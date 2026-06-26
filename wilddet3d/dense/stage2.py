"""JENGA Stage 2: dimension-conditioned amodal box completion.

Per visible-box query: self-attention among queries (mutual "jenga" layout
consistency) + cross-attention over the scene's candidate-dimension tokens
(allowed sizes). Outputs a hard-selectable assignment over the dim tokens
(size = selected dim) plus an actual-center residual and a 6D rotation. Handles
variable per-scene query- and dim-counts via padding + attention masks.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pad_stack(
    tensors: list[Tensor], dim0_max: int, feat_dim: int
) -> tuple[Tensor, Tensor]:
    """Pad a list of ``[Ni, F]`` to ``[B, dim0_max, F]``; return (padded, mask)."""
    b = len(tensors)
    out = tensors[0].new_zeros(b, dim0_max, feat_dim)
    mask = torch.zeros(b, dim0_max, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n:
            out[i, :n] = t
            mask[i, :n] = True
    return out, mask


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, q: Tensor, kv: Tensor, q_pad: Tensor, kv_pad: Tensor) -> Tensor:
        x = self.n1(q)
        x = q + self.self_attn(
            x, x, x, key_padding_mask=~q_pad, need_weights=False
        )[0]
        y = self.n2(x)
        x = x + self.cross_attn(
            y, kv, kv, key_padding_mask=~kv_pad, need_weights=False
        )[0]
        return x + self.ffn(self.n3(x))


class JengaStage2(nn.Module):
    """Dimension-conditioned Stage-2 decoder.

    Args:
        in_ch: fused FPN feature channels (256 for SAM3).
        d_model: decoder width.
        layers: number of self+cross attention blocks.
        heads: attention heads.
    """

    def __init__(
        self, in_ch: int = 256, d_model: int = 512, layers: int = 12, heads: int = 8
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(in_ch, d_model)
        self.obb_embed = nn.Sequential(
            nn.Linear(12, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.dim_embed = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.layers = nn.ModuleList(
            [_DecoderLayer(d_model, heads) for _ in range(layers)]
        )
        self.q_to_assign = nn.Linear(d_model, d_model)
        self.k_to_assign = nn.Linear(d_model, d_model)
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 9)
        )  # 3 center delta + 6 rot

    @staticmethod
    def _sample_feat(feat: Tensor, uv: Tensor) -> Tensor:
        """Bilinear-sample ``feat[C,Hf,Wf]`` at ``uv[Q,2]`` (grid coords) -> ``[Q,C]``."""
        _, hf, wf = feat.shape
        gx = (uv[:, 0] / max(wf - 1, 1)) * 2 - 1
        gy = (uv[:, 1] / max(hf - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        s = F.grid_sample(feat[None], grid, align_corners=True)  # [1,C,1,Q]
        return s[0, :, 0].transpose(0, 1)  # [Q,C]

    def forward(
        self,
        feat: Tensor,
        queries_uv: list[Tensor],
        vis_obb: list[Tensor],
        catalog: list[Tensor],
    ) -> dict[str, Tensor]:
        b = feat.shape[0]
        qmax = max((q.shape[0] for q in queries_uv), default=0)
        kmax = max((c.shape[0] for c in catalog), default=0)
        qmax, kmax = max(qmax, 1), max(kmax, 1)

        q_feats = []
        for i in range(b):
            if queries_uv[i].shape[0]:
                qf = self._sample_feat(feat[i], queries_uv[i])
            else:
                qf = feat.new_zeros(0, feat.shape[1])
            q_feats.append(self.feat_proj(qf) + self.obb_embed(vis_obb[i]))
        q, q_mask = _pad_stack(q_feats, qmax, self.d_model)
        kv, k_mask = _pad_stack([self.dim_embed(c) for c in catalog], kmax, self.d_model)

        for layer in self.layers:
            q = layer(q, kv, q_mask, k_mask)

        qa = self.q_to_assign(q)  # [B,Qmax,d]
        ka = self.k_to_assign(kv)  # [B,Kmax,d]
        logits = torch.einsum("bqd,bkd->bqk", qa, ka) / (self.d_model ** 0.5)
        logits = logits.masked_fill(~k_mask[:, None, :], float("-inf"))
        pose = self.pose_head(q)
        return {
            "assign_logits": logits,
            "center_delta": pose[..., :3],
            "rot6d": pose[..., 3:],
            "q_mask": q_mask,
            "k_mask": k_mask,
        }
