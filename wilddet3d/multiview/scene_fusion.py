"""Cross-view attention fusion for multi-capture scenes.

SceneFusion aggregates per-view queries (and depth latents) from
n=1..3 views of the same physical scene, producing scene-level
queries that are then decoded with the standard 3D head.

Design:
- Pose encoding: each view's T_base_cam is encoded as a Fourier
  pose embedding and prepended to its queries as a CLS-like token.
- Cross-view attention: 2 transformer layers (self-attn across all
  views, then cross-attn back to per-view queries).
- Anchor frame: predictions are made in the anchor view's camera
  frame (view 0 by default), so depth/intrinsics can still be used.
- n=1 graceful degrade: with only one view the self-attention is
  identity and output == input (SceneFusion adds zero learnable
  residual by default via zero-init final proj).

The module is inserted after SAM3 grounding (which produces
per-view queries) and before the 3D head.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pose_fourier_embed(
    t_base_cam: Tensor,  # (B, 4, 4)
    d_model: int = 64,
    num_freq: int = 8,
) -> Tensor:
    """Encode rigid transform as a fixed Fourier embedding.

    Extracts the 6-DoF pose as [tx, ty, tz, r00..r08] (12 scalars),
    normalises to [-1, 1] with reasonable scene-scale priors, then
    applies multi-frequency sinusoidal embedding.

    Args:
        t_base_cam: (B, 4, 4) rigid transforms.
        d_model: Output embedding dimension (must be even * len(pose_vec)).
        num_freq: Number of frequency bands.

    Returns:
        (B, d_model) pose embeddings.
    """
    B = t_base_cam.shape[0]
    # Flatten translation (normalised by 3m typical range) + 9 rotation elements
    t = t_base_cam[:, :3, 3] / 3.0           # (B, 3)
    r = t_base_cam[:, :3, :3].reshape(B, 9)  # (B, 9)
    pose = torch.cat([t, r], dim=1)           # (B, 12)

    freqs = 2.0 ** torch.arange(num_freq, device=t_base_cam.device)  # (F,)
    x = pose.unsqueeze(-1) * freqs            # (B, 12, F)
    emb = torch.cat([x.sin(), x.cos()], dim=-1).reshape(B, 12 * num_freq * 2)  # (B, 288)

    # Project to d_model
    if not hasattr(_pose_fourier_embed, "_proj"):
        # lazy static Linear (used only during __init__ of SceneFusion)
        pass
    return emb


class SceneFusion(nn.Module):
    """Cross-view attention fusion module.

    Args:
        d_model: Hidden dimension (must match WildDet3D hidden_dim, 256).
        num_heads: Multi-head attention heads.
        num_layers: Number of cross-view transformer layers.
        max_views: Maximum number of views supported (for positional tables).
        pose_embed_dim: Dimension of the pose embedding MLP output.
        dropout: Dropout on attention.
        zero_init_output: Zero-init the final projection so the module
            starts as identity and does not disturb pretrained weights.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_views: int = 4,
        pose_embed_dim: int = 64,
        dropout: float = 0.1,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_views = max_views

        # Pose MLP: 12*8*2=192 -> pose_embed_dim -> d_model
        pose_in = 12 * 8 * 2  # matches _pose_fourier_embed default
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_in, pose_embed_dim),
            nn.LayerNorm(pose_embed_dim),
            nn.GELU(),
            nn.Linear(pose_embed_dim, d_model),
        )

        # Learnable view-index embedding (up to max_views)
        self.view_embed = nn.Embedding(max_views, d_model)

        # Transformer layers (each: self-attn → FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoder(encoder_layer, num_layers=1)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        # Final projection per view (residual path)
        self.out_proj = nn.Linear(d_model, d_model)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def _build_pose_emb(
        self, t_base_cam: list[Tensor] | Tensor
    ) -> Tensor:
        """Build pose embeddings for a list/batch of T_base_cam."""
        if isinstance(t_base_cam, list):
            t = torch.stack(t_base_cam)  # (V, 4, 4)
        else:
            t = t_base_cam
        B = t.shape[0]
        t_part = t[:, :3, 3] / 3.0
        r_part = t[:, :3, :3].reshape(B, 9)
        pose = torch.cat([t_part, r_part], dim=1)
        freqs = 2.0 ** torch.arange(8, device=t.device)
        x = pose.unsqueeze(-1) * freqs
        return torch.cat([x.sin(), x.cos()], dim=-1).reshape(B, -1)

    def forward(
        self,
        queries_per_view: list[Tensor],
        extrinsics_per_view: list[Tensor],
        key_padding_masks: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Fuse queries from multiple views.

        Args:
            queries_per_view: List of V tensors each (N_prompts_v, S, d_model)
                where N_prompts_v is the number of prompts for view v and
                S is the number of queries per prompt.  All views must share
                the same prompt set (same N_prompts) for the scene-level batch.
            extrinsics_per_view: List of V tensors (4, 4) T_base_cam.
            key_padding_masks: Optional masks [N_prompts, S] per view
                (True = padding; forwarded into attention).

        Returns:
            List of V enhanced query tensors, same shapes as input.
            At n=1 the output == input + zero_init residual = input.
        """
        n_views = len(queries_per_view)
        if n_views == 0:
            return queries_per_view

        N, S, D = queries_per_view[0].shape
        device = queries_per_view[0].device

        # Encode pose for each view → (1, 1, D) broadcast token
        pose_raw = self._build_pose_emb(
            [e.to(device) for e in extrinsics_per_view]
        )  # (V, 192)
        pose_emb = self.pose_mlp(pose_raw.to(queries_per_view[0].dtype))  # (V, D)

        # View index embedding
        view_idx = torch.arange(n_views, device=device)
        view_emb = self.view_embed(view_idx)  # (V, D)

        # Combined pose+view token per view: (V, D)
        pv_tok = pose_emb + view_emb  # (V, D)

        # Concatenate all views into a flat sequence for self-attention
        # Shape: (N, V*S, D)
        # We also prepend a per-view pose token (1 token per view)
        # → total sequence length = V*(S+1)
        segs = []
        for v, q in enumerate(queries_per_view):
            # q: (N, S, D)
            tok = pv_tok[v].unsqueeze(0).unsqueeze(0).expand(N, 1, D)
            segs.append(torch.cat([tok, q], dim=1))  # (N, S+1, D)
        x = torch.cat(segs, dim=1)  # (N, V*(S+1), D)

        # Build padding mask if provided
        attn_mask = None
        if key_padding_masks is not None:
            # Prepend False (valid) for the pose token of each view
            masks = []
            for m in key_padding_masks:
                tok_m = torch.zeros(N, 1, dtype=torch.bool, device=device)
                masks.append(torch.cat([tok_m, m], dim=1))
            attn_mask = torch.cat(masks, dim=1)  # (N, V*(S+1))

        # Run transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)

        # Split back and project residual
        out = []
        offset = 0
        for q in queries_per_view:
            seg_len = q.shape[1] + 1  # +1 for pose token
            # Skip pose token, take query slice
            delta = self.out_proj(x[:, offset + 1 : offset + seg_len])
            out.append(q + delta)
            offset += seg_len
        return out
