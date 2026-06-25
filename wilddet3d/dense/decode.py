"""Decode dense head outputs into 9-DoF boxes (inference / eval)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from wilddet3d.dense.rotation_utils import rotation_6d_to_matrix


def _nms_peaks(heat: Tensor, kernel: int = 3) -> Tensor:
    """Keep local maxima of a heatmap (CenterNet maxpool NMS)."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    return heat * (hmax == heat).float()


@torch.no_grad()
def decode_dense(
    heatmap: Tensor,
    reg: Tensor,
    k: Tensor,
    stride: float,
    topk: int = 100,
    score_thresh: float = 0.2,
) -> list[dict[str, Tensor]]:
    """Decode dense maps into per-image 9-DoF detections.

    Args:
        heatmap: ``[B, 1, Hf, Wf]`` objectness logits.
        reg: ``[B, 12, Hf, Wf]`` regression maps.
        k: ``[B, 3, 3]`` intrinsics (input resolution).
        stride: input-pixel / FPN-cell ratio.

    Returns:
        per-image dict: ``center`` ``[M,3]``, ``size`` ``[M,3]``, ``R`` ``[M,3,3]``,
        ``score`` ``[M]``.
    """
    b, _, hf, wf = heatmap.shape
    scores_map = _nms_peaks(torch.sigmoid(heatmap))[:, 0]  # [B, Hf, Wf]
    out = []
    for i in range(b):
        s = scores_map[i].reshape(-1)
        n = min(topk, s.numel())
        topv, topi = torch.topk(s, n)
        keep = topv > score_thresh
        topv, topi = topv[keep], topi[keep]
        cy = (topi // wf).float()
        cx = (topi % wf).float()
        r = reg[i].permute(1, 2, 0).reshape(-1, 12)[topi]  # [M, 12]
        u = (cx + 0.5 + r[:, 0]) * stride
        v = (cy + 0.5 + r[:, 1]) * stride
        z = torch.exp(r[:, 2])
        fx, fy = k[i, 0, 0], k[i, 1, 1]
        cx0, cy0 = k[i, 0, 2], k[i, 1, 2]
        center = torch.stack(
            [(u - cx0) * z / fx, (v - cy0) * z / fy, z], dim=-1
        )
        size = torch.exp(r[:, 3:6])
        rot = rotation_6d_to_matrix(r[:, 6:12])
        out.append(
            {"center": center, "size": size, "R": rot, "score": topv}
        )
    return out
