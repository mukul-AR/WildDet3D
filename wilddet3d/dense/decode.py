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


@torch.no_grad()
def decode_jenga(stage1_dets, feat, stride, stage2, catalog, k):
    """Chain Stage-1 visible dets -> Stage-2 actual boxes (argmax dim selection).

    Args:
        stage1_dets: per-image dicts from :func:`decode_dense`
            (``center``/``size``/``R``/``score`` of the *visible* boxes).
        feat: fused FPN map ``[B, C, Hf, Wf]`` (from ``DenseDet3D(..., return_feat=True)``).
        stride: input-pixel / FPN-cell ratio.
        stage2: a :class:`~wilddet3d.dense.stage2.JengaStage2` module.
        catalog: per-image candidate dims ``[Ki, 3]`` (ascending-sorted).
        k: ``[B, 3, 3]`` intrinsics (input resolution).

    Returns:
        per-image dict: ``center`` ``[M,3]``, ``size`` ``[M,3]`` (a catalog row),
        ``R`` ``[M,3,3]``, ``score`` ``[M]``, ``assign`` ``[M]``.
    """
    out = []
    for i, det in enumerate(stage1_dets):
        c = det["center"]
        if c.shape[0] == 0:
            out.append(
                {
                    "center": c,
                    "size": c.new_zeros(0, 3),
                    "R": c.new_zeros(0, 3, 3),
                    "score": det["score"],
                    "assign": c.new_zeros(0, dtype=torch.long),
                }
            )
            continue
        fx, fy = k[i, 0, 0], k[i, 1, 1]
        cx0, cy0 = k[i, 0, 2], k[i, 1, 2]
        z = c[:, 2].clamp_min(1e-3)
        u = (fx * c[:, 0] / z + cx0) / stride
        v = (fy * c[:, 1] / z + cy0) / stride
        uv = torch.stack([u, v], dim=-1)
        vis_obb = torch.cat([c, det["size"], det["R"][:, :2].reshape(-1, 6)], dim=-1)
        res = stage2(feat[i : i + 1], [uv], [vis_obb], [catalog[i]])
        n = c.shape[0]
        assign = res["assign_logits"][0, :n].argmax(-1)
        size = catalog[i][assign]
        center = c + res["center_delta"][0, :n]
        R = rotation_6d_to_matrix(res["rot6d"][0, :n])
        out.append(
            {
                "center": center,
                "size": size,
                "R": R,
                "score": det["score"],
                "assign": assign,
            }
        )
    return out
