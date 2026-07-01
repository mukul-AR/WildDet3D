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
def decode_jenga(stage1_dets, feat, stride, stage2, catalog, k, posthoc_anchor=False):
    """Chain Stage-1 visible dets -> Stage-2 actual boxes.

    The actual box **inherits the visible box's rotation** (verified identical in
    the data); Stage 2 only selects the SKU and places the center. The selected
    SKU's sorted dims are laid onto the box axes by the visible per-axis extent
    order (smallest dim -> axis with smallest visible extent).

    If ``posthoc_anchor`` is set, the actual-box center is **not** taken from the
    Stage-2 ``center_delta``; instead the actual box is anchored so its
    camera-facing (near) face coincides with the visible box's near face, then
    grown away from the camera by the (catalog) actual depth. This is a pure
    geometric correction that tests whether the occluded-tail center error is
    just "center placed too shallow" (model fails to push back by the known SKU
    depth). The shift per local axis j is ``0.5*(actual-visible extent)`` toward
    +Z (away from camera, sign of ``R[2,j]``); only the occluded (depth) axis
    has a meaningful extent gap, so the box grows along the viewing ray.

    Args:
        stage1_dets: per-image dicts from :func:`decode_dense`
            (``center``/``size``/``R``/``score`` of the *visible* boxes).
        feat: fused FPN map ``[B, C, Hf, Wf]`` (from ``DenseDet3D(..., return_feat=True)``).
        stride: input-pixel / FPN-cell ratio.
        stage2: a :class:`~wilddet3d.dense.stage2.JengaStage2` module.
        catalog: per-image candidate dims ``[Ki, 3]`` (ascending-sorted).
        k: ``[B, 3, 3]`` intrinsics (input resolution).

    Returns:
        per-image dict: ``center`` ``[M,3]``, ``size`` ``[M,3]`` (per-axis catalog
        dims), ``R`` ``[M,3,3]`` (visible rotation), ``score`` ``[M]``, ``assign`` ``[M]``.
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
        sku = catalog[i][assign]  # [n, 3] ascending catalog dims
        # arrange SKU dims by the model's PREDICTED per-axis order (learned
        # dim->axis assignment), then snap to the catalog values for valid dims
        order = res["log_size"][0, :n].argsort(dim=-1)  # axes ascending by predicted extent
        size = torch.zeros_like(sku).scatter_(1, order, sku)  # per-axis dims
        R = det["R"]  # rotation inherited from the visible box
        if posthoc_anchor:
            # Domain-invariant center: anchor the actual box's near (camera-facing)
            # face to the visible box's near face, then grow away from the camera
            # by the actual (catalog) depth. Replaces the *learned* center_delta
            # (which doesn't transfer sim->real) with pure geometry.
            away = torch.sign(R[:, 2, :])  # [n,3] which way each axis points in +Z (away from cam)
            away = torch.where(away == 0, torch.ones_like(away), away)
            delta_local = 0.5 * (size - det["size"]) * away  # grow away by half the extent gap
            center = c + torch.einsum("nij,nj->ni", R, delta_local)
        else:
            center = c + res["center_delta"][0, :n]
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
