"""Dense target assignment (CenterNet-3D style) for the prompt-free head.

Projects each GT 3D box center to the FPN grid, draws a penalty-reduced
Gaussian on the objectness heatmap, and writes the 12-scalar 9-DoF regression
target at the peak cell. Parameterization matches ``head.py`` / ``decode.py``:
    [du, dv, log_z, log_w, log_h, log_l, r0..r5].
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """CenterNet Gaussian radius for a box of (height, width) in cells."""
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1 * b1 - 4 * a1 * c1, 0))
    r1 = (b1 - sq1) / (2 * a1)
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 * b2 - 4 * a2 * c2, 0))
    r2 = (b2 - sq2) / (2 * a2)
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def _draw_gaussian(heatmap: Tensor, cx: int, cy: int, radius: int) -> None:
    """Draw a 2D Gaussian peak (in-place, max-merged) on a [H, W] heatmap."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
    ys = torch.arange(-top, bottom, device=heatmap.device).view(-1, 1)
    xs = torch.arange(-left, right, device=heatmap.device).view(1, -1)
    g = torch.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    region = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    torch.maximum(region, g, out=region)


def build_dense_targets(
    centers: list[Tensor],
    sizes: list[Tensor],
    rot6d: list[Tensor],
    box2d: list[Tensor],
    k: Tensor,
    feat_hw: tuple[int, int],
    stride: float,
    device: torch.device,
    size_eps: float = 1e-3,
) -> dict[str, Tensor]:
    """Build dense heatmap + regression targets for a batch.

    Args:
        centers: per-image GT centers (camera frame) ``[Ni, 3]``.
        sizes: per-image GT sizes (w, h, l) ``[Ni, 3]``.
        rot6d: per-image GT 6D rotations (camera frame) ``[Ni, 6]``.
        box2d: per-image projected 2D boxes (x1,y1,x2,y2) in input px ``[Ni, 4]``.
        k: per-image intrinsics ``[B, 3, 3]`` for the input-resolution image.
        feat_hw: (Hf, Wf) FPN grid size.
        stride: input-pixel / FPN-cell ratio.

    Returns:
        dict: ``heatmap`` ``[B, 1, Hf, Wf]``, ``reg`` ``[B, 12, Hf, Wf]``,
        ``pos_mask`` ``[B, Hf, Wf]`` (bool).
    """
    b = len(centers)
    hf, wf = feat_hw
    heatmap = torch.zeros(b, 1, hf, wf, device=device)
    reg = torch.zeros(b, 12, hf, wf, device=device)
    pos = torch.zeros(b, hf, wf, dtype=torch.bool, device=device)

    for i in range(b):
        c = centers[i]
        if c.numel() == 0:
            continue
        fx, fy = k[i, 0, 0], k[i, 1, 1]
        cx0, cy0 = k[i, 0, 2], k[i, 1, 2]
        z = c[:, 2].clamp_min(size_eps)
        u = fx * c[:, 0] / z + cx0
        v = fy * c[:, 1] / z + cy0
        gx = (u / stride)
        gy = (v / stride)
        cellx = gx.long().clamp(0, wf - 1)
        celly = gy.long().clamp(0, hf - 1)

        bw = (box2d[i][:, 2] - box2d[i][:, 0]).clamp_min(1.0) / stride
        bh = (box2d[i][:, 3] - box2d[i][:, 1]).clamp_min(1.0) / stride

        # Sort by depth (far first) so nearer boxes overwrite on collision.
        order = torch.argsort(z, descending=True)
        for j in order.tolist():
            ux, uy = int(cellx[j]), int(celly[j])
            if not (0 <= u[j] / stride < wf and 0 <= v[j] / stride < hf):
                continue
            radius = int(max(0, gaussian_radius((float(bh[j]), float(bw[j])))))
            _draw_gaussian(heatmap[i, 0], ux, uy, radius)
            reg[i, 0, uy, ux] = gx[j] - ux - 0.5
            reg[i, 1, uy, ux] = gy[j] - uy - 0.5
            reg[i, 2, uy, ux] = torch.log(z[j])
            reg[i, 3:6, uy, ux] = torch.log(sizes[i][j].clamp_min(size_eps))
            reg[i, 6:12, uy, ux] = rot6d[i][j]
            pos[i, uy, ux] = True

    return {"heatmap": heatmap, "reg": reg, "pos_mask": pos}
