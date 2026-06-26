"""3D OBB evaluation metrics for the JENGA detector.

Full-rotation 3D IoU has no closed form (per the design doc); we approximate it
by **Monte-Carlo sampling** — sample points in the union AABB of the two oriented
boxes and count inside-each. Pure torch, runs on any device, handles arbitrary
9-DoF rotation. Used for eval metrics (not as a training loss).
"""
from __future__ import annotations

import itertools

import torch
from torch import Tensor

_UNIT_SIGNS = torch.tensor(
    list(itertools.product((-0.5, 0.5), repeat=3)), dtype=torch.float32
)  # [8, 3]


def box_corners(center: Tensor, size: Tensor, r: Tensor) -> Tensor:
    """Oriented-box corners ``[N, 8, 3]`` from center/size/rotation ``[N,...]``."""
    local = _UNIT_SIGNS.to(center)[None] * size[:, None, :]  # [N, 8, 3]
    return center[:, None, :] + torch.einsum("nij,nkj->nki", r, local)


def _inside(points: Tensor, center: Tensor, size: Tensor, r: Tensor) -> Tensor:
    """Mask ``[N, S]`` of which sampled points lie inside each oriented box."""
    rel = points - center[:, None, :]  # [N, S, 3]
    local = torch.einsum("nij,nsj->nsi", r.transpose(-1, -2), rel)  # to box frame
    half = size[:, None, :] * 0.5
    return (local.abs() <= half + 1e-9).all(dim=-1)  # [N, S]


def iou3d_mc(
    c1: Tensor, s1: Tensor, r1: Tensor,
    c2: Tensor, s2: Tensor, r2: Tensor,
    n_samples: int = 8192,
) -> Tensor:
    """Monte-Carlo 3D IoU ``[N]`` between two batches of oriented boxes.

    Samples ``n_samples`` points uniformly in each pair's union AABB and returns
    intersection/union of the inside-counts. Error ~ ``1/sqrt(n_samples)``.
    """
    corners = torch.cat(
        [box_corners(c1, s1, r1), box_corners(c2, s2, r2)], dim=1
    )  # [N, 16, 3]
    lo = corners.min(dim=1).values  # [N, 3]
    hi = corners.max(dim=1).values
    u = torch.rand(c1.shape[0], n_samples, 3, device=c1.device, dtype=c1.dtype)
    pts = lo[:, None, :] + u * (hi - lo)[:, None, :]  # [N, S, 3]
    in1 = _inside(pts, c1, s1, r1)
    in2 = _inside(pts, c2, s2, r2)
    inter = (in1 & in2).float().sum(dim=-1)
    union = (in1 | in2).float().sum(dim=-1).clamp_min(1.0)
    return inter / union
