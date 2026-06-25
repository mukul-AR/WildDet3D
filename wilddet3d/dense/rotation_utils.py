"""Pure-torch rotation utilities for the two-stage 9-DoF head.

Mirrors the reusable pieces of ``wilddet3d/ops/rotation.py`` (6D continuous
rotation representation from Zhou et al. 2019 and the cuboid D2 symmetry group)
without importing ``vis4d`` so the two-stage pipeline runs standalone.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert quaternions (w, x, y, z) to rotation matrices ``[..., 3, 3]``."""
    quaternions = F.normalize(quaternions, dim=-1)
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    o = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert a 6D rotation representation to a rotation matrix.

    The 6D rep is the first two ROWS of R (consistent with
    ``matrix_to_rotation_6d``); Gram-Schmidt recovers the full 3x3 matrix.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to the 6D representation (first two rows)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


# Cuboid (D2) proper-rotation symmetry group as local-frame diagonal sign
# matrices S = diag(s): R' = R @ S leaves a cuboid's geometry unchanged
# (I, Rx(pi), Ry(pi), Rz(pi)).
_CUBOID_SYMMETRY_SIGNS = (
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)


def cuboid_symmetry_rotation_6d(d6: Tensor) -> Tensor:
    """All 4 cuboid-symmetric variants of a 6D rotation rep ``[..., 4, 6]``.

    Every column ``j`` of R is scaled by ``s[j]`` under ``R' = R @ diag(s)``, so
    each row is multiplied element-wise by ``s``: the variant 6D reps are
    ``d6 * [s0, s1, s2, s0, s1, s2]`` (identity variant first).
    """
    signs = d6.new_tensor(_CUBOID_SYMMETRY_SIGNS)  # [4, 3]
    patterns = torch.cat([signs, signs], dim=-1)  # [4, 6]
    return d6.unsqueeze(-2) * patterns


def geodesic_angle(r_pred: Tensor, r_gt: Tensor, eps: float = 1e-6) -> Tensor:
    """Geodesic angle (radians) between two batches of rotations ``[..., 3, 3]``.

    Uses a clamped ``acos`` to avoid the NaN/blow-up at +-1 documented in the
    design doc; for the differentiable loss prefer the chordal distance.
    """
    r12 = torch.matmul(r_pred, r_gt.transpose(-1, -2))
    trace = r12[..., 0, 0] + r12[..., 1, 1] + r12[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def symmetry_min_geodesic(d6_pred: Tensor, d6_gt: Tensor) -> Tensor:
    """Minimum geodesic angle over the cuboid symmetry group (metric, radians).

    Args:
        d6_pred: predicted 6D rotation ``[N, 6]``.
        d6_gt: target 6D rotation ``[N, 6]``.

    Returns:
        Per-sample minimum geodesic angle ``[N]`` (degrees-friendly metric).
    """
    r_pred = rotation_6d_to_matrix(d6_pred)  # [N, 3, 3]
    variants = cuboid_symmetry_rotation_6d(d6_gt)  # [N, 4, 6]
    n, k, _ = variants.shape
    r_var = rotation_6d_to_matrix(variants.reshape(n * k, 6)).reshape(
        n, k, 3, 3
    )
    angles = geodesic_angle(
        r_pred.unsqueeze(1).expand(n, k, 3, 3), r_var
    )  # [N, 4]
    return angles.min(dim=1).values


def rad2deg(x: Tensor) -> Tensor:
    """Radians -> degrees."""
    return x * (180.0 / math.pi)


def symmetry_chordal_loss(d6_pred: Tensor, d6_gt: Tensor) -> Tensor:
    """Per-sample min chordal (squared Frobenius) rotation loss ``[N]``.

    Minimised over the cuboid symmetry group so the model is not penalised for
    predicting a physically identical but differently-labelled rotation.
    """
    r_pred = rotation_6d_to_matrix(d6_pred)  # [N, 3, 3]
    variants = cuboid_symmetry_rotation_6d(d6_gt)  # [N, 4, 6]
    n, k, _ = variants.shape
    r_var = rotation_6d_to_matrix(variants.reshape(n * k, 6)).reshape(n, k, 3, 3)
    diff = r_pred.unsqueeze(1) - r_var  # [N, 4, 3, 3]
    chordal = diff.pow(2).sum(dim=(-1, -2))  # [N, 4]
    return chordal.min(dim=1).values  # [N]
