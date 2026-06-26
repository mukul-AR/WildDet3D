import math

import torch

from wilddet3d.dense.rotation_utils import (
    _sized_symmetry_variants,
    matrix_to_rotation_6d,
    rad2deg,
    sized_symmetry_chordal_loss,
    sized_symmetry_min_geodesic,
)


def _rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_symmetry_group_size_depends_on_equal_axes():
    r = torch.eye(3).expand(3, 3, 3).contiguous()
    sizes = torch.tensor(
        [
            [0.20, 0.30, 0.40],  # generic -> 4
            [0.30, 0.30, 0.50],  # square cross-section -> 8
            [0.30, 0.30, 0.30],  # cube -> 24
        ]
    )
    _, mask = _sized_symmetry_variants(r, sizes)
    assert mask.sum(dim=1).tolist() == [4, 8, 24]


def test_square_box_90deg_about_square_axis_is_free():
    # square cross-section in x,y (equal), long axis z; a 90 deg rotation about z
    # is a true symmetry and must cost ~0 under the size-aware loss.
    size = torch.tensor([[0.30, 0.30, 0.50]])
    r_gt = torch.eye(3)[None]
    d6_gt = matrix_to_rotation_6d(r_gt)
    d6_pred = matrix_to_rotation_6d(_rot_z(90)[None])
    loss = sized_symmetry_chordal_loss(d6_pred, d6_gt, size)
    assert loss.item() < 1e-4
    deg = rad2deg(sized_symmetry_min_geodesic(d6_pred, d6_gt, size))
    assert deg.item() < 1.0


def test_generic_box_90deg_is_penalized():
    # for a generic box the same 90 deg rotation is NOT a symmetry -> large cost
    size = torch.tensor([[0.20, 0.30, 0.40]])
    d6_gt = matrix_to_rotation_6d(torch.eye(3)[None])
    d6_pred = matrix_to_rotation_6d(_rot_z(90)[None])
    deg = rad2deg(sized_symmetry_min_geodesic(d6_pred, d6_gt, size))
    assert deg.item() > 80.0
