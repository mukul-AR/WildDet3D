import math

import torch

from wilddet3d.dense.metrics import corner_distance, iou3d_mc


def _eye(n):
    return torch.eye(3).expand(n, 3, 3).contiguous()


def test_identical_boxes_iou_one():
    torch.manual_seed(0)
    c = torch.zeros(1, 3)
    s = torch.ones(1, 3)
    iou = iou3d_mc(c, s, _eye(1), c.clone(), s.clone(), _eye(1), n_samples=50000)
    assert iou.item() > 0.98


def test_disjoint_boxes_iou_zero():
    torch.manual_seed(0)
    c1 = torch.zeros(1, 3)
    c2 = torch.tensor([[10.0, 0.0, 0.0]])
    s = torch.ones(1, 3)
    iou = iou3d_mc(c1, s, _eye(1), c2, s.clone(), _eye(1), n_samples=20000)
    assert iou.item() == 0.0


def test_half_offset_axis_aligned_iou_known():
    # two unit cubes offset 0.5 in x: intersection 0.5, union 1.5 -> IoU 1/3
    torch.manual_seed(0)
    c1 = torch.zeros(1, 3)
    c2 = torch.tensor([[0.5, 0.0, 0.0]])
    s = torch.ones(1, 3)
    iou = iou3d_mc(c1, s, _eye(1), c2, s.clone(), _eye(1), n_samples=200000)
    assert abs(iou.item() - 1.0 / 3.0) < 0.02


def test_corner_distance_add_vs_adds_under_symmetry():
    # a 180 deg rotation about z is a cuboid symmetry (same physical box).
    c = torch.zeros(1, 3)
    s = torch.tensor([[0.3, 0.4, 0.5]])
    eye = torch.eye(3)[None]
    a = math.pi
    rz = torch.tensor([[[math.cos(a), -math.sin(a), 0.0],
                        [math.sin(a), math.cos(a), 0.0],
                        [0.0, 0.0, 1.0]]])
    add = corner_distance(c, s, rz, c, s, eye, symmetric=False)   # fixed correspondence
    adds = corner_distance(c, s, rz, c, s, eye, symmetric=True)   # ADD-S
    assert add.item() > 0.3       # fixed-ADD sees the flip as large error
    assert adds.item() < 1e-5     # ADD-S: ~0 (it's the same box)
