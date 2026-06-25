#!/usr/bin/env python3
"""Unit smoke tests for the two-stage 9-DoF pipeline (no GPU/data required).

    python scripts/test_two_stage_pipeline.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wilddet3d.twostage.losses import NineDoFLoss  # noqa: E402
from wilddet3d.twostage.model import (  # noqa: E402
    TwoStage9DoF,
    transform_obb_cam_to_base,
)
from wilddet3d.twostage.rotation_utils import (  # noqa: E402
    cuboid_symmetry_rotation_6d,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    symmetry_min_geodesic,
)
from wilddet3d.twostage.stage1_visual import Stage1VisualNet  # noqa: E402
from wilddet3d.twostage.stage2_geometry import Stage2GeometryNet  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def _check(cond: bool, msg: str) -> bool:
    print(f"{PASS if cond else FAIL} {msg}")
    return bool(cond)


def test_rotation() -> bool:
    ok = True
    torch.manual_seed(0)
    q = torch.randn(16, 4)
    r = quaternion_to_matrix(q)
    # orthonormal + det 1
    eye = torch.bmm(r, r.transpose(1, 2))
    ok &= _check(
        (eye - torch.eye(3)).abs().max() < 1e-5, "quat->matrix orthonormal"
    )
    d6 = matrix_to_rotation_6d(r)
    r2 = rotation_6d_to_matrix(d6)
    ok &= _check((r - r2).abs().max() < 1e-5, "6d<->matrix round-trip")

    # cuboid symmetry: R @ diag(s) for the 4 sign patterns
    variants = cuboid_symmetry_rotation_6d(d6)
    ok &= _check(variants.shape == (16, 4, 6), f"variants {variants.shape}")
    signs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    for i, s in enumerate(signs):
        rv = rotation_6d_to_matrix(variants[:, i])
        mat_s = torch.diag(torch.tensor(s, dtype=torch.float32))
        err = (rv - r @ mat_s).abs().max().item()
        ok &= _check(err < 1e-5, f"symmetry variant {s} err={err:.1e}")

    # symmetry geodesic is ~0 for a symmetric flip of the same rotation
    # (floored at ~0.0014 rad by the safe-acos clamp, which is expected).
    flipped = variants[:, 1]
    ang = symmetry_min_geodesic(d6, flipped)
    ok &= _check(ang.max() < 1e-2, "min-geodesic ~0 across symmetry")
    return ok


def test_loss() -> bool:
    ok = True
    torch.manual_seed(1)
    n = 8
    q = torch.randn(n, 4)
    d6 = matrix_to_rotation_6d(quaternion_to_matrix(q))
    pred = {
        "center": torch.randn(n, 3, requires_grad=True),
        "size": torch.rand(n, 3) + 0.2,
        "rot6d": d6.clone().requires_grad_(True),
        "presence": torch.randn(n, requires_grad=True),
    }
    target = {
        "center": torch.randn(n, 3),
        "size": torch.rand(n, 3) + 0.2,
        "rot6d": matrix_to_rotation_6d(quaternion_to_matrix(torch.randn(n, 4))),
        "presence": torch.ones(n),
    }
    loss = NineDoFLoss()
    out = loss(pred, target)
    ok &= _check(out["total"].ndim == 0, "loss is scalar")
    ok &= _check("rot_deg" in out, "rot_deg metric present")
    out["total"].backward()
    ok &= _check(
        pred["center"].grad is not None
        and pred["rot6d"].grad.abs().max() > 0,
        "loss backward grads OK",
    )
    # zero rotation loss when pred == gt
    same = matrix_to_rotation_6d(quaternion_to_matrix(torch.randn(n, 4)))
    out2 = loss(
        {"center": target["center"], "size": target["size"], "rot6d": same},
        {"center": target["center"], "size": target["size"], "rot6d": same},
    )
    ok &= _check(out2["rot"].item() < 1e-5, "rot loss 0 when equal")
    ok &= _check(out2["center"].item() < 1e-6, "center loss 0 when equal")
    return ok


def test_stage1() -> bool:
    ok = True
    net = Stage1VisualNet(ctx_dim=10)
    b = 4
    crop = torch.rand(b, 4, 48, 48)
    ctx = torch.rand(b, 10)
    anchor = torch.randn(b, 3)
    out = net(crop, ctx, anchor)
    ok &= _check(out["center"].shape == (b, 3), "stage1 center shape")
    ok &= _check(out["size"].shape == (b, 3), "stage1 size shape")
    ok &= _check((out["size"] > 0).all(), "stage1 size positive")
    ok &= _check(out["rot6d"].shape == (b, 6), "stage1 rot6d shape")
    # at init center ~= anchor (zero-init center head)
    ok &= _check(
        (out["center"] - anchor).abs().max() < 1e-4, "stage1 center≈anchor init"
    )
    return ok


def test_stage2() -> bool:
    ok = True
    net = Stage2GeometryNet(max_sku=8)
    b = 4
    visual = {
        "center": torch.randn(b, 3),
        "size": torch.rand(b, 3) + 0.2,
        "rot6d": matrix_to_rotation_6d(quaternion_to_matrix(torch.randn(b, 4))),
    }
    walls = torch.randn(b, 3, 5)
    walls[..., 4] = 1.0
    skus = torch.rand(b, 8, 4)
    skus[..., 3] = (torch.rand(b, 8) > 0.3).float()
    out = net(visual, walls, skus)
    ok &= _check(out["center"].shape == (b, 3), "stage2 center shape")
    ok &= _check((out["size"] > 0).all(), "stage2 size positive")
    # identity-init: output ≈ visual input
    ok &= _check(
        (out["center"] - visual["center"]).abs().max() < 1e-4,
        "stage2 center≈visual init",
    )
    ok &= _check(
        (out["size"] - visual["size"]).abs().max() < 1e-4,
        "stage2 size≈visual init",
    )
    # all-invalid SKU scene must not produce NaN
    skus0 = torch.rand(b, 8, 4)
    skus0[..., 3] = 0.0
    out0 = net(visual, walls, skus0)
    ok &= _check(
        torch.isfinite(out0["center"]).all(), "stage2 no-NaN with empty SKU"
    )
    return ok


def test_model() -> bool:
    ok = True
    model = TwoStage9DoF(
        stage1_kwargs={"ctx_dim": 10}, stage2_kwargs={"max_sku": 8}
    )
    b = 4
    batch = {
        "crop": torch.rand(b, 4, 48, 48),
        "ctx": torch.rand(b, 10),
        "anchor_center": torch.randn(b, 3),
        "T_base_cam": torch.eye(4).unsqueeze(0).repeat(b, 1, 1),
        "walls": torch.randn(b, 3, 5),
        "skus": torch.rand(b, 8, 4),
    }
    out = model(batch)
    ok &= _check("stage1" in out and "stage2" in out, "model returns 2 stages")
    # identity transform: stage1_base center == stage1 cam center
    ok &= _check(
        (out["stage1_base"]["center"] - out["stage1"]["center"]).abs().max()
        < 1e-5,
        "cam->base identity transform",
    )

    # frame transform correctness with a non-trivial T
    t = torch.eye(4)
    t[:3, :3] = quaternion_to_matrix(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))[0]
    t[:3, 3] = torch.tensor([1.0, -2.0, 0.3])
    visual = {
        "center": torch.tensor([[0.4, 0.1, 2.0]]),
        "size": torch.tensor([[0.3, 0.2, 0.5]]),
        "rot6d": matrix_to_rotation_6d(torch.eye(3).unsqueeze(0)),
    }
    base = transform_obb_cam_to_base(visual, t.unsqueeze(0))
    expect = t[:3, :3] @ visual["center"][0] + t[:3, 3]
    ok &= _check(
        (base["center"][0] - expect).abs().max() < 1e-5,
        "explicit cam->base center transform",
    )
    return ok


def test_dataset() -> bool:
    cache = "data/anyware_scenes/two_stage_cache/val"
    if not os.path.isdir(cache):
        print(f"  (skip dataset test: {cache} not built)")
        return True
    from wilddet3d.twostage.dataset import TwoStageAnywareDataset

    ds = TwoStageAnywareDataset(cache)
    ok = _check(len(ds) > 0, f"dataset len={len(ds)}")
    s = ds[0]
    ok &= _check(s["crop"].shape[0] == 4, f"crop channels {s['crop'].shape}")
    ok &= _check(s["walls"].shape == (3, 5), "walls shape")
    ok &= _check(s["skus"].shape[1] == 4, "skus shape")
    ok &= _check(s["gt_size"].min() > 0, "gt_size positive")
    return ok


def main() -> None:
    print("=" * 60)
    print("Two-Stage 9-DoF Pipeline Test Suite")
    print("=" * 60)
    results = {
        "rotation/symmetry": test_rotation(),
        "9-DoF loss": test_loss(),
        "stage1 visual": test_stage1(),
        "stage2 geometry": test_stage2(),
        "two-stage model": test_model(),
        "cached dataset": test_dataset(),
    }
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    all_ok = all(results.values())
    print("RESULT:", "ALL PASSED ✓" if all_ok else "SOME FAILED ✗")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
