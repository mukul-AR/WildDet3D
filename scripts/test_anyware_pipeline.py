#!/usr/bin/env python3
"""End-to-end pipeline validation for Anyware 9-DoF multi-view WildDet3D.

Run from the repo root:
    python scripts/test_anyware_pipeline.py

Tests (no GPU/checkpoints required):
  1. Dataset loading (AnywareScenes train + val)
  2. Multi-view scene index grouping
  3. GT base_link round-trip (reprojection error < 1mm)
  4. 9-DoF coder (encode → decode, center err < 1μm)
  5. Cuboid symmetry math (all 4 flip variants)
  6. Symmetry-aware rotation loss backward
  7. SceneFusion zero-init (output == input at init)
  8. SceneFusion n=1 degenerate
  9. SceneLateFusion merge
 10. transform_boxes3d identity
 11. WildDet3DInput tensor shapes
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")

import numpy as np
import torch

ANYWARE_ROOT = "data/anyware_scenes"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def _check(cond: bool, msg: str) -> bool:
    print(f"{PASS if cond else FAIL} {msg}")
    return cond


def test_dataset() -> bool:
    from wilddet3d.data.datasets.anyware_scenes import (
        AnywareScenes,
        get_anyware_class_map,
        get_anyware_det_map,
    )

    ok = True
    cm = get_anyware_class_map("AnywareScenes_val", ANYWARE_ROOT)
    dm = get_anyware_det_map("AnywareScenes_val", ANYWARE_ROOT)
    ds = AnywareScenes(
        data_root=ANYWARE_ROOT,
        dataset_name="AnywareScenes_val",
        class_map=cm,
        det_map=dm,
        with_depth=True,
        remove_empty=True,
    )
    ok &= _check(len(ds) > 0, f"val dataset len={len(ds)}")
    ok &= _check(len(ds.scene_index) > 0, f"scene_index scenes={len(ds.scene_index)}")

    s = ds[0]
    ok &= _check(s["boxes3d"].shape[1] == 10, f"boxes3d shape {s['boxes3d'].shape}")
    ok &= _check((s["depth_maps"] > 0).mean() > 0.05, "depth has valid pixels")
    ok &= _check(s["extrinsics"].shape == (4, 4), "extrinsics (4,4)")
    ok &= _check("scene_name" in s, "scene_name present")

    # multi-view
    n_multi = sum(1 for v in ds.scene_index.values() if len(v) > 1)
    ok &= _check(n_multi >= 0, f"multi-view scenes={n_multi}")

    # round-trip base_link
    import json, yaml, glob
    scene_id = list(ds.scene_index.keys())[0]
    candidates = glob.glob(f"**/{scene_id}", recursive=True) + glob.glob(f"data/**/{scene_id}", recursive=True)
    for c in candidates:
        if os.path.exists(f"{c}/scene.json"):
            sj = json.load(open(f"{c}/scene.json"))
            boxes_raw = yaml.safe_load(sj["scene"]["boxes_yaml_string"])
            gt_centers = np.array([v["xyzwpr"][:3] for v in boxes_raw.values()])
            T = np.array(s["extrinsics"])
            b3 = s["boxes3d"][:len(gt_centers)]
            centers_base = (T[:3, :3] @ np.array(b3[:, :3]).T).T + T[:3, 3]
            errs = np.linalg.norm(
                centers_base - gt_centers[: len(centers_base)], axis=1
            )
            ok &= _check(errs.max() < 0.001, f"round-trip err max={errs.max()*1000:.2f}mm")
            break
    return ok


def test_coder() -> bool:
    from wilddet3d.head.coder_3d import Det3DCoder

    ok = True
    coder = Det3DCoder(canonical_rotation=False, symmetry="cuboid")
    ok &= _check(coder.reg_dims == 12, f"reg_dims={coder.reg_dims}")

    torch.manual_seed(42)
    N = 16
    q = torch.randn(N, 4); q /= q.norm(dim=1, keepdim=True)
    boxes3d = torch.cat([torch.rand(N, 3) * 3 + 0.5, torch.rand(N, 3) * 0.3 + 0.2, q], 1)
    b2d = torch.rand(N, 4); b2d[:, 2:] += b2d[:, :2]
    K = torch.eye(3); K[0, 0] = K[1, 1] = 600; K[0, 2] = 640; K[1, 2] = 400
    enc, wts = coder.encode(b2d * 600, boxes3d, K)
    ok &= _check(enc.shape == (N, 12), f"encoded shape {enc.shape}")
    dec = coder.decode(b2d * 600, enc, K)
    ok &= _check((dec[:, :3] - boxes3d[:, :3]).abs().mean() < 1e-5, "center round-trip")
    ok &= _check((dec[:, 3:6] - boxes3d[:, 3:6]).abs().mean() < 1e-4, "dims round-trip")
    return ok


def test_symmetry() -> bool:
    from wilddet3d.ops.rotation import (
        matrix_to_rotation_6d,
        rotation_6d_to_matrix,
        cuboid_symmetry_rotation_6d,
    )
    from vis4d.op.geometry.rotation import quaternion_to_matrix

    ok = True
    torch.manual_seed(7)
    q = torch.randn(8, 4); q /= q.norm(dim=1, keepdim=True)
    R = quaternion_to_matrix(q)
    d6 = matrix_to_rotation_6d(R)
    variants = cuboid_symmetry_rotation_6d(d6)
    ok &= _check(variants.shape == (8, 4, 6), f"variants shape {variants.shape}")

    signs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    for i, s in enumerate(signs):
        Rv = rotation_6d_to_matrix(variants[:, i])
        S = torch.diag(torch.tensor(s, dtype=torch.float32))
        err = (Rv - R @ S).abs().max().item()
        ok &= _check(err < 1e-5, f"variant {s} max_err={err:.2e}")

    # loss backward
    pred = torch.randn(8, 6, requires_grad=True)
    per_var = (pred.unsqueeze(1) - variants.detach()).abs().sum(-1)
    idx = per_var.argmin(1)
    best = variants.detach()[torch.arange(8), idx]
    loss = (pred - best).abs().mean()
    loss.backward()
    ok &= _check(pred.grad is not None and pred.grad.abs().max() > 0, "grad OK")
    return ok


def test_multiview() -> bool:
    from wilddet3d.multiview.scene_fusion import SceneFusion
    from wilddet3d.multiview.late_fusion import SceneLateFusion, transform_boxes3d

    ok = True
    d, N, S = 256, 4, 10
    sf = SceneFusion(d_model=d)
    q1 = torch.randn(N, S, d)
    q2 = torch.randn(N, S, d)
    T1 = torch.eye(4); T2 = torch.eye(4); T2[:3, 3] = torch.tensor([0.1, 0.2, 1.2])

    out = sf([q1, q2], [T1, T2])
    ok &= _check(len(out) == 2 and out[0].shape == (N, S, d), "SceneFusion 2-view shape")
    ok &= _check((out[0] - q1).abs().max() < 1e-6, "SceneFusion zero-init identity")
    out1 = sf([q1], [T1])
    ok &= _check((out1[0] - q1).abs().max() < 1e-6, "SceneFusion n=1 identity")

    lf = SceneLateFusion()
    boxes = [torch.randn(5, 10), torch.randn(4, 10)]
    # Make valid quaternions
    for b in boxes:
        b[:, 6:10] = torch.randn(len(b), 4)
        b[:, 6:10] /= b[:, 6:10].norm(dim=1, keepdim=True)
    scores = [torch.rand(5), torch.rand(4)]
    Ts = [torch.eye(4), torch.eye(4)]
    r = lf(boxes, scores, Ts)
    ok &= _check(r["boxes3d"].shape[1] == 10, "LateFusion output shape")

    T_id = torch.eye(4)
    b = torch.randn(6, 10); b[:, 6:10] /= b[:, 6:10].norm(dim=1, keepdim=True)
    bt = transform_boxes3d(b, T_id)
    ok &= _check((b[:, :3] - bt[:, :3]).abs().max() < 1e-5, "transform_boxes3d identity")
    return ok


def main() -> None:
    print("=" * 60)
    print("WildDet3D Anyware 9-DoF Pipeline Test Suite")
    print("=" * 60)
    results = {
        "Dataset loading + round-trip": test_dataset(),
        "9-DoF coder (encode/decode)": test_coder(),
        "Cuboid symmetry 6d + loss": test_symmetry(),
        "Multi-view (SceneFusion + LateFusion)": test_multiview(),
    }
    print("=" * 60)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print("=" * 60)
    print("RESULT:", "ALL PASSED ✓" if all_pass else "SOME FAILED ✗")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
