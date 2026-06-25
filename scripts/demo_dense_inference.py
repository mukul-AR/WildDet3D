#!/usr/bin/env python3
"""Prompt-free inference demo for the dense 9-DoF detector.

Loads the trained dense model and runs it on a validation image with NO
prompt of any kind (just RGB + depth + intrinsics in), printing the decoded
9-DoF detections vs the GT box count.

    PYTHONPATH=. .venv/bin/python scripts/demo_dense_inference.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.dataset import DenseAnywareDataset, dense_collate  # noqa: E402
from wilddet3d.dense.decode import decode_dense  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.twostage.rotation_utils import (  # noqa: E402
    rad2deg,
    symmetry_min_geodesic,
    matrix_to_rotation_6d,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ckpt", default="ckpt/dense_9dof/dense_9dof_last.pt")
    ap.add_argument("--base-ckpt", default="ckpt/wilddet3d_stage5_anyware_9dof_2ep.ckpt")
    ap.add_argument("--data-root", default="data/anyware_scenes")
    ap.add_argument("--num-images", type=int, default=3)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--fpn-level", type=int, default=1)
    ap.add_argument("--size", type=int, default=1008)
    args = ap.parse_args()

    base = args.base_ckpt if os.path.exists(args.base_ckpt) else None
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=base, fpn_level=args.fpn_level, train_fusion=True
    )
    sd = torch.load(args.dense_ckpt, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded dense ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().cuda()

    ds = DenseAnywareDataset(args.data_root, "val", args.size)
    print(f"val images: {len(ds)}\n")

    for idx in range(min(args.num_images, len(ds))):
        sample = ds[idx]
        batch = dense_collate([sample])
        img = batch["image"].cuda()
        depth = batch["depth"].cuda()
        k = batch["K"].cuda()

        # *** No prompt of any kind — just RGBD + intrinsics. ***
        with torch.no_grad():
            pred = model(img, depth, k)
        stride = args.size / pred["heatmap"].shape[-1]
        dets = decode_dense(
            pred["heatmap"].float(), pred["reg"].float(), k, stride,
            topk=200, score_thresh=args.score_thresh,
        )[0]

        n_gt = sample["centers"].shape[0]
        n_det = dets["center"].shape[0]
        print(f"image {idx}: GT boxes={n_gt}  detections={n_det}")
        if n_det and n_gt:
            # nearest-center match for a rough quality read-out
            d = torch.cdist(dets["center"].cpu(), sample["centers"])
            j = d.argmin(1)
            ctr_err = d.gather(1, j[:, None]).squeeze(1)
            pred6 = matrix_to_rotation_6d(dets["R"].cpu())
            rot_err = rad2deg(
                symmetry_min_geodesic(pred6, sample["rot6d"][j])
            )
            keep = ctr_err < 0.5  # matched within 50 cm
            if keep.any():
                print(
                    f"  matched={int(keep.sum())}  "
                    f"center_err={ctr_err[keep].median()*100:.1f}cm  "
                    f"size(med)={dets['size'][keep.to(dets['size'].device)].median(0).values.tolist()}  "
                    f"rot_err={rot_err[keep].median():.1f}deg  "
                    f"top_score={dets['score'].max():.2f}"
                )
    print("\nPrompt-free inference OK (RGBD in -> 9-DoF boxes out, no prompts).")


if __name__ == "__main__":
    main()
