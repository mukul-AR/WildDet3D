#!/usr/bin/env python3
"""Visualize the prompt-free dense 9-DoF detector's outputs.

Runs the trained JENGA dense model on validation images (NO prompts) and saves
annotated PNGs with the predicted 9-DoF boxes (green) and GT boxes (red)
projected onto the RGB.

    PYTHONPATH=. .venv/bin/python scripts/visualize_dense_inference.py \
        --num-images 6 --score-thresh 0.3 --out viz_out
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.dataset import DenseAnywareDataset, dense_collate  # noqa: E402
from wilddet3d.dense.decode import decode_dense  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIGNS = np.array(list(itertools.product([-0.5, 0.5], repeat=3)), dtype=np.float32)
_EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if np.sum(_SIGNS[a] != _SIGNS[b]) == 1
]


def project_corners(center, size, r, k):
    """8 OBB corners (camera frame) projected to pixels; returns [8,2], z[8]."""
    corners = (r @ (_SIGNS * size).T).T + center  # [8,3]
    z = corners[:, 2]
    uv = (k @ corners.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def draw_boxes(img, boxes, k, color, thickness=2):
    """Draw projected OBB wireframes; skip boxes with a corner behind camera."""
    for center, size, r in boxes:
        uv, z = project_corners(center, size, r, k)
        if (z <= 0.05).any():
            continue
        for a, b in _EDGES:
            pa = tuple(np.round(uv[a]).astype(int))
            pb = tuple(np.round(uv[b]).astype(int))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def denorm(img_t):
    """ImageNet-normalized CHW tensor -> HWC uint8 BGR for cv2."""
    img = img_t.numpy().transpose(1, 2, 0) * _STD + _MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ckpt", default="ckpt/dense_9dof/dense_9dof_last.pt")
    ap.add_argument("--base-ckpt", default="ckpt/wilddet3d_stage5_anyware_9dof_2ep.ckpt")
    ap.add_argument("--data-root", default="data/anyware_scenes")
    ap.add_argument("--sim-root", default="", help="visualize sim scenes (.../scenes/synth) instead of COCO")
    ap.add_argument("--sim-target", default="actual", choices=["actual", "visible"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--fpn-level", type=int, default=1)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--out", default="viz_out")
    args = ap.parse_args()

    base = args.base_ckpt if os.path.exists(args.base_ckpt) else None
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=base, fpn_level=args.fpn_level, train_fusion=True
    )
    sd = torch.load(args.dense_ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().cuda()

    if args.sim_root:
        from wilddet3d.dense.sim_dataset import SimDenseDataset

        ds = SimDenseDataset(args.sim_root, args.size, args.sim_target)
    else:
        ds = DenseAnywareDataset(args.data_root, args.split, args.size)
    os.makedirs(args.out, exist_ok=True)

    for idx in range(min(args.num_images, len(ds))):
        sample = ds[idx]
        batch = dense_collate([sample])
        with torch.no_grad():
            pred = model(batch["image"].cuda(), batch["depth"].cuda(), batch["K"].cuda())
        stride = args.size / pred["heatmap"].shape[-1]
        dets = decode_dense(
            pred["heatmap"].float(), pred["reg"].float(), batch["K"].cuda(), stride,
            topk=200, score_thresh=args.score_thresh,
        )[0]

        img = denorm(sample["image"])
        k = sample["K"].numpy()
        # GT (red)
        gt = [
            (sample["centers"][i].numpy(), sample["sizes"][i].numpy(),
             _r6(sample["rot6d"][i].numpy()))
            for i in range(sample["centers"].shape[0])
        ]
        draw_boxes(img, gt, k, (0, 0, 255), 1)
        # predictions (green)
        pr = [
            (dets["center"][i].cpu().numpy(), dets["size"][i].cpu().numpy(),
             dets["R"][i].cpu().numpy())
            for i in range(dets["center"].shape[0])
        ]
        draw_boxes(img, pr, k, (0, 255, 0), 2)
        cv2.putText(img, f"pred={len(pr)} (green)  GT={len(gt)} (red)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        path = os.path.join(args.out, f"{args.split}_{idx:03d}.png")
        cv2.imwrite(path, img)
        print(f"saved {path}  (pred={len(pr)}, GT={len(gt)})")

    print(f"\nDone. Open the PNGs in '{args.out}/' to see the 9-DoF detections.")


def _r6(d6):
    """6D -> 3x3 rotation (numpy, Gram-Schmidt; matches rotation_utils)."""
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


if __name__ == "__main__":
    main()
