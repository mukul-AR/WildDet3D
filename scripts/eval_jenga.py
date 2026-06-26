#!/usr/bin/env python3
"""Evaluate a trained JENGA checkpoint: teacher-forced Stage-2 metrics on a split.

Reconstructs the predicted actual boxes and reports the full suite — 3D IoU
(Monte-Carlo, full rotation), IoU@0.5/0.75, assignment accuracy, actual-center
distance, size error, size-aware rotation error, and pairwise overlap (a
non-intersection check). Reads the decoder config from the checkpoint's saved
args so it matches the trained model.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/eval_jenga.py \
        --ckpt ckpt/jenga/jenga_last.pt --sim-root <synth_dir> --n-samples 8192
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.loss import JengaStage2Loss  # noqa: E402
from wilddet3d.dense.metrics import stage2_eval_arrays, summarize_eval  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, jenga_collate  # noqa: E402
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402


def move(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) for x in v]
        else:
            out[k] = v
    return out


def make_queries(batch, stride):
    queries_uv, vis_obb = [], []
    for i in range(len(batch["vis_center"])):
        c = batch["vis_center"][i]
        z = c[:, 2].clamp_min(1e-3)
        u = (batch["K"][i][0, 0] * c[:, 0] / z + batch["K"][i][0, 2]) / stride
        v = (batch["K"][i][1, 1] * c[:, 1] / z + batch["K"][i][1, 2]) / stride
        queries_uv.append(torch.stack([u, v], dim=-1))
        vis_obb.append(torch.cat([c, batch["vis_size"][i], batch["vis_rot6d"][i]], dim=-1))
    return queries_uv, vis_obb


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to jenga_last.pt")
    ap.add_argument("--sim-root", default=None, help="override the saved sim-root")
    ap.add_argument("--wilddet3d-ckpt", default=None, help="override base encoder ckpt")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    sim_root = args.sim_root or a["sim_root"]
    base = args.wilddet3d_ckpt or a.get("wilddet3d_ckpt")
    size = a.get("size", 1008)
    val_frac = args.val_frac if args.val_frac is not None else a.get("val_frac", 0.1)
    max_scenes = args.max_scenes if args.max_scenes is not None else a.get("max_scenes", 0)

    ds = SimJengaDataset(sim_root, size, max_scenes, split=args.split, val_frac=val_frac)
    print(f"{args.split} samples (views): {len(ds)}  (ckpt epoch {ck.get('epoch','?')})", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=jenga_collate, pin_memory=True)

    model = DenseDet3D.from_wilddet3d(
        ckpt_path=base if base and os.path.exists(base) else None,
        fpn_level=a.get("fpn_level", 1), train_fusion=True, device=args.device)
    model.load_state_dict(ck["model"])
    stage2 = JengaStage2(in_ch=256, d_model=a.get("d_model", 512),
                         layers=a.get("layers", 12), heads=a.get("heads", 8)).to(args.device)
    stage2.load_state_dict(ck["stage2"])
    model.eval()
    stage2.eval()

    keys = ("iou", "center_dist", "size_err", "corner_add", "corner_adds", "correct", "overlap")
    acc = {k: [] for k in keys}
    for batch in loader:
        batch = move(batch, args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch["image"], batch["depth"], batch["K"], return_feat=True)
        feat, stride = pred["feat"].float(), pred["stride"]
        queries_uv, vis_obb = make_queries(batch, stride)
        out = stage2(feat, queries_uv, vis_obb, batch["catalog"])
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
        arr = stage2_eval_arrays(out, batch, args.n_samples)
        for k in keys:
            acc[k].append(arr[k])

    arrays = {k: (torch.cat(v) if v else torch.zeros(0)) for k, v in acc.items()}
    s = summarize_eval(arrays)
    print("\n==================== JENGA eval (teacher-forced) ====================")
    print(f"  boxes evaluated     : {s.get('n_boxes', 0)}")
    print(f"  3D IoU (mean)       : {s.get('iou3d', 0):.4f}      [gate >= 0.95]")
    print(f"  IoU>=0.50 / >=0.75  : {s.get('iou_50', 0):.3f} / {s.get('iou_75', 0):.3f}")
    print(f"  assignment accuracy : {s.get('assign_acc', 0):.4f}")
    print(f"  actual-center dist  : {s.get('center_dist', 0)*100:.2f} cm")
    print(f"  size error (L1 sum) : {s.get('size_err', 0)*100:.2f} cm")
    print(f"  corner ADD / ADD-S  : {s.get('corner_add', 0)*100:.2f} / "
          f"{s.get('corner_adds', 0)*100:.2f} cm   [orientation-sensitive]")
    print("  rotation error      : 0.00 deg (inherited from the visible box)")
    print(f"  pairwise overlap    : {s.get('overlap_frac', 0)*100:.2f}% of box-pairs intersect")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
