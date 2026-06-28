#!/usr/bin/env python3
"""Export JENGA end-to-end predicted ACTUAL boxes per scene -> world-frame JSON.

For each scene/camera: run the full inference chain (Stage-1 dense detect ->
visible boxes -> Stage-2 dim-conditioned completion), transform the predicted
camera-frame actual boxes into world frame, and write one JSON per scene
(``<pred-dir>/<scene_id>.json``) in the same ``extrinsic_4x4`` + ``geometry``
format the viz tool already uses for GT boxes. The viewer overlays these.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/jenga_export_pred.py \
        --ckpt ckpt/jenga/jenga_last.pt \
        --scenes-dir /storage/3dl_sim_data/scenes \
        --pred-dir /storage/3dl_sim_data/preds --max-scenes 40 --score-thresh 0.3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.decode import decode_dense, decode_jenga  # noqa: E402
from wilddet3d.dense.jenga_utils import parse_catalog  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.sim_dataset import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _resize_pad,
)
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402


def load_view(cam_dir: str, size: int):
    """Return (image[1,3,H,W], depth[1,1,H,W], K[1,3,3], E[4,4]) for one camera."""
    meta = json.load(open(os.path.join(cam_dir, "metadata.json")))
    intr = meta["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    e = np.array(meta["camera_extrinsic_4x4"], dtype=np.float64)  # cam->world
    rgb = cv2.cvtColor(cv2.imread(os.path.join(cam_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(cam_dir, "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
    rgb_p, scale, px, py = _resize_pad(rgb, size, nearest=False)
    depth_p, _, _, _ = _resize_pad(depth.astype(np.uint16), size, nearest=True)
    img = (rgb_p.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    img = torch.from_numpy(img.transpose(2, 0, 1))[None]
    depth_m = torch.from_numpy((depth_p.astype(np.float32) / 1000.0)[None, None])
    k = torch.tensor(
        [[fx * scale, 0, cx * scale + px], [0, fy * scale, cy * scale + py], [0, 0, 1]],
        dtype=torch.float32,
    )[None]
    return img, depth_m, k, e


def to_world_extrinsic(center: np.ndarray, r: np.ndarray, e: np.ndarray) -> list:
    """Camera-frame box (center, R) + cam->world E -> world 4x4 extrinsic."""
    rw = e[:3, :3] @ r
    tw = e[:3, :3] @ center + e[:3, 3]
    m = np.eye(4)
    m[:3, :3] = rw
    m[:3, 3] = tw
    return m.tolist()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes-dir", default="/storage/3dl_sim_data/scenes")
    ap.add_argument("--pred-dir", default="/storage/3dl_sim_data/preds")
    ap.add_argument("--wilddet3d-ckpt", default=None)
    ap.add_argument("--max-scenes", type=int, default=40)
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (overrides max-scenes)")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    base = args.wilddet3d_ckpt or a.get("wilddet3d_ckpt")
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=base if base and os.path.exists(base) else None,
        fpn_level=a.get("fpn_level", 1), train_fusion=True,
        head_kwargs={"feat_ch": a.get("head_width", 256), "n_convs": a.get("head_convs", 4)},
        device=args.device)
    model.load_state_dict(ck["model"])
    stage2 = JengaStage2(in_ch=256, d_model=a.get("d_model", 512),
                         layers=a.get("layers", 12), heads=a.get("heads", 8)).to(args.device)
    stage2.load_state_dict(ck["stage2"])
    model.eval()
    stage2.eval()

    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",")]
        scene_dirs = [os.path.join(args.scenes_dir, s) for s in scene_ids]
    else:
        scene_dirs = sorted(glob.glob(os.path.join(args.scenes_dir, "synth_*")))[: args.max_scenes]
    os.makedirs(args.pred_dir, exist_ok=True)

    for sd in scene_dirs:
        sid = os.path.basename(sd)
        sj = os.path.join(sd, "scene.json")
        if not os.path.exists(sj):
            continue
        catalog = torch.from_numpy(parse_catalog(sj)).to(args.device)
        pred_boxes, bi = {}, 0
        for cam_dir in sorted(glob.glob(os.path.join(sd, "*camera*"))):
            if not os.path.exists(os.path.join(cam_dir, "metadata.json")):
                continue
            img, depth, k, e = load_view(cam_dir, args.size)
            img, depth, k = img.to(args.device), depth.to(args.device), k.to(args.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                out = model(img, depth, k, return_feat=True)
            heatmap, reg = out["heatmap"].float(), out["reg"].float()
            feat, stride = out["feat"].float(), out["stride"]
            dets = decode_dense(heatmap, reg, k, stride, score_thresh=args.score_thresh)
            res = decode_jenga(dets, feat, stride, stage2, [catalog], k)[0]
            cam = os.path.basename(cam_dir)
            for j in range(res["center"].shape[0]):
                pred_boxes[str(bi)] = {
                    "extrinsic_4x4": to_world_extrinsic(
                        res["center"][j].cpu().numpy().astype(np.float64),
                        res["R"][j].cpu().numpy().astype(np.float64), e),
                    "geometry": res["size"][j].cpu().numpy().tolist(),
                    "score": float(res["score"][j]),
                    "camera": cam,
                }
                bi += 1
        json.dump({"pred_boxes": pred_boxes},
                  open(os.path.join(args.pred_dir, f"{sid}.json"), "w"))
        print(f"{sid}: {len(pred_boxes)} predicted boxes ({len(catalog)} catalog dims)", flush=True)
    print(f"\nwrote predictions -> {args.pred_dir}", flush=True)


if __name__ == "__main__":
    main()
