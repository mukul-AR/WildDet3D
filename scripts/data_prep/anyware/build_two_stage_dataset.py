#!/usr/bin/env python3
"""Build the cached two-stage 9-DoF dataset for Anyware warehouse scenes.

Joins the Omni3D-format COCO annotations (per-view boxes: visual geometry,
2D bbox, SKU prior, intrinsics, extrinsics) produced by
``convert_anyware_to_omni3d.py`` with the original ``scene.json`` (container
walls + scene SKU list), then caches, per box:

    * RGB-D crop (Stage-1 input)              -> rgb.npy [N,3,S,S] uint8,
                                                 depth.npy [N,1,S,S] uint16 (mm)
    * geometric context + anchor center        -> meta.npz: ctx, anchor_center
    * camera->base extrinsics                  -> meta.npz: T_base_cam
    * container walls (L/R/B plane + valid)    -> meta.npz: walls [N,3,5]
    * scene SKU candidates (sorted dims)       -> meta.npz: skus  [N,K,4]
    * GT visual/actual 9-DoF OBB (cam frame)   -> meta.npz: gt_center, gt_size,
                                                 gt_rot6d  (+ confidence, has_sku)

Box-local size ordering matches the box rotation columns: size = geometry
[gx, gy, gz] = reverse of the COCO "dimensions" [W, H, L] field.

Usage:
    python scripts/data_prep/anyware/build_two_stage_dataset.py \
        --data-root data/anyware_scenes --split train --crop-size 48
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import yaml

MAX_DEPTH = 10.0  # meters (warehouse cameras see ~0.3-4 m)
MAX_SKU = 8


def rot_matrix_to_6d(r: np.ndarray) -> np.ndarray:
    """First two rows of R (matches twostage rotation_utils convention)."""
    return r[:2, :].reshape(6).astype(np.float32)


def parse_scene(scene_json_path: str) -> dict | None:
    """Parse walls (L/R/B) and the scene SKU list from scene.json."""
    if not os.path.exists(scene_json_path):
        return None
    with open(scene_json_path) as f:
        sj = json.load(f)
    scene = sj.get("scene", {})
    walls = np.zeros((3, 5), dtype=np.float32)  # left, right, bottom
    container_yaml = scene.get("container_yaml_string", "")
    if container_yaml:
        cont = yaml.safe_load(container_yaml) or {}
        for i, name in enumerate(("left_wall", "right_wall", "bottom_wall")):
            w = cont.get(name)
            if w and w.get("valid", False) and w.get("plane_model"):
                pm = np.array(w["plane_model"], dtype=np.float32)  # [a,b,c,d]
                n = np.linalg.norm(pm[:3]) + 1e-9
                walls[i, :3] = pm[:3] / n
                walls[i, 3] = pm[3] / n
                walls[i, 4] = 1.0

    skus = np.zeros((MAX_SKU, 4), dtype=np.float32)
    skus_yaml = scene.get("skus_yaml_string", "")
    if skus_yaml:
        sku_list = yaml.safe_load(skus_yaml) or []
        if isinstance(sku_list, list):
            k = 0
            for s in sku_list:
                geom = s.get("geometry") if isinstance(s, dict) else None
                if not geom or len(geom) < 3:
                    continue
                dims = np.sort(np.array(geom[:3], dtype=np.float32))  # invariant
                skus[k, :3] = dims
                skus[k, 3] = 1.0
                k += 1
                if k >= MAX_SKU:
                    break
    return {"walls": walls, "skus": skus}


def crop_resize(
    img: np.ndarray,
    box: list[float],
    size: int,
    pad: float = 0.15,
    nearest: bool = False,
) -> np.ndarray:
    """Square crop around the 2D bbox (with padding) resized to size x size.

    Depth maps must use ``nearest=True``: averaging interpolation would blend
    metric depth with invalid (zero) pixels and corrupt the values.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (0.5 + pad)
    half = max(half, 4.0)
    sx1, sy1 = int(round(cx - half)), int(round(cy - half))
    sx2, sy2 = int(round(cx + half)), int(round(cy + half))
    sx1c, sy1c = max(sx1, 0), max(sy1, 0)
    sx2c, sy2c = min(sx2, w), min(sy2, h)
    if sx2c <= sx1c or sy2c <= sy1c:
        return np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    patch = img[sy1c:sy2c, sx1c:sx2c]
    # pad back to square so aspect ratio is preserved
    top, left = sy1c - sy1, sx1c - sx1
    bottom, right = sy2 - sy2c, sx2 - sx2c
    patch = cv2.copyMakeBorder(
        patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)


def bbox_depth_median(depth: np.ndarray, box: list[float]) -> tuple[float, float]:
    """Robust valid-depth median (m) over the central region of a 2D bbox.

    Computed from the original-resolution depth map (never a resized crop) so
    invalid pixels are excluded rather than averaged in.

    Returns:
        (median_depth_m, valid_fraction). median is 0.0 if no valid pixels.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # central 60% of the bbox to avoid edges / background
    mx1 = int(round(max(x1 + 0.2 * bw, 0)))
    my1 = int(round(max(y1 + 0.2 * bh, 0)))
    mx2 = int(round(min(x2 - 0.2 * bw, w)))
    my2 = int(round(min(y2 - 0.2 * bh, h)))
    if mx2 <= mx1 or my2 <= my1:
        return 0.0, 0.0
    region = depth[my1:my2, mx1:mx2]
    valid = region[region > 0]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.median(valid)) / 1000.0, float((region > 0).mean())


def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/anyware_scenes")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--crop-size", type=int, default=48)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    ann_path = os.path.join(
        args.data_root, "annotations", f"AnywareScenes_{args.split}.json"
    )
    with open(ann_path) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Optional scene subset (keep whole scenes together).
    scene_ids = [s["scene_id"] for s in coco["scenes"]]
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]
    keep_scenes = set(scene_ids)

    scene_cache: dict[str, dict | None] = {}
    s = args.crop_size

    rgb_list: list[np.ndarray] = []
    depth_list: list[np.ndarray] = []
    ctx_list, anchor_list, tbc_list = [], [], []
    walls_list, skus_list = [], []
    gt_center, gt_size, gt_rot6d = [], [], []
    conf_list, hassku_list, scene_idx_list = [], [], []
    scene_id_to_idx: dict[str, int] = {}

    img_ids = sorted(images.keys())
    n_img = len(img_ids)
    for ii, img_id in enumerate(img_ids):
        im = images[img_id]
        scene_id = im.get("scene_id", "")
        if scene_id not in keep_scenes:
            continue
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue
        img_path = os.path.join(args.data_root, im["file_path"]) if not os.path.isabs(
            im["file_path"]
        ) else im["file_path"]
        if not os.path.exists(img_path):
            img_path = im["file_path"]
        depth_path = img_path.replace("image.jpg", "depth.png")
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if depth is None:
            depth = np.zeros((h, w), dtype=np.uint16)

        k_mat = np.array(im["K"], dtype=np.float64)
        fx, fy = k_mat[0, 0], k_mat[1, 1]
        cx0, cy0 = k_mat[0, 2], k_mat[1, 2]
        t_base_cam = np.array(im["T_base_cam"], dtype=np.float32)

        scene = scene_cache.get(scene_id)
        if scene_id not in scene_cache:
            scene_dir = os.path.dirname(os.path.dirname(img_path))
            scene = parse_scene(os.path.join(scene_dir, "scene.json"))
            scene_cache[scene_id] = scene
        if scene is None:
            continue
        if scene_id not in scene_id_to_idx:
            scene_id_to_idx[scene_id] = len(scene_id_to_idx)

        for a in anns:
            box2d = a.get("bbox2D_trunc")
            if not box2d or box2d[2] <= box2d[0] or box2d[3] <= box2d[1]:
                continue
            rgb_crop = crop_resize(rgb, box2d, s)  # [s,s,3]
            depth_crop = crop_resize(
                depth.astype(np.uint16), box2d, s, nearest=True
            )  # [s,s]

            # anchor depth from the ORIGINAL-resolution depth (valid pixels)
            depth_med, valid_frac = bbox_depth_median(depth, box2d)
            if depth_med <= 0.0:
                depth_med = 2.0  # fallback so anchor is finite

            # anchor center in camera frame (back-projected bbox center)
            uc = (box2d[0] + box2d[2]) / 2.0
            vc = (box2d[1] + box2d[3]) / 2.0
            anchor = np.array(
                [
                    (uc - cx0) * depth_med / fx,
                    (vc - cy0) * depth_med / fy,
                    depth_med,
                ],
                dtype=np.float32,
            )

            bw = box2d[2] - box2d[0]
            bh = box2d[3] - box2d[1]
            ctx = np.array(
                [
                    uc / w,
                    vc / h,
                    bw / w,
                    bh / h,
                    fx / w,
                    fy / h,
                    cx0 / w,
                    cy0 / h,
                    min(depth_med / MAX_DEPTH, 1.0),
                    valid_frac,
                ],
                dtype=np.float32,
            )

            # GT visual/actual OBB in camera frame.
            center_cam = np.array(a["center_cam"], dtype=np.float32)
            dims = a["dimensions"]  # [W, H, L]
            size_local = np.array(
                [dims[2], dims[1], dims[0]], dtype=np.float32
            )  # [gx, gy, gz] aligned to R columns
            r_cam = np.array(a["R_cam"], dtype=np.float32)
            rot6d = rot_matrix_to_6d(r_cam)

            rgb_list.append(rgb_crop.transpose(2, 0, 1).astype(np.uint8))
            depth_list.append(depth_crop[None].astype(np.uint16))
            ctx_list.append(ctx)
            anchor_list.append(anchor)
            tbc_list.append(t_base_cam)
            walls_list.append(scene["walls"])
            skus_list.append(scene["skus"])
            gt_center.append(center_cam)
            gt_size.append(size_local)
            gt_rot6d.append(rot6d)
            conf_list.append(np.float32(a.get("gt_confidence", 1.0)))
            hassku_list.append(np.float32(bool(a.get("has_sku_prior", False))))
            scene_idx_list.append(np.int64(scene_id_to_idx[scene_id]))

        if (ii + 1) % 50 == 0 or ii + 1 == n_img:
            print(f"  [{ii + 1}/{n_img}] boxes so far: {len(rgb_list)}")

    n = len(rgb_list)
    if n == 0:
        raise SystemExit("No boxes cached - check data paths.")

    out_dir = args.out_dir or os.path.join(
        args.data_root, "two_stage_cache", args.split
    )
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "rgb.npy"), np.stack(rgb_list))
    np.save(os.path.join(out_dir, "depth.npy"), np.stack(depth_list))
    np.savez(
        os.path.join(out_dir, "meta.npz"),
        ctx=np.stack(ctx_list),
        anchor_center=np.stack(anchor_list),
        T_base_cam=np.stack(tbc_list),
        walls=np.stack(walls_list),
        skus=np.stack(skus_list),
        gt_center_cam=np.stack(gt_center),
        gt_size=np.stack(gt_size),
        gt_rot6d_cam=np.stack(gt_rot6d),
        confidence=np.stack(conf_list),
        has_sku=np.stack(hassku_list),
        scene_idx=np.stack(scene_idx_list),
        max_depth=np.float32(MAX_DEPTH),
        crop_size=np.int64(s),
    )
    print(
        f"Cached {n} boxes from {len(scene_id_to_idx)} scenes -> {out_dir}\n"
        f"  rgb.npy {np.stack(rgb_list).shape}  depth.npy "
        f"{np.stack(depth_list).shape}"
    )


if __name__ == "__main__":
    main()
