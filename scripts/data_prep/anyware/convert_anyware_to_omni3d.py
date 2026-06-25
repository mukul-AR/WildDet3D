"""Convert Anyware capture-scene data to Omni3D-format COCO JSON.

Scene layout (synced from s3://anyware-perception-capture-scene-data/):
    {site}/{device}/{date}/place_{sec}_{nanosec}_{hash}/
        scene.json                     # 9-DoF GT boxes in base_link
        {idx}_{camera_name}/
            image.jpg                  # 1280x800 RGB
            depth.png                  # uint16 depth in mm (sparse, ~20% valid)
            metadata.json              # K (plumb_bob) + base_link extrinsics

GT pose convention (verified against anyware-core
anyware_plan/src/common/geometry/geometry.cpp::ConvertWPRToRotation):
    xyzwpr: xyz in meters (base_link), WPR euler angles in DEGREES,
    R = Rz(r) @ Ry(p) @ Rx(w)   (FANUC fixed-axis XYZ)
    geometry: [gx, gy, gz] box extents along the box-local x/y/z axes.

Omni3D/vis4d 10-dim box convention (AxisMode.OPENCV, see
vis4d.op.box.box3d.boxes3d_to_corners):
    boxes3d = [center(3), W, L, H, quat(4)] with local x<->L, y<->H, z<->W.
    Annotation "dimensions" field is [W, H, L] -> [gz, gy, gx].

Output: {out_root}/annotations/AnywareScenes_{train,val}.json
Each image entry carries scene/view metadata (scene_id, view_idx,
T_base_cam 4x4) so multi-view consumers can group views per scene.

Usage:
    python scripts/data_prep/anyware/convert_anyware_to_omni3d.py \
        --scene-roots data/anyware_scenes dataset \
        --out-root data/anyware_scenes
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # depth-based lidar_pts becomes unavailable
    cv2 = None

CATEGORY_NAME = "box"
CATEGORY_ID = 1

# Corner signs matching vis4d boxes3d_to_corners (AxisMode.OPENCV):
# local x extent = L, y extent = H, z extent = W
_CORNER_SIGNS = np.array(
    list(itertools.product([-0.5, 0.5], repeat=3)), dtype=np.float64
)


def wpr_deg_to_rotation(w: float, p: float, r: float) -> np.ndarray:
    """FANUC WPR (degrees) to rotation matrix: R = Rz(r) @ Ry(p) @ Rx(w)."""
    w, p, r = np.radians([w, p, r])
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_xyzw_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x, y, z, w) to rotation matrix."""
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_corners_cam(
    center_cam: np.ndarray, r_cam: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    """8 corners [8, 3] in camera frame. geometry = [gx, gy, gz] local extents."""
    corners_local = _CORNER_SIGNS * geometry  # box local frame
    return (r_cam @ corners_local.T).T + center_cam


def project(corners_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project corners; returns (uv [8, 2], z [8])."""
    z = corners_cam[:, 2]
    uv = (k @ corners_cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def load_scene_boxes(scene_json_path: str) -> list[dict] | None:
    """Parse scene.json -> list of box dicts, or None if invalid scene."""
    with open(scene_json_path) as f:
        sj = json.load(f)
    scene = sj.get("scene", {})
    if not scene.get("success", False):
        return None
    boxes_yaml = scene.get("boxes_yaml_string", "")
    if not boxes_yaml:
        return None
    boxes = yaml.safe_load(boxes_yaml)
    if not boxes:
        return None
    out = []
    for name, b in boxes.items():
        if not b.get("valid", True):
            continue
        # SKU dimension prior. matched_skus[].geometry holds the known
        # box extents [gx, gy, gz] (meters). known_box_extents_used flags
        # whether a SKU was matched (i.e. whether a dim prior is available).
        matched_skus = b.get("matched_skus") or []
        has_sku_prior = bool(matched_skus) and bool(
            b.get("known_box_extents_used", False)
        )
        sku_geometry = None
        sku_uuid = None
        sku_weight = None
        if matched_skus:
            sku = matched_skus[0]
            if sku.get("geometry") is not None:
                sku_geometry = [float(v) for v in sku["geometry"]]
            sku_uuid = sku.get("uuid")
            sku_weight = sku.get("weight")
        out.append(
            {
                "name": name,
                "xyzwpr": b["xyzwpr"],
                "geometry": b["geometry"],
                # SKU-snapped GT (cleaner when present); kept as optional
                # extras, does NOT replace the primary GT above.
                "meet_xyzwpr": b.get("meet_xyzwpr"),
                "meet_geometry": b.get("meet_geometry"),
                "confidence": float(b.get("confidence", 1.0)),
                "occlusion": float(b.get("occlusion", 0.0)),
                "uuid": b.get("uuid", name),
                # Dimension prior
                "has_sku_prior": has_sku_prior,
                "sku_geometry": sku_geometry,   # [gx, gy, gz] local extents
                "sku_uuid": sku_uuid,
                "sku_weight": sku_weight,
            }
        )
    return out


def depth_points_base(cap_dir: str, k: np.ndarray, t_base_cam: np.ndarray,
                      stride: int = 4, max_points: int = 30000) -> np.ndarray | None:
    """Backproject sparse depth to base_link points [N, 3] (subsampled)."""
    if cv2 is None:
        return None
    depth_path = os.path.join(cap_dir, "depth.png")
    if not os.path.exists(depth_path):
        return None
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d[::stride, ::stride]
    v, u = np.nonzero(d)
    if len(v) == 0:
        return None
    if len(v) > max_points:
        sel = np.random.default_rng(0).choice(len(v), max_points, replace=False)
        v, u = v[sel], u[sel]
    z = d[v, u].astype(np.float64) / 1000.0
    u = u * stride
    v = v * stride
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)
    r = t_base_cam[:3, :3]
    t = t_base_cam[:3, 3]
    return (r @ pts_cam.T).T + t


def count_points_in_obb(
    pts_base: np.ndarray | None,
    center_base: np.ndarray,
    r_base: np.ndarray,
    geometry: np.ndarray,
    margin: float = 0.01,
) -> int:
    """Count base_link points inside the oriented box (with margin)."""
    if pts_base is None:
        return -1  # unknown
    diag = np.linalg.norm(geometry)
    near = pts_base[np.linalg.norm(pts_base - center_base, axis=1) < diag]
    if len(near) == 0:
        return 0
    local = (r_base.T @ (near - center_base).T).T
    inside = (np.abs(local) <= geometry / 2 + margin).all(axis=1)
    return int(inside.sum())


def convert_scene(
    scene_dir: str,
    rel_root: str,
    image_id_start: int,
    ann_id_start: int,
    max_depth: float = 20.0,
) -> tuple[list[dict], list[dict], int, int]:
    """Convert one scene dir -> (images, annotations, next_img_id, next_ann_id)."""
    boxes = load_scene_boxes(os.path.join(scene_dir, "scene.json"))
    if boxes is None:
        return [], [], image_id_start, ann_id_start

    scene_id = os.path.basename(scene_dir.rstrip("/"))
    images: list[dict] = []
    annotations: list[dict] = []
    img_id = image_id_start
    ann_id = ann_id_start

    cap_dirs = sorted(
        d
        for d in glob.glob(os.path.join(scene_dir, "*_camera_*"))
        if os.path.isdir(d)
    )
    for view_idx, cap_dir in enumerate(cap_dirs):
        meta_path = os.path.join(cap_dir, "metadata.json")
        img_path = os.path.join(cap_dir, "image.jpg")
        if not (os.path.exists(meta_path) and os.path.exists(img_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        cam_info = meta["camera_info"]
        height, width = cam_info["height"], cam_info["width"]
        k = np.array(cam_info["k"], dtype=np.float64).reshape(3, 3)

        ext = meta["extrinsics"]
        t_vec = np.array([ext["translation"][a] for a in "xyz"])
        q = ext["rotation"]
        r_base_cam = quat_xyzw_to_rotation(q["x"], q["y"], q["z"], q["w"])
        t_base_cam = np.eye(4)
        t_base_cam[:3, :3] = r_base_cam
        t_base_cam[:3, 3] = t_vec
        r_cam_base = r_base_cam.T
        t_cam_base = -r_cam_base @ t_vec

        pts_base = depth_points_base(cap_dir, k, t_base_cam)

        file_path = os.path.relpath(img_path, rel_root)
        image_entry = {
            "id": img_id,
            "file_path": file_path,
            "width": width,
            "height": height,
            "K": k.tolist(),
            # Multi-view metadata (non-standard Omni3D extensions)
            "scene_id": scene_id,
            "view_idx": view_idx,
            "camera_name": meta.get("camera_name", ""),
            "T_base_cam": t_base_cam.tolist(),
            "src_dataset": "AnywareScenes",
        }

        n_anns_this_view = 0
        for b in boxes:
            x, y, z, w_ang, p_ang, r_ang = b["xyzwpr"]
            center_base = np.array([x, y, z])
            geometry = np.array(b["geometry"], dtype=np.float64)
            r_box_base = wpr_deg_to_rotation(w_ang, p_ang, r_ang)

            center_cam = r_cam_base @ center_base + t_cam_base
            r_cam = r_cam_base @ r_box_base

            corners = box_corners_cam(center_cam, r_cam, geometry)
            uv, zs = project(corners, k)

            behind = bool((zs <= 0.05).all())
            if behind:
                continue

            # Unclipped projected bbox
            x1, y1 = uv.min(axis=0)
            x2, y2 = uv.max(axis=0)
            # Clipped to image
            cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
            cx2, cy2 = min(x2, float(width)), min(y2, float(height))
            if cx2 <= cx1 or cy2 <= cy1:
                continue  # entirely outside the image
            full_area = max((x2 - x1) * (y2 - y1), 1e-6)
            clip_area = (cx2 - cx1) * (cy2 - cy1)
            truncation = float(1.0 - clip_area / full_area)

            lidar_pts = count_points_in_obb(
                pts_base, center_base, r_box_base, geometry
            )
            if lidar_pts < 0:
                lidar_pts = 1  # depth unavailable; do not filter

            # dimensions = [W, H, L] = extents along local [z, y, x]
            dimensions = [
                float(geometry[2]),
                float(geometry[1]),
                float(geometry[0]),
            ]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_ID,
                    "category_name": CATEGORY_NAME,
                    "valid3D": True,
                    "behind_camera": False,
                    "center_cam": [float(v) for v in center_cam],
                    "dimensions": dimensions,
                    "R_cam": r_cam.tolist(),
                    "bbox3D_cam": corners.tolist(),
                    "bbox2D_proj": [
                        float(x1), float(y1), float(x2), float(y2)
                    ],
                    "bbox2D_trunc": [
                        float(cx1), float(cy1), float(cx2), float(cy2)
                    ],
                    "bbox2D_tight": [-1, -1, -1, -1],
                    "truncation": truncation,
                    "visibility": -1.0,  # unknown per-view; lidar_pts covers it
                    "lidar_pts": lidar_pts,
                    "segmentation_pts": -1,
                    "depth_error": -1.0,
                    "gt_confidence": b["confidence"],
                    "box_uuid": b["uuid"],
                    # SKU dimension prior. has_sku_prior=False marks the
                    # ~20% of boxes with NO matched SKU (model must predict
                    # dims). sku_dims is in the same [W, H, L] order as
                    # "dimensions" (extents along local z, y, x).
                    "has_sku_prior": bool(b.get("has_sku_prior", False)),
                    "sku_dims": (
                        [
                            float(b["sku_geometry"][2]),
                            float(b["sku_geometry"][1]),
                            float(b["sku_geometry"][0]),
                        ]
                        if b.get("sku_geometry") is not None
                        else [0.0, 0.0, 0.0]
                    ),
                    "sku_uuid": b.get("sku_uuid"),
                }
            )
            ann_id += 1
            n_anns_this_view += 1

        if n_anns_this_view > 0:
            images.append(image_entry)
            img_id += 1
        else:
            ann_id -= n_anns_this_view  # no-op; kept for clarity

    return images, annotations, img_id, ann_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-roots",
        nargs="+",
        required=True,
        help="Roots to scan recursively for place_* scene dirs.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root; writes {out_root}/annotations/AnywareScenes_*.json",
    )
    parser.add_argument(
        "--rel-root",
        default=".",
        help="file_path entries are stored relative to this dir (default cwd).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of scenes for the val split (hash-based, deterministic).",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=1,
        help="Minimum usable views for a scene to be included.",
    )
    args = parser.parse_args()

    scene_dirs = []
    for root in args.scene_roots:
        scene_dirs += [
            d
            for d in glob.glob(os.path.join(root, "**", "place_*"), recursive=True)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "scene.json"))
        ]
    scene_dirs = sorted(set(scene_dirs))
    print(f"Found {len(scene_dirs)} scenes")

    splits = {
        "train": {"images": [], "annotations": [], "scenes": []},
        "val": {"images": [], "annotations": [], "scenes": []},
    }
    img_id, ann_id = 1, 1
    n_skipped = 0
    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir.rstrip("/"))
        # Deterministic split on the scene hash suffix
        hash_suffix = scene_id.split("_")[-1]
        bucket = int(hash_suffix[-2:], 16) / 255.0 if hash_suffix else 0.5
        split = "val" if bucket < args.val_fraction else "train"

        images, annotations, img_id, ann_id = convert_scene(
            scene_dir, args.rel_root, img_id, ann_id
        )
        if len(images) < args.min_views:
            n_skipped += 1
            continue
        splits[split]["images"] += images
        splits[split]["annotations"] += annotations
        splits[split]["scenes"].append(
            {
                "scene_id": scene_id,
                "image_ids": [im["id"] for im in images],
            }
        )

    os.makedirs(os.path.join(args.out_root, "annotations"), exist_ok=True)
    categories = [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]
    for split, data in splits.items():
        out = {
            "info": {"description": f"AnywareScenes_{split}"},
            "images": data["images"],
            "annotations": data["annotations"],
            "categories": categories,
            "scenes": data["scenes"],
        }
        out_path = os.path.join(
            args.out_root, "annotations", f"AnywareScenes_{split}.json"
        )
        with open(out_path, "w") as f:
            json.dump(out, f)
        n_multi = sum(1 for s in data["scenes"] if len(s["image_ids"]) > 1)
        print(
            f"{split}: {len(data['scenes'])} scenes "
            f"({n_multi} multi-view), {len(data['images'])} images, "
            f"{len(data['annotations'])} annotations -> {out_path}"
        )
    print(f"Skipped {n_skipped} scenes (invalid/empty)")


if __name__ == "__main__":
    main()
