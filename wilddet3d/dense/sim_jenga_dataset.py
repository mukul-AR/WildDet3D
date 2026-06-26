"""Sim dataset for the JENGA two-stage head: per view, visible + actual OBBs,
the scene's candidate-dimension catalog, and per-box catalog assignment.

Reuses the RGB-D loading / resize-pad / intrinsic-adjust path of
``SimDenseDataset``; the actual box is axis-canonicalized (ascending extent)
and its size is matched to the scene catalog to produce the assignment index.
A deterministic per-scene hash split yields disjoint train/val sets.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from wilddet3d.dense.jenga_utils import assign_index, parse_catalog
from wilddet3d.dense.sim_dataset import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _SIGNS,
    _resize_pad,
)


def feat_cache_path(cache_dir: str, cam_dir: str) -> str:
    """Deterministic cache filename for a camera view's precomputed feature."""
    key = hashlib.md5(os.path.abspath(cam_dir).encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def _pad_geom(h: int, w: int, size: int) -> tuple[float, int, int]:
    """Aspect-preserving resize-pad geometry (matches ``_resize_pad``)."""
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return scale, (size - nw) // 2, (size - nh) // 2


def _in_split(scene_dir: str, split: str, val_frac: float) -> bool:
    if val_frac <= 0.0:
        return split == "train"
    h = int(hashlib.md5(os.path.basename(scene_dir).encode()).hexdigest(), 16)
    is_val = (h % 1000) < int(val_frac * 1000)
    return is_val if split == "val" else not is_val


class SimJengaDataset(Dataset):
    """Per-view RGBD + visible/actual 9-DoF boxes + scene catalog + assignment.

    Args:
        sim_root: directory containing ``synth_*/`` scene folders.
        size: square SAM3 input size (1008).
        max_scenes: optional cap on scenes (0 = all).
        min_visible: drop boxes with ``visible_fraction`` below this.
        split: ``"train"`` or ``"val"``.
        val_frac: fraction of scenes (by hash) held out for validation.
    """

    def __init__(
        self,
        sim_root: str,
        size: int = 1008,
        max_scenes: int = 0,
        min_visible: float = 0.05,
        split: str = "train",
        val_frac: float = 0.1,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.size = size
        self.min_visible = min_visible
        scene_dirs = sorted(glob.glob(os.path.join(sim_root, "synth_*")))
        if max_scenes > 0:
            scene_dirs = scene_dirs[:max_scenes]
        scene_dirs = [d for d in scene_dirs if _in_split(d, split, val_frac)]
        self.samples: list[tuple[str, str]] = []  # (cam_dir, scene_json)
        for sd in scene_dirs:
            sj = os.path.join(sd, "scene.json")
            if not os.path.exists(sj):
                continue
            for cam in sorted(glob.glob(os.path.join(sd, "*_camera_*"))):
                if os.path.exists(os.path.join(cam, "rgb.png")) and os.path.exists(
                    os.path.join(cam, "metadata.json")
                ):
                    self.samples.append((cam, sj))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        cam, scene_json = self.samples[idx]
        rgb = cv2.cvtColor(
            cv2.imread(os.path.join(cam, "rgb.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        depth = cv2.imread(os.path.join(cam, "depth.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        meta = json.load(open(os.path.join(cam, "metadata.json")))
        intr = meta["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        e_inv = np.linalg.inv(np.array(meta["camera_extrinsic_4x4"], dtype=np.float64))
        r_wc, t_wc = e_inv[:3, :3], e_inv[:3, 3]

        rgb_p, scale, px, py = _resize_pad(rgb, self.size, nearest=False)
        depth_p, _, _, _ = _resize_pad(depth.astype(np.uint16), self.size, nearest=True)
        img = (rgb_p.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        depth_m = torch.from_numpy((depth_p.astype(np.float32) / 1000.0)[None])
        k_adj = np.array(
            [
                [fx * scale, 0, cx * scale + px],
                [0, fy * scale, cy * scale + py],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        catalog = parse_catalog(scene_json)

        def cam_obb(ext, geom):
            c = r_wc @ ext[:3, 3] + t_wc
            r = r_wc @ ext[:3, :3]
            return c.astype(np.float32), r.astype(np.float32), geom.astype(np.float32)

        vis_c, vis_s, vis_r, vis_b = [], [], [], []
        act_c, act_s, act_r, assign = [], [], [], []
        for b in meta["boxes"].values():
            if b.get("visible_fraction", 1.0) < self.min_visible:
                continue
            vc, v_r, vg = cam_obb(
                np.array(b["visible_extrinsic_4x4"], dtype=np.float64),
                np.array(b["visible_geometry"], dtype=np.float32),
            )
            if vc[2] <= 0.05:
                continue
            ac, a_r, ag = cam_obb(
                np.array(b["actual_extrinsic_4x4"], dtype=np.float64),
                np.array(b["actual_geometry"], dtype=np.float32),
            )
            # The actual box shares the visible box's orientation exactly
            # (verified 0 deg dataset-wide), so we store its size per-axis
            # (NATIVE, same frame as visible) and the native rotation. No
            # canonicalization: Stage 2 inherits R from the visible box and only
            # selects the SKU + places the center. Assignment matches the sorted
            # dims to the scene catalog.

            # visible 2D box (native intrinsics) -> resized/padded px
            corners = (v_r @ (_SIGNS * vg).T).T + vc
            zc = np.clip(corners[:, 2], 1e-6, None)
            u = fx * corners[:, 0] / zc + cx
            v = fy * corners[:, 1] / zc + cy
            x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()

            vis_c.append(vc.tolist())
            vis_s.append(vg.tolist())
            vis_r.append(v_r[:2].reshape(6).tolist())
            vis_b.append(
                [x1 * scale + px, y1 * scale + py, x2 * scale + px, y2 * scale + py]
            )
            act_c.append(ac.tolist())
            act_s.append(ag.tolist())  # native per-axis extents
            act_r.append(a_r[:2].reshape(6).tolist())  # native R_act (== R_vis)
            assign.append(assign_index(np.sort(ag), catalog))

        n = len(vis_c)

        def t(x, d):
            return torch.tensor(x, dtype=torch.float32).reshape(n, d)

        return {
            "image": img,
            "depth": depth_m,
            "K": torch.from_numpy(k_adj),
            "vis_center": t(vis_c, 3),
            "vis_size": t(vis_s, 3),
            "vis_rot6d": t(vis_r, 6),
            "vis_box2d": t(vis_b, 4),
            "act_center": t(act_c, 3),
            "act_size": t(act_s, 3),
            "act_rot6d": t(act_r, 6),
            "catalog": torch.from_numpy(catalog),
            "assign": torch.tensor(assign, dtype=torch.long),
        }


def jenga_collate(batch: list[dict]) -> dict:
    """Stack image/depth/K; keep variable-length per-sample fields as lists."""
    keys_list = [
        "vis_center",
        "vis_size",
        "vis_rot6d",
        "vis_box2d",
        "act_center",
        "act_size",
        "act_rot6d",
        "catalog",
        "assign",
    ]
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "K": torch.stack([b["K"] for b in batch]),
    }
    for k in keys_list:
        out[k] = [b[k] for b in batch]
    return out
