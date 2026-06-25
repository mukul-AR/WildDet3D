"""Sim (Isaac/anyware-sim) dataset for the dense 9-DoF detector.

The synth sim format differs from the real AnywareScenes COCO format: each
camera's ``metadata.json`` carries ``intrinsics``, ``camera_extrinsic_4x4``
(camera pose in world), and a per-box dict with BOTH ``visible_`` and
``actual_`` ``extrinsic_4x4`` + ``geometry`` (and ``visible_fraction``). Boxes
are in world frame; camera-frame pose is ``inv(camera_extrinsic_4x4) @ pose``.

This yields the same per-sample dict the dense detector consumes, so the dense
trainer/loss are reused unchanged. ``target`` selects whether the GT geometry
is the ``visible`` (design-doc Stage-1) or ``actual`` (full) box.
"""

from __future__ import annotations

import glob
import itertools
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIGNS = np.array(list(itertools.product([-0.5, 0.5], repeat=3)), dtype=np.float32)


def _resize_pad(img: np.ndarray, size: int, nearest: bool) -> tuple:
    """Aspect-preserving resize to fit ``size`` then pad to ``size`` x ``size``."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    if r.ndim == 3:
        out = np.zeros((size, size, r.shape[2]), dtype=r.dtype)
        out[pad_y : pad_y + nh, pad_x : pad_x + nw] = r
    else:
        out = np.zeros((size, size), dtype=r.dtype)
        out[pad_y : pad_y + nh, pad_x : pad_x + nw] = r
    return out, scale, pad_x, pad_y


class SimDenseDataset(Dataset):
    """Per-camera sim RGBD + camera-frame 9-DoF boxes for the dense detector.

    Args:
        sim_root: directory containing ``synth_*/`` scene folders
            (e.g. ``.../anyware-sim/build/scenes/synth``).
        size: square SAM3 input size (1008).
        target: ``"actual"`` (full box) or ``"visible"`` (observed box).
        max_scenes: optional cap on scenes (0 = all).
        min_visible: drop boxes with ``visible_fraction`` below this.
    """

    def __init__(
        self,
        sim_root: str,
        size: int = 1008,
        target: str = "actual",
        max_scenes: int = 0,
        min_visible: float = 0.05,
    ) -> None:
        super().__init__()
        assert target in ("actual", "visible")
        self.size = size
        self.target = target
        self.min_visible = min_visible
        scene_dirs = sorted(glob.glob(os.path.join(sim_root, "synth_*")))
        if max_scenes > 0:
            scene_dirs = scene_dirs[:max_scenes]
        self.samples: list[str] = []
        for sd in scene_dirs:
            for cam in sorted(glob.glob(os.path.join(sd, "*_camera_*"))):
                if os.path.exists(os.path.join(cam, "rgb.png")) and os.path.exists(
                    os.path.join(cam, "metadata.json")
                ):
                    self.samples.append(cam)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        cam = self.samples[idx]
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
            [[fx * scale, 0, cx * scale + px], [0, fy * scale, cy * scale + py], [0, 0, 1]],
            dtype=np.float32,
        )

        centers, sizes, rot6d, box2d = [], [], [], []
        for b in meta["boxes"].values():
            if b.get("visible_fraction", 1.0) < self.min_visible:
                continue
            ext = np.array(b[f"{self.target}_extrinsic_4x4"], dtype=np.float64)
            geom = np.array(b[f"{self.target}_geometry"], dtype=np.float32)
            c_cam = r_wc @ ext[:3, 3] + t_wc
            if c_cam[2] <= 0.05:
                continue
            r_cam = (r_wc @ ext[:3, :3]).astype(np.float32)
            # project 8 corners (native intrinsics) -> 2D bbox, then resize/pad
            corners = (r_cam @ (_SIGNS * geom).T).T + c_cam.astype(np.float32)
            zc = np.clip(corners[:, 2], 1e-6, None)
            u = fx * corners[:, 0] / zc + cx
            v = fy * corners[:, 1] / zc + cy
            x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()

            centers.append(c_cam.astype(np.float32).tolist())
            sizes.append(geom.tolist())
            rot6d.append(r_cam[:2].reshape(6).tolist())
            box2d.append(
                [x1 * scale + px, y1 * scale + py, x2 * scale + px, y2 * scale + py]
            )

        n = len(centers)
        return {
            "image": img,
            "depth": depth_m,
            "K": torch.from_numpy(k_adj),
            "centers": torch.tensor(centers, dtype=torch.float32).reshape(n, 3),
            "sizes": torch.tensor(sizes, dtype=torch.float32).reshape(n, 3),
            "rot6d": torch.tensor(rot6d, dtype=torch.float32).reshape(n, 6),
            "box2d": torch.tensor(box2d, dtype=torch.float32).reshape(n, 4),
        }


def dense_collate(batch: list[dict]) -> dict:
    """Stack images/depth/K; keep variable-length GT as lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "K": torch.stack([b["K"] for b in batch]),
        "centers": [b["centers"] for b in batch],
        "sizes": [b["sizes"] for b in batch],
        "rot6d": [b["rot6d"] for b in batch],
        "box2d": [b["box2d"] for b in batch],
    }
