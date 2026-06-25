"""Dense detector dataset: full-image RGBD + intrinsics + GT 9-DoF boxes.

Reads the Omni3D-format AnywareScenes COCO JSON (per-box camera-frame GT) and
yields full images resized + padded to the SAM3 input size (1008, RoPE-locked),
depth in metres, the resize/pad-adjusted intrinsics, and per-image GT boxes
(camera-frame center, box-axis size [gx,gy,gz], 6D rotation, projected 2D box).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


class DenseAnywareDataset(Dataset):
    """Full-image RGBD dataset for the prompt-free dense detector.

    Args:
        data_root: AnywareScenes root (with ``annotations/`` + image tree).
        split: ``train`` or ``val``.
        size: square SAM3 input size (must be 1008 for the RoPE backbone).
        max_scenes: optional cap on scenes (0 = all).
    """

    def __init__(
        self,
        data_root: str = "data/anyware_scenes",
        split: str = "train",
        size: int = 1008,
        max_scenes: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.size = size
        with open(
            os.path.join(data_root, "annotations", f"AnywareScenes_{split}.json")
        ) as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)

        keep = None
        if max_scenes > 0:
            keep = {s["scene_id"] for s in coco["scenes"][:max_scenes]}
        # keep only images that exist on disk and have annotations
        self.ids = []
        for iid, im in self.images.items():
            if keep is not None and im.get("scene_id") not in keep:
                continue
            if not anns.get(iid):
                continue
            p = os.path.join(data_root, im["file_path"])
            if os.path.exists(p) or os.path.exists(im["file_path"]):
                self.ids.append(iid)
        self.ids.sort()
        self.anns = anns

    def __len__(self) -> int:
        return len(self.ids)

    def _img_path(self, im: dict) -> str:
        p = os.path.join(self.data_root, im["file_path"])
        return p if os.path.exists(p) else im["file_path"]

    def __getitem__(self, idx: int) -> dict:
        iid = self.ids[idx]
        im = self.images[iid]
        path = self._img_path(im)
        rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(path.replace("image.jpg", "depth.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)

        rgb_p, scale, px, py = _resize_pad(rgb, self.size, nearest=False)
        depth_p, _, _, _ = _resize_pad(
            depth.astype(np.uint16), self.size, nearest=True
        )

        img = rgb_p.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        depth_m = torch.from_numpy(
            (depth_p.astype(np.float32) / 1000.0)[None]
        )

        k = np.array(im["K"], dtype=np.float32)
        k_adj = k.copy()
        k_adj[0, 0] *= scale
        k_adj[1, 1] *= scale
        k_adj[0, 2] = k[0, 2] * scale + px
        k_adj[1, 2] = k[1, 2] * scale + py

        centers, sizes, rot6d, box2d = [], [], [], []
        for a in self.anns[iid]:
            if not a.get("valid3D", True) or a.get("behind_camera", False):
                continue
            c = np.array(a["center_cam"], dtype=np.float32)
            if c[2] <= 0.05:
                continue
            dims = a["dimensions"]  # [W, H, L]
            sizes.append([dims[2], dims[1], dims[0]])  # [gx, gy, gz]
            centers.append(c.tolist())
            r = np.array(a["R_cam"], dtype=np.float32)
            rot6d.append(r[:2].reshape(6).tolist())
            b = a.get("bbox2D_proj") or a.get("bbox2D_trunc") or [0, 0, 1, 1]
            box2d.append(
                [b[0] * scale + px, b[1] * scale + py, b[2] * scale + px, b[3] * scale + py]
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
