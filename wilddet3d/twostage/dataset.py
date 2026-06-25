"""Cached two-stage 9-DoF Anyware dataset.

Loads the crops + metadata produced by
``scripts/data_prep/anyware/build_two_stage_dataset.py`` and yields, per box:

    crop ``[4, S, S]``  (RGB/255 + depth_m / max_depth),
    ctx ``[10]``, anchor_center ``[3]``, T_base_cam ``[4, 4]``,
    walls ``[3, 5]``, skus ``[K, 4]``,
    GT visual OBB (camera frame) and GT actual OBB (base_link frame),
    confidence, has_sku, presence target.

Both the Stage-1 (visual, camera frame) and Stage-2 (actual, base frame)
targets are derived from the same GT OBB; Stage 2 only becomes non-trivial
because its *input* is the (corrupted/occluded) Stage-1 output.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from wilddet3d.twostage.rotation_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


class TwoStageAnywareDataset(Dataset):
    """Cached per-box dataset for two-stage 9-DoF training.

    Args:
        cache_dir: directory holding ``rgb.npy``, ``depth.npy``, ``meta.npz``.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.rgb = np.load(os.path.join(cache_dir, "rgb.npy"), mmap_mode="r")
        self.depth = np.load(
            os.path.join(cache_dir, "depth.npy"), mmap_mode="r"
        )
        meta = np.load(os.path.join(cache_dir, "meta.npz"))
        self.ctx = meta["ctx"].astype(np.float32)
        self.anchor = meta["anchor_center"].astype(np.float32)
        self.tbc = meta["T_base_cam"].astype(np.float32)
        self.walls = meta["walls"].astype(np.float32)
        self.skus = meta["skus"].astype(np.float32)
        self.gt_center_cam = meta["gt_center_cam"].astype(np.float32)
        self.gt_size = meta["gt_size"].astype(np.float32)
        self.gt_rot6d_cam = meta["gt_rot6d_cam"].astype(np.float32)
        self.confidence = meta["confidence"].astype(np.float32)
        self.has_sku = meta["has_sku"].astype(np.float32)
        self.max_depth = float(meta["max_depth"])
        self.crop_size = int(meta["crop_size"])

        self._precompute_base_gt()

    def _precompute_base_gt(self) -> None:
        """Derive the base_link-frame GT OBB from the camera-frame GT once."""
        tbc = torch.from_numpy(self.tbc)
        r_bc = tbc[:, :3, :3]
        t_bc = tbc[:, :3, 3]
        center_cam = torch.from_numpy(self.gt_center_cam)
        rot6d_cam = torch.from_numpy(self.gt_rot6d_cam)
        r_cam = rotation_6d_to_matrix(rot6d_cam)
        center_base = (
            torch.bmm(r_bc, center_cam.unsqueeze(-1)).squeeze(-1) + t_bc
        )
        r_base = torch.bmm(r_bc, r_cam)
        self.gt_center_base = center_base.numpy().astype(np.float32)
        self.gt_rot6d_base = (
            matrix_to_rotation_6d(r_base).numpy().astype(np.float32)
        )

    def __len__(self) -> int:
        return self.rgb.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb = np.asarray(self.rgb[idx], dtype=np.float32) / 255.0  # [3,S,S]
        depth = (
            np.asarray(self.depth[idx], dtype=np.float32) / 1000.0
        )  # mm -> m, [1,S,S]
        depth = np.clip(depth / self.max_depth, 0.0, 1.0)
        crop = np.concatenate([rgb, depth], axis=0)  # [4,S,S]

        return {
            "crop": torch.from_numpy(crop),
            "ctx": torch.from_numpy(self.ctx[idx]),
            "anchor_center": torch.from_numpy(self.anchor[idx]),
            "T_base_cam": torch.from_numpy(self.tbc[idx]),
            "walls": torch.from_numpy(self.walls[idx]),
            "skus": torch.from_numpy(self.skus[idx]),
            "gt_cam_center": torch.from_numpy(self.gt_center_cam[idx]),
            "gt_size": torch.from_numpy(self.gt_size[idx]),
            "gt_cam_rot6d": torch.from_numpy(self.gt_rot6d_cam[idx]),
            "gt_base_center": torch.from_numpy(self.gt_center_base[idx]),
            "gt_base_rot6d": torch.from_numpy(self.gt_rot6d_base[idx]),
            "confidence": torch.tensor(self.confidence[idx]),
            "has_sku": torch.tensor(self.has_sku[idx]),
            "presence": torch.tensor(1.0),
        }
