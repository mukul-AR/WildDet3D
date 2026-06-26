# JENGA Two-Stage Dimension-Conditioned Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stage-2 transformer that, given Stage-1 *visible* box detections plus a per-scene set of candidate `(L,B,H)` dimension triples, infers the *actual* full boxes by hard-selecting one candidate dimension per box and completing its pose.

**Architecture:** Frozen SAM3 + LingBot-depth + EarlyDepthFusion → fused FPN map. Stage 1 = existing dense CenterNet head (`target=visible`). Stage 2 = transformer decoder over per-box queries (self-attention) + candidate-dim tokens (cross-attention); per query it emits an assignment over the dim tokens (size = selected dim) + an actual center residual + 6D rotation. Trained jointly, encoders frozen, Stage 2 teacher-forced on GT visible boxes.

**Tech Stack:** PyTorch 2.8 (cu128), the existing `wilddet3d/dense/*` modules, PyYAML for the scene catalog. Tests via `pytest`.

## Global Constraints

- Input is locked to **1008²** (SAM3 RoPE). `--size 1008`.
- Encoders (SAM3 backbone + LingBot depth) are **always frozen, run under `no_grad`**. Only Stage-1 head + fusion + Stage-2 decoder train.
- Rotation rep is **6D continuous** (first two rows of R); rotation losses are **symmetry-aware over the cuboid D2 group** (`wilddet3d/dense/rotation_utils.py`). Do not add new rotation conventions.
- Candidate dims and assignment use **ascending-sorted** `(L,B,H)`; the actual box's local axes are **canonicalized to ascending extent** so size is sorted and rotation carries orientation.
- Stage-2 decoder defaults: **`d_model=512, layers=12, heads=8`**, all CLI knobs.
- No explicit packing/collision loss (learned-implicit).
- New code lives under `wilddet3d/dense/`; tests under `tests/dense/`. Follow existing file style (module docstring, `from __future__ import annotations`, typed signatures).

---

### Task 1: Canonicalization + catalog helpers (pure functions)

**Files:**
- Create: `wilddet3d/dense/jenga_utils.py`
- Test: `tests/dense/test_jenga_utils.py`

**Interfaces:**
- Produces:
  - `canonicalize_obb(size_xyz: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — returns `(size_sorted[3] ascending, R_canon[3,3] proper rotation)`; reorders R's columns by `argsort(size)` and negates the last column if `det < 0`.
  - `parse_catalog(scene_json_path: str) -> np.ndarray` — `[K,3]` distinct ascending-sorted catalog dims from `skus_yaml_string`.
  - `assign_index(size_sorted: np.ndarray, catalog: np.ndarray) -> int` — index of nearest (min-L1) catalog row.

- [ ] **Step 1: Write failing tests**

```python
# tests/dense/test_jenga_utils.py
import numpy as np
from wilddet3d.dense.jenga_utils import canonicalize_obb, assign_index

def test_canonicalize_sorts_size_and_keeps_proper_rotation():
    R = np.eye(3)
    size = np.array([0.49, 0.34, 0.34])  # unsorted
    s2, R2 = canonicalize_obb(size, R)
    assert np.allclose(s2, [0.34, 0.34, 0.49])           # ascending
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-5) # still a rotation
    # column that had the largest extent (x) must now be last
    assert np.allclose(R2[:, 2], R[:, 0])

def test_canonicalize_fixes_handedness_on_odd_permutation():
    R = np.eye(3)
    size = np.array([0.3, 0.5, 0.4])     # argsort -> [0,2,1] is odd
    _, R2 = canonicalize_obb(size, R)
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-5)

def test_assign_index_picks_nearest_catalog_row():
    cat = np.array([[0.34, 0.34, 0.49], [0.30, 0.40, 0.40]])
    assert assign_index(np.array([0.34, 0.34, 0.49]), cat) == 0
    assert assign_index(np.array([0.31, 0.39, 0.41]), cat) == 1
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_jenga_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: wilddet3d.dense.jenga_utils`.

- [ ] **Step 3: Implement `jenga_utils.py`**

```python
"""Canonicalization + scene-catalog helpers for the JENGA Stage-2 head.

Actual boxes are canonicalized so their local axes are ordered by ascending
extent: size becomes an ascending-sorted triple (matching the sorted catalog
dims used for assignment) and the rotation carries the orientation. Column
reordering can flip handedness; we restore a proper rotation by negating the
last axis (a cuboid D2 symmetry, already modded out by the rotation loss).
"""
from __future__ import annotations

import json

import numpy as np
import yaml


def canonicalize_obb(size_xyz: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder box axes to ascending extent; return (sorted size, proper R)."""
    perm = np.argsort(size_xyz)
    size_sorted = size_xyz[perm].astype(np.float32)
    R_canon = R[:, perm].astype(np.float32)
    if np.linalg.det(R_canon) < 0:
        R_canon[:, -1] *= -1.0
    return size_sorted, R_canon


def parse_catalog(scene_json_path: str) -> np.ndarray:
    """Distinct ascending-sorted catalog dims [K,3] from skus_yaml_string."""
    scene = json.load(open(scene_json_path))
    skus = yaml.safe_load(scene["skus_yaml_string"])["skus"]
    dims = {tuple(sorted(round(float(x), 4) for x in s["geometry"])) for s in skus}
    return np.array(sorted(dims), dtype=np.float32).reshape(-1, 3)


def assign_index(size_sorted: np.ndarray, catalog: np.ndarray) -> int:
    """Index of the nearest (min-L1) catalog row to an ascending-sorted size."""
    return int(np.abs(catalog - size_sorted[None]).sum(axis=1).argmin())
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_jenga_utils.py -v`
Expected: PASS (3 passed). If `pytest` missing: `.venv/bin/python -m pip install pytest` (or `uv pip install --python .venv/bin/python pytest`).

- [ ] **Step 5: Commit**

```bash
git add wilddet3d/dense/jenga_utils.py tests/dense/test_jenga_utils.py
git commit -m "feat(jenga): OBB axis-canonicalization + scene-catalog helpers"
```

---

### Task 2: Dataset — emit visible + actual + catalog + assignment, with train/val split

**Files:**
- Create: `wilddet3d/dense/sim_jenga_dataset.py`
- Test: `tests/dense/test_sim_jenga_dataset.py`

**Interfaces:**
- Consumes: `canonicalize_obb`, `parse_catalog`, `assign_index` (Task 1); `_resize_pad`, `_IMAGENET_MEAN/STD`, `_SIGNS` (import from `wilddet3d.dense.sim_dataset`).
- Produces:
  - `SimJengaDataset(sim_root, size=1008, max_scenes=0, min_visible=0.05, split="train", val_frac=0.1)`; `__getitem__` returns a dict with keys: `image[3,H,W]`, `depth[1,H,W]`, `K[3,3]`, and per-box lists as tensors — `vis_center[N,3]`, `vis_size[N,3]`, `vis_rot6d[N,6]`, `vis_box2d[N,4]`, `act_center[N,3]`, `act_size[N,3]`(sorted), `act_rot6d[N,6]`(canonical), `catalog[K,3]`, `assign[N]`(long).
  - `jenga_collate(batch) -> dict` — stacks image/depth/K; keeps per-sample variable-length fields as Python lists (including `catalog`, which varies in K per scene).

- [ ] **Step 1: Write failing test** (uses real local sim data)

```python
# tests/dense/test_sim_jenga_dataset.py
import os
import pytest
import torch
from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, jenga_collate

ROOT = "/storage/3dl_sim_data/20260625_2skuwallremoval/anyware-sim/build/scenes/synth"

@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_sample_has_visible_actual_catalog_and_assignment():
    ds = SimJengaDataset(ROOT, size=1008, max_scenes=4, split="train", val_frac=0.0)
    s = ds[0]
    n = s["vis_center"].shape[0]
    assert s["act_center"].shape[0] == n and s["assign"].shape[0] == n
    assert s["act_size"].shape == (n, 3) and s["act_rot6d"].shape == (n, 6)
    # actual size is ascending-sorted
    assert torch.all(s["act_size"][:, 0] <= s["act_size"][:, 1] + 1e-4)
    assert torch.all(s["act_size"][:, 1] <= s["act_size"][:, 2] + 1e-4)
    # assignment indexes into the catalog
    K = s["catalog"].shape[0]
    assert K >= 1 and int(s["assign"].max()) < K

@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_train_val_split_is_disjoint():
    tr = SimJengaDataset(ROOT, max_scenes=20, split="train", val_frac=0.2)
    va = SimJengaDataset(ROOT, max_scenes=20, split="val", val_frac=0.2)
    assert set(tr.samples).isdisjoint(set(va.samples))
    assert len(va) > 0 and len(tr) > 0

@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_collate_keeps_catalog_per_sample():
    ds = SimJengaDataset(ROOT, max_scenes=4, split="train", val_frac=0.0)
    b = jenga_collate([ds[0], ds[1]])
    assert b["image"].shape[0] == 2
    assert isinstance(b["catalog"], list) and len(b["catalog"]) == 2
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_sim_jenga_dataset.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `sim_jenga_dataset.py`**

```python
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

from wilddet3d.dense.jenga_utils import assign_index, canonicalize_obb, parse_catalog
from wilddet3d.dense.sim_dataset import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _SIGNS,
    _resize_pad,
)


def _in_split(scene_dir: str, split: str, val_frac: float) -> bool:
    if val_frac <= 0.0:
        return split == "train"
    h = int(hashlib.md5(os.path.basename(scene_dir).encode()).hexdigest(), 16)
    is_val = (h % 1000) < int(val_frac * 1000)
    return is_val if split == "val" else not is_val


class SimJengaDataset(Dataset):
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
            cv2.imread(os.path.join(cam, "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
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
        catalog = parse_catalog(scene_json)

        def cam_obb(ext, geom):
            c = r_wc @ ext[:3, 3] + t_wc
            R = r_wc @ ext[:3, :3]
            return c.astype(np.float32), R.astype(np.float32), geom.astype(np.float32)

        vis_c, vis_s, vis_r, vis_b = [], [], [], []
        act_c, act_s, act_r, assign = [], [], [], []
        for b in meta["boxes"].values():
            if b.get("visible_fraction", 1.0) < self.min_visible:
                continue
            vc, vR, vg = cam_obb(
                np.array(b["visible_extrinsic_4x4"], dtype=np.float64),
                np.array(b["visible_geometry"], dtype=np.float32),
            )
            if vc[2] <= 0.05:
                continue
            ac, aR, ag = cam_obb(
                np.array(b["actual_extrinsic_4x4"], dtype=np.float64),
                np.array(b["actual_geometry"], dtype=np.float32),
            )
            a_size_sorted, a_R_canon = canonicalize_obb(ag, aR)

            # visible 2D box (native intrinsics) -> resized/padded px
            corners = (vR @ (_SIGNS * vg).T).T + vc
            zc = np.clip(corners[:, 2], 1e-6, None)
            u = fx * corners[:, 0] / zc + cx
            v = fy * corners[:, 1] / zc + cy
            x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()

            vis_c.append(vc.tolist())
            vis_s.append(vg.tolist())
            vis_r.append(vR[:2].reshape(6).tolist())
            vis_b.append([x1 * scale + px, y1 * scale + py, x2 * scale + px, y2 * scale + py])
            act_c.append(ac.tolist())
            act_s.append(a_size_sorted.tolist())
            act_r.append(a_R_canon[:2].reshape(6).tolist())
            assign.append(assign_index(a_size_sorted, catalog))

        n = len(vis_c)
        t = lambda x, d: torch.tensor(x, dtype=torch.float32).reshape(n, d)
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
    keys_list = [
        "vis_center", "vis_size", "vis_rot6d", "vis_box2d",
        "act_center", "act_size", "act_rot6d", "catalog", "assign",
    ]
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "K": torch.stack([b["K"] for b in batch]),
    }
    for k in keys_list:
        out[k] = [b[k] for b in batch]
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_sim_jenga_dataset.py -v`
Expected: PASS (3 passed), or SKIPPED if `/storage/3dl_sim_data/...` is absent — in that case run on the box where data exists before relying on it.

- [ ] **Step 5: Commit**

```bash
git add wilddet3d/dense/sim_jenga_dataset.py tests/dense/test_sim_jenga_dataset.py
git commit -m "feat(jenga): sim dataset with visible+actual+catalog+assignment and val split"
```

---

### Task 3: Stage-2 transformer module

**Files:**
- Create: `wilddet3d/dense/stage2.py`
- Test: `tests/dense/test_stage2.py`

**Interfaces:**
- Consumes: `rotation_6d_to_matrix` (rotation_utils), fused feat map from the model (Task 5).
- Produces:
  - `JengaStage2(in_ch=256, d_model=512, layers=12, heads=8)` with
    `forward(feat[B,C,Hf,Wf], queries_uv[list of [Qi,2] feat-grid coords], vis_obb[list of [Qi,10]], catalog[list of [Ki,3]]) -> dict` returning padded `assign_logits[B,Qmax,Kmax]`, `center_delta[B,Qmax,3]`, `rot6d[B,Qmax,6]`, `q_mask[B,Qmax]`(bool), `k_mask[B,Qmax? no -> B,Kmax]`(bool).
  - `vis_obb` row layout: `[cx,cy,cz, log_z, w,h,l, r0,r1,...]` is overkill — use `[cx,cy,cz, w,h,l, r0..r5]` = 12? Keep simple: **`vis_obb` row = concat(vis_center[3], vis_size[3], vis_rot6d[6]) = 9+? = 12`**. Document as 12-dim.

- [ ] **Step 1: Write failing test** (synthetic tensors, CPU)

```python
# tests/dense/test_stage2.py
import torch
from wilddet3d.dense.stage2 import JengaStage2

def test_forward_shapes_and_masking_variable_q_and_k():
    m = JengaStage2(in_ch=16, d_model=32, layers=2, heads=4).eval()
    B, C, Hf, Wf = 2, 16, 12, 12
    feat = torch.randn(B, C, Hf, Wf)
    queries_uv = [torch.rand(3, 2) * (Wf - 1), torch.rand(1, 2) * (Wf - 1)]
    vis_obb = [torch.randn(3, 12), torch.randn(1, 12)]
    catalog = [torch.rand(2, 3), torch.rand(4, 3)]
    out = m(feat, queries_uv, vis_obb, catalog)
    assert out["assign_logits"].shape == (2, 3, 4)  # Qmax=3, Kmax=4
    assert out["center_delta"].shape == (2, 3, 3)
    assert out["rot6d"].shape == (2, 3, 6)
    assert out["q_mask"].sum().item() == 4         # 3 + 1 real queries
    # padded dim-token slots must be masked out (logit -> -inf) for sample 0
    assert torch.isinf(out["assign_logits"][0, 0, 2:]).all()

def test_backward_runs():
    m = JengaStage2(in_ch=16, d_model=32, layers=2, heads=4)
    feat = torch.randn(1, 16, 12, 12, requires_grad=True)
    out = m(feat, [torch.rand(2, 2) * 11], [torch.randn(2, 12)], [torch.rand(3, 3)])
    out["center_delta"].sum().backward()
    assert feat.grad is not None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_stage2.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `stage2.py`**

```python
"""JENGA Stage 2: dimension-conditioned amodal box completion.

Per visible-box query: self-attention among queries (mutual "jenga" layout
consistency) + cross-attention over the scene's candidate-dimension tokens
(allowed sizes). Outputs a hard-selectable assignment over the dim tokens
(size = selected dim) plus an actual-center residual and a 6D rotation. Handles
variable per-scene query- and dim-counts via padding + attention masks.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pad_stack(tensors: list[Tensor], dim0_max: int, feat_dim: int) -> tuple[Tensor, Tensor]:
    """Pad a list of [Ni, F] to [B, dim0_max, F]; return (padded, mask[B,dim0_max])."""
    b = len(tensors)
    out = tensors[0].new_zeros(b, dim0_max, feat_dim)
    mask = torch.zeros(b, dim0_max, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n:
            out[i, :n] = t
            mask[i, :n] = True
    return out, mask


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, q: Tensor, kv: Tensor, q_pad: Tensor, kv_pad: Tensor) -> Tensor:
        x = self.n1(q)
        x = q + self.self_attn(x, x, x, key_padding_mask=~q_pad, need_weights=False)[0]
        y = self.n2(x)
        x = x + self.cross_attn(y, kv, kv, key_padding_mask=~kv_pad, need_weights=False)[0]
        return x + self.ffn(self.n3(x))


class JengaStage2(nn.Module):
    def __init__(self, in_ch: int = 256, d_model: int = 512, layers: int = 12, heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(in_ch, d_model)
        self.obb_embed = nn.Sequential(
            nn.Linear(12, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.dim_embed = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.layers = nn.ModuleList([_DecoderLayer(d_model, heads) for _ in range(layers)])
        self.q_to_assign = nn.Linear(d_model, d_model)
        self.k_to_assign = nn.Linear(d_model, d_model)
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 9)  # 3 center delta + 6 rot
        )

    @staticmethod
    def _sample_feat(feat: Tensor, uv: Tensor) -> Tensor:
        """Bilinear-sample feat [C,Hf,Wf] at uv [Q,2] (feat-grid coords) -> [Q,C]."""
        _, hf, wf = feat.shape
        gx = (uv[:, 0] / max(wf - 1, 1)) * 2 - 1
        gy = (uv[:, 1] / max(hf - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        s = F.grid_sample(feat[None], grid, align_corners=True)  # [1,C,1,Q]
        return s[0, :, 0].transpose(0, 1)  # [Q,C]

    def forward(
        self,
        feat: Tensor,
        queries_uv: list[Tensor],
        vis_obb: list[Tensor],
        catalog: list[Tensor],
    ) -> dict[str, Tensor]:
        b = feat.shape[0]
        qmax = max((q.shape[0] for q in queries_uv), default=0)
        kmax = max((c.shape[0] for c in catalog), default=0)
        qmax, kmax = max(qmax, 1), max(kmax, 1)

        q_feats = []
        for i in range(b):
            qf = self._sample_feat(feat[i], queries_uv[i]) if queries_uv[i].shape[0] else feat.new_zeros(0, feat.shape[1])
            q_feats.append(self.feat_proj(qf) + self.obb_embed(vis_obb[i]))
        q, q_mask = _pad_stack(q_feats, qmax, self.d_model)
        kv, k_mask = _pad_stack([self.dim_embed(c) for c in catalog], kmax, self.d_model)

        for layer in self.layers:
            q = layer(q, kv, q_mask, k_mask)

        qa = self.q_to_assign(q)               # [B,Qmax,d]
        ka = self.k_to_assign(kv)              # [B,Kmax,d]
        logits = torch.einsum("bqd,bkd->bqk", qa, ka) / (self.d_model ** 0.5)
        logits = logits.masked_fill(~k_mask[:, None, :], float("-inf"))
        pose = self.pose_head(q)
        return {
            "assign_logits": logits,
            "center_delta": pose[..., :3],
            "rot6d": pose[..., 3:],
            "q_mask": q_mask,
            "k_mask": k_mask,
        }
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_stage2.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add wilddet3d/dense/stage2.py tests/dense/test_stage2.py
git commit -m "feat(jenga): Stage-2 dim-conditioned transformer decoder"
```

---

### Task 4: Stage-2 loss

**Files:**
- Modify: `wilddet3d/dense/loss.py` (append a class; do not change `DenseDet3DLoss`)
- Test: `tests/dense/test_stage2_loss.py`

**Interfaces:**
- Consumes: `JengaStage2` outputs (Task 3); `symmetry_chordal_loss`, `symmetry_min_geodesic`, `rad2deg` (rotation_utils); `act_center`, `act_rot6d`, `assign`, `catalog`, `vis_center` (dataset).
- Produces:
  - `JengaStage2Loss(w_assign=1.0, w_center=1.0, w_rot=1.0)` with
    `forward(out, batch) -> dict` keys `assign`, `center`, `rot`, `total`, `rot_deg`, `assign_acc`, `num_q`. `out["center_delta"]` is added to per-query GT `vis_center` to predict `act_center`.

- [ ] **Step 1: Write failing test**

```python
# tests/dense/test_stage2_loss.py
import torch
from wilddet3d.dense.loss import JengaStage2Loss

def _fake(B=2, Q=3, K=4):
    out = {
        "assign_logits": torch.randn(B, Q, K, requires_grad=True),
        "center_delta": torch.randn(B, Q, 3, requires_grad=True),
        "rot6d": torch.randn(B, Q, 6, requires_grad=True),
        "q_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "k_mask": torch.ones(B, K, dtype=torch.bool),
    }
    batch = {
        "vis_center": [torch.zeros(3, 3), torch.zeros(2, 3)],
        "act_center": [torch.randn(3, 3), torch.randn(2, 3)],
        "act_rot6d": [torch.tensor([[1.,0,0,0,1,0]]*3), torch.tensor([[1.,0,0,0,1,0]]*2)],
        "assign": [torch.tensor([0, 1, 2]), torch.tensor([3, 0])],
    }
    return out, batch

def test_loss_is_finite_and_differentiable():
    loss = JengaStage2Loss()
    out, batch = _fake()
    d = loss(out, batch)
    for k in ("assign", "center", "rot", "total"):
        assert torch.isfinite(d[k]).all()
    d["total"].backward()
    assert out["assign_logits"].grad is not None
    assert 0.0 <= d["assign_acc"].item() <= 1.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_stage2_loss.py -v`
Expected: FAIL — `JengaStage2Loss` undefined.

- [ ] **Step 3: Append `JengaStage2Loss` to `loss.py`**

```python
class JengaStage2Loss(nn.Module):
    """Assignment CE + actual-center L1 + symmetry-aware rotation, masked per query."""

    def __init__(self, w_assign: float = 1.0, w_center: float = 1.0, w_rot: float = 1.0) -> None:
        super().__init__()
        self.w_assign, self.w_center, self.w_rot = w_assign, w_center, w_rot

    def forward(self, out: dict, batch: dict) -> dict:
        device = out["assign_logits"].device
        logits, dc, rot6 = out["assign_logits"], out["center_delta"], out["rot6d"]
        b = logits.shape[0]
        pa, pc, pr, ta, tc, tr = [], [], [], [], [], []
        for i in range(b):
            n = int(out["q_mask"][i].sum())
            if n == 0:
                continue
            pa.append(logits[i, :n])                                   # [n,K]
            pc.append(dc[i, :n] + batch["vis_center"][i])              # predicted actual center
            pr.append(rot6[i, :n])
            ta.append(batch["assign"][i])
            tc.append(batch["act_center"][i])
            tr.append(batch["act_rot6d"][i])
        if not pa:
            z = torch.zeros((), device=device)
            return {"assign": z, "center": z, "rot": z, "total": z,
                    "rot_deg": z, "assign_acc": z, "num_q": torch.tensor(0.0, device=device)}
        pa_c = torch.cat(pa); ta_c = torch.cat(ta).to(device)
        pc_c = torch.cat(pc); tc_c = torch.cat(tc).to(device)
        pr_c = torch.cat(pr); tr_c = torch.cat(tr).to(device)
        loss_assign = F.cross_entropy(pa_c, ta_c)
        loss_center = (pc_c - tc_c).abs().mean()
        loss_rot = symmetry_chordal_loss(pr_c, tr_c).mean()
        total = self.w_assign * loss_assign + self.w_center * loss_center + self.w_rot * loss_rot
        with torch.no_grad():
            acc = (pa_c.argmax(-1) == ta_c).float().mean()
            rot_deg = rad2deg(symmetry_min_geodesic(pr_c, tr_c)).mean()
        return {"assign": loss_assign, "center": loss_center, "rot": loss_rot,
                "total": total, "rot_deg": rot_deg, "assign_acc": acc,
                "num_q": torch.tensor(float(ta_c.numel()), device=device)}
```

Add to the imports at the top of `loss.py`: `from torch.nn import functional as F`.

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_stage2_loss.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add wilddet3d/dense/loss.py tests/dense/test_stage2_loss.py
git commit -m "feat(jenga): Stage-2 loss (assignment CE + actual center + rotation)"
```

---

### Task 5: Model wiring — expose fused feat + run Stage 2 (teacher-forced)

**Files:**
- Modify: `wilddet3d/dense/model.py`
- Test: `tests/dense/test_model_feat.py`

**Interfaces:**
- Consumes: existing `DenseDet3D` (frozen encoders + Stage-1 head); `JengaStage2` (Task 3).
- Produces:
  - `DenseDet3D.forward(images, depth, k, return_feat=False)` — when `return_feat`, the returned dict also includes `feat` (the fused FPN map `[B,256,Hf,Wf]`) and `stride` (`size/Hf`).
  - The fused-feature computation is unchanged; we only stop discarding `feat`.

- [ ] **Step 1: Write failing test** (CPU, monkeypatched encoders — no checkpoint)

```python
# tests/dense/test_model_feat.py
import torch
from wilddet3d.dense.head import DenseConvHead
from wilddet3d.dense.model import DenseDet3D

def test_forward_can_return_fused_feat(monkeypatch):
    # Build a DenseDet3D with stub encoders to test the return_feat plumbing only.
    m = DenseDet3D.__new__(DenseDet3D)
    torch.nn.Module.__init__(m)
    m.fpn_level = 0
    m.head = DenseConvHead(in_ch=8)
    def fake_fused(images, depth, k):
        return [torch.randn(images.shape[0], 8, 9, 9)]
    monkeypatch.setattr(m, "_fused_feats", fake_fused, raising=False)
    out = m.forward(torch.randn(1, 3, 36, 36), torch.randn(1, 1, 36, 36),
                    torch.eye(3)[None], return_feat=True)
    assert out["feat"].shape == (1, 8, 9, 9)
    assert "heatmap" in out and "reg" in out
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_model_feat.py -v`
Expected: FAIL — `forward()` has no `return_feat`; no `_fused_feats`.

- [ ] **Step 3: Refactor `model.py` forward**

Split the fused-feature computation into a helper and add `return_feat`:

```python
    def _fused_feats(self, images: Tensor, depth: Tensor, k: Tensor) -> list[Tensor]:
        _, _, h, w = images.shape
        with torch.no_grad():
            backbone_out = self.backbone.forward_image(self._to_sam3(images))
            geom = self.geometry_backend(
                images=images, depth_feats=None, intrinsics=k, image_hw=(h, w),
                depth_gt=depth, depth_mask=None, padding=None,
            )
        depth_latents = geom["depth_latents"]
        depth_latents_hw = geom.get("aux", {}).get("depth_latents_hw")
        backbone_fpn = backbone_out["backbone_fpn"]
        if not isinstance(backbone_fpn, list):
            backbone_fpn = [backbone_fpn]
        fused = self.early_depth_fusion(
            visual_feats=backbone_fpn, depth_latents=depth_latents,
            depth_latents_hw=depth_latents_hw,
        )
        if not isinstance(fused, (list, tuple)):
            fused = [fused]
        return list(fused)

    def forward(self, images: Tensor, depth: Tensor, k: Tensor, return_feat: bool = False) -> dict[str, Tensor]:
        feat = self._fused_feats(images, depth, k)[self.fpn_level]
        out = self.head(feat)
        if return_feat:
            out["feat"] = feat
            out["stride"] = images.shape[-1] / feat.shape[-2]
        return out
```

(Delete the old inline body of `forward`.)

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_model_feat.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add wilddet3d/dense/model.py tests/dense/test_model_feat.py
git commit -m "refactor(jenga): expose fused FPN feat from DenseDet3D.forward"
```

---

### Task 6: Train script — combined Stage-1 + teacher-forced Stage-2, val loop, CLI knobs

**Files:**
- Create: `scripts/train_jenga.py` (start from a copy of `scripts/train_dense_9dof.py`)

**Interfaces:**
- Consumes: `SimJengaDataset`, `jenga_collate` (Task 2); `DenseDet3D` w/ `return_feat` (Task 5); `JengaStage2` (Task 3); `DenseDet3DLoss` + `JengaStage2Loss` (Task 4); `build_dense_targets` (existing — fed the **visible** boxes for Stage 1).
- Produces: a runnable trainer. New CLI flags beyond `train_dense_9dof.py`: `--d-model 512 --layers 12 --heads 8 --val-frac 0.1 --w-assign 1.0 --w-center 1.0 --w-rot2 1.0`. Stage-1 targets use `batch["vis_*"]`. Stage-2 queries are the **GT visible** centers projected to the feat grid (teacher forcing).

Key implementation notes (fold into the copied script):
- Build `train`/`val` datasets with `split=` and `--val-frac`.
- `model = DenseDet3D.from_wilddet3d(...)`; `stage2 = JengaStage2(in_ch=256, d_model=..., layers=..., heads=...).to(device)`.
- Optimizer over `model` trainables **+** `stage2.parameters()`.
- Per batch:
  - `pred = model(image, depth, K, return_feat=True)`; `feat, stride = pred["feat"], pred["stride"]`.
  - Stage-1: `tgt = build_dense_targets(batch["vis_center"], batch["vis_size"], batch["vis_rot6d"], batch["vis_box2d"], K, (Hf,Wf), stride, device)`; `l1 = DenseDet3DLoss()(pred, tgt)`.
  - Build Stage-2 queries from GT visible centers: project each `vis_center[i]` to pixels with `K[i]` (`u=fx*x/z+cx0`, `v=fy*y/z+cy0`), divide by `stride` → `queries_uv[i] [Qi,2]`; `vis_obb[i] = cat([vis_center, vis_size, vis_rot6d], -1) [Qi,12]`.
  - `out = stage2(feat, queries_uv, vis_obb, batch["catalog"])`; `l2 = JengaStage2Loss(w_assign,w_center,w_rot2)(out, batch)`.
  - `total = l1["total"] + l2["total"]`; backward over both.
- W&B: log `train/stage1_*` (reuse existing keys) and `train/assign`, `train/center`, `train/rot2`, `train/assign_acc`.
- After each epoch, run a **val loop** (no grad): average `assign_acc`, Stage-2 `rot_deg`, center L1; log `val/assign_acc`, `val/rot_deg`, `val/center`.
- Save `{"model":..., "stage2": stage2.state_dict(), "epoch":..., "args":...}` to `args.out/jenga_last.pt`.

- [ ] **Step 1: Copy + edit**

```bash
cp scripts/train_dense_9dof.py scripts/train_jenga.py
```
Then edit per the notes above (imports, datasets, stage2 build, combined loss, query projection, val loop, save keys, new args).

- [ ] **Step 2: Smoke test — 1 epoch, tiny subset, no W&B, CPU-or-CUDA**

Run (local box with sim data + checkpoint present):
```bash
PYTHONPATH=. .venv/bin/python scripts/train_jenga.py \
  --sim-root /storage/3dl_sim_data/20260625_2skuwallremoval/anyware-sim/build/scenes/synth \
  --max-scenes 6 --epochs 1 --batch-size 2 --val-frac 0.34 \
  --d-model 128 --layers 2 --heads 4 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga-smoke --no-wandb
```
Expected: prints `sim train/val samples`, trainable param count (Stage-1 + Stage-2), one epoch with finite combined loss, a `val/assign_acc` line, and `saved -> ckpt/jenga-smoke/jenga_last.pt`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_jenga.py
git commit -m "feat(jenga): two-stage trainer (Stage-1 visible + teacher-forced Stage-2) + val loop"
```

---

### Task 7: Inference chaining (decode) + local 10-epoch mock

**Files:**
- Modify: `wilddet3d/dense/decode.py` (add `decode_jenga`)
- Test: `tests/dense/test_decode_jenga.py`

**Interfaces:**
- Consumes: `decode_dense` (existing, Stage-1 visible peaks), `JengaStage2`, `rotation_6d_to_matrix`.
- Produces:
  - `decode_jenga(stage1_dets, feat, stride, stage2, catalog, k) -> list[dict]` — for each image: project Stage-1 visible centers → feat-grid queries → run `stage2` → per box: `size = catalog[argmax assign]`, `center = vis_center + center_delta`, `R = rotation_6d_to_matrix(rot6d)`, `score` from Stage 1. Returns `center[M,3], size[M,3], R[M,3,3], score[M], assign[M]`.

- [ ] **Step 1: Write failing test** (synthetic; reuse `JengaStage2(in_ch=8,...)`)

```python
# tests/dense/test_decode_jenga.py
import torch
from wilddet3d.dense.stage2 import JengaStage2
from wilddet3d.dense.decode import decode_jenga

def test_decode_selects_catalog_dim_and_completes_pose():
    s2 = JengaStage2(in_ch=8, d_model=32, layers=2, heads=4).eval()
    feat = torch.randn(1, 8, 12, 12)
    dets = [{"center": torch.tensor([[0.1, 0.0, 1.0], [0.0, 0.1, 2.0]]),
             "size": torch.rand(2, 3), "R": torch.eye(3).expand(2, 3, 3),
             "score": torch.tensor([0.9, 0.8])}]
    catalog = [torch.tensor([[0.3, 0.34, 0.49], [0.3, 0.4, 0.4]])]
    k = torch.tensor([[[500., 0, 6.], [0, 500., 6.], [0, 0, 1.]]])
    out = decode_jenga(dets, feat, stride=84.0, stage2=s2, catalog=catalog, k=k)
    o = out[0]
    assert o["size"].shape == (2, 3) and o["center"].shape == (2, 3)
    assert o["R"].shape == (2, 3, 3)
    # selected size must be an exact catalog row
    for s in o["size"]:
        assert torch.any(torch.all(torch.isclose(catalog[0], s, atol=1e-5), dim=1))
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_decode_jenga.py -v`
Expected: FAIL — `decode_jenga` undefined.

- [ ] **Step 3: Implement `decode_jenga`** in `decode.py`

```python
@torch.no_grad()
def decode_jenga(stage1_dets, feat, stride, stage2, catalog, k):
    """Chain Stage-1 visible dets -> Stage-2 actual boxes (argmax dim selection)."""
    out = []
    for i, det in enumerate(stage1_dets):
        c = det["center"]
        if c.shape[0] == 0:
            out.append({"center": c, "size": c.new_zeros(0, 3),
                        "R": c.new_zeros(0, 3, 3), "score": det["score"],
                        "assign": c.new_zeros(0, dtype=torch.long)})
            continue
        fx, fy = k[i, 0, 0], k[i, 1, 1]
        cx0, cy0 = k[i, 0, 2], k[i, 1, 2]
        z = c[:, 2].clamp_min(1e-3)
        u = (fx * c[:, 0] / z + cx0) / stride
        v = (fy * c[:, 1] / z + cy0) / stride
        uv = torch.stack([u, v], dim=-1)
        vis_obb = torch.cat([c, det["size"], det["R"][:, :2].reshape(-1, 6)], dim=-1)
        res = stage2(feat[i : i + 1], [uv], [vis_obb], [catalog[i]])
        n = c.shape[0]
        assign = res["assign_logits"][0, :n].argmax(-1)
        size = catalog[i][assign]
        center = c + res["center_delta"][0, :n]
        from wilddet3d.dense.rotation_utils import rotation_6d_to_matrix
        R = rotation_6d_to_matrix(res["rot6d"][0, :n])
        out.append({"center": center, "size": size, "R": R,
                    "score": det["score"], "assign": assign})
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/test_decode_jenga.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full unit suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/ -v`
Expected: all pass (data-dependent dataset tests may SKIP if `/storage/...` absent).

- [ ] **Step 6: Local 10-epoch mock (integration test)**

Run on the local box (data + ckpt present), full requested decoder size, small scene subset:
```bash
PYTHONPATH=. .venv/bin/python scripts/train_jenga.py \
  --sim-root /storage/3dl_sim_data/20260625_2skuwallremoval/anyware-sim/build/scenes/synth \
  --max-scenes 60 --epochs 10 --batch-size 2 --val-frac 0.15 \
  --d-model 512 --layers 12 --heads 8 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga-mock10 --no-wandb
```
Expected: combined loss trends down; `train/assign_acc` rises; `val/assign_acc` printed each epoch; ckpt saved. Record the final train/val numbers in the PR/commit message.

- [ ] **Step 7: Commit**

```bash
git add wilddet3d/dense/decode.py tests/dense/test_decode_jenga.py
git commit -m "feat(jenga): inference chaining decode_jenga + local mock validated"
```

---

## Rollout (post-plan, not a code task)

After the local 10-epoch mock is green: rsync to the H100, launch `testing-2` (full data, `--epochs 12 --batch-size 8 --d-model 512 --layers 12 --heads 8 --val-frac 0.1 --wandb --wandb-entity anyware-robotics --wandb-run-name testing-2`). Kill `testing-1` once `testing-2` is confirmed training. Build the real `vis4d_cuda_ops` on the H100 to add 3D-IoU to the val metrics (separate task).

## Self-Review notes

- **Spec coverage:** Stage 1 (existing head, visible target) ✓ Task 6; Stage 2 decoder ✓ Task 3; dim tokens/queries/self+cross attn ✓ Task 3; hard selection ✓ Tasks 3/7; dataset visible+actual+catalog+assign ✓ Task 2; canonicalization ✓ Task 1; val split ✓ Task 2/6; teacher forcing ✓ Task 6; combined loss ✓ Tasks 4/6; CLI knobs ✓ Task 6; decode chaining ✓ Task 7; rollout ✓ above. 3D-IoU deferred (needs CUDA ops) — noted.
- **Placeholders:** none — every code step has concrete code.
- **Type consistency:** `vis_obb` is 12-dim (`center3+size3+rot6`) in Tasks 3/6/7; `assign_logits[B,Qmax,Kmax]`, `center_delta[B,Qmax,3]`, `rot6d[B,Qmax,6]`, `q_mask`/`k_mask` consistent across Tasks 3/4/7; `decode_jenga` consumes `decode_dense` dict keys (`center/size/R/score`).
