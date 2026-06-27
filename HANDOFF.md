# JENGA — Handoff (Anyware two-stage 9-DoF warehouse box detector)

> **Prompt-free, single-view 9-DoF 3D box detector** for warehouse unloading.
> Frozen SAM3 + LingBot-Depth encoders → **Stage 1** (visible-box detection) →
> **Stage 2** (dimension-conditioned: inherits orientation, picks the scene SKU,
> places the full *actual* box). Trains on Isaac/anyware-sim data.
> Branch: `visible_actual_estimation` (pushed to `github.com/mukul-AR/WildDet3D`).

---

## 0. TL;DR

- **Two-stage model** (`wilddet3d/dense/`): frozen encoders → Stage-1 dense head
  (visible 9-DoF box) → Stage-2 transformer (dim-conditioned actual box).
- **Stage 2 inherits rotation** from the visible box — verified visible==actual
  rotation = **0°** dataset-wide. It only **selects the scene's SKU** + predicts
  **per-axis extents** (the dim→axis assignment) + a **center** residual.
- **Data:** fixed 2-SKU sim dump (corner-fix), **3,978 scenes**. Old data deprecated.
- **Current best:** `testing-5` — teacher-forced 3D IoU **0.893** (size_err
  collapsed 11.3 → 1.5 cm, ADD 1.9 cm) via inherited rotation + learned per-axis
  assignment + ADD loss. See §6. (Gate 0.95; this is teacher-forced — the
  end-to-end Stage-1→Stage-2 number is still TBD.)
- **Gate:** 3D IoU ≥ 0.95, rotation < 5° median (Perception-V3 doc).

```bash
# train (fixed data, frozen encoders, W&B opt-in)
PYTHONPATH=. .venv/bin/python scripts/train_jenga.py \
  --sim-root ~/data_fixed --epochs 12 --batch-size 16 --workers 14 --val-frac 0.1 \
  --d-model 512 --layers 12 --heads 8 \
  --wandb --wandb-entity anyware-robotics --wandb-run-name testing-N \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jengaN
```

---

## 1. Architecture (`wilddet3d/dense/`)

```
RGB-D (1008²) + per-scene SKU catalog + K
        │
   ❄ SAM3 ViT-L/14 (RGB) ─┐
   ❄ LingBot-Depth DINOv2 ─┼─ ❄/🔥 EarlyDepthFusion ─► fused FPN feat [B,256,144,144]
                            │   (fusion trainable, 0.07M)
        ┌───────────────────┴───────────────────┐
   🔥 STAGE 1 (dense head, ~5.4M)          (same feat)
   per-cell heatmap + 12-ch 9-DoF              │
   VISIBLE OBB                                 │
        │ visible boxes (R, size, center)      │
        ▼  → queries (teacher-forced GT in train)
   🔥 STAGE 2 (transformer, d=512/12L/8H, ~52M)
   queries (feat sample + OBB embed) self-attn + cross-attn over SKU dim tokens
   → assignment over scene SKUs  +  per-axis log-extents  +  center Δ
   → actual box = R_visible (inherited) + SKU dims @ predicted axis-order + center
```

- **Frozen** (no_grad): SAM3 backbone + LingBot-Depth backbone (~1.1B total).
  Optionally fine-tunable at low LR — see §4.
- **Trainable** (~57M): EarlyDepthFusion + Stage-1 head + Stage-2 transformer.
- **Input locked 1008²** (SAM3 RoPE).
- Files: `model.py` (DenseDet3D + frozen encoders + `from_wilddet3d`), `head.py`
  (Stage-1 dense head), `stage2.py` (Stage-2 transformer), `targets.py`,
  `loss.py`, `decode.py`, `metrics.py`, `sim_jenga_dataset.py`,
  `rotation_utils.py`, `jenga_utils.py`.

### Relationship to WildDet3D
We **keep WildDet3D's frozen encoders + fusion** and **replaced its head**:
original WildDet3D is **7-DoF (yaw-only) + prompt-conditioned**; ours is **9-DoF
(full rotation), prompt-free, two-stage**. The 6D rotation rep (Zhou 2019) is used
in Stage 1; Stage 2 inherits it.

---

## 2. The key design decisions (and why)

1. **Stage 2 inherits rotation, doesn't predict it.** Measured: visible & actual
   boxes share orientation **exactly 0°** (camera sees the front face). So
   orientation is observed via the visible box; Stage 2 has no rotation
   output/loss/symmetry/canonicalization. This removed the part that was going
   wrong.
2. **Per-scene SKU catalog is an *input*, never baked into weights.** Real
   catalogs are open-ended. Each scene has **1–2 SKUs** (12% single = no choice,
   88% binary; only ~5% are "face-ambiguous" — same front face, different depth —
   and need scene-level reasoning, handled by Stage-2 self-attention).
3. **Stage 2 predicts per-axis extents** (the dim→axis assignment), not just
   "which SKU". Even with orientation fixed, *which SKU dim goes on which axis* is
   an open decision; getting it wrong is the dominant error (see §6 testing-4).
4. **Corner-distance (ADD) loss + metric.** 3D IoU saturates on cube-ish boxes
   (rotating/mis-assigning barely changes volume) → it can't supervise the axis
   assignment. ADD (corner-to-corner, fixed correspondence) does. ADD-S is the
   symmetry-tolerant variant.

---

## 3. Data — fixed 2-SKU sim dump

```
<root>/synth_<hash>/
    scene.json                         # world-frame boxes {extrinsic_4x4, geometry, sku}, skus_yaml_string
    0_camera_pole_{bottom,left,right}/  # 3 cams/scene
        rgb.png, depth.png (uint16 mm), metadata.json
metadata.json: intrinsics, camera_extrinsic_4x4 (cam->world),
    boxes{uuid: {visible_extrinsic_4x4, visible_geometry,
                 actual_extrinsic_4x4, actual_geometry, visible_fraction, sku}}
```

- **Source:** `s3://anyware-perception-data-dumps/dataset-unload-3Dlearning/20260625_3D-Learning-Dump_2-SKU.tar.gz`
  (13.4 GB, the **corner-fixed** dump — earlier data had bad visible corners).
- **Local:** `/storage/3dl_sim_data/20260625_fixed/synth_*` (3,978 scenes).
- **H100:** `~/data_fixed/synth_*`.
- `SimJengaDataset` yields, per camera view: RGB-D (1008², resize+pad,
  ImageNet-norm), pad-adjusted K, **visible** OBBs, **actual** OBBs (native
  per-axis size + native rotation == visible), **scene catalog** (sorted dims),
  per-box **assignment** index, all camera-frame. Hash-based **train/val split**
  (`--val-frac`).
- ⚠️ Old buggy data (`/storage/3dl_sim_data/scenes`, ~5,281 scenes) is deprecated;
  don't train on it.

---

## 4. Loss & training (`scripts/train_jenga.py`)

- **Stage 1** (`DenseDet3DLoss`, on the *visible* box): focal heatmap + L1 offset
  + L1 log-depth + L1 log-size + **4-symmetry chordal rotation** (NOT size-aware —
  see §6 testing-3).
- **Stage 2** (`JengaStage2Loss`): `assign CE + center L1 + per-axis size L1 +
  ADD corner`. No rotation term (inherited).
- **Teacher forcing:** Stage 2 trains on GT visible boxes; inference chains
  Stage-1 predictions → Stage-2 (`decode_jenga`). ⚠️ Val metrics are therefore
  *optimistic* (assume perfect visible detection). **Stage-1 quality is never
  measured end-to-end yet** — a known gap.
- **Joint loss**, encoders frozen by default; single GPU.
- **Optional encoder fine-tune:** `--encoder-lr 1e-5` unfreezes **SAM3 (RGB)**;
  `--depth-encoder-lr` unfreezes the depth backbone (default 0 = frozen). Uses
  discriminative LR groups. Heavy (batch must drop to ~2–4). Not yet run.
- CLI knobs: `--d-model/--layers/--heads`, `--w-assign/--w-center/--w-size/--w-add`,
  `--val-frac`, `--encoder-lr/--depth-encoder-lr`.

---

## 5. Metrics & eval

- `wilddet3d/dense/metrics.py`: **Monte-Carlo 3D IoU** (full rotation, pure-torch;
  pytorch3d/CUDA ops unavailable on this stack), IoU@0.5/0.75, center dist, size
  err, **ADD / ADD-S** (corner distance), pairwise overlap (non-intersection),
  assign acc. Logged per-epoch in the val loop + W&B.
- `scripts/eval_jenga.py --ckpt <jenga_last.pt> --n-samples 8192` — scores a
  checkpoint with the full suite (teacher-forced).
- `scripts/jenga_export_pred.py` — runs the **full chain** (Stage-1 → Stage-2)
  per scene, writes world-frame predicted boxes as `<pred-dir>/<scene>.json` for
  the viz tool.
- 18 unit tests in `tests/dense/` (`PYTHONPATH=. .venv/bin/python -m pytest tests/dense/`).

---

## 6. Runs (W&B project `jenga-9dof`, entity **`anyware-robotics`**)

| run | design | data | 3D IoU | notes |
|---|---|---|---|---|
| testing-1 | old single dense head | old | — | killed |
| **testing-2** | old (predicts rotation, canonical frame) | old | **0.809** | hid the per-axis issue via canonicalization |
| testing-3 | + size-aware symmetry loss | old | 0.742 | **worse → reverted** (see memory) |
| **testing-4** | corrected (inherit rotation, **heuristic** placement) | fixed | **0.730** | **size err 11.3 cm** — exposed the per-axis bug |
| **testing-5** | corrected + **learned per-axis + ADD loss** | fixed | **0.893** | ✅ **current best** — size_err 11.3→1.5 cm, ADD 1.9 cm, @.75 0.89, center 1.7 cm |

**Diagnosis from testing-4 → fix in testing-5:** testing-4 had assign acc 0.945,
center 2.8 cm, rotation 0°, but **size err 11.3 cm** and ADD 5.5 cm → the dominant
error was the **dim→axis placement** (right SKU, wrong axes). Making that placement
learned + ADD-supervised (testing-5) collapsed size err to **1.5 cm** and lifted
3D IoU 0.73 → **0.893**. All numbers are **teacher-forced** (GT visible boxes); the
end-to-end (Stage-1 predictions → Stage-2) number is the next thing to measure.

⚠️ **W&B entity is `anyware-robotics`** (the script default `mukul-ganwal` fails).

---

## 7. Lambda H100 deployment

```bash
ssh ubuntu@209.20.157.13          # key-based, 1× H100 80GB
```
- Code synced via **rsync from the dev box** (the box has no GitHub auth):
  `rsync -az --exclude='/.venv/' --exclude='/ckpt/' ... ./ ubuntu@209.20.157.13:~/WildDet3D/`
- venv: `bash scripts/setup_venv.sh` (torch 2.8 cu128, vis4d, MoGe, weights,
  `vis4d_cuda_ops` stub). `wandb` + `pytest` installed separately. `uv` installed.
- Data: `~/data_fixed/` (extracted fixed dump). W&B key in `~/.wandb_env`.
- Long jobs run in remote `tmux` (`tmux new -s testing-N`). Checkpoints in
  `~/WildDet3D/ckpt/jengaN/jenga_last.pt`.
- ⚠️ Ephemeral disk — re-sync repo/data + re-run setup on a fresh boot.

---

## 8. Viz tool (GT vs predicted)

- Web viewer at **`/storage/3dl_sim_data/viz_tool`** (Three.js + python server),
  modified to overlay **predicted boxes (red)** vs **GT (green)** via `--pred-dir`.
- Generate preds: `scripts/jenga_export_pred.py --ckpt ... --scenes-dir ... --pred-dir ...`
- Run (in **your** terminal — a sandboxed server gets killed):
  `/home/mukul/WildDet3D/.venv/bin/python /storage/3dl_sim_data/viz_tool/server.py --root <dir> --pred-dir <preds>` → http://localhost:8000
- ⚠️ Current preds are from **testing-2 on OLD data** — re-export from a
  fixed-data model (e.g. testing-5) before trusting the overlay.

---

## 9. Next steps

1. **Watch `testing-5`** — does `size_err` collapse from 11 cm → ~1 cm and IoU
   climb past 0.73 (toward/above 0.81)? If yes, the learned per-axis fix worked.
2. **Re-export viz** from the best fixed-data model → inspect GT vs predicted.
3. **SAM3 fine-tune** (`testing-6`, `--encoder-lr 1e-5`, batch ~4) — lets Stage 1
   reshape features (the head does all adaptation when encoders are frozen).
4. **Measure Stage 1 end-to-end** (visible-box recall / errors) — it's never been
   measured directly; if it's the bottleneck, consider a stronger/multi-scale head.
5. **Real `vis4d_cuda_ops`** on the H100 (Hopper) for exact 3D-IoU (currently MC).
6. Scale sim data; DDP for multi-GPU.

(Checkpoints, `.venv/`, `pretrained/`, `data/`, `wandb/`, `viz_out/` are git-ignored.)
