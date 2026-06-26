# JENGA — Two-Stage, Dimension-Conditioned 9-DoF Detector (Design)

**Date:** 2026-06-25 · **Branch:** `visible_actual_estimation`

## Motivation

Real warehouse catalogs are open-ended (effectively infinite SKUs), so any model
that bakes a fixed catalog into its weights (a class head / learned SKU vocab) is a
dead end. We decouple perception from catalog knowledge:

- **Stage 1 (catalog-blind):** detect the **visible** boxes from RGB-D.
- **Stage 2 (dimension-conditioned, learned-implicit):** given the visible boxes
  **plus a per-scene set of candidate `(L,B,H)` dimension triples** (a manifest — no
  ids, no classes, just raw dims), infer the **actual** full boxes — for each box,
  *select which candidate dimension it is* and complete its full pose, so the whole
  set fits together as a physically consistent packed scene ("the JENGA").

The catalog never enters the weights; it is purely a variable-length input. This is
exactly representable in the current sim data (per box: `visible_*`, `actual_*`; per
scene: catalog dims from `scene.json`'s `skus_yaml_string`).

## Architecture — one model, frozen encoders, two heads

Frozen SAM3 + LingBot-depth + EarlyDepthFusion (from the WildDet3D stage-2 ckpt,
`no_grad`), tapped at the fused FPN level (default 1 = 144², 256-ch), as today.

### Stage 1 — visible detection
Reuse the existing dense CenterNet-3D head (`wilddet3d/dense/head.py`) with
`target="visible"`: per-cell objectness heatmap + 12-DoF visible OBB
`[du, dv, log_z, log-size(3), 6D rot]`. Unchanged mechanics; only the supervision
target is the visible box.

### Stage 2 — dimension-conditioned fitting (new)
A small transformer decoder. **Defaults: `d_model=512, layers=12, heads=8`** (start
big per decision; shrink from curves). All sizes are CLI knobs.

- **Dim tokens:** each scene-catalog triple, **sorted** (permutation-canonical) → MLP
  → token. Scene with *k* SKUs → *k* tokens; variable *k* handled by attention.
- **Queries:** one per visible box = pooled encoder feature at the box's center cell
  (bilinear sample of the fused FPN map) + an embedding of the visible OBB.
- **Decoder block (×layers):** self-attention among queries (mutual "jenga"
  consistency) + cross-attention over the dim tokens (allowed sizes).
- **Per-query outputs:**
  - **Assignment** logits over the *k* dim tokens. Actual **size = selected dim**:
    softmax-weighted combination at train time, argmax at inference (**hard
    selection** — option i; actual size is always a valid catalog dim by
    construction).
  - **Actual pose:** center as a residual off the visible center (+ log-depth) and a
    6D rotation.

## Data / supervision

Extend the sim dataset (new `sim_jenga_dataset.py` or extend `SimDenseDataset`) to
yield, per camera view:
- visible OBBs (current path, `target=visible`),
- actual OBBs (`actual_*`),
- the **scene catalog dim set** (distinct sorted `(L,B,H)` from `skus_yaml_string`,
  parsed with PyYAML),
- per-box **GT assignment index** = index of the box's `actual_geometry` (sorted)
  within the catalog dim set.

**Train/val split** by scene hash (90/10) — prerequisite for pruning decisions.

Val metrics: assignment accuracy, actual center/rot error, actual-box 3D-IoU (once
real `vis4d_cuda_ops` is built on the H100).

## Training

- **Teacher forcing:** Stage 2 trains on **GT visible boxes** as queries (decoupled
  from Stage-1 quality). At inference it chains off Stage-1 predictions. Optional
  later: fine-tune Stage 2 on predicted boxes.
- **Joint run, encoders frozen.** Combined loss:
  - Stage 1: existing visible detection terms (focal heatmap + offset + depth + size
    + symmetry rotation).
  - Stage 2: assignment cross-entropy + actual-center L1 + actual-depth L1 +
    symmetry-aware chordal rotation. **No explicit packing loss** (learned-implicit).
- All loss weights, decoder dims, and the split fraction are CLI knobs.

## Config & rollout

- New/extended train script (`scripts/train_jenga.py` or extend
  `train_dense_9dof.py`) exposing `--d-model --layers --heads`, Stage-2 loss weights,
  `--val-frac`, plus existing flags. W&B logs both stages' losses + val metrics.
- **Rollout:** build → **local 10-epoch mock** on a scene subset (both heads train,
  val metrics log, ckpt saves) → push to H100 → launch **`testing-2`**. Leave
  `testing-1` (single-head, target=actual baseline) running until `testing-2` is
  ready, then kill it.

## Files touched

- `wilddet3d/dense/sim_dataset.py` — emit visible + actual + catalog + assignment idx;
  scene split.
- `wilddet3d/dense/stage2.py` (new) — dim tokens, query builder, transformer decoder,
  hard-selection size + pose heads.
- `wilddet3d/dense/loss.py` — add Stage-2 loss (assignment CE + actual pose).
- `wilddet3d/dense/model.py` — wire Stage 2 onto the fused FPN tap; expose a combined
  forward (Stage-1 dense maps + Stage-2 per-query outputs).
- `wilddet3d/dense/decode.py` — inference chaining (Stage-1 peaks → queries → Stage-2
  actual boxes); argmax dim selection.
- `scripts/train_*.py` — combined loss, teacher-forced Stage-2, val loop, new CLI.
- `scripts/visualize_dense_inference.py` — draw actual (selected-dim) boxes.

## Risks / open items

- **Teacher-forcing gap:** Stage 2 sees GT visible at train, predicted at inference.
  Mitigate later with predicted-box fine-tuning if the gap shows in val.
- **Overfitting** the large Stage-2 head on ~5.3k scenes (data will grow): the val
  split is how we catch it; shrink `d_model/layers` if val ≪ train.
- **Assignment ambiguity** for near-identical catalog dims (~1 cm pairs): CE may be
  noisy but geometrically harmless; monitor assignment accuracy vs size error
  separately.
- **3D-IoU eval** needs the real `vis4d_cuda_ops` built on the H100 (Hopper).
