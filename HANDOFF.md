# JENGA — Handoff (Anyware two-stage 9-DoF warehouse box detector)

> **Prompt-free, single-view 9-DoF 3D box detector** for warehouse unloading.
> Frozen SAM3 + LingBot-Depth encoders → **Stage 1** (visible-box detection) →
> **Stage 2** (dimension-conditioned: inherits orientation, picks the scene SKU,
> places the full *actual* box). Trains on Isaac/anyware-sim data.
> Canonical branch: **`nearface-anchor`** — the superset of all work, merged to
> `main` (2026-07-07). All other branches (`visible_actual_estimation`,
> `depth-encoder-unfreeze`, `multiview-fusion`) are fully contained in it.
> Pushed to `github.com/mukul-AR/WildDet3D`.

---

## 0. TL;DR

- **Two-stage model** (`wilddet3d/dense/`): frozen encoders → Stage-1 dense head
  (visible 9-DoF box) → Stage-2 transformer (dim-conditioned actual box).
- **Stage 2 inherits rotation** from the visible box — verified visible==actual
  rotation = **0°** dataset-wide. It only **selects the scene's SKU** + predicts
  **per-axis extents** (the dim→axis assignment) + a **center** residual.
- **Data:** combined corner-fixed sim, **13,202 scenes** (archived
  `s3://anyware-perception-data-dumps/dataset-unload-3Dlearning/20260627_3D-Sim-Data-13k-2SKU.tar.gz`).
- **Current best:** `real-train-1` — **end-to-end** (Stage-1 detections → Stage-2,
  the deployment-real setting). On **graspable boxes (vis_frac ≥ 0.6)**: 3D IoU
  **0.893**, recall@0.5 **0.986**, recall@0.75 **0.959**; front-of-stack
  (≥0.9): IoU **0.902**. All-boxes IoU 0.857 (dragged by the occluded tail).
  Detection recall 0.976. See §6.
- **Reporting:** headline metric is the **graspable subset** (`vis_frac ≥ 0.6`) —
  in unloading, heavily-occluded boxes are picked later (once un-occluded), so
  their IoU isn't actionable now (we still emit them for collision safety).
- **Gate:** 3D IoU ≥ 0.95 (on graspable boxes), rotation < 5° median.
- **Status (2026-07-06):** current best is **`visweight46k-1`** (0.924 graspable
  E2E, `ckpt/visweight46k1`, pulled to dev box). Active change: **near-face
  center reparameterization** (branch `nearface-anchor`) — Stage 2 now places the
  actual center as **visible near-face + a learned box-local offset** instead of a
  residual off the visible center. Motivation: the sim labels the **visible depth
  extent as a 0.5 m placeholder** when the depth dim is unobserved (**47.5%** of
  boxes; verified), which makes the visible center — and therefore the old
  `center_delta` target — **bimodal** along the ray. The **front-face plane is
  shared exactly** between visible and actual boxes (0 mm, verified), so anchoring
  the actual box there gives a **unimodal** target. This is the untried quadrant
  after the three depth-axis dead ends (finer grid, depth-unfreeze, hard geometric
  anchor). **Status: `nearface-1` is RUNNING on the H100** (visweight46k recipe on
  46k, launched 2026-07-06, ~epoch 6/20, ETA ~2 days, `ckpt/nearface1`) — a clean
  single-variable A/B vs the 0.924 baseline. Judge by `jenga_eval_e2e.py`; the
  mechanism check is whether the *partial* (placeholder) bucket rises toward the
  *corner-match* bucket (§5). Canonical branch is now `nearface-anchor` (== `main`).

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
5. **Stage 2 anchors the center on the visible near-face, not the visible center.**
   The sim labels the visible depth extent as a **0.5 m placeholder** when the
   depth dim is unobserved (47.5% of boxes) → the visible *center* is bimodal
   along the ray, so the old `center_delta` (residual off the visible center) had
   a bimodal target. The **front-face plane is shared exactly** (0 mm) between
   visible and actual, so Stage 2 predicts `face_delta` = a box-local offset from
   the visible **near-face center** (`decode.visible_near_face`): unimodal target,
   convention-invariant anchor. Under occlusion the actual face-center differs from
   the visible one only **in-plane** (0 mm at vis≥0.9 → 35 mm at vis 0.6–0.9), which
   is what `face_delta` learns; the along-ray placement is `+ SKU_depth/2` (known).

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
- **Teacher forcing vs predicted:** `--stage2-input gt` trains Stage 2 on GT
  visible boxes (optimistic val); `--stage2-input predicted` (+ `--tf-warmup-epochs`)
  feeds Stage-1's own detections — the deployment-real setting used by real-train-1.
  End-to-end quality is measured by `jenga_eval_e2e.py` (§5).
- **Joint loss**, encoders frozen by default; single GPU.
- **Encoder fine-tune:** `--encoder-lr` unfreezes **SAM3 (RGB)**; `--depth-encoder-lr`
  unfreezes the **LingBot-depth** backbone; `--depth-unfreeze-blocks N` unfreezes
  only the **last N/24 depth blocks + final norm** (memory-bounded partial fine-tune,
  optimizer groups by `requires_grad`). Active in `depth-unfreeze-1` (last 4 @ 1e-6;
  batch 8 = 18/80 GB — large headroom).
- **RGB feature cache (`--feat-cache-dir`) — ~1.66×/step.** The SAM3 RGB backbone is
  frozen and the dataset applies **no augmentation**, so its FPN output is a
  deterministic function of the image → cacheable. `scripts/precompute_feat_cache.py`
  writes one fp16 `.npy` per view (just the head-read level `backbone_fpn[fpn_level]`);
  the trainer then **skips the SAM3 forward**. Measured **~1.66×/step** on an
  RTX 5090 (SAM3 = ~40% of a fwd+bwd step; the depth path is ~equal cost — so a
  *fully-frozen* run could also cache the fused features for ~5×, but while the depth
  encoder is unfrozen only the RGB half is cacheable). **Opt-in**, defaults
  byte-identical; predictions (`heatmap`/`reg`) verified **bitwise-identical** to the
  uncached path (raw `feat` 0.14% from fp16 storage). Manifest-validated
  (ckpt/fpn/size) + guarded to require the RGB backbone frozen (`--encoder-lr 0`).
  Cache for `data_combined` (~39.6k views @ `[256,144,144]`) ≈ **420 GB** on disk
  (fits the H100's 776 GB; ~half streams from NVMe), ~1 h one-time precompute.
- CLI knobs: `--d-model/--layers/--heads`, `--w-assign/--w-center/--w-size/--w-add`,
  `--val-frac`, `--encoder-lr/--depth-encoder-lr`, `--feat-cache-dir`.

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
| **testing-5** | corrected + learned per-axis + ADD loss | fixed 4k | **0.893** (TF) | teacher-forced; size_err 11.3→1.5 cm, ADD 1.9 cm |
| **real-train-1** | + **predicted boxes** (no teacher forcing) + bigger Stage-1 (78M) | **13.2k** | **0.893** (E2E graspable) | ✅ **current best** — train/test gap closed; 20 ep (5 GT-warmup → 15 predicted) |
| real-train-2 | finer FPN (`fpn-0`, 288²) + head 384/4 + vis-weight 0.6 | 13.2k | 0.828 (E2E graspable) | ❌ **killed @ e18** — finer grid worse on every metric + over-predicting; dead end (see diagnostic) |
| **depth-unfreeze-1** | r1 config + LingBot-depth last-4-block unfreeze @ 1e-6 | 13.2k | **0.880** (E2E graspable) | ❌ **regressed** vs r1's 0.893 (worse S1 1.60 / S2 1.86 cm center too). In-train val was **misleading** — showed 0.890 > r1 0.878, the *opposite* of the authoritative e2e. batch-8 confound. `ckpt/real3` |
| **visweight46k-1** | r1 config + **vis-weight 0.6** (down-weight occluded Stage-2 loss) | **46k** | **0.924** (E2E graspable, 46k val) | ✅ **current best** — front 0.932, all 0.903, recall@.5 0.996, S2 ctr 1.59 cm; in-train val (0.911) *matched* e2e (no lying). ⚠️ vis-weight effect **unattributed** (confounded with 46k data — see §9.1). `ckpt/visweight46k1` |
| **nearface-1** | visweight46k recipe + **near-face center anchor** (see §2.5) | **46k** | ⏳ **RUNNING** (launched 2026-07-06, ~epoch 6/20, ~2 days) | Tests the unimodal near-face reparameterization vs the 0.924 baseline. Clean single-variable A/B (identical recipe). tmux `nearface` on H100, `~/nearface.log`, done-flag `~/nearface.done` → `ckpt/nearface1`. **Judge by `jenga_eval_e2e.py`.** Prediction: lifts the *partial* (placeholder) bucket toward the *corner-match* bucket (§5). |

**`real-train-1` end-to-end eval (recall-inclusive, `jenga_eval_e2e.py`):**
| subset | mean IoU | recall@.5 | recall@.75 |
|---|---|---|---|
| all boxes | 0.857 | 0.950 | 0.906 |
| **graspable (vis≥.6)** | **0.893** | 0.986 | 0.959 |
| front (vis≥.9) | 0.902 | 0.989 | 0.974 |

**Diagnostic (`jenga_eval_e2e.py` tail breakdown):** rotation is **solved**
(0.3–0.4° everywhere — the inherit-rotation design works). The IoU tail is
**occlusion**: `vis_frac<0.3 → IoU 0.57`, `>0.9 → 0.90`. Well-visible boxes have
center 1.5 cm / size 0 / rot 0.3° — the **only residual on graspable boxes is
center precision**.

**Per-axis center diagnostic (free — from exported preds, no GPU; rotate pred−GT
error into the camera frame).** On graspable boxes the error is **zero-mean,
isotropic variance**: no bias, no correlation with box size (so a geometric
depth-anchor has nothing to exploit). **Image-plane is already saturated at
~0.46 cm (≈ ½ pixel)**; the worst axis is **depth (0.74 cm, 43% of the energy),
and it grows with range** (0.93 cm < 1.5 m → 1.47 cm at 2–2.5 m). This is exactly
**why finer FPN (`real-train-2`) failed** — it sharpens the already-saturated
image-plane axis. The lever is therefore **letting the depth features adapt**
(`depth-unfreeze-1`), not grid resolution. *Tabled:* `w_add=0` (ADD-S cleanup,
only if other levers are exhausted); multi-view triangulation — **54% of graspable
boxes are single-camera**, so it's only a partial fix. Sim depth is clean and ZED
gives good metric depth, so this precision work transfers.

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

0. **`nearface-1` (RUNNING) — near-face center reparameterization.** The active
   experiment (§2.5): Stage 2 places the actual center as visible near-face + a
   learned box-local `face_delta` (unimodal target) instead of a residual off the
   bimodal visible center. visweight46k recipe on 46k, tmux `nearface` on the H100,
   `~/nearface.log` / done-flag `~/nearface.done` → `ckpt/nearface1`. **When done:**
   `jenga_eval_e2e.py` on the 46k val vs the **0.924** baseline. Reads: (a) does
   graspable E2E IoU beat 0.924? (b) mechanism — did the *partial* bucket rise toward
   the *corner-match* bucket (§5)? **If better:** promote to best, re-eval on real
   (`jenga_infer_real.py`) — the domain-invariant near-face anchor should also help the
   real depth-shallow bias. **If not:** the two commits (near-face) are a clean revert;
   `visweight46k1` stays best. *Follow-up idea (only after this validates):* corner-match
   as a **training** signal — but aim it at the *harder* partial boxes, and resume from
   the epoch-5 ckpt rather than restart.
1. **`visweight46k-1` (DONE) — prior best model: 0.924 graspable e2e.** r1 config +
   `--vis-weight-thresh 0.6` on the **46k** set, batch 24, 20 ep, no cache →
   `ckpt/visweight46k1`. **E2E (46k val): graspable IoU 0.924 (front 0.932, all 0.903),
   recall@.5 0.996, S2 center 1.59 cm** — and the in-train val (0.911) *matched* the e2e
   (no depth-unfreeze-style lying, so it's trustworthy). **BUT the win is confounded:** on
   the *same* 46k val set, r1 scores only **0.690** graspable (0.893→0.690 — r1 is
   out-of-distribution on 46k's 2× SKU variety; even Stage-1 recall drops to 0.82). So
   **46k is a genuinely harder/distinct distribution**, and the +0.23 gap is dominated by
   "trained on 46k vs not," NOT the vis-weighting. The **vis-weighting's own effect is
   unattributed** — isolating it needs a 46k-WITHOUT-vis-weight baseline (~3 days), which
   isn't worth it vs the appearance gap (§9.2). Deployment (real) is unchanged by this run.
2. **THE real lever = the sim→real APPEARANCE gap (confirmed NOT calibration).** On a
   real capture (`data/place_*`; GT in `boxes_yaml_string`; `jenga_infer_real.py`):
   sim↔real intrinsics are **byte-identical** (the sim is calibrated to the real pole
   rig — same fx/fy/FOV/camera positions), yet on real the model gives **rotation 9°
   (sim 0.3°), center 17 cm, depth-shallow −5.8 cm**. Geometry is perfect → the gap is
   pure **appearance** (real RGB + real-estimated depth ≠ sim-rendered clean RGB/depth).
   More sim data won't fix it (46k = same occlusion mix as 13.2k, only +2× SKU variety).
   Levers: (a) real-data fine-tune (needs more labeled `place_*` captures); (b) **RGB +
   depth domain randomization / augmentation** in sim — the dataset currently does
   **ZERO augmentation**, so large headroom. *Rejected:* train on LingBot-sim-depth
   (not what the deployment camera outputs).
3. **Geometric near-face anchor** (`--posthoc-anchor`, eval + infer): replaces the
   learned center_delta with "anchor the actual box's camera-facing face to the visible
   box's near face, grow back by the SKU depth." Domain-invariant → on real it **halves
   the depth shallow-bias** (−5.8→−2.8 cm); on sim it **regresses graspable 0.893→0.861**
   (neutral on front vf≥.9, breaks under mutual occlusion). Optional high-vis/real
   inference toggle, not a default (refined depth already made real depth error minor).
4. **Tabled levers:** `w_add=0` (ADD-S cleanup); multi-view fusion (partial — 54% of
   graspable boxes are single-camera). NOT yet: SAM3 (RGB) fine-tune (premature on sim RGB).
5. **Infra:** real `vis4d_cuda_ops` on the H100 (Hopper) for exact 3D-IoU (currently MC);
   per-camera viz filter; DDP. Big-dataset I/O: **stage to local NVMe** (46k enumerates in
   2.3 s local vs minutes on NFS); the RGB feature cache is **not** worth it for large
   sets — a cached feat (10.6 MB) is 8× a raw image, so on NFS it's *more* I/O to save
   idle compute; only `data_combined` (RAM-cached, local) is a case where it helps.

### Confirmed dead ends (don't re-run)
- **Finer FPN grid** (`--fpn-level 0`, real-train-2): E2E graspable 0.893 → **0.828**.
- **Size-aware symmetry loss** (testing-3): IoU 0.81 → 0.74.
- **Depth-encoder unfreeze** (`depth-unfreeze-1`, last-4 blocks @ 1e-6, batch 8): E2E
  graspable 0.893 → **0.880** (+ worse S1/S2 center). ⚠️ in-train val showed the
  *opposite* — **always confirm with `jenga_eval_e2e.py`, not the training val print**.
- **Geometric center anchor on sim** (`--posthoc-anchor`): graspable 0.893 → **0.861**
  (it helps only the *real* depth-bias; see §9.3).

### Tooling added (this session)
- `scripts/jenga_eval_e2e.py` — end-to-end eval: recall@IoU + **visibility-
  stratified** report (all / graspable / front) + tail diagnostic + **corner-match
  ("fully observed") stratum**: a GT-only flag for boxes whose visible corners
  coincide with actual corners (depth snapped, no in-plane crop). Sharper "we fully
  see it" signal than vis_frac (44% of vis≥.9 boxes still carry placeholder depth).
  Verified on `visweight46k1`: **fully-observed 0.934 vs partial 0.904 IoU**; within
  graspable 0.935 vs 0.923 — adds signal beyond vis_frac. But the gap is modest:
  placeholder-depth boxes are still well-recovered via the SKU catalog (assign 0.98),
  so "not fully observed" ≠ "badly estimated"; occlusion (vis<.3 → 0.74) is the real
  IoU driver. Parameterization-independent, so it reads on both old and near-face models.
- ADD-S (symmetry-tolerant) corner loss; occlusion-aware loss weighting
  (`--vis-weight-thresh`); per-epoch **graspable-IoU** in the val print.
- `--stage2-input predicted` (+ `--tf-warmup-epochs`, `--resume`), bigger Stage-1
  head (`--head-width/--head-convs`), encoder fine-tune flags (`--encoder-lr`).
- `scripts/precompute_feat_cache.py` + `--feat-cache-dir` — cache the frozen SAM3
  RGB FPN to skip its forward (~1.66×/step; opt-in, manifest-validated, verified
  prediction-identical). See §4. Recipe: precompute once, then add the one flag.
  ⚠️ Only worth it for small **local, RAM-cached** sets (`data_combined`); for large
  sets the ~1.5 TB cache doesn't fit locally and on NFS it's *more* I/O than raw images.
- `scripts/jenga_infer_real.py` — **real-capture** inference (ROS `camera_info` +
  quaternion extrinsics, `image.jpg`, `refined_depth.png`): GT/S1/S2 HTML overlays +
  center / **signed-depth** / **symmetry-aware rotation** error vs GT + `--posthoc-anchor`.
- `scripts/jenga_pcd_viz_real.py` — 3D colored point-cloud viewer (base_link, multi-cam).
- `--posthoc-anchor` (`decode_jenga` / eval / infer) — geometric near-face center anchor (§9.3).
- **Loader resilience**: `SimJengaDataset.__getitem__` skips corrupt/unreadable
  `rgb.png` (cv2→None → advance to next sample) so one bad file can't kill a long run.

(Checkpoints, `.venv/`, `pretrained/`, `data/`, `wandb/`, `viz_out/` are git-ignored.)
