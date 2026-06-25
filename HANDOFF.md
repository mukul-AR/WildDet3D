# JENGA — Handoff (Anyware 9-DoF Warehouse Box Detection)

> Prompt-free 9-DoF (full-pose) 3D box detector for warehouse unloading, built
> on WildDet3D. **Branch: `visible_actual_estimation`.**
> Design doc: *Perception V3: WildDet3D-based 3D Box Detector for Warehouse Unloading*
> (Confluence, Software Group). The most up-to-date intent is **9-DoF + prompt-free**.

---

## 0. TL;DR — what exists now

Three things were built and trained, in order. **JENGA (the dense, prompt-free
detector) is the one to carry forward.**

| # | Thing | Prompt? | Status | Checkpoint |
|---|---|---|---|---|
| 1 | **Two-stage** (visible→actual) detector | n/a | runnable, trained (smoke) | `ckpt/two_stage_9dof/` |
| 2 | **Full WildDet3D 9-DoF** (`stage5`) | text-prompted | trained 2 ep | `ckpt/wilddet3d_stage5_anyware_9dof_2ep.ckpt` (8.3 GB) |
| 3 | **JENGA dense detector** (prompt-free) | **none** | trained 6 ep ✅ | `ckpt/dense_9dof/dense_9dof_last.pt` (4.5 GB) |

**JENGA result (6 epochs, head-only):** loss 5.66→1.25, rotation error
**53°→5.4°** (doc gate is <5°). Detects the right box count + position;
orientation tightens with more training. See `viz_out/*.png`.

Stage 5's role now: it **warehouse-adapted the SAM3/depth/fusion encoders**;
JENGA reuses those frozen features. You can also train JENGA straight from the
downloaded stage-2 checkpoint (skip stage 5).

---

## 1. Environment (RTX 5090 / Blackwell sm_120, 24 GB)

`uv` venv at `.venv` (Python 3.11). One-time setup:

```bash
bash scripts/setup_venv.sh
```

What it installs / why (see comments in the script):
- **torch 2.8.0 + cu128** — the README's pinned `torch==2.5.1/cu121` does **NOT**
  support Blackwell (sm_120); cu128 + torch ≥2.7 is required.
- **vis4d 1.0.0** (+ `numpy<2`), **utils3d** (from GitHub), **shapely**.
- **MoGe** cloned to `third_party/moge` (depth-backend training losses).
- **`vis4d_cuda_ops` stub** (`scripts/vis4d_cuda_ops_stub.py` → site-packages):
  the real CUDA ext is eval-only (3D-IoU / rotated NMS / deformable attn) and
  needs a CUDA-12.8 `nvcc` to build for sm_120; training doesn't use it.
- **Weights (public HF):** stage-2 ckpt (`allenai/WildDet3D`, 4.7 GB) + LingBot
  depth backbone (`robbyant/lingbot-depth-pretrain-vitl-14-v0.5`, 1.3 GB →
  `pretrained/lingbot-depth/postrain-dc-vitl14/model.pt`).

**Gotchas baked in:**
- `facebook/sam3` is **gated** → we build SAM3 *structure-only*
  (`load_from_HF=False`); all encoder weights come from the stage-2 ckpt
  (vis4d loads non-strict, strips the `model.` PL prefix).
- Input resolution is **locked at 1008²** (SAM3 RoPE `freqs_cis`); you cannot
  shrink inputs to save memory.
- On a different GPU (Ampere/Ada: 4090 / A6000 / A100), setup is **easier** —
  `vis4d_cuda_ops` builds normally, no stub needed.

---

## 2. Data

| Split | Scenes | Images | Boxes | Note |
|---|---|---|---|---|
| train | 438 | 1,066 | 28,479 | **~99% one site (inhouse P602)** |
| val | 50 | 116 | 2,713 | same site as train |

- Raw scenes: `data/anyware_scenes/<site>/.../place_*/{image.jpg, depth.png,
  metadata.json}` + `scene.json` (9-DoF GT, walls, SKU list).
- Converted COCO JSON: `data/anyware_scenes/annotations/AnywareScenes_{train,val}.json`
  (per-box `center_cam`, `R_cam`, `dimensions`, `bbox2D`, `sku_dims`, K, T_base_cam).
  Built by `scripts/data_prep/anyware/convert_anyware_to_omni3d.py`.
- More data available: **~21K real scenes on s3** (`anyware-perception-capture-scene-data`:
  inhouse 11K, saddle_creek 7.6K, kontoor 2.5K, all with 9-DoF GT) and **~4.9K sim
  scenes** locally (`/storage/3dl_sim_data/synth`).

**Data recommendation (frozen-head training is data-efficient):**
- Dev/iterate: ~1K (current) is fine.
- Trustworthy single-view V1: **~3–5K real, balanced across all 3 sites** +
  **~10K sim** for rotation diversity (real is mostly upright). **Split by SITE**
  for val (current val is the same site → optimistic).
- Only go >10–15K real if you **unfreeze the encoders**.

---

## 3. JENGA — the prompt-free dense detector (`wilddet3d/dense/`)

This is the design doc's "core surgery": drop the text encoder + prompt decoder
+ query 3D head; add a dense conv head on the **depth-fused SAM3 FPN**.

- `model.py` — `DenseDet3D`: reuses SAM3 backbone + LingBot depth +
  `EarlyDepthFusion` (init from a WildDet3D ckpt, **frozen, run under `no_grad`**),
  + a dense head on a fused FPN level. `from_wilddet3d(ckpt, fpn_level, train_fusion)`.
- `head.py` — conv tower → per-cell objectness heatmap `[B,1,Hf,Wf]` + 12-ch reg
  `[du, dv, log_z, log-size(3), 6D-rot(6)]`.
- `targets.py` — project GT 3D centers to the FPN grid, CenterNet Gaussian
  heatmap, per-cell 9-DoF targets.
- `loss.py` — penalty-reduced focal (heatmap) + L1 (offset/log-depth/log-size) +
  **cuboid symmetry-aware** chordal rotation loss.
- `dataset.py` — full 1008² RGBD + resize/pad-adjusted K + GT boxes (camera frame).
- `decode.py` — heatmap peaks → 9-DoF boxes (for inference/viz).

Tap point: `backbone_out["backbone_fpn"]` **after** `EarlyDepthFusion`
(`wilddet3d/model.py:715-733`). FPN levels: 0=288², 1=144² (default), 2=72²
(256-ch each). Depth backend is **100% frozen** (your preference).

### Train JENGA
```bash
PYTHONPATH=. WD3D_SKIP_DEPTH_LOSS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/train_dense_9dof.py \
    --epochs 6 --batch-size 2 --lr 2e-4 --fpn-level 1 \
    --wilddet3d-ckpt ckpt/wilddet3d_stage5_anyware_9dof_2ep.ckpt \
    --out ckpt/dense_9dof
```
- ~5 min/epoch on the 5090, ~9 GB GPU (only 4.82M trainable: head + fusion).
- W&B: add `--wandb` (or `WD3D_WANDB=1`). Defaults to project `jenga-9dof`,
  entity `mukul-ganwal`. **Set `WANDB_API_KEY` in the env — never committed.**
- To skip stage 5: pass `--wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt`.

### Inference
```bash
# Visualize (green = pred, red = GT) -> viz_out/*.png
PYTHONPATH=. .venv/bin/python scripts/visualize_dense_inference.py \
  --num-images 8 --score-thresh 0.3 --out viz_out

# Numbers (counts + center/size/rot error)
PYTHONPATH=. .venv/bin/python scripts/demo_dense_inference.py --num-images 6
```
Flags: `--score-thresh` (lower = more boxes), `--split train|val`, `--dense-ckpt`,
`--base-ckpt`. (Raw-folder RGBD inference, no GT needed, is a TODO — easy to add.)

---

## 4. Full WildDet3D `stage5` (prompt-based) — `configs/training/stage5_anyware_9dof_train.py`

The text-prompted 9-DoF detector (uses a **fixed** `"cardboard box"` prompt →
all boxes in one shot). Its checkpoint provides JENGA's frozen encoders.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WD3D_SKIP_DEPTH_LOSS=1 \
MIXED_PRECISION=bf16 PYTHONPATH=. .venv/bin/vis4d fit \
  --config configs/training/stage5_anyware_9dof_train.py --gpus 1 \
  --ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt
```
- Env-overridable: `WD3D_EPOCHS, WD3D_LR, WD3D_BS, WD3D_ACCUM, WD3D_SHAPE,
  WD3D_BACKBONE_FREEZE, WD3D_LINGBOT_FREEZE` (default 24 = depth fully frozen),
  `WD3D_LIMIT_TRAIN_BATCHES`.
- W&B: `WD3D_WANDB=1` (+ `WD3D_WANDB_PROJECT`, `WD3D_WANDB_ENTITY`, `WD3D_RUN_NAME`).
- On 24 GB: depth loss is skipped (OOMs at 1008²), batch 1 + grad-accum, val off
  (3D evaluator needs the cuda ops). On **48/80 GB** you can drop these hacks.
- Multi-GPU: `--gpus 8` (vis4d DDP already wired).
- Sweep harness: `scripts/sweep_anyware_9dof.sh` (LR sweep; 1e-5 was best).

### The "stages" (why the numbering)
Each stage = a fine-tune of the previous **checkpoint**; you start from a
**downloaded** mid-point, you do NOT retrain 1–3:
- 1/2/3 = upstream Ai2 (Omni3D → all-data → prompt-tune); **downloaded**.
- 4a = teammate's Anyware 9-DoF config (`stage4a_anyware_9dof.py`).
- 5 = our Blackwell-runnable Anyware 9-DoF training (from stage-2).
- JENGA = **not a stage**; dense prompt-free head on stage-5's frozen features.

---

## 5. Two-stage (visible→actual) detector — `wilddet3d/twostage/`

The branch's namesake. Stage 1: RGB-D crop → *visible* OBB. Stage 2: visible OBB
+ container **walls** + scene **SKU** candidates → *actual* OBB (fills occluded
depth, snaps to SKU dims). Self-contained pure-torch (no vis4d).
- Build cache: `scripts/data_prep/anyware/build_two_stage_dataset.py --split {train,val}`
- Train: `scripts/train_two_stage.py --epochs 10`
- Tests: `scripts/test_two_stage_pipeline.py`
This is a separate line of work from JENGA; keep if pursuing the explicit
visible→actual / SKU-snapping idea, otherwise JENGA subsumes single-view detection.

---

## 6. Weights & Biases

Wired into **both** trainers; **opt-in**, **no secrets committed**.
```bash
export WANDB_API_KEY=...        # or: wandb login
# dense:  add --wandb
# stage5: set WD3D_WANDB=1
```
Project `jenga-9dof`, entity `mukul-ganwal` (override via `WD3D_WANDB_PROJECT` /
`WD3D_WANDB_ENTITY`). vis4d routes `pl_module.log(...)` → `trainer.logger`, so a
`WandbLogger` receives all losses/metrics.

---

## 7. GPU / scaling

- **JENGA dense head (frozen encoders):** single GPU is plenty (~9 GB). One
  A100 is overkill-but-fine; 8×A100 underused unless you add DDP.
- **Full fine-tune (unfreeze RGB, depth loss on, batch 2–4):** **48 GB**
  (RTX 6000 Ada / L40S / A6000) is the sweet spot; 24 GB is tight (OOM hacks).
- **Sim-pretrain (10–50K) / parallel sweeps:** **8×A100 80 GB** — best value,
  DDP for the full model, or 8 parallel experiments.
- Recommended freeze recipe for the big run: **keep depth frozen**, optionally
  thaw the SAM3 RGB encoder at low LR.
- Inference is cheap (deploy target in the doc: 12 GB A3500 @ FP16).

---

## 8. Known limitations / gotchas

- 3D-IoU **eval** isn't wired (needs `vis4d_cuda_ops`, not built for sm_120).
  Build it on an Ampere/Ada/CUDA-12.8 box to enable the IoU≥0.95 gate metric.
- Val rotation error (11–28°) > train (5.4°): held-out val + the demo's rough
  nearest-center matching inflate it; also only 6 epochs head-only + single-site.
- Resolution fixed at 1008² (SAM3 RoPE).
- Data is single-site (P602) → generalization unproven; add multi-site + sim.
- `wilddet3d/multiview/*` (from base commit `bbfac93`) is **unused dead code**
  in our pipeline — inert, kept because the load-bearing 9-DoF/symmetry/dataset
  pieces share that commit.

---

## 9. Next steps (suggested order)

1. **Scale data:** pull a balanced multi-site real subset from s3 (use the
   convert script) + wire the 4.9K local sim scenes; split val by site.
2. **Longer JENGA run** on 8×A100 (more epochs; optionally `train_fusion` + thaw
   RGB at low LR, depth frozen). Log to W&B.
3. **Raw-folder inference mode** (`--image-dir` RGBD, no GT) for new captures.
4. **3D-IoU eval** (build cuda ops on Ampere/Ada) to measure the doc gates.
5. (Optional) **Rename** `stage5`→`jenga_encoder_pretrain`, `dense`→`jenga` to
   kill the naming confusion.
6. (Optional) **DDP** for the dense trainer if you want it to use all 8 GPUs.

---

## 10. File map

```
wilddet3d/dense/                 JENGA prompt-free dense detector
  head|targets|loss|model|dataset|decode.py
wilddet3d/twostage/              visible->actual two-stage detector
configs/training/stage5_anyware_9dof_train.py   full WildDet3D 9-DoF train (vis4d)
scripts/
  setup_venv.sh                  one-time env (torch cu128, vis4d, MoGe, weights, stub)
  vis4d_cuda_ops_stub.py         eval-only CUDA-ops stub for sm_120
  train_dense_9dof.py            train JENGA (+ W&B)
  visualize_dense_inference.py   draw predicted/GT 9-DoF boxes on RGB
  demo_dense_inference.py        prompt-free inference (counts + metrics)
  train_two_stage.py             train two-stage detector
  sweep_anyware_9dof.sh          LR sweep for the full model
  data_prep/anyware/
    convert_anyware_to_omni3d.py     raw scenes -> COCO JSON
    build_two_stage_dataset.py       cached two-stage dataset
ckpt/
  wilddet3d_stage2_alldata_12e_v1.0.pt    downloaded base (Ai2)
  wilddet3d_stage5_anyware_9dof_2ep.ckpt  warehouse-adapted encoders
  dense_9dof/dense_9dof_last.pt           ** the JENGA model **
```

(Checkpoints, `.venv/`, `pretrained/`, `data/`, `vis4d-workspace/`, `wandb/`,
`viz_out/` are git-ignored.)
