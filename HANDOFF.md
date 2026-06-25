# JENGA — Handoff (Anyware 9-DoF Warehouse Box Detection)

> **Prompt-free, single-view 9-DoF 3D box detector** for warehouse unloading,
> built on WildDet3D's frozen SAM3 + LingBot-Depth encoders with a dense conv
> head. **Trains on Isaac/anyware-sim data only.** Branch: `visible_actual_estimation`
> (pushed to `github.com/mukul-AR/WildDet3D`).

The repo is **stripped to one path**: sim data → dense JENGA detector. All
prompt-based / two-stage / multiview / real-COCO code was removed.

---

## 0. TL;DR

- **Model:** `wilddet3d/dense/` — dense CenterNet-3D head over the depth-fused
  SAM3 FPN. No text, no prompts, no per-object input.
- **Data:** sim only (`SimDenseDataset`) — each camera's `metadata.json` has, per
  box, `visible_`/`actual_` `extrinsic_4x4` + `geometry` (+`visible_fraction`);
  boxes are world-frame, `center_cam = inv(camera_extrinsic_4x4) @ pose`.
- **Encoders:** frozen, loaded from the downloaded stage-2 WildDet3D checkpoint;
  only the dense head (+fusion) trains (~4.8M params).
- **Verified:** 5-epoch local sim run trains cleanly (loss ↓, rot-err ↓, ckpt saved).

```bash
PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \
  --sim-root <anyware-sim>/build/scenes/synth --sim-target actual \
  --epochs 6 --batch-size 2 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga
```

---

## 1. Environment (`scripts/setup_venv.sh`)

uv venv at `.venv` (Python 3.11). One-time: `bash scripts/setup_venv.sh`.
- **torch 2.8.0 + cu128** (README's 2.5.1/cu121 does NOT support Blackwell
  sm_120; cu128 also works on A100/H100).
- vis4d 1.0.0 (+ `numpy<2`), utils3d, shapely, **MoGe** (`third_party/moge`).
- **`vis4d_cuda_ops` stub** (`scripts/vis4d_cuda_ops_stub.py`): the real ext is
  eval-only and needs CUDA-12.8 nvcc for sm_120; training doesn't use it. On
  Ampere/Hopper you can build the real one (enables 3D-IoU eval).
- Weights (public HF): stage-2 ckpt (`allenai/WildDet3D`, 4.7 GB) + LingBot
  depth backbone (`robbyant/...`, 1.3 GB).
- `facebook/sam3` is **gated**: SAM3 is built structure-only; its weights come
  from the stage-2 ckpt. Input is **locked to 1008²** (SAM3 RoPE).

---

## 2. Data — sim (the only format)

```
<root>/anyware-sim/build/scenes/synth/synth_<hash>/
    scene.json                         # world-frame boxes, container, skus
    {idx}_camera_pole_{bottom,left,right}/   # 3 cams/scene
        rgb.png, depth.png (uint16 mm), metadata.json
metadata.json:
    intrinsics {fx,fy,cx,cy,width,height}
    camera_extrinsic_4x4               # camera pose in world (cam->world)
    boxes{uuid: {visible_extrinsic_4x4, visible_geometry,
                 actual_extrinsic_4x4,  actual_geometry, visible_fraction, sku}}
```
`SimDenseDataset(sim_root, size=1008, target="actual"|"visible", max_scenes)`
yields one sample **per camera view** (single-view): RGB-D (1008², resize+pad,
ImageNet-norm), pad-adjusted K, and camera-frame GT (`center`, box-axis `size`,
6D `rot`, projected `box2d`). `target` picks the visible (Stage-1) or actual
(full) box. Real captures lack a visible/actual split → we standardised on sim
(which renders both).

---

## 3. Model — `wilddet3d/dense/`

- `model.py` `DenseDet3D` — reuses SAM3 backbone + LingBot depth +
  `EarlyDepthFusion` (init from a WildDet3D ckpt via `from_wilddet3d(...)`,
  **frozen, `no_grad`**) + dense head on a fused FPN level (default 1 = 144²,
  256-ch). Tap point: `backbone_out["backbone_fpn"]` after fusion.
- `head.py` — per-cell objectness heatmap + 12-ch reg `[du, dv, log_z,
  log-size(3), 6D rot(6)]`.
- `targets.py` — project GT centers to the FPN grid, CenterNet Gaussian
  heatmap, per-cell 9-DoF targets.
- `loss.py`, `rotation_utils.py`, `sim_dataset.py`, `decode.py` (see below).

Depth backend + SAM3 backbone are **always frozen**; only the dense head
(+fusion) trains.

---

## 4. Loss (`wilddet3d/dense/loss.py`)

### 4.1 Current (single-target)
`DenseDet3DLoss` predicts **one** OBB per cell, supervised against **one** target
(`visible` OR `actual`, set by `--sim-target`):
```
L = λ_hm·focal(heatmap) + λ_off·L1(du,dv) + λ_z·L1(log_z)
    + λ_size·L1(log w,h,l) + λ_rot·symmetry_chordal_rotation
```
- **Heatmap / "number of boxes":** there is **no explicit count term**. The
  count is *emergent* — focal loss puts a Gaussian peak at each GT center and
  suppresses elsewhere; a missed box (low score at a true center) or a
  hallucination (high score on an empty cell) is penalised per-cell. At
  inference the box count = heatmap peaks above `--score-thresh` (a tunable knob).
- **Rotation:** 6D continuous rep (Zhou 2019) → Gram-Schmidt → R. Loss is the
  **chordal distance**, minimised over the **4 cuboid symmetries** (a box looks
  identical flipped 180°), so the model isn't punished for a physically-correct
  but differently-labelled orientation. Reported metric `rot_deg` =
  symmetry-aware geodesic angle in degrees (the "how close is the orientation"
  number in the logs). Captures the **full 3D** rotation (tilt/lean), not yaw-only.

### 4.2 Planned (richer multi-task — uses the sim's visible+actual+SKU)
The single-target loss underuses the data. Next upgrade keeps ONE dense
detector but makes the head multi-task:
```
heatmap(1) + visible_OBB(12) + actual_OBB(12)
L = λ_hm·focal + λ_vis·OBB(visible) + λ_act·OBB(actual) + λ_sku·snap(actual_size → nearest scene SKU)
```
- **visible** ← `visible_extrinsic_4x4`+`visible_geometry` (what the camera sees).
- **actual** ← `actual_extrinsic_4x4`+`actual_geometry` (full box; the deploy output).
- **SKU snap** ← rotation-invariant L1 pulling predicted *actual* dims to the
  nearest catalog SKU (from `scene.json` `skus_yaml_string`); acts as a catalog
  prior so predictions are valid known sizes. (Optionally a SKU-classification
  head instead.) This is the visible→actual→SKU pipeline the branch is named for,
  now with **real labels** (no faked occlusion). **Not implemented yet** — the
  current head/loss is single-target.

---

## 5. Train / Visualize / Inference

```bash
# Train (sim) — W&B opt-in via --wandb
PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \
  --sim-root <synth_dir> --sim-target actual --epochs 6 --batch-size 2 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga [--wandb]

# Visualize predicted (green) vs GT (red) 9-DoF boxes -> viz_out/*.png
PYTHONPATH=. .venv/bin/python scripts/visualize_dense_inference.py \
  --sim-root <synth_dir> --sim-target actual \
  --dense-ckpt ckpt/jenga/dense_9dof_last.pt --num-images 8 --score-thresh 0.3

# decode_dense (wilddet3d/dense/decode.py): heatmap peaks -> 9-DoF boxes
```
~5 min/epoch on a 24 GB GPU at batch 2 (encoders frozen, ~9 GB used); faster on
H100/A100. Single-GPU (no DDP yet). Inference ≈ 295 ms/img bf16 on a laptop
5090 (dominated by the frozen encoders, not the head).

---

## 6. Weights & Biases

Opt-in, **no secrets committed**. `export WANDB_API_KEY=...` then `--wandb`.
Defaults: project **`jenga-9dof`**, entity **`mukul-ganwal`** (override via
`WD3D_WANDB_PROJECT` / `WD3D_WANDB_ENTITY` / `WD3D_RUN_NAME`). Logs per-step +
per-epoch loss components, `rot_deg`, `num_pos`, LR.

---

## 7. GPU / scaling

- Frozen-head training: single GPU is plenty (~9 GB). **1× H100 80 GB** is the
  current sweet spot.
- **8× A100/H100** only for sim-pretrain on lots of data, parallel sweeps, or a
  full fine-tune with DDP (then add DDP to the trainer).

---

## 8. Lambda H100 deployment

### SSH / access
```bash
ssh ubuntu@209.20.157.13          # user: ubuntu, key-based (no password)
```
- Instance: **1× NVIDIA H100 PCIe 80 GB**, 26 vCPU, ~968 GB free disk.
- Image: **Ubuntu 22.04 LTS + Lambda Stack** (default).
- Local gateway: `tmux 3dl` (on the dev box, for connecting/attaching).
- Remote long-running jobs should run inside a **remote tmux** (e.g.
  `tmux new -s testing-1`) so they survive disconnects.

### Status — what's been done on the box
| Item | Status | Detail |
|---|---|---|
| SSH access | ✅ done | key-based, verified |
| Sim dataset (19 GB) | ✅ transferred | `~/20260625_2skuwallremoval.tar.gz` (rsync from local `/storage/3dl_sim_data/`) |
| Repo | ⚠️ stale | `~/WildDet3D` at commit `9474d17` — **pre-cleanup** (still has `twostage` etc.); needs updating to `2dc6ad3` |
| Sim data extracted | ❌ pending | tarball not yet unpacked |
| venv | ❌ pending | `setup_venv.sh` not run |
| `testing-1` training | ❌ pending | not started |

### Finish the deployment (run `testing-1`)
```bash
ssh ubuntu@209.20.157.13
cd ~/WildDet3D
# 1. update to the cleaned code (re-rsync from dev box, or if the box has
#    GitHub access:)  git fetch origin && git reset --hard origin/visible_actual_estimation
# 2. extract the sim data
tar xzf ~/20260625_2skuwallremoval.tar.gz -C ~          # -> ~/anyware-sim/build/scenes/synth
# 3. environment (torch cu128, vis4d, MoGe, weights, stub)
bash scripts/setup_venv.sh
# 4. (optional) build the REAL vis4d_cuda_ops here (Hopper) to enable 3D-IoU eval
# 5. train, logging to W&B as run "testing-1"
export WANDB_API_KEY=...
tmux new -s testing-1
PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \
  --sim-root ~/anyware-sim/build/scenes/synth --sim-target actual \
  --epochs 6 --batch-size 4 --wandb --wandb-run-name testing-1 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga
```
⚠️ Lambda instances are **ephemeral** — disk wipes on termination. Use a
persistent filesystem (region-locked) for repo/venv/data, or re-sync each boot
(`setup_venv.sh` re-pulls the model weights itself).

---

## 9. File map

```
wilddet3d/dense/   head | targets | loss | model | decode | sim_dataset | rotation_utils
scripts/
  setup_venv.sh                  env (torch cu128, vis4d, MoGe, weights, stub)
  vis4d_cuda_ops_stub.py         eval-only CUDA-ops stub for sm_120
  train_dense_9dof.py            train JENGA on sim (+ W&B)
  visualize_dense_inference.py   draw predicted/GT 9-DoF boxes on RGB
ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt   downloaded base (frozen encoders)
```
Kept WildDet3D core (`wilddet3d/{model,inference,depth,head,ops,loss}.py`,
`configs/base/`) — the dense model reuses it for the frozen encoders. Upstream
scaffolding (`demo/`, `configs/eval/`, other-dataset `data_prep/`,
`scripts/benchmark_inference.py`) is left untouched; strip further if desired.
Removed in `2dc6ad3`: prompt-based `stage*` configs, `wilddet3d/twostage`,
`wilddet3d/multiview`, real-COCO loaders/data-prep, `scripts/lambda/`, sweep.

---

## 10. Next steps

1. **Run `testing-1`** on the H100 (sim, W&B).
2. **Richer multi-task loss** (§4.2): visible + actual + SKU-snap head/loss.
3. Train on the full sim set; more epochs.
4. Raw-folder inference mode (RGBD in, no GT) for new captures.
5. Build real CUDA ops on the H100 → 3D-IoU eval (doc gate IoU≥0.95).
6. (Optional) DDP for multi-GPU; rename `dense`→`jenga`.

(Checkpoints, `.venv/`, `pretrained/`, `data/`, `vis4d-workspace/`, `wandb/`,
`viz_out/` are git-ignored.)
