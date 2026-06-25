# JENGA — Handoff (Anyware 9-DoF Warehouse Box Detection)

> **Prompt-free, single-view 9-DoF 3D box detector** for warehouse unloading,
> built on WildDet3D's frozen SAM3 + LingBot-Depth encoders with a dense conv
> head. **Trains on Isaac/anyware-sim data only.** Branch: `visible_actual_estimation`.

This repo has been **stripped to one path**: sim data → dense JENGA detector.
All prompt-based / two-stage / multiview / real-COCO code was removed.

---

## 0. TL;DR

- **Model:** `wilddet3d/dense/` — dense CenterNet-3D head over the depth-fused
  SAM3 FPN. No text, no prompts, no per-object input.
- **Data:** sim only (`SimDenseDataset`) — each camera's `metadata.json` has per
  box `visible_`/`actual_` `extrinsic_4x4` + `geometry` (+`visible_fraction`);
  boxes are world-frame, `center_cam = inv(camera_extrinsic_4x4) @ pose`.
- **Encoders:** frozen, loaded from the downloaded stage-2 WildDet3D checkpoint;
  only the dense head (+fusion) trains (~4.8M params).
- **Verified:** 5-epoch local run on sim trains cleanly (loss ↓, ckpt saved).

```bash
PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \
  --sim-root <anyware-sim>/build/scenes/synth --sim-target actual \
  --epochs 6 --batch-size 2 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga
```

---

## 1. Environment (`scripts/setup_venv.sh`)

uv venv at `.venv` (Python 3.11). One-time:
```bash
bash scripts/setup_venv.sh
```
- **torch 2.8.0 + cu128** (README's 2.5.1/cu121 does NOT support Blackwell
  sm_120; on A100/H100 cu128 also works fine).
- vis4d 1.0.0 (+ `numpy<2`), utils3d, shapely, **MoGe** (`third_party/moge`).
- **`vis4d_cuda_ops` stub** (`scripts/vis4d_cuda_ops_stub.py`): the real ext is
  eval-only and needs CUDA-12.8 nvcc for sm_120; training doesn't use it. On
  Ampere/Hopper you can build the real one (enables 3D-IoU eval).
- Weights (public HF): stage-2 ckpt (`allenai/WildDet3D`, 4.7 GB) + LingBot
  depth backbone (`robbyant/...`, 1.3 GB → `pretrained/lingbot-depth/...`).
- `facebook/sam3` is **gated**: SAM3 is built structure-only; its weights come
  from the stage-2 ckpt. Input is **locked to 1008²** (SAM3 RoPE).

---

## 2. Data — sim (the only format)

Sim scene tree (Isaac/anyware-sim):
```
<root>/anyware-sim/build/scenes/synth/synth_<hash>/
    scene.json                         # world-frame boxes, container, skus
    {idx}_camera_pole_{bottom,left,right}/
        rgb.png, depth.png (uint16 mm), metadata.json
metadata.json:
    intrinsics {fx,fy,cx,cy,width,height}
    camera_extrinsic_4x4               # camera pose in world (cam->world)
    boxes{uuid: {visible_extrinsic_4x4, visible_geometry,
                 actual_extrinsic_4x4,  actual_geometry, visible_fraction}}
```
`SimDenseDataset(sim_root, size=1008, target="actual"|"visible", max_scenes)`
yields one sample **per camera view** (single-view): RGB-D (1008², resize+pad,
ImageNet-norm), pad-adjusted K, and camera-frame GT (`center`, box-axis `size`,
6D `rot`, projected `box2d`). `target` picks the visible (Stage-1) or actual
(full) box.

> Real captures lack a visible/actual split, so we standardised on sim (which
> renders both). To use real data you'd need the capture/sim pipeline to emit
> `visible_`+`actual_` the same way.

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
- `loss.py` — focal heatmap + L1 (offset/log-depth/log-size) + cuboid
  symmetry-aware chordal rotation.
- `sim_dataset.py` — `SimDenseDataset` + `dense_collate` (self-contained).
- `rotation_utils.py` — 6D⇄matrix, cuboid symmetry, geodesic, chordal loss.
- `decode.py` — heatmap peaks → 9-DoF boxes (inference/viz).

Depth backend + SAM3 backbone are **always frozen** (run under `no_grad`);
only the dense head (+fusion) trains.

---

## 4. Train / Visualize

```bash
# Train (sim) — W&B opt-in via --wandb (project jenga-9dof; set WANDB_API_KEY)
PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \
  --sim-root <synth_dir> --sim-target actual --epochs 6 --batch-size 2 \
  --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga \
  [--wandb]

# Visualize predicted (green) vs GT (red) 9-DoF boxes -> viz_out/*.png
PYTHONPATH=. .venv/bin/python scripts/visualize_dense_inference.py \
  --sim-root <synth_dir> --sim-target actual \
  --dense-ckpt ckpt/jenga/dense_9dof_last.pt --num-images 8 --score-thresh 0.3
```
~5 min/epoch on a 24 GB GPU at batch 2 (encoders frozen, ~9 GB used); faster on
H100/A100. Single-GPU (no DDP yet — easy to add when data scales).

---

## 5. GPU

- Frozen-head training: single GPU is plenty (~9 GB). **1× H100 80 GB** is the
  sweet spot right now (fast, cheap, room for a full fine-tune).
- Scale to **8× A100/H100** only for sim-pretrain on lots of data, parallel
  sweeps, or a full fine-tune with DDP (then add DDP to the trainer).
- Inference: ~295 ms/image bf16 on a laptop 5090 (dominated by the frozen
  encoders, not the head); much faster on datacenter GPUs / with torch.compile.

---

## 6. File map (after cleanup)

```
wilddet3d/dense/                 the model (head|targets|loss|model|decode
                                 |sim_dataset|rotation_utils)
scripts/
  setup_venv.sh                  env (torch cu128, vis4d, MoGe, weights, stub)
  vis4d_cuda_ops_stub.py         eval-only CUDA-ops stub for sm_120
  train_dense_9dof.py            train JENGA on sim (+ W&B)
  visualize_dense_inference.py   draw predicted/GT 9-DoF boxes on RGB
ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt   downloaded base (frozen encoders)
```
Kept WildDet3D core (`wilddet3d/{model,inference,depth,head,ops,loss}.py`,
`configs/base/`) — the dense model reuses it to build/load the encoders.
Upstream scaffolding (`demo/`, `configs/eval/`, other-dataset `data_prep/`,
`scripts/benchmark_inference.py`) is left untouched; strip further if desired.

Removed: prompt-based training configs (`stage*`), two-stage
(`wilddet3d/twostage`), multiview (`wilddet3d/multiview`), real-COCO loaders +
data-prep + anyware dataset config, `scripts/lambda/`.

---

## 7. Next steps

1. Train on the full sim set on the H100 (more epochs; W&B).
2. Add a raw-folder inference mode (RGBD in, no GT) for new captures.
3. Build the real CUDA ops on the H100 → 3D-IoU eval (doc gate IoU≥0.95).
4. (Optional) two-stage visible→actual on sim's real labels; DDP for multi-GPU;
   rename `dense`→`jenga`.

(Checkpoints, `.venv/`, `pretrained/`, `data/`, `vis4d-workspace/`, `wandb/`,
`viz_out/` are git-ignored.)
