# Running Anyware 9-DoF fine-tune on a cloud GPU box (Lambda 1–2× H100)

Plug-and-play setup for fine-tuning WildDet3D on Anyware warehouse scenes with
full 9-DoF rotation. Designed for a fresh GPU instance.

## TL;DR

```bash
# On the GPU box, inside the cloned repo:
bash scripts/lambda/setup_env.sh          # env + submodules + MoGe + weights (~13 GB)
bash scripts/lambda/prepare_data.sh 400   # S3 sync + convert (needs AWS creds)
bash scripts/lambda/train_stage4a.sh 1    # fine-tune on 1 GPU (use 2 for 2x H100)
```

## What each script does

| Script | Action |
|--------|--------|
| `setup_env.sh` | Inits submodules, clones MoGe, creates `wilddet3d` conda env, installs torch 2.5.1+cu121 / vis4d / vis4d_cuda_ops / requirements, downloads lingbot-depth encoder → `pretrained/` and the stage2 checkpoint → `ckpt/`. SAM3 backbone auto-downloads at model build. |
| `prepare_data.sh` | `aws s3 sync` scenes from `s3://anyware-perception-capture-scene-data/`, then runs the converter to produce `data/anyware_scenes/annotations/AnywareScenes_{train,val}.json`. Arg = scenes per prefix. |
| `train_stage4a.sh` | Launches `vis4d fit` on the stage4a config with bf16. Arg = number of GPUs. |

## Hardware sizing

- **1× H100 80 GB:** `SAMPLES_PER_GPU=4` at 1008² fits comfortably (matches the
  stage2 per-GPU recipe). Room for 6–8 if you want.
- **2× H100:** pass `2` to `train_stage4a.sh`; DDP gives effective batch 8, ~2× faster.
- The dataset is small (~440 scenes), so this is a short fine-tune, not a from-scratch run.

## Prerequisites

- AWS credentials with read access to the capture-scene bucket (`aws configure`
  or instance role). Data stays in the cloud — no need to copy scenes off your laptop.
- A HuggingFace token is **not** required (all weights are public).
- CUDA 12.x driver (default on Lambda images). torch 2.5.1+cu121 works on H100 (sm_90).

## Verify before training

```bash
conda run -n wilddet3d python scripts/test_anyware_pipeline.py
```

All four test groups should pass (dataset, 9-DoF coder, symmetry loss, multi-view).

## Notes

- The stage4a config (`configs/training/stage4a_anyware_9dof.py`) uses
  `canonical_rotation=False` + `symmetry="cuboid"` for full 9-DoF, and
  `use_depth_input_test=True` so sensor depth is used at inference.
- To also train the multi-view fusion module (`SceneFusion`), a Stage 4b config
  is the next step — not included here yet.
