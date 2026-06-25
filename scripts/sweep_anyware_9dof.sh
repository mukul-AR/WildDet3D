#!/usr/bin/env bash
# Hyperparameter sweep for the full WildDet3D 9-DoF Anyware training.
# Each run is short (capped train batches) to compare learning-rate stability
# on a single 24 GB GPU. Final loss is parsed from each run's log.
set -u
cd "$(dirname "$0")/.."

CKPT=ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt
LIMIT=${LIMIT:-0.12}
EPOCHS=${EPOCHS:-1}
RESULTS=/tmp/wd3d_sweep_results.txt
: > "$RESULTS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WD3D_SKIP_DEPTH_LOSS=1
export MIXED_PRECISION=bf16
export WD3D_SHAPE=1008
export WD3D_WORKERS=2
export PYTHONPATH=.

for LR in 1e-5 2e-5 5e-5; do
  echo "=== sweep LR=$LR (epochs=$EPOCHS limit=$LIMIT) ==="
  LOG=/tmp/wd3d_sweep_lr_${LR}.log
  WD3D_LR=$LR WD3D_EPOCHS=$EPOCHS WD3D_LIMIT_TRAIN_BATCHES=$LIMIT \
    .venv/bin/vis4d fit \
      --config configs/training/stage5_anyware_9dof_train.py \
      --gpus 1 --ckpt "$CKPT" > "$LOG" 2>&1
  # last reported total loss
  LASTLOSS=$(grep -oE "loss: [0-9.]+" "$LOG" | tail -1 | awk '{print $2}')
  NAN=$(grep -c -i "nan" "$LOG")
  echo "LR=$LR  final_loss=${LASTLOSS:-NA}  nan_hits=$NAN" | tee -a "$RESULTS"
done

echo "=== SWEEP DONE ==="
cat "$RESULTS"
