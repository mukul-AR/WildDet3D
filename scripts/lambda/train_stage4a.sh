#!/usr/bin/env bash
# Launch Anyware 9-DoF single-view fine-tune on 1-2 H100 GPUs.
#
# Usage:
#   bash scripts/lambda/train_stage4a.sh [NUM_GPUS]
#
# Effective batch = NUM_GPUS * SAMPLES_PER_GPU (4). H100 80GB fits batch 4/GPU
# at 1008x1008 comfortably. bf16 mixed precision is enabled for speed/memory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-wilddet3d}"
NUM_GPUS="${1:-1}"
CKPT="${CKPT:-ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt}"
CONFIG="configs/training/stage4a_anyware_9dof.py"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: fine-tune checkpoint not found: $CKPT"
    echo "Run scripts/lambda/setup_env.sh first."
    exit 1
fi

echo "[train] config=$CONFIG gpus=$NUM_GPUS ckpt=$CKPT precision=bf16"

# bf16 is read by configs/base/pl.py via MIXED_PRECISION env var.
MIXED_PRECISION=bf16 \
conda run -n "${ENV_NAME}" --no-capture-output \
    vis4d fit --config "$CONFIG" --gpus "$NUM_GPUS" --ckpt "$CKPT"
