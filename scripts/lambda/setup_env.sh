#!/usr/bin/env bash
# Set up a fresh GPU box (e.g. Lambda 1-2x H100) for WildDet3D Anyware 9-DoF
# fine-tuning. Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/lambda/setup_env.sh
#
# Assumes: conda available, CUDA 12.x driver, this repo already cloned.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[setup] repo root: $REPO_ROOT"

ENV_NAME="${ENV_NAME:-wilddet3d}"

# ----------------------------------------------------------------------
# 1. Submodules (sam3, lingbot_depth) + MoGe (training-only loss helpers)
# ----------------------------------------------------------------------
echo "[setup] syncing submodules..."
git submodule update --init --recursive

if [ ! -d third_party/moge ]; then
    echo "[setup] cloning MoGe (needed by lingbot depth training loss)..."
    git clone --depth 1 https://github.com/microsoft/moge.git third_party/moge
else
    echo "[setup] MoGe already present."
fi

# ----------------------------------------------------------------------
# 2. Conda env
# ----------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] creating conda env '${ENV_NAME}' (python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo "[setup] conda env '${ENV_NAME}' already exists."
fi

# Resolve env python without needing `conda activate` in non-interactive shell
ENV_PY="$(conda run -n "${ENV_NAME}" which python)"
echo "[setup] env python: $ENV_PY"

# ----------------------------------------------------------------------
# 3. Python deps (follows README install order)
# ----------------------------------------------------------------------
echo "[setup] installing torch 2.5.1 + cu121..."
"$ENV_PY" -m pip install --quiet torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[setup] installing vis4d..."
"$ENV_PY" -m pip install --quiet vis4d==1.0.0

echo "[setup] building vis4d_cuda_ops from source..."
"$ENV_PY" -m pip install --quiet \
    git+https://github.com/SysCV/vis4d_cuda_ops.git \
    --no-build-isolation --no-cache-dir

echo "[setup] installing remaining requirements..."
"$ENV_PY" -m pip install --quiet -r requirements.txt

# MoGe is imported via sys.path (third_party/moge); ensure its runtime deps:
"$ENV_PY" -m pip install --quiet huggingface_hub click || true

# ----------------------------------------------------------------------
# 4. Weights
#    - lingbot depth encoder -> pretrained/ (path the config expects)
#    - stage2 WildDet3D checkpoint -> ckpt/ (fine-tune starting point)
#    - SAM3 backbone auto-downloads from HF at model-build time
# ----------------------------------------------------------------------
echo "[setup] downloading lingbot-depth encoder weights..."
mkdir -p pretrained/lingbot-depth/postrain-dc-vitl14
"$ENV_PY" -m huggingface_hub.commands.huggingface_cli download \
    robbyant/lingbot-depth-postrain-dc-vitl14 model.pt \
    --local-dir pretrained/lingbot-depth/postrain-dc-vitl14 2>/dev/null \
    || "$ENV_PY" -c "from huggingface_hub import hf_hub_download; import shutil; \
p=hf_hub_download('robbyant/lingbot-depth-postrain-dc-vitl14','model.pt'); \
shutil.copy(p,'pretrained/lingbot-depth/postrain-dc-vitl14/model.pt'); \
print('lingbot model.pt ->', 'pretrained/lingbot-depth/postrain-dc-vitl14/model.pt')"

echo "[setup] downloading stage2 WildDet3D checkpoint (~4.7 GB)..."
mkdir -p ckpt
"$ENV_PY" -c "from huggingface_hub import hf_hub_download; import shutil,os; \
p=hf_hub_download('allenai/WildDet3D','wilddet3d_stage2_alldata_12e_v1.0.pt'); \
dst='ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt'; \
shutil.copy(p,dst) if not os.path.exists(dst) else None; \
print('stage2 ckpt ->', dst)"

echo ""
echo "[setup] DONE. Verify with:"
echo "    conda run -n ${ENV_NAME} python scripts/test_anyware_pipeline.py"
