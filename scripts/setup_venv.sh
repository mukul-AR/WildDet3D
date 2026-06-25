#!/usr/bin/env bash
# Reproducible environment for the FULL WildDet3D (vis4d) training on an
# RTX 5090 / Blackwell (sm_120) GPU. Builds a uv venv at .venv.
#
#   bash scripts/setup_venv.sh
#
# Notes / why each step:
#  - torch 2.8.0+cu128: the README pins 2.5.1+cu121 which does NOT support
#    Blackwell sm_120; cu128 + torch>=2.7 is required.
#  - vis4d 1.0.0 only needs torch>=2.0 (its 2.5.1 pin is soft) but numpy<2.
#  - vis4d_cuda_ops (SysCV) is a CUDA extension needing CUDA 12.8 nvcc to build
#    for sm_120; it is eval-only (3D-IoU / rotated NMS / deformable attn) and
#    NOT used by WildDet3D training, so we install a lightweight stub instead.
#  - MoGe (third_party/moge) provides the depth-backend training losses.
#  - SAM3 weights come from facebook/sam3 (GATED); we never download them —
#    the stage-2 checkpoint already contains all encoder weights.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  "vis4d==1.0.0" "numpy==1.26.4" shapely \
  einops timm transformers huggingface_hub \
  "ftfy==6.1.1" regex "iopath>=0.1.10" "setuptools<80" \
  opencv-python matplotlib pycocotools pyquaternion scipy terminaltables \
  ml_collections tqdm \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"

# MoGe (depth-backend training losses), added to sys.path by wilddet3d/__init__.
[ -d third_party/moge ] || git clone --depth 1 https://github.com/microsoft/moge.git third_party/moge

# Stub the eval-only CUDA extension so the vis4d CLI imports on sm_120.
cp scripts/vis4d_cuda_ops_stub.py \
   "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/vis4d_cuda_ops.py"

# Pretrained weights (public): stage-2 checkpoint + LingBot depth backbone.
.venv/bin/python - <<'PY'
import os
from huggingface_hub import hf_hub_download
hf_hub_download("allenai/WildDet3D", "wilddet3d_stage2_alldata_12e_v1.0.pt", local_dir="ckpt")
os.makedirs("pretrained/lingbot-depth/postrain-dc-vitl14", exist_ok=True)
hf_hub_download("robbyant/lingbot-depth-pretrain-vitl-14-v0.5", "model.pt",
                local_dir="pretrained/lingbot-depth/postrain-dc-vitl14")
print("weights downloaded")
PY

echo "Setup complete. Train (sim data -> dense JENGA detector) with:"
echo "  PYTHONPATH=. .venv/bin/python scripts/train_dense_9dof.py \\"
echo "    --sim-root <anyware-sim>/build/scenes/synth --epochs 6 \\"
echo "    --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt"
