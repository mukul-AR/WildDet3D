#!/usr/bin/env bash
# Sync Anyware capture scenes from S3 and convert to Omni3D-format COCO JSON
# for training/eval. Run on the GPU box after setup_env.sh.
#
# Usage:
#   bash scripts/lambda/prepare_data.sh [N_SCENES_PER_PREFIX]
#
# Requires: AWS credentials with read access to
#   s3://anyware-perception-capture-scene-data/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-wilddet3d}"
ENV_PY="$(conda run -n "${ENV_NAME}" which python)"
N_PER_PREFIX="${1:-400}"   # scenes per (site/device/date) prefix
DATA_ROOT="data/anyware_scenes"
BUCKET="s3://anyware-perception-capture-scene-data"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "[data] syncing up to ${N_PER_PREFIX} scenes per prefix from ${BUCKET}..."
# Adjust prefixes / dates as needed. These cover all 3 real sites.
for pfx in \
    inhouse/P602/20260529 inhouse/P602/20260530 \
    saddle_creek/P603/20260527 saddle_creek/P603/20260528 \
    kontoor/P607/20260511 kontoor/P607/20260512; do
    echo "[data]   prefix $pfx"
    aws s3 ls "${BUCKET}/${pfx}/" 2>/dev/null | awk '{print $2}' \
        | head -"${N_PER_PREFIX}" | while read -r scene; do
        [ -z "$scene" ] && continue
        aws s3 sync "${BUCKET}/${pfx}/${scene}" "${pfx}/${scene}" \
            --only-show-errors \
            --exclude "*image_boxes.jpg" --exclude "*scene_3d.html"
    done
done
cd "$REPO_ROOT"

N_SCENES=$(find "$DATA_ROOT" -name scene.json | wc -l)
echo "[data] synced ${N_SCENES} scenes. Converting to COCO JSON..."

"$ENV_PY" scripts/data_prep/anyware/convert_anyware_to_omni3d.py \
    --scene-roots "$DATA_ROOT" \
    --out-root "$DATA_ROOT" \
    --val-fraction 0.1

echo "[data] DONE. Annotations at ${DATA_ROOT}/annotations/"
