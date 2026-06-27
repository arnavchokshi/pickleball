#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAST_SAM_ROOT="${FAST_SAM_ROOT:-/opt/fast-sam-3d-body}"
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
ENV_NAME="${FAST_SAM_ENV_NAME:-fast_sam_3d_body}"
OUT_DIR="${1:-$ROOT/runs/phase0/fast_sam_profile}"
IMAGE_PATH="${FAST_SAM_IMAGE_PATH:-$FAST_SAM_ROOT/notebook/images/dancing.jpg}"
TIMEOUT_SECONDS="${FAST_SAM_TIMEOUT_SECONDS:-900}"
WARMUP_RUNS="${FAST_SAM_WARMUP_RUNS:-1}"
BENCHMARK_RUNS="${FAST_SAM_BENCHMARK_RUNS:-5}"

CHECKPOINT_SRC="${FAST_SAM_CHECKPOINT:-/workspace/checkpoints/body4d/sam-3d-body-dinov3/model.ckpt}"
MHR_SRC="${FAST_SAM_MHR_MODEL:-/workspace/checkpoints/body4d/sam-3d-body-dinov3/assets/mhr_model.pt}"
CHECKPOINT_DIR="$FAST_SAM_ROOT/checkpoints/sam-3d-body-dinov3"

if [ ! -d "$FAST_SAM_ROOT" ]; then
  echo "missing Fast-SAM-3D-Body repo at $FAST_SAM_ROOT" >&2
  exit 66
fi

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  echo "missing conda setup at $CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 66
fi

if [ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]; then
  echo "missing conda env $ENV_NAME; run scripts/racketsport/install_fast_sam_env.sh first" >&2
  exit 66
fi

mkdir -p "$OUT_DIR" "$CHECKPOINT_DIR/assets"

if [ -f "$CHECKPOINT_SRC" ]; then
  ln -sfn "$CHECKPOINT_SRC" "$CHECKPOINT_DIR/model.ckpt"
else
  echo "missing Fast-SAM checkpoint at $CHECKPOINT_SRC" >&2
  exit 66
fi

if [ -f "$MHR_SRC" ]; then
  ln -sfn "$MHR_SRC" "$CHECKPOINT_DIR/assets/mhr_model.pt"
else
  echo "missing SAM-3D-Body MHR model at $MHR_SRC" >&2
  exit 66
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$FAST_SAM_ROOT"

runner=("$ROOT/scripts/gpu-eval-run.sh")
if command -v timeout >/dev/null 2>&1; then
  runner+=(timeout "$TIMEOUT_SECONDS")
fi

"${runner[@]}" python profile_nsight.py \
  --image_path "$IMAGE_PATH" \
  --output_dir "$OUT_DIR" \
  --detector_model yolo11n.pt \
  --warmup "$WARMUP_RUNS" \
  --runs "$BENCHMARK_RUNS"
