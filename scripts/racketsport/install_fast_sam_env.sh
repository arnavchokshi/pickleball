#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
ENV_NAME="${FAST_SAM_ENV_NAME:-fast_sam_3d_body}"
WORKSPACE_CACHE="${WORKSPACE_CACHE:-/workspace/.cache}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_CACHE/pip}"
export TMPDIR="${TMPDIR:-$WORKSPACE_CACHE/build}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if "$CONDA_ROOT/bin/conda" tos --help >/dev/null 2>&1; then
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

if ! "$CONDA_ROOT/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_ROOT/bin/conda" create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

# Fast-SAM-3D-Body's detectron2 build expects nvcc. Keep this in a separate
# env so the running body4d pod-agent environment stays stable.
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y
conda install -c conda-forge "gcc_linux-64=13.*" "gxx_linux-64=13.*" ninja -y

export CC="${CC:-x86_64-conda-linux-gnu-cc}"
export CXX="${CXX:-x86_64-conda-linux-gnu-c++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

python -m pip install \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
  dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
  webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs \
  ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black \
  pycocotools tensorboard huggingface_hub ultralytics \
  tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnx onnxruntime-gpu nvtx \
  smplx numpy scipy tqdm pyzmq

python -m pip install git+https://github.com/microsoft/MoGe.git
python -m pip install chumpy --no-build-isolation
python -m pip install \
  'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
  --no-build-isolation --no-deps

python - <<'PY'
import importlib.util
for mod in ("torch", "detectron2", "ultralytics", "cv2", "nvtx"):
    if importlib.util.find_spec(mod) is None:
        raise SystemExit(f"missing {mod}")
print("fast_sam_3d_body env ready")
PY
