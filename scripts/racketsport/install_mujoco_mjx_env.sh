#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${MUJOCO_MJX_ENV_NAME:-racketsport_mjx}"
ENV_PATH="${MUJOCO_MJX_ENV_PATH:-/opt/conda/envs/${ENV_NAME}}"
PYTHON_VERSION="${MUJOCO_MJX_PYTHON_VERSION:-3.11}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install "jax[cuda12]" "mujoco-mjx[warp]"
"${ENV_PATH}/bin/python" "${ROOT}/scripts/racketsport/smoke_mujoco_mjx.py"
