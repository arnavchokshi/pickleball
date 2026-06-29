#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${MUJOCO_MJX_ENV_NAME:-racketsport_mjx}"
ENV_PATH="${MUJOCO_MJX_ENV_PATH:-/opt/conda/envs/${ENV_NAME}}"
ENV_PATH_WAS_SET="${MUJOCO_MJX_ENV_PATH+x}"
PYTHON_VERSION="${MUJOCO_MJX_PYTHON_VERSION:-3.11}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/racketsport/install_mujoco_mjx_env.sh

Installs the MuJoCo MJX/Warp environment and runs the local MJX smoke check.
Set MUJOCO_MJX_ENV_PATH to create/use a specific conda environment path.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac
if [ "$#" -gt 0 ]; then
  usage >&2
  exit 64
fi

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  if [[ -n "${ENV_PATH_WAS_SET}" ]]; then
    conda create -y -p "${ENV_PATH}" "python=${PYTHON_VERSION}" pip
  else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
  fi
fi

"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install "jax[cuda12]" "mujoco-mjx[warp]"
"${ENV_PATH}/bin/python" "${ROOT}/scripts/racketsport/smoke_mujoco_mjx.py"
