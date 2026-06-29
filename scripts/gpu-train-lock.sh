#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu-train-lock.sh <command> [args...]

Runs a full-GPU/training command while holding the exclusive shared-H100 lock.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac
if [ "$#" -eq 0 ]; then
  usage >&2
  exit 64
fi

LEASE_ROOT="${GPU_LEASE_ROOT:-/run/gpu-lease}"
if ! mkdir -p "$LEASE_ROOT" 2>/dev/null; then
  LEASE_ROOT="${TMPDIR:-/tmp}/gpu-lease"
  mkdir -p "$LEASE_ROOT"
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "gpu-train-lock: flock is required for shared-H100 lease enforcement" >&2
  exit 69
fi

exec {full_gpu_fd}>"$LEASE_ROOT/full-gpu.lock"
flock "$full_gpu_fd"
"$@"
