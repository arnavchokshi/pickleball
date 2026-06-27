#!/usr/bin/env bash
set -euo pipefail

LEASE_ROOT="${GPU_LEASE_ROOT:-/run/gpu-lease}"
mkdir -p "$LEASE_ROOT"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

flock "$LEASE_ROOT/full-gpu.lock" "$@"

