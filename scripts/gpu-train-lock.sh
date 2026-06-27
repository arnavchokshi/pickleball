#!/usr/bin/env bash
set -euo pipefail

LEASE_ROOT="${GPU_LEASE_ROOT:-/run/gpu-lease}"
if ! mkdir -p "$LEASE_ROOT" 2>/dev/null; then
  LEASE_ROOT="${TMPDIR:-/tmp}/gpu-lease"
  mkdir -p "$LEASE_ROOT"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "gpu-train-lock: flock unavailable; running without cross-process lock" >&2
  "$@"
  exit $?
fi

flock "$LEASE_ROOT/full-gpu.lock" "$@"
