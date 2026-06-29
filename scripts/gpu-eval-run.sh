#!/usr/bin/env bash
set -euo pipefail

LEASE_ROOT="${GPU_LEASE_ROOT:-/run/gpu-lease}"
SLOTS_DIR="$LEASE_ROOT/slots"
HEARTBEAT_DIR="$LEASE_ROOT/heartbeat"
FULL_GPU_LOCK="$LEASE_ROOT/full-gpu.lock"
if ! mkdir -p "$SLOTS_DIR" "$HEARTBEAT_DIR" 2>/dev/null; then
  LEASE_ROOT="${TMPDIR:-/tmp}/gpu-lease"
  SLOTS_DIR="$LEASE_ROOT/slots"
  HEARTBEAT_DIR="$LEASE_ROOT/heartbeat"
  FULL_GPU_LOCK="$LEASE_ROOT/full-gpu.lock"
  mkdir -p "$SLOTS_DIR" "$HEARTBEAT_DIR"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

slot_files=("$SLOTS_DIR"/slot*.lock)
if [ ! -e "${slot_files[0]}" ]; then
  mkdir -p "$SLOTS_DIR"
  : > "$SLOTS_DIR/slot0.lock"
  if [ ! -f "$SLOTS_DIR/slot0.uuid" ]; then
    echo "${CUDA_VISIBLE_DEVICES:-0}" > "$SLOTS_DIR/slot0.uuid"
  fi
  slot_files=("$SLOTS_DIR"/slot*.lock)
fi
for lock_file in "${slot_files[@]}"; do
  slot_name="$(basename "$lock_file" .lock)"
  uuid_file="$SLOTS_DIR/$slot_name.uuid"
  if [ ! -f "$uuid_file" ]; then
    echo "${CUDA_VISIBLE_DEVICES:-0}" > "$uuid_file"
  fi
done

run_with_slot() (
  local lock_file="$1"
  shift
  local slot_name
  slot_name="$(basename "$lock_file" .lock)"
  local uuid_file="$SLOTS_DIR/$slot_name.uuid"
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES="$(cat "$uuid_file")"
  local heartbeat="$HEARTBEAT_DIR/$$"
  printf 'pid=%s slot=%s ts=%s\n' "$$" "$slot_name" "$(date +%s)" > "$heartbeat"
  trap 'rm -f "$heartbeat"' EXIT
  "$@"
)

if ! command -v flock >/dev/null 2>&1; then
  echo "gpu-eval-run: flock is required for shared-H100 lease enforcement" >&2
  exit 69
fi

exec {full_gpu_fd}>"$FULL_GPU_LOCK"
flock -s "$full_gpu_fd"

for lock_file in "${slot_files[@]}"; do
  exec {lock_fd}>"$lock_file"
  if flock -n "$lock_fd"; then
    run_with_slot "$lock_file" "$@"
    exit $?
  fi
done

exec {lock_fd}>"${slot_files[0]}"
flock "$lock_fd"
run_with_slot "${slot_files[0]}" "$@"
