#!/usr/bin/env bash
set -euo pipefail

LEASE_ROOT="${GPU_LEASE_ROOT:-/run/gpu-lease}"
SLOTS_DIR="$LEASE_ROOT/slots"
HEARTBEAT_DIR="$LEASE_ROOT/heartbeat"
mkdir -p "$SLOTS_DIR" "$HEARTBEAT_DIR"

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

run_with_slot() {
  local lock_file="$1"
  local slot_name
  slot_name="$(basename "$lock_file" .lock)"
  local uuid_file="$SLOTS_DIR/$slot_name.uuid"
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES="$(cat "$uuid_file")"
  local heartbeat="$HEARTBEAT_DIR/$$"
  printf 'pid=%s slot=%s ts=%s\n' "$$" "$slot_name" "$(date +%s)" > "$heartbeat"
  trap 'rm -f "$heartbeat"' EXIT
  "$@"
}

for lock_file in "${slot_files[@]}"; do
  if flock -n "$lock_file" bash -c 'run_with_slot "$0" "${@:1}"' "$lock_file" "$@"; then
    exit 0
  fi
done

flock "${slot_files[0]}" bash -c 'run_with_slot "$0" "${@:1}"' "${slot_files[0]}" "$@"

