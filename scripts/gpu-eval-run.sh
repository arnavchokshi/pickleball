#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu-eval-run.sh <command> [args...]

Runs a GPU eval/inference command under the shared-H100 slot lease.

Env vars:
  GPU_LEASE_ROOT      lease directory root (default: /run/gpu-lease, falls back
                       to ${TMPDIR:-/tmp}/gpu-lease if that cannot be created --
                       see the WARNING this script prints when that happens).
  GPU_LOCK_TIMEOUT_S   optional wait timeout in seconds for the initial shared
                       full-gpu.lock acquisition (default: wait forever, same
                       as before this flag existed). On timeout, prints the
                       full-gpu.lock.meta holder (written by
                       scripts/gpu-train-lock.sh while it holds the exclusive
                       lock) if present, and exits 75.
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
SLOTS_DIR="$LEASE_ROOT/slots"
HEARTBEAT_DIR="$LEASE_ROOT/heartbeat"
FULL_GPU_LOCK="$LEASE_ROOT/full-gpu.lock"
if ! mkdir -p "$SLOTS_DIR" "$HEARTBEAT_DIR" 2>/dev/null; then
  PRIMARY_LEASE_ROOT="$LEASE_ROOT"
  LEASE_ROOT="${TMPDIR:-/tmp}/gpu-lease"
  SLOTS_DIR="$LEASE_ROOT/slots"
  HEARTBEAT_DIR="$LEASE_ROOT/heartbeat"
  FULL_GPU_LOCK="$LEASE_ROOT/full-gpu.lock"
  mkdir -p "$SLOTS_DIR" "$HEARTBEAT_DIR"
  echo "gpu-eval-run: WARNING: could not use $PRIMARY_LEASE_ROOT; falling back to" \
       "$LEASE_ROOT (per-user TMPDIR). Another agent/user with a different TMPDIR" \
       "will NOT share this lock -- coordinate a shared GPU_LEASE_ROOT on this host" \
       "if concurrent GPU jobs are expected." >&2
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
  # First line's `pid=... slot=... ts=...` shape is kept byte-for-byte as
  # before for any existing reader; the extra lines below are additive
  # holder metadata (review_harden_20260702.md finding 6).
  {
    printf 'pid=%s slot=%s ts=%s\n' "$$" "$slot_name" "$(date +%s)"
    printf 'ppid=%s user=%s host=%s\n' "$PPID" "${USER:-$(id -un 2>/dev/null || echo unknown)}" "$(hostname 2>/dev/null || echo unknown)"
    printf 'cwd=%s\n' "$PWD"
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'command=%s\n' "$*"
  } > "$heartbeat"
  trap 'rm -f "$heartbeat"' EXIT
  "$@"
)

if ! command -v flock >/dev/null 2>&1; then
  echo "gpu-eval-run: flock is required for shared-H100 lease enforcement" >&2
  exit 69
fi

exec {full_gpu_fd}>"$FULL_GPU_LOCK"
if [ -n "${GPU_LOCK_TIMEOUT_S:-}" ]; then
  if ! flock -s -w "$GPU_LOCK_TIMEOUT_S" "$full_gpu_fd"; then
    echo "gpu-eval-run: timed out after ${GPU_LOCK_TIMEOUT_S}s waiting for $FULL_GPU_LOCK" \
         "(likely held exclusively by a training job via gpu-train-lock.sh)" >&2
    META_FILE="$FULL_GPU_LOCK.meta"
    if [ -f "$META_FILE" ]; then
      echo "gpu-eval-run: current holder metadata ($META_FILE):" >&2
      cat "$META_FILE" >&2 || true
    fi
    exit 75
  fi
else
  flock -s "$full_gpu_fd"
fi

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
