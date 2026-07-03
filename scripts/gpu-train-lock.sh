#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu-train-lock.sh <command> [args...]

Runs a full-GPU/training command while holding the exclusive shared-H100 lock.

Env vars:
  GPU_LEASE_ROOT      lease directory root (default: /run/gpu-lease, falls back
                       to ${TMPDIR:-/tmp}/gpu-lease if that cannot be created --
                       see the WARNING this script prints when that happens).
  GPU_LOCK_TIMEOUT_S   optional wait timeout in seconds for the exclusive lock
                       (default: wait forever, same as before this flag existed).
                       On timeout, prints the current holder's metadata (see
                       full-gpu.lock.meta below) and exits 75.

While the lock is held, this script writes "$LEASE_ROOT/full-gpu.lock.meta"
(pid/ppid/user/host/cwd/started_at_utc/command) and removes it on exit, purely
as a diagnostic -- the kernel flock on full-gpu.lock remains the sole
correctness mechanism (a crashed holder still releases the real lock even if
this metadata file is left behind by a hard kill; treat stale metadata as
diagnostic only, never as an additional gate).
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
  PRIMARY_LEASE_ROOT="$LEASE_ROOT"
  LEASE_ROOT="${TMPDIR:-/tmp}/gpu-lease"
  mkdir -p "$LEASE_ROOT"
  echo "gpu-train-lock: WARNING: could not use $PRIMARY_LEASE_ROOT; falling back to" \
       "$LEASE_ROOT (per-user TMPDIR). Another agent/user with a different TMPDIR" \
       "will NOT share this lock -- coordinate a shared GPU_LEASE_ROOT on this host" \
       "if concurrent GPU jobs are expected." >&2
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "gpu-train-lock: flock is required for shared-H100 lease enforcement" >&2
  exit 69
fi

LOCK_FILE="$LEASE_ROOT/full-gpu.lock"
META_FILE="$LOCK_FILE.meta"

exec {full_gpu_fd}>"$LOCK_FILE"
if [ -n "${GPU_LOCK_TIMEOUT_S:-}" ]; then
  if ! flock -w "$GPU_LOCK_TIMEOUT_S" "$full_gpu_fd"; then
    echo "gpu-train-lock: timed out after ${GPU_LOCK_TIMEOUT_S}s waiting for $LOCK_FILE" >&2
    if [ -f "$META_FILE" ]; then
      echo "gpu-train-lock: current holder metadata ($META_FILE):" >&2
      cat "$META_FILE" >&2 || true
    fi
    exit 75
  fi
else
  flock "$full_gpu_fd"
fi

# Metadata is best-effort bookkeeping written after the real flock is already
# held; a failure here must never abort the training command itself.
{
  printf 'pid=%s\n' "$$"
  printf 'ppid=%s\n' "$PPID"
  printf 'user=%s\n' "${USER:-$(id -un 2>/dev/null || echo unknown)}"
  printf 'host=%s\n' "$(hostname 2>/dev/null || echo unknown)"
  printf 'cwd=%s\n' "$PWD"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'lane_run_id=%s\n' "${GPU_LOCK_LANE_ID:-}"
  printf 'command=%s\n' "$*"
} >"$META_FILE" 2>/dev/null || true
trap 'rm -f "$META_FILE"' EXIT

"$@"
