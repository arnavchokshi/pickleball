#!/usr/bin/env bash
# Fleet restart protocol:
# 1. Refresh configs/ssh/a100_known_hosts for the VM's current external IP.
# 2. Update runs/manager/gpu_fleet.md with that current host.
# 3. Always pass the host explicitly to BODY dispatch (--remote-host in process_video.py,
#    --host in scripts/racketsport/remote_body_dispatch.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KNOWN_HOSTS="$ROOT/configs/ssh/a100_known_hosts"
HOST=""
ALIAS=""
SSH_KEY="${HOME}/.ssh/google_compute_engine"
SSH_USER="arnavchokshi"
KEYSCAN_FILE=""
SKIP_CONNECTIVITY=0

usage() {
  cat >&2 <<'EOF'
usage: scripts/fleet/refresh_remote_host.sh --host <ip> [--alias <name>] [options]

Options:
  --known-hosts <path>          known_hosts file to update
  --ssh-key <path>              fleet SSH key (default: ~/.ssh/google_compute_engine)
  --ssh-user <user>             fleet SSH user (default: arnavchokshi)
  --keyscan-file <path>         offline fixture for tests; skips live ssh-keyscan
  --skip-connectivity-check     update file only; connectivity check runs at fleet start
EOF
}

while (($#)); do
  case "$1" in
    --host)
      HOST="${2:-}"; shift 2 ;;
    --alias)
      ALIAS="${2:-}"; shift 2 ;;
    --known-hosts)
      KNOWN_HOSTS="${2:-}"; shift 2 ;;
    --ssh-key)
      SSH_KEY="${2:-}"; shift 2 ;;
    --ssh-user)
      SSH_USER="${2:-}"; shift 2 ;;
    --keyscan-file)
      KEYSCAN_FILE="${2:-}"; shift 2 ;;
    --skip-connectivity-check)
      SKIP_CONNECTIVITY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "--host <ip> is required; read the current value from runs/manager/gpu_fleet.md" >&2
  exit 2
fi

mkdir -p "$(dirname "$KNOWN_HOSTS")"
touch "$KNOWN_HOSTS"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

scan_raw="$tmp_dir/keyscan.raw"
scan_normalized="$tmp_dir/keyscan.normalized"
filtered="$tmp_dir/known_hosts.filtered"

if [[ -n "$KEYSCAN_FILE" ]]; then
  cp "$KEYSCAN_FILE" "$scan_raw"
else
  ssh-keyscan -T 10 "$HOST" > "$scan_raw"
fi

awk -v host="$HOST" -v alias="$ALIAS" '
  /^[[:space:]]*$/ { next }
  /^[#]/ { next }
  NF >= 3 {
    hosts = host
    if (alias != "") {
      hosts = host "," alias
    }
    print hosts " " $2 " " $3
  }
' "$scan_raw" > "$scan_normalized"

if [[ ! -s "$scan_normalized" ]]; then
  echo "FAIL refresh_remote_host host=$HOST reason=no_keyscan_entries" >&2
  exit 1
fi

awk -v host="$HOST" -v alias="$ALIAS" '
  /^[[:space:]]*$/ { print; next }
  /^[#]/ { print; next }
  {
    n = split($1, names, ",")
    for (i = 1; i <= n; i++) {
      if (names[i] == host || (alias != "" && names[i] == alias)) {
        next
      }
    }
    print
  }
' "$KNOWN_HOSTS" > "$filtered"

cat "$filtered" "$scan_normalized" > "$KNOWN_HOSTS"

if [[ "$SKIP_CONNECTIVITY" -eq 1 ]]; then
  echo "PASS refresh_remote_host host=$HOST alias=${ALIAS:-} known_hosts=$KNOWN_HOSTS connectivity_check=skipped_runs_at_fleet_start"
  exit 0
fi

if ssh \
  -i "$SSH_KEY" \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$KNOWN_HOSTS" \
  "${SSH_USER}@${HOST}" true; then
  echo "PASS refresh_remote_host host=$HOST alias=${ALIAS:-} known_hosts=$KNOWN_HOSTS connectivity_check=passed"
else
  echo "FAIL refresh_remote_host host=$HOST alias=${ALIAS:-} known_hosts=$KNOWN_HOSTS connectivity_check=failed" >&2
  exit 1
fi
