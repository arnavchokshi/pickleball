#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/arnavchokshi/coldstart_20260706/repo}"
SELF_IP="${SELF_IP:?SELF_IP is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUT="${OUT_ROOT}/${RUN_ID}"
STATUS="${OUT_ROOT}/${RUN_ID}.status"
STDOUT="${OUT_ROOT}/${RUN_ID}.stdout.json"
STDERR="${OUT_ROOT}/${RUN_ID}.stderr.log"

mkdir -p "${OUT_ROOT}"
cd "${REPO}"
touch /tmp/ns06_eff_heartbeat
printf '%s START out=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${OUT}" > "${STATUS}"
START_NS="$(date +%s%N)"
set +e
"${REPO}/.venv/bin/python3" scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --out "${OUT}" \
  --remote-host "${SELF_IP}" \
  --force \
  --json > "${STDOUT}" 2> "${STDERR}"
RC=$?
set -e
END_NS="$(date +%s%N)"
WALL_SECONDS="$(awk -v start="${START_NS}" -v end="${END_NS}" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"
touch /tmp/ns06_eff_heartbeat
printf '%s END exit=%s wall_seconds=%s out=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RC}" "${WALL_SECONDS}" "${OUT}" >> "${STATUS}"
exit "${RC}"
