#!/usr/bin/env bash
set -u
LANE_DIR=/home/arnavchokshi/coldstart_20260706/fullgame_demo_20260728
BODY_PY=/home/arnavchokshi/coldstart_20260706/body_runtime/body_venv/bin/python
KNOWN_HOSTS=/Users/arnavchokshi/Desktop/pickleball/configs/ssh/a100_known_hosts
KEY=/Users/arnavchokshi/.ssh/google_compute_engine
HOST=arnavchokshi@104.198.129.228
POLL_S=900
ITER=0

ssh_remote() {
  ssh -F /dev/null -o "UserKnownHostsFile=${KNOWN_HOSTS}" -o StrictHostKeyChecking=yes \
      -i "${KEY}" -o ConnectTimeout=15 "${HOST}" "$@"
}

REMOTE_SCRIPT=$(cat <<REMOTE
LANE_DIR=${LANE_DIR}
if pgrep -af 'scripts/racketsport/process_video.py' >/dev/null 2>&1; then ALIVE=RUNNING; else ALIVE=EXITED; fi
UTIL=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
echo "ALIVE=\$ALIVE GPU=[\$UTIL]"
${BODY_PY} - <<'PYEOF' 2>/dev/null
import json, glob, os
lane = "${LANE_DIR}"
found = False
for p in sorted(glob.glob(os.path.join(lane, "out", "**", "PIPELINE_SUMMARY.json"), recursive=True)):
    found = True
    try:
        d = json.load(open(p))
    except Exception as e:
        print("SUMMARY_PARSE_ERROR", p, e)
        continue
    stages = d.get("stages") or d.get("stage_results") or []
    print("SUMMARY", p, "overall_status=", d.get("status"))
    for s in stages:
        if isinstance(s, dict):
            print("  STAGE", s.get("stage"), s.get("status"), "wall_s=", s.get("wall_seconds"))
if not found:
    print("NO_SUMMARY_YET")
PYEOF
echo '--- stderr tail ---'
tail -n 15 \${LANE_DIR}/run_stderr.log 2>/dev/null
echo '--- stdout tail ---'
tail -n 8 \${LANE_DIR}/run_stdout.log 2>/dev/null
REMOTE
)

while true; do
  ITER=$((ITER+1))
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  REMOTE_OUT=$(ssh_remote "$REMOTE_SCRIPT" 2>&1)
  printf '[%s] iter=%s\n%s\n' "$TS" "$ITER" "$REMOTE_OUT"
  if printf '%s' "$REMOTE_OUT" | grep -qE "ALIVE=EXITED"; then
    echo "PROCESS_EXITED - ending monitor loop"
    break
  fi
  if printf '%s' "$REMOTE_OUT" | grep -qiE "Traceback|CUDA out of memory|Killed|OOM-killer|Segmentation fault"; then
    echo "POSSIBLE_FAILURE_SIGNATURE_DETECTED"
  fi
  sleep "$POLL_S"
done
