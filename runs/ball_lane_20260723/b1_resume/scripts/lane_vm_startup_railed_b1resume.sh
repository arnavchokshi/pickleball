#!/usr/bin/env bash
# ball_b1_resume WS1.1 (2026-07-23) boot-armed startup script.
# Derived from runs/lanes/ball_b1_gpu_resume_20260722/scripts/lane_vm_startup_railed_b1.sh (v2):
# same rail-first ordering, broadened idle-watchdog pattern (+ sha256sum/shasum for staging),
# preemption watcher, CUDA compute mode DEFAULT.
# Changes vs v2: rail +480 (8h hard wall — sized for reuse-x4/fresh-x3 3.6-5.3h expected with
# margin; T4 worst case at rail ~= $3.2, under the $10 per-run cap), plus a 60s heartbeat
# writer at /var/tmp/b1_resume_heartbeat.txt (builder liveness + log size).
set -euo pipefail

LOG=/var/tmp/b1resume_boot.log
exec > >(tee -a "$LOG") 2>&1

echo "=== ball_b1_resume_20260723 startup $(date -u +%FT%TZ) ==="

# 1. HARD WALL RAIL FIRST (before anything else can race it).
sudo shutdown -P +480 "lane rail: ball_b1_resume_20260723, 8h hard wall" || {
  echo "FATAL: could not arm shutdown rail" >&2
  exit 1
}
test -f /run/systemd/shutdown/scheduled || {
  echo "FATAL: shutdown rail did not register" >&2
  exit 1
}
date -u +%FT%TZ > /var/tmp/b1resume_boot_rail_armed.txt
cat /run/systemd/shutdown/scheduled >> /var/tmp/b1resume_boot_rail_armed.txt

# 2. Idle watchdog: power off after 30 minutes with none of the expected
#    lane processes running (6 x 5-min polls). Broadened pattern per the
#    2026-07-21 false-shutdown fix, plus sha256sum/shasum (staging verify).
( while sleep 300; do
    if ! pgrep -f '\.venv/bin/python|scripts/racketsport|build_pbvision_ball_sst|run_wasb_ball|ffprobe|ffmpeg|curl|yt-dlp|git|pip|rsync|scp|sha256sum|shasum|apt-get|dpkg' >/dev/null 2>&1; then
      IDLE_COUNT_FILE=/var/tmp/b1resume_idle_count
      count=$(cat "$IDLE_COUNT_FILE" 2>/dev/null || echo 0)
      count=$((count + 1))
      echo "$count" > "$IDLE_COUNT_FILE"
      if [ "$count" -ge 6 ]; then
        echo "b1resume idle watchdog: 30min no lane activity, powering off" >> "$LOG"
        sudo shutdown -P now "b1resume idle watchdog"
      fi
    else
      echo 0 > /var/tmp/b1resume_idle_count
    fi
  done ) & disown

# 3. Preemption watcher.
( while sleep 5; do
    if curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/preempted | grep -q TRUE; then
      touch /tmp/PREEMPTED; break
    fi
  done ) & disown

# 4. Heartbeat writer: every 60s record UTC time, builder pid (or NONE), and
#    build log size so a monitor can see forward progress with one cat.
( while sleep 60; do
    {
      date -u +%FT%TZ
      pgrep -f build_pbvision_ball_sst >/dev/null 2>&1 \
        && echo "builder=RUNNING pid=$(pgrep -f build_pbvision_ball_sst | head -1)" \
        || echo "builder=NOT_RUNNING"
      BUILD_LOG="$(getent passwd arnavchokshi | cut -d: -f6)/pickleball/runs/lanes/ball_data_regroup_20260722/b1_resume_20260723.log"
      [ -f "$BUILD_LOG" ] && echo "log_bytes=$(stat -c %s "$BUILD_LOG")" || echo "log_bytes=absent"
      [ -f /tmp/PREEMPTED ] && echo "PREEMPTED=TRUE" || true
    } > /var/tmp/b1_resume_heartbeat.txt 2>&1
  done ) & disown

# 5. CUDA compute mode: pipeline DEFAULT (do NOT set EXCLUSIVE_PROCESS).
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "lane_vm_startup: CUDA compute mode DEFAULT"
  sudo nvidia-smi -c DEFAULT || echo "WARN: nvidia-smi compute-mode set failed" >&2
fi

# 6. Restored-snapshot disk: repo clone, venv, WASB-SBDT, checkpoint and staged
#    media already exist. Do NOT re-clone/reset. Just record the pin found.
REAL_HOME="$(getent passwd arnavchokshi | cut -d: -f6)"
if [ -d "$REAL_HOME/pickleball/.git" ]; then
  cd "$REAL_HOME/pickleball"
  echo "ball_b1_resume_20260723: existing clone found at $(git rev-parse HEAD 2>/dev/null || echo unknown)" >> "$LOG"
else
  echo "WARN: no existing clone found on this restored disk" >> "$LOG"
fi

echo "=== ball_b1_resume_20260723 startup complete $(date -u +%FT%TZ) ==="
