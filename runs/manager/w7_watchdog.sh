#!/bin/bash
# Wave-7 manager watchdog — READ-ONLY early-warning loop (never mutates anything).
# Exits (notifying the manager session) on FIRST anomaly, or after MAX_HOURS with a heartbeat.
# Classes: A preemption/VM-stop  B cost/overcount/age  C lane stall  D codex quota wall
#          E board regression on origin  F gcloud auth challenge  G OWNER EXPORT LANDED (positive wake)
set -u
ROOT=/Users/arnavchokshi/Desktop/pickleball
CYCLE_SECS=600
MAX_HOURS=7
STALE_CODEX_MIN=45
VM_MAX_AGE_HOURS=6
EXPECTED_VMS="pickleball-h100-w7ball"
CODEX_LANES="w7_p22checklist_20260709 w7_browserbypass_20260709 w7_ghostviewer_20260709 paddlewire_p31_20260709"

anomaly() { echo "WATCHDOG-ANOMALY[$1]: $2"; exit 2; }
now() { date +%s; }

START=$(now); CYCLES=0
while true; do
  CYCLES=$((CYCLES+1))

  # --- G: owner CVAT export landed (label-ingest trigger — positive event, wake immediately) ---
  NEWZIPS=$(find "$ROOT/cvat_upload/exports" -name "*.zip" -newer "$ROOT/runs/manager/.w7_export_epoch" 2>/dev/null | grep -v court_keypoints_metric15 | head -3)
  [ -n "$NEWZIPS" ] && anomaly G "owner export zip(s) landed — fire the label ingest: $NEWZIPS"

  # --- A/B/F: fleet state (also the auth canary) ---
  FLEET_JSON=$(gcloud compute instances list --filter=labels.fable-fleet=pickleball \
    --format="csv[no-heading](name,status,creationTimestamp)" 2>&1)
  if echo "$FLEET_JSON" | grep -qiE "reauth|credential|invalid_grant|login required"; then
    if [ "${AUTH_DOWN:-0}" = "1" ]; then
      : # known auth-dead state (typed STOP already surfaced); keep G/C/D/E canaries alive
    else
      anomaly F "gcloud auth challenge fired mid-wave: $(echo "$FLEET_JSON" | head -2)"
    fi
  fi
  if echo "$FLEET_JSON" | grep -qiE "^ERROR"; then
    anomaly F "gcloud list failed: $(echo "$FLEET_JSON" | head -2)"
  fi
  RUNNING_COUNT=$(echo "$FLEET_JSON" | grep -c ",RUNNING," || true)
  [ "$RUNNING_COUNT" -gt 4 ] && anomaly B "more than 4 fleet VMs RUNNING: $FLEET_JSON"
  for VM in $EXPECTED_VMS; do
    LINE=$(echo "$FLEET_JSON" | grep "^$VM," || true)
    [ -z "$LINE" ] && continue
    STATUS=$(echo "$LINE" | cut -d, -f2)
    if [ "$STATUS" = "STOPPED" ] || [ "$STATUS" = "TERMINATED" ]; then
      anomaly A "$VM is $STATUS (preemption/idle-stop suspect; lane not done since VM not deleted)"
    fi
    CREATED=$(echo "$LINE" | cut -d, -f3)
    CR_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${CREATED%%.*}" +%s 2>/dev/null || echo 0)
    if [ "$CR_EPOCH" -gt 0 ] && [ $(( ($(now) - CR_EPOCH) / 3600 )) -ge $VM_MAX_AGE_HOURS ]; then
      anomaly B "$VM age exceeds ${VM_MAX_AGE_HOURS}h budget (created $CREATED) — idle-burn suspect (lane wall cap is 5h)"
    fi
  done

  # --- C/D: codex lane liveness + quota wall (error-shaped TAIL lines only, w5 lesson) ---
  for LANE in $CODEX_LANES; do
    D="$ROOT/runs/lanes/$LANE"
    [ -f "$D/report.json" ] && continue
    if [ -f "$D/log.txt" ]; then
      if tail -n 8 "$D/log.txt" | grep -qiE "you've hit your usage limit|usage limit reached|429 too many requests|stream error.*rate"; then
        anomaly D "$LANE log ENDS with a quota/rate-limit wall (tail of $D/log.txt)"
      fi
      AGE_MIN=$(( ( $(now) - $(stat -f %m "$D/log.txt") ) / 60 ))
      [ "$AGE_MIN" -ge $STALE_CODEX_MIN ] && anomaly C "$LANE log silent ${AGE_MIN}min with no report.json — stall/death suspect"
    fi
  done

  # --- E: board regression on origin (the infra-PR drop class) ---
  git -C "$ROOT" fetch -q origin 2>/dev/null
  BC=$(git -C "$ROOT" show origin/main:BUILD_CHECKLIST.md 2>/dev/null)
  if [ -n "$BC" ]; then
    echo "$BC" | grep -q "WAVE-6 COMPLETE" || anomaly E "origin BUILD_CHECKLIST lost the [WAVE-6 COMPLETE] bullet"
  fi

  if [ $(( ($(now) - START) / 3600 )) -ge $MAX_HOURS ]; then
    echo "WATCHDOG-HEARTBEAT: $CYCLES clean cycles over ${MAX_HOURS}h; no anomalies."
    exit 0
  fi
  sleep $CYCLE_SECS
done
