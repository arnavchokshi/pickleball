#!/usr/bin/env bash
# trk_detbench_20260716 dispatch-2 provisioning ladder (AMENDMENT 1).
# Stops at first success. 120s inter-attempt backoff. Emits one status line per attempt.
set -u
ROOT="/Users/arnavchokshi/Desktop/pickleball"
SNAP="projects/gifted-electron-498923-h1/global/snapshots/pickleball-fleet-snap-20260709-w7close"
LABELS="fable-lane=trk_detbench_20260716,fable-fleet=pickleball,owner=arnavchokshi"
STARTUP="$ROOT/scripts/fleet/lane_vm_startup.sh"
LOG="$ROOT/runs/lanes/trk_detbench_20260716/logs/provision_attempts.log"

# tier|machine_type|zone|vm_name
LADDER=(
  "h100|a3-highgpu-1g|asia-southeast1-b|pickleball-h100-detbench"
  "h100|a3-highgpu-1g|us-central1-a|pickleball-h100-detbench"
  "a100-80|a2-ultragpu-1g|asia-southeast1-c|pickleball-a100-detbench"
  "a100-80|a2-ultragpu-1g|us-central1-a|pickleball-a100-detbench"
  "a100-80|a2-ultragpu-1g|us-east4-c|pickleball-a100-detbench"
  "a100-80|a2-ultragpu-1g|europe-west4-a|pickleball-a100-detbench"
  "a100-40|a2-highgpu-1g|asia-southeast1-b|pickleball-a100-detbench"
  "a100-40|a2-highgpu-1g|us-central1-a|pickleball-a100-detbench"
  "a100-40|a2-highgpu-1g|europe-west4-b|pickleball-a100-detbench"
)

n=0
for entry in "${LADDER[@]}"; do
  n=$((n+1))
  IFS='|' read -r tier mt zone vm <<< "$entry"
  if [ $n -gt 1 ]; then
    echo "BACKOFF 120s before attempt $n"
    sleep 120
  fi
  echo "ATTEMPT $n tier=$tier mt=$mt zone=$zone vm=$vm at $(date -u +%H:%M:%SZ)"
  out=$(gcloud compute instances create "$vm" \
    --zone="$zone" \
    --machine-type="$mt" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --create-disk=auto-delete=yes,boot=yes,device-name="$vm",mode=rw,size=200,type=pd-balanced,source-snapshot="$SNAP" \
    --labels="$LABELS" \
    --metadata-from-file=startup-script="$STARTUP" \
    --format="value(name,zone,status)" 2>&1)
  rc=$?
  {
    echo "=== dispatch2 attempt $n $(date -u +%Y-%m-%dT%H:%M:%SZ) tier=$tier mt=$mt zone=$zone rc=$rc ==="
    echo "$out"
  } >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "SUCCESS attempt=$n tier=$tier mt=$mt zone=$zone vm=$vm"
    echo "$out"
    exit 0
  else
    # one-line reason for the event stream
    reason=$(echo "$out" | grep -o "ZONE_RESOURCE_POOL_EXHAUSTED[A-Z_]*" | head -1)
    [ -z "$reason" ] && reason=$(echo "$out" | grep -m1 "ERROR" | cut -c1-160)
    echo "FAIL attempt=$n tier=$tier zone=$zone reason=${reason:-unknown}"
  fi
done
echo "LADDER_EXHAUSTED all ${#LADDER[@]} attempts failed"
exit 1
