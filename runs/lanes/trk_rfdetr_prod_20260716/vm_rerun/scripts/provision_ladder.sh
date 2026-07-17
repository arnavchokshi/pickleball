#!/usr/bin/env bash
# trk_rfdetr_prod_20260716/vm_rerun provisioning ladder. Stops at first success. 120s backoff.
set -u
ROOT="/Users/arnavchokshi/Desktop/pickleball"
SNAP="projects/gifted-electron-498923-h1/global/snapshots/pickleball-fleet-snap-20260709-w7close"
LABELS="fable-lane=trk_rfdetr_prod_20260716,fable-fleet=pickleball,owner=arnavchokshi"
STARTUP="$ROOT/scripts/fleet/lane_vm_startup.sh"
LOG="$ROOT/runs/lanes/trk_rfdetr_prod_20260716/vm_rerun/logs/provision_attempts.log"
VM="pickleball-gpu-rfdetrflip"

LADDER=(
  "h100|a3-highgpu-1g|us-central1-a"
  "h100|a3-highgpu-1g|asia-southeast1-b"
  "a100-80|a2-ultragpu-1g|us-central1-a"
  "a100-80|a2-ultragpu-1g|asia-southeast1-c"
  "a100-80|a2-ultragpu-1g|us-east4-c"
  "a100-80|a2-ultragpu-1g|europe-west4-a"
  "a100-40|a2-highgpu-1g|us-central1-a"
  "a100-40|a2-highgpu-1g|asia-southeast1-b"
  "a100-40|a2-highgpu-1g|europe-west4-b"
)

n=0
for entry in "${LADDER[@]}"; do
  n=$((n+1))
  IFS='|' read -r tier mt zone <<< "$entry"
  if [ $n -gt 1 ]; then echo "BACKOFF 120s before attempt $n"; sleep 120; fi
  echo "ATTEMPT $n tier=$tier mt=$mt zone=$zone at $(date -u +%H:%M:%SZ)"
  out=$(gcloud compute instances create "$VM" \
    --zone="$zone" --machine-type="$mt" \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --create-disk=auto-delete=yes,boot=yes,device-name="$VM",mode=rw,size=200,type=pd-balanced,source-snapshot="$SNAP" \
    --labels="$LABELS" \
    --metadata-from-file=startup-script="$STARTUP" \
    --format="value(name,zone,status)" 2>&1)
  rc=$?
  { echo "=== attempt $n $(date -u +%Y-%m-%dT%H:%M:%SZ) tier=$tier zone=$zone rc=$rc ==="; echo "$out"; } >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "SUCCESS attempt=$n tier=$tier mt=$mt zone=$zone"
    echo "$out"
    exit 0
  else
    reason=$(echo "$out" | grep -o "ZONE_RESOURCE_POOL_EXHAUSTED[A-Z_]*" | head -1)
    [ -z "$reason" ] && reason=$(echo "$out" | grep -m1 "ERROR" | cut -c1-160)
    echo "FAIL attempt=$n tier=$tier zone=$zone reason=${reason:-unknown}"
  fi
done
echo "LADDER_EXHAUSTED"
exit 1
