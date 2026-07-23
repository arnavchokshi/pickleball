# E-v2 Sonnet GPU run plan

This is the verbatim execution leg for the one registered `EV2_RECIPE` arm. It starts only after
the Track-D manager's joint review commit is on `main` and includes this lane's ignored `runs/`
artifacts with `git add -f`. Do not create a branch, change a value, reuse a probe checkpoint, or
invoke the owner judge outside section 7.

Registered machine: a fresh `pickleball-gpu-ev2` in `us-central1-f`, exact
`a2-highgpu-1g` Spot shape (one NVIDIA A100-SXM4-40GB), exact
`pickleball-cache-image-20260722` image from family `pickleball-cache`, a 200 GB
`pd-balanced` auto-delete boot disk, and the shared zonal
`pickleball-cache-data-usc1f` disk attached read-only as device `cache`. The cache image contains
repo `e1e2184df` plus the upgraded `.venv` (`torch 2.13.0+cu130`,
`torchvision 0.28.0+cu130`, CUDA 13.0); the VM fetches and checks out the exact dispatch
`RUN_COMMIT` in that baked repo and does not clone or build another environment. The prior E1 VM
`pickleball-gpu-abc` was deleted on 2026-07-22 and is not reused.  
GPU output root: `runs/lanes/trackD_ev2_gpu_20260722/`.  
Final state remains `VERIFIED=0` until a later protected promotion gate.

## 0. Fail-closed training-input gate is a dispatch prerequisite

Before any training input is read, the provisioned VM must run
`scripts/racketsport/verify_training_inputs.py` from the pinned `RUN_COMMIT` and preserve its
successful machine-readable proof as
`runs/lanes/trackD_ev2_gpu_20260722/gate_proof.json`. `RUN_COMMIT` must contain the separately
reviewed Track-E cache-safety revision from `trackE_cache_safety_20260723`. The serialized
integration owner fills `REVIEWED_SAFETY_REVISION_SHA` with that exact reviewed commit SHA at
dispatch time; this plan deliberately does not guess it now. The controller proves that revision
is an ancestor of `RUN_COMMIT`, the VM proves the script exists at `RUN_COMMIT`, and sections 3-7
remain forbidden unless the gate exits zero and the proof says `status=PASS`. Missing, stale, malformed,
or failed proof is a registered terminal failure, not permission to bypass the gate or retry under
this registration. In particular, dispatch is forbidden until the ledger at `RUN_COMMIT`
queue-authorizes every declared pre-existing input and the in-run Stage-P/Stage-F generated-input
asset contracts used by the refreshed proofs; current local ledger state is not an override. The
executable invocation is in section 1 after integrity-only staging and before the setup trap is
disarmed.

## 1. Create the exact fresh VM, arm rails, bootstrap, and stage inputs

Run on the control workstation from the repository root:

```bash
set -euo pipefail
cd /Users/arnavchokshi/Desktop/pickleball
test "$(git branch --show-current)" = main

INSTANCE=pickleball-gpu-ev2
ZONE=us-central1-f
PROJECT=gifted-electron-498923-h1
CACHE_IMAGE=pickleball-cache-image-20260722
CACHE_DISK=pickleball-cache-data-usc1f
REVIEWED_SAFETY_REVISION_SHA=${REVIEWED_SAFETY_REVISION_SHA:?integration owner must fill the exact reviewed Track-E safety revision SHA at dispatch}
test "$(printf '%s' "$REVIEWED_SAFETY_REVISION_SHA" | tr -cd '0-9a-f' | wc -c | tr -d ' ')" = 40
RUN_COMMIT=$(git rev-parse main)
test "$(printf '%s' "$RUN_COMMIT" | tr -cd '0-9a-f' | wc -c | tr -d ' ')" = 40
git fetch origin main
git merge-base --is-ancestor "$RUN_COMMIT" origin/main
git merge-base --is-ancestor "$REVIEWED_SAFETY_REVISION_SHA" "$RUN_COMMIT"
git cat-file -e "$RUN_COMMIT:scripts/racketsport/verify_training_inputs.py"
for lane_artifact in \
  REGISTRATION.md VM_RUN_PLAN.md INPUT_LOCK.json RATE_MEDIA_LOCK.json \
  CODE_SHA256SUMS CONTROL_RESULTS.json REPORT.md spec.md report.json \
  CROSS_TRACK_ASSUMPTIONS.md REPAIR_BRIEF_R1.md REPAIR_BRIEF_R2.md RESUME.md \
  report_repair1.json report_repair2.json
do
  git cat-file -e \
    "$RUN_COMMIT:runs/lanes/trackD_ev2_design_20260722/$lane_artifact"
  git diff --exit-code "$RUN_COMMIT" -- \
    "runs/lanes/trackD_ev2_design_20260722/$lane_artifact"
done

# BEGIN EV2_F5_FROZEN_COMMIT_CODE_PREFLIGHT
verify_frozen_commit_code_bytes() {
  frozen_commit="$1"
  sums_blob_path="$2"
  while read -r expected_sha code_path; do
    test -n "$expected_sha"
    test -n "$code_path"
    git cat-file -e "$frozen_commit:$code_path"
    actual_sha=$(git show "$frozen_commit:$code_path" | shasum -a 256 | awk '{print $1}')
    test "$actual_sha" = "$expected_sha"
  done < <(git show "$frozen_commit:$sums_blob_path")
}
# Prove every reviewed runtime byte is a blob in RUN_COMMIT. Both the digest
# list and every hashed target are read with git show from that commit; no
# working-tree byte can satisfy this boundary.
verify_frozen_commit_code_bytes \
  "$RUN_COMMIT" \
  runs/lanes/trackD_ev2_design_20260722/CODE_SHA256SUMS
# END EV2_F5_FROZEN_COMMIT_CODE_PREFLIGHT

# Reject stale inputs or code before creating a billable resource. Keep the
# equivalent VM-side checks later to prove transfer integrity.
shasum -a 256 -c <<'SHA256'
f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json
9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5  runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json
84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b  runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json
f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082  runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt
81df518a85ce891b4b2da1b494f8b123979367d9b70432b4c9af850f4a88792c  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/agreement_decisions.jsonl
SHA256
shasum -a 256 -c runs/lanes/trackD_ev2_design_20260722/CODE_SHA256SUMS

# Prove the exact 38 train + 2 validation media inventory before provisioning.
RATE_MEDIA_LOCK=runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json
RATE_MEDIA_LOCK_SHA256=79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee
.venv/bin/python - "$RATE_MEDIA_LOCK" "$RATE_MEDIA_LOCK_SHA256" <<'PY'
import sys
from pathlib import Path
from scripts.racketsport.finetune_event_head import validate_registered_rate_media_inventory
result = validate_registered_rate_media_inventory(
    Path('data/online_harvest_20260706/rallies'), Path(sys.argv[1]), sys.argv[2]
)
assert len(result['train']) == 38 and len(result['validation']) == 2
assert result['train_total_frames'] == 57025
assert result['train_total_duration_s'] == 2063.1827083333333
PY

ACTIVE_GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)
test -n "$ACTIVE_GCLOUD_ACCOUNT"
if gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  echo "$INSTANCE already exists; this registration requires one fresh VM" >&2
  exit 64
fi
if gcloud compute disks describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  echo "orphan disk $INSTANCE already exists; this registration requires fresh storage" >&2
  exit 64
fi
gcloud compute images describe "$CACHE_IMAGE" --project="$PROJECT" --format=json \
  > /tmp/trackD_ev2_cache_image.json
test "$(jq -r '.status' /tmp/trackD_ev2_cache_image.json)" = READY
test "$(jq -r '.family' /tmp/trackD_ev2_cache_image.json)" = pickleball-cache
gcloud compute disks describe "$CACHE_DISK" --project="$PROJECT" --zone="$ZONE" --format=json \
  > /tmp/trackD_ev2_cache_disk.json
test "$(jq -r '.status' /tmp/trackD_ev2_cache_disk.json)" = READY

# Fail closed on the registered usc1f shape and quota. A stockout is a registered failure that
# requires a new registration; never fall back to another zone.
gcloud compute machine-types describe a2-highgpu-1g \
  --project="$PROJECT" --zone="$ZONE" >/tmp/trackD_ev2_machine_type.json
gcloud compute accelerator-types describe nvidia-tesla-a100 \
  --project="$PROJECT" --zone="$ZONE" >/tmp/trackD_ev2_accelerator_type.json
gcloud compute regions describe us-central1 --project="$PROJECT" --format=json \
  > /tmp/trackD_ev2_region_quota.json
.venv/bin/python - /tmp/trackD_ev2_region_quota.json <<'PY'
import json
import sys
from pathlib import Path

region = json.loads(Path(sys.argv[1]).read_text())
quotas = {row['metric']: row for row in region['quotas']}
a100 = [row for name, row in quotas.items() if 'A100' in name]
assert a100, sorted(quotas)
assert max(float(row['limit']) - float(row['usage']) for row in a100) >= 1, a100
preemptible = quotas.get('PREEMPTIBLE_CPUS')
assert preemptible is not None, sorted(quotas)
assert float(preemptible['limit']) - float(preemptible['usage']) >= 12, preemptible
PY

# Mechanically obtain a fresh public-list-price proof from Google's authoritative
# Cloud Billing Catalog API. No operator-entered rate or source is accepted.
# The selected evidence is registered to us-central1-f and covers A2 Spot vCPU,
# RAM, and A100 GPU, plus the disposable 250 GiB boot disk and one in-use external
# IPv4. The already-existing shared cache disk/image are not charged to this run.
# Staleness is measured from
# this live retrieval (maximum 15 minutes before use), not from a SKU's
# effectiveTime: an unchanged current price can legitimately have an older
# effective date. The authoritative list API itself supplies the latest price.
PRICE_API='https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?currencyCode=USD&pageSize=5000'
PRICE_NDJSON=/tmp/trackD_ev2_compute_skus.ndjson
PRICE_CATALOG=/tmp/trackD_ev2_compute_skus.json
PRICE_PROOF=/tmp/trackD_ev2_price_proof.json
: > "$PRICE_NDJSON"
PAGE_TOKEN=
while :; do
  PAGE=/tmp/trackD_ev2_compute_skus_page.json
  PAGE_URL="$PRICE_API"
  if test -n "$PAGE_TOKEN"; then PAGE_URL="$PAGE_URL&pageToken=$PAGE_TOKEN"; fi
  curl --fail --silent --show-error --location \
    -H 'Cache-Control: no-cache' \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    "$PAGE_URL" > "$PAGE"
  jq -c '.skus[]' "$PAGE" >> "$PRICE_NDJSON"
  PAGE_TOKEN=$(jq -r '.nextPageToken // empty' "$PAGE")
  test -n "$PAGE_TOKEN" || break
done
jq -s --arg retrieved_utc "$(date -u +%FT%TZ)" --arg source "$PRICE_API" \
  '{retrieved_utc:$retrieved_utc,source:$source,skus:.}' "$PRICE_NDJSON" > "$PRICE_CATALOG"

.venv/bin/python - "$PRICE_CATALOG" "$PRICE_PROOF" <<'PY'
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text())
now = datetime.now(timezone.utc)
retrieved = datetime.fromisoformat(catalog['retrieved_utc'].replace('Z', '+00:00'))
assert 0 <= (now - retrieved).total_seconds() <= 900

def select(name, *, sku_id, description, usage_type, allow_global=False):
    matches = []
    for sku in catalog['skus']:
        category = sku.get('category', {})
        if (
            sku.get('skuId') == sku_id
            and str(sku.get('description', '')) == description
            and category.get('usageType') == usage_type
            and (
                'us-central1' in sku.get('serviceRegions', [])
                or (
                    allow_global
                    and (
                        not sku.get('serviceRegions')
                        or 'global' in sku.get('serviceRegions', [])
                    )
                )
            )
        ):
            matches.append(sku)
    assert len(matches) == 1, (name, [(x.get('skuId'), x.get('description')) for x in matches])
    sku = matches[0]
    assert sku.get('pricingInfo'), (name, 'missing pricingInfo')
    pricing = max(sku['pricingInfo'], key=lambda row: row['effectiveTime'])
    effective = datetime.fromisoformat(pricing['effectiveTime'].replace('Z', '+00:00'))
    assert effective <= now, (name, pricing['effectiveTime'], 'future price')
    expression = pricing['pricingExpression']
    rates = expression['tieredRates']
    assert len(rates) == 1 and float(rates[0].get('startUsageAmount', 0)) == 0
    money = rates[0]['unitPrice']
    assert money['currencyCode'] == 'USD'
    unit_price = float(money.get('units', 0)) + int(money.get('nanos', 0)) / 1e9
    assert math.isfinite(unit_price) and unit_price >= 0
    return {
        'component': name,
        'sku_id': sku['skuId'],
        'description': sku['description'],
        'category': sku.get('category', {}),
        'usage_type': usage_type,
        'service_regions': sku['serviceRegions'],
        'effective_time': pricing['effectiveTime'],
        'usage_unit': expression['usageUnit'],
        'unit_price_usd': unit_price,
    }

components = [
    select('a2_spot_vcpu', sku_id='3178-715E-CFB6', description='Spot Preemptible A2 Instance Core running in Americas', usage_type='Preemptible'),
    select('a2_spot_ram_gib', sku_id='65A3-16DB-D57A', description='Spot Preemptible A2 Instance Ram running in Americas', usage_type='Preemptible'),
    select('a100_spot_gpu', sku_id='39D4-516A-0317', description='Nvidia Tesla A100 GPU attached to Spot Preemptible VMs running in Americas', usage_type='Preemptible'),
    select('balanced_pd_gib_month', sku_id='6AE1-525F-8B80', description='Balanced PD Capacity', usage_type='OnDemand'),
    select('external_ipv4_hour', sku_id='4AF8-7C1F-39C4', description='External IP Charge on a Spot Preemptible VM', usage_type='OnDemand', allow_global=True),
]
by_name = {row['component']: row for row in components}
assert by_name['a2_spot_vcpu']['usage_unit'] in {'h', 'hour'}
assert by_name['a2_spot_ram_gib']['usage_unit'] in {'GiBy.h', 'gibibyte hour'}
assert by_name['a100_spot_gpu']['usage_unit'] in {'h', 'hour'}
assert by_name['balanced_pd_gib_month']['usage_unit'] in {'GiBy.mo', 'gibibyte month'}
assert by_name['external_ipv4_hour']['usage_unit'] in {'h', 'hour'}

component_hourly = {
    'a2_spot_vcpu_12': by_name['a2_spot_vcpu']['unit_price_usd'] * 12,
    'a2_spot_ram_85_gib': by_name['a2_spot_ram_gib']['unit_price_usd'] * 85,
    'a100_spot_gpu_1': by_name['a100_spot_gpu']['unit_price_usd'],
    'balanced_pd_250_gib': by_name['balanced_pd_gib_month']['unit_price_usd'] * 250 / 730,
    'external_ipv4_1': by_name['external_ipv4_hour']['unit_price_usd'],
}
rate = sum(component_hourly.values())
assert math.isfinite(rate) and 0 < rate <= 3.30, (rate, component_hourly)
proof = {
    'artifact_type': 'event_ev2_authoritative_price_proof',
    'verified': False,
    'retrieved_utc': catalog['retrieved_utc'],
    'source': catalog['source'],
    'source_authority': 'Google Cloud Billing Catalog API v1 latest public USD pricing',
    'region': 'us-central1',
    'zone': 'us-central1-f',
    'machine_type': 'a2-highgpu-1g',
    'machine_shape': {'vcpu': 12, 'ram_gib': 85, 'a100_40gb_gpu': 1},
    'disk': {'type': 'pd-balanced', 'gib': 250, 'monthly_hours': 730},
    'external_ipv4_count': 1,
    'components': components,
    'component_hourly_usd': component_hourly,
    'all_in_instance_rate_usd_per_hour': rate,
}
Path(sys.argv[2]).write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
print(json.dumps(proof, sort_keys=True))
PY
ALL_IN_INSTANCE_RATE_USD_PER_HOUR=$(jq -r '.all_in_instance_rate_usd_per_hour' "$PRICE_PROOF")
A100_RATE_SOURCE=$(jq -r '.source' "$PRICE_PROOF")

cat > /tmp/trackD_ev2_startup.sh <<'STARTUP'
#!/usr/bin/env bash
set -euo pipefail
LOG=/var/tmp/trackD_ev2_startup.log
exec > >(tee -a "$LOG") 2>&1

# This boot-time rail is armed before package setup, media staging, or training.
sudo shutdown -P +350 'trackD E-v2 350-minute all-in hard wall'
test -f /run/systemd/shutdown/scheduled
date -u +%FT%TZ > /var/tmp/trackD_ev2_boot_rail_armed.txt
cat /run/systemd/shutdown/scheduled >> /var/tmp/trackD_ev2_boot_rail_armed.txt

( while sleep 5; do
    if curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/preempted | grep -q TRUE; then
      touch /tmp/PREEMPTED
      break
    fi
  done ) & disown

# Fail closed after 45 minutes without setup/training/eval/transfer activity.
( while sleep 300; do
    if ! pgrep -f 'apt-get|pip|git|curl|scp|tar|train_event_head\.py|finetune_event_head\.py|eval_event_head\.py' >/dev/null 2>&1; then
      count=$(cat /var/tmp/trackD_ev2_idle_count 2>/dev/null || echo 0)
      count=$((count + 1))
      echo "$count" > /var/tmp/trackD_ev2_idle_count
      if test "$count" -ge 9; then
        sudo shutdown -P now 'trackD E-v2 idle watchdog'
      fi
    else
      echo 0 > /var/tmp/trackD_ev2_idle_count
    fi
  done ) & disown

# The registered pipeline-safe mode is DEFAULT. Setup later rechecks it live.
for _ in $(seq 1 120); do
  if nvidia-smi >/dev/null 2>&1; then
    sudo nvidia-smi -c DEFAULT
    date -u +%FT%TZ > /var/tmp/trackD_ev2_startup_complete.txt
    exit 0
  fi
  sleep 5
done
echo 'GPU did not become available within ten minutes' >&2
exit 1
STARTUP
chmod 755 /tmp/trackD_ev2_startup.sh
bash -n /tmp/trackD_ev2_startup.sh

# BEGIN EV2_F6_CREATE_AND_SETUP_TEARDOWN
EV2_STATE_DIR=${EV2_STATE_DIR:-/tmp/trackD_ev2_controller_state}
if test -e "$EV2_STATE_DIR"; then
  echo "stale E-v2 controller state exists: $EV2_STATE_DIR" >&2
  exit 64
fi
mkdir -m 700 "$EV2_STATE_DIR"
INSTANCE_ID_FILE="$EV2_STATE_DIR/instance_id"
INSTANCE_JSON_FILE="$EV2_STATE_DIR/trackD_ev2_instance.json"
DISK_JSON_FILE="$EV2_STATE_DIR/trackD_ev2_disk.json"
WATCHDOG_PID_FILE="$EV2_STATE_DIR/watchdog.pid"
TEARDOWN_CONFIRMED_FILE="$EV2_STATE_DIR/teardown_confirmed"
CONTROLLER_DELETE_ON_EXIT=1
INSTANCE_CREATE_ATTEMPTED=0

controller_instance_is_original() {
  current_id=$(gcloud compute instances describe "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" --format='value(id)' 2>/dev/null || true)
  if ! test -f "$INSTANCE_ID_FILE"; then
    # The name was proven absent immediately before this one create attempt. If
    # capture failed, any now-present same-name resource belongs to that attempt.
    test "$INSTANCE_CREATE_ATTEMPTED" = 1
    return
  fi
  expected_id=$(tr -d '\r\n' < "$INSTANCE_ID_FILE")
  test -n "$expected_id"
  test -z "$current_id" || test "$current_id" = "$expected_id"
}
cancel_original_controller_watchdog() {
  if test -f "$WATCHDOG_PID_FILE"; then
    watchdog_pid=$(tr -d '\r\n' < "$WATCHDOG_PID_FILE")
    if kill -0 "$watchdog_pid" 2>/dev/null; then
      watchdog_command=$(ps -p "$watchdog_pid" -o command=)
      expected_id=$(tr -d '\r\n' < "$INSTANCE_ID_FILE")
      case "$watchdog_command" in
        *pickleball-gpu-ev2*us-central1-f*"$expected_id"*) kill "$watchdog_pid"; wait "$watchdog_pid" 2>/dev/null || true ;;
        *) echo 'identity-bound watchdog PID mismatch; refusing to signal' >&2; return 43 ;;
      esac
    fi
  fi
}
controller_confirm_deleted() {
  if gcloud compute instances describe "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
    echo 'instance still exists after setup-failure delete' >&2
    return 44
  fi
  if gcloud compute disks describe "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
    echo 'boot disk still exists after setup-failure delete' >&2
    return 44
  fi
  date -u +%FT%TZ > "$TEARDOWN_CONFIRMED_FILE"
}
controller_delete_on_failure() {
  if test "$CONTROLLER_DELETE_ON_EXIT" = 1 && test "$INSTANCE_CREATE_ATTEMPTED" = 1; then
    controller_instance_is_original || {
      echo 'refusing teardown: same-name instance has a different provider ID' >&2
      return 43
    }
    # Tolerantly recover any setup-era evidence that exists. The content-blind
    # spend bootstrap is the fallback when the guest never created SPEND_GUARD.
    mkdir -p runs/lanes/trackD_ev2_design_20260722/vm_pull/setup_failure
    cp /tmp/trackD_ev2_spend_bootstrap.json \
      runs/lanes/trackD_ev2_design_20260722/vm_pull/setup_failure/ 2>/dev/null || true
    cp /tmp/trackD_ev2_price_proof.json \
      runs/lanes/trackD_ev2_design_20260722/vm_pull/setup_failure/ 2>/dev/null || true
    gcloud compute scp --recurse \
      "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/trackD_ev2_gpu_20260722" \
      runs/lanes/trackD_ev2_design_20260722/vm_pull/setup_failure/ \
      --project="$PROJECT" --zone="$ZONE" 2>/dev/null || true
    cancel_original_controller_watchdog || true
    # The cache disk and cache image are shared fleet resources. Detach/leave them
    # intact; delete only this VM and its auto-delete boot disk.
    gcloud compute instances detach-disk "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" --device-name=cache || true
    gcloud compute instances delete "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" --delete-disks=boot --quiet || true
    gcloud compute disks delete "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" --quiet || true
    controller_confirm_deleted
  fi
}
trap controller_delete_on_failure EXIT INT TERM
# END EV2_F6_CREATE_AND_SETUP_TEARDOWN

INSTANCE_CREATE_ATTEMPTED=1
if ! gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT \
  --maintenance-policy=TERMINATE \
  --instance-termination-action=DELETE \
  --max-run-duration=21000s \
  --image="$CACHE_IMAGE" \
  --image-project="$PROJECT" \
  --disk=name="$CACHE_DISK",mode=ro,device-name=cache,auto-delete=no \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-balanced \
  --boot-disk-device-name="$INSTANCE" \
  --boot-disk-auto-delete \
  --labels=fable-fleet=pickleball,fable-lane=trackd_ev2_20260722,owner=arnavchokshi \
  --metadata=fable-role=training,fable-cuda-compute-mode=DEFAULT \
  --metadata-from-file=startup-script=/tmp/trackD_ev2_startup.sh
then
  echo 'usc1f A100 Spot create failed; registered failure, ABORT with no zone fallback' >&2
  exit 69
fi

gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
  --format='value(id)' > "$INSTANCE_ID_FILE.tmp"
test -s "$INSTANCE_ID_FILE.tmp"
mv "$INSTANCE_ID_FILE.tmp" "$INSTANCE_ID_FILE"
INSTANCE_ID=$(tr -d '\r\n' < "$INSTANCE_ID_FILE")
test -n "$INSTANCE_ID"
gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --format=json \
  > "$INSTANCE_JSON_FILE.tmp"
mv "$INSTANCE_JSON_FILE.tmp" "$INSTANCE_JSON_FILE"
gcloud compute disks describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --format=json \
  > "$DISK_JSON_FILE.tmp"
mv "$DISK_JSON_FILE.tmp" "$DISK_JSON_FILE"
test "$(jq -r '.id' "$INSTANCE_JSON_FILE")" = "$INSTANCE_ID"
.venv/bin/python - \
  "$INSTANCE_JSON_FILE" "$DISK_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path

instance = json.loads(Path(sys.argv[1]).read_text())
disk = json.loads(Path(sys.argv[2]).read_text())
basename = lambda value: str(value).rstrip('/').rsplit('/', 1)[-1]
assert basename(instance['machineType']) == 'a2-highgpu-1g', instance['machineType']
assert instance['scheduling']['provisioningModel'] == 'SPOT', instance['scheduling']
assert instance['scheduling']['onHostMaintenance'] == 'TERMINATE', instance['scheduling']
assert instance['scheduling']['instanceTerminationAction'] == 'DELETE', instance['scheduling']
assert int(instance['scheduling']['maxRunDuration']['seconds']) == 21000, instance['scheduling']
accelerators = instance.get('guestAccelerators', [])
assert len(accelerators) == 1, accelerators
assert basename(accelerators[0]['acceleratorType']) == 'nvidia-tesla-a100', accelerators
assert int(accelerators[0]['acceleratorCount']) == 1, accelerators
assert int(disk['sizeGb']) == 250, disk
assert basename(disk['type']) == 'pd-balanced', disk['type']
assert '/projects/gifted-electron-498923-h1/' in disk['sourceImage'], disk['sourceImage']
assert basename(disk['sourceImage']).startswith(
    'pickleball-cache-image-20260722'
), disk['sourceImage']
attachments = instance.get('disks', [])
assert len(attachments) == 2, attachments
boot = next(row for row in attachments if row['boot'] is True)
cache = next(row for row in attachments if row['deviceName'] == 'cache')
assert boot['autoDelete'] is True, boot
assert boot['deviceName'] == 'pickleball-gpu-ev2', boot
assert basename(boot['source']) == 'pickleball-gpu-ev2', boot
assert cache['boot'] is False, cache
assert cache['autoDelete'] is False, cache
assert cache['mode'] == 'READ_ONLY', cache
assert basename(cache['source']) == 'pickleball-cache-data-usc1f', cache
interfaces = instance.get('networkInterfaces', [])
assert len(interfaces) == 1, interfaces
access_configs = interfaces[0].get('accessConfigs', [])
assert len(access_configs) == 1, access_configs
assert access_configs[0]['type'] == 'ONE_TO_ONE_NAT', access_configs
assert access_configs[0].get('natIP'), access_configs
assert instance['status'] == 'RUNNING', instance['status']
print({
    'machine_type': basename(instance['machineType']),
    'provisioning_model': instance['scheduling']['provisioningModel'],
    'accelerators': accelerators,
    'disk_gb': int(disk['sizeGb']),
    'disk_type': basename(disk['type']),
    'source_image': basename(disk['sourceImage']),
})
PY

# creationTimestamp is a conservative provider-origin clock: it begins before SSH,
# cache attach/mount, exact checkout, retained transports, the gate, probes,
# training, scoring, and pull-back.
INSTANCE_START_UTC=$(jq -r '.creationTimestamp' "$INSTANCE_JSON_FILE")
test -n "$INSTANCE_START_UTC"
.venv/bin/python - \
  "$INSTANCE_START_UTC" \
  "$ALL_IN_INSTANCE_RATE_USD_PER_HOUR" \
  "$A100_RATE_SOURCE" \
  "$RUN_COMMIT" \
  "$PRICE_PROOF" \
  /tmp/trackD_ev2_spend_bootstrap.json <<'PY'
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

start = datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
rate = float(sys.argv[2])
assert start.tzinfo is not None
assert math.isfinite(rate) and 0.0 < rate <= 3.30
assert sys.argv[3].strip()
assert len(sys.argv[4]) == 40 and all(c in '0123456789abcdef' for c in sys.argv[4])
price_proof_path = Path(sys.argv[5])
price_proof = json.loads(price_proof_path.read_text())
assert float(price_proof['all_in_instance_rate_usd_per_hour']) == rate
all_in_cap_minutes = min(350, math.floor(19.50 * 60.0 / rate))
assert all_in_cap_minutes >= 60
start_epoch = int(start.timestamp())
payload = {
    'instance_start_utc': start.astimezone(timezone.utc).isoformat(),
    'instance_start_epoch': start_epoch,
    'clock_source': 'gce_creationTimestamp',
    'all_in_instance_rate_usd_per_hour': rate,
    'rate_source': sys.argv[3].strip(),
    'run_commit': sys.argv[4],
    'price_proof': price_proof,
    'instance': 'pickleball-gpu-ev2',
    'zone': 'us-central1-f',
    'project': 'gifted-electron-498923-h1',
    'all_in_cap_minutes': all_in_cap_minutes,
    'deadline_epoch': start_epoch + all_in_cap_minutes * 60,
}
Path(sys.argv[6]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY

ALL_IN_DEADLINE_EPOCH=$(jq -r '.deadline_epoch' /tmp/trackD_ev2_spend_bootstrap.json)
REMAINING_SECONDS=$((ALL_IN_DEADLINE_EPOCH - $(date +%s)))
GUEST_WATCHDOG_MINUTES=$((REMAINING_SECONDS / 60))
test "$GUEST_WATCHDOG_MINUTES" -gt 0

# This controller watchdog survives the setup shell. At the absolute deadline
# it stops compute, then detaches the shared cache disk and deletes the disposable
# VM/boot disk ten minutes later if the
# normal verified pull path has not already stopped the VM and cancelled it.
# The immediate EXIT/INT/TERM trap remains armed through setup and staging.
nohup bash -c 'sleep "$1"; current=$(gcloud compute instances describe "$2" --project="$4" --zone="$3" --format="value(id)" 2>/dev/null || true); test "$current" = "$5" || exit 43; gcloud compute instances stop "$2" --project="$4" --zone="$3"; sleep 600; current=$(gcloud compute instances describe "$2" --project="$4" --zone="$3" --format="value(id)" 2>/dev/null || true); test -z "$current" || test "$current" = "$5" || exit 43; gcloud compute instances detach-disk "$2" --project="$4" --zone="$3" --device-name=cache || true; gcloud compute instances delete "$2" --project="$4" --zone="$3" --delete-disks=boot --quiet' \
  _ "$REMAINING_SECONDS" "$INSTANCE" "$ZONE" "$PROJECT" "$INSTANCE_ID" \
  >/tmp/trackD_ev2_controller_watchdog.log 2>&1 &
CONTROLLER_WATCHDOG_PID=$!
printf '%s\n' "$CONTROLLER_WATCHDOG_PID" > "$WATCHDOG_PID_FILE.tmp"
mv "$WATCHDOG_PID_FILE.tmp" "$WATCHDOG_PID_FILE"

# Bound SSH readiness. The boot-time and provider rails are already active.
SSH_READY=0
for _ in $(seq 1 60); do
  if gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command true; then
    SSH_READY=1
    break
  fi
  sleep 5
done
test "$SSH_READY" = 1

# Do not race the metadata startup script's first shutdown schedule. Its
# completion marker is written only after the boot rail and CUDA DEFAULT mode.
STARTUP_READY=0
for _ in $(seq 1 120); do
  if gcloud compute ssh "arnavchokshi@$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" --command \
    'test -s /var/tmp/trackD_ev2_boot_rail_armed.txt && test -s /var/tmp/trackD_ev2_startup_complete.txt'
  then
    STARTUP_READY=1
    break
  fi
  sleep 5
done
test "$STARTUP_READY" = 1

gcloud compute scp \
  /tmp/trackD_ev2_spend_bootstrap.json \
  "$INSTANCE_JSON_FILE" \
  "$DISK_JSON_FILE" \
  /tmp/trackD_ev2_price_proof.json \
  /tmp/trackD_ev2_startup.sh \
  "arnavchokshi@$INSTANCE:/tmp/" \
  --project="$PROJECT" --zone="$ZONE"
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  "sudo shutdown -c || true; sudo shutdown -P +$GUEST_WATCHDOG_MINUTES 'trackD E-v2 price-aware hard wall'; test -f /run/systemd/shutdown/scheduled; sudo cat /run/systemd/shutdown/scheduled | tee /tmp/trackD_ev2_price_aware_rail.txt >/dev/null"

# Mount the shared fleet cache read-only using the GPU-verified device path.
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  'sudo mkdir -p /cache && sudo mount -o ro /dev/disk/by-id/google-cache /cache && mountpoint -q /cache && findmnt -no OPTIONS /cache | tr "," "\n" | grep -Fx ro'

# Bootstrap the baked repo at exact RUN_COMMIT; clone and environment rebuild are removed.
cat > /tmp/trackD_ev2_vm_setup.sh <<'SETUP'
#!/usr/bin/env bash
set -euo pipefail
RUN_COMMIT="$1"
test ! -e /home/arnavchokshi/pickleball
test -d /home/arnavchokshi/coldstart_20260706/repo/.git
ln -s /home/arnavchokshi/coldstart_20260706/repo /home/arnavchokshi/pickleball
cd /home/arnavchokshi/pickleball
git fetch origin main
git cat-file -e "$RUN_COMMIT^{commit}"
git checkout -B main "$RUN_COMMIT"
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = "$RUN_COMMIT"
test -f scripts/racketsport/verify_training_inputs.py

# Register the exact runtime inherited from the cache image; do not install or mutate it.
test -x .venv/bin/python
.venv/bin/python - <<'PY'
import cv2
import numpy
import scipy
import torch
import torchvision
assert torch.cuda.is_available(), torch.__version__
assert torch.__version__ == '2.13.0+cu130', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torchvision.__version__ == '0.28.0+cu130', torchvision.__version__
print({
    'cv2': cv2.__version__,
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'cuda': torch.version.cuda,
})
PY
.venv/bin/python -m pip freeze | sort > /tmp/trackD_ev2_python_environment.txt
SETUP
chmod 755 /tmp/trackD_ev2_vm_setup.sh
bash -n /tmp/trackD_ev2_vm_setup.sh
gcloud compute scp /tmp/trackD_ev2_vm_setup.sh \
  "arnavchokshi@$INSTANCE:/tmp/trackD_ev2_vm_setup.sh" --project="$PROJECT" --zone="$ZONE"
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  "bash /tmp/trackD_ev2_vm_setup.sh '$RUN_COMMIT'"

# Transport split is intentionally conservative:
# - SWAPPED to shared cache: only the six Stage-P pb.vision media rows proven below.
# - REGISTERED PATH UNCHANGED: the 40-MP4 owner rally-media universe, owner/manifests,
#   Stage-P and hard-negative manifests/decisions, and frozen T20 checkpoint retain
#   their exact existing SCP/tar paths. CACHE_MANIFEST does not prove those exact
#   objects at those pins, so similarly named cache rows are not substitutes.
#
# Stage the immutable 40-file rally-media universe without reading the owner
# manifest at all. This makes the later one-touch judge executable while Stage F
# still filters to and opens only split=train. Media presence is not label access.
find data/online_harvest_20260706/rallies -type f -name '*.mp4' -print | sort \
  > /tmp/trackD_ev2_owner_media.txt
test "$(wc -l < /tmp/trackD_ev2_owner_media.txt | tr -d ' ')" = 40
while IFS= read -r media_path; do
  shasum -a 256 "$media_path"
done < /tmp/trackD_ev2_owner_media.txt > /tmp/trackD_ev2_owner_media_SHA256SUMS
tar -cf /tmp/trackD_ev2_owner_media.tar -T /tmp/trackD_ev2_owner_media.txt

gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  'mkdir -p /home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2 /home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull/abc_out /home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull/inputs /home/arnavchokshi/pickleball/runs/lanes/ball_event_abc_20260720/inputs'

gcloud compute scp \
  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json \
  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/agreement_decisions.jsonl \
  "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/" \
  --project="$PROJECT" --zone="$ZONE"
gcloud compute scp \
  runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json \
  "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull/abc_out/" \
  --project="$PROJECT" --zone="$ZONE"
gcloud compute scp \
  runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt \
  "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/abc_experiment_20260721/vm_pull/inputs/" \
  --project="$PROJECT" --zone="$ZONE"
gcloud compute scp \
  runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
  "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/ball_event_abc_20260720/inputs/" \
  --project="$PROJECT" --zone="$ZONE"
gcloud compute scp \
  /tmp/trackD_ev2_owner_media.tar \
  /tmp/trackD_ev2_owner_media_SHA256SUMS \
  "arnavchokshi@$INSTANCE:/tmp/" --project="$PROJECT" --zone="$ZONE"
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  'tar -xf /tmp/trackD_ev2_owner_media.tar -C /home/arnavchokshi/pickleball && cd /home/arnavchokshi/pickleball && sha256sum -c /tmp/trackD_ev2_owner_media_SHA256SUMS && .venv/bin/python -c "from pathlib import Path; from scripts.racketsport.finetune_event_head import validate_registered_rate_media_inventory as v; r=v(Path(\"data/online_harvest_20260706/rallies\"), Path(\"runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json\"), \"79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee\"); assert len(r[\"train\"]) == 38 and len(r[\"validation\"]) == 2"'

# Recreate the exact registered Stage-P absolute paths as symlinks to the shared
# read-only cache and verify every source video against the UNCHANGED INPUT_LOCK pins.
# Eligibility is load-bearing: a row is swappable only when flags contains the exact
# `sha256_matches` token and contains none of SHA256_MISMATCH,
# COMPARE_ONLY_NEVER_TRAIN, or QUARANTINED*. The mismatch-flagged 83gyqyc10y8f row
# is not an E-v2 teacher clip, but it is explicitly ineligible under this rule.
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/arnavchokshi/pickleball
mkdir -p /home/arnavchokshi/pbv_media_root
: > /tmp/trackD_ev2_stage_p_media_SHA256SUMS
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

input_lock = json.loads(
    Path('runs/lanes/trackD_ev2_design_20260722/INPUT_LOCK.json').read_text()
)
cache_manifest = json.loads(
    Path('runs/lanes/trackE_fleetcache_20260722/CACHE_MANIFEST.json').read_text()
)
pins = input_lock['inputs']['stage_p_public_media']['sha256_by_video_id']
selected = [
    '143sf3gdwxsa',
    '98z43hspqz13',
    'st0epgnab7dr',
    'td2szayjwtrj',
    'utasf5hnozwz',
    'xkadsq9bli3h',
]
assert set(pins) == set(selected), (sorted(pins), selected)
rows = {row['id']: row for row in cache_manifest['media']['pbvision']}
for video_id in selected:
    row = rows[video_id]
    flags = set(row.get('flags', []))
    assert 'sha256_matches' in flags, (video_id, sorted(flags))
    assert 'SHA256_MISMATCH' not in flags, (video_id, sorted(flags))
    assert 'COMPARE_ONLY_NEVER_TRAIN' not in flags, (video_id, sorted(flags))
    assert not any(flag.startswith('QUARANTINED') for flag in flags), (
        video_id, sorted(flags)
    )
    assert row['expected_sha256'] == pins[video_id], (video_id, row, pins[video_id])
    source = Path(row['path'])
    assert str(source).startswith('/cache/'), source
    target = Path('/home/arnavchokshi/pbv_media_root') / video_id / 'max.mp4'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    target.symlink_to(source)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == pins[video_id], (video_id, digest, pins[video_id])
    with Path('/tmp/trackD_ev2_stage_p_media_SHA256SUMS').open('a') as handle:
        handle.write(f'{digest}  {target}\n')

excluded = rows['83gyqyc10y8f']
excluded_flags = set(excluded.get('flags', []))
assert (
    'SHA256_MISMATCH' in excluded_flags
    or 'COMPARE_ONLY_NEVER_TRAIN' in excluded_flags
    or any(flag.startswith('QUARANTINED') for flag in excluded_flags)
), excluded
assert '83gyqyc10y8f' not in pins
PY
REMOTE

# Step 0: materialize the gate's reviewed training_input_manifest schema and run
# the separately landed, fail-closed CLI before any training input read. This
# first proof covers Stage-P's pre-existing inputs and remains preserved as the
# required gate_proof.json. Stage-F refreshes separate 900-second proofs later
# for its exact then-existing inputs; it never overwrites this step-0 artifact.
gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/arnavchokshi/pickleball
EV2_GPU_OUT=runs/lanes/trackD_ev2_gpu_20260722
mkdir -p "$EV2_GPU_OUT"
.venv/bin/python - "$EV2_GPU_OUT/training_inputs_step0.json" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(
    Path('runs/lanes/trackD_ev2_design_20260722/INPUT_LOCK.json').read_text()
)
inputs = [
    {
        'path': lock['inputs']['stage_p_agreement_manifest']['path'],
        'asset_id': 'event_abc_vm_pull_20260721',
        'sha256': lock['inputs']['stage_p_agreement_manifest']['sha256'],
    },
    {
        'path': lock['inputs']['stage_p_init_checkpoint']['path'],
        'asset_id': 'event_abc_vm_pull_20260721',
        'sha256': lock['inputs']['stage_p_init_checkpoint']['sha256'],
    },
]
for video_id, sha256 in sorted(
    lock['inputs']['stage_p_public_media']['sha256_by_video_id'].items()
):
    inputs.append({
        'path': f'/home/arnavchokshi/pbv_media_root/{video_id}/max.mp4',
        'asset_id': 'pbvision_gallery_20260719',
        'source_id': video_id,
        'sha256': sha256,
    })
Path(sys.argv[1]).write_text(json.dumps({
    'schema_version': 1,
    'artifact_type': 'training_input_manifest',
    'inputs': inputs,
}, indent=2, sort_keys=True) + '\n')
PY
.venv/bin/python scripts/racketsport/verify_training_inputs.py \
  --inputs "$EV2_GPU_OUT/training_inputs_step0.json" \
  --ledger runs/manager/data_ledger.json \
  --cache-manifest runs/lanes/trackE_fleetcache_20260722/CACHE_MANIFEST.json \
  --repo-root . \
  --gate-proof "$EV2_GPU_OUT/gate_proof.json"
.venv/bin/python - "$EV2_GPU_OUT/gate_proof.json" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

proof = json.loads(Path(sys.argv[1]).read_text())
assert proof['status'] == 'PASS', proof
assert proof['repo_head_sha'] == sys.argv[2], proof
PY
REMOTE

# All setup, integrity-only staging, and step-0 gating are complete. Expected
# setup is 20 minutes total: about 4 minutes for cache-image boot/attach (the
# fleet smoke measured 3.6 minutes), plus 16 minutes for exact checkout, the
# retained ~1.01 GiB owner-media transport, small registered inputs, hashes, and
# gate proof. Independent provider, boot, controller, and guest rails remain armed.
CONTROLLER_DELETE_ON_EXIT=0
trap - EXIT INT TERM
```

If any source file, cache attach/mount, gate/proof, SHA check, clean-main assertion,
exact-shape check, or immutable media staging step fails, stop. In particular, an usc1f
capacity failure is `ABORT`, never a zone fallback. Do not substitute a VM, image,
accelerator, input, media file, owner-validation row, or protected artifact.

## 2. VM preflight and immutable checks

Open one VM shell and keep all remaining commands in that shell:

```bash
gcloud compute ssh arnavchokshi@pickleball-gpu-ev2 \
  --project=gifted-electron-498923-h1 --zone=us-central1-f
```

Then run:

```bash
set -euo pipefail
cd /home/arnavchokshi/pickleball
test "$(git branch --show-current)" = main

EV2_GPU_OUT=runs/lanes/trackD_ev2_gpu_20260722
STEP0_GATE_PROOF="$EV2_GPU_OUT/gate_proof.json"
STAGE_P_MANIFEST=runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json
HARDNEG_INVALID=runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json
OWNER_MANIFEST=runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json
T20_INIT=runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt
P_PROBE="$EV2_GPU_OUT/stage_p_probe"
P_FULL="$EV2_GPU_OUT/stage_p_full"
F_PROBE="$EV2_GPU_OUT/stage_f_probe"
F_FULL="$EV2_GPU_OUT/stage_f_full"
OWNER_SCORE="$EV2_GPU_OUT/ev2_owner41_once.json"
OWNER_SCORE_MARKER="$EV2_GPU_OUT/OWNER41_SCORE_TOKEN_CONSUMED"
SPEND_GUARD="$EV2_GPU_OUT/SPEND_GUARD.json"
RATE_MEDIA_LOCK=runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json
RATE_MEDIA_LOCK_SHA256=79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee

test -d "$EV2_GPU_OUT"
test -s "$STEP0_GATE_PROOF"
.venv/bin/python - "$STEP0_GATE_PROOF" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

proof = json.loads(Path(sys.argv[1]).read_text())
assert proof.get('pass') is True or proof.get('status') == 'pass', proof
assert proof.get('repo_head_sha') == sys.argv[2], proof
PY
cp /tmp/trackD_ev2_python_environment.txt "$EV2_GPU_OUT/python_environment.txt"
cp /tmp/trackD_ev2_spend_bootstrap.json "$EV2_GPU_OUT/spend_bootstrap.json"
cp /tmp/trackD_ev2_price_proof.json "$EV2_GPU_OUT/price_proof.json"
cp /tmp/trackD_ev2_instance.json "$EV2_GPU_OUT/gce_instance.json"
cp /tmp/trackD_ev2_disk.json "$EV2_GPU_OUT/gce_disk.json"
cp /tmp/trackD_ev2_startup.sh "$EV2_GPU_OUT/startup_script.sh"
cp /tmp/trackD_ev2_vm_setup.sh "$EV2_GPU_OUT/vm_setup.sh"
cp /tmp/trackD_ev2_owner_media_SHA256SUMS "$EV2_GPU_OUT/owner_media_SHA256SUMS"
cp /tmp/trackD_ev2_stage_p_media_SHA256SUMS "$EV2_GPU_OUT/stage_p_media_SHA256SUMS"
cp /tmp/trackD_ev2_price_aware_rail.txt "$EV2_GPU_OUT/price_aware_rail.txt"
cp /var/tmp/trackD_ev2_boot_rail_armed.txt "$EV2_GPU_OUT/boot_rail_armed.txt"
cp /var/tmp/trackD_ev2_startup_complete.txt "$EV2_GPU_OUT/startup_complete.txt"
cp /var/tmp/trackD_ev2_startup.log "$EV2_GPU_OUT/startup.log"

record_terminal_verdict() {
  .venv/bin/python - "$EV2_GPU_OUT/VERDICT.json" "$1" "$OWNER_SCORE_MARKER" <<'PY'
import json
import sys
from pathlib import Path
owner_token_consumed = Path(sys.argv[3]).exists()
Path(sys.argv[1]).write_text(json.dumps({
    'artifact_type': 'event_ev2_registered_verdict',
    'verified': False,
    'owner41_score_calls': int(owner_token_consumed),
    'protected50_score_calls': 0,
    'verdict': sys.argv[2],
}, indent=2, sort_keys=True) + '\n')
PY
}

record_incomplete_on_error() {
  if ! test -f "$EV2_GPU_OUT/VERDICT.json"; then
    record_terminal_verdict EVENT_EV2_RUN_INCOMPLETE
  fi
}
trap record_incomplete_on_error ERR

# The controller captured the authoritative all-in Spot rate and the provider's
# creation timestamp before any VM setup. Derive the durable run guard
# from that immutable bootstrap; never re-prompt or restart the spend clock.
SPEND_BOOTSTRAP=/tmp/trackD_ev2_spend_bootstrap.json
test -f "$SPEND_BOOTSTRAP"
.venv/bin/python - "$SPEND_BOOTSTRAP" "$SPEND_GUARD" <<'PY'
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

bootstrap_path, output = sys.argv[1:]
bootstrap = json.loads(Path(bootstrap_path).read_text())
start_text = bootstrap['instance_start_utc']
start = datetime.fromisoformat(start_text.replace('Z', '+00:00'))
assert start.tzinfo is not None, start_text
rate = float(bootstrap['all_in_instance_rate_usd_per_hour'])
assert math.isfinite(rate) and 0.0 < rate <= 3.30, rate
rate_source = str(bootstrap['rate_source']).strip()
assert rate_source, 'quote source is required'
price_proof = bootstrap['price_proof']
assert price_proof['source'] == rate_source
assert float(price_proof['all_in_instance_rate_usd_per_hour']) == rate
assert len(price_proof['components']) == 5
assert bootstrap['clock_source'] == 'gce_creationTimestamp'
assert bootstrap['instance'] == 'pickleball-gpu-ev2'
assert bootstrap['zone'] == 'us-central1-f'
assert bootstrap['project'] == 'gifted-electron-498923-h1'

# Hourly resources are planned below $19.50, while $0.50 is reserved for
# non-hourly network/rounding. The $3.30/hour admission ceiling makes the
# 350-minute hourly projection at most $19.25, leaving another $0.25 for stop
# latency beneath the $20 hard cap. This clock includes the expected 20-minute
# cache-image boot/attach + retained transports + gate setup, at most 300
# registered compute minutes, and the 30-minute pull/hash/stop reserve:
# 20 + 300 + 30 = 350 minutes.
spend_limited_minutes = math.floor(19.50 * 60.0 / rate)
all_in_cap_minutes = min(350, spend_limited_minutes)
assert all_in_cap_minutes >= 60, all_in_cap_minutes
start_epoch = int(start.timestamp())
assert int(bootstrap['instance_start_epoch']) == start_epoch
assert int(bootstrap['all_in_cap_minutes']) == all_in_cap_minutes
assert int(bootstrap['deadline_epoch']) == start_epoch + 60 * all_in_cap_minutes
payload = {
    'artifact_type': 'event_ev2_all_in_spend_guard',
    'verified': False,
    'instance_start_utc': start.astimezone(timezone.utc).isoformat(),
    'instance_start_epoch': start_epoch,
    'all_in_instance_rate_usd_per_hour': rate,
    'rate_source': rate_source.strip(),
    'price_proof': price_proof,
    'hard_spend_cap_usd': 20.0,
    'planning_spend_ceiling_usd': 19.50,
    'all_in_cap_minutes': all_in_cap_minutes,
    'deadline_epoch': start_epoch + 60 * all_in_cap_minutes,
    'maximum_projected_spend_usd': rate * all_in_cap_minutes / 60.0,
    'expected_setup_minutes': 20,
    'registered_compute_schedule_cap_minutes': 300,
    'pull_hash_stop_reserve_minutes': 30,
}
assert payload['maximum_projected_spend_usd'] <= 19.50
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY

ALL_IN_DEADLINE_EPOCH=$(jq -r '.deadline_epoch' "$SPEND_GUARD")
test "$(git rev-parse HEAD)" = "$(jq -r '.run_commit' "$SPEND_BOOTSTRAP")"
WATCHDOG_MINUTES=$(( (ALL_IN_DEADLINE_EPOCH - $(date +%s)) / 60 ))
test "$WATCHDOG_MINUTES" -gt 0
# Section 1 already armed independent controller and guest watchdogs from this
# exact deadline. Re-arming here could move the guest deadline, so only verify
# that the setup reserve has not been exhausted.

require_all_in_budget() {
  local needed_minutes="$1"
  local remaining_minutes=$(( (ALL_IN_DEADLINE_EPOCH - $(date +%s)) / 60 ))
  if test "$remaining_minutes" -lt "$needed_minutes"; then
    echo "all-in spend clock has ${remaining_minutes}m, needs ${needed_minutes}m" >&2
    return 42
  fi
}

refresh_stage_f_gate_proof() {
  local proof_path="$1"
  local manifest_path="$2"
  local generated_input_path="${3:-}"
  .venv/bin/python - \
    "$manifest_path" "$P_FULL/train_manifest.json" "$generated_input_path" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
stage_p_train_manifest = Path(sys.argv[2])
generated_input_path = sys.argv[3]
inputs = [
    {
        'path': 'runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json',
        'asset_id': 'event_abc_inputs_20260720',
        'sha256': '84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b',
    },
    {
        'path': 'runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json',
        'asset_id': 'event_abc_vm_pull_20260721',
        'sha256': '9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5',
    },
    {
        'path': 'runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json',
        'asset_id': 'event_abc_vm_pull_20260721',
        'sha256': 'f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7',
    },
    {
        'path': str(stage_p_train_manifest),
        'asset_id': 'event_ev2_generated_stage_p_20260723',
    },
    {
        'path': 'data/online_harvest_20260706/rallies',
        'asset_id': 'online_harvest_20260706',
    },
    {
        'path': 'runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json',
        'asset_id': 'online_harvest_20260706',
        'sha256': '79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee',
    },
]
if generated_input_path:
    inputs.append({
        'path': generated_input_path,
        'asset_id': 'event_ev2_generated_stage_f_probe_20260723',
    })
output.write_text(json.dumps({
    'schema_version': 1,
    'artifact_type': 'training_input_manifest',
    'inputs': inputs,
}, indent=2, sort_keys=True) + '\n')
PY
  .venv/bin/python scripts/racketsport/verify_training_inputs.py \
    --inputs "$manifest_path" \
    --ledger runs/manager/data_ledger.json \
    --cache-manifest runs/lanes/trackE_fleetcache_20260722/CACHE_MANIFEST.json \
    --repo-root . \
    --gate-proof "$proof_path"
}

sha256sum -c <<'SHA256'
f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json
9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5  runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json
84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b  runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json
f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082  runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt
81df518a85ce891b4b2da1b494f8b123979367d9b70432b4c9af850f4a88792c  runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/agreement_decisions.jsonl
SHA256

sha256sum -c runs/lanes/trackD_ev2_design_20260722/CODE_SHA256SUMS

git diff --exit-code 9bbd8011828631b4cc7df4afdf3b1932e758914a -- \
  scripts/racketsport/eval_event_head.py \
  threed/racketsport/event_head/matcher.py

nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader,nounits | \
  .venv/bin/python -c 'import sys; rows=[x.strip() for x in sys.stdin if x.strip()]; assert len(rows)==1, rows; name,mem,mode=[x.strip() for x in rows[0].rsplit(",",2)]; assert name == "NVIDIA A100-SXM4-40GB", rows; assert 40000 <= int(mem) < 50000, rows; assert mode == "Default", rows; print(rows[0])'

if pgrep -af 'train_event_head.py|finetune_event_head.py|eval_event_head.py'; then
  echo 'another event process is active; aborting' >&2
  exit 1
fi
```

Verify only training media. The owner loop reads `split` on all envelope rows but opens and hashes
media fields only for the 61 train rows:

```bash
.venv/bin/python - "$STAGE_P_MANIFEST" "$HARDNEG_INVALID" "$OWNER_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

expected = {}
for raw_path in sys.argv[1:3]:
    manifest = json.loads(Path(raw_path).read_text())
    assert all(row.get('split') == 'train' for row in manifest['rows'])
    for row in manifest['rows']:
        path = Path(row['video_path'])
        expected[str(path)] = row['source_video_sha256']

owner = json.loads(Path(sys.argv[3]).read_text())
train_count = 0
val_count = 0
for row in owner['rows']:
    split = row.get('split')
    if split == 'val':
        val_count += 1
        continue
    assert split == 'train'
    train_count += 1
    expected[str(Path(row['video_path']))] = row['video_sha256']
assert (train_count, val_count) == (61, 41)
for raw_path, expected_sha in sorted(expected.items()):
    path = Path(raw_path)
    assert path.is_file(), path
    assert digest(path) == expected_sha, path
print({'media_files_verified': len(expected), 'owner_train_rows': train_count, 'owner_val_rows_unopened': val_count})
PY
```

The spend artifact and shutdown watchdog are binding. Do not cancel or extend the watchdog. A
rate above `$3.30/hour`, a missing source, or insufficient remaining all-in time is
`EVENT_EV2_RUN_INCOMPLETE`; it is not permission to restart the VM under this registration.

## 3. Stage-P 100-step throughput probe

The probe includes the step-100 internal validation/threshold sweep. Its weights and threshold lock
are diagnostic only and are never consumed by the full run.

```bash
require_all_in_budget 45  # 15m probe + 30m pull/hash/stop reserve
timeout --signal=TERM --kill-after=30s 15m \
  .venv/bin/python scripts/racketsport/train_event_head.py \
  --full \
  --manifest "$STAGE_P_MANIFEST" \
  --device cuda \
  --out "$P_PROBE" \
  --weights none \
  --steps 100 \
  --image-size 224 \
  --window-frames 64 \
  --batch-size 8 \
  --lr 0.001 \
  --val-every 100 \
  --seed 20260722 \
  --max-wall-minutes 180 \
  --init-checkpoint-model-only "$T20_INIT" \
  --corpus-kind pbvision-agreement \
  --internal-val-source-count 1 \
  --sqrt-frequency-class-weights \
  --label-dilation-frames 1 \
  --label-dilation-soft-weight 0.5 \
  --label-assignment hungarian \
  --assignment-max-shift-frames 2 \
  --assignment-class-cost-weight 1.0 \
  --assignment-temporal-cost-weight 0.25 \
  --offset-regression-head \
  --offset-loss-weight 0.2 \
  --validation-thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --validation-nms-radius 2 \
  --stride-frames 32 \
  --num-workers 4 \
  --prefetch-factor 2

jq -e '.status == "complete" and .completed_steps == 100 and .target_steps == 100 and .honest_partial == false and .decode_threshold_lock != null' \
  "$P_PROBE/train_manifest.json"

P_CAP_MINUTES=$(.venv/bin/python - "$P_PROBE/train_manifest.json" <<'PY'
import json
import math
import sys
report = json.load(open(sys.argv[1]))
assert report['completed_steps'] == report['target_steps'] == 100
print(math.ceil(1.5 * float(report['elapsed_s']) / 100 * 1000 / 60))
PY
)
P_PROBE_MINUTES=$(.venv/bin/python - "$P_PROBE/train_manifest.json" <<'PY'
import json
import math
import sys
print(math.ceil(float(json.load(open(sys.argv[1]))['elapsed_s']) / 60))
PY
)
test "$P_CAP_MINUTES" -le 180
echo "registered Stage-P cap: $P_CAP_MINUTES minutes"
```

## 4. Full Stage P

This restarts from T20; it does not resume the probe.

```bash
require_all_in_budget $((P_CAP_MINUTES + 30))
if ! timeout --signal=TERM --kill-after=30s "${P_CAP_MINUTES}m" \
  .venv/bin/python scripts/racketsport/train_event_head.py \
  --full \
  --manifest "$STAGE_P_MANIFEST" \
  --device cuda \
  --out "$P_FULL" \
  --weights none \
  --steps 1000 \
  --image-size 224 \
  --window-frames 64 \
  --batch-size 8 \
  --lr 0.001 \
  --val-every 100 \
  --seed 20260722 \
  --max-wall-minutes "$P_CAP_MINUTES" \
  --init-checkpoint-model-only "$T20_INIT" \
  --corpus-kind pbvision-agreement \
  --internal-val-source-count 1 \
  --sqrt-frequency-class-weights \
  --label-dilation-frames 1 \
  --label-dilation-soft-weight 0.5 \
  --label-assignment hungarian \
  --assignment-max-shift-frames 2 \
  --assignment-class-cost-weight 1.0 \
  --assignment-temporal-cost-weight 0.25 \
  --offset-regression-head \
  --offset-loss-weight 0.2 \
  --validation-thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --validation-nms-radius 2 \
  --stride-frames 32 \
  --num-workers 4 \
  --prefetch-factor 2
then
  record_terminal_verdict EVENT_EV2_RUN_INCOMPLETE
  exit 31
fi

if ! jq -e '.status == "complete" and .completed_steps == 1000 and .target_steps == 1000 and .honest_partial == false and .all_losses_finite == true and .decode_threshold_lock != null' \
  "$P_FULL/train_manifest.json"
then
  record_terminal_verdict EVENT_EV2_RUN_INCOMPLETE
  exit 31
fi

.venv/bin/python - "$P_FULL/train_manifest.json" <<'PY'
import json
import math
import sys
report = json.load(open(sys.argv[1]))
config = report['config']
assert (report['train_windows'], report['val_windows']) == (963, 226)
assert config['internal_val_source_videos'] == ['st0epgnab7dr']
assert config['train_window_sample_weight_counts'] == {'0.25': 642, '0.5': 321}
expected_counts = [57563.0, 1035.5, 888.5]
expected_weights = [1.0, 7.45584135131073, 8.049019765763125]
assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(config['class_counts_loss_eligible_dense_mass'], expected_counts))
assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(config['class_weights'], expected_weights))
print({'stage_p_train_windows': 963, 'stage_p_val_windows': 226, 'sqrt_weights': expected_weights})
PY

P_LOCK="$P_FULL/stage_p_decode_threshold_lock.json"
jq -e '.owner_val_used == false and .nms_radius_frames == 2 and .checkpoint_step >= 100 and .checkpoint_step <= 1000' "$P_LOCK"
P_LOCK_EXPECTED=$(jq -r '.decode_threshold_lock_sha256' "$P_FULL/train_manifest.json")
P_LOCK_ACTUAL=$(sha256sum "$P_LOCK" | awk '{print $1}')
test "$P_LOCK_ACTUAL" = "$P_LOCK_EXPECTED"
P_BEST=$(jq -r '.best_checkpoint' "$P_FULL/train_manifest.json")
P_BEST_SHA=$(sha256sum "$P_BEST" | awk '{print $1}')
test "$P_BEST_SHA" = "$(jq -r '.checkpoint_sha256' "$P_LOCK")"
LOCKED_THRESHOLD=$(jq -r '.threshold' "$P_LOCK")
```

## 5. Stage-F 100-step throughput probe

The probe repeats deterministic hard-negative mining, skips only terminal pre-score guards, and is
explicitly owner-score-ineligible.

```bash
require_all_in_budget 80  # 30m probe + 20m guard probe + 30m reserve
F_PROBE_GATE_PROOF="$EV2_GPU_OUT/gate_proof_stage_f_probe.json"
refresh_stage_f_gate_proof \
  "$F_PROBE_GATE_PROOF" "$EV2_GPU_OUT/training_inputs_stage_f_probe.json"
timeout --signal=TERM --kill-after=30s 30m \
  .venv/bin/python scripts/racketsport/finetune_event_head.py \
  --owner-manifest "$OWNER_MANIFEST" \
  --owner-manifest-sha256 84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b \
  --init-checkpoint-model-only "$P_BEST" \
  --init-checkpoint-sha256 "$P_BEST_SHA" \
  --stage-p-threshold-lock "$P_LOCK" \
  --stage-p-train-manifest "$P_FULL/train_manifest.json" \
  --gate-proof "$F_PROBE_GATE_PROOF" \
  --out "$F_PROBE" \
  --device cuda \
  --steps 100 \
  --image-size 224 \
  --window-frames 64 \
  --batch-size 8 \
  --lr 0.001 \
  --val-every 100 \
  --seed 20260722 \
  --stride-frames 32 \
  --num-workers 4 \
  --checkpoint-selection final-step \
  --probe-only \
  --class-weighting sqrt-frequency \
  --assignment-mode fixed \
  --assignment-max-shift-frames 0 \
  --assignment-class-cost-weight 1.0 \
  --assignment-temporal-cost-weight 0.25 \
  --label-dilation-frames 1 \
  --label-neighbor-positive-weight 0.5 \
  --offset-loss-weight 0.2 \
  --offset-smooth-l1-beta 1.0 \
  --hard-negative-invalid-manifest "$HARDNEG_INVALID" \
  --hard-negative-invalid-manifest-sha256 9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5 \
  --hard-negative-repaired-manifest "$STAGE_P_MANIFEST" \
  --hard-negative-repaired-manifest-sha256 f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7 \
  --hard-negative-expected-candidates 262 \
  --hard-negative-top-k 96 \
  --hard-negative-batch-size 4 \
  --hard-negative-excluded-source-video st0epgnab7dr \
  --hard-negative-loss-cap 0.5 \
  --internal-decode-threshold "$LOCKED_THRESHOLD" \
  --expected-owner-train-negative-rows 21 \
  --internal-owner-negative-max-fp 2 \
  --internal-audio-only-max-fired-rows 26 \
  --internal-rate-min-per-s 0.3 \
  --internal-rate-max-per-s 1.0 \
  --owner-media-root data/online_harvest_20260706/rallies \
  --rate-media-inventory "$RATE_MEDIA_LOCK" \
  --rate-media-inventory-sha256 "$RATE_MEDIA_LOCK_SHA256" \
  --owner-train-source-video 73VurrTKCZ8 \
  --owner-train-source-video Ezz6HDNHlnk \
  --owner-train-source-video _L0HVmAlCQI \
  --owner-train-source-video wBu8bC4OfUY \
  --owner-validation-source-video HyUqT7zFiwk \
  --owner-validation-source-video zwCtH_i1_S4 \
  --expected-owner-train-media-paths 38 \
  --expected-owner-train-source-videos 4 \
  --expected-owner-train-rows 61 \
  --expected-owner-val-rows 41 \
  --max-wall-minutes 180

jq -e '.status == "complete_probe_only" and .probe_only == true and .owner_score_eligible == false and .validation_windows == 0 and .owner_validation_rows_uninspected == 41 and .hard_negative_candidate_rows == 262 and .hard_negative_train_windows == 96 and .config.batch_size_human == 8 and .config.batch_size_hard_negative == 4 and .config.owner_batch_sampler_policy == "seeded_permutation_with_deterministic_reshuffle_wrap_top_up_exact_8" and .internal_guards.status == "not_run_probe_only"' \
  "$F_PROBE/finetune_manifest.json"

F_CAP_MINUTES=$(.venv/bin/python - "$F_PROBE/finetune_manifest.json" <<'PY'
import json
import math
import sys
report = json.load(open(sys.argv[1]))
assert report['completed_steps'] == report['target_steps'] == 100
training = float(report['elapsed_training_s'])
one_time = max(0.0, float(report['elapsed_total_s']) - training)
print(math.ceil(1.5 * (one_time + training / 100 * 1000) / 60))
PY
)
F_PROBE_MINUTES=$(.venv/bin/python - "$F_PROBE/finetune_manifest.json" <<'PY'
import json
import math
import sys
print(math.ceil(float(json.load(open(sys.argv[1]))['elapsed_total_s']) / 60))
PY
)
test "$F_CAP_MINUTES" -le 180
echo "registered Stage-F cap: $F_CAP_MINUTES minutes"
```

Measure the otherwise one-time terminal guard workload on the score-ineligible probe checkpoint.
Its pass/fail value is ignored; only the elapsed time is used. This consumes train-side media and
the 262 source-clean audio-only candidates, never owner-validation or protected data.

```bash
require_all_in_budget 50  # 20m guard probe + 30m reserve
F_PROBE_CHECKPOINT=$(jq -r '.best_checkpoint' "$F_PROBE/finetune_manifest.json")
F_GUARD_PROBE="$EV2_GPU_OUT/stage_f_guard_probe.json"
F_GUARD_GATE_PROOF="$EV2_GPU_OUT/gate_proof_stage_f_guard_probe.json"
refresh_stage_f_gate_proof \
  "$F_GUARD_GATE_PROOF" \
  "$EV2_GPU_OUT/training_inputs_stage_f_guard_probe.json" \
  "$F_PROBE_CHECKPOINT"
timeout --signal=TERM --kill-after=30s 20m \
  .venv/bin/python - \
  "$OWNER_MANIFEST" \
  "$HARDNEG_INVALID" \
  "$STAGE_P_MANIFEST" \
  "$F_PROBE_CHECKPOINT" \
  "$LOCKED_THRESHOLD" \
  "$F_GUARD_PROBE" \
  "$F_GUARD_GATE_PROOF" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch

from scripts.racketsport.finetune_event_head import (
    derive_audio_only_hard_negative_pool,
    run_internal_stage_f_guards,
    validate_stage_f_owner_manifest,
)
from scripts.racketsport.verify_training_inputs import assert_gate_proof
from threed.racketsport.event_head.model import load_checkpoint

owner_path, invalid_path, repaired_path, checkpoint_path, threshold, output, gate_proof = sys.argv[1:]
assert_gate_proof(
    Path(gate_proof),
    repo_root=Path.cwd(),
    required_input_paths=[
        Path(owner_path),
        Path(invalid_path),
        Path(repaired_path),
        Path(checkpoint_path),
        Path('data/online_harvest_20260706/rallies'),
        Path('runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json'),
    ],
)
owner, _ = validate_stage_f_owner_manifest(
    Path(owner_path),
    owner_manifest_sha256='84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b',
    window_frames=64,
    expected_owner_train_rows=61,
    expected_owner_val_rows=41,
)
candidates, _ = derive_audio_only_hard_negative_pool(
    Path(invalid_path),
    Path(repaired_path),
    invalid_manifest_sha256='9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5',
    repaired_manifest_sha256='f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7',
    expected_candidates=262,
    window_frames=64,
    excluded_source_video_ids=('st0epgnab7dr',),
    expected_raw_candidates=292,
    expected_excluded_source_rows=30,
)
model, _ = load_checkpoint(checkpoint_path, device='cuda')
started = time.monotonic()
guard = run_internal_stage_f_guards(
    model,
    owner,
    candidates,
    image_size=224,
    window_frames=64,
    batch_size=8,
    device=torch.device('cuda'),
    num_workers=4,
    seed=20260722,
    threshold=float(threshold),
    owner_media_root=Path('data/online_harvest_20260706/rallies'),
    train_source_video_ids=('73VurrTKCZ8', 'Ezz6HDNHlnk', '_L0HVmAlCQI', 'wBu8bC4OfUY'),
    validation_source_video_ids=('HyUqT7zFiwk', 'zwCtH_i1_S4'),
    expected_owner_negative_rows=21,
    owner_negative_max_fp=2,
    audio_only_max_fired_rows=26,
    rate_min_per_s=0.3,
    rate_max_per_s=1.0,
    expected_train_media_paths=38,
    expected_train_source_videos=4,
    rate_media_inventory_path=Path('runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json'),
    rate_media_inventory_sha256='79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee',
)
artifact = {
    'artifact_type': 'event_ev2_train_side_guard_throughput_probe',
    'verified': False,
    'score_eligible': False,
    'elapsed_s': time.monotonic() - started,
    'guard_result_for_timing_only': guard,
}
Path(output).write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
print(json.dumps({'elapsed_s': artifact['elapsed_s'], 'score_eligible': False}))
PY

F_GUARD_PROBE_MINUTES=$(.venv/bin/python - "$F_GUARD_PROBE" <<'PY'
import json
import math
import sys
print(math.ceil(float(json.load(open(sys.argv[1]))['elapsed_s']) / 60))
PY
)
F_GUARD_CAP_MINUTES=$(.venv/bin/python - "$F_GUARD_PROBE" <<'PY'
import json
import math
import sys
print(math.ceil(1.5 * float(json.load(open(sys.argv[1]))['elapsed_s']) / 60))
PY
)
F_TOTAL_CAP_MINUTES=$((F_CAP_MINUTES + F_GUARD_CAP_MINUTES))
TOTAL_REGISTERED_MINUTES=$((P_PROBE_MINUTES + P_CAP_MINUTES + F_PROBE_MINUTES + F_GUARD_PROBE_MINUTES + F_TOTAL_CAP_MINUTES + 30))
ALL_IN_REGISTERED_MAX_MINUTES=$((20 + TOTAL_REGISTERED_MINUTES + 30))
test "$F_TOTAL_CAP_MINUTES" -le 210
test "$TOTAL_REGISTERED_MINUTES" -le 300
test "$ALL_IN_REGISTERED_MAX_MINUTES" -le 350
require_all_in_budget $((F_TOTAL_CAP_MINUTES + 60))
echo "registered compute schedule cap: $TOTAL_REGISTERED_MINUTES minutes"
echo "registered all-in arithmetic: 20m setup + ${TOTAL_REGISTERED_MINUTES}m compute + 30m pull/hash/stop = ${ALL_IN_REGISTERED_MAX_MINUTES}m"
```

## 6. Full Stage F and train-side stop gate

This repeats mining from the immutable Stage-P checkpoint; it does not resume the probe. The
terminal checkpoint is fixed at step 1000. The command returns an artifact even when a guard fails,
so the `jq` assertion is binding.

```bash
require_all_in_budget $((F_TOTAL_CAP_MINUTES + 60))
F_FULL_GATE_PROOF="$EV2_GPU_OUT/gate_proof_stage_f_full.json"
refresh_stage_f_gate_proof \
  "$F_FULL_GATE_PROOF" "$EV2_GPU_OUT/training_inputs_stage_f_full.json"
if ! timeout --signal=TERM --kill-after=30s "${F_TOTAL_CAP_MINUTES}m" \
  .venv/bin/python scripts/racketsport/finetune_event_head.py \
  --owner-manifest "$OWNER_MANIFEST" \
  --owner-manifest-sha256 84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b \
  --init-checkpoint-model-only "$P_BEST" \
  --init-checkpoint-sha256 "$P_BEST_SHA" \
  --stage-p-threshold-lock "$P_LOCK" \
  --stage-p-train-manifest "$P_FULL/train_manifest.json" \
  --gate-proof "$F_FULL_GATE_PROOF" \
  --out "$F_FULL" \
  --device cuda \
  --steps 1000 \
  --image-size 224 \
  --window-frames 64 \
  --batch-size 8 \
  --lr 0.001 \
  --val-every 100 \
  --seed 20260722 \
  --stride-frames 32 \
  --num-workers 4 \
  --checkpoint-selection final-step \
  --class-weighting sqrt-frequency \
  --assignment-mode fixed \
  --assignment-max-shift-frames 0 \
  --assignment-class-cost-weight 1.0 \
  --assignment-temporal-cost-weight 0.25 \
  --label-dilation-frames 1 \
  --label-neighbor-positive-weight 0.5 \
  --offset-loss-weight 0.2 \
  --offset-smooth-l1-beta 1.0 \
  --hard-negative-invalid-manifest "$HARDNEG_INVALID" \
  --hard-negative-invalid-manifest-sha256 9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5 \
  --hard-negative-repaired-manifest "$STAGE_P_MANIFEST" \
  --hard-negative-repaired-manifest-sha256 f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7 \
  --hard-negative-expected-candidates 262 \
  --hard-negative-top-k 96 \
  --hard-negative-batch-size 4 \
  --hard-negative-excluded-source-video st0epgnab7dr \
  --hard-negative-loss-cap 0.5 \
  --internal-decode-threshold "$LOCKED_THRESHOLD" \
  --expected-owner-train-negative-rows 21 \
  --internal-owner-negative-max-fp 2 \
  --internal-audio-only-max-fired-rows 26 \
  --internal-rate-min-per-s 0.3 \
  --internal-rate-max-per-s 1.0 \
  --owner-media-root data/online_harvest_20260706/rallies \
  --rate-media-inventory "$RATE_MEDIA_LOCK" \
  --rate-media-inventory-sha256 "$RATE_MEDIA_LOCK_SHA256" \
  --owner-train-source-video 73VurrTKCZ8 \
  --owner-train-source-video Ezz6HDNHlnk \
  --owner-train-source-video _L0HVmAlCQI \
  --owner-train-source-video wBu8bC4OfUY \
  --owner-validation-source-video HyUqT7zFiwk \
  --owner-validation-source-video zwCtH_i1_S4 \
  --expected-owner-train-media-paths 38 \
  --expected-owner-train-source-videos 4 \
  --expected-owner-train-rows 61 \
  --expected-owner-val-rows 41 \
  --max-wall-minutes "$F_CAP_MINUTES"
then
  record_terminal_verdict EVENT_EV2_RUN_INCOMPLETE
  exit 31
fi

if ! jq -e '.status == "complete" and .completed_steps == 1000 and .target_steps == 1000 and .all_losses_finite == true and .probe_only == false and .validation_windows == 0 and .owner_validation_rows_uninspected == 41 and .hard_negative_candidate_rows == 262 and .hard_negative_train_windows == 96 and .config.batch_size_human == 8 and .config.batch_size_hard_negative == 4 and .config.max_wall_scope == "hard_negative_mining_plus_optimizer" and .owner_score_eligible == true and .internal_guards.pass == true and .internal_guards.checks.audio_only_rows_with_predictions.denominator_rows == 262 and .internal_guards.checks.audio_only_rows_with_predictions.maximum == 26 and .internal_guards.full_video_train_source_proxy.unique_media_path_count == 38 and .internal_guards.full_video_train_source_proxy.distinct_source_video_count == 4 and .internal_guards.full_video_train_source_proxy.train_validation_path_overlap == [] and .provenance.protected_inventory_opened == false' \
  "$F_FULL/finetune_manifest.json"
then
  if test -f "$F_FULL/finetune_manifest.json" && \
    test "$(jq -r '.status' "$F_FULL/finetune_manifest.json")" = complete_internal_guard_fail
  then
    record_terminal_verdict EVENT_EV2_INTERNAL_GUARD_FAIL_NO_SCORE
    exit 12
  fi
  record_terminal_verdict EVENT_EV2_RUN_INCOMPLETE
  exit 31
fi

F_CHECKPOINT=$(jq -r '.best_checkpoint' "$F_FULL/finetune_manifest.json")
.venv/bin/python - "$F_CHECKPOINT" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert payload['completed_steps'] == payload['config']['steps'] == 1000
assert payload['config']['checkpoint_selection'] == 'final-step'
assert payload['config']['owner_score_eligible'] is True
assert payload['checkpoint_role'] == 'terminal_step_internal_guards_pass'
print({'terminal_checkpoint': sys.argv[1], 'completed_steps': 1000})
PY
```

If the `jq` assertion fails, record `EVENT_EV2_INTERNAL_GUARD_FAIL_NO_SCORE`, stop, and do not run
section 7.

## 7. The one owner-41 score

This is the only command in the entire plan that may inspect val-row fields, verify val media, or
construct/score owner-41. The marker is created **before** the integrity check and evaluation; if
either action starts and fails, the token is still consumed and may not be retried under this
registration. The one 30-minute timeout covers both media hashing and judge inference.

```bash
require_all_in_budget 60  # 30m judge + 30m pull/hash/stop reserve
test ! -e "$OWNER_SCORE_MARKER"
test ! -e "$OWNER_SCORE"
STAGE_V_ONCE_SCRIPT="$EV2_GPU_OUT/stage_v_once.sh"
cat > "$STAGE_V_ONCE_SCRIPT" <<'STAGEV'
#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="$1"
OWNER_MANIFEST="$2"
INTEGRITY_OUT="$3"
F_CHECKPOINT="$4"
OWNER_SCORE="$5"
LOCKED_THRESHOLD="$6"

# The caller writes the token marker before launching this script. Verify the
# two frozen val media objects against their manifest pins before inference; a
# mismatch is terminal and is never a reason to replace media or retry.
"$PYTHON_BIN" - "$OWNER_MANIFEST" "$INTEGRITY_OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
rows = [row for row in manifest['rows'] if row.get('split') == 'val']
assert len(rows) == 41
expected = {}
for row in rows:
    path = Path(row['video_path'])
    digest = str(row['video_sha256'])
    assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    previous = expected.setdefault(str(path), digest)
    assert previous == digest
assert len(expected) == 2

observed = {}
for raw_path, expected_digest in sorted(expected.items()):
    path = Path(raw_path)
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    actual = value.hexdigest()
    assert actual == expected_digest, path
    observed[raw_path] = actual

artifact = {
    'artifact_type': 'event_ev2_owner41_media_integrity',
    'verified': False,
    'owner41_token_already_consumed': True,
    'validation_rows': len(rows),
    'unique_media_paths': len(expected),
    'sha256_by_path': observed,
    'status': 'pass',
}
Path(sys.argv[2]).write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
print(json.dumps({'validation_rows': len(rows), 'unique_media_paths': len(expected), 'status': 'pass'}))
PY

exec "$PYTHON_BIN" scripts/racketsport/eval_event_head.py \
  --checkpoint "$F_CHECKPOINT" \
  --mode owner-val \
  --out "$OWNER_SCORE" \
  --threshold "$LOCKED_THRESHOLD" \
  --device cuda \
  --arm B \
  --seed 20260722 \
  --manifest "$OWNER_MANIFEST"
STAGEV
chmod 700 "$STAGE_V_ONCE_SCRIPT"

date -u +%Y-%m-%dT%H:%M:%SZ > "$OWNER_SCORE_MARKER"
timeout --signal=TERM --kill-after=30s 30m bash "$STAGE_V_ONCE_SCRIPT" \
  .venv/bin/python \
  "$OWNER_MANIFEST" \
  "$EV2_GPU_OUT/OWNER41_MEDIA_INTEGRITY.json" \
  "$F_CHECKPOINT" \
  "$OWNER_SCORE" \
  "$LOCKED_THRESHOLD"
```

Apply the registered gate and write the terminal verdict:

```bash
.venv/bin/python - "$OWNER_SCORE" "$EV2_GPU_OUT/VERDICT.json" <<'PY'
import json
import sys
from pathlib import Path

score = json.loads(Path(sys.argv[1]).read_text())
checks = {
    'macro_f1_at_2': float(score['macro_f1_at_2']) >= 0.13043478260869568,
    'negative_false_positives': int(score['negative_false_positives']) <= 2,
    'negative_denominator': int(score['negative_rows']) == 22,
    'full_video_rate': 0.3 <= float(score['full_video_events_per_second']) <= 1.0,
    'selection_rows': int(score['selection_rows']) == 41,
    'steps': int(score['completed_steps']) == int(score['target_steps']) == 1000,
    'protected_50_untouched': score['protected_50_touched'] is False,
    'arm': score['arm'] == 'B',
    'seed': int(score['seed']) == 20260722,
}
# The frozen judge collects timing errors only from matches already admitted
# within +/-2 frames. Thus macro_f1_at_2 > 0 implies every timing sample is <=2;
# timing_error_p90_frames is descriptive only and is deliberately absent from
# checks. With no matches the judge reports the window_frames sentinel (64).
passed = all(checks.values())
result = {
    'artifact_type': 'event_ev2_registered_verdict',
    'verified': False,
    'owner41_score_calls': 1,
    'protected50_score_calls': 0,
    'checks': checks,
    'verdict': (
        'EVENT_EV2_RECIPE_REPAIR_PASS'
        if passed else 'EVENT_EV2_RECIPE_REPAIR_NO_LIFT'
    ),
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if passed else 10)
PY
```

A nonzero gate exit is the final negative verdict, not permission to change a threshold or score
again. Never invoke `--mode protected-seed` in this experiment.

## 8. Pull, hash, stop, and delete the disposable VM

From the control workstation, run this single tolerant finalizer after **every** post-create exit:
PASS, guard-stop, scored FAIL, SHA failure, OOM, probe/full-stage timeout, budget failure, setup
failure, or Stage-V timeout. It pulls whatever exists, falls back to the controller's content-blind
spend bootstrap, detaches and preserves the shared cache disk/image, stops/deletes only the
original provider instance ID plus its boot disk, and cancels that instance's identity-bound
watchdog.

```bash
set -uo pipefail
cd /Users/arnavchokshi/Desktop/pickleball
INSTANCE=pickleball-gpu-ev2
ZONE=us-central1-f
PROJECT=gifted-electron-498923-h1
CACHE_DISK=pickleball-cache-data-usc1f
mkdir -p runs/lanes/trackD_ev2_design_20260722/vm_pull
PULLED=runs/lanes/trackD_ev2_design_20260722/vm_pull/trackD_ev2_gpu_20260722
mkdir -p "$PULLED"
EV2_STATE_DIR=${EV2_STATE_DIR:-/tmp/trackD_ev2_controller_state}
INSTANCE_ID_FILE="$EV2_STATE_DIR/instance_id"
WATCHDOG_PID_FILE="$EV2_STATE_DIR/watchdog.pid"
TEARDOWN_CONFIRMED_FILE="$EV2_STATE_DIR/teardown_confirmed"
EXPECTED_INSTANCE_ID=$(tr -d '\r\n' < "$INSTANCE_ID_FILE")
test -n "$EXPECTED_INSTANCE_ID"
TEARDOWN_DONE=0
# BEGIN EV2_F6_TOLERANT_FINALIZER
instance_is_original() {
  current_id=$(gcloud compute instances describe "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" --format='value(id)' 2>/dev/null || true)
  test -z "$current_id" || test "$current_id" = "$EXPECTED_INSTANCE_ID"
}
cleanup_disposable_vm() {
  if test "$TEARDOWN_DONE" = 0; then
    if instance_is_original; then
      # DETACH-never-delete the shared cache disk; the shared cache image is never
      # a deletion target. Delete only this VM and its auto-delete boot disk.
      gcloud compute instances detach-disk "$INSTANCE" \
        --project="$PROJECT" --zone="$ZONE" --device-name=cache || true
      gcloud compute instances delete "$INSTANCE" \
        --project="$PROJECT" --zone="$ZONE" --delete-disks=boot --quiet || true
      gcloud compute disks delete "$INSTANCE" \
        --project="$PROJECT" --zone="$ZONE" --quiet || true
    else
      echo 'refusing cleanup of a recycled same-name instance' >&2
      return 43
    fi
    if gcloud compute instances describe "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
      echo 'instance still exists after tolerant finalizer delete' >&2
      return 44
    fi
    if gcloud compute disks describe "$INSTANCE" \
      --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
      echo 'boot disk still exists after tolerant finalizer delete' >&2
      return 44
    fi
    date -u +%FT%TZ > "$TEARDOWN_CONFIRMED_FILE"
    TEARDOWN_DONE=1
  fi
}
# END EV2_F6_TOLERANT_FINALIZER
trap cleanup_disposable_vm EXIT INT TERM
REMOTE_SHA_OK=0
if instance_is_original && gcloud compute ssh "arnavchokshi@$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command \
  'cd /home/arnavchokshi/pickleball/runs/lanes/trackD_ev2_gpu_20260722 && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > /tmp/trackD_ev2_gpu_20260722_SHA256SUMS && mv /tmp/trackD_ev2_gpu_20260722_SHA256SUMS SHA256SUMS'
then
  REMOTE_SHA_OK=1
fi
if instance_is_original; then
  gcloud compute scp --recurse \
  "arnavchokshi@$INSTANCE:/home/arnavchokshi/pickleball/runs/lanes/trackD_ev2_gpu_20260722" \
  runs/lanes/trackD_ev2_design_20260722/vm_pull/ \
  --project="$PROJECT" --zone="$ZONE" || true
fi
if test "$REMOTE_SHA_OK" = 1 && test -f "$PULLED/SHA256SUMS"; then
  (cd "$PULLED" && sha256sum -c SHA256SUMS) || true
fi
cp /tmp/trackD_ev2_spend_bootstrap.json "$PULLED/spend_bootstrap.json" 2>/dev/null || true
cp /tmp/trackD_ev2_price_proof.json "$PULLED/price_proof.json" 2>/dev/null || true
if instance_is_original; then
  gcloud compute instances stop "$INSTANCE" --project="$PROJECT" --zone="$ZONE" || true
fi
INSTANCE_STOP_UTC=$(gcloud compute instances describe "$INSTANCE" \
  --project="$PROJECT" --zone="$ZONE" --format='value(lastStopTimestamp)' 2>/dev/null || true)
if test -z "$INSTANCE_STOP_UTC"; then INSTANCE_STOP_UTC=$(date -u +%FT%TZ); fi
# The provider stop above must return successfully before the durable controller
# watchdog is cancelled. Refuse to signal a recycled PID with a different command.
if test -f "$WATCHDOG_PID_FILE"; then
  CONTROLLER_WATCHDOG_PID=$(tr -d '\r\n' < "$WATCHDOG_PID_FILE")
  if kill -0 "$CONTROLLER_WATCHDOG_PID" 2>/dev/null; then
    CONTROLLER_WATCHDOG_COMMAND=$(ps -p "$CONTROLLER_WATCHDOG_PID" -o command=)
    case "$CONTROLLER_WATCHDOG_COMMAND" in
      *pickleball-gpu-ev2*us-central1-f*"$EXPECTED_INSTANCE_ID"*) kill "$CONTROLLER_WATCHDOG_PID"; wait "$CONTROLLER_WATCHDOG_PID" 2>/dev/null || true ;;
      *) echo 'watchdog PID command mismatch; refusing to signal it' >&2; exit 43 ;;
    esac
  fi
fi
SPEND_SOURCE="$PULLED/SPEND_GUARD.json"
if ! test -f "$SPEND_SOURCE"; then SPEND_SOURCE="$PULLED/spend_bootstrap.json"; fi
.venv/bin/python - \
  "$SPEND_SOURCE" \
  "$PULLED/CONTROLLER_SPEND_ACTUAL.json" \
  "$INSTANCE_STOP_UTC" <<'PY'
from datetime import datetime
import json
import sys
from pathlib import Path

guard = json.loads(Path(sys.argv[1]).read_text())
stopped = datetime.fromisoformat(sys.argv[3].replace('Z', '+00:00'))
stop_epoch = int(stopped.timestamp())
elapsed_s = stop_epoch - int(guard['instance_start_epoch'])
assert elapsed_s >= 0
hourly_resource_spend = (
    elapsed_s / 3600.0 * float(guard['all_in_instance_rate_usd_per_hour'])
)
non_hourly_reserve = 0.50
spend = hourly_resource_spend + non_hourly_reserve
result = {
    'artifact_type': 'event_ev2_controller_spend_actual',
    'verified': False,
    'instance_start_utc': guard['instance_start_utc'],
    'confirmed_stopped_utc': stopped.isoformat(),
    'all_in_elapsed_s': elapsed_s,
    'all_in_instance_rate_usd_per_hour': guard['all_in_instance_rate_usd_per_hour'],
    'rate_source': guard['rate_source'],
    'hourly_resource_spend_upper_bound_usd': hourly_resource_spend,
    'non_hourly_network_and_rounding_reserve_usd': non_hourly_reserve,
    'spend_usd_upper_bound': spend,
    'hard_spend_cap_usd': 20.0,
    'pass': spend <= 20.0,
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
assert result['pass'], result
PY

.venv/bin/python - "$PULLED/VERDICT.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        'artifact_type': 'event_ev2_registered_verdict',
        'verified': False,
        'owner41_score_calls': None,
        'protected50_score_calls': 0,
        'verdict': 'EVENT_EV2_RUN_INCOMPLETE',
        'reason': 'controller_finalizer_recovered_no_guest_verdict',
    }, indent=2, sort_keys=True) + '\n')
PY

(cd "$PULLED" && \
  find . -type f ! -name FINAL_SHA256SUMS -print | LC_ALL=C sort | \
  while IFS= read -r artifact_path; do shasum -a 256 "$artifact_path"; done \
    > FINAL_SHA256SUMS && \
  shasum -a 256 -c FINAL_SHA256SUMS)

# The verified pull is durable locally. Use the same trap-backed finalizer on
# the success route and require its durable delete confirmation before disarm.
cleanup_disposable_vm
test -s "$TEARDOWN_CONFIRMED_FILE"
trap - EXIT INT TERM
```

The GPU handoff must report the measured probe times, computed caps, total A100 time, spot price,
spend, all SHA results, guard artifact, whether the owner token was consumed, and the exact verdict.

## 9. PASS-only pending handoff to the serialized integration owner

The executable GPU plan does **not** edit `configs/racketsport/best_stack.json` or its revision-
pinned test. If and only if the pulled verdict is `EVENT_EV2_RECIPE_REPAIR_PASS`, emit one immutable
candidate payload for the Track-D integration owner. That owner validates and applies any later
best-stack/test edit in a separate serialized transaction; a half-applied production mutation is
therefore impossible in this GPU lane.

```bash
set -euo pipefail
cd /Users/arnavchokshi/Desktop/pickleball
test "$(git branch --show-current)" = main
PULLED=runs/lanes/trackD_ev2_design_20260722/vm_pull/trackD_ev2_gpu_20260722

.venv/bin/python - "$PULLED" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

pulled = Path(sys.argv[1])
verdict_path = pulled / 'VERDICT.json'
checkpoint = pulled / 'stage_f_full/best_event_head_finetuned.pt'
score_path = pulled / 'ev2_owner41_once.json'
lock_path = pulled / 'stage_p_full/stage_p_decode_threshold_lock.json'
verdict = json.loads(verdict_path.read_text())
assert verdict['verdict'] == 'EVENT_EV2_RECIPE_REPAIR_PASS'
assert verdict['owner41_score_calls'] == 1
assert verdict['protected50_score_calls'] == 0
score = json.loads(score_path.read_text())
lock = json.loads(lock_path.read_text())
payload = {
    'artifact_type': 'event_ev2_best_stack_pending_handoff',
    'verified': False,
    'integration_owner_required': True,
    'production_files_mutated': [],
    'entry_key': 'events.ev2_checkpoint',
    'status': 'PENDING',
    'value': {
        'enabled': False,
        'kind': 'local_path',
        'path': str(checkpoint),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'decode_threshold': float(lock['threshold']),
        'nms_radius_frames': 2,
    },
    'gate': {
        'name': 'EVENT_EV2_RECIPE_REPAIR_PASS',
        'bar': 'macro-F1>=0.13043478260869568; negFP<=2/22; rate 0.3-1.0/s',
    },
    'evidence': {
        'verdict': str(verdict_path),
        'score': str(score_path),
        'threshold_lock': str(lock_path),
        'owner41_macro_f1_at_2': float(score['macro_f1_at_2']),
        'owner41_negative_false_positives': int(score['negative_false_positives']),
        'owner41_full_video_events_per_second': float(score['full_video_events_per_second']),
        'descriptive_timing_error_p90_frames': float(score['timing_error_p90_frames']),
    },
    'notes': (
        'Disabled PENDING RGB-only candidate. VERIFIED=0 remains binding. The serialized '
        'integration owner must validate both production files fully before any mutation.'
    ),
}
output = pulled / 'BEST_STACK_PENDING.json'
assert not output.exists()
temporary = pulled / '.BEST_STACK_PENDING.json.tmp'
temporary.unlink(missing_ok=True)
try:
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    with temporary.open('r+b') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    directory_fd = os.open(pulled, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps({'pending_handoff': str(output), 'production_files_mutated': []}))
PY

(cd "$PULLED" && \
  find . -type f ! -name FINAL_SHA256SUMS -print | LC_ALL=C sort | \
  while IFS= read -r artifact_path; do shasum -a 256 "$artifact_path"; done \
    > FINAL_SHA256SUMS && \
  shasum -a 256 -c FINAL_SHA256SUMS)
git diff --exit-code -- configs/racketsport/best_stack.json tests/racketsport/test_best_stack_manifest.py
```

On any non-PASS verdict, section 9 is forbidden and no pending handoff is emitted.
`sequence_dp.py` wiring remains a separate lane even on PASS.
