# STATUS: BLOCKED / INCOMPLETE DRAFT — DO NOT RUN AS-IS (ultra review3 must-fix, 2026-07-20)
The "dependency-packaged" and handoff-PASS claims below are RETRACTED: the mission assumes a pinned
repo/payload this bundle does not provision; OSNet/torchreid/clips/frozen GT/calibrations/diagnosis
fixtures are checked but NOT packaged; the scorer can exit 0 on recorded errors (needs fail-closed
assertions on error count, both clip rows, near-miss fidelity, and every preregistered threshold);
the scaffold startup script does not arm teardown. A runnable scorecard recipe must package+hash the
frozen code/data/env and fail closed on all of the above.

# A100 micro-session plan — player selection frozen card

This is a dependency-packaged manager handoff, not an executed result. The local bundle
has only a static dependency dry-run; the GPU evaluation remains `NO-ATTEMPT` and
`VERIFIED=0`. Burlington and Wolverine are protected internal evaluation clips. Do not use
their labels for fitting, threshold changes, enrollment refreshes, or training. Run once at
the registered defaults and report misses as misses.

## 0. Local preflight and transfer bundle

Run from `/Users/arnavchokshi/Desktop/pickleball`. This reuses the score-faithful variant-P
protocol and frozen inputs from `trk_rfdetr_prod_20260716/vm_rerun`.

```bash
set -euo pipefail
LANE=runs/lanes/trkL_selection_impl_20260719
PREV=runs/lanes/trk_rfdetr_prod_20260716/vm_rerun
FEEDERS=runs/lanes/trk_detbench_20260716/scripts
mkdir -p "$LANE/vm_eval/local" "$LANE/vm_eval/pulled"

git branch --show-current | tee "$LANE/vm_eval/local/branch.txt"
test "$(git branch --show-current)" = main
.venv/bin/python -m pytest -q tests/racketsport/test_player_selection.py | tee "$LANE/vm_eval/local/focused.txt"
test -s "$FEEDERS/candidate_detector_feeder.py"
test -s "$FEEDERS/feeder_yolo26m_botsort_arm0b.py"

tar -czf "$LANE/vm_eval/local/selection_code.tar.gz" \
  threed/racketsport/player_selection.py \
  threed/racketsport/schemas/__init__.py \
  scripts/racketsport/select_players_from_pool.py \
  docs/racketsport/player_selection_report_schema.json \
  tests/racketsport/test_player_selection.py
cp "$PREV/scripts/vm_mission.sh" "$LANE/vm_eval/local/variant_p_vm_mission.sh"
cp "$FEEDERS/candidate_detector_feeder.py" "$LANE/vm_eval/local/"
cp "$FEEDERS/feeder_yolo26m_botsort_arm0b.py" "$LANE/vm_eval/local/"
cp "$PREV/p_inputs/"*.rfdetr_p.json "$LANE/vm_eval/local/"

(cd "$LANE/vm_eval/local" && shasum -a 256 * > LOCAL_SHA256.txt)
```

## 1. Provision exactly one spot A100 and arm teardown first

Use the prior successful snapshot and A100 ladder. Check the fleet before creating anything;
do not disturb another lane's VM. The budget cap is `$2`, with a 45-minute hard rail.

```bash
set -euo pipefail
PROJECT=gifted-electron-498923-h1
VM=pickleball-gpu-trkl-selection
ZONE=asia-southeast1-c
SNAP=projects/$PROJECT/global/snapshots/pickleball-fleet-snap-20260709-w7close
gcloud compute instances list --project "$PROJECT" \
  --filter='labels.fable-fleet=pickleball' \
  --format='table(name,zone,status,labels.fable-lane)'

gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
  --machine-type=a2-ultragpu-1g --provisioning-model=SPOT \
  --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --create-disk=auto-delete=yes,boot=yes,device-name="$VM",mode=rw,size=200,type=pd-balanced,source-snapshot="$SNAP" \
  --labels=fable-lane=trkL-selection-eval-20260719,fable-fleet=pickleball,owner=arnavchokshi \
  --metadata-from-file=startup-script=scripts/fleet/lane_vm_startup.sh

gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command \
  'sudo shutdown -P +45 && cat /run/systemd/shutdown/scheduled && nvidia-smi'
```

If `asia-southeast1-c` is out of stock, stop and use the existing provision ladder's next
A100-80 zone; do not change GPU class after the env-fidelity gate starts.

Transfer and prove both sides:

```bash
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command \
  'mkdir -p /tmp/trkl_selection_payload/local /tmp/p_inputs'
gcloud compute scp --project "$PROJECT" --zone "$ZONE" \
  "$LANE/vm_eval/local/"* "$VM:/tmp/trkl_selection_payload/local/"
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command '
  set -euo pipefail
  cd /tmp/trkl_selection_payload/local
  sha256sum -c LOCAL_SHA256.txt
  cd ~/coldstart_20260706/repo
  tar -xzf /tmp/trkl_selection_payload/local/selection_code.tar.gz
  cp /tmp/trkl_selection_payload/local/*.rfdetr_p.json /tmp/p_inputs/
  cp /tmp/trkl_selection_payload/local/candidate_detector_feeder.py /tmp/
  cp /tmp/trkl_selection_payload/local/feeder_yolo26m_botsort_arm0b.py /tmp/
  cp /tmp/trkl_selection_payload/local/variant_p_vm_mission.sh /tmp/variant_p_vm_mission.sh
  chmod +x /tmp/variant_p_vm_mission.sh \
    /tmp/candidate_detector_feeder.py \
    /tmp/feeder_yolo26m_botsort_arm0b.py
  test -s /tmp/p_inputs/burlington_gold_0300_low_steep_corner.rfdetr_p.json
  test -s /tmp/p_inputs/wolverine_mixed_0200_mid_steep_corner.rfdetr_p.json
  test -s /tmp/candidate_detector_feeder.py
  test -s /tmp/feeder_yolo26m_botsort_arm0b.py
  test -s models/checkpoints/osnet_x1_0_market1501.pt
  test -s configs/racketsport/botsort_no_reid_loose.yaml
  test -s scripts/racketsport/run_raw_pool_person_authority.py
  test -s scripts/racketsport/score_person_track_sources.py
  test -d runs/lanes/trk_flip_20260713/frozen_gt
  for clip in burlington_gold_0300_low_steep_corner wolverine_mixed_0200_mid_steep_corner; do
    test -s "eval_clips/ball/$clip/source.mp4"
    test -s "runs/lanes/trk_flip_20260713/preflip_production/$clip/court_calibration.json"
  done
  bash -n /tmp/variant_p_vm_mission.sh
  .venv/bin/python3 -c "import cv2, numpy, torch, torchreid, ultralytics"
  .venv/bin/python3 /tmp/candidate_detector_feeder.py --help >/dev/null
  .venv/bin/python -m pytest -q tests/racketsport/test_player_selection.py
'
```

## 2. Mandatory environment-fidelity gate — unmodified variant P first

Rebuild both variant-P rows through the unmodified association/scoring path before running
selection. Stop immediately if any listed scalar differs by more than `1e-9`.

```bash
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command '
  set -euo pipefail
  cd ~/coldstart_20260706/repo
  rm -rf rfdetrflip_out
  bash /tmp/variant_p_vm_mission.sh pool_p
  bash /tmp/variant_p_vm_mission.sh assoc_p burlington_gold_0300_low_steep_corner
  bash /tmp/variant_p_vm_mission.sh assoc_p wolverine_mixed_0200_mid_steep_corner
  bash /tmp/variant_p_vm_mission.sh score env_fidelity
  .venv/bin/python - <<"PY"
import json
from pathlib import Path

path = Path("rfdetrflip_out/scores/env_fidelity/person_track_gt_scoring_report.json")
report = json.loads(path.read_text())
want = {
    "burlington_gold_0300_low_steep_corner": {
        "idf1": 0.9220183486238532,
        "four_player_coverage": 0.9933333333333333,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "far_off_court_false_positive_frames": 0,
    },
    "wolverine_mixed_0200_mid_steep_corner": {
        "idf1": 0.8036253776435045,
        "four_player_coverage": 0.7233333333333334,
        "id_switches": 1,
        "true_spectator_or_background_false_positives": 4,
        "far_off_court_false_positive_frames": 0,
    },
}
source = next(item for item in report["sources"] if item["track_source_id"].endswith("/rfdetr_l_p"))
rows = {row["clip_id"]: row for row in source["rows"]}
for clip, expected in want.items():
    for key, target in expected.items():
        actual = rows[clip][key]
        if abs(float(actual) - float(target)) > 1e-9:
            raise SystemExit(f"ENV_FIDELITY_FAIL {clip} {key}: {actual} vs {target}")
print("ENV_FIDELITY_PASS_WITHIN_1E-9")
PY
'
```

## 3. Run selection once on both clips, then score once

No threshold flags exist: the frozen dataclass defaults are the only operating point.

```bash
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command '
  set -euo pipefail
  cd ~/coldstart_20260706/repo
  V=.venv/bin/python
  OUT=rfdetrflip_out
  AUTH=$OUT/selection_authoritative
  SEL=$OUT/selection_scored_projection
  rm -rf "$AUTH" "$SEL" "$OUT/scores/selection"
  for clip in burlington_gold_0300_low_steep_corner wolverine_mixed_0200_mid_steep_corner; do
    IN=$OUT/scored/$clip/rfdetr_l_p
    POOL=$OUT/pools/rfdetr_p/$clip
    A=$AUTH/$clip/rfdetr_l_p_selection
    O=$SEL/$clip/rfdetr_l_p_selection
    mkdir -p "$A" "$O"
    "$V" scripts/racketsport/select_players_from_pool.py \
      --enable-selection \
      --tracks "$IN/tracks.json" \
      --raw-pool "$POOL/tracked_detections.json" \
      --embeddings "$IN/reid_embeddings.json" \
      --calibration "$IN/court_calibration.json" \
      --out-tracks "$A/tracks.json" \
      --report "$A/player_selection_report.json"
    "$V" - "$A/tracks.json" "$O/tracks.json" <<"PY"
import hashlib, json, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
payload = json.loads(source.read_text())
for player in payload["players"]:
    for frame in player["frames"]:
        frame.pop("interpolated", None)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
attestation = {
    "artifact_type": "racketsport_player_selection_scoring_projection",
    "authoritative_tracks": str(source),
    "authoritative_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "projection_tracks": str(target),
    "projection_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "only_removed_field": "players[].frames[].interpolated",
    "reason": "frozen scorer uses the pre-integration strict TrackFrame schema",
    "not_product_output": True,
}
(target.parent / "SCORING_PROJECTION.json").write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
PY
    cp "$IN/metrics.json" "$IN/court_calibration.json" "$O/"
  done
  "$V" scripts/racketsport/score_person_track_sources.py \
    --cvat-root runs/lanes/trk_flip_20260713/frozen_gt \
    --runs-root "$SEL" --out-dir "$OUT/scores/selection" \
    --iou-threshold 0.5 --expected-players 4
  find "$OUT/selection_authoritative" "$OUT/selection_scored_projection" \
    "$OUT/scores/env_fidelity" "$OUT/scores/selection" \
    -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/trkl_selection_ALL_SHA256.txt"
  tar -czf /tmp/trkl_selection_pull.tar.gz \
    "$OUT/selection_authoritative" "$OUT/selection_scored_projection" \
    "$OUT/scores/env_fidelity" "$OUT/scores/selection" \
    "$OUT/trkl_selection_ALL_SHA256.txt"
  sha256sum /tmp/trkl_selection_pull.tar.gz
'
```

## 4. Pull with two-sided SHA-256, verify, and tear down

```bash
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command \
  'sha256sum /tmp/trkl_selection_pull.tar.gz' | tee "$LANE/vm_eval/pulled/REMOTE_TARBALL_SHA256.txt"
gcloud compute scp --project "$PROJECT" --zone "$ZONE" \
  "$VM:/tmp/trkl_selection_pull.tar.gz" "$LANE/vm_eval/pulled/"
(cd "$LANE/vm_eval/pulled" && shasum -a 256 trkl_selection_pull.tar.gz) \
  | tee "$LANE/vm_eval/pulled/LOCAL_TARBALL_SHA256.txt"
diff <(awk '{print $1}' "$LANE/vm_eval/pulled/REMOTE_TARBALL_SHA256.txt") \
     <(awk '{print $1}' "$LANE/vm_eval/pulled/LOCAL_TARBALL_SHA256.txt")
tar -xzf "$LANE/vm_eval/pulled/trkl_selection_pull.tar.gz" -C "$LANE/vm_eval/pulled"
(cd "$LANE/vm_eval/pulled" && sha256sum -c rfdetrflip_out/trkl_selection_ALL_SHA256.txt)

gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet
gcloud compute instances list --project "$PROJECT" --filter="name=$VM" --format='value(name,status)'
gcloud compute disks list --project "$PROJECT" --filter="name=$VM" --format='value(name,status)'
```

Both final list commands must print zero rows. Record VM wall time and cost before ruling.

## 5. Pre-registered acceptance table (verbatim)

| clip | axis | variant P (baseline to beat) | FULL PASS bar | notes |
|---|---|---|---|---|
| wolverine | spectFP | 4 | **0** | hard |
| wolverine | switches | 1 | **0** | hard |
| wolverine | far-off-court FP | 0 | 0 | hard |
| wolverine | near-miss rate | 0.1244 (breach) | ≤ 0.10 | CF predicts 0.0986 |
| wolverine | IDF1 | 0.8036 | ≥ 0.8036; target ≥ 0.8516 | CF3 upper bound 0.8519 |
| wolverine | cov4 | 0.7233 (contains ~0.107 synthetic padding) | ≥ 0.7233 via layer-C recovery | 0.6167 ≤ cov4 < 0.7233 with all other axes green = PARTIAL — coordinator ruling, stated verbatim as recovery shortfall |
| burlington | all axes | 0.9220 / 0.9933 / 0 / 0 / 0 | no degradation: IDF1 ≥ 0.9200, cov4 ≥ 0.9917 (≤1 frame), FP axes stay 0 | veto rules structurally cannot fire (6 synth frames, max run 3 / 1.14 m) |

FULL PASS requires every row. Do not tune, rerun with changed values, or use Outdoor/Indoor.
Only FULL PASS permits the manager to draft a preview/default flip proposal; this lane itself
makes no best-stack change.
