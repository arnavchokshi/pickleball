#!/usr/bin/env bash
# vm_rerun mission script — runs ON the VM after bootstrap. Assumes:
#  - repo at ~/coldstart_20260706/repo pinned to cabdb8cf, payload tar unpacked (frozen pools/GT/OSNet)
#  - torchreid installed in .venv
#  - /tmp/p_inputs/*.rfdetr_p.json + /tmp/candidate_detector_feeder.py + /tmp/feeder_yolo26m_botsort_arm0b.py present
# Usage: bash vm_mission.sh <step>   steps: arm0a | pool_p | assoc_p | score <tag> | mech_confirm
set -euo pipefail
R=~/coldstart_20260706/repo
V=$R/.venv/bin
cd $R
CLIPS="burlington_gold_0300_low_steep_corner wolverine_mixed_0200_mid_steep_corner"
OUT=rfdetrflip_out
mkdir -p $OUT/scored $OUT/scores $OUT/pools

step="${1:?step}"

case "$step" in
arm0a)
  for clip in $CLIPS; do
    POOL=runs/lanes/trk_flip_20260713/preflip_production/$clip
    O=$OUT/scored/$clip/arm0a_repro
    $V/python3 scripts/racketsport/run_raw_pool_person_authority.py \
      --clip-id $clip --candidate arm0a_repro \
      --video eval_clips/ball/$clip/source.mp4 \
      --raw-pool-dir $POOL --calibration $POOL/court_calibration.json \
      --out-dir $O \
      --reid-model models/checkpoints/osnet_x1_0_market1501.pt \
      --reid-backend osnet --court-margin-m 1.0 --expected-players 4 2>&1 | tail -2
    cp $POOL/metrics.json $POOL/court_calibration.json $O/
    touch /tmp/lane_heartbeat
  done
  ;;
pool_p)
  for clip in $CLIPS; do
    $V/python3 /tmp/candidate_detector_feeder.py \
      --detector raw-json --raw-json /tmp/p_inputs/$clip.rfdetr_p.json \
      --video eval_clips/ball/$clip/source.mp4 --clip-id $clip \
      --out-dir $OUT/pools/rfdetr_p/$clip \
      --tracker-yaml configs/racketsport/botsort_no_reid_loose.yaml \
      --person-class-id 1 --conf 0.18
    touch /tmp/lane_heartbeat
  done
  ;;
assoc_p)
  clip="${2:?clip}"
  CAL=runs/lanes/trk_flip_20260713/preflip_production/$clip/court_calibration.json
  O=$OUT/scored/$clip/rfdetr_l_p
  $V/python3 scripts/racketsport/run_raw_pool_person_authority.py \
    --clip-id $clip --candidate rfdetr_l_p \
    --video eval_clips/ball/$clip/source.mp4 \
    --raw-pool-dir $OUT/pools/rfdetr_p/$clip \
    --calibration $CAL --out-dir $O \
    --reid-model models/checkpoints/osnet_x1_0_market1501.pt \
    --reid-backend osnet --court-margin-m 1.0 --expected-players 4 2>&1 | tail -2
  cp $OUT/pools/rfdetr_p/$clip/metrics.json $O/
  cp $CAL $O/
  touch /tmp/lane_heartbeat
  ;;
score)
  tag="${2:?tag}"
  $V/python3 scripts/racketsport/score_person_track_sources.py \
    --cvat-root runs/lanes/trk_flip_20260713/frozen_gt \
    --runs-root $OUT/scored \
    --out-dir $OUT/scores/$tag \
    --iou-threshold 0.5 --expected-players 4
  touch /tmp/lane_heartbeat
  ;;
mech_confirm)
  # YOLO26m at the POOLDIAG-confirmed generator operating point (conf .18, imgsz 960)
  for clip in $CLIPS; do
    $V/python3 /tmp/feeder_yolo26m_botsort_arm0b.py \
      --video eval_clips/ball/$clip/source.mp4 \
      --weights models/checkpoints/yolo26m.pt \
      --tracker-yaml configs/racketsport/botsort_no_reid_loose.yaml \
      --out-dir $OUT/pools/yolo_mech018_960/$clip \
      --clip-id $clip --conf 0.18 --imgsz 960
    CAL=runs/lanes/trk_flip_20260713/preflip_production/$clip/court_calibration.json
    O=$OUT/scored/$clip/yolo_mech018_960
    $V/python3 scripts/racketsport/run_raw_pool_person_authority.py \
      --clip-id $clip --candidate yolo_mech018_960 \
      --video eval_clips/ball/$clip/source.mp4 \
      --raw-pool-dir $OUT/pools/yolo_mech018_960/$clip \
      --calibration $CAL --out-dir $O \
      --reid-model models/checkpoints/osnet_x1_0_market1501.pt \
      --reid-backend osnet --court-margin-m 1.0 --expected-players 4 2>&1 | tail -2
    cp $OUT/pools/yolo_mech018_960/$clip/metrics.json $O/
    cp $CAL $O/
    touch /tmp/lane_heartbeat
  done
  ;;
*) echo "unknown step $step"; exit 2;;
esac
echo "STEP_${step}_DONE"
