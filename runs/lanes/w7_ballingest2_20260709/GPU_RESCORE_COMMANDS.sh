#!/usr/bin/env bash
# Prepare-only GPU re-score for w7_ballingest2_20260709.
# Run on the GPU checkout after the rebuilt corpus artifacts are present.
# This scores INTERNAL-val reviewed corpus clips only; it does not read held-out labels.
#
# Optional w7 retrain winners can be appended without editing this file:
#   W7_EXTRA_CANDIDATES="w7_candidate=/path/to/checkpoint.pt w7_winner=/path/to/winner.pt" bash GPU_RESCORE_COMMANDS.sh
set -euo pipefail

REPO="${REPO:-/home/arnavchokshi/coldstart_20260706/repo}"
cd "$REPO"

LANE_ROOT="${LANE_ROOT:-runs/lanes/w7_ballingest2_20260709}"
CORPUS_ROOT="${CORPUS_ROOT:-$LANE_ROOT/reviewed_corpus}"
OUT_ROOT="${OUT_ROOT:-$LANE_ROOT/gpu_rescore}"
WASB_REPO="${WASB_REPO:-third_party/WASB-SBDT}"
DEVICE="${DEVICE:-cuda}"

declare -A CANDIDATE_CKPT=(
  [official_tennis_control]="${OFFICIAL_TENNIS_CONTROL_CKPT:-models/checkpoints/wasb/wasb_tennis_best.pth.tar}"
  [stage1_official]="${STAGE1_OFFICIAL_CKPT:-runs/lanes/w5_ballretrain_20260707/stage1_official/checkpoints/latest.pt}"
  [seed_official]="${SEED_OFFICIAL_CKPT:-runs/lanes/w5_ballretrain_20260707/seed_official/checkpoints/latest.pt}"
)

CANDIDATES=(
  official_tennis_control
  stage1_official
  seed_official
)

if [[ -n "${W7_EXTRA_CANDIDATES:-}" ]]; then
  for spec in ${W7_EXTRA_CANDIDATES}; do
    if [[ "$spec" != *=* ]]; then
      echo "W7_EXTRA_CANDIDATES entries must be name=/checkpoint/path, got: $spec" >&2
      exit 2
    fi
    name="${spec%%=*}"
    checkpoint="${spec#*=}"
    CANDIDATE_CKPT["$name"]="$checkpoint"
    CANDIDATES+=("$name")
  done
fi

CLIPS=(
  73VurrTKCZ8_rally_0001
  73VurrTKCZ8_rally_0002
  73VurrTKCZ8_rally_0003
  73VurrTKCZ8_rally_0004
  73VurrTKCZ8_rally_0005
  73VurrTKCZ8_rally_0006
  73VurrTKCZ8_rally_0007
  73VurrTKCZ8_rally_0008
  Ezz6HDNHlnk_rally_0001
  Ezz6HDNHlnk_rally_0002
  Ezz6HDNHlnk_rally_0003
  Ezz6HDNHlnk_rally_0004
  Ezz6HDNHlnk_rally_0005
  Ezz6HDNHlnk_rally_0006
  Ezz6HDNHlnk_rally_0007
  Ezz6HDNHlnk_rally_0008
  HyUqT7zFiwk_rally_0001
  _L0HVmAlCQI_rally_0001
  _L0HVmAlCQI_rally_0002
  _L0HVmAlCQI_rally_0003
  _L0HVmAlCQI_rally_0005
  _L0HVmAlCQI_rally_0007
  _L0HVmAlCQI_rally_0008
  _L0HVmAlCQI_rally_0009
  _L0HVmAlCQI_rally_0010
  _L0HVmAlCQI_rally_0011
  _L0HVmAlCQI_rally_0012
  _L0HVmAlCQI_rally_0013
  _L0HVmAlCQI_rally_0014
  _L0HVmAlCQI_rally_0015
  _L0HVmAlCQI_rally_0016
  _L0HVmAlCQI_rally_0017
  _L0HVmAlCQI_rally_0018
  _L0HVmAlCQI_rally_0019
  wBu8bC4OfUY_rally_0001
  wBu8bC4OfUY_rally_0002
  wBu8bC4OfUY_rally_0003
  zwCtH_i1_S4_rally_0001
)

declare -A FPS_BY_SOURCE=(
  [73VurrTKCZ8]="30"
  [Ezz6HDNHlnk]="23.98"
  [HyUqT7zFiwk]="30"
  [_L0HVmAlCQI]="30"
  [wBu8bC4OfUY]="30"
  [zwCtH_i1_S4]="30"
)

for candidate in "${CANDIDATES[@]}"; do
  checkpoint="${CANDIDATE_CKPT[$candidate]}"
  test -f "$checkpoint"
  for clip in "${CLIPS[@]}"; do
    source_id="${clip%%_rally_*}"
    video="data/online_harvest_20260706/rallies/${source_id}/${clip}.mp4"
    fps="${FPS_BY_SOURCE[$source_id]}"
    out_dir="$OUT_ROOT/$candidate/$clip/wasb"
    test -f "$video"
    mkdir -p "$out_dir"
    .venv/bin/python scripts/racketsport/run_wasb_ball.py \
      --checkpoint "$checkpoint" \
      --wasb-repo "$WASB_REPO" \
      --video "$video" \
      --fps "$fps" \
      --candidate-top-k 5 \
      --visible-threshold 0.5 \
      --input-preprocessing official \
      --device "$DEVICE" \
      --out "$out_dir/ball_track.json" \
      --metadata-out "$out_dir/ball_track_metadata.json"
  done
done

loso_args=()
for candidate in "${CANDIDATES[@]}"; do
  for clip in "${CLIPS[@]}"; do
    loso_args+=(--candidate-track "${candidate}=${clip}=${OUT_ROOT}/${candidate}/${clip}/wasb/ball_track.json")
  done
done

.venv/bin/python scripts/racketsport/ball_loso_validation.py \
  --cvat-root "$CORPUS_ROOT" \
  --out-dir "$OUT_ROOT/loso" \
  "${loso_args[@]}"
