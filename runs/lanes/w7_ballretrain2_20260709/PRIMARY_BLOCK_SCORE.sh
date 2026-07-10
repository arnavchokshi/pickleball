#!/usr/bin/env bash
# PRIMARY curve-read scoring block for w7_ballretrain2_20260709.
# Adapted VERBATIM (same CORPUS_ROOT, same 20-clip list, same FPS map, same harness flags)
# from runs/lanes/w6_labelingest_20260708/GPU_RESCORE_COMMANDS.sh -- the SAME block the
# 1k-checkpoint point (w7_ballretrain_20260709) used -- per manager ruling: "the measuring
# stick must not move." Candidates swapped to official_tennis_control + A_seed_official_aug
# (banked from w7_ballretrain) + E3k_seed_official_aug (this lane, trained on the 3,026-row
# corpus). Run on the GPU checkout.
set -euo pipefail

REPO="${REPO:-/home/arnavchokshi/coldstart_20260706/repo}"
cd "$REPO"

LANE_ROOT="runs/lanes/w7_ballretrain2_20260709"
CORPUS_ROOT="runs/lanes/w6_labelingest_20260708/reviewed_corpus"
OUT_ROOT="$LANE_ROOT/primary_block_score"
WASB_REPO="third_party/WASB-SBDT"

declare -A CANDIDATE_CKPT=(
  [official_tennis_control]="models/checkpoints/wasb/wasb_tennis_best.pth.tar"
  [A_seed_official_aug]="runs/lanes/w7_ballretrain_20260709/arm3_finetunes/A_seed_official_aug/checkpoints/latest.pt"
  [E3k_seed_official_aug]="$LANE_ROOT/arm_finetunes/E3k_seed_official_aug/checkpoints/latest.pt"
)

CANDIDATES=(
  official_tennis_control
  A_seed_official_aug
  E3k_seed_official_aug
)

# Identical to w6_labelingest_20260708/GPU_RESCORE_COMMANDS.sh CLIPS list (the 1121-row,
# 20-clip corpus the 1k point was measured on).
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
  wBu8bC4OfUY_rally_0001
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
      --device cuda \
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
