#!/usr/bin/env bash
# NEW-BLOCK (3,026-row, 40-clip, mixed-provenance) scoring for w7_ballretrain2_20260709.
# Per manager ruling: report this block SPLIT three ways (full / clean / confirmed-prelabel)
# via classify_provenance.py, labeled "mixed-provenance, non-comparable absolutes,
# ordering-only". Candidates: official_tennis_control (fresh re-run -- prior control
# ball_track.json files lived on the now-deleted first VM instance and were not preserved)
# + E3k_seed_official_aug.
set -euo pipefail

REPO="${REPO:-/home/arnavchokshi/coldstart_20260706/repo}"
cd "$REPO"

LANE_ROOT="runs/lanes/w7_ballretrain2_20260709"
CORPUS_ROOT="runs/lanes/w7_ballingest4_20260709/reviewed_corpus"
OUT_ROOT="$LANE_ROOT/new_block_score"
WASB_REPO="third_party/WASB-SBDT"

declare -A CANDIDATE_CKPT=(
  [official_tennis_control]="models/checkpoints/wasb/wasb_tennis_best.pth.tar"
  [E3k_matched_seed_official_aug]="$LANE_ROOT/arm_finetunes/E3k_matched_seed_official_aug/checkpoints/latest.pt"
)

CANDIDATES=(
  official_tennis_control
  E3k_matched_seed_official_aug
)

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
  _L0HVmAlCQI_rally_0004
  _L0HVmAlCQI_rally_0005
  _L0HVmAlCQI_rally_0006
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
