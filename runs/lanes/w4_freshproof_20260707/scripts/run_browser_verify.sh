#!/bin/bash
# Browser verify for w4_freshproof: verify_process_video_viewer.py against each pulled
# replay_viewer_manifest.json (NOT PIPELINE_SUMMARY.json), run locally on the Mac from the MAIN repo
# (Vite serves repo root; manifests were built with --vite-allow-root = main repo).
set -uo pipefail
REPO=/Users/arnavchokshi/Desktop/pickleball
LANE="$REPO/runs/lanes/w4_freshproof_20260707"
cd "$REPO"
for pair in "outdoor outdoor_webcam_iynbd_1500_long_high_baseline" "burlington burlington_gold_0300_low_steep_corner" "wolverine wolverine_mixed_0200_mid_steep_corner" "img1605 owner_IMG_1605_8a193402780b"; do
  short=$(echo "$pair" | cut -d' ' -f1); clip=$(echo "$pair" | cut -d' ' -f2)
  M="$LANE/$short/$clip/$clip/replay_viewer_manifest.json"
  O="$LANE/browser_verify/$short"
  mkdir -p "$O"
  echo "=== browser verify $short start $(date -u +%FT%TZ) ==="
  if [ ! -f "$M" ]; then echo "$short: MANIFEST MISSING at $M"; continue; fi
  python3 scripts/racketsport/verify_process_video_viewer.py --manifest "$M" --out-dir "$O" > "$O/verify_stdout.log" 2>&1
  echo "$short exit=$?"
  tail -3 "$O/verify_stdout.log"
done
echo "=== browser verify done $(date -u +%FT%TZ) ==="
