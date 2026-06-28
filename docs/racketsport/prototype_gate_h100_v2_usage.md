# Prototype Gate H100 v2 Usage

Date: 2026-06-28

This is the current runnable prototype path for the accepted-four pickleball
clips. It is set up for qualitative review and held-out benchmark comparison.
It is not a production BALL gate and does not make BALL `VERIFIED`.

## Accepted Clips

Use only these clips for the current prototype gate:

- `burlington_gold_0300_low_steep_corner`
- `wolverine_mixed_0200_mid_steep_corner`
- `outdoor_webcam_iynbd_1500_long_high_baseline`
- `indoor_doubles_fwuks_0500_long_mid_baseline`

Do not use `side_view_game5_0100_high_side_fence` for this gate. It is
`DEFERRED_REJECTED_SIDE_FISHEYE`.

## Current Usable Ball Artifact

For each accepted clip, the current strict no-click review track is:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json
```

Its watchable overlay is:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4
```

This track is built from:

- target-court-filtered TrackNetV3 output
- the motion-consistent TrackNet temporal path
- VballNet Fast and VballNet V1 verifier tracks
- a local trajectory-consistency filter that hides points far away from the
  surrounding before/after path

It does not read `ball_points.json`, click-corrected tracks, or any human-click
oracle output. The human clicks are held-out benchmark labels only.

## Current Benchmark Read

Latest benchmark summary:

```text
runs/eval0/prototype_gate_h100_v2/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.md
```

Aggregate result on the four accepted clips:

| candidate | hit recall | p90 error px | hidden FP | teleports | score |
|---|---:|---:|---:|---:|---:|
| `fusion_temporal_vball100` | 0.563 | 46.784 | 0.425 | 8 | 0.235 |
| `fusion_temporal_vball100_localtraj` | 0.509 | 38.851 | 0.294 | 0 | 0.280 |
| `tracknet_court_temporal_path` | 0.415 | 35.043 | 0.325 | 5 | 0.169 |
| `tracknet_raw` | 0.694 | 431.866 | 0.950 | 71 | -0.243 |

Interpretation: `fusion_temporal_vball100_localtraj` is the current best strict
review artifact because it removes the remaining teleport-style jumps and lowers
hidden false positives. It hides more uncertain ball frames than the looser
fusion track, so use `fusion_temporal_vball100` as the recall comparison track.

## Rebuild The Current Review Tracks

Run from repo root. The same commands work locally for existing artifacts and on
the H100 under `/workspace/pickleball` after `git pull --ff-only`.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  base="$RUN_ROOT/$clip/tracknet_smoke_0000_0010"

  python scripts/racketsport/fuse_ball_tracks.py \
    --primary-ball-track "$base/ball_track_target_court_120px.json" \
    --stable-ball-track "$base/ball_track_target_court_temporal.json" \
    --verifier-ball-track "$base/vballnet_fast/ball_track.json" \
    --verifier-ball-track "$base/vballnet_v1/ball_track.json" \
    --outlier-distance-px 100 \
    --out "$base/ball_track_fusion_temporal_vball100.json" \
    --summary-out "$base/ball_track_fusion_temporal_vball100_summary.json"

  python scripts/racketsport/filter_ball_temporal.py \
    --mode local_trajectory \
    --ball-track "$base/ball_track_fusion_temporal_vball100.json" \
    --local-trajectory-window-frames 20 \
    --local-trajectory-max-error-px 80 \
    --local-trajectory-min-pair-predictions 4 \
    --max-iterations 3 \
    --out "$base/ball_track_fusion_temporal_vball100_localtraj.json" \
    --summary-out "$base/ball_track_fusion_temporal_vball100_localtraj_summary.json"

  python scripts/racketsport/render_ball_track_overlay.py \
    --video "$base/input_0000_0010.mp4" \
    --ball-track "$base/ball_track_fusion_temporal_vball100_localtraj.json" \
    --out "$base/ball_track_fusion_temporal_vball100_localtraj_overlay.mp4"

  ffmpeg -y \
    -i "$base/ball_track_fusion_temporal_vball100_localtraj_overlay.mp4" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$base/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"
done
```

## Rerun The Held-Out Benchmark

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2

python scripts/racketsport/benchmark_ball_trackers.py \
  --run-root "$RUN_ROOT" \
  --review-root "$RUN_ROOT/ball_click_review_30" \
  --clip burlington_gold_0300_low_steep_corner \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --clip outdoor_webcam_iynbd_1500_long_high_baseline \
  --clip indoor_doubles_fwuks_0500_long_mid_baseline \
  --candidate "tracknet_raw=tracknet_smoke_0000_0010/ball_track_0000_0010.json" \
  --candidate "tracknet_court_temporal_path=tracknet_smoke_0000_0010/ball_track_target_court_temporal.json" \
  --candidate "fusion_temporal_vball100:generalizable_two_model_fusion=tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100.json" \
  --candidate "fusion_temporal_vball100_localtraj:generalizable_two_model_fusion=tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json" \
  --out-json "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.json" \
  --out-md "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.md"
```

## Sync And Verify On H100

```bash
gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && git pull --ff-only'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && /opt/conda/envs/fast_sam_3d_body/bin/python -m pytest -q -p no:cacheprovider tests/racketsport/test_ball_temporal_filter.py tests/racketsport/test_ball_benchmark.py tests/racketsport/test_ball_model_fusion.py'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
```

Expected focused test result at the current commit: `9 passed`.

## If The Output Looks Wrong

- Background balls are selected: check the court calibration overlay first, then
  rerun the target-court filter before fusion.
- The ball jumps to a far-away false detection: use the local-trajectory track
  above, or lower `--local-trajectory-max-error-px` for a stricter pass.
- The real ball disappears too often: compare against
  `ball_track_fusion_temporal_vball100.json`; if that is better, raise
  `--local-trajectory-max-error-px` or keep both artifacts for review.
- Court lines or paddle flashes are selected as the ball: this needs better
  model-side candidate generation, not just post-processing.

## Explicit Next Steps

1. Build a real BALL StageRunner that runs the best no-click track path through
   the orchestrator and writes the BALL contract artifacts fail-closed.
2. Improve candidate generation, not just filtering: obtain usable TrackNetV4
   weights or fine-tune TrackNetV3/V4/V5 on pickleball clips with neighboring
   courts, small balls, occlusions, and high-baseline viewpoints.
3. Extend from 10-second smoke windows to longer windows, then full clips, using
   the same benchmark command and keeping human clicks held out.
4. Add event/contact outputs only after the ball path is stable enough: audio
   pop, wrist-velocity peak, ball inflection, and doubles attribution.
5. Replace the prototype benchmark with the full BALL acceptance gate: ball F1
   at least 0.90, false positives below 5%, and contact timing within plus/minus
   2 frames on a representative labeled set.
6. Keep `BALL-1`, `BALL-3`, and `BALL-4` unverified until the real gate passes
   through the spine on real clips.
