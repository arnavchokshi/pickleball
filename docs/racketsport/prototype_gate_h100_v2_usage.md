# Prototype Gate H100 v2 Runbook

Last audited: 2026-06-29

This is the current runnable prototype path for the accepted-four pickleball
clips. It exists for qualitative review, held-out benchmark comparison, and
artifact-readiness debugging. It is not a production gate and does not make any
pipeline phase `VERIFIED`.

For product/status truth, prefer the top truth snapshot in `BUILD_CHECKLIST.md`
and the capability matrix in `CAPABILITIES.md`. This file should not duplicate
mutable artifact counts beyond stable paths and commands.

## Accepted Clips

Use only these clips for this prototype gate:

- `burlington_gold_0300_low_steep_corner`
- `wolverine_mixed_0200_mid_steep_corner`
- `outdoor_webcam_iynbd_1500_long_high_baseline`
- `indoor_doubles_fwuks_0500_long_mid_baseline`

Do not use `side_view_game5_0100_high_side_fence`; it is
`DEFERRED_REJECTED_SIDE_FISHEYE`.

Burlington is retired for court calibration because fisheye curvature bends the
court lines. It remains useful for BODY/player/ball/paddle smoke and review,
but it must not prove no-tap or line-based court calibration.

## Current Truth

- **CAL:** manual/review calibration artifacts exist; automatic court evidence
  remains fail-closed unless trusted line/top-net evidence is present.
- **TRK:** player-track overlays and candidates are qualitative review evidence.
  Tracking is not verified until IDF1, spectator rejection, ID-switch, and
  throughput gates pass on labeled clips.
- **BALL:** the strict no-click review track is a prototype review artifact.
  Human clicks are held-out benchmark labels only. BALL remains unverified until
  real ball/contact gates pass through the spine.
- **BODY:** limited scheduled H100 BODY smoke outputs exist for some clips, but
  BODY remains blocked without world-MPJPE, full-window coverage, and related
  BODY gate evidence. Outdoor currently remains missing BODY mesh output.
- **RKT:** box-derived paddle candidates are preview-only. They must not promote
  into canonical `racket_pose.json`; RKT needs true paddle corners, CAD/reference
  pose, or ArUco/AprilTag/reference labels plus Phase 6 evaluation.
- **RPL:** static review GLBs and browser review pages are inspection aids only.
  Production animated GLB/USDZ export and replay validation are still missing.
- **E2E:** `pipeline_readiness_e2e.json` reports artifact plus semantic blockers.
  It is not an accuracy or performance gate and can lag regenerated artifacts.

## Review Entrypoints

If an accepted-four CVAT-labeled dataset is being produced, use that export as
the preferred reviewed-label source for player boxes/tracks/IDs and downstream
TRK/BODY gates. The localhost correction UI below remains useful for quick
triage, but it is not the next required user action in that case. Likewise, if
the separate racket annotation workflow is producing true paddle-face corners,
masks/keypoints, CAD/reference pose, or ArUco/reference labels, consume that
export instead of duplicating paddle review in this runbook.

Main generated packet:

```text
runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review.html
```

Serve from repo root so packet links and Three.js pages can load local assets:

```bash
python -m http.server 8878 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8878/runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review.html
```

Direct correction UI:

```bash
python scripts/racketsport/review_input_server.py --port 8765
```

Then open `http://127.0.0.1:8765`. Save review inputs to
`runs/review_inputs/pickleball_cv_review_latest.json` and export them through
the corrections/contact-window scripts below.

Strict no-click ball review track per clip:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json
```

Strict no-click overlay per clip:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4
```

Use `ball_track_fusion_temporal_vball100.json` only as the looser recall
comparison track. Neither artifact reads `ball_points.json` or click-corrected
tracks.

## Common Variables

Run commands from the repo root.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)
```

## Rebuild Order

Use the builders below rather than hand-editing generated artifacts. Rebuild the
review packet after changing clip artifacts.

### Ball Review Track

For each clip, rebuild in this order:

```bash
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

ffmpeg -y -i "$base/ball_track_fusion_temporal_vball100_localtraj_overlay.mp4" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$base/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"
```

### Contact Windows

Machine cue fusion requires all three cue families:

- `audio_onsets.json`
- `wrist_velocity_peaks.json`
- `ball_inflections.json`

Build or refresh them with:

```bash
python scripts/racketsport/build_ball_inflections.py \
  --virtual-world "$RUN_ROOT/$clip/virtual_world.json" \
  --out "$RUN_ROOT/$clip/ball_inflections.json"

python scripts/racketsport/build_audio_onsets.py \
  --input "$RUN_ROOT/$clip/tracknet_smoke_0000_0010/input_0000_0010.mp4" \
  --out "$RUN_ROOT/$clip/audio_onsets.json" \
  --clip "$clip" \
  --start-s 0 \
  --duration-s 10 \
  --analysis-sample-rate-hz 16000

python scripts/racketsport/build_wrist_velocity_peaks.py \
  --skeleton3d "$RUN_ROOT/$clip/skeleton3d.json" \
  --out "$RUN_ROOT/$clip/wrist_velocity_peaks.json" \
  --allow-missing

python scripts/racketsport/build_contact_windows_from_cues.py \
  --audio-onsets "$RUN_ROOT/$clip/audio_onsets.json" \
  --wrist-velocity-peaks "$RUN_ROOT/$clip/wrist_velocity_peaks.json" \
  --ball-inflections "$RUN_ROOT/$clip/ball_inflections.json" \
  --tracks "$RUN_ROOT/$clip/tracks.json" \
  --out "$RUN_ROOT/$clip/contact_windows.json"
```

If any cue family is missing, blocked, empty, or temporally inconsistent, the
BALL path must fail closed. Do not treat empty contact windows as a BALL pass.

For human review candidates:

```bash
python scripts/racketsport/build_contact_window_candidates.py \
  --events "$RUN_ROOT/$clip/labels/events.json" \
  --out "$RUN_ROOT/$clip/contact_window_candidates.json"

python scripts/racketsport/promote_contact_windows.py \
  --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
  --template-out "$RUN_ROOT/$clip/contact_window_review.json"

python scripts/racketsport/render_contact_window_review.py \
  --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
  --review "$RUN_ROOT/$clip/contact_window_review.json" \
  --out-html "$RUN_ROOT/$clip/contact_window_review.html"
```

Apply saved review UI inputs, then promote accepted contacts:

```bash
python scripts/racketsport/apply_review_inputs_to_contact_review.py \
  --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
  --review "$RUN_ROOT/$clip/contact_window_review.json" \
  --review-input runs/review_inputs/pickleball_cv_review_latest.json \
  --clip "$clip" \
  --out-review "$RUN_ROOT/$clip/contact_window_review.json"

python scripts/racketsport/promote_contact_windows.py \
  --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
  --review "$RUN_ROOT/$clip/contact_window_review.json" \
  --out-contact-windows "$RUN_ROOT/$clip/contact_windows.json"
```

### BODY, Readiness, World, And Replay Review

```bash
python scripts/racketsport/build_body_compute_execution.py \
  --tracks "$RUN_ROOT/$clip/tracks.json" \
  --frame-compute-plan "$RUN_ROOT/$clip/frame_compute_plan.json" \
  --out "$RUN_ROOT/$clip/body_compute_execution.json"

python scripts/racketsport/build_body_mesh_readiness.py \
  --clip "$clip" \
  --smpl-motion "$RUN_ROOT/$clip/smpl_motion.json" \
  --skeleton3d "$RUN_ROOT/$clip/skeleton3d.json" \
  --frame-compute-plan "$RUN_ROOT/$clip/frame_compute_plan.json" \
  --body-compute-execution "$RUN_ROOT/$clip/body_compute_execution.json" \
  --out "$RUN_ROOT/$clip/body_mesh_readiness.json"

python scripts/racketsport/validate_pipeline_artifacts.py \
  --run-dir "$RUN_ROOT/$clip" \
  --stage e2e \
  --out "$RUN_ROOT/$clip/pipeline_readiness_e2e.json"

python scripts/racketsport/build_virtual_world.py \
  --court-calibration "$RUN_ROOT/$clip/court_calibration.json" \
  --tracks "$RUN_ROOT/$clip/tracks.json" \
  --ball-track "$RUN_ROOT/$clip/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json" \
  --smpl-motion "$RUN_ROOT/$clip/smpl_motion.json" \
  --skeleton3d "$RUN_ROOT/$clip/skeleton3d.json" \
  --out "$RUN_ROOT/$clip/virtual_world.json"

python scripts/racketsport/build_virtual_world_review.py \
  --virtual-world "$RUN_ROOT/$clip/virtual_world.json" \
  --out-html "$RUN_ROOT/$clip/virtual_world.html" \
  --index-out "$RUN_ROOT/$clip/virtual_world_review_index.json" \
  --title "$clip Virtual World"

python scripts/racketsport/build_replay_review_export.py \
  --virtual-world "$RUN_ROOT/$clip/virtual_world_paddle_preview.json" \
  --out-dir "$RUN_ROOT/$clip/replay_review" \
  --scene-out "$RUN_ROOT/$clip/replay_scene.json"
```

The replay command writes static review GLBs only. It does not produce the
production animated replay.

### Paddle Preview And RKT Readiness

Use this path only for review. It must not replace canonical RKT promotion.

```bash
python scripts/racketsport/build_racket_pose_preview.py \
  --court-calibration "$RUN_ROOT/$clip/court_calibration.json" \
  --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json" \
  --out "$RUN_ROOT/$clip/racket_pose_preview.json"

python scripts/racketsport/build_racket_pose_readiness.py \
  --clip "$clip" \
  --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json" \
  --racket-pose-preview "$RUN_ROOT/$clip/racket_pose_preview.json" \
  --out "$RUN_ROOT/$clip/racket_pose_readiness.json"

python scripts/racketsport/build_racket_promotion_audit.py \
  --clip "$clip" \
  --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json" \
  --racket-pose-preview "$RUN_ROOT/$clip/racket_pose_preview.json" \
  --out "$RUN_ROOT/$clip/racket_promotion_audit.json"

python scripts/racketsport/build_virtual_world.py \
  --court-calibration "$RUN_ROOT/$clip/court_calibration.json" \
  --tracks "$RUN_ROOT/$clip/tracks.json" \
  --ball-track "$RUN_ROOT/$clip/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json" \
  --racket-pose "$RUN_ROOT/$clip/racket_pose_preview.json" \
  --smpl-motion "$RUN_ROOT/$clip/smpl_motion.json" \
  --skeleton3d "$RUN_ROOT/$clip/skeleton3d.json" \
  --out "$RUN_ROOT/$clip/virtual_world_paddle_preview.json"
```

### Packet And Benchmark

Rebuild the packet after artifact changes:

```bash
python scripts/racketsport/build_review_packet.py \
  --run-root "$RUN_ROOT" \
  --out-dir runs/review_packets/prototype_gate_h100_v2
```

Rerun the held-out ball benchmark:

```bash
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

Record fresh benchmark/test output before citing any count as current evidence.

## H100 Sync And Checks

Run only after the local commit is pushed and the remote container should pick
up the same commit.

```bash
gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && git pull --ff-only && git rev-parse --short HEAD'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && /opt/conda/envs/fast_sam_3d_body/bin/python -m pytest -q -p no:cacheprovider tests/racketsport/test_ball_temporal_filter.py tests/racketsport/test_ball_benchmark.py tests/racketsport/test_ball_model_fusion.py'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
```

## If Output Looks Wrong

- Background balls are selected: inspect the calibration overlay first, then
  rerun target-court filtering before fusion.
- The ball jumps to a distant false detection: use the local-trajectory track or
  lower `--local-trajectory-max-error-px`.
- The real ball disappears too often: compare against
  `ball_track_fusion_temporal_vball100.json` and keep both artifacts for review
  if recall/precision tradeoffs are unresolved.
- Court lines or paddle flashes are selected as the ball: improve model-side
  candidate generation; do not treat post-processing alone as BALL verification.
- Racket pose appears plausible from boxes: keep it preview-only until true
  paddle evidence and RKT reference labels exist.

## Major Remaining Work

1. Replace the prototype BALL adapter with real TrackNet/audio/wrist/inflection
   cue generation through the orchestrator, then pass BALL F1/contact gates.
2. Improve ball candidate generation with TrackNetV4/V5 or fine-tuned
   TrackNetV3/V4/V5 on pickleball-specific edge cases.
3. Extend from 10-second smoke windows to longer windows, then full clips.
4. Collect true paddle-face corners or CAD/reference/ArUco labels and rerun RKT
   promotion/evaluation.
5. Add BODY world-MPJPE/full-window labels or equivalent evaluator evidence.
6. Replace review-only static GLBs with production animated GLB/USDZ replay.
7. Keep BALL, BODY, RKT, TRK, and RPL unverified until their documented gates
   pass on representative clips.
