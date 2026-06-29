# Prototype Gate H100 v2 TrackNet and Body Smoke

Date: 2026-06-27; updated 2026-06-28

This is a prototype smoke record, not an acceptance gate. None of these artifacts make BALL, BODY, or E2E `VERIFIED`; they only prove real H100 runtime seams on the accepted-four prototype clips.

Archived evidence snapshot. The canonical current prototype runbook is `docs/racketsport/prototype_gate_h100_v2_usage.md`; use this file only for the original TrackNet/BODY smoke provenance.

## Accepted Clips

- `burlington_gold_0300_low_steep_corner`
- `wolverine_mixed_0200_mid_steep_corner`
- `outdoor_webcam_iynbd_1500_long_high_baseline`
- `indoor_doubles_fwuks_0500_long_mid_baseline`

Rejected/deferred clip remains `side_view_game5_0100_high_side_fence` because it is side/fisheye-distorted and not representative for this prototype gate.

## TrackNetV3 Smoke Windows

Each accepted clip has a 10 second smoke under:

`runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/`

Artifacts per clip:

- `input_0000_0010.mp4`
- `predictions/input_0000_0010_ball.csv`
- `ball_track_0000_0010.json`
- `ball_track_0000_0010_run.json`
- `ball_track_overlay_h264.mp4`
- `ball_track_overlay_run.json`
- `window_manifest.json`

Summary:

| Clip | Input FPS | Frames | Visible Frames | H.264 Overlay |
| --- | ---: | ---: | ---: | --- |
| `burlington_gold_0300_low_steep_corner` | 59.94005994 | 600 | 599 | `runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/tracknet_smoke_0000_0010/ball_track_overlay_h264.mp4` |
| `wolverine_mixed_0200_mid_steep_corner` | 30.0 | 300 | 288 | `runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/tracknet_smoke_0000_0010/ball_track_overlay_h264.mp4` |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | 60.0 | 600 | 600 | `runs/eval0/prototype_gate_h100_v2/outdoor_webcam_iynbd_1500_long_high_baseline/tracknet_smoke_0000_0010/ball_track_overlay_h264.mp4` |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | 29.97002997 | 300 | 300 | `runs/eval0/prototype_gate_h100_v2/indoor_doubles_fwuks_0500_long_mid_baseline/tracknet_smoke_0000_0010/ball_track_overlay_h264.mp4` |

Timing note: the prototype label metadata rounds some source frame rates to 30 or 60 fps. The smoke `ball_track_0000_0010.json` files were corrected from the saved TrackNet CSVs using `ffprobe avg_frame_rate` from each smoke input file.

TrackNet note: upstream `--video_range` is only for background median sampling. It does not trim prediction frames. The smoke windows were physically cut with `ffmpeg -ss 0 -t 10` before TrackNet inference.

## FastSAM-3D-Body Probe

One real probe was run on Burlington frame 0 using the four tracked player boxes at `t=0`:

`runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/body_probe_0000/sam3dbody_probe.json`

Result:

- `person_count`: 4
- `status`: `probe_only_not_verified`
- runtime: `/opt/fast-sam-3d-body` with `/opt/conda/envs/fast_sam_3d_body`
- checkpoint dir: `/opt/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3`
- detected output keys include `pred_vertices`, `pred_keypoints_3d`, `pred_joint_coords`, `pred_cam_t`, `pred_pose_raw`, `body_pose_params`, `hand_pose_params`, `shape_params`, and `global_rot`.

This proves `setup_sam_3d_body(...).process_one_image(...)` can run on a prototype pickleball frame with tracked boxes. It does not yet prove the court-world SMPL conversion, foot contact, physics, or BODY acceptance gates.

## H100 BODY Scheduled Spine Smoke

On 2026-06-28, the scheduled BODY spine ran successfully on the three accepted
clips whose current `frame_compute_plan.json` requested `world_mesh` work:
Burlington, Wolverine, and Indoor doubles. The run used `/workspace/pickleball`
inside the H100 container with `/opt/fast-sam-3d-body`,
`/opt/conda/envs/fast_sam_3d_body`, and the local
`/workspace/checkpoints/body4d/sam-3d-body-dinov3/model.ckpt` checkpoint.
Outdoor webcam was not forced through Fast SAM-3D-Body because its current
frame plan schedules zero world-mesh BODY frames and correctly remains
fail-closed.

Command:

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
for clip in \
  burlington_gold_0300_low_steep_corner \
  wolverine_mixed_0200_mid_steep_corner \
  indoor_doubles_fwuks_0500_long_mid_baseline; do
  scripts/gpu-eval-run.sh /opt/conda/envs/fast_sam_3d_body/bin/python \
    -m threed.racketsport.orchestrator \
    --clip "$clip" \
    --inputs "$RUN_ROOT/$clip" \
    --out "$RUN_ROOT/$clip" \
    --stage body \
    --tracking-mode precomputed
done
```

Result summary:

| Clip | Pipeline status | BODY frames | BODY player-frames | Mesh player-frames | Mesh vertices/frame |
| --- | --- | ---: | ---: | ---: | ---: |
| `burlington_gold_0300_low_steep_corner` | `pass` | 3 | 9 | 9 | 18,439 |
| `wolverine_mixed_0200_mid_steep_corner` | `pass` | 4 | 12 | 12 | 18,439 |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | `pass` | 2 | 6 | 6 | 18,439 |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | not run | 0 | 0 | 0 | n/a |

Every successful BODY run reported:

- `grounding`: `camera_extrinsics_plus_track_footpoint_court_z0`
- `world_frame`: `court_Z0`
- `body_compute_mode`: `adaptive_frame_compute_plan`
- `verified_model_ids`: `fast_sam_3d_body_dinov3`, `sam_3d_body_mhr_model`,
  `moge_2_vitl_normal`, `yolo26m`
- `body_model_path`: `/workspace/checkpoints/body4d/sam-3d-body-dinov3/model.ckpt`
- `detector_model_path`: `/workspace/checkpoints/body4d/yolo26/yolo26m.pt`

Pulled-back local artifacts:

- `runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/smpl_motion.json`
- `runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/skeleton3d.json`
- `runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/smpl_motion.json`
- `runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/skeleton3d.json`
- `runs/eval0/prototype_gate_h100_v2/indoor_doubles_fwuks_0500_long_mid_baseline/smpl_motion.json`
- `runs/eval0/prototype_gate_h100_v2/indoor_doubles_fwuks_0500_long_mid_baseline/skeleton3d.json`
- refreshed per-clip `pipeline_run.json`, `body_compute_execution.json`,
  `body_mesh_readiness.json`, `pipeline_readiness_e2e.json`, and
  `virtual_world.json`.

This proves the registered `BodyStageRunner` can run on the H100 through the
fail-closed spine on scheduled local test-video frames and write schema
artifacts. It is still not BODY `VERIFIED`: no labeled world-MPJPE gate has
passed, Outdoor currently has no scheduled world-mesh BODY frames, and
foot/contact/physics accuracy remains unmeasured.

## Floor Placement, Physics Scaffold, and Replay Viewer

On 2026-06-28, the accepted-four `virtual_world.json` files were regenerated
with per-player floor placement fields derived from court-calibrated track
footpoints and available BODY world meshes. The same pass wrote
`physics_refinement.json` through the existing CPU physics-refinement scaffold
and generated `replay_viewer_manifest.json` for the local React/Three.js
viewer.

Generated summary at snapshot time:

| Clip | Mesh player-frames | Floor player-frames | Physics artifact | FOOT-2 done |
| --- | ---: | ---: | --- | --- |
| `burlington_gold_0300_low_steep_corner` | 9 | 220 | `cpu_fallback_scaffold` | false |
| `wolverine_mixed_0200_mid_steep_corner` | 12 | 129 | `cpu_fallback_scaffold` | false |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | 0 | 1144 | `cpu_fallback_scaffold` | false |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | 6 | 569 | `cpu_fallback_scaffold` | false |

The browser viewer was run locally against the Burlington manifest at
`http://127.0.0.1:5173/` and rendered the local video, player box overlay,
court, BODY mesh/joint lines, floor markers, and physics status. A saved
browser screenshot passed an image-level nonblank check at:

```text
runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/replay_viewer_browser_check.jpg
```

This proves the local review viewer path loaded real generated artifacts at the
time of this snapshot. It does not prove production replay export or physics
verification. The snapshot physics artifact remained `cpu_fallback_scaffold`; it does not run MuJoCo/MJX,
PhysPT, PHC/PULSE, or FOOT-2, and no positive foot-contact frames are present
in the snapshot BODY artifacts.

## Ball Review Track At Snapshot Time

The accepted-four TrackNet smoke windows now also have no-click model-fusion and
local-trajectory review artifacts under each clip's `tracknet_smoke_0000_0010/`
directory:

- `ball_track_fusion_temporal_vball100.json`
- `ball_track_fusion_temporal_vball100_summary.json`
- `ball_track_fusion_temporal_vball100_localtraj.json`
- `ball_track_fusion_temporal_vball100_localtraj_summary.json`
- `ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4`

The local-trajectory variant is the current strict review track. On the four
accepted clips it removes benchmark teleports and lowers hidden false positives
relative to the looser fusion track, but it also hides more uncertain ball
frames. This remains `filtered_not_gate_verified`; it is not BALL `VERIFIED`.

Machine cue fusion still emits no machine-derived contact events unless all
three cue families are present: `audio_onsets.json`, `wrist_velocity_peaks.json`,
and `ball_inflections.json`. When they are present and temporally agree, the
BALL runner fuses them into schema-valid contact events. The accepted-four
canonical `contact_windows.json` files currently exist from explicit human
review inputs, with `player_id` untrusted/null; they schedule limited BODY
review work (Burlington 3 frames / 9 player-frames, Wolverine 4 / 12, Outdoor 0
/ 0, Indoor 2 / 6) but do not make BALL or BODY `VERIFIED`.

Usage and rerun commands are documented in
`docs/racketsport/prototype_gate_h100_v2_usage.md`.
