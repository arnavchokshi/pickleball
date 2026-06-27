# Prototype Gate H100 v2 TrackNet and Body Smoke

Date: 2026-06-27

This is a prototype smoke record, not an acceptance gate. None of these artifacts make BALL, BODY, or E2E `VERIFIED`; they only prove real H100 runtime seams on the accepted-four prototype clips.

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
