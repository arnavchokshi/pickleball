# one_world_v1 field verification

Date: 2026-07-16. All probes were read-only. `EXIT=0` is the real, unpiped exit
code. The complete machine-readable output is `field_probe_output.json`; the
extracts below are trimmed, not reconstructed. Unless stated otherwise, the
artifact root is
`runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/`.

## One command that verifies all exercised artifacts

```console
$ python3 runs/lanes/oneworld_design_20260716/field_probe.py > runs/lanes/oneworld_design_20260716/field_probe_output.json; rc=$?; printf 'EXIT=%s\n' "$rc"; exit "$rc"
EXIT=0
```

The probe loads every named file, fails if a required exercised field is
missing, records Python types, and emits samples. The following per-artifact
extracts come from that output.

## Calibration and trust

Command:

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["court_calibration.json"]; print({k:d[k] for k in ("coordinate_frame","intrinsics","extrinsics","metric_confidence","capture_quality","source","coordinate_contract","inline_trust_band","companion_court_trust_band")})'
{'coordinate_frame': {'type': 'str', 'value': 'court_netcenter_z_up_m'}, 'intrinsics': {'keys': ['cx', 'cy', 'dist', 'fx', 'fy', 'source'], 'type': 'dict'}, 'extrinsics': {'keys': ['R', 'camera_height_m', 't'], 'type': 'dict'}, 'metric_confidence': {'type': 'str', 'value': 'low'}, 'capture_quality': {'keys': ['grade', 'reasons'], 'type': 'dict'}, 'source': {'type': 'str', 'value': 'metric_15pt_reviewed'}, 'coordinate_contract': {'type': 'NoneType', 'value': None}, 'inline_trust_band': {'type': 'NoneType', 'value': None}, 'companion_court_trust_band': {'stage': 'CAL', 'gate_id': 'court_calibration_pck5px_gate', 'gate_status': 'metric15_unverified', 'badge': 'preview', ...}}
EXIT=0
```

`homography` is a 3x3 list; `intrinsics.fx/fy/cx/cy` and `dist` are native
pixel-camera values; `extrinsics.R/t` use world-to-camera OpenCV column
convention; `camera_height_m` and the world are metres. This legacy artifact
has neither the additive `coordinate_contract` nor inline `trust_band`.
`trust_bands.json.court` is therefore the actual band source. The pass must
inherit `preview/metric15_unverified`, never synthesize a stronger band.

The two requested calibration-status vocabularies are also real, but are not
inline fields on this Wolverine calibration:

```console
$ python3 -c 'from pathlib import Path; p=Path("scripts/racketsport/process_video.py"); s=p.read_text(); print("corrected_unverified", s.count("corrected_unverified")); p=Path("threed/racketsport/orchestrator.py"); s=p.read_text(); print("line_evidence_solved_preview", s.count("line_evidence_solved_preview"))'
corrected_unverified 2
line_evidence_solved_preview 4
EXIT=0
```

The design treats either vocabulary as preview/low-confidence and carries it
through unchanged.

## Tracks and placement generations

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json")); print(d["tracks.json"]); print(d["placement.json"])'
tracks: fps=30.0 (float), player_count=4, player fields=[frames,id,role,side], frame fields=[bbox,conf,t,world_xy], sample t=0.0, world_xy=[-1.5903862315,-2.6787989680], conf=0.9342419505; repair_markers=None
placement: fps=30.0; frame fields=[covariance_m2,frame_idx,fused_world_xy,original_world_xy,signals,smoothed_world_xy,source_counts,stance,t]; sample frame_idx=0, t=0.0, fused_world_xy=[-1.4935363119,-2.6604292703], smoothed_world_xy=[-1.5903862315,-2.6787989680], covariance_m2=2x2; signals bbox/native2d used with xy,sigma_m,covariance_m2 and sam3d used=false reason=missing
EXIT=0
```

Thus `TrackFrame.world_xy`, `conf`, bbox, `t`, IDs/role/side, and all fallback
placement signal/covariance/provenance fields are real. Both are court-plane
metres at 30 fps. `placement.homography_pixel_convention` is absent and
`undistort_applied=false`; this is provenance, not permission to relabel pixels.

The Track-I artifact appeared while this lane was running, but no `SCHEMA.md`
was present. It is exercised on a *different* Wolverine run and cannot be
joined to the manager v5.1 run (player IDs are 18-21 instead of 1-4 and its
input hashes point at `w3_freshworlds_20260707`):

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["placement_trajectory_refined.json"]; print({k:d[k] for k in ("path","same_run_as_manager_wolverine","artifact_type","coordinate_space","world_frame","fps","preview_band","VERIFIED","player_id","refinement_fields","refinement_sample")})'
{'same_run_as_manager_wolverine': False, 'artifact_type': 'placement_trajectory_refined', 'coordinate_space': 'world_court_netcenter_z_up_m', 'world_frame': 'court_Z0', 'fps': 30.0, 'preview_band': True, 'VERIFIED': 0, 'player_id': 18, 'refinement_fields': ['correction_convention','correction_magnitude_m','covariance_m2','provenance','refined_foot_positions','refined_transl_world','rigid_correction_xyz_m'], 'refinement_sample': {'refined_transl_world': [-1.75125,5.96156,0.0], 'rigid_correction_xyz_m':[0.0,0.0,-0.0], 'covariance_m2': 3x3, 'correction_magnitude_m':0.0, 'provenance': {'evidence': {'body':...,'court_plane':...,'plant':...,'smoothness':...,'trk':...}, 'plant_anchored':False}}}
EXIT=0
```

The preferred generation is buildable, including covariance and effective
per-term weights, but absent for the target run. v1 must use `placement.json`
there. It must reject a refined artifact whose input hashes/run identity,
player IDs, fps, or frame-time joins do not match the target run.

Player-repair markers are generation-2 code output, not present in the v5.1
`tracks.json`. The producer writes them in `repair_summary.json.summary`:

```console
$ python3 -c 'from pathlib import Path; p=Path("threed/racketsport/player_id_repair.py"); L=p.read_text().splitlines(); print("\n".join(f"{i+1}: {L[i]}" for i in range(467,479)))'
468: confidence_repairs = tuple(
469:     {
470:         "player_id": player_idx + 1,
471:         "frame_index": detection.frame_idx,
472:         "t": detection.frame_idx / fps,
473:         "conf": max(0.0, min(1.0, detection.conf)),
474:         "conf_source": detection.conf_source,
475:         "repaired": True,
...
EXIT=0
```

The concrete gap-fill marker is
`conf_source="interpolated_endpoint_min_capped_0_35"`. Absence in a legacy run
means “repair provenance unavailable,” not “measured.”

## BODY motion and wrist schema

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["smpl_motion.json"]; print(d)'
fps=30.0 (float), model='sam3dbody_world_joints', world_frame='court_Z0', player_count=4, skeleton_stride=None; frame fields=[body_pose,foot_contact,foot_lock,frame_idx,global_orient,grf,joint_conf,joints_world,left_hand_pose,mesh_vertices_world,right_hand_pose,t,temporal_smoothing_reset,track_world_xy,transl_world]; sample frame_idx=0, t=0.0, transl_world=[-1.6920796003,-3.0117146001,0.0], joints_world_count=70, idx9=[-1.8060191617,-3.0617157085,0.7800266585], idx10=[-1.6448032048,-3.1248648668,0.7717128269], confidences=0.9342419505/0.9342419505
EXIT=0
```

The motion is court-Z-up metres at 30 fps. `frame_idx` is present. Contrary to
the requested ideal contract, `skeleton_stride` is not serialized in this
artifact; v1 must infer the *observed cadence* from sorted frame-index deltas
and record that fact. It may not assume the runner config.

```console
$ python3 -c 'from threed.racketsport.joint_schema import BODY_17_JOINT_NAMES as n; print(n[9],n[10],len(n))'
left_wrist right_wrist 17
EXIT=0
```

The first 17 of the 70 joints follow BODY_17; wrists are indices 9/10. BODY
frames are sparse: v5.1 has player frame counts 286/300/272/244 over frames
0..299. Missing wrist frames are therefore normal and require the bounded
interpolation/absence policy in `DESIGN.md`.

## BALL 2D, candidates, repair provenance, and 3D generations

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["ball_track.json"]; print(d)'
fps=30.0, source='wasb', frame fields=[approx,conf,speed_mps,spin_rpm,t,visible,world_xyz,xy], sample t=0.0 xy=[354.676941,307.531799] conf=0.86482644 visible=true world_xyz=null approx=false; confidence_repair_marker=None; ball_candidates_sibling_exists=false
EXIT=0
```

`xy` is native WASB image pixels; `t` is seconds; `conf` is unitless. This run
has 300 frames, 242 visible, and no raw 3D. `visibility_level` is optional in
the schema and absent here. Generation-2 repaired samples keep `approx=true`
and write the exact marker to the filter summary:

```console
$ python3 -c 'from pathlib import Path; L=Path("threed/racketsport/ball_temporal_filter.py").read_text().splitlines(); print("\n".join(f"{i+1}: {L[i]}" for i in range(484,497)))'
485: frame["conf"] = min(float(left["conf"]), float(right["conf"])) * 0.5
486: frame["visible"] = True
487: frame["approx"] = True
...
494: "conf_source": "interpolated_endpoint_min_half",
495: "repaired": True,
EXIT=0
```

`ball_candidates.json` is absent in v5.1. Its strict code schema is exercised
elsewhere but is optional here:

```console
$ python3 -c 'from threed.racketsport.schemas import BallCandidates; print(BallCandidates.model_json_schema()["properties"].keys()); print(BallCandidates.model_json_schema()["$defs"]["BallCandidate"]["properties"].keys())'
dict_keys(['schema_version','artifact_type','fps','source','source_mode','input_preprocessing','primary_output','max_candidates_per_frame','nms_radius_px','not_ground_truth','candidate_prediction','provenance','frames'])
dict_keys(['xy','score','source_detector'])
EXIT=0
```

Arc-solved generation 1 is real:

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["ball_track_arc_solved.json"]; print({k:d[k] for k in ("status","policy","kill_reasons","frame_fields","frame_sample","anchor_fields","segment_fields","segment_sample","segment_budget_exceeded","ball_arc_render_sibling_exists")})'
status='ran'; policy={feeds_world_only:true,render_only:true,not_for_detection_metrics:true,outdoor_indoor_labels_read:false}; kill_reasons=[]; frame fields=[approx,arc_solver,band,conf,not_for_detection_metrics,render_only,sigma_m,source,speed_mps,spin_rpm,t,visible,world_xyz,xy]; sample world_xyz=[-0.377492154,7.063090754,0.432489464], conf=0.86482644, sigma_m=2.001858, band='anchored_measured'; anchor fields=[anchor_id,details,frame,immovable,kind,sigma_m,source,status,t,world_xyz]; segment fields include anchors_used,frame_start/end,reprojection_rmse_px,physical_sanity,status; sample rmse=8.622576 status='fit'; segment_budget_exceeded=false; ball_arc_render_sibling_exists=false
EXIT=0
```

This run has 11 segments and 295 world samples, but it is render-only and not
valid for BALL detection metrics. The repository radius constant is real:

```console
$ python3 -c 'from threed.racketsport.ball_arc_solver import BALL_RADIUS_M; print(BALL_RADIUS_M)'
0.0371
EXIT=0
```

Generation-2 `ball_arc_render.json` is code-only/unexercised in the target run.
The strict schema has `segments[].anchor_types/anchor_frames/confidence/
flight_sanity_verdict/reprojection_rmse_px`, plus dense `samples[].t/
frame_float/world_xyz/confidence/band/bridge`; all are `render_only=true` and
`not_for_detection_metrics=true`. The writer is
`threed/racketsport/ball_arc_chain.py:320-328`. A guard emits the literal
`segment_budget_exceeded` and `missing_segment_count` rather than fabricating a
segment (`ball_arc_solver.py:2721-2733,2839-2849`). v1 consumes this loud
degrade as-is.

## Audio and events

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json")); print(d["audio_onsets_v2.json"]); print(d["contact_windows.json"])'
audio: frame_rate=30.0, status='blocked', not_gate_verified=true, trusted_for_contact=false, onset_count=0, onset_fields=[], pop_band_ratio_first=null
contacts: event_count=24; fields=[confidence,frame,player_id,sources,t,trust_band_note,type,window]; sample type='contact', t=0.175, frame=5, player_id=3, confidence=0.6808895, sources={ball_inflection:0.76187,wrist_vel:0.599909}, window={t0:0.14,t1:0.23,importance:0.6808895}; refined_sibling_exists=false
EXIT=0
```

Wolverine proves the absence path: audio is a valid blocked artifact with zero
onsets, and `contact_windows_refined_v1.json` is absent. The producing code at
`audio_onsets_v2.py:385-401` writes `time_s/raw_time_s/analysis_time_s/score/
onset_strength/source/window_start_s/window_end_s` and `features.{spectral_flux,
high_frequency_content,band_energy_delta,pop_band_ratio}`. Those feature fields
are generation-2-unexercised on this run. Audio remains soft, bounded,
non-gating, and may never create a contact by itself.

## Paddle generations and typed lift

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["racket_pose_estimate.json"]; print(d)'
world_frame='court_Z0', translation_unit='m', render_only=true, not_for_detection_metrics=true, trust='estimated_from_wrist'; frame fields=[ambiguous,conf,confidence_provenance,frame,not_for_detection_metrics,pose_se3,proxy_inputs,render_only,reprojection_error_px,source,t,translation_unit,trust,trust_band,world_frame]; sample frame=0 t=0.0 pose_se3={R:3x3,t:[-1.42549,-2.52260,0.72708]} conf=0.934242 reprojection_error_px=null ambiguous=false source='wrist_proxy'; racket_pose=false hypotheses=false racket_candidates=false
EXIT=0
```

The only target-run artifact is a court-world wrist proxy, not true paddle
6DoF. It cannot resolve IPPE ambiguity and stays unresolved.

Generation 2 is code-only/unexercised on this run. The writer at
`racket_stage_runner.py:189-209` emits `racket_pose.json` and
`racket_pose_hypotheses.json` with `world_frame="camera"` and
`translation_unit="cm"`. Each hypotheses frame contains `primary_pose`,
`alt_pose`, `candidate_reprojection_errors_px`, `ambiguity_margin_px`, and
`ambiguous` (`racket_stage_runner.py:289-314`). It deliberately retains the
second IPPE pose.

There is currently no Pydantic schema class or `ARTIFACT_MODELS` entry for
`racket_pose_hypotheses`; the writer and its tests are the only contract. This
is a real schema gap. The implementation slice must add a strict
`extra="forbid"` model before one-world consumes generation 2.

The camera-to-world path is real and typed:

```console
$ python3 -c 'from threed.racketsport import coordinates as c; print(c.translation_to_metres([1,2,3],input_unit="cm")); print(c.invert_extrinsics.__name__,c.camera_to_world_points.__name__,c.project_world_points.__name__)'
(0.01, 0.02, 0.03)
invert_extrinsics camera_to_world_points project_world_points
EXIT=0
```

`coordinates.py:580-612` pins `camera = R @ world + t` and its inverse;
`translation_to_metres` is lines 615-643. No ad-hoc matrix convention is
needed.

## Net, court zones, rallies, and baseline world

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json")); print(d["net_plane.json"]); print(d["court_zones.json"]); print(d["rally_spans.json"]); print(d["virtual_world.json"])'
net: plane.point=[0,0,0], plane.normal=[0,1,0], endpoints=[[-3.3528,0,0.9144],[3.3528,0,0.9144]], center_height_in=34, post_height_in=36
zones: court polygon=[[-3.048,-6.7056],[3.048,-6.7056],[3.048,6.7056],[-3.048,6.7056]] metres; named service/NVZ zones present
rallies: one span t0=0.0 t1=10.0 sources=[ball,player_motion], not_ground_truth=true
world: artifact_type='racketsport_virtual_world', world_frame='court_Z0', fps=30.0; summary includes player/ball/paddle coverage and warnings
EXIT=0
```

These values provide computable court/net signed distances, out-of-bounds
flags, rally-frame denominators, and the required no-fusion baseline.

## Measurement reality and protected surfaces

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/field_probe_output.json"))["demo_reality"]; print(d)'
ball_fields=[bounces,fps,frames,input_preprocessing,schema_version,source], ball_world_xyz_count=0, court_coordinate_frame=null, court_metric_confidence=null, has_tracks=false, has_smpl_motion=false, has_contact_windows=false, has_audio_onsets_v2=false
EXIT=0
```

The demo is exactly the advertised partial surface: 2D BALL + old calibration
only. `PIPELINE_SUMMARY.json.video_path` points to
`data/pbvision_11min_20260713/source_video.mp4`, so audio is CPU-derivable, but
not yet minted. Player/wrist/contact fusion and nonzero four-player world
coverage are impossible without GPU regeneration, which v1 will not do.

Only Wolverine and Burlington protected labels were opened for the following
inventory; no label values are inputs to fusion:

```console
$ python3 -c 'import json; from pathlib import Path; [(lambda r,e,b,c: print(clip,{"event_frames":len(e["frames"]),"ball_items":len(b["items"]),"ball_review_items":len(b["review_items"]),"cal_image_points":len(c["image_pts"]),"cal_world_points":len(c["world_pts"]),"event_not_ground_truth":e["not_ground_truth"],"ball_not_ground_truth":b["not_ground_truth"]}))(Path("eval_clips/ball")/clip/"labels",json.load(open(Path("eval_clips/ball")/clip/"labels/events.json")),json.load(open(Path("eval_clips/ball")/clip/"labels/ball_points.json")),json.load(open(Path("eval_clips/ball")/clip/"labels/court_calibration_metric15pt.json"))) for clip in ["wolverine_mixed_0200_mid_steep_corner","burlington_gold_0300_low_steep_corner"]]'
Wolverine: event_frames=8, ball_items=30, ball_review_items=30, calibration image/world points=15/15, event_not_ground_truth=true, ball_not_ground_truth=true
Burlington: event_frames=8, ball_items=30, ball_review_items=30, calibration image/world points=15/15, event_not_ground_truth=true, ball_not_ground_truth=true
EXIT=0
```

They are evaluation-only, independent-ish review surfaces—not ground truth by
their own flags, never training inputs. Outdoor/Indoor labels are prohibited.

## Baseline feasibility proof

```console
$ python3 runs/lanes/oneworld_design_20260716/baseline_probe.py --run-dir runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z --confidence-threshold 0.5 --output runs/lanes/oneworld_design_20260716/baseline_probe_output.json >/dev/null; rc=$?; printf 'EXIT=%s\n' "$rc"; exit "$rc"
EXIT=0
```

```console
$ python3 -c 'import json; d=json.load(open("runs/lanes/oneworld_design_20260716/baseline_probe_output.json"))["metrics"]; c=d["ball_at_contact_to_hitter_wrist"]; w=d["world_coverage"]; print({"contact_events":c["event_count"],"computable":c["computable_event_count"],"center_median_m":c["center_distance_m"]["median"],"center_p90_m":c["center_distance_m"]["p90_nearest_rank"],"volume_median_m":c["wrist_volume_residual_m"]["median"],"volume_p90_m":c["wrist_volume_residual_m"]["p90_nearest_rank"],"world_coverage":w["coverage_fraction"],"complete_frames":w["simultaneously_world_placed_frame_count"],"rally_frames":w["rally_frame_count"],"threshold":w["confidence_threshold"]})'
{'contact_events': 24, 'computable': 24, 'center_median_m': 8.130821455698724, 'center_p90_m': 11.322165421763817, 'volume_median_m': 7.973721455698724, 'volume_p90_m': 11.165065421763817, 'world_coverage': 0.39, 'complete_frames': 117, 'rally_frames': 300, 'threshold': 0.5}
EXIT=0
```

These are internal preview baselines, `VERIFIED=0`. The enormous contact
distances are a real mismatch, not a successful contact model. Exact 30-fps
frame joins exist for all 24 events, so the mismatch is not papered over as a
timebase failure.
