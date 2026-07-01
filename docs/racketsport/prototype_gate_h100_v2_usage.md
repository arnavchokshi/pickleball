# Prototype Gate H100 v2 Runbook

Last audited: 2026-06-30

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

- **GPU substrate:** the historical H100 acceptance VM/container is not
  currently present. The running A100 VM can execute CUDA diagnostics and small
  prototype jobs, but it is missing `/workspace`, `/workspace/pickleball`, and
  `/opt/conda/envs/fast_sam_3d_body/bin/python`. A100 artifacts do not satisfy
  rows that explicitly require H100 acceptance unless the row is changed to
  allow A100 evidence. The 2026-06-30 A100 substrate run proves bounded Outdoor
  0-1150 TRK replay can run through `/home/arnavchokshi/pickleball/.venv` under
  `scripts/gpu-eval-run.sh`, but that candidate is diagnostic-only: IDF1
  0.3230, MOTA 0.2527, precision 0.6451, recall 0.6629, and 204 ID switches.
  Current work is A100-first while the H100 path is absent, but old H100-named
  gates remain unpassed until rewritten or rerun. Upstream TrackNetV3 and the
  official `TrackNet_best.pt` / `InpaintNet_best.pt` checkpoints are now
  hash-verified on A100, and the A100 has now produced full-CVAT TrackNetV3
  tracks for Burlington, Wolverine, and Outdoor 0-1150. This proves runtime and
  horizon coverage only; BALL quality gates remain unpassed.
- **CAL:** manual/review calibration artifacts exist; automatic court evidence
  remains fail-closed unless trusted line/top-net evidence is present. The
  2026-06-30 no-tap diagnostics ran inference, but max confidence was
  0.097079 locally and 0.126557 in the existing A100/CUDA artifact, with 0
  keypoints surviving the 0.5 threshold. Lower-threshold correspondences are
  diagnostic only, and the validation sweep at
  `runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_validation_eval/`
  still blocks every active clip on reprojection/world/temporal sanity. The
  goal-continuation CPU rerun at
  `runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_goal_continuation_cal_cpu_eval/`
  again blocks all three active clips: thresholds 0.5/0.12/0.1 produce no
  solvePnP samples, and lower thresholds fail reprojection, world, temporal,
  line, or border sanity. CAL-3 remains unverified.
- **TRK:** player-track overlays and candidates are qualitative review evidence.
  Tracking is not verified until IDF1, spectator rejection, ID-switch, and
  throughput gates pass on labeled clips. The current CVAT scoring pass found
  no promotable source. The latest diagnostic-only advancement now extracts
  identity-switch events, transition summaries, and temporal coverage, but the
  best full-source mean IDF1 is still 0.6397, worst IDF1 is 0.4983, and total
  switches are 82. Outdoor player coverage is the first repair target: its
  best-IDF1 source predicts only frames 0-599 while CVAT spans 0-1150, leaving
  551 GT frames and 2080 player boxes after the last prediction. The full-horizon
  A100 tiled candidate scores only IDF1 0.4482; blind repair/pruning reduces
  switches and false positives but drops recall and exact-four coverage. The
  first source-only seeded gap-fill from `tracked_detections.json` reaches only
  IDF1 0.4515, leaves switches/exact-four unchanged, and adds off-court false
  positives. The global source-only selector is also diagnostic-only: its best
  fail-closed variant drops to IDF1 0.4141, while the high-cardinality variant
  adds 109 off-court false positives. Source-only appearance/ReID diagnostics
  found no persisted embeddings or crop files in the candidate pool. The
  follow-up crop inventory exported source-pixel crops but still found 0 real
  persisted tracker embeddings. The exact A100 detector checkpoint has now been
  recovered locally at
  `runs/detect/runs/gpu/train_player_yolo26/a100_player_yolo26n_20260630_044525/weights/best.pt`
  with sha256 `511fcd11a37bae0f9f20bc66b02876b6a3e1e7e678fa24076be157b7c144e11f`.
  A learned export at
  `runs/phase2/trk_reid_embedding_export_20260630T171133Z/person_reid_embeddings.json`
  contains 6,918 YOLO26n crop embeddings. The selector/CLI now consumes that
  export as a weak diagnostic appearance cost and scores
  `runs/phase2/trk_embedding_source_selection_20260630T223103Z/`, but it is
  still `promote_trk=false`: best swept IDF1 is 0.4314 with 115 switches,
  1379 spectator/background FP, and 0.5209 four-player coverage. The
  goal-continuation analysis at
  `runs/phase2/trk_reid_embedding_analysis_20260630T_goal_continuation/`
  found track-centroid cosine min/mean/max 0.963704/0.986169/0.999300, so
  embedding-only repair is unsafe; stronger embedding-aware repair remains
  diagnostic until labeled gates pass.
- **BALL:** the strict no-click review track is a prototype review artifact.
  Human clicks are held-out benchmark labels only. BALL remains unverified until
  real ball/contact gates pass through the spine. The historical pre-A100 CVAT ball benchmark
  evaluated 1304/1524 visible labels because older Outdoor ball
  track artifacts stopped before the full 0..1150 CVAT window. The Outdoor
  horizon scan found 84 schema-valid candidates and 0 full-horizon candidates
  through frame 1150 before the A100 rerun. The A100 runtime probe now verifies
  the official TrackNetV3 checkpoints and generates full-horizon TrackNetV3
  outputs for all three CVAT clips; the current full-CVAT A100 rerun evaluates all 1524 visible labels. Raw A100
  TrackNet still fails the gate (F1@20 0.554, recall@20 0.610, precision@20
  0.508, 61 teleports, 320 hidden false positives), and label-free filters
  reduce errors only by dropping too much recall. The dense CVAT-to-TrackNet
  label artifact at `runs/ball_tracknet_cvat_dataset_20260630T171321Z/` is now
  materialized at `runs/ball_tracknet_cvat_dataset_20260630T175124Z_materialized/`
  and has a first A100 fine-tune benchmark at
  `runs/ball_tracknet_finetune_20260630T1812Z/cvat_benchmark/tracknet_dense_hidden_finetune_vs_cvat_20260630T183151Z.json`.
  It improves to F1@20 0.626, recall@20 0.647, precision@20 0.606,
  hidden-FP rate 0.342, and 2 teleports, but still does not pass BALL or
  contact-timing gates. The companion error analysis at
  `runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/`
  keeps the next iteration scoped: Burlington/Wolverine need hidden-negative
  suppression, while Outdoor needs localization/recall improvement. That
  directory now includes annotated `review_sheets/` for visible errors, hidden
  false positives, missing visible predictions, and teleports. The A100
  checkpoint audit found the fine-tune `TrackNet_best.pt` at epoch 19 is better
  than `TrackNet_cur.pt` at epoch 21, so do not blindly resume the existing save
  dir. The hard-negative materializer now consumes
  `ball_hard_negative_iteration_plan.json` into a fresh dataset at
  `runs/ball_tracknet_cvat_dataset_20260630T211745Z_hard_negative_context8_repeat3_materialized/`
  with Outdoor held out, 15 generated oversample windows, 5 unique windows, and
  480 oversampled frames. The fresh A100 resume from epoch 19 stopped after the
  epoch-20 validation signal degraded, and the scored epoch-20 candidate at
  `runs/ball_tracknet_hardneg_finetune_20260630T2130Z/cvat_benchmark/tracknet_hardneg_epoch20_vs_cvat_20260630T2200Z.json`
  regressed to F1@20 0.569, recall@20 0.583, precision@20 0.555, hidden-FP
  rate 0.342, and 1 teleport. Burlington/Wolverine improved, but held-out
  Outdoor fell to F1@20 0.296 and hit recall 0.340, so this hard-negative-only
  iteration is diagnostic-only and must not be promoted. The follow-up CPU
  local-search postprocess at `runs/ball_local_search_postprocess_20260630T225735Z/`
  evaluated all 1,524 CVAT visible labels and is diagnostic-only rejected too:
  dense local-search postprocess regresses to F1@20 0.487 / precision 0.429 /
  recall 0.563 / hidden-FP 0.970 / 10 teleports, and hard-negative local-search
  postprocess regresses to F1@20 0.502 / precision 0.490 / recall 0.514 /
  hidden-FP 0.342 / 18 teleports. PB-MAT assets are still missing.
- **BODY:** limited scheduled H100 BODY smoke outputs exist for some clips, and
  fresh A100 reset-bound bbox-scaled full-track smokes now prove the current
  Fast SAM-3D-Body path can submit real videos and emit scheduled world-joint
  and mesh outputs for BODY-safe player-frames. `BodyStageRunner` invokes Fast
  SAM once per tracked player bbox, scales track bboxes into the materialized
  BODY frame size before cropping, caps grounded BODY root motion at 8 m/s by
  default, resets temporal smoothing when a carried state would exceed the
  0.75 m track-anchor bound, and submit-style smokes preserve their full-track
  execution manifest after review helpers run. Burlington
  `runs/body_joint_goal_smoke_20260630T001407/a100_body_video_smoke_burlington_bboxscaled_resetcap075_runtime_20260701T001500Z/`
  produced 149 scheduled frames / 265 mesh+joint player-frames and
  `body_full_clip_gate.json` passes at coverage 0.974265. Wolverine
  `runs/body_joint_goal_smoke_20260630T001407/a100_body_video_smoke_wolverine_bboxscaled_resetcap075_runtime_20260701T002500Z/`
  produced 80 scheduled frames / 163 mesh+joint player-frames and
  `body_full_clip_gate.json` passes at coverage 0.964497. Both
  `body_joint_quality.json` files have empty structural quality
  blockers/warnings and remain `quality_checked_needs_accuracy_gate` with
  `missing_world_mpjpe_gate`. The selected overlay warnings now add
  `body_joint_overlay_warning_review_required`. The reset-bound runs reduce max track-anchor residual
  to 0.748754 m / 0.721875 m and the selected review overlays have
  0 selected overlay alignment failures with 9 / 2 warnings. The synced compact
  `body_gate_report_resetcap075.json` can score finalized labels from local
  `body_world_label_packet.json` predictions without pulling the huge remote
  motion files back first, and `body_joint_quality_from_packet.json` sidecars
  keep compact finite/coverage/floor/track-anchor plausibility checkable. The
  regenerated report also surfaces BODY label review as `blocked_finalization`:
  Burlington has 27 selected samples, Wolverine has 20, both draft templates
  still have 0 accepted samples, `selected_samples_have_overlay_warnings`
  is active, and `body_world_label_finalization_blocked` blocks promotion. BODY remains diagnostic/review evidence, not
  promotion, until representative world-MPJPE labels or equivalent evaluator
  evidence pass. Outdoor currently remains
  missing
  trusted BODY mesh output because current player coverage schedules 0 trusted
  world-mesh frames; the A100 TRK candidate-local diagnostic scheduling remains
  untrusted because that TRK source is `do_not_promote`. The live recheck
  `runs/body_unblock_20260630T092502Z/live_recheck_20260630T170731Z.json`
  remains the canonical H100/promotion blocker: local manifest preflight fails
  18/18 declared H100 checkpoint files, `body4d-gcp-prod` is absent, and the
  A100 has only a partial noncanonical home-path SAM-3D-Body setup without
  MoGe/canonical runtime paths. The later bbox-scaled A100 smokes supersede that
  artifact for Burlington/Wolverine diagnostic availability only; they do not
  make BODY safe for canonical promotion.
- **RKT:** box-derived paddle candidates are preview-only. They must not promote
  into canonical `racket_pose.json`; RKT needs true paddle corners, CAD/reference
  pose, or ArUco/AprilTag/reference labels plus Phase 6 evaluation. The
  goal-continuation RKT/SHOT/RPL audit remains `blocked_not_production_ready`,
  and the runtime readiness check is `blocked`: 0/6 model components are
  runtime-ready, asset readiness is false, and RKT GPU smoke/promotion is not
  allowed. The consolidated review manifest at
  `runs/cvat_imports/2026_06_30/rkt_true_corner_review_manifest_20260630_goal_continuation/`
  is review-only: 755 accepted-four required labels, 174 rendered crops,
  2,211 CVAT rectangle-derived candidates, and 0 true-corner/reference labels.
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

Current reviewed-label package:

```text
runs/cvat_imports/2026_06_30/manifest.json
runs/cvat_imports/2026_06_30/data1_substitute/manifest.json
runs/cvat_imports/2026_06_30/data1_substitute/DATA1_SUBSTITUTE_report.md
runs/cvat_imports/2026_06_30/data1_substitute/indoor_missing_export_report.json
runs/cvat_imports/2026_06_30/data1_substitute/INDOOR_MISSING_EXPORT_REPORT.md
runs/cvat_imports/2026_06_30/yolo_datasets/player/data.yaml
runs/cvat_imports/2026_06_30/yolo_datasets/paddle/data.yaml
runs/cvat_imports/2026_06_30/yolo_datasets/ball/data.yaml
runs/cvat_imports/2026_06_30/yolo_datasets/combined/data.yaml
```

This package currently uses Burlington, Wolverine, and Outdoor only. Outdoor is
capped at frame 1150; frame 1151 onward is discarded. Video 4 / Indoor remains
pending and is not required for the current three-clip detector setup. The CVAT
`ball` ellipses are converted into detector boxes during import. The CVAT
`paddle` annotations are rectangles, not true paddle-face corners; use them for
detector training and review-only RKT candidate/crop generation, not canonical
`racket_pose.json` promotion.

The DATA-1 substitute package is blocked, not promoted: source videos are 4/4
present, CVAT exports and reviewed-box imports are 3/4 present, Indoor is
missing `cvat_upload/exports/04_indoor_doubles_fwuks_0500_long_mid_baseline_cvat_for_video_1.1.zip`,
and 43 inputs are missing. Generated label skeletons are planning placeholders
only and are marked `not_ground_truth=true`.

Related 2026-06-30 status/evidence artifacts:

```text
runs/gpu_diagnostics/a100_runtime_20260630_codex.json
runs/a100_substrate_20260630T0809Z/A100_FEASIBILITY_REPORT.md
runs/a100_substrate_20260630T0809Z/outdoor_0000_1150_mobile_replay/metrics.json
runs/a100_substrate_20260630T0809Z/outdoor_0000_1150_mobile_replay/timing.json
runs/a100_trk_candidate_score_20260630T081736Z/A100_TRK_CANDIDATE_SCORE_REPORT.md
runs/a100_trk_candidate_score_20260630T081736Z/scoring/person_track_gt_scoring_report.json
runs/phase2/trk_a100_tiled_repair_20260630T082849Z/TRK_A100_TILED_REPAIR_REPORT.md
runs/phase2/trk_a100_tiled_repair_20260630T082849Z/repair_comparison_summary.json
runs/phase2/trk_source_selection_20260630T084549Z/TRK_SOURCE_SELECTION_REPORT.md
runs/phase2/trk_source_selection_20260630T084549Z/source_selection_comparison_summary.json
runs/phase2/trk_global_source_selection_20260630T085520Z/TRK_GLOBAL_SOURCE_SELECTION_REPORT.md
runs/phase2/trk_global_source_selection_20260630T085520Z/source_selection_comparison_summary.json
runs/phase2/trk_reid_diagnostics_20260630T090857Z/TRK_REID_DIAGNOSTICS_REPORT.md
runs/phase2/trk_reid_diagnostics_20260630T090857Z/source_appearance_diagnostics.json
runs/phase2/trk_reid_next_step_20260630T092540Z/NEXT_STEP.md
runs/phase2/trk_reid_next_step_20260630T092540Z/next_step_summary.json
runs/phase2/trk_reid_embedding_export_20260630T171133Z/person_reid_embeddings.json
runs/phase2/trk_reid_embedding_export_20260630T171133Z/person_reid_embeddings_512.json
runs/phase2/trk_reid_embedding_analysis_20260630T_goal_continuation/trk_reid_embedding_analysis.json
runs/phase2/trk_reid_embedding_analysis_20260630T_goal_continuation/TRK_REID_EMBEDDING_ANALYSIS.md
runs/phase2/trk_reid_embedding_consumption_plan_20260630_goal_continuation/trk_reid_embedding_consumption_plan.json
runs/phase2/trk_reid_embedding_consumption_plan_20260630_goal_continuation/TRK_REID_EMBEDDING_CONSUMPTION_PLAN.md
runs/a100_detector_eval_20260630/summary.json
runs/cvat_imports/2026_06_30/coverage_sanity_report.json
runs/cvat_imports/2026_06_30/data1_next_step_20260630T092447Z/DATA1_NEXT_STEP.md
runs/cvat_imports/2026_06_30/data1_next_step_20260630T092447Z/data1_next_step_inventory.json
runs/cvat_imports/2026_06_30/data1_next_step_20260630T170631Z/doc_ready_truth_summary.md
runs/cvat_imports/2026_06_30/data1_next_step_20260630T170631Z/data1_missing_input_manifest.json
runs/cvat_imports/2026_06_30/data1_indoor_export_recheck_20260630_goal_continuation/indoor_cvat_export_recheck.json
runs/cvat_imports/2026_06_30/data1_indoor_export_recheck_20260630_goal_continuation/INDOOR_CVAT_EXPORT_RECHECK.md
runs/phase2/person_track_gt_scoring_20260630/TRK_FAILURE_ANALYSIS_20260630.json
runs/phase2/person_track_gt_scoring_20260630/PERSON_TRACK_GT_SCORING_REPORT_before_switch_diagnostics_20260630.md
runs/phase2/person_track_gt_scoring_20260630/PERSON_TRACK_GT_SCORING_REPORT_after_switch_diagnostics_20260630.md
runs/phase2/person_track_gt_scoring_20260630/TRK_OUTDOOR_PLAYER_ID_DIAGNOSTICS_20260630.json
runs/phase2/person_track_gt_scoring_20260630/TRK_OUTDOOR_PLAYER_ID_DIAGNOSTICS_20260630.md
runs/phase2/trk_outdoor_cvat_review_proxy_20260630/outdoor_webcam_iynbd_1500_long_high_baseline/cvat_player_review_proxy_report.json
runs/phase2/trk_outdoor_cvat_review_proxy_20260630/outdoor_webcam_iynbd_1500_long_high_baseline/CVAT_PLAYER_REVIEW_PROXY_REPORT.md
runs/eval0/prototype_gate_h100_v2/ball_tracker_benchmark/cvat_ball_horizon_candidate_benchmark_20260630.json
runs/eval0/prototype_gate_h100_v2/ball_tracker_benchmark/outdoor_ball_candidate_horizon_scan_20260630.json
runs/ball_a100_runtime_20260630T083848Z/BALL_A100_RUNTIME_BLOCKER.md
runs/ball_a100_runtime_20260630T083848Z/runtime_blocker_report.json
runs/ball_a100_runtime_20260630T090538Z/BALL_A100_OFFICIAL_TRACKNET_CHECKPOINTS.md
runs/ball_a100_runtime_20260630T090538Z/runtime_blocker_report.json
runs/ball_a100_runtime_20260630T090538Z/a100_smoke_after_parse/ball_track.json
runs/ball_a100_runtime_20260630T090538Z/a100_smoke_after_parse/tracknet_smoke_metadata.json
runs/ball_a100_full_cvat_20260630T0925Z/full_cvat_tracknet_summary.json
runs/ball_a100_full_cvat_20260630T0925Z/cvat_benchmark/tracknet_full_vs_existing_20260630.json
runs/ball_a100_full_cvat_20260630T0925Z/cvat_benchmark/tracknet_full_postprocess_vs_existing_20260630.json
runs/ball_a100_full_cvat_20260630T0925Z/BALL_A100_FULL_CVAT_POSTPROCESS_REPORT.md
runs/ball_tracknet_cvat_dataset_20260630T171321Z/ball_tracknet_cvat_dataset_manifest.json
runs/ball_tracknet_cvat_dataset_20260630T171321Z/ball_tracknet_cvat_dataset_manifest.md
runs/ball_tracknet_cvat_dataset_20260630T175124Z_materialized/ball_tracknet_cvat_dataset_manifest.json
runs/ball_tracknet_cvat_dataset_20260630T175124Z_materialized/ball_tracknet_cvat_dataset_manifest.md
runs/ball_tracknet_finetune_20260630T1812Z/cvat_benchmark/tracknet_dense_hidden_finetune_vs_cvat_20260630T183151Z.json
runs/ball_tracknet_finetune_20260630T1812Z/cvat_benchmark/tracknet_dense_hidden_finetune_vs_cvat_20260630T183151Z.md
runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/tracknet_dense_hidden_error_analysis.json
runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/TRACKNET_DENSE_HIDDEN_ERROR_ANALYSIS.md
runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/review_sheets/tracknet_dense_hidden_error_review_sheets.json
runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/hard_negative_plan/ball_hard_negative_iteration_plan.json
runs/ball_tracknet_finetune_20260630T1812Z/error_analysis_20260630_goal_continuation/hard_negative_plan/BALL_HARD_NEGATIVE_ITERATION_PLAN.md
runs/ball_tracknet_cvat_dataset_20260630T211745Z_hard_negative_context8_repeat3_materialized/ball_tracknet_cvat_dataset_manifest.json
runs/ball_tracknet_cvat_dataset_20260630T211745Z_hard_negative_context8_repeat3_materialized/ball_tracknet_cvat_dataset_manifest.md
runs/ball_tracknet_hardneg_finetune_20260630T2130Z/BALL_HARDNEG_A100_RUN_REPORT.md
runs/ball_tracknet_hardneg_finetune_20260630T2130Z/ball_hardneg_a100_run_report.json
runs/ball_tracknet_hardneg_finetune_20260630T2130Z/cvat_benchmark/tracknet_hardneg_epoch20_vs_cvat_20260630T2200Z.json
runs/ball_tracknet_hardneg_finetune_20260630T2130Z/cvat_benchmark/tracknet_hardneg_epoch20_vs_cvat_20260630T2200Z.md
runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_calibration_report_20260630_codex.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead_verify_current/court_keypoint_no_tap_eval_cpu.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead_verify_current/court_keypoint_no_tap_eval_cpu.md
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead_verify_current/overlay_review/cal_keypoint_overlay_review.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead_verify_current/overlay_review/CAL_KEYPOINT_OVERLAY_REVIEW.md
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead/court_keypoint_no_tap_threshold_diagnostic.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_lead/court_keypoint_no_tap_threshold_diagnostic.md
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_validation_eval/court_keypoint_no_tap_validation_eval.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_cal_validation_eval/court_keypoint_no_tap_validation_eval.md
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_goal_continuation_cal_dry_run/court_keypoint_no_tap_eval_dry_run.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_goal_continuation_cal_dry_run/court_keypoint_no_tap_eval_dry_run.md
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_goal_continuation_cal_cpu_eval/court_keypoint_no_tap_eval_cpu.json
runs/pickleball_pretraining/court_keypoint_20260628/diagnostic_20260630_goal_continuation_cal_cpu_eval/court_keypoint_no_tap_eval_cpu.md
runs/body_runtime_diagnostics/body_runtime_diagnostics_20260630.json
runs/body_schedule_from_a100_trk_20260630T090000Z/BODY_SCHEDULE_IMPACT_REPORT.md
runs/body_schedule_from_a100_trk_20260630T090000Z/body_schedule_impact_report.json
runs/body_unblock_20260630T092502Z/BODY_UNBLOCK_20260630.md
runs/body_unblock_20260630T092502Z/body_unblock_report.json
runs/body_unblock_20260630T092502Z/live_recheck_20260630T170731Z.json
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T170539Z/rkt_shot_replay_lane_audit.json
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T170539Z/RKT_SHOT_REPLAY_LANE_AUDIT.md
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T_goal_continuation/rkt_shot_replay_lane_audit.json
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T_goal_continuation/RKT_SHOT_REPLAY_LANE_AUDIT.md
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T_goal_continuation/racket_model_runtime_readiness.json
runs/cvat_imports/2026_06_30/rkt_shot_replay_lane_audit_20260630T_goal_continuation/RACKET_MODEL_RUNTIME_READINESS.md
runs/ios_diagnostics/2026-06-30_record_first_device_smoke.md
runs/ios_diagnostics/2026-06-30-record-first-capture-smoke.json
runs/ios_diagnostics/2026-06-30_record_first_locked_device_pull.md
```

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

## CAL No-Tap Court-Keypoint Prep

The trained court-keypoint checkpoint is a pretraining artifact only. This
runner validates that the local CAL/no-tap evaluation inputs are present and
schema-compatible, selects active court-calibration clips, compares reviewed
top-net evidence against `court_line_evidence.json`, and preserves
`verified=false` / `not_cal3_verified=true` in its output. It does not mark
CAL-3 verified.

Local CPU-safe dry run:

```bash
python scripts/racketsport/evaluate_court_keypoint_no_tap.py \
  --run-root runs/eval0/prototype_gate_h100_v2 \
  --checkpoint runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt \
  --metrics runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_metrics.json \
  --out runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_eval_dry_run.json \
  --dry-run \
  --device cpu \
  --frames-per-clip 5 \
  --thresholds 0.5,0.12,0.1,0.08,0.05,0.02 \
  --markdown-out runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_eval_dry_run.md
```

Expected local selection: Wolverine, Outdoor, and Indoor are active when their
reviewed court/top-net evidence remains ready. Burlington is skipped by default
because it is retired for court calibration.

Exact H100 gate command after `body4d-gcp-prod` is recreated and the container
path/env is healthy:

```bash
cd /workspace/pickleball
PYTHONPATH=/workspace/pickleball /opt/conda/envs/fast_sam_3d_body/bin/python \
  scripts/racketsport/evaluate_court_keypoint_no_tap.py \
  --run-root runs/eval0/prototype_gate_h100_v2 \
  --checkpoint runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt \
  --metrics runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_metrics.json \
  --out runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_eval_h100.json \
  --frames-per-clip 5 \
  --device cuda \
  --thresholds 0.5,0.12,0.1,0.08,0.05,0.02 \
  --markdown-out runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_eval_h100.md
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

Selected-sample BODY world-joint visual check:

```bash
python scripts/racketsport/build_body_world_label_review_overlay.py \
  --run-dir "$RUN_ROOT/$clip"
```

Compact packet quality check, useful when `smpl_motion.json` and
`skeleton3d.json` stay on the GPU machine:

```bash
python scripts/racketsport/build_body_joint_quality.py \
  --clip "$clip" \
  --body-compute-execution "$RUN_ROOT/$clip/body_compute_execution.json" \
  --body-world-label-packet "$RUN_ROOT/$clip/body_world_label_packet.json" \
  --out "$RUN_ROOT/$clip/body_joint_quality_from_packet.json"
```

The current reset-bound bbox-scaled Burlington and Wolverine full-track runs
render selected review frames and pass structural full-clip coverage. Local
copies now include `body_mesh_readiness.json`, review queue/template/frame
assets, selected overlays, compact `body_world_label_packet.json`,
`body_joint_quality_from_packet.json`, and
`runs/body_joint_goal_smoke_20260630T001407/body_gate_report_resetcap075.json`.
Burlington reports 0 selected overlay alignment failures / 9 warnings over 27
selected samples, and Wolverine reports 0 selected overlay alignment failures /
2 warnings over 20. The packet-quality sidecars have no compact quality
blockers for all 265 / 163 packet samples, but they warn that current synced
packets lack temporal-reset metadata for compact-only root-speed checks and do
not include mesh vertices. New label packets preserve `temporal_smoothing_reset`
metadata for future compact checks. The compact gate report now shows packet
quality as `warning` and can score finalized labels from
`body_world_label_packet.json` without local huge motion-file copies, but the
current report is still blocked on `missing_world_mpjpe_gate` and
`body_joint_overlay_warning_review_required`. Promotion still needs
representative reviewed/equivalent world-MPJPE evidence, resolved selected
overlay warnings, and broader trusted TRK/BODY coverage including Outdoor.

The replay command writes static review GLBs only. It does not produce the
production animated replay.

### Paddle Preview And RKT Readiness

Use this path only for review. It must not replace canonical RKT promotion.

For the current CVAT package, convert reviewed `paddle` rectangles into
review-only candidates and build the readiness/audit reports:

```bash
manifest=runs/cvat_imports/2026_06_30/manifest.json
jq -c '.clips[]' "$manifest" | while read -r clip_json; do
  clip=$(jq -r '.clip_id' <<< "$clip_json")
  fps=$(jq -r '.fps' <<< "$clip_json")
  reviewed=$(jq -r '.reviewed_boxes' <<< "$clip_json")
  out_dir=$(jq -r '.import_dir' <<< "$clip_json")
  candidates="$out_dir/racket_candidates_from_cvat_paddle_boxes.json"

  python scripts/racketsport/build_racket_candidates.py \
    --cvat-reviewed-boxes "$reviewed" \
    --fps "$fps" \
    --out "$candidates"

  python scripts/racketsport/build_racket_pose_readiness.py \
    --clip "$clip" \
    --racket-candidates "$candidates" \
    --out "$out_dir/racket_pose_readiness_from_cvat_paddle_boxes.json"

  python scripts/racketsport/build_racket_promotion_audit.py \
    --clip "$clip" \
    --racket-candidates "$candidates" \
    --out "$out_dir/racket_promotion_audit_from_cvat_paddle_boxes.json"

  python scripts/racketsport/build_paddle_true_corner_review.py \
    --clip "$clip" \
    --racket-candidates "$candidates" \
    --max-required-labels 48 \
    --out "$out_dir/paddle_true_corner_review_from_cvat_paddle_boxes.json"
done

python scripts/racketsport/build_racket_model_runtime_readiness.py \
  --out runs/cvat_imports/2026_06_30/racket_model_runtime_readiness.json
```

Expected current status: `blocked_preview_only` for the per-clip pose readiness
reports, `safe_preview_only` for promotion audits, and `blocked` for model/runtime
readiness until true face-corner labels, CAD/reference assets, reference images,
ArUco/AprilTag/reference pose GT, and RKT pose evaluation exist.

The current true-corner review handoff is consolidated here:

```text
runs/cvat_imports/2026_06_30/rkt_true_corner_review_manifest_20260630_goal_continuation/
```

That manifest is `review_only_no_rkt_promotion`; it points reviewers to the
accepted-four crop sheets and CVAT paddle-box review candidates while preserving
the rule that rectangle corners are not true paddle-face corners.

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

Current live GCP state from the 2026-06-30 docs/status pass:

- `body4d-gcp-prod` is absent, so the H100 commands below are valid only after
  the controller recreates the H100 VM and the pod-agent container is healthy.
- `body4d-waker-ctrl` is running in `us-central1-a`.
- `pickleball-a100-spot-ase1a` is running in `asia-southeast1-a` with one idle
  A100-SXM4-40GB. Its repo checkout is `/home/arnavchokshi/pickleball_git` on
  `main` at `fb2169d` with untracked `cvat_upload/` and YOLO26 weight files; the
  file-copy tree `/home/arnavchokshi/pickleball` has a CUDA-enabled `.venv`.
- The A100 VM does not have `/workspace`, `/workspace/pickleball`, or
  `/opt/conda/envs/fast_sam_3d_body/bin/python`; the H100 commands in this
  runbook will not run there without explicit adaptation.
- A100 runs are diagnostic unless a specific gate explicitly accepts A100. Do
  not record them as H100 phase verification.

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

1. Improve ball candidate generation with TrackNetV4/V5, PB-MAT, or another
   TrackNetV3/V4/V5 iteration on pickleball-specific edge cases; the raw
   full-CVAT A100 TrackNetV3 run has horizon coverage but fails
   F1/FP/teleport quality, and the first dense hidden-negative A100 fine-tune
   improves those metrics without reaching BALL/contact gates. Use the
   goal-continuation error analysis to mine hard negatives and visible-error
   spans, and start from the epoch-19 fine-tune best checkpoint rather than the
   worse epoch-21 current checkpoint. The hard-negative-only and local-search postprocess paths are rejected; the next useful BALL iteration must target
   Outdoor localization/recall or a different model-side candidate, then rerun
   the same reviewed F1/hidden-FP/teleport/contact checks.
2. Run BALL cue fusion/contact timing only after the ball candidate quality
   improves enough to support contact evidence.
3. Export/import the missing Indoor CVAT `for video 1.1` task before claiming
   DATA-1 progress.
4. Collect true paddle-face corners or CAD/reference/ArUco labels and rerun RKT
   promotion/evaluation.
5. Repair BODY/TRK temporal continuity and coverage before promotion: current
   reset-bound bbox-scaled A100 BODY smokes prove real BODY output on two
   submitted videos, pass structural full-clip coverage for Burlington/Wolverine,
   and have 0 selected overlay alignment failures. `body_gate_report_resetcap075.json`
   now shows the two BODY world-label review bundles are `blocked_finalization`
   with 27 / 20 selected samples, 0 accepted template samples, and selected
   overlay-warning samples, so `body_world_label_finalization_blocked` is active
   and no finalized representative world-MPJPE labels or equivalent evaluator
   evidence exist. Cover the broader
   trusted clip set including Outdoor, fill the BODY world-label review-bundle
   templates, and run the bundle `finalize_command`. Neither current A100 result
   replaces the accuracy gate.
6. Replace review-only static GLBs with production animated GLB/USDZ replay.
7. Keep BALL, BODY, RKT, TRK, and RPL unverified until their documented gates
   pass on representative clips.
8. For TRK specifically, the exact A100 detector checkpoint has been recovered
   locally and real crop embeddings have been persisted, but centroid similarity
   is high. The first weak-cost embedding-aware selector has been scored and is
   still diagnostic-only, so stronger appearance-aware repair or detector/tracker
   training must keep same-frame/court/source constraints and full labeled
   scoring before any promotion discussion.
