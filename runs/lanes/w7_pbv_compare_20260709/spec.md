# LANE w7_pbv_compare_20260709 — OURS vs PB.VISION on the SAME wolverine footage (READ-ONLY analysis)

## HARD RULES
READ-ONLY on all existing artifacts; outputs ONLY under runs/lanes/w7_pbv_compare_20260709/. No commits. No repo source edits. This is measurement — no verdict spin; where alignment is impossible, say so.

## INPUTS
- THEIRS (pb.vision, owner-exported, wolverine upload vid=emwt3u5kzavy): runs/research_ball3d_20260709/pbvision_cv_export/{cv_export.json,insights.json,stats.json}. cv structure: camera.cameraSegments[0] (fps 30, fov, 3D position/orientation, court_points u/v+confidence+spread), sessions[0].rallies[0] = frame_index 45 + 252 frames, each frame: actions{ball,bounce,net,shot,ball_radius,...} u/v+confidence, balls.ball.court_position{x,y,z}+interpolated flag, player_court_positions (FEET units, their court frame; rally spans start_ms 1500-9866 per insights.json).
- OURS (fresh full-stack production runs on wolverine_mixed_0200_mid_steep_corner): runs/lanes/w7_critique_20260709/wolv_world/ (run1: ball_track.json 2D, ball_track_arc_solved.json, ball_track_physics_filled.json, ball_arc_render.json, ball_inflections.json, ball_bounce_candidates.json, contact_windows.json, court_calibration.json, net_plane.json, virtual_world.json, PIPELINE_SUMMARY.json, match_stats.json) and runs/lanes/w7_speedgate_20260709/critique_world/wolverine_run6/ (run6, same artifact set — use for run-to-run robustness of any conclusion). Prior diagnosis: runs/lanes/w7_ball3ddiag_20260709/DIAGNOSIS.md.

## DELIVERABLES (each an acceptance row with numbers)
1. ALIGNMENT (prerequisite, honesty-critical): establish the mapping between their rally frames and our clip frames. Our eval clip is a ~10s excerpt of the same footage; align via cross-correlation of the 2D ball signals (their actions.ball u/v peaks x their frame grid vs our WASB track normalized to u/v) + sanity via court_points vs our calibration image_pts. Report the offset, correlation strength, and residual uncertainty. If alignment confidence is weak, STOP the frame-level comparisons and do only the alignment-free ones (say so loudly).
2. UNIT/FRAME RECONCILIATION: their court frame (feet, origin/axes inferred from player positions + court dims 44ft x 20ft) -> our metric court frame. Document the inferred transform + its checks.
3. 2D BALL AGREEMENT: on aligned frames, their actions.ball u/v vs our WASB 2D — coverage each, agreement distribution (px at 1920x1080), where each has detections the other lacks.
4. 3D BALL HEAD-TO-HEAD: their court_position (non-interpolated frames only, flag-aware) vs our solved/rendered 3D on aligned frames — per-axis deltas, Bland-Altman (bias + 95% LoA) for height/apex/speed; SEPARATELY compare on (a) our 2 well-fit segments vs (b) our fallback segments. Hypothesis to test: on well-fit segments we are close; on fallback we are absurd while they stay plausible via interpolation+provenance.
5. INTERPOLATION POLICY READ: their interpolated=true fraction + where (occlusions/gaps) vs our fallback fraction — quantify how much of their "great look" is honest interpolation coverage.
6. BOUNCE PSEUDO-GT: for bounce events both systems see, back-project the 2D bounce pixel onto our calibrated court plane = pseudo-GT landing point; score BOTH systems' landing points against it.
7. CAMERA CROSS-CHECK: their solved camera (position/fov, feet) vs our 15-pt PnP calibration (metric) — do the two geometries agree (project our court corners through their model and vice versa)?
8. STATS LAYER GLANCE (bonus, brief): their stats.json/insights.json vs our match_stats.json on comparable quantities (player movement ft vs m, rally span) — one table, no deep dive.

## REPORT
Self-write runs/lanes/w7_pbv_compare_20260709/report.json (lane_report.schema.json structure) + COMPARISON.md with all tables. .venv/bin/python, MPLBACKEND=Agg, numpy/scipy only. Facts; the manager rules.
