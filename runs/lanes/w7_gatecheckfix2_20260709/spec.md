# LANE w7_gatecheckfix2_20260709 — canonical GATE-1b harness: pass pred_cam_t through the re-ground path

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. THRESHOLDS ARE UNTOUCHABLE: gate_1b <=1mm and mesh-skel <=5mm bars, gate statistics definitions, and eval protocol stay byte-identical — this lane fixes a MEASUREMENT BUG only (manager ruling, BUILD_CHECKLIST [W7 P22 GPU MEASUREMENT DECISIVE 2026-07-09]). Owned files: scripts/racketsport/gate_check_body_decode.py + its tests ONLY. Do NOT touch: threed/racketsport/mhr_decode.py + hmr_deep.py (landed, verified), synthetic_body_decode_gate.py (landed), process_video.py/best_stack.json (tierprov lane), import_w6_labelpack_tasks.py + vite config (micro-debt lane), evaluate_court_keypoint_owner_gate.py (court-kp lane). Artifacts under runs/lanes/w7_gatecheckfix2_20260709/ only.

## THE BUG (verified code fact from the GPU errand — read runs/lanes/w7_p22gate_20260709/ evidence first)
scripts/racketsport/gate_check_body_decode.py:385-393 calls mhr_decode.ground_decoded_camera_frame(...) WITHOUT pred_cam_t= — inside the harness's own re-decode+re-ground path apply_pred_cam_t_once is a no-op. Production (hmr_deep.py, fixed+verified this wave) applies cam_t exactly once, so the harness systematically measures a ~262mm false divergence (raw pred_keypoints_3d grounded WITH cam_t = p95 23.4mm vs the harness's 262.3mm on the identical run — runs/lanes/w7_p22gate_20260709/arm_c_field_quant_report.json).

## OBJECTIVE
1. Fix: the harness's re-ground path reads pred_cam_t from the raw emit it already parses and passes it to ground_decoded_camera_frame with the exactly-once semantics (honor the pred_cam_t_already_applied escape hatch exactly as production does — mirror hmr_deep.py's landed convention, do not invent a new one).
2. Tests: (a) a unit test that FAILS if the harness re-ground path omits pred_cam_t (synthetic fixture with a known nonzero cam_t: harness result must reflect exactly-once application; assert a mutation-detectable numeric outcome, not just "ran"); (b) a regression test that gate_1a computation is bit-unaffected by the change; (c) escape-hatch fixture test (pre-translated raw source -> no double application).
3. HONEST SCOPE: the real fixed-harness GATE-1b re-measurement needs the raw body_mesh monolith (VM-only, 714MB — not local). Do NOT fake it. Your acceptance is fixture-level; state plainly that the live re-measurement rides the next scheduled GPU run.

## SELF-VERIFICATION
MPLBACKEND=Agg: tests/racketsport/test_gate_check_body_decode*.py + your new tests + tests/racketsport/test_mhr_decode.py (import-compat) + scaffold index test. Fix what you introduce; prove pre-existing at HEAD.

## REPORT
Self-write runs/lanes/w7_gatecheckfix2_20260709/report.json (lane_report.schema.json structure): the exact-keys acceptance rows (gate_1b_world_round_trip.*, mesh_skeleton_divergence.*), proof the thresholds/statistics are unchanged (diff summary), changes file:line, full_suite honest, BEST-STACK DELTA (expected none — harness fix), honest_issues, next.
