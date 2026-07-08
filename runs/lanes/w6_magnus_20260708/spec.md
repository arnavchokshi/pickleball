# LANE w6_magnus_20260708 — P1-4b STEP 2: add scalar Magnus S to the arc-solver fit (wave-6 queue #3; TECH_BLUEPRINTS BALL 3D pillar STEP 2, UNLOCKED by BVP span v2)

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes. Working-tree changes only in your OWNED FILES. Manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or runs/manager/ boards — proposed bullet text goes in your report.
- Protected eval clips are EVAL-ONLY. Burlington/Wolverine internal scoring allowed. Outdoor/Indoor LABELS are NEVER read without a pre-registered heldout_eval_ledger.md row — the D.3 gates below use label-free per-segment status/endpoint_error/reprojection metrics plus the internal-val F1 suite exactly as the w5_bvpspan lane measured them; if any step seems to need held-out labels, STOP and report.
- Honest reporting; PASS with full_suite.failed>0 not proven pre-existing = rejected.
- .venv/bin/python; importorskip("torch"); MPLBACKEND=Agg on wide runs.
- Artifacts under runs/lanes/w6_magnus_20260708/ ONLY. Other lanes' run dirs READ-ONLY.
- ALL line numbers below have DRIFTED — re-grep function names at HEAD before editing.
- Kill criteria are COMMITMENTS: if one fires, the answer is what this spec says, never threshold tuning.

## FILE OWNERSHIP (exclusive this wave)
- OWNED: threed/racketsport/ball_arc_solver.py + its test files + runs/lanes/w6_magnus_20260708/**.
- READ-ONLY: threed/racketsport/flight_simulator.py (you COPY constants/logic from it; NEVER import solver<-simulator; never edit it), runs/lanes/w5_bvpspan_20260707/** + runs/lanes/w5_bvpspan_verify_20260707/** (measurement precedent), runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json (frozen D.3 baselines).
- DO NOT TOUCH: process_video.py + remote_body_dispatch.py (w6_gate1b_knob lane), CAPABILITIES.md + train_ball_stage2.py (w6_instrudocs lane), cvat_upload/** (w6_labelpack lane), web/replay/**, ios/**.

## OBJECTIVE
Let the solver fit ONE extra scalar S per segment so top/backspin arcs stop being free-fit approximations. Port EXACTLY the simulator's proven Magnus derivative — do NOT invent new physics. STEP 1 (BVP span protection v2) landed 792fa5fc6 with fresh D.3(e) floors Burlington 0.7727272727 / Wolverine 0.875000 — those floors are your regression baseline.

## THE DESIGN (pinned, from TECH_BLUEPRINTS BALL 3D STEP 2 — do exactly this, no re-litigation)
- State extension: NONE. Keep 6-state (x,y,z,vx,vy,vz). S is a per-segment CONSTANT parameter, not a 7th integrated state.
- Spin axis FIXED per segment: `axis = (vy0/norm, -vx0/norm, 0)` from the segment's initial horizontal velocity (copy `_spin_axis_for_velocity`, flight_simulator.py ~:902).
- Force term (copy from `_rk4_step_with_magnus`, flight_simulator.py ~:729): inside `deriv` of a new `_rk4_step_magnus` in ball_arc_solver.py (mirror `_rk4_step` ~:3922), after drag accel add lift: `lift_dir = unit(cross(axis, v_hat))`; `lift_k = 0.5*rho*pi*r^2/mass`; `lift_acc = lift_k * speed^2 * (0.195 * S)`; add `lift_acc*lift_dir` to (ax,ay,az). Define `STEYN_CL_PER_SPIN=0.195` in the solver too (do NOT import from simulator — solver must not depend on it).
- S-threading DECISION (exact): add `spin_scalar: float = 0.0` keyword arg to `_integrate_positions` (~:3888) AND `_rk4_step` (~:3922) plus the new `_rk4_step_magnus`; route to the magnus stepper when `abs(spin_scalar)>1e-12`, else plain `_rk4_step` (analytic no-drag shortcut only for spin_scalar==0.0). Thread spin_scalar ONLY from the three fit paths that OWN a segment S (`_fit_free_flight_segment_once` ~:611, `_refine_bvp_endpoints` ~:963, `_solve_bvp_shooting` ~:845); ALL other ~15 `_integrate_positions` call sites pass the default 0.0. Do NOT put S on `PhysicsParameters` (shared object -> cross-segment spin leakage).
- Fit change: S is the extra least-squares parameter in ALL THREE paths a segment can traverse — the velocity fit in `_fit_free_flight_segment_once`, the [dp0,dp1,dt0,dt1] vector in `_refine_bvp_endpoints`, and `_solve_bvp_shooting` must receive the current S so its integration matches (every path carries the SAME S). Init S0=0.0. BOUNDS: `|S| <= 0.8`, hard-clip. Regularize toward 0: residual `sqrt(lambda)*S` with `lambda=0.05` (spin must not absorb detector noise on short/side-view segments).
- Gate: only FIT S when segment has >= 6 inlier observations AND view-geometry confidence is not back-view-degraded; STEP 5 (per-segment view confidence) has NOT landed, so use the interim proxy: require >= 8 inliers.
- Config flag `fit_spin_scalar` (default per acceptance outcome — see kill criterion) so S can ship dormant.

## ACCEPTANCE (exact, all measured; use the SAME measurement path the w5_bvpspan lanes used — read their spec/report first and reuse their harness invocations)
(a) Owned pytest green INCLUDING a new test: a known-S simulator trajectory (`generate_trajectory_pair`, spin_scalar=0.5) round-trips through the solver to recovered `|S_hat - 0.5| <= 0.15`.
(b) On the 3-clip set eval_clips/ball/{burlington_gold_0300_low_steep_corner, wolverine_mixed_0200_mid_steep_corner, outdoor_webcam_iynbd_1500_long_high_baseline}: `reprojection_rmse_px` mean does NOT increase on any clip, and IMPROVES on the steepest-launch high-arc segment wolverine_mixed_0200 seg6.
(c) D.3(a)/(b)/(c)/(d) all still PASS (frozen baselines: the acceptance list of runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json + the w5_bvpspan_verify floors; protected spans delta must stay 0.0).
(d) Internal-val F1 no >1pt regression — the ONLY gating scorer is `label_f1_at_20px` from run_ball_tracking_eval_suite (the training-harness proxy f1_at_20px NEVER gates). Fresh floors to hold: Burlington 0.7727272727 / Wolverine 0.8750.
(e) DIAGNOSTIC ONLY (no bar, no tuning): report the img1605 arc census row (eval_clips/ball/owner_IMG_1605_8a193402780b) before/after — the anchor-starvation quality gap is on watch; you report the number, you do NOT chase it.
(f) FULL wide blast-radius suite green (MPLBACKEND=Agg; court benchmark split standalone if needed).
If any leg STRUCTURALLY requires a fresh GPU run (not cached artifacts), complete every offline-measurable leg, mark PARTIAL, and report the exact GPU command needed — GPU spend is predictor-gated by the manager.

## KILL CRITERION (a commitment)
If enabling S regresses ANY D.3 gate or inflates reprojection rmse on >=1 clip: default S OFF (`fit_spin_scalar=False`, S pinned 0), ship the plumbing DORMANT, and record which segments benefited. Do NOT attempt 3-axis spin (UNIDENTIFIABLE — frozen ruling). Do NOT attempt spin-SIGN disambiguation (parked on H13).

## REPORT (schema-enforced)
objective_result vs (a)-(f); full_suite line; per-clip/per-segment before-after table (rmse, D.3 statuses, endpoint errors, S values fitted with inlier counts); img1605 census row; CHANGES file:line; HONEST ISSUES; proposed BUILD_CHECKLIST bullet; NEXT.
