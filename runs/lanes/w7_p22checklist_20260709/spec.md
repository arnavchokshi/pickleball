# LANE w7_p22checklist_20260709 — P2-2 decode-fidelity R1 CHECKLIST (NOT archaeology)

## HARD RULES
No branches. No commits (manager commits). Protected clips: Outdoor/Indoor labels NEVER touched (no ledger row exists); Burlington/Wolverine internal scoring OK. Honest reporting; .venv/bin/python; MPLBACKEND=Agg for suites. Do NOT edit: scripts/racketsport/process_video.py, configs/racketsport/best_stack.json, tests/racketsport/test_process_video.py, tests/racketsport/test_best_stack_manifest.py (paddle lane owns); web/**, server/** (other lanes); third_party/** (vendor pins — propose-diff ONLY, never edit). Other lanes' run dirs are read-only evidence; your artifacts go under runs/lanes/w7_p22checklist_20260709/ only. lambda_foot=0, smoother UNWIRED, latent-interp OFF all stay untouched.

## OBJECTIVE
Run the R1 decode-fidelity CHECKLIST (runs/research_w6refresh_20260709/RULINGS.md R1 — read it first, it is binding) against the GATE-1b legitimate FAIL: gate_1b_world_round_trip 262.35mm (target <=1mm) and mesh_skeleton_divergence 53.50mm p95 (target <=5mm), evidence runs/lanes/w6_close_errand_20260708/gate1b_raw_arm_report.json + gpu_instrument_wolverine_mixed_0200_raw_postchain/. This is an enumerated checklist with a ceiling rule, not open-ended debugging.

## CHECKLIST (each item = an acceptance row with evidence)
(a) AUDIT our decode path (threed/racketsport/mhr_decode.py + the callers that build world skeletons/meshes) against vendored MHR conversion.py@4debaacf L472-516: the 100x cm/m scale must hit BOTH pred_vertices AND pred_cam_t at the same point; the axis-flip ([:,[1,2]]*=-1) is branch-dependent. Report file:line for every scale/axis application on our side, with a verdict per application (correct/incorrect/ambiguous).
(b) PROVE pred_cam_t is added EXACTLY ONCE end-to-end (trace + a unit test that fails on double/missing application).
(c) ADJUDICATE which field the harness reads as "skeleton": pred_keypoints_3d lacks central spine joints vs 127-joint pred_joint_coords (upstream GH issue #34). Quantify OFFLINE from the banked w6 raw-arm artifacts how much of the 262.35mm / 53.50mm is field-definition mismatch (recompute divergence under the alternate field if the banked sidecars allow; if not, say so honestly).
(d) The world-skeleton placement formula is OUR extrapolation (conversion.py never reads joint fields) — audit it as such; document the exact formula and its assumptions in the report.
(e) BUILD the SYNTHETIC render round-trip gate as a standing CLI instrument: authored mesh with known scale/pose/camera -> render -> SAM-3D-Body -> decode -> measure (MetricHMSR recipe). New CLI under scripts/racketsport/ (your naming), registered in the scaffold index, with a direct-CLI reference test, runnable CPU for small N (a GPU arm is a separate manager errand — design the CLI so the same command runs on the VM).
(f) CEILING-RULE VERDICT (report-only): if (a)-(d) clear and the residual is family-normal (~50mm p95; MetricHMSR 3DPW PVE 62.7mm), recommend the WORKAROUND switch (per-track identity/scale locking + latent-space smoothing per arXiv:2512.21573) + a PROPOSED gate recalibration. Do NOT change any gate threshold or eval protocol in code — that is a manager+owner ruling. If instead you find a real decode defect, you have bounded fix authority in threed/racketsport/mhr_decode*.py + its tests; a fix needing vendor or fenced files = STOP and report the proposed diff inline (no .patch files).
(g) 30-sec check (R5d): is arXiv 2603.15603 the same lineage we benched in w5_fastbody? One line + link in the report. NOTE: your sandbox has NO network — if you cannot resolve this offline from runs/lanes/w5_fastbody_bench_20260708/ evidence, mark it UNRESOLVED-needs-network; do not guess.

## EVIDENCE TO READ FIRST
runs/research_w6refresh_20260709/RULINGS.md (R1) · runs/lanes/w6_close_errand_20260708/gate1b_raw_arm_report.json · runs/lanes/w6_gatecheckfix_20260708/ (canonical harness lane) · scripts/racketsport/gate_check_body_decode.py · runs/lanes/w5_p22latent_20260707/report.json.

## OWNED FILES
threed/racketsport/mhr_decode.py (+ siblings mhr_decode*.py), NEW scripts/racketsport/<synthetic gate CLI>.py, tests/racketsport/test_mhr_decode*.py + new CLI test, scaffold index registration line. Nothing else.

## SELF-VERIFICATION (MANDATORY)
Full blast-radius suite before declaring done: MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_mhr_decode.py tests/racketsport/test_gate_check_body_decode*.py <your new tests> tests/racketsport/test_body_runner_real_path.py -q, plus the scaffold-index test. Fix every failure you introduced; prove pre-existing ones at HEAD. Known concurrent-lane churn: process_video/best_stack test files may be dirty (paddle lane) — classify those CROSS-LANE-SUSPECT, do not fix them.

## REPORT
Structured JSON self-written to runs/lanes/w7_p22checklist_20260709/report.json (lane_report.schema.json field structure): objective_result, acceptance rows (a)-(g) with EXACT metric keys (gate_1b_world_round_trip.joints_world_p95_abs_error_mm, mesh_skeleton_divergence.p95_mm), changes file:line, full_suite honest, BEST-STACK DELTA (any decode fix = PENDING entry proposal with named gate; no promotion), honest_issues, next. Iterate until acceptance passes or a genuine blocker — never tune a threshold to pass.
