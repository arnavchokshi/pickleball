# LANE w7_p22_verify_20260709 — ADVERSARIAL VERIFY of w7_p22checklist (PRE-STAGED; dispatch after its report.json lands)

## HARD RULES
Read-only on repo source except: you may RUN the lane's new synthetic-gate CLI and tests. Your defect proofs live under runs/lanes/w7_p22_verify_20260709/ ONLY. No commits. Do not touch process_video.py/best_stack.json (paddle lane) or web/, server/.

## MISSION
Attack the R1 checklist verdicts in runs/lanes/w7_p22checklist_20260709/report.json before the manager rules on decode-defect vs ceiling-rule. Fraud classes to hunt:
1. VACUOUS TESTS: mutation-check the pred_cam_t exactly-once test — introduce a deliberate double-application in a scratch copy (runs/lanes/w7_p22_verify_20260709/ scratch, never repo) and confirm the lane's test FAILS on it. A test that passes on both = vacuous (wave-4 class).
2. FIELD ADJUDICATION MATH: independently recompute the claimed field-mismatch contribution to 262.35mm from the banked w6 raw-arm artifacts. If the lane says "not computable offline", verify that claim (are the sidecars really insufficient?).
3. SYNTHETIC GATE HONESTY: the authored-mesh ground truth must be INDEPENDENT of the decode path under test (no importing the decoder to author the truth — self-referential gate = unfailable). Trace the CLI's data flow; run it CPU small-N; confirm it CAN fail (feed it a deliberately mis-scaled input and watch it fail).
4. CEILING-RULE VERDICT: if the lane recommends the workaround, check every checklist item (a)-(d) has POSITIVE evidence of clearing (file:line verdicts), not absence-of-evidence. A "cleared" audit row with no traced application = not cleared. If it claims a defect fix, verify the fix moves gate_1b_world_round_trip.joints_world_p95_abs_error_mm / mesh_skeleton_divergence.p95_mm on a real recompute (CPU-feasible subset OK; GPU rerun = flag for the manager errand).
5. SCOPE CREEP: confirm no gate thresholds/eval protocol changed in code and lambda_foot/smoother/latent-interp remain untouched (git diff scan).
Executable proofs for every allegation. Verdict per item: CONFIRMED-VALID / REFUTED (proof) / UNVERIFIABLE.

## REPORT
Self-write runs/lanes/w7_p22_verify_20260709/report.json (lane_report.schema.json structure): verdict table 1-5, proof paths, honest_issues, next. Repairs are scored by YOUR unmodified harness.
