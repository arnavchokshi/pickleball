# LANE w7_gatecheckfix2_verify_20260709 — ADVERSARIAL VERIFY of the GATE-1b harness pred_cam_t fix

## HARD RULES
Read-only on repo source; scratch mutations under runs/lanes/w7_gatecheckfix2_verify_20260709/ ONLY. No commits. Do not touch other lanes' files (process_video.py/best_stack.json, evaluate_court_keypoint_owner_gate.py, import_w6_labelpack_tasks.py).

## MISSION — try to REFUTE the fix (runs/lanes/w7_gatecheckfix2_20260709/report.json, latest commit)
1. VACUOUS-TEST mutation check: scratch-copy gate_check_body_decode.py; (i) remove the pred_cam_t= argument at the re-ground call; (ii) separately make it double-apply. The lane's new tests must FAIL on both mutants. A test passing on any mutant = vacuous.
2. SILENT-FALLBACK attack (the wave-5 FAST_SAM_PYTHON degrade class): when the raw SAM3D chunk index is ABSENT or unreadable, does the harness quietly compute without pred_cam_t (silently reproducing the old 262mm-class measurement bug) or does it surface the degrade loudly (explicit status/provenance field or hard failure)? Write an executable proof either way. If silent: that is a CONFIRMED defect — report it with the exact code path.
3. THRESHOLD/STATISTIC drift: independently re-derive (not just re-run their grep) that gate_1b/mesh-divergence thresholds, percentile math, and pass/fail logic are byte-equivalent to the pre-fix definitions (git show HEAD~1 diff review).
4. ESCAPE-HATCH abuse: can pred_cam_t_already_applied arrive true for an ordinary raw source via any default/inference (silently skipping application)? Trace where the flag originates in the harness input path.
Verdicts per item: CONFIRMED-VALID / REFUTED(proof) / UNVERIFIABLE. Executable proofs for all allegations.

## REPORT
Self-write runs/lanes/w7_gatecheckfix2_verify_20260709/report.json (lane_report.schema.json structure): verdict table 1-4, proof paths, honest_issues, next.
