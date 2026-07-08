# LANE w6_footpinstub_20260708 — fix the raw-postchain foot_pin stub crash (GPU-proven defect, deterministic)

## HARD RULES (binding)
- NO branches/commits/pushes; no board edits (bullet text in report). Protected clips EVAL-ONLY. .venv/bin/python; MPLBACKEND=Agg. Artifacts under runs/lanes/w6_footpinstub_20260708/ only. Re-grep symbols at HEAD (line numbers drift).

## FILE OWNERSHIP (exclusive)
- OWNED: threed/racketsport/worldhmr.py, threed/racketsport/body_grounding_quality.py (read; edit ONLY if the fix genuinely belongs there), their test files.
- DO NOT TOUCH: threed/racketsport/ball_arc_solver.py + tests (w6_magnus RUNNING), owner-ball-label ingest files incl. tests/racketsport/test_owner_ball_label_ingest.py (w6_labelingest RUNNING), court/calibration files (CALV1 session), cvat_upload/**, web/replay/**.

## THE DEFECT (from the live H100 errand, runs/lanes/w6_gpu_instrument_20260708/arm1_raw_postchain_evidence/)
`--body-postchain raw` (or `--no-body-foot-pin`) crashes AFTER SAM-3D inference: `foot_lock_gate_stream missing artifact_size_policy`. Root cause pinned by the errand: `_bypassed_foot_pin_metrics()` (worldhmr.py ~:736-743) builds a stub foot_lock_gate_stream WITHOUT `artifact_size_policy`; the validator (body_grounding_quality.py ~:121-123) requires that key on every foot_lock_gate_stream. Consequence: raw runs write body_raw_grounded_joints.json but never serialize skeleton3d/body_mesh/phase timing.

## THE FIX (pinned)
- The bypass stub must satisfy the SAME schema the real producer emits: include `artifact_size_policy` (mirror the real producer's value/shape; grep where the non-bypassed path sets it) and audit the stub for ANY other keys the validator or downstream consumers require (do not fix only the one key that happened to crash first — enumerate the validator's required keys and satisfy all of them in the stub, marked as bypassed provenance).
- Do NOT weaken the validator. The loud-degrade contract stands: a raw run must still be unmistakably raw in every summary.

## ACCEPTANCE
1. New regression test: the deterministic CPU fixture run with `--body-postchain raw` (and separately `--no-body-foot-pin`) passes THROUGH the foot_lock_gate_stream validator and serializes the full artifact set (skeleton3d + mesh-index path reachable + bypass summaries) — i.e., the exact path that crashed on GPU now completes on the fixture. This closes the contract-test gap that let the defect through.
2. Existing per-stage bypass tests + gate1b contract tests still green; no-flag path still byte-identical on the fixture.
3. Focused blast-radius suites green (worldhmr, body_grounding_quality, body runner real-path, process_video BODY tests); full wide suite NOT required (tree concurrently dirty; wave-close adjudication is final).

## REPORT (schema-enforced)
objective_result; full_suite (scoped); CHANGES file:line; the enumerated stub-key audit; HONEST ISSUES; proposed bullet; NEXT.
