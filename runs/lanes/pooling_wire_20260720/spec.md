# pooling_wire_20260720 — wire cross-frame line pooling into the calibration evidence seam (default-OFF)

Codex gpt-5.6-sol ultra. The farline diagnostic (runs/lanes/farline_diag_20260720/{RESULTS.md,FINAL_RECORD.json}) PROVED T14's cross-frame pooling recovers the gate-blocking far_centerline on the real Drill clip (4/96 baseline → 63 support frames, 0.357px p90, counterfactual auto_calibration_ready=TRUE). Wire it so a real pipeline run can use it. Default-OFF; ultra re-review before commit; then ONE GPU replay re-run proves it end-to-end. VERIFIED=0.

## HARD RULES
- NO commits/pushes. Ultra re-review before commit. Default-OFF: flag OFF ⇒ byte-identical behavior (golden test). No gate-gaming: do NOT loosen the evidence-readiness bar — pooling must EARN readiness by producing real accepted lines at the existing bar. Raw per-frame evidence IMMUTABLE (pooled = separate artifact with provenance, per the static_cal spec's design that was reviewed clean on this point).
- Focused + wide suite (MPLBACKEND=Agg), attribute failures. Honest reporting.

## YOUR FILES: scripts/racketsport/process_video.py (ONLY the calibration/court-line-evidence seam: where court_line_evidence readiness is computed pre-tracking), threed/racketsport/court_line_keypoints.py + the T14 modules already in the tree (court robustness/pooling), the farline_diag lane's proven pooling routine (reuse its exact logic — it is the evidence-backed reference), new tests, lane dir. Nothing else.

## DESIGN
1. New flag (config/CLI, default OFF): when ON, after per-frame line evidence collection, run the PROVEN pooling routine over the sampled frames (static-camera assumption; reuse the diagnostic's parameters exactly — no retuning) and emit court_line_evidence_pooled.json (separate artifact: contributing frames, per-line support, residuals, provenance).
2. The evidence-readiness check may consume pooled lines ONLY as additional accepted-line evidence at the SAME acceptance bar; provenance marks pooled lines as pooled_static. The calibration solve then proceeds exactly as today.
3. A static-consistency guard: if per-frame drift across the sampled frames exceeds the diagnostic's dispersion bound, pooling ABSTAINS (typed) — never pools a moving camera.
4. Tests: OFF byte-identical; ON with the Drill clip's pulled evidence fixtures reproduces the diagnostic's recovery (63 support, gate-ready) from fixtures (no video decode in tests); moving-camera fixture abstains; determinism.

## ACCEPTANCE: OFF byte-identical proven; ON reproduces the diagnostic numbers from fixtures; readiness bar UNCHANGED (assert no threshold edits — diff must show none); focused + wide suite attributed. Emit RERUN_CMD.md: the exact process_video command (flag ON) for the GPU replay re-run on the Drill clip.
