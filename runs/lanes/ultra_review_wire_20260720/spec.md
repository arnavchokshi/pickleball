# ultra_review_wire_20260720 — final review: pooling wired into the calibration seam (commit gate)

REVIEW-ONLY, gpt-5.6-sol ultra, tight scope. The pooling_wire_20260720 lane wired the PROVEN farline pooling (4/96→63 frames, 0.357px on the real Drill clip) into process_video's calibration evidence seam, default-OFF. Its self-report: OFF byte-identical PASS, fixture-reproduced recovery PASS, thresholds-unchanged PASS, focused+wide PASS. Verify before commit.

VERIFY (live tree — process_video.py calibration seam, court_line_keypoints.py, the pooling module, new tests):
1. Default-OFF truly byte-identical (trace the flag; no behavior leak when OFF).
2. NO threshold/gate loosening anywhere in the diff — pooled lines must pass the SAME acceptance bar; readiness logic unchanged except accepting pooled evidence as additional lines with pooled_static provenance.
3. Raw per-frame evidence immutable; pooled output a SEPARATE artifact with contributing frames + residuals + provenance.
4. Static-consistency guard: pooling abstains (typed) on drift; confirm the guard is real and its bound comes from the diagnostic, not invented.
5. The fixture test genuinely reproduces the diagnostic's recovery (63 support, gate-ready) — not a mock that would pass vacuously.
6. RERUN_CMD.md's command is correct for the GPU re-run (flag ON, Drill clip, frozen stack otherwise).

OUTPUT: verdict COMMIT_OK / COMMIT_WITH_FIXES (exact) / DO_NOT_COMMIT + file:line. One line to the manager.
