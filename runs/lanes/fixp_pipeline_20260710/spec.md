# LANE fixp_pipeline_20260710 — pipeline honest-wiring wave (owner deep-review rulings, Wave P)

Ground truth: runs/research_deepreview_20260710/RULINGS.md + runs/lanes/dr_pipeline_20260710/
FINDINGS.md. Read both FIRST. You are the process_video/orchestrator integration owner this wave.

## HARD RULES
- No branches, no commits, no git add (manager commits after ruling).
- FILE OWNERSHIP: scripts/racketsport/process_video.py, threed/racketsport/orchestrator.py,
  threed/racketsport/camera_motion.py, threed/racketsport/virtual_world.py, matching tests under
  tests/racketsport/, and runs/lanes/fixp_pipeline_20260710/**.
  FORBIDDEN: web/**, ios/**, server/**, court files (court_calibration_metric15.py, train_court_*,
  evaluate_court_*, calibrate_harvest_courts.py, run_court_line_keypoints.py — live court wave),
  scripts/racketsport/import_w6_labelpack_tasks.py, scripts/racketsport/ingest_owner_ball_labels.py
  + their tests (owned by a concurrent labeling session, uncommitted edits — do not touch or revert),
  runs/manager/**.
- Protected clips: Outdoor/Indoor labels NEVER. Burlington/Wolverine artifacts OK.
- Every behavior ships a red→green test same-pass. End with the WIDE suite
  (MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport tests/server -x-less; report full
  pass/fail counts; failures must be proven pre-existing).
- Honest reporting. No promotion claims; VERIFIED=0.

## MISSION
1. FAIL-LOUD STACK ASSETS: `_attempt_global_association` silently degrades when opts.reid_model is
   missing (process_video.py:1589-1597). Ruling: a missing best-stack-pinned asset must either
   (a) fail the tracking stage loudly, or (b) proceed ONLY with an explicit machine-readable
   degradation record (stage summary + PIPELINE_SUMMARY 'degraded_reasons' + manifest surface),
   defaulting to (a) for stack-pinned assets. Audit for the same pattern on other stack-pinned
   assets (court model, calibration curves, ball ckpt) and normalize. Tests: missing-asset run
   raises/records exactly as specified.
2. MANIFEST REPRESENTATION TRUTH: `skeleton_only` is currently chosen whenever mesh artifacts are
   absent, without checking skeleton3d.json (process_video.py:3819-3835). Add honest states
   (`body_missing` / `track_only` / `skeleton_only` / `mesh`) derived from validated world counts +
   actual artifact presence; bundle stays `partial` when neither skeleton nor mesh exists for a
   tracked player. Align with server bundle_policy semantics (read-only; do not edit server/**).
3. SIDE-EFFECT-FREE REVIEW WRITERS: after BODY failure, `_write_best_effort_review_artifacts`
   regenerates/overwrites authoritative frame_compute_plan.json + body_compute_execution.json
   (orchestrator.py:2756-2838). Move review outputs to namespaced review sidecars (e.g.
   *.review_after_failure.json); authoritative stage outputs become immutable after their stage
   completes. Test: failed-BODY run leaves the original plan byte-identical + review sidecar exists.
4. CAMERA_MOTION FRAME REBASE: reference frames from parent-video calibration indices fail on
   excerpts ("reference frame 109050 outside processed frame range", camera_motion.py:88-102,
   1495-1505). Fix: detect out-of-range reference, remap via clip provenance if available, else
   degrade to explicit `camera_motion_unavailable(reason=unrebased_parent_reference)` instead of
   stage failure. Tests for both paths.
5. FAIL-CLOSED VERDICT PROVENANCE: persist the per-segment fail-closed verdict map (computed in
   apply_ball_track_arc_solved_overlay, virtual_world.py:371-558) into a world-adjacent sidecar
   (e.g. ball_fail_closed_verdicts.json) + reference it from the world summary WITHOUT breaking the
   strict world schema. Test: sidecar content equals ball_arc_render summary verdicts.
6. (bounded) 1200-CAP VISIBILITY: do NOT change the cap. Emit an explicit per-run exclusion record
   (count + frame ids excluded by the cap) into the frames-stage summary so NS-04.2 can rule on
   policy with data. Test: zwcth-class plan reports 115 excluded indices.

## REPORT (runs/lanes/fixp_pipeline_20260710/)
report.json via schema; per item implemented/partial/skipped + test names + file:line; wide-suite
counts; HONEST ISSUES; dated bullet for the manager. BEST-STACK DELTA: (b)? NO — these are
correctness/honesty fixes, no stack entry flips: state (c) none, with one exception — if item 1
lands as fail-loud default, note it as a BEHAVIOR default change (not a model promotion) for the
manager's best_stack provenance note.
