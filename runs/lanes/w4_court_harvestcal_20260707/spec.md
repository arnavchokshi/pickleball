# LANE w4_court_harvestcal_20260707 — per-source harvest court calibration from owner CVAT court-kp labels

## OBJECTIVE
Turn the owner's court-keypoint labels (task 13 export: 49 usable points across 5 frames; sources
graded 3 full / 2 tennis-overlay partial / 1 declared-skip) into FROZEN per-source court
calibrations for harvest footage, plus a coverage map over the 40 prelabeled rally clips. This is
wave-4 queue #4: it structurally unlocks the physics-gated ball chain (deferred SST teacher) on
covered harvest clips — the 3D chain CLIs hard-require court calibration. Harvest sources are
static tripod games: ONE calibration per source video covers all its rally clips.

## EVIDENCE TO READ FIRST (state each as a CHECK result in your report)
- `cvat_upload/court_keypoints_20260707/`: `label_spec.json` (keypoint schema), `taskset_manifest.json`,
  `package_manifest.json`, `packages/`, `frames/`, `import_report.json`, `validation_report.json`,
  `OWNER_COURT_KP_GUIDE.md`, `import_court_kp_tasks.py` — understand exactly which keypoints exist
  per source/frame and their template correspondence.
- The owner review README (git log — commit b451f4cf6 "review README (3 full / 2 tennis-overlay
  partial / 1 declared-skip w/ stray-drop rule)"): apply its stray-drop rule literally.
- `threed/racketsport/court_calibration_metric15.py` (`fit_single_view_metric_camera` ~:303 —
  re-grep) + the court template keypoint definitions (`court_templates.py`). CHECK: the minimum
  point count/configuration the metric solver needs; whether tennis-overlay partial sources meet it.
- Rally-clip↔source mapping: `data/online_harvest_20260706/manifest.json` + `prelabels/` (40 clips)
  + `rallies/`.

## DESIGN (pinned shape)
NEW `scripts/racketsport/calibrate_harvest_courts.py`:
1. Per SOURCE video: gather its labeled frames' points from the imported export (apply the
   stray-drop rule; drop the declared-skip source).
2. Solve per-frame metric calibration via the EXISTING metric solver (consume
   `court_calibration_metric15.py` as a library — zero-distortion default; do NOT modify court
   calibration code, do NOT invent a new solver).
3. Cross-frame consistency per source (static tripod ⇒ solved cameras should agree): report
   translation/rotation/focal spread across that source's frames; pick the frame(s) whose solve has
   the best reprojection as the frozen calibration (document the selection rule you implement).
4. Emit per-source artifact `data/online_harvest_20260706/court_calibrations/<source_id>.json`:
   the frozen calibration + provenance (export path, frames used, points used, stray-drops applied)
   + per-frame reprojection stats + `calibration_grade` (below).
5. Emit `data/online_harvest_20260706/court_calibrations/coverage_report.json`: ALL 40 prelabeled
   rally clips → {source_id, calibrated: bool, grade}.
6. Report (do NOT wire) the chain handoff: CHECK how `scripts/racketsport/run_ball_chain.py`
   receives court calibration for a clip today (grep its CLI/args); write in your report the exact
   invocation shape a GPU lane would use to run the physics-gated chain on a covered harvest clip
   with your artifact. If no external-calibration input exists, say exactly what seam is missing.
   You do NOT edit run_ball_chain or any pipeline file.

## ACCEPTANCE (exact keys, per source, in the report table)
- Full-labeled sources: reprojection `median_px <= 4.8` AND `p95_px <= 12.3` on that source's
  labeled points → `calibration_grade = "manual_bar"`.
- Between that and `p95_px <= 20.0` → `calibration_grade = "auto_bar"` (usable for teacher-gating
  trials, flagged).
- Worse, or solver-infeasible (insufficient points) → `calibration_grade = "failed"` with the reason.
- Coverage: the 40-clip table, with an honest count of clips covered at each grade.
- Unit test(s) for the point-gathering + grading logic (fixture-based); the real solve runs as the
  lane's measured execution (deterministic, CPU).
- Full blast radius: `.venv/bin/python -m pytest tests/racketsport/test_scaffold_tool_index.py
  <your new test file> -q` plus any test file your grep shows touching
  `court_calibration_metric15` consumers you altered (you should be altering NONE — consume-only).
  Register the new CLI in the scaffold index in this SAME lane.

## KILL (commitments)
- <2 sources reach even `auto_bar` → the physics-gated teacher stays deferred: bank the negative
  with per-source evidence; that is a PASS for this lane if honestly measured.
- Points insufficient for the metric solver on a source → grade it `failed` with the count; do not
  bend the solver or invent correspondences.

## OWNED FILES (anti-collision fence)
NEW: `scripts/racketsport/calibrate_harvest_courts.py`, `tests/racketsport/test_calibrate_harvest_courts.py`,
`data/online_harvest_20260706/court_calibrations/` (new dir), your lane dir. READ-ONLY: everything
under `cvat_upload/court_keypoints_20260707/`, `court_calibration_metric15.py`, `court_templates.py`,
`run_ball_chain.py`. DO NOT TOUCH: `process_video.py`/`orchestrator.py` (fenced), `ios/**`,
`runs/manager/**`, eval labels, held-out ledger, the two HARVEST held-out reservations
(`pwxNwFfYQlQ`, `vQhtz8l6VqU` — if any labeled frame traces to them, that is a defect: STOP and report).

## DISCIPLINE
`.venv/bin/python`; no git branch/commit/push; no network; no new root-level .md;
`pytest.importorskip("torch")` if torch sneaks into a test (it should not); prove pre-existing
failures fail at HEAD.

## STRUCTURED REPORT
Acceptance table: per-source grade + median/p95 reproj + frames/points used; the 40-clip coverage
count; the chain-handoff invocation (or the missing-seam statement). honest_issues: labeling
quality problems you noticed (feed the owner's next court-kp session). next: what unlocks when
more sources are labeled.
