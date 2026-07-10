# COURT-DATA-1 — re-solve harvest calibrations on corrected r2 labels + pseudo-label projector

## HARD RULES
- No branches, no commits, no pushes. Manager commits at ruling.
- Read NORTH_STAR_ROADMAP.md (CAL rows, §2.3, standing rules) before acting.
- 4 protected eval clips (eval_clips/**: Burlington/Wolverine/Outdoor/Indoor) are EVAL-ONLY.
  This lane must not read their labels at all. Held-out harvest sources pwxNwFfYQlQ and
  vQhtz8l6VqU stay excluded (existing script constants — preserve).
- Honest reporting. If a bar fails, report FAIL — never bend thresholds (manual 4.8/12.3,
  auto p95 20.0 are FROZEN).
- Run the WIDE test suite at the end with MPLBACKEND=Agg using .venv/bin/python -m pytest
  tests/racketsport (court evals REQUIRE .venv python — anaconda flips results).
- All artifacts under runs/lanes/court_data1_20260709/. Other lanes' run dirs are READ-ONLY
  evidence. data/online_harvest_20260706/court_calibrations/ is READ-ONLY for this lane
  (write re-solved calibrations to YOUR lane dir; manager promotes at ruling).
- Every new CLI ships its direct-CLI reference test same-lane.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/calibrate_harvest_courts.py (additive flag only; default behavior byte-identical)
- NEW scripts/racketsport/project_court_pseudo_labels.py
- NEW tests/racketsport/test_project_court_pseudo_labels.py
- tests/racketsport/test_calibrate_harvest_courts.py IF it exists (extend, don't rewrite)
- runs/lanes/court_data1_20260709/**
Do NOT touch: calibrate_charuco_device.py (live lane), process_video.py, any BODY/decode file,
train_court_keypoint_heatmap.py, evaluate_court_keypoint_owner_gate.py.

## EVIDENCE TO READ FIRST
- runs/lanes/w7_courtkpingest_20260709/gt_corpus_manifest_r2.json — the corrected r2 owner
  court GT (3 usable sources / 5 FULL15 frames). Verify md5s before consuming any file.
- runs/lanes/w7_courtkpingest_20260709/gt_roots/corrected_r2/<src>/labels/court_keypoints.json
- runs/lanes/w4_court_harvestcal_20260707/report.json — the OLD solve: 73VurrTKCZ8
  manual_bar 2.93px median; HyUqT7zFiwk FAILED 7.01/36.2; zwCtH_i1_S4 FAILED 9.28/32.2
  (those solves used PRE-correction labels — that is the motivation for this lane).
- scripts/racketsport/calibrate_harvest_courts.py + threed/racketsport/court_calibration_metric15.py.

## MISSION
1. **Corrected-label re-solve.** Add an additive input mode to calibrate_harvest_courts.py
   (e.g. --corrected-gt-root <dir>) that consumes the r2 per-source court_keypoints.json
   label files instead of the old CVAT XML, leaving every default byte-identical. Re-solve
   ALL sources available in r2 (73VurrTKCZ8, HyUqT7zFiwk incl frame 14564, zwCtH_i1_S4;
   multi-frame sources should use all their frames — report per-frame AND pooled residuals).
   Output solved calibrations + coverage report to runs/lanes/court_data1_20260709/court_calibrations_r2/.
   Also report the 3 owner-rejected sources exactly as the old run did (skip/partial rules
   unchanged). Compare old-vs-new per source in the report.
2. **Pseudo-label projector.** NEW scripts/racketsport/project_court_pseudo_labels.py:
   for every source whose r2 solve reached manual_bar or auto_bar, project the 15 metric
   template keypoints through the solved calibration into frame coordinates for every rally
   clip of that source (data/online_harvest_20260706/rallies/<src>/*.mp4), emitting a corpus
   manifest JSONL under the lane dir: one row per sampled frame (default stride 5, flag
   --stride) with {source_id, clip_path (repo-relative), frame_index, image_size, 15 named
   keypoints px, per-kp in_frame flag, source_grade, calibration_residual_summary,
   projector_version}. Do NOT extract frames to disk (Mac disk is tight); rows reference
   video+frame like the ball corpus.
3. **Static-camera check.** Per clip, verify the static-camera assumption with a cheap
   measurable method (e.g. sparse feature/ECC drift between the calibration reference frame
   and >=8 uniformly sampled frames, measured in px at the court region). Rows from any clip
   whose drift p95 exceeds 2.0px get excluded_reason="camera_motion" (kept in manifest,
   excluded from the default view). Document the method + numbers per clip.
4. **QA overlays.** Render 20 random overlay images per calibrated source (projected court
   skeleton drawn on the frame) to runs/lanes/court_data1_20260709/qa_overlays/<src>/ for
   manager inspection.
5. **Self-consistency test.** In the new test: project template->px on a synthetic-known
   calibration and re-solve from those px; recovered calibration must reproduce projected
   points within 0.1px. Plus a direct-CLI reference test on a tiny fixture.

## ACCEPTANCE (numbers)
- A1: r2 re-solve completes for all 3 usable sources; per-source grade + median/p95 px
  reported vs FROZEN bars; 73VurrTKCZ8 r2 median <= 3.5px (regression sanity vs 2.93).
- A2: old-vs-new comparison table for all 6 original sources (73VurrTKCZ8/HyUqT7zFiwk/
  zwCtH_i1_S4/_L0HVmAlCQI/wBu8bC4OfUY/Ezz6HDNHlnk).
- A3: corpus manifest exists with >=1 calibrated source, schema-validated rows, per-clip
  static-check numbers, and row counts per source/clip in the report.
- A4: self-consistency 0.1px test green; direct-CLI tests green; focused tests green;
  wide suite failures==0 or every failure proven pre-existing (name the proof).
- A5: zero file writes outside {owned scripts/tests, lane dir}.
- KILL/HONEST: if HyUqT7zFiwk or zwCtH_i1_S4 still fail both bars on r2 labels, that is a
  FINDING not a failure — build the corpus from whatever passes and quantify what a 1-source
  corpus means (row counts, viewpoint count).

## BEST-STACK DELTA (mandatory in report)
Expected (c) NO stack delta: this is data preparation; no model/policy default changes.
State it explicitly in the report.

## REPORT
Write the structured report per the output schema (objective_result, acceptance rows with
baseline/after/target/verdict, full_suite counts, honest_issues, artifacts list). Also
append a dated bullet to runs/lanes/court_data1_20260709/HANDOFF.md summarizing corpus
size + calibrated-source count for the training lane that follows.
