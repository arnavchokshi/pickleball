# Dispatch: static_cal_firstlock_20260717 — static single-lock CAL + pooling + k1 fix

## HARD RULES (all lanes)
- NO git branches, NO commits, NO pushes. The manager commits after ruling on your report.
- Read first: NORTH_STAR_ROADMAP.md (SS2.1 + SS5 queue), AGENTS.md, the RUNBOOK section for your area.
- The 4 protected eval clips are EVAL-ONLY (Burlington/Wolverine internal scoring allowed; Outdoor/Indoor labels NEVER). The 50-row owner event seed is PROTECTED EVAL, never training.
- VERIFIED=0 is binding. Preview band. Nothing you produce is a promotion. Honest reporting: misses are misses; no threshold shopping — pre-registered values are THE values.
- Sandbox limits: no network/DNS, no MPS (CPU fallback), no localhost binds, no xcodebuild. Anything needing GPU/network = a documented handoff plan in your lane dir, never faked.
- Run your focused tests AND the wide blast-radius suite (MPLBACKEND=Agg, real exit codes); attribute every failure: yours vs pre-existing (known pre-existing: sandbox socket-bind failures).
- Artifacts under YOUR lane dir only. Other lanes run dirs are READ-ONLY evidence.
- Final message = the schema-enforced structured report: objective_result PASS/FAIL/NO-ATTEMPT vs the pre-registered numbers, full_suite counts + attribution, HONEST_ISSUES, artifacts, BEST-STACK DELTA (a promotes / b pending-dormant entry / c none + why).
- CONCURRENT LANES ARE LIVE (file-disjoint — do NOT touch their files): trkL_selection_impl_20260719 (new player_selection files + list_scaffold_tools additive), static_cal_firstlock_20260717 (court_calibration.py, court_line_keypoints.py, process_video.py calibration seams), trk_rfdetr_integrate_20260717 (orchestrator.py, models/MANIFEST.json, configs/racketsport/best_stack.json, RUNBOOK tracking lines), event_head_corpus_20260719 (threed/racketsport/event_head/**, scripts/racketsport/eval_event_head.py, scripts/racketsport/train_event_head.py).

## YOUR FULL SPEC
runs/lanes/static_cal_firstlock_20260717/spec.md — read it fully and implement EXACTLY (SS3 design: single-lock path, cross-frame line-evidence pooling, zero-distortion k1 config fix, static-consistency guard). Code seams are verified-present in spec SS2.

## FILE OWNERSHIP
YOURS: threed/racketsport/court_calibration.py; threed/racketsport/court_line_keypoints.py; scripts/racketsport/process_video.py ONLY the calibration seams (_stage_calibration, _court_calibration_needs_correction, _court_line_evidence_ready, _missing_required_court_line_count + minimal private helpers; you are the SOLE process_video.py owner right now — keep the diff surgical); new court_lock schema/build code + tests (tests/racketsport/test_static_cal_*.py, new files); runs/lanes/static_cal_firstlock_20260717/.
DO NOT touch scripts/racketsport/list_scaffold_tools.py or configs/racketsport/best_stack.json — if you add a CLI or want a stack entry, put registration/entry text as inline diff hunks in your report for the manager.

## DATA
pb.vision demo: eval_clips/ball/pbvision_11min_20260713/ incl. labels/court_calibration_metric15pt.json (metric_15pt_reviewed seed, commit 1075cee57). Owner semantics: unmarked review points = explicitly not-in-frame. EVAL-ONLY. Raw per-frame solves/evidence IMMUTABLE — court_lock.json is a separate refinement artifact with provenance+covariance. Static internal clips: committed artifacts under eval_clips/ and runs/ evidence.

## ACCEPTANCE = spec.md SS4 verbatim, scored PER criterion (adopt/reject/partial each):
1 reuse-lock residual within ~1px of per-frame; 2 in/out stops abstaining on the demo (19.16px -> toward ~2.6px, metric_confidence above abstention threshold); 3 no metric_15pt_reviewed authority regression; 4 static-guard flags a synthetically panned clip and NOT the static clips; 5 determinism + selection-OFF byte-identical + immutability.

## BEST-STACK DELTA
(b) expected: propose a PENDING default-OFF entry for the static-lock path as inline text in your report; do NOT edit best_stack.json (owned by trk_rfdetr_integrate).
