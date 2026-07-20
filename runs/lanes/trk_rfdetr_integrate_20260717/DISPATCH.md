# Dispatch: trk_rfdetr_integrate_20260717 — RF-DETR-L production integration (branch 2b, conf 0.18)

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
runs/lanes/trk_rfdetr_integrate_20260717/spec.md — implement EXACTLY. OPERATING POINT resolved: branch 2b_ship_for_demo_at_0.18, conf floor 0.18, RF-DETR-L native 704. Read-first list in the spec (FLIP_DECISION_INPUT, FLIP_PROPOSAL, PREREG_conf030, vm_rerun/report.json, POOLDIAG_PHASE1, orchestrator seam, detbench checkpoint pins, best_stack rev-12 margin-flip honesty template).

## FILE OWNERSHIP
YOURS (per spec fence): threed/racketsport/orchestrator.py (sole owner); models/MANIFEST.json; configs/racketsport/best_stack.json; RUNBOOK tracking-section lines; your named new tests; runs/lanes/trk_rfdetr_integrate_20260717/.
READ-ONLY: scripts/racketsport/process_video.py (you RUN it, never edit), court_*, player_*, event_head/**, list_scaffold_tools.py, ball_arc files.

## CRITICAL NOTES-TEXT CORRECTION (North Star SS5 row 6)
FLIP_PROPOSAL.md's mandated notes text asserts the wolverine ghosts are HIGH-CONFIDENCE detections — P0-I forensics DISPROVED this: the 4 wolverine spectator FPs are the association's own fabricated bridge (f45-86, conf pinned 0.35, runs/lanes/trkL_selection_20260717/GHOST_DIAGNOSIS.md), not detector output. Carry the regression NUMBERS verbatim, but correct the causal claim to the P0-I-consistent statement, and record old-vs-new notes text in your report for manager review.

## SANDBOX REALITY (spec item 3)
The GPU-class production reproduction (both clips within 0.0001) CANNOT run in your sandbox — Mac CPU association is NOT score-faithful (standing finding). Build the full injection path + kill-switch + manifest/stack entries + tests; run honest local checks only (unit tests, schema checks, a tiny CPU smoke clearly labeled non-score-faithful). Emit VM_REPRO_PLAN.md: exact commands/env for the manager's GPU session; targets burlington IDF1 0.922018 / cov4 0.993333 / 0 / 0 / 0, wolverine 0.803625 / 0.723333 / 1 / 4 / 0. Your objective_result for the reproduction bar = NO-ATTEMPT (blocked-on-GPU) — never a fake pass. If the rfdetr package is absent locally, mock the detector interface for unit tests and document install per spec item 4 (+ gpu_fleet snapshot re-bake note text in your report, not the ledger).

## BEST-STACK DELTA
(b): tracking.person_detector rfdetr_large_2026 WIRED_DEFAULT trust_band preview, do_not_promote true, rev 12->13, wolverine regression verbatim-with-correction. STATE EXPLICITLY: the flip is staged in-tree but only counts after the manager's GPU reproduction passes; kill-switch fallback to yolo26m preserved and tested.
