# Lane event_head_corpus_20260719 — event-head scale-up CODE LEG (Levers 2+3 + mandatory fixes)

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

## YOUR FULL CONTEXT
runs/lanes/event_head_pretrain_20260716/SCALE_UP_SPEC.md — read fully. You implement the LOCAL CODE portions (SS2 Levers 2-3, SS4 fixes 1/4/5). Lever 1 (VM-side video staging) and the training run are the MANAGER'S GPU leg — not yours; do not attempt network.

## FILE OWNERSHIP
YOURS: threed/racketsport/event_head/** (esp. datasets.py); scripts/racketsport/eval_event_head.py; scripts/racketsport/train_event_head.py (provenance guard + DataLoader workers only); their tests; runs/lanes/event_head_corpus_20260719/.
READ-ONLY: everything else, incl. the protected 50-row owner event seed (its in-code EVAL-ONLY hard-fail must remain intact and tested).

## MISSION
1. Multi-window-per-row extraction: sliding stride 32 over each row's full num_frames (replaces the events[0]-only single window); keep source-disjoint splits + loss-masked union semantics exactly; REBALANCE splits to ~70/15/15 by parent video (current 226 train / 282 val is backwards); determinism test — byte-identical manifest for identical config.
2. DataLoader num_workers + prefetch_factor over the existing on-the-fly decode contract (NO frame cache — disk rule stands). Decode-bound 20% GPU util is the target defect.
3. eval_event_head.py window fix: window param defaults to the checkpoint's train_manifest.config.window_frames; assert-match at load (loud failure on mismatch). PROVE on the existing 07-16 checkpoint: reproduce the 9 TP / 0 FP matched-window result through your FIXED CLI (evidence: runs/lanes/event_head_pretrain_20260716/eval/matched_window64_eval.json; harness logs/matched_window_eval.py).
4. Guard the unguarded git rev-parse in train_event_head.py provenance (safe on a no-.git mirror).
5. AppleDouble hygiene at tar/glob ingest points (strip ._* / COPYFILE_DISABLE note).
6. Training-lane standing acceptance: push ONE sample through the training dataloader AND the production inference preprocessor; assert identical tensors or document the stamped mapping.

## ACCEPTANCE
- Measured window count on the CURRENT staged 226 media-present train rows via the new extractor (report the multiplier; projection says x3-12 local, x68 post-Lever-1).
- Split rebalance documented by parent video; builder totals still reconcile EXACTLY (33,791 / 4,271 / 36,484).
- Matched-window eval reproduced via the fixed CLI; determinism gate green; focused + wide suite with attribution.
- Emit VM_RUN_PLAN.md: updated exact VM-side sequence for the manager (Lever 1 h264 fetch + decode-verify per SCALE_UP_SPEC, then train with your new dataloader, A100-40/L4/T4 ladder, wall cap 5h, $10 cap, boot-armed rail, matched-window eval >=50 clips w/ threshold sweep).

## BEST-STACK DELTA
(c) none this lane — the checkpoint and its MANIFEST PENDING entry come from the GPU leg.
