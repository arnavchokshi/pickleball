# Lane spec — trk_rfdetr_integrate_20260717 (Codex gpt-5.6-sol high; Track F closing deliverable)

Coordinator flip ruling 2026-07-17: wire RF-DETR-L into the production tracking path as an
owner-directed PREVIEW detector selection (same pattern as the 2026-07-13 margin flip: preview
band, do_not_promote honesty, production reproduction bar, rev bump). VERIFIED=0; this is a
stack selection, never an accuracy promotion; the fresh-clip full gate stays unmet.

## OPERATING POINT (filled by manager after the preregistered conf-0.30 single-shot)

RESOLVED by manager 2026-07-17 (conf-0.30 preregistered shot FAILED — wolverine worse on every
axis; both threshold attempts now spent):

- BRANCH: **2b_ship_for_demo_at_0.18**
- Detector conf floor: **0.18**; RF-DETR-L native 704 input (no imgsz analog).
- Reproduction targets (frozen card, both clips, within 0.0001, GPU-class env, via the REAL
  production entry) — source runs/lanes/trk_rfdetr_prod_20260716/vm_rerun/report.json:
  - burlington: IDF1 0.922018 / cov4 0.993333 / 0 sw / 0 spectFP / 0 farFP
  - wolverine:  IDF1 0.803625 / cov4 0.723333 / 1 sw / 4 spectFP / 0 farFP
- best_stack notes MUST carry the verbatim regression text from
  runs/lanes/trk_rfdetr_prod_20260716/FLIP_PROPOSAL.md (do not paraphrase).
- STATUS: dispatch-ready handoff; NOT dispatched by Track F (no GPU in the wrap-up window).

## Read first

runs/lanes/trk_rfdetr_prod_20260716/{FLIP_DECISION_INPUT.md,FLIP_PROPOSAL.md,PREREG_conf030.md,
vm_rerun/report.json,vm_rerun/POOLDIAG_PHASE1.md}; threed/racketsport/orchestrator.py
(RealYOLO26BoTSORTReIDTrackingRunner — the seam); runs/lanes/trk_detbench_20260716/report.json
(checkpoint pins); configs/racketsport/best_stack.json rev 12 patterns (the margin-flip entry is
the honesty template).

## Work

1. **Detector-injection path in orchestrator.py**: a new runner (or a detector strategy inside
   the existing one) that runs RF-DETR-L per frame (pinned checkpoint via models/MANIFEST.json
   entry, sha256 0f4e20e1…, person class id 1, conf floor per OPERATING POINT) and feeds
   per-frame detections through BOTSORT (`botsort_no_reid_loose.yaml`) via its `update()` API —
   the construction proven exactly equivalent to the frozen pools at matched operating point
   (vm_rerun zero-delta proof). Emit the SAME pool artifacts (tracked_detections.json,
   raw_tracked_detections.json, metrics.json w/ counts + operating-point provenance) so every
   downstream consumer is untouched. Downstream association/ReID/margins: zero changes.
2. **Selection surface**: models/MANIFEST.json entry `rfdetr_large_2026` (license Apache-2.0,
   url + sha256); best_stack.json entry `tracking.person_detector` — owner-directed selection,
   status WIRED_DEFAULT only per this ruling, `trust_band: preview`, `do_not_promote: true`,
   provenance = this lane + the trk_rfdetr_prod evidence chain, `proven_against` = the exact
   card numbers incl. [2b only] the wolverine regression stated VERBATIM in notes; rev bump
   12→13. Kill-switch: explicit flag/manifest fallback to yolo26m preserved and tested.
3. **Production reproduction bar (the gate of this lane)**: through the REAL production entry
   (scripts/racketsport/process_video.py tracking stage) on both eval clips, on a GPU-class
   environment (Mac CPU is NOT score-faithful — standing finding), reproduce the OPERATING
   POINT reproduction targets within 0.0001 on every axis. Miss = NO flip, report honestly.
4. **Tests**: focused units for the injection path (detector-output→pool schema, conf floor,
   class mapping, kill-switch fallback, manifest resolution failure = loud), plus the wide
   suite with real exit codes. rfdetr dependency: document install (runtime deps unpinned per
   repo policy) + add to the fleet snapshot re-bake list in gpu_fleet.md notes.
5. **Docs**: RUNBOOK tracking section note (detector selection + kill switch); no North Star
   edits (manager owns that).

## Rules

Serialize: this lane is the sole orchestrator.py owner while live (ledger row enforces);
ball_arc files untouched; no association/threshold changes of any kind; one-shot reproduction
(no tuning to hit the bar — a miss is a finding). Fence: threed/racketsport/orchestrator.py,
models/MANIFEST.json, configs/racketsport/best_stack.json, RUNBOOK tracking lines, named tests,
lane dir. NO commits (manager commits after ruling). Report: report.json + reproduction table +
wide-suite exit codes.
