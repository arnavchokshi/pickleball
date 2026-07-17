# Lane spec — trk_rfdetr_prod_20260716 (Codex gpt-5.6-sol high, CPU/MPS local, Track F)

OWNER DIRECTIVE 2026-07-16 (via coordinator): RF-DETR-L through the PRODUCTION association stack
on the frozen 2-clip card. Hypothesis: production pool construction + court-margin gate kills the
wolverine spectator FPs seen under the detbench feeder protocol while burlington stays ~0.99 cov4.
If it passes BOTH clips on ALL axes vs the frozen YOLO26m baseline → draft the owner-directed
preview-flip PROPOSAL (no code flip in this lane). License: directive waives concern; RF-DETR-L
is Apache-2.0 (strictly better than the AGPL incumbent) — record posture, proceed.

Read first: runs/lanes/trk_detbench_20260716/{DECISION_TABLE.md,report.json} (the feeder numbers
+ FEEDER_DRIFT), runs/lanes/trk_pooldiag_20260716/spec.md (M1-M5 mechanisms — Phase 1 of that
spec EXECUTES inside this lane), runs/research_trk_rkt_20260716/benchmark_spec_trk.md (frozen
protocol), runs/lanes/trk_detbench_20260716/spec.md (pins).

## Environment (fence-safe)

Lane-local venv (NEVER mutate the repo .venv): python3 -m venv .lane_venv inside the lane dir;
install rfdetr + torchreid + torch + repo test deps there; PYTHONPATH=repo root. Network allowed
for pip + the pinned checkpoint (rf-detr-large-2026.pth from the storage.googleapis.com/rfdetr
URL; record sha256 — must match the detbench report's). Device: MPS preferred, CPU fallback
(900 frames total; minutes either way). NO GPU fleet.

## Protocol (gates in order; STOP typed on any gate failure)

1. **ENV-FIDELITY GATE:** reproduce detbench arm 0a locally — run_raw_pool_person_authority
   (margin 1.0, OSNet models/checkpoints/osnet_x1_0_market1501.pt sha256 2809d322…, all other
   knobs default) on the FROZEN trk_flip production pools + frozen scorer
   (--cvat-root runs/lanes/trk_flip_20260713/frozen_gt, IoU 0.5, expected-players 4).
   Must match the preflip pins within 0.0001 (burl 0.8830775881/0.7116666667, wolv
   0.8515962036/0.76, 0 switches). This proves the LOCAL env is score-faithful.
2. **POOL-CONSTRUCTION ATTRIBUTION (pooldiag Phase 1, executed here):** diff the frozen
   production pools (runs/lanes/trk_flip_20260713/{default,preflip}_production/<clip>/) against
   the detbench feeder pools (runs/lanes/trk_detbench_20260716/vm_pull/detbench_out/pools/
   arm0b_*). Per frame: detection counts, conf histograms, GT-matched presence asymmetries,
   track fragmentation, spectator-region detections. Mark M1 (persisted-pool filtering),
   M2 (BOTSORT lifecycle), M3 (GMC), M4 (preprocessing), M5 (version) each
   CONFIRMED/EXCLUDED/UNRESOLVED with file-level evidence. Deliverable: POOLDIAG_PHASE1.md
   (also satisfies the booked trk_pooldiag spec Phase 1).
3. **PRODUCTION-EQUIVALENT RF-DETR-L POOL:** run RF-DETR-L (native 704, person class verified,
   conf floor 0.05) per frame on both eval clips (eval_clips/ball/<clip>/source.mp4); construct
   the pool applying the CONFIRMED production-construction rules from step 2 (e.g. the same
   BOTSORT config + persistence/filter rule + GMC setting that production uses). Document every
   construction choice with its step-2 evidence line. If step 2 leaves the rule UNRESOLVED,
   construct BOTH variants (feeder-style and best-hypothesis-production-style) and score both —
   do not silently pick one.
4. **SCORE THE FROZEN CARD:** association (step-1 exact settings) → frozen scorer, both clips.
   Axes vs the frozen YOLO26m baseline (0a pins): IDF1, cov4, switches=0, spectator FP=0,
   far-off-court FP=0. Also report vs the detbench feeder arm-1 numbers (burl 0.9204/0.9967;
   wolv 0.7955/0.7267/1sw/16spectFP) to show what pool construction changed.
5. **VERDICT:**
   - PASS = both clips, all axes ≥ baseline with zero FP/switch violations. Then write
     FLIP_PROPOSAL.md: best_stack candidate entry sketch (new detector entry, preview band,
     do_not_promote honesty, production-reproduction bar, rev bump), the code seam required
     (orchestrator detector-injection point per the recon: RealYOLO26BoTSORTReIDTrackingRunner
     hardcodes yolo26m — name the exact integration-lane work), runtime + license lines. NO code
     changes, NO best_stack edit — proposal only, manager takes it to the coordinator for the
     flip ruling.
   - FAIL any axis = report which, with the per-frame FP/miss evidence; no proposal.

## Fences + budget

Write ONLY runs/lanes/trk_rfdetr_prod_20260716/**. Read-only everywhere else. No pipeline code,
no configs, no best_stack, no commits, no GPU. The concurrently-running lane
owner_person_labels_20260716 owns its own dirs — do not touch. Budget: aim ≤2.5h wall; the
step-2 forensics is bounded to ≤45 min (mark UNRESOLVED and proceed with both variants rather
than digging).

## Report

report.json + final message: gate outcomes in order, the M1-M5 table, the decision table (both
clips × {baseline, RF-DETR-L-prod[, variant B]} × all axes + runtime ms/frame on MPS/CPU),
verdict, and if PASS the FLIP_PROPOSAL.md path. VERIFIED=0; a preview flip is an owner-directed
stack selection, never an accuracy promotion; the fresh-clip full gate (cov4 ≥0.95 etc.) remains
unmet on this historical card regardless of outcome.
