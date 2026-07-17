# FLIP PROPOSAL — RF-DETR-L person detector, ship-for-demo preview selection

Track F manager, 2026-07-17. Coordinator ruling 2b (owner standing posture "good enough").
Status: **PROPOSAL + INTEGRATION SPEC HANDED OFF — NOT EXECUTED** (no GPU authorized in the
wrap-up window; a half-integrated detector is worse than none, per the coordinator's own rule).
VERIFIED=0. This is an owner-directed stack SELECTION, never an accuracy promotion. The
NS-03.TRK fresh-clip full gate (IDF1 ≥0.85, 0 switches, 0 spectator FP, 0 far-off-court FP,
cov4 ≥0.95, every fresh clip) remains UNMET and unchanged by this proposal.

## What is proposed

Select **RF-DETR-L** (Apache-2.0, `rf-detr-large-2026.pth`, sha256 pinned in
runs/lanes/trk_detbench_20260716/report.json, person class id 1, native 704) as the production
person detector at the **production-equivalent operating point: conf floor 0.18**, feeding the
UNCHANGED frozen association stack (margin 1.0m + OSNet, rev-12 defaults), with:

- `trust_band: preview`, `do_not_promote: true`, best_stack rev 12 → 13;
- the wolverine regression stated VERBATIM in the entry notes (text below);
- a kill-switch fallback to yolo26m preserved and tested;
- provenance = the full evidence chain (detbench card → vm_rerun → this proposal).

## The evidence (frozen card, frozen scorer, GPU-class environment)

| clip | arm | IDF1 | cov4 | sw | spectFP | farFP |
|---|---|---|---|---|---|---|
| burlington | YOLO26m baseline (pin) | 0.8831 | 0.7117 | 0 | 0 | 0 |
| burlington | **RF-DETR-L @0.18** | **0.9220** | **0.9933** | 0 | 0 | 0 |
| wolverine | YOLO26m baseline (pin) | 0.8516 | 0.7600 | 0 | 0 | 0 |
| wolverine | **RF-DETR-L @0.18** | 0.8036 | 0.7233 | 1 | 4 | 0 |

VERBATIM NOTES TEXT for the best_stack entry (must appear unedited):
> Owner-directed preview selection 2026-07-17, NOT an accuracy promotion. On the frozen
> historical-internal card this detector IMPROVES burlington (IDF1 0.8831→0.9220, cov4
> 0.7117→0.9933, all FP axes zero) and REGRESSES wolverine (IDF1 0.8516→0.8036, cov4
> 0.7600→0.7233, 0→1 identity switch, 0→4 true-spectator false-positive frames). The
> wolverine spectator ghosts are HIGH-CONFIDENCE detections: a preregistered conf-0.30
> attempt made every wolverine axis worse, so threshold suppression is exhausted. The
> owner ruling is that high-confidence spectator detection is correct detector behavior;
> the fix belongs in a downstream selection layer (court-footpoint prior + 4-slot identity
> pruning), owned by a separate track. TRK stays preview; VERIFIED=0.

## Preregistered threshold attempts: BOTH SPENT (stop rule satisfied, honestly)

1. conf 0.18 (production-equivalent): burlington clean gain; wolverine 1 sw + 4 spectFP.
2. conf 0.30 (preregistered single shot, PREREG_conf030.md): **FAILED — wolverine worse on
   every axis.** Decisive negative: the ghosts are high-confidence, so no threshold fixes them.
   Detector-side suppression of this failure mode is EXHAUSTED. Evidence: vm_conf030/report.json.

This is the useful finding of the night: it converts "tune the detector" into "build the
selection layer" — precisely the build-our-own item the Track F TRK research identified
(TRK_CROSSCHECK_RULING.md §"Where we build our own tech": owned-data negatives + 4-slot
enrollment gallery with open-set rejection + soft court-footpoint prior with truncation-aware
uncertainty). Ghost forensics feed that track.

## Integration lane (the closing deliverable — SPEC READY, NOT DISPATCHED)

`runs/lanes/trk_rfdetr_integrate_20260717/spec.md` is complete and dispatch-ready. Its branch
block resolves to: **BRANCH 2b_ship_for_demo_at_0.18**; reproduction targets = the four rows
above (burlington 0.9220/0.9933/0/0/0; wolverine 0.8036/0.7233/1/4/0), matched within 0.0001
through the REAL production entry (process_video tracking stage) on a GPU-class environment.
Requirements it must honor: orchestrator detector-injection seam (RealYOLO26BoTSORTReIDTracking-
Runner hardcodes yolo26m via MANIFEST), BOTSORT `update()` pool construction proven equivalent at
matched operating point (vm_rerun zero-delta proof), zero association changes, kill-switch,
MANIFEST entry + best_stack rev bump with the verbatim notes, focused tests + wide suite,
snapshot re-bake note for the rfdetr dependency.

Why handed off, not run: it needs a multi-hour orchestrator change plus a GPU-class reproduction
run; no GPU is authorized in this window. Landing it half-done would leave a partially flipped
stack — the exact failure the coordinator ruled against.
