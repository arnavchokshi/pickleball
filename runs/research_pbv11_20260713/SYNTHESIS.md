# pb.vision 11-min dual-model synthesis — 2026-07-13 (Fable ruling)

Sources: runs/research_pbv11_20260713/FORENSICS_11MIN.md (codex gpt-5.6-sol xhigh, 8/8 PASS) +
Fable 5-lens workflow wf_7e3c1be5-e38 w/ adversarial re-computation (all load-bearing claims
CONFIRMED to the digit). Data: data/pbvision_11min_20260713 (cv sha 4ccd08fd, video sha 272a2132,
pb.vision's OWN demo video — R&D reference ONLY, never GT/training). VERIFIED=0.

## Cross-model agreed truth (disagreements: none material)
1. THEIR PIPELINE: detector heads -> ONE global ballistic track per rally (track posterior exported
   as frame confidence; a 0.891 threshold on it reproduces 96.1% of selection; interpolates <=6-frame
   holes; terminates on physics/rally-end) -> rally window = track extent +- fixed pads (~0.9s pre-
   serve / ~1.1s tail) -> event anchors at trajectory-fit times w/ per-event NMS -> typed exception
   subpolicies (serves constructed from contact not 2D peak; out_of_sequence flags 32 shots/16
   bounces; 9 elevated bounces all flagged) -> in-sequence bounces 154/154 EXACTLY z=radius, nets
   pinned to net plane -> out-of-range landings CLAMPED in-bounds w/ corrected=true (3/256, silent-
   ish) -> rally with zero track DROPPED from product entirely (rally-level fail-closed).
2. EMISSION TRUTH: 75.6% in-rally coverage is DELIBERATE serve-contact->terminal-event windows;
   interior tracking holes are tiny (23 spans, max 24 frames/0.8s across 11 min). The right target
   is continuous emission serve->terminal, not %-of-rally.
3. BALLISTICS: pure gravity (drag NOT identifiable even on 130 long arcs); 93% of segments pass the
   re-integration diagnostic; event delta-v 39-41x interior (scale-confirmed).
4. RADIUS CUE: universal ORDINAL cue (Spearman>=0.5 in 41/41 rallies) but linear strength is
   confidence-gated (R2 0.274 raw -> 0.706 at radius-conf>=0.7 keeping 95.7%) and compressed
   (log-log slope -0.696 vs -1 pinhole) — usable as a WEIGHTED residual w/ per-rally
   calibration/abstention, not a universal linear law.
5. PRODUCT: their fact breadth ~10x ours (shot taxonomy, score state, win-prob strip, highlights,
   Wilson-CI coach advice) BUT in/out + landing calls are BINARY w/ zero uncertainty, low-conf shots
   ship full facts, and net_impact_score=0.5 is a shipped placeholder. OUR typed-abstention/audit
   architecture is ahead on honesty — keep it; expand fact breadth behind it (NS-05 map in both
   source docs).

## The program (supersedes the 07-12 reproduction map; kills stand: TT3D/DP/bridge/fusion-as-built)
- MOVE 1 (GPU, dispatched): baseline head-to-head — our promoted stack once on the 697s demo video,
  frozen identity, immutable per-rally ours-vs-PB scorecard + union event-candidate set staged for a
  15-min owner review.
- MOVE 2 (codex, dispatched): event-evidence quality on the same clip (audio order correction +
  blur-aware shot/bounce proposals; this clip HAS audio — closes BL-E's untested-cue gap).
  Gate: p/r>=0.90 within +-2 frames on the reviewed union set; contact p90<=40ms; zero 2D regression.
- MOVE 3 (codex, dispatched isolated): ball_global_track candidate — whole-rally robust ballistic
  track over ALL candidates incl. below-threshold (membership gating, not per-frame acceptance),
  track posterior + <=6-frame interpolation + typed exception policies + fixed-pad windows +
  rally-level fail-closed; radius as confidence-gated weighted residual; two-ended robust fit.
  Offline-scored vs internal cards + (after MOVE 1) the 11-min scorecard. UNWIRED candidate.
- Recovery-v2 re-enable + solver wiring only behind MOVE 2/3 gates. Landing/in-out promotion still
  requires NS-02 independent truth.

## ADDENDUM 2026-07-13 ~03:0x — MOVE-3 v1 verdict (real below-threshold pools)
ball_global_track REJECTED_INTERNAL, stronger than v0: real MPS WASB regeneration produced genuine
0.05-floor pools (987/4624 sub-0.5 candidates; real-inference byte-parity re-proven, closing the
MOVE-2 fake-harness caveat) and the UNMODIFIED module still refused both rallies with the IDENTICAL
131-frame membership hole — 129/131 hole frames DO carry candidates; the gate rejects them as
trajectory-inconsistent. Input starvation REFUTED as the cause. PRECISE REMAINING LEVERS:
(1) event boundaries INSIDE long segments (131 frames ~ 4.4s spans multiple shots; single-segment
ballistic membership cannot bridge unmarked direction changes) -> MOVE-2's audio-anchored event
chain applies on real captures WITH audio (internal cards have none); (2) detector recall/positional
accuracy in fast/occluded spans -> NS-02/03 label flywheel + learned radius/detector heads.
Evidence: runs/lanes/ball_gt_rescore_20260713/ (scoring_table_v1_real.json, parity proof, real pools).
