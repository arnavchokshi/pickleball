# Queue #1 forensics — fusion contact refusals × fabricated-bridge spans (RULING)

Manager (Fable), 2026-07-19. Read-only forensics per NORTH_STAR §5 row 1. Inputs verified by
sha256: tracks.json f9d2f46c… == the fusion's pinned input (the PRODUCTION-stack tracks the
fusion consumed). Full tables: xref.json in this dir.

## Findings

1. **Fabrication is present in the production tracks the fusion consumed** — 8 synthetic
   conf-pinned-0.35 runs ≥3f (p2 f45-55, f112-118; p3 f173-180, f186-212, f247-270, f272-276,
   f278-291; p4 f266-289) totaling ~120 frames. Production bridges are geometrically tamer than
   the RF-DETR-variant's f45-86 10.4m monster (max disp 0.49m, no net-cross), but they reach the
   fusion unlabeled — P0-I confirmed at the exact consumption point.
2. **Overlap is REAL but MINORITY: 5/24 refused contacts** sit inside a synthetic span ±2f
   (f44, f49 → p2 f45-55; f112, f116, f118 → p2 f112-118). 10/24 contacts sit on frames where
   <4 players have any real detection (f49: ZERO real detections).
3. **The refusal wall does NOT dissolve into fabrication**: 19/24 refusals stand on fully-real
   frames with wrist residuals 1.6-15.7m; overlapping contacts' residuals (2.4-10.2m) are in the
   same range as non-overlapping. The declared contact WINDOWS (the owner's 29/50-failed
   auto-label class) + missing trained event detection remain the dominant explanation, exactly
   as the standing BALL ruling says.
4. **True cov4 quantified**: only 156/300 frames (0.520) have 4/4 REAL detections vs the
   reported 0.7233 → synthetic padding = 0.203 on the production stack (worse than the ~0.107
   noted for the RF-DETR variant card).
5. **NEW FINDING (same trust family, distinct defect): exported world_xy is piecewise-CONSTANT**
   — held stale across long spans (p2: frozen 137f f163-299 while its bbox moves 82px; distinct
   world_xy counts per player in xref.json). No per-frame provenance marker distinguishes held
   from measured placement. Does NOT explain the refusals (those use BODY wrists), but must be
   routed to the selection-layer lane + Track C as a provenance gap.

## RULING: the ball program does NOT reorder.

The pre-registered question — "if the refusals overlap the bridges, the fusion's impossible
contacts are partly OUR fabrication and the ball program reorders around fixing fabrication
first" — resolves NO on the production evidence: fabrication contaminates a 5/24 minority and
the residual magnitudes are indistinguishable between contaminated and clean frames. The
trained-event-head wall (queue #5) remains the binding ball blocker; the P0-I fix (queue #2-3,
lane DISPATCHED) remains binding for the trust contract and for honest cov4, not as the
contact-refusal explanation. Both proceed in parallel as dispatched. Post-fix, the fusion
should be re-run on de-fabricated tracks to re-score the 5 contaminated contacts.

VERIFIED=0 unchanged. No promotion. Evidence: xref.json (tables), this file (ruling).
