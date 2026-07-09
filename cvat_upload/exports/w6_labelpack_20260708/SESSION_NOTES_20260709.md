# w6 owner ball sessions — standing rulings (owner, 2026-07-09)

Apply to `w6_ball_sst_ball_session_01` export (filed here) AND all future w6 owner sessions
unless superseded.

## Ruling 1 — multi-ball frames: SKIP double-boxed frames, no owner rework
Some frames genuinely show 2+ balls. Where the owner boxed BOTH balls, ingest must
skip-and-account the frame (existing multiple_ball_boxes path — already deterministic).
Known in s01: frame indices 104, 257, 269, 418, 458, 461. In the other multi-ball frames
the owner boxed ONLY the in-play ball — those are valid single-box positives as-is.
Do not ask the owner to clean double-boxed frames; the skip loss is accepted.

## Ruling 2 — visibility_level is UNINFORMATIVE in owner w6 sessions
Owner left the dropdown at default `clear` throughout; many blurred balls are tagged
`clear`. Owner explicitly declines to backfill. Therefore for w6 owner sessions:
- Treat labels as BOX-POSITION-ONLY supervision (BlurBall center convention still applies).
- Do NOT use `visibility_level` from these sessions for stats, loss weighting,
  visibility-conditioned training, or per-visibility eval slices.
- Reviewed-absent negatives (deleted boxes) remain fully valid.
w5 session_01 visibility tags predate this ruling and remain usable.

## s01 export stats (validated against prelabels 2026-07-09)
640/640 frames; 71 reviewed-absent negatives; 108 corrected boxes (median move 985px —
disagreement mining confirmed working); 461 prelabels confirmed ≤2px; 6 double-box frames
(skip per Ruling 1).
