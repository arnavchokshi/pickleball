# T23 pbv_replay ruling (Fable, 2026-07-20): HONEST PARTIAL — high diagnostic value, not the win

Frozen production stack ran end-to-end on a FRESH real pickleball clip (Drill Session 186s, never
processed before). Result per the planner's own bar: NOT a success (core entities missing) but an
honest, browser-verified partial + the sharpest product diagnosis of the night:

- WORKS: real viewer, zero page errors, real 2D ball track 81.8% of frames (low_confidence band),
  honest HUD ("Players 0 / Ball not visible"); NOTHING fabricated; every gap a typed abstention.
- BLOCKED (the finding): auto court-line evidence found EVERY required line EXCEPT far_centerline
  across the whole 186s → fail-closed calibration gate → 0/4 players, no BODY, no placement. On a
  REAL fresh clip, the product's binding wall is exactly the research's conclusion: court
  line-detection robustness. ONE missing line cost the entire people/3D stack.
- ball_arc exceeded its 20-min cap AGAIN on a fresh clip (per-segment guard insufficient at stage
  level) — killed cleanly per directive, --no-ball-arc reuse worked. Known scaling defect, reconfirmed.
- Ops: $2-3.3, torn down + list-confirmed. Two-sided sha on all artifacts.

## ROUTES
1. → T14 (court line-hardening, running): this clip is the PERFECT real-world test case — static
   camera, 186s of frames, one missing line (far_centerline). Cross-frame pooling/lookalike
   handling should recover it. When T14 lands, score it on THIS clip too (artifacts + line evidence
   pulled at runs/lanes/pbv_replay_20260720/vm_pull/). If T14 recovers far_centerline → re-run the
   replay → players/BODY unblock → THE user-visible chain on fresh clips.
2. → ball_arc stage-level wall-clock cap belongs in the runner (typed abstention), not just
   per-segment budgets — queue as a small lane.
3. Screenshot evidence: vm_pull/viewer_screenshot.png (real video, real viewer, honest zeros).
