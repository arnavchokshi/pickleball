# Wolverine foot-slide 0.0378 m flag — root-cause diagnosis

Lane: `wolverine_slide_diag_20260728`. Date: 2026-07-28. Evidence commit:
`3f6782b` (full write-up also appended to `runs/manager/inflight_lanes.md`).
Written by the orchestrator from the lane agent's returned findings (its
harness refused report-file writes). Read-only diagnosis; no code changed.

## Verdict

**Not a real regression** from the July-25 foot/toe repairs or the 2026-07-28
always-on default flip, and not diffuse noise: sub-centimeter run-to-run BODY
(SAM-3D) joint-output variability for one player landed on the wrong side of a
static, unmodified hysteresis threshold in the existing plant-phase detector.

## Key evidence

- Of 43 candidate foot-lock phases (`body_grounding_quality.json`, identical
  count/rejection breakdown in both runs), 42 measure slide at machine epsilon
  (~1e-15 m) in BOTH runs. The single exception: tonight's extra candidate
  phase `2:left:142-145:19` (player 2, left foot, 4 frames, 0.13 s) with
  `slide_m = 0.03777` — the entire reported 0.0378 m FAIL. The baseline run has
  NO phase in that range (sequence jumps 129-130 → 165-168).
- Raw `skeleton3d.json` positions for that player/foot/window are mm-to-~1cm
  apart between runs and show the foot in continuous fast motion in BOTH runs.
  Reconstructed speeds sit at `threed/racketsport/foot_contact.py`'s 0.75/1.25
  m/s hysteresis band (unmodified since its original commit, verified via
  `git log -S`); the baseline's speeds run ~0.1-0.3 m/s higher, staying
  "moving," while tonight's marginally slower estimate dipped into "contact"
  for 4 frames.
- Player 2 tracking coverage is byte-identical (282 frames) in both runs —
  tracking/selection ruled out.
- The two real code changes between the runs were traced and ruled out: the
  July-25 `placement_v2_temporal_support_state` detector feeds a different
  artifact (`foot_contact_phases.json` → trajectory/grounding refine), and the
  slide gate (`worldhmr.py:_contact_gate_stream_for_skeleton3d()`) reads only
  `skeleton3d.json` via its own detector; `placement_trajectory_refine`'s
  output is never read by the slide gate.
- Correction to the dispatch premise: `grounding_refine`'s typed revert on
  wolverine is IDENTICAL in both runs (same message, same residual pattern);
  the July-24 "case5 reverted" note refers to a different clip.

## Recommended fix (implemented separately in `plant_phase_plausibility_20260728`)

Add a plausibility cross-check in the BODY-direct contact-phase detector:
reject/flag a candidate phase whose own `max_speed_mps` is inconsistent with a
genuine plant, rather than trusting height/speed hysteresis alone. The failing
phase here is 4 frames (0.13 s) of a fast-moving foot.
