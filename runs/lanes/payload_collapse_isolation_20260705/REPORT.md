# ISOLATION VERDICT (manager-completed 2026-07-05 ~17:3x PDT)

The dispatched Sonnet agent completed the legacy E2E run (runs/payload_collapse_isolation_20260705T223802Z,
wall 804.9s, --fetch-body-monoliths legacy assembly, current worldhmr) but died before the diff step.
The manager ran the decisive A-vs-B comparison locally (1e-6 float tolerance, path-strings/timing ignored):

A = legacy assembly   (runs/payload_collapse_isolation_20260705T223802Z/source)
B = array-native slim (runs/body_payload_collapse_verify_20260705T221027Z/source)
Both ran the SAME current worldhmr.py (md5-verified by the livecheck + spec step 1).

| artifact | result |
|---|---|
| body_full_clip_gate.json | value-identical (0 non-timing diffs) — gate semantics unaffected |
| body_grounding_quality.json | 1 real diff: grounding_metrics.world_joint_visual_smoothing.stance_lower_body_frames_protected **828 (legacy) -> 0 (array-native)** |
| skeleton3d.json | 14,863 value diffs; sampled magnitudes ~1-3mm on lower-body joints — consistent with stance lower-body smoothing protection NOT being applied in the array-native path |

## VERDICT: NOT ACCURACY-NEUTRAL YET — named root cause, default pipeline UNAFFECTED
The array-native path does not thread stance/contact provenance into
_smooth_grounded_frames_stance_aware (stance lower-body protection never engages: 828->0 protected
frames). Divergence is mm-scale and below every gate threshold, but it is a real behavioral gap, not
noise. Because experimental_body_array_native defaults to False (orchestrator.py:672) and
body_array_native.py must be explicitly enabled, the DEFAULT pipeline output is unchanged.

Positive control: the earlier livecheck already confirmed the speed effect (BODY stage wall
618.5 -> 473.0s with array-native; assembly 171.6s -> 0).

## BOOKED NEXT STEP (speed P-queue, first item)
Fix stance/contact-phase threading in threed/racketsport/body_array_native.py so
stance_lower_body_frames_protected matches legacy on the same inputs (add a legacy-vs-native
stance-protection equality test to test_body_serialization_timing.py), THEN flip
experimental_body_array_native default and re-verify. Until then the flag stays opt-in OFF.
