# ADDENDUM A10-A12 to oneworld_impl_20260716 (owner win condition, 2026-07-16)

This resumes your session with three ADDITIONAL acceptance items. Everything in spec.md stands
unless explicitly amended here. PRIORITY RULING (owner): demo-visible completeness with honest
bands > per-metric polish. If time-boxed, A10-A12 outrank hitting the M1 <=0.60m target — attempt
M1 honestly and report what you got.

Owner win condition (verbatim intent): "all our demo videos having 4 people on/near the court in
the correct position, each holding a paddle in semi-decent orientation, and super good tracking
of the ball in the 3D world where we are always getting its location and placing it correctly
and seeing when it hits a paddle, floor, net etc." OneWorldV1 IS this deliverable. Bands stay
honest; nothing predicted is ever labeled measured; VERIFIED=0 unchanged.

## A10 — paddle ALWAYS emits a display pose with its band (design amendment 1)
OneWorldPaddleState gains `display_pose_world: SE3 | None` + `display_tier`:
- gen-2 resolved: winning hypothesis (unchanged) — display_tier "resolved".
- gen-2 UNRESOLVED: still emit the BEST-evidence hypothesis as display_pose_world
  (highest §3.6 combined score even below the M>=0.25 resolve bar; deterministic tie-break by
  lower reprojection error IS allowed for DISPLAY ONLY — record rule "display_tiebreak_reproj",
  never for `status`, which stays "unresolved"), retained_hypotheses keep BOTH poses —
  display_tier "unresolved_best_evidence".
- gen-1 wrist proxy: carry the existing proxy pose as display_pose_world —
  display_tier/status "unresolved_legacy_wrist_proxy".
- Only when NO paddle input exists at all for that player/frame: display_pose_world null +
  absence sentinel (unchanged).
This is display-band carriage, NOT promotion: status/ambiguity flags/confidence and both
retained hypotheses are unchanged; provenance records the display-choice rule + margin.
TESTS: unresolved synthetic swing still yields display_pose_world + intact retained_hypotheses +
status unresolved; display_tier never upgrades a trust band.

## A11 — typed events are first-class viewer entities
Add consolidated `events: list[OneWorldEvent]` to OneWorldV1:
`{event_index, type: Literal["paddle_contact","floor_bounce","net_contact","net_cross"],
t, frame, world_location_raw: Vector3|None, world_location_refined: Vector3|None,
hitter_id: int|None (paddle_contact only), confidence, trust_band, evidence: (the A8b
contact_evidence_vector for paddle contacts; surface-prior weights for bounce/net),
provenance: OneWorldRuleProvenance}`.
Sourced from the existing contacts/bounces refinement lists + Stage B net handling (net_cross
carries its time residual and NO positional pull, per DESIGN.md §3.4). The refinement lists
stay; events[] is the viewer-facing consolidation (Track H timeline glyphs + 3D markers).
TESTS: all four types emitted from synthetic inputs with correct locations/bands; net_cross has
no position mutation.

## A12 — ball continuity chain (design amendment 2: banded estimates instead of display holes)
Per-frame ball emission becomes tiered via `estimate_tier`:
1. "arc_measured" — arc/solved world_xyz (existing behavior).
2. "physics_predicted" — ballistic bridge between adjacent supported samples/anchors ONLY
   (no extrapolation beyond 0.5s from support); ConfidenceProvenance.predictor set,
   horizon_frames counted, predicted_sigma_m grows with horizon; approx=true.
3. "ray_court_projection" — frames with ONLY 2D detection: typed-coordinates camera ray
   reported as its court-plane intersection + ray direction; explicit `altitude_unknown: true`;
   large stated sigma; band low_confidence/preview.
4. absent — ONLY when not even a 2D detection exists (absence sentinel unchanged).
HONESTY RAILS (mandatory): estimate_tier + not-measured provenance on every non-tier-1 sample;
M1/M2/M5 metrics computed ONLY on tier-1 (+ contact-refined) samples — the continuity chain can
NEVER feed a metric; M3 world coverage reported STRATIFIED: `coverage_measured` (tier 1 only —
comparable to the frozen 0.39 baseline) AND `coverage_with_predicted` (tiers 1-3, labeled).
The A6 demo partial MUST use this chain (2D ray projections + bounce priors) so the demo ball
never simply vanishes — with tier counts reported.
TESTS: no tier>=2 sample ever carries a measured band; predictor/horizon/sigma populated;
tier-2 refuses to bridge gaps >0.5s from support; metrics reject tier>=2 inputs; ball absent
only when 2D is also absent.

## Report additions
- acceptance rows for A10/A11/A12 + the stratified M3 pair.
- honest_issues: tier distribution on Wolverine + demo (how much of the chain is predicted vs
  measured — the owner sees this number).
