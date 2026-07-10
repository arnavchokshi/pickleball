# demo_beststack_20260710 — REPORT

Owner directive: demo video of the week's improvements on the best pipeline, everything spun up,
best-stack discipline going forward. Session: Fable bg job bd000ff4, worktree demo-beststack-20260709
branched at committed HEAD d47b399a1. Draft PR: https://github.com/arnavchokshi/pickleball/pull/12

## 1. Fail-closed 3D ball emission — LANDED (code + tests + manifest)

- `threed/racketsport/virtual_world.py`: `ball_arc_segment_fail_closed_verdicts` + fail-closed
  boundary inside `apply_ball_track_arc_solved_overlay` (the documented single place world_xyz is
  decided). Criteria: fallback/weak status AND (inliers < 3 OR outliers > inliers OR max reprojection
  > 40px OR spatial sanity violation {outside_court_volume, apex_height_implausible,
  net_clearance_below_slack}). `fit` segments always trusted; speed-policy-only violations stay
  visible. Missing fit statistics on a fallback segment = not trusted (fails closed).
- `threed/racketsport/ball_arc_chain.py`: same verdicts gate the `ball_arc_render.json` dense viewer
  trail (same-artifact segment_id join — never against ball_flight_sanity.json whose ids are shifted).
- `configs/racketsport/best_stack.json` revision 11: `ball.world_emission_fail_closed` WIRED_DEFAULT
  with provenance + measured before/after. Default selection, NOT a VERIFIED claim.
- Tests: 12 new in `tests/racketsport/test_virtual_world_ball_failclosed.py`; 2 existing tests
  updated to the new contract; 132 passed / 0 failed across affected suites. Known fresh-worktree
  environmental failure (gitignored runs/ fixture) deselected after `git check-ignore` confirmation.

### Real-artifact proof (w7_critique wolverine world, re-composed with the fix)

| metric | before (fail-open) | after (fail-closed) |
|---|---|---|
| suppressed segments | none | 0, 2, 3, 4, 6, 8 (= the diagnosed junk) |
| emitted ball frames | 300/300 | 75/300, all plausible |
| max ball height | 23.530 m | 0.968 m |
| max consecutive speed | 41.0 m/s | 10.3 m/s |
| gate bands | 236 measured · 0 hidden | 59 measured · 224 hidden |
| viewer HUD (live headless capture) | "237/300 measured · 0 hidden" | "54/300 measured · 235 hidden" + honest coverage notice |

Honest residuals: kept segment 5 (a true `fit`) still reaches y≈10m behind the baseline — monocular
depth ambiguity on a genuine fit, owned by the TT3D anchor-search lane, not emission policy. Kept
segments 7/9/10 violate the solver's 3 m/s minimum (pixel-consistent slow arcs) and stay visible by
design.

## 2. GPU best-stack E2E (Sonnet lane demo_beststack_gpu_20260710)

VM pickleball-h100-demo1 (H100 spot, w7close snapshot), cold fresh clip dirs, code = this branch.
See `runs/lanes/demo_beststack_gpu_20260710/report.json` (main checkout) for the run attestation
(`best_stack.resolved` revision 11 + fail_closed provenance + md5-verified pulls). RESULTS PENDING
AT WRITING — appended below when the lane reports.

## 3. Demo video

Assembled from: live headless viewer captures (before/after worlds, world view), CPU overlay renders
(tracking / skeleton / 2D ball), ink-on-cream title cards with measured numbers, fresh-run segments
from the GPU pull. Delivered to ~/Desktop. Assembly evidence in main checkout
`runs/lanes/demo_beststack_render_20260710/`.

## 4. Standing directive recorded

Owner (2026-07-10): from now on, wins get promoted into the default E2E when they beat their gates —
the promotion loop is: candidate beats pre-registered held-out bar -> flip the best_stack.json entry
-> same day. The eval reset (ns02) makes this executable. Demo runs may select PENDING candidates
via explicit CLI overrides but must be labeled; truth runs use manifest defaults only.

## BEST-STACK DELTA

(a)-adjacent: adds `ball.world_emission_fail_closed` WIRED_DEFAULT at revision 11 (emission-policy
default flip per the ball-3D diagnosis + research SHIP-FIRST ruling + owner demo directive; not an
accuracy promotion — no gate metric changed, VERIFIED=0 binding).
