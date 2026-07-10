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

VM pickleball-h100-demo1 (H100 spot, w7close snapshot, first-attempt ase1-b create), cold fresh
clip dirs, VM repo checked out at this branch (0c110de verified).

**Wolverine fresh cold run — COMPLETE, attested:**
- `PIPELINE_SUMMARY.json` status `partial` (input-quality advisory degraded_input only — same as
  every wolverine run); all substantive stages ran.
- **Total wall 379.5s** (body 270.7s, tracking 15.8s, ball 6.4s, calibration 1.4s) — vs 489.4s
  w7speed mean and 2141s historical: the ns06 efficiency default is live. 5.6x vs a week ago.
- **`best_stack.resolved.manifest_revision: 11`** — the run attests it consumed tonight's stack,
  fail-closed ball entry included. This is the per-run stack attestation the owner asked for.
- Fresh-run fail-closed proof: `ball_arc_render.json` summary reports enabled=true, suppressed
  segments [0, 2, 3, 4, 6, 8]; max emitted ball z 0.968m. Identical to the local re-composition —
  reproducible.

**Owner-critique 45s excerpt** — cut from the local rally parent, schema-valid tapped-corner
calibration trio (NOT the harvest-source format that killed the last cold attempt).
- Run 1 (zwcth45s_demo_20260710): honest PARTIAL — calibration/tracking/placement/ball/arc/world
  all ran (the strict-schema calibration path WORKS cold now), but the manager launched it without
  `--body-local`, so BODY degraded instantly per the RUNBOOK's documented no-default-remote-host
  behavior → no meshes/paddle; manifest also degraded (video outside the Vite allow root).
  Manager error, attributed; artifacts kept as evidence.
- Run 2 (zwcth45s_demo_r2_20260710): full cold rerun with `--body-local`, video in-tree. PARTIAL,
  454.6s, revision 11 attested; manifest fixed; but **BODY failed with a frames-schedule mismatch**:
  "missing BODY frame image for frame 41; expected body_frames/frame_000041.jpg" — the frame plan
  kept 658/1315 tracked frames (stride-2 schedule) while BODY iterated a frame outside the kept
  set. Cold directory, so NOT stale-cache: a genuine scheduling defect on this clip class
  (P0-D dependency-mismatch family; wolverine 244/705 unaffected; NEW BUG, booked as a follow-up
  lane — candidate interaction between the bounded skeleton schedule and the BODY frame iterator
  on non-eval cold clips).
- Run 2 owner-clip fail-closed proof (pulled artifacts): render summary enabled=true, suppressed
  segments [0,2,3,4,5] of 8; 58 emitted frames, max z 1.69m; gate bands 45 measured / 1279 hidden
  on 1350 frames. The policy generalizes beyond wolverine.
- Run 3 (zwcth45s_demo_r3_20260710): `--body-skeleton-stride 1` (the proven w7_critique config for
  this clip class), on a second VM pickleball-h100-demo2 (us-central1-a — both ase1 H100 zones
  STOCKOUT at attempt time) after the lane's early teardown killed the first r3. Results below.

**Lane report adjudication (manager):** the GPU lane's report claims the world-overlay half of the
fix is "only reachable via a standalone repair script, not the default pipeline". REFUTED with
artifact evidence: the fresh pipeline-produced `confidence_gated_world.json` matches the manager's
local overlay re-composition frame-for-frame (75 emitted / 59 measured / 224 hidden / max z
0.968m), which is only possible if `apply_ball_track_arc_solved_overlay` ran inside the world
stage (`build_virtual_world_state` line ~129). The lane's other findings stand, including its
correct observation that the run-level `arc_solved_overlay` provenance block does not persist
into the world artifact (schema drop — follow-up above), which is likely what misled it.
Teardown-race lesson: the manager's re-scope message pointed the lane at the r1 output path;
the lane found it complete, pulled r1+r2, and deleted the VM while the manager's r3 was
mid-flight. Standing rule for next time: a re-scope message to a lane sharing a VM must name
EVERY live run dir and an explicit do-not-teardown-before condition.
Ops note: the Sonnet GPU lane went idle after the wolverine run; the manager staged inputs and
launched both zwcth runs directly, then re-scoped the lane via queued message to
pull/teardown/report only.

**Known residual (booked):** the per-segment fail-closed verdict map does not persist into
`virtual_world.json` (strict world schema drops the overlay block); behavior provenance lives in
`ball_arc_render.json` summary + the confidence-gate hidden bands. Follow-up: persist the verdict
map in a world-adjacent sidecar or extend the world schema.

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
