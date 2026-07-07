# LANE w4_bvp_20260707 — P1-4a: stabilize anchor-first BVP so good intervals keep fit status

## OBJECTIVE
Eliminate the D.3(b) regression from the committed BVP anchor-first WIP: after reselection, 5
exactly-named currently-good baseline intervals (Burlington seg0/seg13-adjacent ×4, Wolverine
seg6-region ×1) LOSE `fit*` status. Fix per the pre-ruled recipe below, prove on the frozen
baseline, and RUN the internal-val F1 check that the prior lane skipped. This is the dependency
spine's highest-leverage artifact (P1-4 full-flight arcs) — correctness over cleverness.

## EVIDENCE TO READ FIRST
- `runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json` — the WIP lane's own acceptance list:
  D.3(a) PASS, D.3(b) FAIL (the 5 intervals, with baseline/after per-interval fields), D.3(c)/(d)
  PASS, D.3(e) never run. Root-cause note: cached event-subset scoring uses a "ballistic proxy"
  while final fits run BVP/refinement; LOO reuses the segment anchor-BVP arc instead of re-solving.
- `TECH_BLUEPRINTS.md` BALL-3D pillar STEP 1 (the pre-ruled recipe — follow it literally).

## FILE TARGETS (re-grep every function name at HEAD FIRST — recorded line numbers HAVE drifted)
`threed/racketsport/ball_arc_solver.py`: `_fit_flight_segment_once` (~:486), `_select_event_subset`
(~:2234), the `selection_scoring="ballistic_initial_guess_no_bvp"` branch (~:841) inside
`_solve_bvp_shooting` (~:845), `_leave_one_out_validation` (~:3623), `_refine_bvp_endpoints` (~:963).

## RECIPE (order MANDATORY — do not reorder, do not invent physics)
(a2 — try FIRST, cheap) Anchor-preservation rule: any interval that was `fit*` in the frozen
baseline and still has ≥ `min_segment_observations` (3) inliers may NOT be demoted below
`fit_bvp_fallback` without a strictly-worse `endpoint_error_m`.
(a1 — ONLY if a2 leaves D.3(b) failing; expensive) Make selection scoring use the SAME
BVP/refinement cost the final fit uses (remove the ballistic-proxy shortcut) on JUST the affected
intervals.
(b) Re-solve BVP per-holdout in LOO rather than reusing the segment arc.
Solver constants UNCHANGED: `integrator_max_step_s=1/240`, `min_segment_dt_s=0.045`,
`max_reprojection_inlier_px=18.0`, `robust_pixel_sigma=6.0`.

## ACCEPTANCE (exact — measured through the real solver path, no lane-local surrogate math)
1. D.3(b): the 5 named intervals keep `status.startswith("fit")` with `endpoint_error_m <=
   baseline`, compared against the per-interval `baseline`/`after` fields in the
   `ball_p3a_bvp_anchor_first_20260705/report.json` acceptance entries. If that report lacks exact
   per-interval numbers: `git stash` your fix, regenerate the frozen baseline from pre-change code
   FIRST, unstash, then compare. Never compare against a baseline your own change produced.
2. D.3(a)/(c)/(d) do NOT regress (violators still convert; 0 out-of-bounds render; Outdoor
   violation_fraction stays ≤ 0.142857 — Outdoor numbers here come from the EXISTING banked
   artifacts/replays only; you must NOT touch held-out labels, and the ledger stays untouched).
3. D.3(e) — RUN IT THIS TIME: re-run the arc-solve + scoring replay on Burlington+Wolverine from
   existing candidate sidecars (CPU) and diff the product metric `label_f1_at_20px` (from
   `threed/racketsport/ball_benchmark.py` via `scripts/racketsport/run_ball_tracking_eval_suite.py`)
   pre- vs post-change: no regression >1pt on either clip. CHECK FIRST that the replay path exists
   locally (the prior lane ran these clips — find its run root); if a required artifact is missing,
   report exactly which file and STOP that sub-check rather than fabricating.
4. Owned pytest green: `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py -q`
   PLUS the full ball blast radius: grep `tests/racketsport/` for every test file importing
   `ball_arc_solver` (list them in the report) and run them ALL.

## KILL CRITERIA (commitments, not suggestions)
- If BVP-exact selection scoring (a1) blows CPU runtime unbounded: keep the proxy for SEARCH, ship
  the anchor-preservation guard (a2) as the fix, and RECORD the measured runtime delta.
- If good intervals cannot be preserved without breaking D.3(a): STOP → report as blocked
  (needs-advice) with the evidence. Do not trade one gate for another silently.

## OWNED FILES (anti-collision fence)
`threed/racketsport/ball_arc_solver.py`, `tests/racketsport/test_ball_arc_solver.py`, your lane
dir. DO NOT TOUCH: `train_ball_pretrain.py`, any `ball_sst_*`/`train_ball_stage2*` (another live
lane owns trainer-side), `run_wasb_ball.py`, `fuse_ball_tracks.py`, `process_video.py`,
`orchestrator.py` (fenced), `ios/**`, `runs/manager/**`, eval labels, held-out ledger.

## DISCIPLINE
`.venv/bin/python`; `pytest.importorskip("torch")` for any torch-touching test; no git
branch/commit/push; no network; do not create root-level .md files; fresh-GPU chain re-runs are
the MANAGER's wave-close job, not yours — your proof is the frozen-baseline replay + suites; if a
pre-existing failure appears, prove it fails at HEAD before your change.

## STRUCTURED REPORT
Acceptance table with per-interval status + endpoint_error_m (baseline vs after), D.3(e) F1 diff
per clip, runtime delta if a1 was tried. honest_issues: anything still fragile (e.g. remaining
proxy-vs-final-cost divergence risk). next: what you'd do next (Magnus STEP 2 readiness), for the
manager to weigh — not to auto-execute.
