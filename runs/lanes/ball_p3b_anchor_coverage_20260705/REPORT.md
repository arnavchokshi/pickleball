# Lane P3-B Anchor Coverage Recovery

Status: PASS_WITH_CONTAINED_WOLVERINE

## Summary

- Phase B bounce-bounce discovery is active in the current solver: it only accepts a solver-proposed intermediate bounce when the parent bounce-bounce segment is a gate failure, there are enough interior observations, the split lifts inlier fraction above the fit gate, both child arcs are court-plausible, and residual reduction beats `selection_split_penalty`.
- Synthetic positive/negative tests cover recovery and no-spurious-anchor containment.
- The real Wolverine seg6 fixture now exercises discovery-enabled containment: no valid intermediate anchor is accepted, frame 200 stays `fit_bvp_fallback`/`arc_weak`, and render samples stay in court volume.
- Three-clip default-chain rerun passed: all clips `ran`, zero `fit` segments with zero inliers or court-volume violation, zero Phase A-good segment regressions, and zero out-of-bounds render samples.

## Wolverine 148 Cutoff Verdict

Verdict: WIRING-BUG-DOCUMENTED_NOT_FIXED_BY_THIS_LANE.

This is not a detection-coverage gap. In `runs/manager_rebuild_wolverine_20260702T23Z`, current `ball_track.json` has 242 visible frames through frame 299, with 138 visible frames after 148. `wrist_velocity_peaks.json` has 259 peaks through frame 297, with 114 peaks after frame 148. Regenerating ball inflections from the current ball track yields 53 candidates through frame 295, with 24 after frame 148; fusing those regenerated ball cues with current wrist peaks yields 42 contact events through frame 295, with 21 after frame 148.

The stored `ball_inflections.json` is stale/inconsistent: it has candidates with frame ids after 148, but their `time_s` values are frame/60-era timestamps while current `ball_track.json` is 30 fps. Example: stored inflection frame 156 has `time_s=2.6`, while current ball-track frame 156 has `t=5.2`. Contact fusion matches by `time_s` and emits frames from current fps, so late-rally cues collapse into <=148 output frames. Stored `contact_windows.json` therefore has 24 events and stops at frame 148 / 4.925s.

I did not patch event-stage reuse/wiring in this lane because the approved Phase B file scope is `ball_arc_solver.py` plus tests, with read-only event-input diagnosis.

## Wolverine 6.66s Verdict

Verdict: CONTAINED-ONLY.

The Phase B three-clip rerun did not find an acceptable Wolverine intermediate solver anchor. Frame 200 is still contained by segment 6:

- `t`: 6.666666666666667
- `world_xyz`: `[-0.70918917, 5.146088199, 4.138838425]`
- `band`: `arc_weak`
- `segment_status`: `fit_bvp_fallback`
- `inside_court_volume`: true

This is safe containment, not near-net recovery.

## Three-Clip Acceptance

Audit artifact: `runs/lanes/ball_p3b_anchor_coverage_20260705/three_clip_acceptance_audit.json`

- Burlington: unchanged from Phase A, 13 segments, 588 world-xyz frames, 2883 render samples, 0 out-of-bounds render samples.
- Wolverine: unchanged from Phase A, 10 segments, 290 world-xyz frames, 1400 render samples, 0 out-of-bounds render samples, frame 200 contained-only.
- Outdoor: improved with accepted gate-failure split `solver_bounce_000388` at frame 388. Parent inlier fraction 0.036585 -> split inlier fraction 0.457831; residual score 2959.061626 -> 307.50481; score gain 2651.556816 > `selection_split_penalty` 0.25.

## Accepted Risks

- Mandatory `auto_bounce_candidate` anchors still cannot be rejected by `_select_event_subset`; Phase A gates only contain bad mandatory arcs.
- A confidently wrong wrist track can still force a wrong contact anchor and therefore a wrong arc; Phase A gates contain but do not repair that input.

## Verification

- `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py -q -k 'bounce_bounce_gate_failure'` -> 2 passed, 35 deselected.
- `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py -q -k 'wolverine_seg6 or bounce_bounce_gate_failure'` -> 3 passed, 34 deselected.
- `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py -q` -> 37 passed.
- Three-clip default-chain rerun into `runs/lanes/ball_p3b_anchor_coverage_20260705/three_clip_default_chain_r2` -> 3 clips `ran`.
- `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py tests/racketsport/test_ball_*.py tests/racketsport/test_schemas.py tests/racketsport/test_scaffold*.py -q` -> 368 passed.
- `cd web/replay && npm test` -> 14 files / 182 tests passed.
