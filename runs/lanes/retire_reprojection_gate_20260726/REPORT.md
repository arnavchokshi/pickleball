# Retiring reprojection error as a 3D quality gate

**Lane:** BALL / GATE-POLICY - **Branch:** `fix-reproj-20260726` (forked from `f29145a`)
**Date:** 2026-07-26 - **VERIFIED=0** - this is a correctness fix to how output is *judged*.
The 3D output itself is not improved by anything here, and in several places it is now
suppressed where it used to render.

---

## The finding in one paragraph

Reprojection error cannot see depth. Sliding a solved ball a full metre along its own
camera ray changes its reprojection by 1.6e-13 px - zero to machine precision. One view in
the external validation showed 0.323 px median reprojection, sub-pixel and below every
threshold in this repository, sitting alongside 0.305 m median and 1.381 m p95 3D error;
of a 0.3118 m mean total error, 0.3115 m was along the depth axis and 0.0047 m in the image
plane. That is geometry, not a defect to be tuned away. So any acceptance decision resting
on a reprojection threshold is measuring something it does not claim to measure, and this
change removes reprojection from every such decision - while keeping it, and relabelling
it, everywhere it was doing honest 2D work.

Source: `runs/lanes/tt3d_external_validation_20260726/report.json`.

---

## What that cost us, measured

The clearest case is
`runs/lanes/ballarc_anchorfusion_20260716/wolverine_no_soft_current/ball_track_arc_solved.json`:

| | before | after |
|---|---|---|
| frames emitted above 15 m of altitude | **79** | **0** |
| maximum emitted ball height | **23.53 m** | 14.97 m |
| `sigma_m` claimed on those 79 frames | **0.196641** | - |
| band on those 79 frames | `arc_weak` | `hidden` |
| flight-sanity reasons recorded for them | `["fit_bvp_fallback"]` | suppressed |

A ball twenty-three metres in the air, carrying a claimed 20 cm uncertainty, rendering.
No check flagged it as implausible, because there was no upper bound on height anywhere:
`_court_volume_bounds` in both `ball_arc_solver.py` and `ball_flight_sanity.py` bounded
x, y and z-minimum, and stopped there.

Across all 15 committed `ball_track_arc_solved.json` artifacts:

| band | frames before | frames after |
|---|---|---|
| `anchored_measured` | 565 | **565** |
| `arc_interpolated` | 159 | **159** |
| `arc_extrapolated` | 24 | **24** |
| `arc_weak` | 16549 | 3683 |
| `hidden` | 56471 | 69337 |

**12866 previously-emitted 3D positions are now suppressed. Every single one came from
`arc_weak`. Not one frame was lost from any confident band.** That is the shape the task
predicted, confirmed against solver output rather than assumed.

### On the owner-label claim

The task cited 19 owner labels at
`runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json` showing
`arc_weak` prefills wrong by 2.5-24.8 m. **That file does not exist**, at `f29145a` or on
this branch. The only committed label set from that lane is
`demo_labels/outdoor_webcam/ball_human_labels.json`: 15 labels, 3 prefills, bands
`anchored_measured` and `arc_interpolated` only, no `arc_weak` prefill anywhere, prefill
deltas 0.016-0.034 m. The wolverine launch command appears in that lane's `REPORT.md` but
its output was never committed.

So the claim as stated is unverified. The *phenomenon* is not: it was confirmed directly
against the solver artifacts above, which is stronger evidence anyway - it does not depend
on a human's judgement of where the ball was.

---

## Site classification

Full per-site detail with file:line is in `report.json`. 111 sites reviewed across
`threed/`, `scripts/`, `server/` (zero hits), `web/replay/src/`, `configs/`, `docs/`,
`tests/`.

### KEEP - legitimate (96 sites)

**Calibration residuals (61).** `court_calibration.py`
(`CALIBRATION_REPROJECTION_MEDIAN_GATE_PX = 8.0`, `P95 = 15.0`, `passes_reprojection_gate`),
`court_calibration_metric15.py`, `court_keypoint_eval.py`, `court_line_evidence.py`,
`court_static_lock.py`, `court_structured_metrics.py`, `court_partial_geometry.py`,
`drift_guard.py`, `net_anchor_court.py`, `eval/calib_eval.py`, `calibrate_harvest_courts.py`,
`calibrate_charuco_device.py`, `train_court_structured_v3.py` and the rest. Reprojecting a
2D correspondence against a 2D prediction is the correct and complete measure of a
calibration's image-space fit. Untouched.

**2D consistency (21).** The arc solver's `max_reprojection_inlier_px = 18.0`
inlier/outlier partition (which sightings belong to this arc), `ball_temporal_filter.py`
ballistic outlier rejection, `ball_ransac_arc_gate.py`, `ball_2d_post_gate.py`,
`camera_motion.py` RANSAC inliers, `ball_global_track.py` candidate membership,
`court_proposal_optimizer.py` MAGSAC. Untouched.

**Diagnostics (14).** `ball_solver_characterization.py` (which already carried
`"reprojection_blind_to_depth": True`), `eval/ball_metric3d_eval.py`
(`"reprojection_out_of_scope": True`), the TypeScript viewer fields. Untouched.

### KEEP AND FLAG - illegitimate but not mine to change (9 sites)

Reprojection RMSE is folded into a **3D uncertainty** in four places. Sizing a depth sigma
from a quantity blind to depth is exactly the error this task is about, but these belong to
the concurrent anisotropic-uncertainty lane and two of them are explicitly fenced. Flagged,
not edited:

- `ball_arc_solver.py:6748` `_frame_sigma` - `min(0.25, rmse * 0.01)` into the frame sigma.
- `ball_arc_solver.py:2896` `anchor_sigma_for_bounce` - **fenced**, sibling agent owns it.
- `ball_physics_fill.py:946` `_physics3d_uncertainty_m` - same pattern.
- `shot_taxonomy.py:346,364` - sizes the 3D landing ellipse and adds
  `0.15 / (1 + rmse/20)` to shot confidence. It consumes segments the arc solver has
  already banded, so it inherits the cap added upstream.

Three more are kept because they only ever **suppress**, and removing a fail-closed check
is a loosening: `one_world_v1.py` reprojection-regression kills, `ball_3d_events_gate.py`
5 px arc-residual block, the RKT paddle 6DoF rejections (out of lane; already box-derived
preview with zero true corner labels).

Two are kept because their input is a genuine calibration residual mapped into metres *on
the court plane*, where the mapping is real: `capture_quality.py` -> court trust band, and
`ball_inout_uncertainty.py` `sigma_reproj_m = p95 * ground_sample_distance`.

### CHANGED - reprojection was standing in for a 3D judgement (6 sites)

| Site | What it did | What it does now |
|---|---|---|
| `virtual_world.py:377-460` `ball_arc_segment_fail_closed_verdicts` | Trusted a fallback segment on inliers + bounded reprojection. Docstring said so. | Policy `v2`. Reprojection and inlier terms **kept**, documented as 2D. Depth-aware `segment_physical_sanity_violations` added. Every verdict stamped `depth_unvalidated: true` and carries `reprojection_error_px_diagnostic`. |
| `ball_arc_solver.py:5646` `_solved_frames` | Awarded `anchored_measured` - the strongest band, the one the label tool prefills from - purely on reprojection-inlier membership. | Inlier membership still assigns the band, but plausibility can only **lower** it, never raise it. Every emitted frame gains `depth_unvalidated: true`. |
| `ball_flight_sanity.py` | The gate meant to catch what reprojection could not: no upper z bound, and it only reached frames inside an evaluated segment. | Whole-track per-frame sweep, segment-independent, with a height ceiling. Absurd positions suppressed with **no BVP-fallback exemption**. `schema_version` 2 -> 3. |
| `ball_arc_chain.py:1416` `_segment_confidence` | `base *= 1 - rmse/48`, and that confidence decides the rendered band at the 0.45 threshold. | Penalty **kept** (removing a penalty loosens). Hard cap added: an implausible segment is pinned to 0.44, below the `arc_weak` threshold. |
| `ball_physics3d.py:107` `reconstruct_bounce_arcs_from_image_track` | Docstring: "The fit is accepted only when reprojection error stays under `max_reprojection_rmse_px`." | Bound **kept** as a 2D condition, docstring corrected. New `no_fit_under_physical_plausibility_gate` status. `summary()` always reports `depth_validated: false`, accepted results included. |
| `ball_physics_fill.py:317` `fill_ball_track_physics` | Admitted a frame into the 3D fill on image-plane evidence alone. | Reprojection rejection **kept**. Plausibility rejection added beside it, plus `implausible_rejected_frame_count` and `depth_validated: false`. |

Plus `trust_band.derive_ball_trust_band`, which now states that ball depth is unvalidated
and cites the measurement.

---

## Nothing was loosened

This is the constraint that shaped the change most. Twice the reprojection term was removed
outright and had to be put back.

**Segment verdicts.** Old (v1, reprojection-based) versus new (v2) over all **1203**
segments in every committed artifact:

```
segments: 1203   trusted_both: 902   suppressed_both: 301
newly_allowed: 0   newly_suppressed: 0
```

Exact parity. The first draft, which deleted the `max_reprojection_error_above_bound`
reason, allowed 2 previously-suppressed segments through - so the reason was restored and
reclassified as a 2D check instead. Which is also the sharpest measurement of how little
work it was doing: **2 of 301 suppressed segments are rejected by reprojection alone.**
Every other one is independently caught by the inlier/outlier evidence check or by spatial
sanity.

Worth reading alongside that: segment 2 of the wolverine clip, the one with the 23.52 m
apex, had `max_reprojection_error_px = 3585 px`. The reprojection gate did fire on it. It
just fired for the wrong reason, and on segment 3 - two frames underground, apex a
perfectly plausible 4.77 m - it fired identically. The number was never distinguishing the
two.

**Frame emission.** All 12866 newly suppressed frames come from `arc_weak`; 0 from
`anchored_measured`, `arc_interpolated` or `arc_extrapolated`. Suppression only increased.

---

## The replacement

`threed/racketsport/ball_position_plausibility.py`, policy `ball_position_plausibility_v1`.

| Tier | Bound | Consequence |
|---|---|---|
| implausible | z outside [-0.10, 8.0] m, or outside court footprint + 4.0 m | band -> `arc_weak`, `depth_unvalidated` stamped, **position kept** |
| absurd | z outside [-0.50, 15.0] m, or outside court footprint + 10.0 m | `world_xyz` and `sigma_m` -> null, band `hidden` |
| speed | > 35 m/s plausible ceiling, > 60 m/s absurd | upper ceiling only |

**Why two tiers.** This project's own measured calibration plane residual is 0.10-0.23 m
median. A single hard floor at -0.10 m would suppress genuinely grounded balls on
calibration error alone. Demotion is the honest response at ten centimetres underground;
suppression is the honest response at half a metre underground or fifteen metres up.

**Why the speed check only has an upper ceiling.** `virtual_world.py` already carried a
documented rationale (w7_ball3ddiag segments 7/9/10) that a slow-but-pixel-consistent arc
is depth-ambiguous, not junk, and hiding it trades honest coverage for nothing. That
reasoning is preserved: `speed_violations()` never flags a slow arc.

**What it does not do.** This is a necessary condition, never a sufficient one. A ball
solved three metres too deep along the camera ray but still over the court passes every
bound above. Nothing in this pipeline validates ball depth today. `depth_unvalidated: true`
is now on every emitted 3D ball frame, and that is the whole claim.

**Seam for the anisotropic uncertainty work.**
`evaluate_ball_track_plausibility(frames, depth_sigma_m_by_frame=None, sigma_multiple=3.0)`.
Off by default. When supplied it is purely descriptive - each frame report gains
`depth_sigma_m` and `overage_within_claimed_sigma` - and it never changes a verdict. There
is a test asserting a sigma wide enough to cover the overage cannot rescue an absurd frame.
No hard dependency on that lane landing.

---

## Verification

All commands run with `/Users/arnavchokshi/Desktop/pickleball/.venv/bin/python` (3.14.6)
and `PYTHONPATH=.`; the worktree has no `.venv` of its own. `models/checkpoints/*` and two
`runs/` paths referenced by `best_stack.json` are gitignored and absent from the worktree,
so they were symlinked from the main checkout to let the suite collect. No tracked file was
touched by that.

```
pytest tests/racketsport/test_ball_position_plausibility.py tests/racketsport/test_ball_flight_sanity.py -q
  24 passed                                                              exit 0
```

Focused blast-radius suite (`-k 'ball or arc or virtual_world or trust or shot or one_world
or flight or physics or plausib or characterize or chain or solver'`):

```
HEAD  156143c : 41 failed, 1371 passed, 4 skipped, 3664 deselected  (298.00s)   exit 1
base  f29145a : 41 failed, 1352 passed, 4 skipped, 3664 deselected  (314.85s)   exit 1
```

Attribution, by `comm` over the sorted `FAILED` node ids:

```
only at HEAD (caused by this change) : (empty)
only at base (fixed or flaky)        : (empty)
```

**The 41 failing node ids are byte-identical at base and at HEAD. Zero failures caused by
this change; 19 passing tests added.** The pre-existing failures are artifact- and
media-dependent (`test_build_pbvision_ball_sst.py`, `test_ball_wasb_dataset.py`,
`test_build_person_fewshot_pack.py`, `test_coords_parity_real_fixture.py`,
`test_flight_simulator.py`, `test_schemas.py`) and fail the same way with this branch
checked out to `f29145a`.

AGENTS.md structure checks:

```
scripts/racketsport/list_scaffold_tools.py --root .      exit 0    (base: exit 0)
scripts/racketsport/audit_dead_code.py --root .          exit 1    (base: exit 1)
scripts/racketsport/audit_storage_policy.py --root .     exit 1    (base: exit 1)
```

`audit_dead_code` fails identically at base, on the same two unknowns
(`scripts/tt3d/run_tt3d_validation.py`, `scripts/tt3d/tt3d_adapter.py`). Its
`python_sources` count moves 646 -> 647 and `unknown_python_sources` stays at 2, confirming
the new module is recognised. `audit_storage_policy` exit 1 is pre-existing and unchanged,
as the task noted.

### Tests added

`tests/racketsport/test_ball_position_plausibility.py` (18 tests). The regression cases are
real solver output, not invented: the five frames between 15.1 m and 23.52 m and the three
underground frames from `wolverine_no_soft_current`; real off-court values (|x| = 28.98 m,
|y| = 13.39 m) split across the two tiers; the real 23.52 m apex classified at segment level
alongside its 3585 px reprojection; end-to-end suppression of a 23.52 m frame in a track
with **no segments at all**, where every segment-scoped check is inert; and suppression
surviving the BVP-fallback exemption, since all 79 real frames sat on a `fit_bvp_fallback`
segment and that exemption is precisely what let them through.

### Tests modified

`test_flight_sanity_leaves_clean_parabola_untouched` used a synthetic "clean" parabola that
ends at z = -0.103 m - underground, which the new ground-plane bound correctly demotes. The
motion was raised 0.2 m/s so it ends at +0.097 m and the test asserts what it always meant
to assert; **the original motion is now covered by a new sibling test that asserts the
demotion explicitly**, so no coverage was traded away.

`test_flight_sanity_flags_only_outside_court_volume_frames_from_solver_config` asserts
`schema_version == 3` and now expects both `outside_court_volume` and
`outside_court_footprint` in the reasons, because the independent plausibility sweep agrees
with the segment-scoped court-volume check on that frame.

---

## Left open

- The four 3D-sigma sites that size a depth uncertainty from reprojection RMSE. Flagged
  above, deliberately not edited - the anisotropic-uncertainty lane owns them and two are
  explicitly fenced.
- `shot_taxonomy.py` landing ellipse and shot confidence. Same flaw, inherits the upstream
  cap; changing shot semantics from the gate lane was not warranted.
- RKT paddle 6DoF (`racket_stage_runner.py`, `racket_pose_preview.py`, `racket6dof.py`)
  drops frames above 6 px and sets pose confidence from reprojection. Structurally the same
  error in another lane.
- Depth itself is still unvalidated. This change makes the output honest about that; it
  does not fix it.
