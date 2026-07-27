# CAL: why `dist` was zero, and what fixing it actually bought

Lane: `calib_distortion_fit_20260726` · branch `fix-calib-20260726` off `f29145a` · 2026-07-26
**VERIFIED=0.** This is an engineering improvement to a fit. It promotes no capability, and
the ball is not verified. Raw solves were not modified; refined artifacts sit in
`refined/` with provenance and are not promoted.

---

## 1. Root cause: `dist` was zero for two separate reasons, and only one is the gate

The brief asked whether distortion is never estimated, estimated then discarded, or gated
behind a threshold that never fires. The answer is **estimated then discarded by a gate
that could not fire, because the fit carried a much larger error the gate was blind to.**

### 1a. The gate was scored on training residual, so it was not evidence

`court_calibration_metric15.fit_single_view_metric_camera` fit a zero-distortion camera and
a `cv2.calibrateCamera` k1/k2 camera on **all 15 points**, then compared them on the
**median reprojection error of those same 15 points**, accepting k1/k2 only on a 15%
improvement. A lower residual on the points you fit is what two extra parameters do; it is
not evidence that the parameters are real. Measured improvements at `f29145a`:

| clip | zero-dist train median | k1k2 train median | training improvement | gate fires? |
|---|---|---|---|---|
| burlington | 36.878 px | 26.418 px | +28.4% | yes |
| indoor | 21.767 px | 21.654 px | +0.5% | no |
| outdoor | 29.087 px | 31.241 px | -7.4% | no |
| wolverine | 31.376 px | 31.405 px | -0.1% | no |
| owner_IMG_1605 | 18.155 px | 18.241 px | -0.5% | no |
| **pbvision demo seed (the named defect)** | **19.165 px** | **17.298 px** | **+9.7%** | **no** |

The demo seed -- the 19.16 px solve the North Star names -- missed the 15% gate at 9.7%.

`cv2.calibrateCamera` was also unconstrained. On the outdoor clip it returned
**k1 = +6.96**, which is not a camera: the radial map folds back on itself well inside the
frame, so the inverse a consumer needs does not exist.

### 1b. The dominant error was a mis-specified object point, not distortion

`court_keypoint_net.PICKLEBALL_KEYPOINTS` declares the 3 net keypoints at
`post_net_height_m` = **0.9144 m** ("net top at left sideline"). **Every reviewed label set
in this repo marks the net line where it meets the court, ~0 m.**

Measured, not assumed: fit the camera on the 12 floor points only, back-project each net
label's ray, and intersect the vertical net plane `Y = 0`. The intersection's height is the
implied label height and its `X` is an independent sanity check.

| clip | implied net-label height (3 pts) | implied X vs expected |
|---|---|---|
| burlington | +0.116 / +0.081 / +0.043 m | -2.42/+0.46/+2.99 vs -3.048/0/+3.048 |
| indoor | +0.094 / +0.091 / +0.044 m | -3.02/-0.04/+3.12 |
| outdoor | +0.130 / +0.119 / +0.052 m | -2.94/-0.01/+2.97 |
| wolverine | +0.080 / +0.040 / +0.054 m | -3.17/-0.13/+2.72 |
| owner_IMG_1605 | +0.008 / +0.040 / +0.025 m | -3.03/-0.01/+3.06 |
| pbvision demo seed | +0.079 / +0.051 / +0.032 m | -2.64/+0.14/+3.02 |

0.008-0.130 m on six independent clips, with X landing within a few cm of the expected
sideline/centre. Not 0.914 m.

A 0.9 m world-model error on 3 of 15 correspondences is a **40-120 px systematic residual
on 20% of the points**. Distortion cannot explain a mis-specified object point, so no
`k1` could ever move the median enough to clear any gate. That is why the gate "never
fired": it was being asked to fix the wrong thing.

**Corroboration.** The shipped artifacts in `eval_clips/` record `world_pts` with the net
at **z = 0.0** and are numerically identical to what the current code produces *if* the net
z is forced back to 0 (burlington reproduces bit-for-bit: fx 1252.7783900978502,
dist [-0.30035182958629364, 0.09861181595540636], median 6.3901815068357175, p95
19.782933882839167). The artifacts were generated before the taxonomy change and never
regenerated. The frozen-digest test
`test_real_burlington_fixture_preserves_legacy_numeric_payload_and_adds_typed_contract`
**was already failing at `f29145a`** for exactly this reason -- a silent regression from
6.39 px to 26.4 px median that nobody re-froze.

The repo already half-knew this. `court_keypoint_geometric_loss.py` carries a "NET-LINE
CAVEAT" asserting the owner labels are at net top, and works around it by *excluding* the
net points from its homography and colinearity terms. The owner_IMG_1605 calibration does
the same (`net_points_excluded_from_fit`, net residuals 66-93 px). Both are working around
a wrong constant rather than measuring it. `court_line_keypoints.py`,
`net_anchor_court.py` and `court_keypoint_lines.py` all treat the same three names as
*floor line intersections*, which is the convention the labels actually follow.

### 1c. A third defect: `intrinsics.dist` was inert downstream

Neither `ball_arc_solver.pixel_ray_world` nor `ball_arc_solver._project_world_point` read
`intrinsics.dist`. Both were pure pinhole. So a fitted `k1` changed **nothing** downstream
-- while the focal length it was fit alongside changed everything. Two clips already ship a
nonzero k1 (burlington -0.300, indoor -0.256) and were being consumed with the bare
pinhole. Without fixing this, fitting distortion is a no-op by construction and the
deliverable would have been unmeasurable.

---

## 2. What was fitted, and the acceptance criterion

### Protocol

**Leave-one-out cross-validation, 15 folds.** Each fold refits focal length, distortion
coefficients *and* pose on 14 correspondences and scores the 1 it never saw. Training
residual is now descriptive only and is never used to select.

**Two things are selected, in this order:**

1. **Net-keypoint label height** -- `net_top_as_declared` (0.9144 m) vs `ground_net_line`
   (0 m), whichever has the lower held-out score. Same parameter count, so no gate:
   this is a discrete world-model fact. Selected per clip, not hardcoded, so a future clip
   genuinely labelled at net top still fits. **`court_keypoint_net.py` is not modified** --
   it is the court/training lane's taxonomy.
2. **Radial distortion**, in increasing complexity `zero -> k1 -> k1+k2`. Each candidate is
   compared against the **currently accepted** model (not the previous candidate), so a
   k1+k2 pair that works where k1 alone does not is still reachable -- radial terms trade
   off against focal length, and this happens in practice. Acceptance needs a
   `distortion_improvement_threshold` (15%, unchanged) reduction in the held-out score.

**Regularisation.** k1 in [-0.60, 0.35], k2 in [-0.50, 0.50], tangential and k3 fixed at
zero. The binding constraint is not the box but **radial invertibility over this frame**:
the model must reach every observed image radius while still strictly increasing. That
kills folded solutions (k1 = -0.4276 peaks at f(r) = 0.589 while a 1920x1080 frame at
fx = 1253 needs r = 0.879) and it is frame-dependent, as it must be -- the same -0.4276 is
perfectly fine on a 2514 px lens. Fitting is a coarse joint (focal, k1) grid seed followed
by a bounded Nelder-Mead, because the (focal, k1) surface is a long curved valley that a
1D-then-1D search misses and a cold `calibrateCamera` walks out of.

### Which held-out quantity decides

Both are computed and recorded for every candidate; `selection_metric` picks which one
ranks them.

- **`held_out_median_plane_error_m` (default).** The held-out pixel is pushed through the
  exact downstream bounce path -- undistort, world ray, intersect the plane at that point's
  own world height -- and compared in **metres**. This is the quantity the North Star calls
  the binding floor, and it weights each correspondence by real metric consequence. A
  pixel-space residual over-weights points near the camera, where a pixel is centimetres,
  against the far baseline, where a pixel is decimetres.
- **`held_out_median_px`.** The classical reprojection criterion.

They agree on 5 of 6 clips. Where they disagree (indoor) the disagreement is informative:
k1,k2 cuts held-out reprojection 6.60 -> 5.56 px **while making held-out plane error worse,
0.191 -> 0.234 m**. Selecting on pixels would have taken that trade.

### What got selected

| clip | net height | distortion accepted | held-out plane error (m): zero / k1 / k1k2 |
|---|---|---|---|
| burlington | ground | **k1 = -0.1789** | 0.3475 / **0.2693** / 0.2683 (k1k2 only -0.4% over k1 -> rejected) |
| indoor | ground | **none** | **0.1908** / 0.2546 / 0.2338 |
| outdoor | ground | **none** | **0.1273** / 0.1304 / 0.1296 |
| wolverine | ground | **none** | **0.1582** / 0.1660 / 0.2003 |
| owner_IMG_1605 | ground | **none** | **0.1065** / 0.1120 / 0.1153 |
| pbvision demo seed | ground | **k1 = -0.1312** | 0.2648 / **0.1769** / 0.1715 (k1k2 only -3.1% over k1 -> rejected) |

**Three of six clips get distortion refused on held-out evidence.** That is a measured
negative and it is preserved. The demo seed's k1 = -0.1312 is the same sign and order as
the k1 = -0.28 the line-evidence solve found on that video; the k1+k2 candidate the gate
declined sits at k1 = -0.2985, k2 = +0.2078, which is closer still -- but it does not earn
its second parameter on held-out data and was not taken.

---

## 3. Before/after: three states, because two changes are involved

To avoid reporting the seam fix and the refit as one lump:

- **A** = `f29145a`: shipped artifact read by a distortion-blind consumer.
- **B** = shipped artifact, distortion-aware consumer (1c fix only).
- **C** = refined artifact, distortion-aware consumer (full change).

### Calibration-residual floor, metres (median)

`ball_label_geometry.calibration_plane_residuals`, the number the labelling tool reports.
**Read the caveat in 3.1 before using the in-sample column.**

| clip | A (in-sample) | B | C | A held-out | B held-out | C held-out |
|---|---|---|---|---|---|---|
| burlington | 0.3684 | 0.1907 | **0.2196** | 0.3475 | 0.2683 | **0.2693** |
| indoor | 0.2319 | 0.1990 | **0.1835** | 0.1908 | 0.2338 | **0.1908** |
| outdoor | 0.1006 | 0.1006 | **0.1008** | 0.1273 | 0.1273 | **0.1273** |
| wolverine | 0.1268 | 0.1268 | **0.1261** | 0.1582 | 0.1582 | **0.1582** |
| owner_IMG_1605 | 0.0933 | 0.0933 | **0.0899** | 2.4198 | 2.4198 | **0.1065** |
| **pbvision demo seed** | **1.9062** | 1.9062 | **0.1444** | **2.7452** | 2.7452 | **0.1770** |

Floor-only (12 court-plane correspondences, so the improvement cannot be attributed
solely to the 3 net points whose declared height changed): burlington 0.3565 -> 0.1975,
indoor 0.1129 -> 0.1182, outdoor 0.0949 -> 0.0951, wolverine 0.1155 -> 0.1138,
owner_IMG_1605 0.0731 -> 0.0799, demo seed **1.7018 -> 0.1383**.

### 3.1 The reported "floor" is an in-sample number

`calibration_plane_residuals` back-projects **the calibration's own fitted
correspondences**. It is a training residual, so a model with more parameters always looks
better on it. The 0.101 / 0.127 / 0.232 / 0.928 figures the labelling tool derived inherit
this. The held-out columns above are the same measurement on data each fold never saw, and
they are the trustworthy ones. This matters for reading burlington: state B's 0.1907
in-sample looks better than C's 0.2196, while held-out they are indistinguishable
(0.2683 vs 0.2693) -- the signature of the extra k2 buying in-sample fit only.

### Honest summary

- **The named defect is fixed.** The demo seed goes 19.165 -> 3.676 px median (p95
  51.71 -> 11.42), in-sample floor **1.906 -> 0.144 m**, held-out floor **2.745 -> 0.177 m**.
- **owner_IMG_1605's declared correspondence set is repaired.** Its held-out floor was
  2.42 m because the artifact declares net points at 0.9144 m while its own labels are at
  ~0 and its fit *excluded* them -- an internally inconsistent artifact. Now 0.107 m. Its
  in-sample floor barely moves (0.0933 -> 0.0899) because the shipped camera was fit
  floor-only and was already reasonable.
- **Three clips are essentially unchanged** (outdoor, wolverine, indoor held-out). Their
  shipped artifacts predate the taxonomy regression and were already fit with net z = 0.
- **Burlington is a wash held-out and a regression in-sample.** See section 6.

---

## 4. `metric_confidence` and in/out abstention

`_confidence_from_reprojection` is **deliberately unchanged** (same training thresholds),
so before/after is apples-to-apples. An earlier revision of this change capped it by the
held-out median; that was reverted -- `metric_confidence` is what the in/out gate abstains
on, and moving that threshold is trust-band policy owned elsewhere. Under the held-out rule
indoor and outdoor would both drop med -> low; that cost is published as
`advisory_held_out_metric_confidence=...` in `capture_quality.reasons` for the gate owner
to decide on, not applied here.

| clip | metric_confidence A -> C | advisory (held-out rule) |
|---|---|---|
| burlington | low -> low | low |
| indoor | med -> med | low |
| outdoor | med -> med | low |
| wolverine | low -> low | low |
| owner_IMG_1605 | med -> med | med |
| **pbvision demo seed** | **low -> med** | low |

**Only the demo seed's confidence rises**, low -> med. Authority is unchanged everywhere:
`source` and `intrinsics.source` stay `metric_15pt_reviewed`, and
`reviewed_15pt_correspondences` is still emitted, so
`orchestrator.TRUSTED_INTRINSICS_SOURCES` and `REVIEWED_GATE_MARKERS` see exactly what they
saw before. A better fit is not a new authority class.

### Does in/out stop abstaining?

`ball_line_calls` returns `unknown` when the bounce's distance to the nearest boundary is
inside the uncertainty radius, and `ball_inout_uncertainty.bounce_geometric_uncertainty_m`
builds that radius from `reprojection_error_px.p95 * GSD` plus elevation parallax. Radius
at four canonical near-line spots, A -> C, metres:

| clip | near baseline | far baseline | left sideline | right sideline |
|---|---|---|---|---|
| burlington | 0.622 -> 0.660 | 4.933 -> 4.903 | 2.425 -> 2.463 | 0.862 -> **1.173** |
| indoor | 0.876 -> 0.883 | 4.368 -> 4.355 | 0.764 -> 0.758 | 0.937 -> 0.926 |
| outdoor | 1.266 -> 1.266 | 2.473 -> 2.473 | 0.340 -> 0.340 | 0.454 -> 0.454 |
| wolverine | 0.616 -> 0.615 | 3.926 -> 3.928 | 0.507 -> 0.507 | 2.102 -> 2.103 |
| owner_IMG_1605 | 0.659 -> 0.658 | 3.732 -> 3.741 | 1.628 -> 1.633 | 0.415 -> 0.410 |
| **demo seed** | **1.233 -> 0.717** | **7.268 -> 4.387** | **3.611 -> 2.453** | **3.258 -> 1.037** |

**Answer: no, in/out does not stop abstaining, except partially on the demo seed.** Even
after the fix, radii of 0.4-4.9 m dwarf the ~0.02 m line width a real in/out call needs.
The demo seed's dominant uncertainty term flips from `calibration_reprojection` to
`camera_geometry_elevation_parallax` on all four spots, which is the meaningful structural
change: calibration is no longer the binding term there. Elevation parallax -- one bounce
detection window of vertical travel at a low grazing angle -- is now the wall, and it is a
capture-geometry and temporal-resolution problem, not a calibration one.

**A finding for the gate owner:** the in/out radius consumes
`reprojection_error_px.p95`, an **in-sample** number, as if it were a generalisation
estimate. An overfit model therefore *narrows* the in/out radius. Burlington demonstrates
this: the honestly-selected k1 model has a worse in-sample p95 (25.8 vs 19.8) and so a
*wider* radius, while being indistinguishable held-out. The uncertainty model should read a
held-out residual; `held_out_median_reprojection_*px` is now published in
`capture_quality.reasons` for that purpose.

---

## 5. The owner's reviewed bounce labels

`runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json` (read
read-only from the ball-label lane's worktree; it is not present at `f29145a`) holds **19
owner labels, of which 7 are `kind: bounce`** -- the ones whose `uncertainty_basis` carries
a calibration floor. The other 12 are 7 `free_flight` and 5 `near_player`, whose sigma is a
human depth prior, not a calibration term.

| | before (A) | after (C) |
|---|---|---|
| calibration plane residual floor | 0.126773 m | 0.126132 m |
| bounce sigma along ray, median | 0.225264 m | 0.224934 m |
| bounce sigma along ray, min / max | 0.159563 / 0.281397 m | 0.158953 / 0.281298 m |

**They do not materially improve.** Wolverine's shipped calibration already had the net
points at z = 0 and the refit confirms zero distortion is the right model for that camera,
so there was nothing to recover. This is an honest negative for that clip. The
`outdoor_webcam` label file in the same lane contains 0 labels, so there is nothing to
measure there.

---

## 6. Recommendation on promotion

Refined artifacts are in `refined/<clip>/court_calibration_metric15pt_refined.json` with a
`provenance` block. Nothing was promoted into `eval_clips/`.

- **Promote:** `pbvision_11min_20260713_demo_seed` (the named defect: 13x in-sample /
  15x held-out floor improvement, confidence low -> med) and `owner_IMG_1605_8a193402780b`
  (repairs an internally inconsistent `world_pts`).
- **Promote, marginal:** `indoor_doubles` (held-out 0.234 -> 0.191; drops a k1,k2 pair that
  was hurting metric accuracy).
- **Do NOT promote:** `burlington_gold`. Held-out is a wash and the in-sample p95 regresses
  19.8 -> 25.8 px, which widens its in/out radius under the current (in-sample) uncertainty
  model. Its shipped k1,k2 artifact is already correct on net height.
- **No change needed:** `outdoor_webcam`, `wolverine_mixed`.

**Separately and more urgently than any promotion:** the `intrinsics.dist` seam fix (1c)
applies to the artifacts already in `eval_clips/`. Burlington's floor goes
0.3684 -> 0.1907 m and indoor's 0.2319 -> 0.1990 m with **no artifact change at all** --
those clips ship a k1 that the consumer was throwing away.

---

## 7. Test evidence

All commands run from the worktree root, real exit codes, no pipes.

### Focused, after the change

```
$ python3 -m pytest tests/racketsport/test_court_calibration_metric15.py -q -p no:cacheprovider
19 passed, 6 warnings in 121.46s          exit 0

$ python3 -m pytest tests/racketsport/test_camera_distortion.py -q
10 passed in 0.54s                        exit 0

$ python3 -m pytest tests/racketsport/test_ball_arc_solver_distortion_seams.py -q
5 passed, 6 warnings in 0.83s             exit 0
```

`test_court_calibration_metric15.py` was **12 passed / 1 failed** at `f29145a` (the stale
frozen digest, 1b). It is 19 passed now -- 5 new tests plus the repaired one.

### Focused calibration + ball set, base vs after

At `f29145a`:

```
$ python3 -m pytest \
    tests/racketsport/test_court_calibration_metric15.py \
    tests/racketsport/test_court_calibration.py \
    tests/racketsport/test_court_calibration_distortion.py \
    tests/racketsport/test_ball_arc_solver.py \
    tests/racketsport/test_ball_inout_gate.py \
    tests/racketsport/test_ball_inout_uncertainty.py \
    tests/racketsport/test_ball_court_calibration_gate.py -q
3 failed, 114 passed, 1 skipped, 6 warnings in 73.55s     exit 1
```

Pre-existing failures at `f29145a`:

1. `test_court_calibration_metric15.py::test_real_burlington_fixture_preserves_legacy_numeric_payload_and_adds_typed_contract` -- stale digest; **fixed by this change**.
2. `test_ball_arc_solver.py::test_fit_flight_segment_recovers_simulated_scalar_magnus_spin` -- pre-existing, untouched.
3. `test_ball_arc_solver.py::test_wolverine_seg6_fixture_falls_back_to_anchor_bvp_and_render_samples_stay_in_bounds` -- pre-existing, untouched.

After the seam fix, the same ball set was `2 failed, 87 passed, 1 skipped` -- the same two
pre-existing `test_ball_arc_solver.py` failures, no new ones.

### Wide

See `wide_suite.txt` in this directory for the full run and its exit code.

`tests/racketsport/test_ball_stage_runner.py` fails to **collect** at both `f29145a` and
here: `BestStackManifestError: best_stack entry 'ball.wasb_checkpoint' points at missing
path models/checkpoints/wasb/wasb_tennis_best.pth.tar`. Missing model artifact in this
environment, not a code defect and not caused by this change.

### AGENTS.md structure checks

```
$ python3 scripts/racketsport/list_scaffold_tools.py --root .            exit 0   (output byte-identical to base)
$ python3 scripts/racketsport/audit_dead_code.py --root .                exit 1   (PRE-EXISTING)
$ python3 scripts/racketsport/audit_storage_policy.py --root . --json    exit 1   (PRE-EXISTING)
```

- `audit_dead_code` exits 1 at base and here with the **same two** unknown files,
  `scripts/tt3d/run_tt3d_validation.py` and `scripts/tt3d/tt3d_adapter.py`. The only diff
  against base is `python_sources: 646 -> 647`, the new `camera_distortion.py`, which is
  recognised.
- `audit_storage_policy` exits 1 at base and here with the same
  `NotADirectoryError: ... /.git` -- the checker assumes `.git` is a directory and a git
  worktree makes it a file. Confirmed unchanged, as the brief expected.

---

## 8. Scope kept

- `court_structured_solver.py`, `court_static_lock.py`, `court_keypoint_net.py` and the
  court training path: **not touched.** The net-height fix lives in the metric fit, which
  is why it is a *selected* hypothesis rather than an edited constant.
- `anchor_sigma_for_bounce`, `_ray_plane_pixel_sigma`, the sigma consumer near
  `ball_arc_solver.py:2357`, gate/trust-band policy: **not touched.** Two surgical edits
  were made in `ball_arc_solver.py`, at `pixel_ray_world` and `_project_world_point`, both
  no-ops under zero distortion and both pinned by test.
- `ball_physics3d._project_world_array` left alone: it declares
  `output_space=PIXELS_UNDISTORTED_NATIVE`, an honest typed contract rather than an
  oversight. Follow-up for whoever feeds it distorted detections.
- Raw solves are immutable. Nothing in `eval_clips/` or
  `runs/lanes/pbv11_headtohead_20260713/` was written.

## 9. Reproduce

```
python3 runs/lanes/calib_distortion_fit_20260726/refit_and_measure.py
```

Writes `report.json` and `refined/`. ~2 minutes, CPU only.
