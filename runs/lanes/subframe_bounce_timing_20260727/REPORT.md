# Sub-frame bounce timing

Lane `subframe_bounce_timing_20260727`, branch `subframe-20260727`, forked from `e209112`.
CPU only. Default OFF. **VERIFIED=0 for pickleball.**

---

## 1. The defect

A bounce happens *between* video frames. The nearest frame therefore still shows the ball
slightly above the court, and forcing that frame's camera ray down to `z = BALL_RADIUS_M`
walks past the true contact along the ray by

```
overshoot = height_above_plane / |ray_direction_z|
```

`height_above_plane` is never negative, so the overshoot is never negative either. That is why
the bounce-anchor error is a systematic push away from the camera rather than scatter.

The previous lane (`sigma_anisotropic_fix_20260726`) modelled this term correctly and reported
it as `bias_along_ray_m`, but the anchor itself was still built at the marked frame. Reporting
a bias is not the same as not having one. **This lane attacks the instant.**

## 2. Method: `image_branch_kink_v1`

World position is continuous through a bounce; world velocity is not. Projection is a smooth
map, so the 2D track inherits exactly that structure -- continuous position, discontinuous
velocity -- meaning the image track has a **kink at the contact instant and nowhere else**.

For a candidate instant `t_c`, fit two polynomial branches that share a meeting point:

```
p(t) = p_c + b_pre  * (t - t_c) + c_pre  * (t - t_c)^2      t <= t_c
p(t) = p_c + b_post * (t - t_c) + c_post * (t - t_c)^2      t >= t_c
```

The shared `p_c` encodes position continuity; independent `b_pre`/`b_post` encode the velocity
discontinuity that *is* the bounce. Only `t_c` enters non-linearly, so the branch coefficients
come from a linear solve at each grid point and the search is a 1-D scan over +-1 frame plus a
parabolic step. The two image axes are independent given `t_c`, so each is a 5x5
normal-equation solve -- no numeric dependency, no optimizer.

The fit returns the instant **and** the ball's image position at that instant from the same
solve. `build_bounce_anchor` then takes the ray through that sub-frame pixel instead of the
marked frame's pixel. At the true contact the ball genuinely is at `z = ball_radius`, so
`height_above_plane` goes to zero and the overshoot goes with it.

Two properties worth naming:

- **The estimator never reads the calibration, the court, or the physics constants.** It is
  pure 2D-track geometry, so it degrades independently of the court solve.
- **The meeting point is two-sided.** If `t_c` is estimated early, the incoming branch pulls
  `p_c` above the plane and the outgoing branch pulls it below; the least-squares compromise
  lands near the true contact pixel. First-order timing error largely cancels in the pixel.
  That cancellation is *measured* below -- it is deliberately **not** claimed by the
  uncertainty model (section 5).

### Guards

Refinement abstains rather than guesses. Guards run on a fully populated timing object, so a
refusal carries its own diagnostics.

| Guard | Rule | Why |
|---|---|---|
| `rejected_search_bound` | optimum pinned against the +-1-frame window | the kink is outside the window this lane may move within: the marked frame is wrong by more than a frame, which is a *detection* defect and must not be papered over by sliding the anchor |
| `rejected_displacement` | refined pixel more than `0.6 x` the local inter-frame step from the marked pixel | contact is within half a frame of the marked frame; a bigger jump means the fit ran away, not the ball |
| `rejected_weak_kink` | velocity jump below 5% of branch speed | no measurable discontinuity, no localisable contact |
| `insufficient_observations` | fewer than `order + 1` samples on a side | nothing to extrapolate from |

### What was NOT used

**Audio.** `audio_onsets_v2.py` has ~1 ms hop with sub-hop parabolic refinement, and
`AcousticPropagationModel` in `timebase.py` already models the ~3 ms/m propagation delay that
matters at court scale. It was not wired in, for a measurable reason: TT3D is synthetic
observations with **no audio track at all**, so there is no way to measure whether audio
refinement helps. Adding it would have meant shipping an unmeasured second timing source next
to a measured one. The directive was to use audio only where it measurably helps; here it
measurably could not be evaluated. This remains the obvious next lane once a pickleball clip
with sound *and* independent 3D bounce ground truth exists.

---

## 3. HEADLINE: TT3D bounce-position error, before and after

Same 816 bounces (136 scored trajectories x 3 views x 2 noise conditions), both anchors built
in the **same pass of the same run**, so this is not a comparison across runs. The BEFORE
column reproduces `runs/lanes/tt3d_external_validation_20260726/report.json` exactly.

```
PYTHONPATH=. python3 scripts/tt3d/run_tt3d_validation.py \
  --out runs/lanes/subframe_bounce_timing_20260727/tt3d_report.json
exit 0
```

### 3D bounce position error (m), pooled over 816 bounces

| | mean | median | p90 | p95 | max |
|---|---|---|---|---|---|
| **BEFORE** (marked frame) | 0.114123 | **0.091096** | **0.245462** | **0.316761** | **0.449812** |
| **AFTER** (sub-frame) | 0.044324 | **0.030562** | **0.090303** | **0.124241** | **0.406093** |
| change | -61.2% | **-66.4%** | **-63.2%** | **-60.8%** | **-9.7%** |

### Systematic along-ray bias (signed mean, m)

| | bias |
|---|---|
| **BEFORE** | **+0.099272** |
| **AFTER** | **-0.007621** |

The systematic component collapses by 92%, and the small residual has flipped sign -- it is no
longer a one-sided push away from the camera. This is the result the lane was built to get:
the error that was *systematic* is now essentially *zero-mean*, which is the difference between
a bias you must model and noise you can average down.

### Error decomposition (m)

| axis | | median | p90 | p95 |
|---|---|---|---|---|
| depth (along ray) | BEFORE | 0.080501 | 0.232973 | 0.307799 |
| depth (along ray) | AFTER | 0.026496 | 0.082532 | 0.117699 |
| image plane | BEFORE | 0.030924 | 0.087289 | 0.110882 |
| image plane | AFTER | 0.010867 | 0.033184 | 0.048095 |

Both axes improve, which is expected: the anchor now sits on the plane at the right place, so
the plane-constraint shadow of the along-ray error shrinks with it.

### Per view / condition

| view / condition | refined | BEFORE median | AFTER median | BEFORE p95 | AFTER p95 | BEFORE bias | AFTER bias |
|---|---|---|---|---|---|---|---|
| back / no_noise | 132/136 | 0.0837 | 0.0247 | 0.3272 | 0.0736 | +0.1083 | -0.0089 |
| back / noise | 129/136 | 0.0993 | 0.0489 | 0.3315 | 0.2121 | +0.1070 | -0.0038 |
| side / no_noise | 133/136 | 0.0736 | 0.0182 | 0.2021 | 0.0554 | +0.0677 | -0.0068 |
| side / noise | 131/136 | 0.0829 | 0.0346 | 0.2032 | 0.0908 | +0.0703 | -0.0015 |
| oblique / no_noise | 133/136 | 0.1150 | 0.0294 | 0.3458 | 0.0956 | +0.1236 | -0.0144 |
| oblique / noise | 131/136 | 0.1061 | 0.0487 | 0.3475 | 0.1600 | +0.1188 | -0.0103 |

Injected pixel noise (~2.4 px) roughly doubles the refined error but does not restore the bias,
which is consistent with the original finding that the driver is timing, not detection noise.

### Guard behaviour

| status | count |
|---|---|
| `refined` | 789 (96.7%) |
| `rejected_search_bound` | 18 |
| `rejected_displacement` | 9 |

The 27 abstentions keep the marked-frame anchor. Abstaining is why the max improves at all
rather than regressing.

### Timing accuracy

| quantity | value |
|---|---|
| mean timing error | -0.00005 s (0.001 frames) |
| rms timing error | 0.004924 s (0.123 frames) |
| p95 abs timing error | 0.011508 s |
| reported timing sigma (median) | 0.010000 s |
| reported / measured | **2.03** (conservative) |

---

## 4. MEASURED NEGATIVE: the downstream trajectory does not uniformly improve

The product question is not only "is the anchor better" but "is the reconstructed trajectory
better". E2 refits the post-bounce weak flight segment from the sub-frame anchor -- same
observations, same config, only the seeding anchor differs.

| E2 weak-segment 3D error (m) | n points | mean | median | p90 | p95 | max |
|---|---|---|---|---|---|---|
| **BEFORE** (marked-frame anchor) | 2676 | 0.330516 | 0.133641 | 0.791943 | 1.403043 | 4.745549 |
| **AFTER** (sub-frame anchor) | 2576 | 0.348387 | **0.107142** | **0.913952** | **1.713333** | **5.091333** |

**The median improves 19.8% and every tail statistic gets worse.** Fitted trajectories drop
from 498 to 473. This is a real negative and it is not being buried: a decisively better
bounce anchor did **not** translate into a uniformly better downstream trajectory on TT3D.

Two caveats on the negative, both stated so it can be judged rather than explained away:

- E2 is measurably **nondeterministic run to run**. Before touching any code, a fresh baseline
  run of the unmodified script produced p90 0.792 / p95 1.403 where the committed report
  records p90 0.675 / p95 1.217 on the same data. The tail movement here is of comparable size
  to that run-to-run spread, so the direction is suggestive rather than settled.
- The refined anchor also shifts the anchor's **time**, which changes the segment's start and
  therefore the weak fit's conditioning, not just its seed position. Disentangling the position
  change from the time change is not attempted here.

Either way, the honest reading is: **the bounce anchor is decisively better; the downstream
weak-segment reconstruction is not demonstrated to be better, and its tails may be worse.**
That alone is sufficient reason for this to stay default OFF.

---

## 5. Keeping the uncertainty honest

Two things could have gone wrong here, and both were live risks.

**(a) The pixel-noise timing sigma was measurably optimistic.** Propagating pixel noise through
the branch crossing predicts a 1-sigma of ~0.9 ms -- but the *measured* spread on TT3D is
4.9 ms rms, **5.4x wider**, because the dominant error is the local polynomial branch model,
not the pixel. Shipping the pixel-only number would have been the same class of defect this
lane exists to remove. The reported instant is therefore floored at
`DEFAULT_SUBFRAME_TIMING_SD_FLOOR_FRAMES = 0.25` frame intervals: a round declared prior chosen
to sit **above** the table-tennis measurement (0.123) and still below the unrefined half-frame
uniform sigma (`1/(2*sqrt(3))` = 0.289). It is not fitted to pickleball and has not been
measured on pickleball. The result is a reported sigma 2.03x the measured spread --
conservative, and stated as such.

**(b) The bias term was reduced by the timing error removed, and no more.**
`anchor_uncertainty_for_bounce` gains one argument, `subframe_timing_sd_s`. When supplied, `dt`
becomes zero-mean Gaussian with that sigma instead of `U(-half_frame, +half_frame)`. The
one-sided `h = v_vertical * |dt|` height term **keeps its shape and its sign** -- `|dt|` is
still non-negative, so the modelled overshoot shrinks but does not vanish. Nothing claims the
two-sided meeting-point cancellation described in section 2.

The consequence is visible and deliberate: the model still reports **+0.055 to +0.092 m** of
bias while the **measured** residual is **-0.001 to -0.014 m**. The model is conservative by
roughly 6-8x on the bias. That is the correct direction to be wrong, and it is recorded here
rather than tuned away.

### Sigma calibration, per view

```
PYTHONPATH=. python3 scripts/tt3d/sigma_calibration.py \
  --out runs/lanes/subframe_bounce_timing_20260727/sigma_calibration.json
exit 0
```

| view / condition | BEFORE depth rms | BEFORE ratio | BEFORE cov | BEFORE bias | AFTER depth rms | AFTER ratio | AFTER cov | AFTER bias |
|---|---|---|---|---|---|---|---|---|
| back / no_noise | 0.1469 | 1.55 | 0.56 | +0.1083 | 0.0539 | 0.59 | 0.96 | -0.0089 |
| back / noise | 0.1545 | 1.63 | 0.47 | +0.1070 | 0.0974 | 1.06 | 0.78 | -0.0038 |
| side / no_noise | 0.0856 | 1.30 | 0.57 | +0.0677 | 0.0253 | 0.41 | 0.97 | -0.0068 |
| side / noise | 0.0914 | 1.39 | 0.51 | +0.0703 | 0.0421 | 0.68 | 0.86 | -0.0015 |
| oblique / no_noise | 0.1587 | 1.64 | 0.46 | +0.1236 | 0.0439 | 0.48 | 0.96 | -0.0144 |
| oblique / noise | 0.1626 | 1.69 | 0.49 | +0.1188 | 0.0725 | 0.79 | 0.86 | -0.0103 |

`ratio` is measured RMS / reported sigma; `cov` is the fraction within 1 sigma. The reported
sigma went from **optimistic by 1.30-1.69x** to **conservative at 0.41-1.06x**, and depth
coverage went from 0.46-0.57 to 0.78-0.97. The anchor is no longer over-trusted on its worst
axis.

---

## 6. The owner's pickleball bounce labels

```
PYTHONPATH=. python3 scripts/tt3d/owner_bounce_label_check.py
exit 0
```

Clip `wolverine_mixed_0200_mid_steep_corner`, 30 fps, real pickleball.

**On the label count.** At base commit `e209112` the label file
`runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json` holds 19 labels of
which **7** are `kind == "bounce"`. The brief said 8. The owner has since added an 8th (frame
299) in the main worktree, uncommitted at the time of writing; it is scored below as an
addendum, read-only, without importing anything from main. Two bounces carry `anchored_measured`
prefills, matching the quoted 0.316 / 0.276 m deltas.

**Second, what these labels can and cannot show.** Each label's 3D position is itself a
ray-plane intersection at a marked frame, produced by the same code path as the solver anchor.
It therefore **inherits the same sub-frame overshoot** and is *not* independent ground truth on
the depth axis. These labels can measure the **size** of the correction on real pickleball;
they cannot prove the position got more accurate.

| frame | status | dt (frames) | pixel shift | 3D anchor shift (m) | reported bias before -> after (m) |
|---|---|---|---|---|---|
| 12 | `rejected_search_bound` | -- | -- | 0.0000 | 0.2753 -> 0.2753 |
| 25 | `rejected_search_bound` | -- | -- | 0.0000 | 0.3403 -> 0.3403 |
| 104 | `no_observation_at_frame` | -- | -- | 0.0000 | 0.0793 -> 0.0793 |
| 157 | `refined` | -0.356 | 3.73 px | 0.2262 | 0.3006 -> 0.2364 |
| 217 | `refined` | -0.311 | 4.54 px | 0.1944 | 0.2748 -> 0.2222 |
| 272 | `refined` | -0.129 | 2.47 px | 0.0512 | 0.1866 -> 0.1481 |
| 289 | `refined` | -0.200 | 1.05 px | 0.1772 | 0.1777 -> 0.1445 |

**Findings.**

1. **The correction is large on real pickleball.** The 4 refined labels move by 0.051, 0.177,
   0.194 and 0.226 m (median **0.19 m**) -- comparable to the whole claimed sigma of 0.18 m. At
   30 fps with a grazing court camera, sub-frame timing is not a second-order effect.
2. **Every estimated `dt` is negative** (-0.13 to -0.36 frames): contact consistently happened
   *before* the frame that was marked. That is the expected signature of marking the first
   frame where the ball reads as being at the ground.
3. **The two `anchored_measured` labels did not move, and should not have.** At frames 12 and
   25 the 2D track's image-y is strictly monotone across the whole +-1-frame window -- there is
   no kink to find. The detector has no sighting at frames 8-9 (near label 12) or frame 28
   (near label 25), which is where contact actually falls. The marked frame is off by more than
   a frame, so the search-bound guard fires and the anchor is left alone. **The 0.316 and
   0.276 m deltas are therefore unchanged.** Repairing those would require moving the anchor
   several frames, which is a bounce-*detection* fix, not a timing fix -- and TT3D says this
   method cannot repair a mis-marked frame anyway (section 7).
4. The estimator fired on 4 of 7 and abstained cleanly on 3, on real gappy detector tracks.

**Addendum: the owner's 8th bounce label.** Re-run against the newer 8-bounce set in the main
worktree (`--labels`, read-only; nothing copied into this branch), output at
`owner_label_check_8labels.json`:

```
PYTHONPATH=. python3 scripts/tt3d/owner_bounce_label_check.py \
  --labels /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json
exit 0
```

The new label is frame **299** on a 300-frame clip, so there are not three post-contact
sightings to fit an outgoing branch: status `insufficient_observations`, anchor unmoved. **The
full-set result is 4 refined of 8, 4 abstained**, and the four that move are the same four. No
conclusion changes.

**Honest bottom line on pickleball: the owner labels do not confirm an accuracy improvement,
because they cannot. They confirm the correction is large, consistently signed, and that the
guards behave on real data.**

---

## 7. Limits

- **TT3D is TABLE TENNIS.** 25 fps, a 40 mm ball with ~3.3x the drag of a pickleball, and
  synthetic 2D observations. The timing **geometry** transfers exactly -- a bounce falls between
  frames in every sport, and the kink argument is sport-independent. The **magnitudes do not**.
  `VERIFIED=0` for pickleball stands.
- **The downstream trajectory metric did not uniformly improve** (section 4). Median better,
  tails worse, inside a metric with known run-to-run spread.
- **It cannot fix a mis-marked bounce frame.** Measured directly: with the marked frame
  displaced by +-1 frame, the guards refuse **648 of 816** bounces (79%), and across the whole
  set the systematic along-ray bias only moves from **+0.444 to +0.399 m** -- against +0.099 ->
  -0.008 m when the frame is marked correctly. The method fixes *timing*, not *detection*. This
  is by design: the +-1-frame search bound is a fence, not a tuning knob.
- **The refined `t` moves the anchor's time**, so segment boundaries shift downstream. That is a
  real behaviour change and is exactly why the knob is default OFF.
- **The timing sigma floor (0.25 frame intervals) is a declared prior, not a pickleball
  measurement.** It is conservative against table tennis by 2.03x. It has never been checked
  against a pickleball bounce.
- **The modelled bias remains ~6-8x larger than the measured residual bias.** Conservative, and
  deliberately not tuned to the error it predicts.
- **The branch order and sample count (quadratic, 6 per side) were selected by sweep on TT3D.**
  A sensitivity table over order in {1,2}, samples in {3..7} and displacement cap in
  {0.35, 0.5, 0.6, 0.8} is in the lane's working notes; the chosen point is not a sharp
  optimum (samples 5, 6 and 7 are within 0.002 m of each other on median) but it is tuned on
  table tennis and inherits that sport's frame rate.
- **No independent pickleball 3D bounce ground truth exists.** Until it does, the gate in
  `best_stack` cannot be satisfied by anything in this report.
- **Cost: ~25 ms per bounce** (81 grid points x 2 image axes x a 5x5 solve, pure Python), i.e.
  ~1.5 s on a 60-bounce clip. Acceptable while default OFF; the grid could be halved with the
  parabolic step absorbing the loss if this is ever switched on.
- **Audio was not used.** See section 2.

---

## 8. Test evidence

Real commands, real exit codes, unpiped.

| command | result | exit |
|---|---|---|
| `python3 -m pytest tests/racketsport/test_ball_subframe_bounce_timing.py -q` | 16 passed | 0 |
| `python3 -m pytest tests/racketsport/test_ball_arc_solver.py -q` (HEAD) | 64 passed, 1 failed, 1 skipped | 1 |
| `python3 -m pytest tests/racketsport/test_ball_arc_solver.py -q` (base e209112) | 64 passed, 1 failed, 1 skipped | 1 |
| `python3 scripts/tt3d/run_tt3d_validation.py` | wrote report | 0 |
| `python3 scripts/tt3d/sigma_calibration.py` | wrote report | 0 |
| `python3 scripts/tt3d/owner_bounce_label_check.py` | 4/7 refined | 0 |

### Failure attribution

`test_ball_arc_solver.py::test_wolverine_seg6_fixture_falls_back_to_anchor_bvp_and_render_samples_stay_in_bounds`
fails **identically at base commit e209112 and at HEAD** (64 passed / 1 failed / 1 skipped in
both). Verified by checking out the base file and re-running. **Pre-existing, not caused by
this lane.**

### Blast radius, base vs HEAD

Every test module that mentions `ball_arc_solver`, `best_stack`, `build_bounce_anchor`,
`anchor_uncertainty` or `BallArcSolverConfig` -- 26 modules, 325 tests -- run at both commits
with `--continue-on-collection-errors`.

```
cat blast.txt | xargs python3 -m pytest -q -p no:randomly --continue-on-collection-errors
HEAD  : 25 failed, 299 passed, 1 skipped, 7 errors    exit 1
base  : 27 failed, 281 passed, 1 skipped, 7 errors    exit 1   (16 fewer tests: the new file)
```

Diffing the sorted `FAILED`/`ERROR` lists:

- **Failures present at HEAD but not at base: ZERO.** No regression.
- 32 failures/errors are common to both.
- Two failures appear only at BASE and pass at HEAD --
  `test_flight_simulator.py::test_generate_corpus_is_deterministic_and_fast_for_small_cpu_sample`
  and `test_ball_arc_solver.py::test_fit_flight_segment_recovers_simulated_scalar_magnus_spin`.
  Both are wall-clock / optimizer-sensitive. Claimed as flakiness, not as a fix; `magnus_spin`
  was then re-run in isolation at BOTH commits and **fails at base and at HEAD alike**, which is
  what flaky means here rather than fixed.

`test_ball_arc_solver.py` is itself contention-sensitive: it reports 1 failure when run alone
and 3 when the machine is loaded, the two extras being `magnus_spin` and
`test_fit_flight_segment_shoots_drag_bvp_to_anchor_with_diagnostics`. The latter passes in
isolation at HEAD. Only the `wolverine_seg6` failure is stable, and it is stable at base too.

### Wide suite

`python3 -m pytest tests/racketsport -q` aborts with **30 collection errors** at HEAD **and at
base e209112** -- byte-identical lists of the 30 modules, diffed. Cause is environmental, not
code: `threed.racketsport.best_stack` fails to load because
`models/checkpoints/wasb/wasb_tennis_best.pth.tar` is gitignored and absent from this worktree,
and 30 test modules import it at collection time. **Pre-existing.** See section 8 addendum for
the continue-on-collection-errors run.

### Byte-identity when OFF

Two independent proofs, both in `tests/racketsport/test_ball_subframe_bounce_timing.py`:

1. `test_default_bounce_anchor_payload_is_byte_identical_to_the_base_commit` hashes the default
   `build_bounce_anchor` payload and compares to
   `58dd3851a35b35284937141304b3bf2c5c179a46a8af3a19491e744be39266ac`, generated by running the
   identical script against the base commit's `ball_arc_solver.py`. A single new key inside
   `uncertainty.terms` breaks it. This is why the unrefined path deliberately emits **no**
   `timing_model` key.
2. `test_solver_default_never_calls_the_estimator` monkeypatches `refine_bounce_contact_time` to
   raise and runs `solve_ball_arc_track` with the default config through the reviewed-bounce
   anchor path. Off does not merely produce the same bytes; the new code is never reached.
   `test_enabling_the_knob_reaches_the_estimator` proves that assertion is not vacuous.

### AGENTS.md structure checks

| check | HEAD | base e209112 | verdict |
|---|---|---|---|
| `list_scaffold_tools.py --root .` | 0 | 0 | unchanged |
| `audit_dead_code.py --root .` | 1 | 1 | **pre-existing fail**, unchanged status |
| `audit_storage_policy.py --root . --json` | 1 | 1 | **pre-existing fail**, unchanged, as expected |

`audit_dead_code` reports `status: fail` at both commits, but **this lane makes it one entry
worse and that is not hidden**. At base the unknown set is 2 files
(`scripts/tt3d/run_tt3d_validation.py`, `scripts/tt3d/tt3d_adapter.py`); at HEAD it is 3, the
extra one being this lane's `scripts/tt3d/owner_bounce_label_check.py`. It is the same class of
entry as the two already there: a validation CLI referenced only from lane evidence. It cannot
be resolved by documenting it here, because `runs/` is in the audit's `IGNORED_PARTS` and lane
reports are invisible to it. Resolving all three needs a decision about where external-validation
CLIs are registered, which is a repo-policy question this lane did not take unilaterally.

---

## 9. Files

| path | what |
|---|---|
| `threed/racketsport/ball_arc_solver.py` | `refine_bounce_contact_time`, `SubFrameBounceTiming`, `subframe_timing=` on `build_bounce_anchor`, `subframe_timing_sd_s=` on `anchor_uncertainty_for_bounce`, `enable_subframe_bounce_timing` config knob |
| `tests/racketsport/test_ball_subframe_bounce_timing.py` | 16 tests |
| `scripts/tt3d/run_tt3d_validation.py` | before/after in one pass |
| `scripts/tt3d/sigma_calibration.py` | sub-frame uncertainty calibration |
| `scripts/tt3d/owner_bounce_label_check.py` | real-pickleball magnitude check |
| `configs/racketsport/best_stack.json` | `ball.subframe_bounce_timing`, PENDING, `do_not_promote`, revision 16 |
| `runs/lanes/subframe_bounce_timing_20260727/` | `report.json`, `tt3d_report.json`, `sigma_calibration.json`, `owner_label_check.json` |

## 10. Verdict

Sub-frame bounce timing **removes most of the bounce-anchor error on TT3D** (median
0.0911 -> 0.0306 m, p95 0.3168 -> 0.1242 m) and **removes the systematic component almost
entirely** (+0.0993 -> -0.0076 m), on 816 bounces scored against external ground truth, with the
baseline reproducing the committed report exactly.

The downstream weak-segment trajectory metric **did not follow**: median improved 19.8% while
p90/p95/max regressed. That negative is preserved, not smoothed over.

This is table tennis. It is **not** a pickleball capability claim, it is **not** promoted, and it
stays **default OFF** behind `ball.subframe_bounce_timing` (PENDING, `do_not_promote`) until it is
measured against independent pickleball 3D bounce ground truth. `VERIFIED=0`.
