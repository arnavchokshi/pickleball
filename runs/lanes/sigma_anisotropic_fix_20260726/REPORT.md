# Anisotropic, calibration-floored, bias-aware bounce-anchor uncertainty

**Lane:** `sigma_anisotropic_fix_20260726` · **Branch:** `fix-sigma-20260726` · **Base:** `f29145a`
**Date:** 2026-07-26 · **VERIFIED = 0 for pickleball.**

## What this is, and what it is not

This is a **correctness fix, not an accuracy improvement.** The ball is no more accurate
afterwards than it was before. `anchor_sigma_for_bounce` was stating an uncertainty that was
wrong in three separate ways; it now states one that is materially less wrong. The number it
reports is **larger** — it claims **less** confidence than it used to. That is the correct
outcome, and any reading of this report that treats it as a capability gain is a misreading.

External validation is on **table tennis** (TT3D). It is not a pickleball promotion.

## The defect

`anchor_sigma_for_bounce` at base `f29145a` returned one scalar for an error that is neither
isotropic nor zero-mean. Measured against 139 TT3D trajectories with multi-camera ground truth,
136 scored bounces per view/condition:

| | measured |
|---|---|
| Optimistic on depth | RMS **1.65–2.97×** the reported sigma; only **29–47 %** of depth errors inside 1σ against the 68.3 % a Gaussian implies |
| Conservative in the image plane | **0.36–0.88×**; **78–97 %** coverage |
| Biased | **+0.068 to +0.124 m**, always directed **away from the camera** |
| Consumed isotropically | `ball_arc_solver.py:2357` divided all three world components by the same scalar |
| Calibration floor omitted | our clips measure **0.101 m** outdoor / **0.127 m** wolverine / **0.232 m** indoor; the estimate contained none of it |
| Silently non-radius-agnostic | `_ray_plane_pixel_sigma` was called with the module constant `BALL_RADIUS_M`, not the caller's `ball_radius_m` |

## Root cause of the bias — exact geometry, not a fudge

A bounce almost never lands on a sampled frame. The pixel used for the anchor is up to half a
frame interval from true contact, and during that interval the ball is still **above** the
plane. Forcing that pixel's ray down to `z = ball_radius` overshoots **along the ray** by exactly

```
overshoot = height_above_plane / |ray_direction_z|
```

The height is never negative, so the overshoot is never negative: a one-sided, systematic error
pointed away from the camera. Checked against TT3D's ground-truth ball height at each observed
frame, that single relation explains **73–97 %** of the along-ray error variance per view. It is
a geometric identity, not a fitted correction.

The same interval lets the ball travel sideways, which is genuinely zero-mean and belongs in
sigma rather than in the bias. On the side view that term predicts 5.10 m/s × 0.0116 s =
**0.059 m** of cross-track error; the measured cross-track RMS is **0.059 m**.

## What was built

`anchor_uncertainty_for_bounce` → `BounceAnchorUncertainty`, carrying `sigma_along_ray_m`,
`sigma_perp_m`, `bias_along_ray_m`, `ray_direction_unit`, `sigma_xyz_m`. Four first-principles
terms, none fitted to the error it predicts:

1. **Pixel sensitivity** — finite-difference the ray-plane intersection at ±`pixel_sigma_px`,
   decomposed along and across the ray (`ray_frame_plane_pixel_sigma`).
2. **Sub-frame timing** — the mechanism above. Its **mean** becomes `bias_along_ray_m`; only its
   **spread** enters sigma. A systematic error is never folded into a random-noise term.
3. **Calibration residual floor** — measured from the calibration's own reviewed correspondences
   (`calibration_plane_residuals`), previously omitted entirely.
4. **Plane-constraint shadow** — the anchor is pinned to the plane, so an along-ray error `a`
   forces an across-ray component `a·|ray_z|` just to stay on it.

The consumer at `ball_arc_solver.py:2357`, and the four other anchor-residual sites that share
the helper and the argument, now use `_ray_scaled_vec`: the offset is split into along-ray and
across-ray parts and each is divided by its own sigma. This is an **exact** whitening for the
modelled covariance `σ_along² dd^T + σ_perp² (I − dd^T)`, not an approximation. Anchors with no
attached uncertainty take the exact previous isotropic path, so the change is default-safe.

`sigma_xyz_from_ray` and `calibration_plane_residuals` moved from `ball_label_geometry` into
`ball_arc_solver`, which now owns the single definition; the labeller re-exports them and
`bounce_depth_sigma_m` delegates instead of separately recomputing a click term the solver sigma
already partly contained.

`_ray_plane_pixel_sigma` is deleted. Its replacement takes the plane height as a required
positional argument, so it cannot silently ignore the caller's ball radius.

## Before / after, per view

`PYTHONPATH=. python3 scripts/tt3d/sigma_calibration.py` → **exit 0**. The harness reproduces the
pre-fix formula verbatim, bug included, so both rows come from one run; the BEFORE row matches the
previously committed baseline exactly.

### BEFORE — isotropic scalar at `f29145a`

| view/cond | σ (m) | depth RMS | depth ratio | depth cov | image RMS | image ratio | image cov | depth bias |
|---|---|---|---|---|---|---|---|---|
| back/no_noise | 0.0566 | 0.1469 | **2.59** | **0.38** | 0.0201 | 0.36 | 0.97 | **+0.1083** |
| back/noise | 0.0566 | 0.1545 | **2.73** | **0.29** | 0.0223 | 0.39 | 0.97 | **+0.1070** |
| side/no_noise | 0.0517 | 0.0856 | **1.65** | **0.47** | 0.0453 | 0.88 | 0.79 | **+0.0677** |
| side/noise | 0.0517 | 0.0914 | **1.77** | **0.42** | 0.0457 | 0.88 | 0.78 | **+0.0703** |
| oblique/no_noise | 0.0549 | 0.1587 | **2.89** | **0.33** | 0.0408 | 0.74 | 0.83 | **+0.1236** |
| oblique/noise | 0.0548 | 0.1626 | **2.97** | **0.30** | 0.0413 | 0.75 | 0.83 | **+0.1188** |

### AFTER — ray-aligned, bias removed by the consumer

Speeds supplied as model input from TT3D's own measured contact physics (1.94 m/s vertical,
5.10 m/s horizontal, cv 0.3) — properties of table tennis, not parameters fitted to the error.

| view/cond | σ_along | σ_perp | aniso | bias | depth ratio | depth cov | image ratio | image cov | residual bias |
|---|---|---|---|---|---|---|---|---|---|
| back/no_noise | 0.0947 | 0.0509 | 1.9 | +0.1113 | **1.04** | **0.68** | 0.40 | 0.96 | **−0.0014** |
| back/noise | 0.0947 | 0.0509 | 1.9 | +0.1113 | **1.16** | **0.64** | 0.44 | 0.94 | **−0.0027** |
| side/no_noise | 0.0658 | 0.0512 | 1.3 | +0.0703 | **0.78** | **0.86** | 0.88 | 0.79 | **−0.0027** |
| side/noise | 0.0658 | 0.0512 | 1.3 | +0.0704 | **0.87** | **0.76** | 0.89 | 0.78 | **−0.0002** |
| oblique/no_noise | 0.0967 | 0.0505 | 1.9 | +0.1182 | **1.00** | **0.72** | 0.81 | 0.80 | **+0.0042** |
| oblique/noise | 0.0964 | 0.0505 | 1.9 | +0.1181 | **1.12** | **0.65** | 0.82 | 0.80 | **−0.0004** |

**Depth ratio 1.65–2.97 → 0.78–1.16. Depth coverage 0.29–0.47 → 0.64–0.86. Systematic bias
+0.068…+0.124 m → −0.003…+0.004 m.** Image-plane ratio 0.36–0.88 → 0.40–0.89.

### AFTER — bias reported but NOT removed (the shipped default)

`apply_bias_correction` is **off** by default: applying it moves the anchor, and moving the anchor
changes solver output. The bias is reported for consumers to act on, never silently applied.

| view/cond | depth ratio | depth cov | depth bias |
|---|---|---|---|
| back/no_noise | 1.55 | 0.56 | +0.1083 |
| back/noise | 1.63 | 0.47 | +0.1070 |
| side/no_noise | 1.30 | 0.57 | +0.0677 |
| side/noise | 1.39 | 0.51 | +0.0703 |
| oblique/no_noise | 1.64 | 0.46 | +0.1236 |
| oblique/noise | 1.69 | 0.49 | +0.1188 |

Better than 1.65–2.97 / 0.29–0.47, but **the bias still dominates until a consumer acts on it.**
Stated rather than hidden.

## Why anisotropy, and not just a bigger scalar

The obvious objection is that the old number could simply have been scaled up. Scoring all three
axes against the **corrected** sigma squeezed back into one scalar (the worst axis):

| view/cond | σ (m) | depth ratio | depth cov | image ratio | image cov |
|---|---|---|---|---|---|
| back/no_noise | 0.0947 | 1.04 | 0.68 | **0.21** | **1.00** |
| back/noise | 0.0947 | 1.16 | 0.64 | **0.24** | **1.00** |
| side/no_noise | 0.0658 | 0.78 | 0.86 | 0.69 | 0.87 |
| side/noise | 0.0658 | 0.87 | 0.76 | 0.69 | 0.87 |
| oblique/no_noise | 0.0967 | 1.00 | 0.72 | **0.42** | **0.96** |
| oblique/noise | 0.0964 | 1.12 | 0.65 | **0.43** | **0.96** |

Depth becomes honest and the image plane goes to ratio **0.21–0.69** at 87–100 % coverage — a
different lie in the opposite direction. The ray-aligned form reaches **0.40–0.89** at 78–96 % on
the same data. Two genuinely different accuracies need two numbers. That is the design argument.

## Sensitivity to the one prior the model cannot derive

The bias needs a per-sport bounce vertical-speed prior. Run with the **library pickleball
defaults** (4.0 / 8.0 m/s) on table-tennis data, i.e. deliberately the wrong sport's prior:

| view/cond | σ_along | bias | depth ratio | depth cov | image ratio | image cov | residual bias |
|---|---|---|---|---|---|---|---|
| back/no_noise | 0.1729 | +0.2294 | 0.88 | 0.61 | 0.24 | 1.00 | **−0.1179** |
| back/noise | 0.1729 | +0.2294 | 0.93 | 0.65 | 0.26 | 1.00 | **−0.1191** |
| side/no_noise | 0.1190 | +0.1450 | 0.78 | 0.71 | 0.53 | 0.93 | **−0.0776** |
| side/noise | 0.1191 | +0.1451 | 0.79 | 0.76 | 0.53 | 0.92 | **−0.0751** |
| oblique/no_noise | 0.1806 | +0.2436 | 0.86 | 0.62 | 0.48 | 0.95 | **−0.1225** |
| oblique/noise | 0.1804 | +0.2435 | 0.92 | 0.61 | 0.49 | 0.95 | **−0.1270** |

A 2× wrong speed prior over-corrects the bias by ~2× and leaves a residual bias of similar
magnitude to the original defect with the sign flipped. **The bias term is only as good as that
prior**, which is exactly why the correction is opt-in and gated rather than default-on.

## The 68.3 % target cannot be met at ratio 1.0, and should not be

`rms_ratio ≈ 1.0` and `coverage ≈ 0.683` coincide **only for a Gaussian error.** The dominant term
here is sub-frame timing, which is uniform in `dt`. For a uniform error at ratio 1.0 the correct
coverage is **0.577**, not 0.683. The measured 0.64–0.86 at ratio 0.78–1.16 is consistent with a
mixture of a uniform timing term and a Gaussian pixel term. Inflating sigma by ~1.18× would move
coverage to 0.683 and move the ratio away from 1.0. Since the solver uses sigma as a
**least-squares weight**, where RMS-matching is the statistically correct criterion, the default
is RMS-matched and both numbers are reported. Nothing was tuned toward whichever looked better.

## Owner labels — honest, and inconclusive

The 19 owner-made wolverine labels
(`runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json` — owner-made
today, **uncommitted** in the main worktree and therefore absent from this branch; read
read-only) contain 7 bounce labels, all `prefill_corrected`. Pipeline-prefill error against the
owner's corrected label:

| label | frame | band | 3D err | along-ray | across | old claimed σ | corrected σ_along | corrected bias |
|---|---|---|---|---|---|---|---|---|
| f000012 | 12 | anchored_measured | 0.316 | +0.313 | 0.047 | 0.181 | 0.266 | +0.275 |
| f000025 | 25 | anchored_measured | 0.276 | −0.275 | 0.029 | 0.181 | 0.319 | +0.340 |
| f000104 | 104 | arc_weak | **24.753** | +0.086 | 24.753 | 0.197 | 0.155 | +0.079 |
| f000157 | 157 | arc_weak | 0.376 | +0.157 | 0.342 | 0.180 | 0.290 | +0.301 |
| f000217 | 217 | arc_weak | 1.572 | +1.569 | 0.108 | 0.182 | 0.269 | +0.275 |
| f000272 | 272 | arc_weak | 0.662 | −0.662 | 0.027 | 0.182 | 0.208 | +0.187 |
| f000289 | 289 | arc_weak | 0.210 | +0.205 | 0.047 | 0.183 | 0.203 | +0.178 |

**Verdict: inconclusive. It does not confirm the fix at the expected rate.** 0/7 were inside the
old claimed 1σ; **1/7** are inside the corrected anisotropic 1σ, **2/7** after removing the
modelled bias. That is nowhere near 68 %, and the reason is that five of these seven prefills are
`arc_weak` — arc-solver output rather than measured bounce anchors — including a **24.75 m**
error. No bounce-anchor uncertainty model can or should cover an arc-solver failure of that kind;
it is a different defect on a different lane.

Restricted to the two `anchored_measured` prefills, the only ones this sigma actually describes:
errors 0.316 / 0.276 m against corrected σ_along 0.266 / 0.319 m, i.e. |error|/σ of 1.18 and
0.86 — consistent with the TT3D result, 1 of 2 inside. **With n = 2 that is an anecdote, not a
calibration.** The corrected σ (median 0.266 m) is 2.2× the old claimed 0.182 m, which is the
direction the wolverine evidence supports; the coverage rate is not established.

Sign check: f000012 errs +0.313 m away from the camera as modelled, but f000025 errs −0.275 m
toward it. **The modelled bias sign is not confirmed on wolverine.**

On the 4 committed outdoor demo bounce labels the corrected σ_along is **0.175–0.205 m** against
the old solver's 0.100–0.111 m. That independently converges on the labelling tool's own
separately derived 0.165–0.197 m: the tool was right and the solver was ~1.8× optimistic, which
is what this fix removes. All 3 prefill deltas there (0.016–0.034 m) sit inside the corrected
bound.

## Evidence and reproduction

- TT3D data re-fetched at commit `a2ef524ea0400262d6808db6cacf4a0b90bd0ad7`; **all 977 files
  match** `runs/lanes/tt3d_external_validation_20260726/data_manifest.sha256` byte-for-byte.
- Upstream repo carries **no LICENSE**. Internal validation only: never train on it, never ship
  it, never admit it to the data ledger.
- `PYTHONPATH=. python3 scripts/tt3d/sigma_calibration.py` → exit 0; writes
  `runs/lanes/tt3d_external_validation_20260726/sigma_calibration.json`.
- Machine-readable numbers: `runs/lanes/sigma_anisotropic_fix_20260726/report.json`.

## Test evidence

Interpreter: `/Users/arnavchokshi/Desktop/pickleball/.venv/bin/python` (3.14.6), the one AGENTS.md
prescribes. Four gitignored artifacts referenced by `best_stack.json`
(`models/checkpoints/wasb/wasb_tennis_best.pth.tar`,
`models/checkpoints/court_unet_v2/court_model_v2.pt`, and two `runs/` calibration paths) are
absent from this worktree and were **symlinked** from the main checkout so the manifest loads and
collection succeeds. Without them 29 test modules fail at import, at base as well. No tracked file
was touched by that.

**Focused — the new behaviour**

```
pytest tests/racketsport/test_ball_arc_solver.py -p no:randomly \
  -k "uncertainty or ray_scaled or ray_whitening or calibration_residual or radius_agnostic or isotropic or bounce_uncertainty"
  9 passed, 57 deselected                                                    exit 0
```

**Blast radius** — `tests/racketsport tests/tt3d -p no:randomly -k "ball or arc or solver or label
or anchor or sigma or uncertain or tt3d or world or trust"`:

```
HEAD 203cc79 : 39 failed, 1612 passed, 4 skipped, 3442 deselected  (495.00s)   exit 1
base f29145a : 38 failed, 1605 passed, 4 skipped, 3442 deselected  (276.25s)   exit 1
```

Attribution by `comm` over the sorted `FAILED` node ids:

```
only at base (fixed or flaky)        : (empty)
only at HEAD (candidate regression)  : test_ball_arc_solver.py::test_wolverine_seg6_fixture_falls_back_to_anchor_bvp_and_render_samples_stay_in_bounds
```

The counts reconcile exactly: 8 tests added, and `wolverine_seg6` moved from passed to failed
(1605 − 1 + 8 = 1612; 38 + 1 = 39).

**That one candidate regression is not one.** It is bounded by `SEGMENT_WALL_CLOCK_BUDGET_S = 5.0`,
and this fixture sits right on the boundary. Run alone on a quiet machine it fails **2 of 3 times
at base** and 4 of 4 at HEAD — a coin flip, not a verdict. Timing it directly with the budget
lifted to 600 s, so the measurement is not clipped by the cap it is testing:

```
HEAD: status=ran seg0=fit_bvp_fallback times=[5.55, 5.05, 5.06] median=5.06s
BASE: status=ran seg0=fit_bvp_fallback times=[5.71, 5.36, 6.47] median=5.71s
```

Both produce exactly the statuses the test asserts, and **HEAD is faster than base**, so the change
did not push it over the budget. Independently, an earlier focused batch of
`test_ball_arc_solver.py + test_ball_label_studio.py + test_ball_anchor_evidence.py` under a
different interpreter gave **2 failed, 118 passed, 1 skipped, exit 1 — byte-identical failure names
at base and at HEAD**, with this same test failing at base.

The 38 shared pre-existing failures are media- and artifact-dependent
(`test_build_pbvision_ball_sst.py`, `test_ball_wasb_dataset.py`, `test_ball_stage2_training.py`,
`test_build_person_fewshot_pack.py`, `test_coords_parity_real_fixture.py`, `test_schemas.py` and
others) and fail identically with this worktree checked out to `f29145a`.

**AGENTS.md structure checks**, HEAD vs base, same interpreter:

```
list_scaffold_tools.py --root .          exit 0    (base: exit 0)
audit_dead_code.py     --root .          exit 1    (base: exit 1)
audit_storage_policy.py --root . --json  exit 1    (base: exit 1)
```

`audit_dead_code` reports `python_sources: 646` and `unknown_python_sources: 2` on **both** sides,
the same two pre-existing unknowns (`scripts/tt3d/run_tt3d_validation.py`,
`scripts/tt3d/tt3d_adapter.py`). No new module was introduced, so nothing new is unregistered.
`audit_storage_policy` exit 1 is pre-existing as the task noted; diffing its JSON at base against
HEAD, **every violation list is identical** (16 large tracked files, 128 missing-allowed untracked
source files, 13 generated artifacts). The only difference anywhere in the payload is
`directory_usage_bytes.runs` growing 310451152 → 310503368, i.e. this lane's own report.json.
Unchanged, not worsened.


## What remains imperfect

1. **The bias needs a per-sport vertical-speed prior.** The one input the model cannot derive; it
   scales the bias linearly. The library defaults are pickleball priors that have **not** been
   measured against pickleball ground truth.
2. **`sigma_perp` assumes an unknown ball heading**, splitting sub-frame horizontal travel
   isotropically in the plane. In the TT3D back view the ball flies along the camera axis with
   almost no cross-track motion, so `sigma_perp` is over-stated there by ~2.5× (image ratio
   0.40–0.44). Fixing this needs the ball velocity, which the uncertainty function is not given.
3. **The ray-frame covariance is assumed diagonal.** Any real along/across correlation is
   unmodelled. The whitening is exact for the model, not for reality.
4. **TT3D could not exercise the calibration-residual floor at all.** Its calibration is synthetic
   and exact — the mapping reproduces its own pixels to 5.4e-13 px — so that term was 0
   throughout. The floor is evidenced only by our clips' measured residuals and the outdoor label
   set.
5. **Bias correction is off by default**, so in the shipped posture the depth ratio is 1.30–1.69
   and coverage 0.46–0.57.
6. **The wolverine owner labels did not confirm the corrected coverage rate** (2/7), for the
   reasons above.
7. **External validation is table tennis.** VERIFIED = 0 for pickleball. The `best_stack` entry
   `ball.bounce_anchor_uncertainty` is **PENDING / `do_not_promote`** with a pickleball
   calibration gate.
