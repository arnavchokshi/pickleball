# PB Vision 2D to 3D lift reverse-engineering

Date: 2026-07-12  
Status: competitor-reference diagnostic only; `VERIFIED=0` unchanged.  
Protected Outdoor/Indoor clips were not read. PB output is not GT, training data,
or promotion evidence.

## Bottom line

The export is the Wolverine 300-frame clip already processed locally. The
match is strong enough for frame-level comparison: offset 0, standardized 2D
correlation `r=0.9492` on 137 paired detections, median paired court-point
residual `5.67 px`, and the same 30 fps / 300-frame camera span. The local
source is 1920x1080, 10.000 s, SHA-256
`7f6c33b7cfd94a063405b68708d37d968cc1850e7435aa875f5b30f0afb6cb4b`.

The strongest data-only hypothesis is a temporally selected, event-anchored,
piecewise ballistic trajectory. Shot and bounce event positions reproject to
their selected event pixels to numerical precision; selected bounces are
exactly at a constant ball-radius height; interior motion is smooth and close
to gravity; velocity changes concentrate at event boundaries. Ordinary ball
samples are allowed to deviate substantially from the raw selected ball pixel,
which is consistent with robust arc smoothing rather than exact per-frame ray
fitting.

That is a method-family inference, not source-code knowledge. The export does
not identify the optimizer, whether radius entered its objective, whether a
Kalman/UKF was used, or whether drag/Magnus terms were active. Fitting gravity
plus quadratic drag to the exported arcs does not identify meaningful drag on
most segments.

Our gap is not 2D coverage or the shared court camera. It is primarily event
segmentation/anchoring plus emission policy, with monocular depth constraints
and outlier robustness next. The current raw arc emits 3D on all 252 aligned
rally frames even though 229 are `fit_bvp_fallback`; the confidence-gated
world cuts this to 58/252, below PB's 183/252. The right target is recovery with
honest provenance, not either fail-open raw output or blanket suppression.

## 1. Exhaustive schema inventory

`schema_inventory.json` recursively inventories every leaf path, array path,
and object-key shape in all three supplied files. It found:

| File | Version | Top-level role | Leaf paths | Array paths | Object-shape paths |
|---|---:|---|---:|---:|---:|
| `cv_export.json` | 2.1.0 | camera, framewise CV/action heads, selected 3D, players | 75 | 10 | 30 |
| `insights.json` | 4.10.0 | one rally, six shots, trajectories, outcomes, coaching/derived facts | 255 | 138 | 55 |
| `stats.json` | 2.2.0 | game and four per-player aggregate statistic objects | 364 | 4 | 88 |

`insights.json` and `stats.json` share session ID `vid=emwt3u5kzavy`, session
index 0, type `game`, and four players. Insights contains one rally, six shots,
four player records, two highlights, four coach-advice records, and 120 derived
stats keys. Stats contains four player objects with 48 top-level keys each.
Those downstream files repeat the selected event endpoints/peaks and establish
rally structure, but they do not expose lift internals.

### Camera and coordinate frame

| Field | Export value | Interpretation and evidence |
|---|---|---|
| `camera.fps` | 30 | Matches the local clip. |
| `cameraSegments` | one, `s=0`, `e=299` | One static solve for the whole 300-frame clip. |
| `fov` | 1.212824996 rad = 69.490 deg | Horizontal-FOV interpretation matches our solve within 0.274 deg. |
| `position` | `[-6.0347, 51.8335, 5.6473]` | Feet in PB's corner-origin court frame. |
| `orientation` | pitch -0.24555, roll -0.01322, yaw 5.39428 | Radians; Euler convention is not documented and was inferred from court correspondences. |
| `court_points` | 12 normalized `u,v` | Confidence 0.924129 to 0.980230; spread 0.001061 to 0.007502. Spread units/meaning are not documented. |

The court frame is feet: x spans 0 to 20 across court, y spans 0 to 44 along
court, and z is height. The evidence-backed transform into our meter frame is:

```text
x_ours_m = (x_pb_ft - 10) * 0.3048
y_ours_m = (22 - y_pb_ft) * 0.3048
z_ours_m = z_pb_ft * 0.3048
```

After this transform, the camera centers differ by `0.02694 m`. Assigning the
12 unnamed PB court observations to 12 unique points in our matched reviewed
calibration gives PB self-reprojection p50/RMS/p95 `3.88/6.36/11.93 px` and
PB-camera vs our-camera court projection p50/p95 `8.93/13.00 px`.

### Temporal and per-frame structure

The single rally starts at global frame 45 and contains 252 records (45..296),
matching `insights` time 1500..9866 ms. Every frame has:

- top-level frame confidence;
- four `player_court_positions` in the feet frame;
- four normalized court player points with confidence;
- action heads `ball`, `bounce`, `net`, `shot`, `ball_projection`, and
  `player_projection`, each with normalized `u,v,confidence`;
- `ball_radius` on 205/252 frames, with normalized center, confidence, and
  `radius_v`.

`balls.selected` exists on 183/252 frames and chooses exactly one typed 3D
record:

| Selected type | Frames | Extra payload |
|---|---:|---|
| `ball` | 173 | `court_position`, `interpolated` |
| `shot` | 6 | position, player, speed variants, peak, usually height over net |
| `bounce` | 3 | position at constant z |
| `net` | 1 | terminal net-contact position |

The typed event rows are part of the 3D stream. Counting only `balls.ball`
understates PB coverage as 173 instead of 183.

## 2. Data-only method hypothesis

| Claim | Evidence | Strength / limit |
|---|---|---|
| The shipped 3D is piecewise event-anchored, not one unconstrained per-frame filter. | Six shot, three bounce, and one net typed positions partition the stream. Median velocity change is `14.109 m/s` at eight usable event boundaries versus `0.316 m/s` inside arcs, a `44.61x` ratio. Insights trajectories reuse the event endpoints. | Strong method-family evidence; exact segmentation algorithm unknown. |
| Bounce anchors are explicitly pinned to ball center at the court plane plus radius. | All three selected bounce z values equal `0.12383333 ft = 0.037744399 m`; max deviation from their median is below `3e-16 m`. | Strong. It does not prove how the bounce time/candidate was selected. |
| Event anchors are hard geometric constraints. | All 6 shot and all 3 bounce positions reproject through the inferred camera to their selected action pixel within `1.5e-12 px`; the net event differs by 19.05 px. | Strong for shots/bounces under the inferred camera convention. |
| Ordinary ball observations are robustly smoothed/rejected rather than exactly ray-fit. | Ordinary ball 3D to selected ball 2D error is `25.35 px` p50, `213.18 px` p95, `745.53 px` max. Even selected-action confidence >=0.5 has p50/p95 `22.84/103.18 px`. | Strong evidence of non-exact reprojection; could also include undocumented distortion or different internal candidates. |
| Candidate selection is temporal/global, not `argmax(confidence)` or a simple 0.5 threshold. | Selected type is the action argmax on only `148/183 = 80.87%`. The 69 omitted frames have max-action confidence p50/p95/max `0.542/0.938/0.968`. Seven bounce-confidence local peaks exceed 0.5, but only three are selected bounces. | Strong behavior inference; specific DP/beam/Viterbi method unknown. |
| Explicit interpolation is rare and omission is deliberate. | Combined output is two runs, frames 73..125 (53) and 132..261 (130), with missing runs 45..72 (28), 126..131 (6), 262..296 (35). Only frame 116 is flagged interpolated: `1/183 = 0.546%`. | Strong export behavior. `interpolated=false` must not be called measured: 39 selected actions have confidence <0.5. |
| Apparent radius is a useful depth cue and may be used. | On 168 frames with both radius and 3D, `radius_v` vs inverse camera depth has Pearson `r=0.850`, Spearman `0.774`, linear `R2=0.722`; median relative residual is 7.04%. | Strong cue value, hypothesis-only causal use. |
| Motion is ballistic between events. | Gravity+drag re-integration passes the declared no-GT rule on all 9 evaluable intervals. Segment RMSE p50/max is `0.023/0.135 m`; observed acceleration p50/p95 is `9.49/12.00 m/s2`; no step exceeds 35 m/s. Residual >=5 Hz energy fraction is 0.035..0.177 (median 0.097). | Strong piecewise-ballistic signature. Drag/Magnus are not identifiable: fitted drag is nearly zero on most intervals and two longer arcs dominate residual. |

The most economical reconstruction hypothesis is therefore:

1. a static court/camera solve from normalized court-point distributions;
2. multiple per-frame action/candidate heads, including ball radius and
   player/ball projections;
3. temporal selection of a rally-consistent event sequence, rejecting many
   high-confidence isolated action peaks;
4. hard shot/bounce endpoint constraints, with bounces at z=ball radius;
5. piecewise ballistic optimization/smoothing that may depart from noisy 2D;
6. conservative output-window and gap policy, with sparse explicit
   interpolation provenance.

This shape is also consistent with PB's public statements about equations of
motion over imprecise CV, but every claim above is supported from the supplied
data and does not depend on the public statement.

## 3. Three no-GT pillars on PB 3D

### Physics re-integration

Each interval between selected shot/bounce/net events was independently fit by
gravity plus quadratic drag, optimizing initial velocity and nonnegative drag.
The transparent diagnostic pass rule is RMSE <=0.15 m, p95 <=0.30 m, speed
<=35 m/s, and z >=-0.05 m.

| Segment frames | n | RMSE m | p95 m | max m | max speed m/s | Pass |
|---|---:|---:|---:|---:|---:|---|
| 73..103 | 31 | 0.1281 | 0.2394 | 0.3065 | 14.98 | yes |
| 104..111 | 8 | 0.0293 | 0.0446 | 0.0481 | 3.92 | yes |
| 112..156 | 39 | 0.0093 | 0.0146 | 0.0189 | 11.41 | yes |
| 157..170 | 14 | 0.0030 | 0.0044 | 0.0056 | 3.98 | yes |
| 171..202 | 32 | 0.1350 | 0.2521 | 0.3228 | 8.91 | yes |
| 203..216 | 14 | 0.0035 | 0.0052 | 0.0066 | 15.01 | yes |
| 217..225 | 9 | 0.0231 | 0.0357 | 0.0394 | 4.39 | yes |
| 226..249 | 24 | 0.0499 | 0.0874 | 0.1132 | 9.61 | yes |
| 250..260 | 11 | 0.0166 | 0.0256 | 0.0298 | 6.53 | yes |

This is self-consistency, not accuracy. Our raw parametric arcs re-integrate to
sub-nanometer numerical residual even when they are spatially absurd, proving
why this pillar cannot stand alone.

### Court-plane bounce pseudo-GT

The three selected PB bounces are frames 104, 157, and 217, with selected
confidence 0.924, 0.972, and 0.747. All are exactly at inferred radius
0.037744399 m. Raw confidence peaks alone are not pseudo-GT: peaks at 132,
216/218, 274, and 289 were not all selected and can be far from the plane.

### Internal reprojection

| Slice | n | p50 px | p95 px | max px |
|---|---:|---:|---:|---:|
| All selected 3D | 183 | 23.37 | 206.37 | 745.53 |
| Ordinary ball | 173 | 25.35 | 213.18 | 745.53 |
| Shot anchors | 6 | ~0 | ~0 | <1e-12 |
| Bounce anchors | 3 | ~0 | ~0 | <1.5e-12 |
| Net event | 1 | 19.05 | 19.05 | 19.05 |

The export has no distortion fields and no documented Euler convention, so
ordinary-ball tail error may include unexported camera details. Exact event
reprojection under the same inferred convention makes the anchor-vs-track
contrast real.

## 4. Clip match and local outputs

`insights.session.vid` is `emwt3u5kzavy`; the export itself has no filename or
content hash. The source mapping is nevertheless decisive for comparison:

| Evidence | Result |
|---|---|
| Duration/grid | PB camera 0..299 at 30 fps; local Wolverine 300 frames, 10.000 s at 30 fps |
| Rally timing | PB 45..296 / 1500..9866 ms falls inside the local grid exactly |
| 2D track alignment | offset 0, `r=0.9492`, 137 pairs; runner-up lag +1 is lower (`r=0.9421`) and has 20.27 px median error vs 5.69 px |
| Court imagery | 12 unique matches, 5.67 px median residual |
| Camera | center 2.69 cm apart; horizontal FOV 0.274 deg apart |

Processed local outputs exist in:

- `runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner/`
- `runs/lanes/demo_beststack_render_20260710/after_wolv/`
- `runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner/`

No owner action is required for clip identity. A cryptographic PB-source match
would require PB's original uploaded video hash, which the export does not
contain.

## 5. Ours versus PB on the matched frames

### Coverage, smoothness, physics, bounces, and net

| Metric | PB selected 3D | Our raw arc | Our confidence-gated world |
|---|---:|---:|---:|
| 2D visible/detected over rally | 148/252 = 58.7% | 203/252 = 80.6% | same source |
| 3D emitted over rally | 183/252 = 72.6% | 252/252 = 100% | 58/252 = 23.0% |
| Explicit interpolation / weak fallback | 1/183 = 0.55% interpolated | 229/252 = 90.9% fallback | 196/252 hidden-no-prediction display band |
| Physics-plausible intervals | 9/9 = 100% | 7/11 = 63.6% | 5/5 emitted intervals = 100% |
| Reprojection p50/p95 | 23.4/206.4 px overall | fit 2.52/15.30 px; fallback 455/3865 px | emitted overall 1.89/132.46 px |
| Speed p95/max | 14.62/15.01 m/s | 24.12/32.81 m/s | 9.54/10.28 m/s |
| Acceleration p95 | 12.00 m/s2 | 25.96 m/s2 | 10.98 m/s2 |
| Teleports >35 m/s | 0 | 0 | 0 |
| Net crossings / positive | 5/5 | 1/0, clearance -0.733 m | 0/0 |

At PB's three selected bounces, our raw z-minus-radius absolute errors are
21.247 m, 0.356 m, and 0.0875 m. Our declared bounce anchors themselves are
mathematically pinned at radius, but only two of the three PB bounces have an
our declared bounce within two frames; frame 104 is missed by 53 frames. That
separates “anchor constrained correctly” from “correct anchor selected.”

### 2D and 3D agreement

Our 2D coverage advantage is real: PB-only 11 frames, ours-only 66, both 137.
On overlaps, 91.24% are within 20 px, p50 is 5.69 px, but eight exceed 50 px
and p95 is 163.76 px.

Using all 183 typed PB 3D samples (differences are ours minus PB):

| Our output/status | n | 3D delta p50/p95/max m | Mean xyz delta m |
|---|---:|---|---|
| Raw fit | 23 | 4.64 / 4.95 / 5.13 | included in aggregate |
| Raw fallback | 160 | 5.63 / 23.91 / 24.76 | included in aggregate |
| Raw aggregate | 183 | 5.08 / 23.85 / 24.76 | `[-0.047, +4.610, +6.904]` |
| Gated emitted overlap | 30 | 4.65 / 5.17 / 5.40 | `[+1.395, +4.494, -0.386]` |

The remaining normal-fit disagreement is mostly planar, not height. Neither
system is truth, so 4.65 m is disagreement, not our measured error. It is large
enough to reject “normal fit matches PB in 3D.”

## 6. Gap decomposition

| Failure component | Attribution | Measured evidence | Reproduction implication |
|---|---|---|---|
| Segmentation / anchoring | Primary structural gap | Our current artifact has 0 contact anchors, 10 bounce anchors, 11 segments, and 9/11 fallback. PB exports 6 shots, 3 bounces, 1 net; only 2/3 PB bounces match our anchors within 2 frames. PB event delta-v is 44.6x interior. | Joint anchor search, multimodal contact anchors, then global DP if local search remains unstable. |
| Camera / court scale | Ruled down as dominant | Camera-center delta 2.69 cm, FOV delta 0.274 deg, camera-to-camera court projection p50 8.93 px. Yet our fit-only paired 3D differs 4.64 m p50. | Do not spend the next experiment on another camera solve; preserve camera checks as a control. |
| Fit robustness / monocular depth | Major residual gap | 8 paired 2D outliers >50 px; fallback reprojection p50 455 px. Fit reprojection is good at 2.52 px p50 but still disagrees 4.64 m in 3D. PB radius-depth R2 is 0.722 while our artifact reports size observation count 0. | Separate inlier selection from segmentation, pin both ends, and wire radius as an independent depth residual. |
| Gap filling / emission | Largest immediate visible gap | PB emits 72.6% and flags 0.55% interpolated. Raw ours emits 100% with 90.9% fallback; gated ours emits only 23.0%, leaving a 125-frame gap to PB coverage. | UKF/physics recovery must sit between raw fail-open and blanket suppression, with predicted provenance. |

## 7. Ranked reproduction map

1. **UKF seeded only from adjacent accepted fits, behind fail-closed gates.**
   Closes the immediate 125-frame coverage deficit between gated ours (58) and
   PB (183) without re-exposing 229 weak fallback frames. Expected measurable
   effect is up to +49.6 coverage points toward comparator parity, never a claim
   of accuracy. Kill if recovered intervals reduce the current emitted 5/5
   physics-plausible rate, create >35 m/s steps, worsen bounce/net plausibility,
   or are labeled measured.
2. **TT3D joint anchor-state search (core landed; run a fresh frozen harness
   candidate).** Closes 9/11 fallback and the missed frame-104 bounce. Expected
   effect per the standing kill is at least four recovered segments
   (`fallback <5/11`). Kill if a fresh same-input candidate remains >=5/11
   fallback; then candidate density/event evidence, not hard anchor state, is
   the bottleneck.
3. **Ball-radius depth residual (#9).** Closes the planar/depth ambiguity left
   after camera and 2D reprojection look good. PB's cue has `r=0.850`,
   `R2=0.722` on 168 frames; ours consumed 0 size observations. Expected first
   effect is nonzero supported size residuals and lower leave-one-out depth
   instability; a meter claim is not estimable without independent GT. Kill if
   held-out reprojection, bounce pseudo-GT, and segment stability do not improve
   under the same frozen 2D/camera inputs.
4. **Audio ordering plus BlurBall/WASB blur anchor boosters.** Closes 0 our
   contact anchors versus six typed PB shot boundaries and the missed bounce.
   It is the cheapest segmentation evidence experiment. Expected effect is
   additional correctly timed candidate anchors before any solver change. Kill
   if frozen reviewed event timing/precision does not improve, even if this
   competitor sequence looks closer.
5. **Both-ends pinning plus a dedicated inlier pass.** Closes the 8-frame >50
   px 2D tail, fallback reprojection p50 455 px, and remaining 4.64 m planar
   ambiguity on fit output. Expected effect is lower held-out/LOO reprojection
   and endpoint sensitivity, not forced PB agreement. Kill if the two accepted
   segments' held-out residual/endpoint stability and bounce/net pseudo-GT do
   not improve.
6. **Whole-rally DP segmentation.** Closes global over-segmentation (10 bounce
   anchors, no contacts) if the cheaper local anchor routes fail. Expected
   structural effect is fewer spurious boundaries and stable shot/bounce
   intervals across detector perturbations. Kill if it does not reduce fallback
   and boundary instability on the frozen clip set; do not tune it to PB's
   three bounces because PB is not GT.

Ranking #1 is the fastest honest visual parity path; #2-#5 are the accuracy
path. UKF cannot substitute for independent 3D ground truth.

## 8. Harness

```bash
.venv/bin/python runs/lanes/pbv_reveng_20260712/compare_vs_pbvision.py --pb-export runs/research_ball3d_20260709/pbvision_cv_export/cv_export.json --ours <our-run-dir-or-ball/world.json> --frame-offset auto --output <scorecard.json>
```

Supported our inputs are directories or JSON containing `frames[]` or
`ball.frames[]`, including plain 2D `ball_track.json`,
`ball_track_arc_solved.json`, physics-filled tracks, `virtual_world.json`, and
`confidence_gated_world.json`. Sibling ball track, arc, and calibration are
discovered when present; explicit overrides exist. The output records all
thresholds, the alignment mapping, per-segment physics rows, bounce and
reprojection pillars, coverage, smoothness, net crossings, Bland-Altman height
bias/limits, and policy disclaimers. Tests cover PB-only, raw arc, directory
world resolution, plain 2D ball track, auto alignment, and byte-equivalent
determinism after sorted JSON serialization.

## Honest limits

- PB is a comparator, not GT. Agreement is not accuracy.
- The Euler convention and feet transform are inferred, although court and
  event reprojection strongly support them.
- The PB export omits semantic identities for its 12 court points; PB-only
  court reprojection is therefore not computed without the matched local
  calibration.
- `interpolated=false` is not proven to mean directly measured. Thirty-nine
  selected rows have selected-action confidence below 0.5.
- Radius-depth correlation does not prove causal use in PB's optimizer.
- Physics self-consistency cannot detect a wrong but internally integrated arc.
- The physics pass thresholds are lane diagnostics, not promotion gates.
- PB's selected bounce sequence is not reviewed bounce GT. Confidence peaks
  can be false or duplicate.
- The source match is not cryptographic because the PB export has no source
  hash or filename.
- Only one current raw run and its gated world are scored here; TT3D/radius/UKF
  candidates require fresh same-harness runs.
- No best-stack, source, data, config, branch, commit, or staged-file change was
  made. Best-stack delta: none.
