# W7 OURS vs PB.Vision on the same Wolverine footage

## Result

Frame-level comparison is permitted. The strongest mapping is:

`ours_frame = PB_global_frame = PB_rally_local_frame + 45`

The integer offset is `0` on the global 300-frame grid. Standardized `u/v`
position cross-correlation is `r=0.9492` on 137 high-confidence paired
detections; the independent 12-point court match has `5.67 px` median
residual. The peak is broad by one frame, so all event timing keeps a
conservative `+/-1 frame` residual uncertainty.

The head-to-head does **not** support a blanket claim that our normal-fit 3D
is close to PB. The one normal-fit segment inside PB's 3D coverage is close in
height and speed but is displaced by about `1.34 m` cross-court and `4.49 m`
along-court. Our fallback output is much worse: its paired height bias is
`+7.923 m`, and the worst comparable segment has a `+20.700 m` apex delta.
PB's emitted 3D remains within a plausible height range, but that is not due
to flagged interpolation: only `1/173` emitted PB 3D frames is marked
interpolated and `79/252` rally frames have no PB 3D at all.

This is comparator measurement, not ground truth. Product status remains
`VERIFIED=0`.

## Acceptance summary

| # | Deliverable | Numeric result | Status |
|---:|---|---|---|
| 1 | Alignment | global offset `0`; `u/v r=0.9492`; court median `5.67 px`; `+/-1 frame` uncertainty | measured; frame-level comparison allowed |
| 2 | Unit/frame reconciliation | explicit feet-to-meters transform; camera-center check `2.69 cm` | measured |
| 3 | 2D BALL | coverage `58.73%` PB vs `80.56%` ours; paired p50/p95 `5.69/163.76 px` | measured |
| 4 | 3D BALL | normal-fit paired height bias `-0.347 m`; fallback `+7.923 m`; core run1/run6 arrays identical | measured; hypothesis only partially supported |
| 5 | Interpolation policy | PB true interpolation `0.58%` of emitted; ours fallback `87.67%` of all frames | measured |
| 6 | Bounce pseudo-GT | two common events; mean landing errors PB `0.220 m`, ours `0.269 m` | measured, non-independent pseudo-GT |
| 7 | Camera | center delta `0.0269 m`; FOV delta `0.274 deg`; cross-projection median `8.93 px` | geometries agree |
| 8 | Stats glance | PB `8.366 s`, `57.32 ft`; ours about `10.0 s`, `19.626 m` | reported; windows/IDs not directly matched |

## Inputs and comparison policy

- PB owner export: `cv_export.json`, `insights.json`, and `stats.json` for
  `vid=emwt3u5kzavy`.
- Ours run1: the nested
  `wolverine_mixed_0200_mid_steep_corner/` directory under
  `runs/lanes/w7_critique_20260709/wolv_world/`.
- Ours run6: `runs/lanes/w7_speedgate_20260709/critique_world/wolverine_run6/`.
- Image coordinates are evaluated at `1920x1080`.
- A PB 2D detection means `actions.ball.confidence >= 0.5`. The export has an
  `actions.ball` record on every rally frame, including very low-confidence
  guesses, so counting every record as a detection would be misleading.
- PB 3D uses only `balls.ball.court_position` records with
  `interpolated=false`.
- Ours 3D uses the frame-aligned `ball_track_arc_solved.json`. Dense render
  comparisons use only frame times bracketed by samples from the same segment;
  extrapolating to omitted render times is not allowed.
- Differences are `ours - PB` after coordinate conversion.
- Bland-Altman limits of agreement are `mean difference +/- 1.96 * sample SD`.

## 1. Alignment

PB's rally begins at global frame `45`, has `252` frames, and spans global
frames `45..296`. At 30 fps this is `251/30 = 8.3667 s` between first and last
sample, matching `insights.json`'s `1500..9866 ms` span.

For every integer lag, PB `u/v` and our visible WASB `x/1920,y/1080` were
standardized by axis and correlated on co-detected frames. The selected lag
uses the confidence policy above.

| Mapping candidate (`ours = PB_global + lag`) | Joint standardized `u/v` correlation | Paired n | Median direct pixel error |
|---:|---:|---:|---:|
| `-1` | `0.9372` | 136 | `12.15 px` |
| **`0`** | **`0.9492`** | **137** | **`5.69 px`** |
| `+1` | `0.9421` | 138 | `20.27 px` |
| `+2` | `0.9115` | 138 | `36.95 px` |

A parabolic interpolation through the `-1/0/+1` position-correlation samples
places the peak at approximately `+0.13 frame`. That does not justify
sub-frame precision. The confidence/detection-envelope cross-correlation is
weak and broad (`r=0.336` at lag `+1`, versus `0.315` at lag `0`), hence the
reported `+/-1 frame` residual event-time uncertainty.

### Independent court-point sanity check

The 12 PB court points map one-to-one to 12 of our 15 reviewed calibration
points. No duplicate nearest neighbor is used.

| Court-point comparison | n | median | p95 | max |
|---|---:|---:|---:|---:|
| PB exported points vs our reviewed image points | 12 | `5.67 px` | `18.98 px` | `23.57 px` |

The position correlation, the large direct-error advantage at lag 0, the
court-point match, and the exact timing-span match are jointly strong enough
to continue. The temporal confidence envelope alone would not be.

## 2. Unit and frame reconciliation

PB uses feet with a corner-origin court frame: `x=0..20 ft` across the court
and `y=0..44 ft` along it. Our frame is meters, centered at the net, with
`x=+/-3.048 m` and `y=+/-6.7056 m`. The inferred transform is:

```text
x_ours_m = (x_PB_ft - 10) * 0.3048
y_ours_m = (22 - y_PB_ft) * 0.3048
z_ours_m = z_PB_ft * 0.3048
```

The negative sign on the along-court axis is required by both the court-point
ordering and the camera position.

| Check | PB after transform | Ours | Difference |
|---|---|---|---:|
| Camera center x/y/z | `[-4.8874,-9.0933,1.7213] m` | `[-4.8698,-9.0770,1.7337] m` | `0.0269 m` Euclidean |
| Court width/length | `20/44 ft = 6.096/13.4112 m` | `6.096/13.4112 m` | `0` |
| Shared observed court points | 12 PB points | 12 unique points from our 15 | median `5.67 px` |

PB player positions range from `x=-0.82..18.58 ft` and
`y=-5.95..47.57 ft`; the small excursions outside regulation bounds are
consistent with player feet, not a different unit scale.

## 3. 2D BALL agreement

### Coverage over the aligned rally

| State | Frames | Fraction of 252 |
|---|---:|---:|
| PB detected (`confidence >= 0.5`) | 148 | `58.73%` |
| Ours visible | 203 | `80.56%` |
| Both | 137 | `54.37%` |
| PB only | 11 | `4.37%` |
| Ours only | 66 | `26.19%` |
| Neither | 38 | `15.08%` |

### Pixel agreement on the 137 common detections

| Metric | Error at 1920x1080 |
|---|---:|
| mean / RMSE | `21.92 / 69.59 px` |
| p25 / p50 / p75 | `3.82 / 5.69 / 9.50 px` |
| p90 / p95 / p99 | `15.56 / 163.76 / 333.63 px` |
| max | `493.15 px` |
| `<=5 / <=10 / <=20 px` | `43.80% / 76.64% / 91.24%` |
| `<=50 px` | `94.16%` |

Most paired points agree closely, but the tail is real and cannot be hidden by
the median. Eight paired frames exceed 50 px. The largest disagreements are
frame 133 (`493.15 px`), frame 124 (`337.39 px`), and frame 123
(`326.86 px`); PB is high-confidence on all three while the tracks select
different image locations.

## 4. 3D BALL head-to-head

PB emits 173 3D frames. Removing its one `interpolated=true` record leaves
172 paired frames. These cover our segments 2 through 7. Normal-fit segment 1
(`12..25`) is outside PB's emitted 3D range, so only normal-fit segment 5 can
be evaluated.

### Per-axis and Euclidean differences

| Our segment class | n | Segment IDs | mean delta x/y/z (m) | p95 absolute x/y/z (m) | Euclidean p50 / p95 / max (m) |
|---|---:|---|---|---|---|
| all | 172 | 2,3,4,5,6,7 | `[-0.056,+4.573,+6.865]` | `[2.445,12.441,21.633]` | `5.047 / 23.781 / 24.591` |
| normal fit | 22 | 5 only | `[+1.338,+4.490,-0.347]` | `[1.483,4.694,0.486]` | `4.649 / 4.952 / 5.130` |
| BVP fallback | 150 | 2,3,4,6,7 | `[-0.260,+4.585,+7.923]` | `[2.622,12.686,21.686]` | `5.569 / 23.886 / 24.591` |

Paired PB heights range from `0.056..2.831 m`. Paired fallback heights from
ours range from `-0.102..23.530 m`. The normal-fit segment's height and speed
are fairly close, but its planar position is not.

### Bland-Altman

| Quantity | Class | n | Bias (`ours-PB`) | 95% LoA |
|---|---|---:|---:|---:|
| per-frame height | all | 172 | `+6.865 m` | `[-9.908,+23.638] m` |
| per-frame height | normal fit | 22 | `-0.347 m` | `[-0.502,-0.193] m` |
| per-frame height | fallback | 150 | `+7.923 m` | `[-9.079,+24.924] m` |
| consecutive-frame speed | all | 161 | `-2.058 m/s` | `[-15.587,+11.471] m/s` |
| consecutive-frame speed | normal fit | 20 | `-0.150 m/s` | `[-3.304,+3.004] m/s` |
| consecutive-frame speed | fallback | 137 | `-2.518 m/s` | `[-16.868,+11.832] m/s` |
| per-segment paired-frame apex | all | 6 | `+3.545 m` | `[-13.116,+20.205] m` |
| per-segment paired-frame apex | normal fit | 1 | `-0.315 m` | not estimable (`n=1`) |
| per-segment paired-frame apex | fallback | 5 | `+4.317 m` | `[-13.844,+22.477] m` |

For speed, PB's paired median/p95/max are `9.127/14.641/15.007 m/s`; ours
are `7.016/14.897/23.300 m/s`. Similar marginal p95 values do not imply
frame-level agreement; the wide limits of agreement show that errors cancel.

### Apex by comparable segment

| Segment | Our status | Paired n | Ours apex | PB apex | Delta |
|---:|---|---:|---:|---:|---:|
| 2 | fallback | 74 | `23.530 m` | `2.831 m` | `+20.700 m` |
| 3 | fallback | 56 | `4.776 m` | `2.126 m` | `+2.650 m` |
| 4 | fallback | 7 | `0.037 m` | `0.556 m` | `-0.519 m` |
| 5 | normal fit | 22 | `0.968 m` | `1.283 m` | `-0.315 m` |
| 6 | fallback | 7 | `0.048 m` | `0.794 m` | `-0.747 m` |
| 7 | fallback | 6 | `0.334 m` | `0.835 m` | `-0.500 m` |

### Dense render cross-check

The dense renderer omits portions of invalid/out-of-volume arcs, so it is
evaluated only where samples bracket the frame time in the same segment. It
brackets 233/300 solved frames and 162/172 PB non-interpolated pairs.

| Render comparison | n | Result |
|---|---:|---|
| dense render vs solved 3D, bracketed frames | 233 | p50/p95/max `0.035/0.408/0.669 m` |
| render vs PB, normal fit | 22 | Euclidean p50/p95 `4.644/4.916 m` |
| render vs PB, fallback | 140 | Euclidean p50/p95 `5.672/23.935 m` |

Rendering does not repair the 3D disagreement.

### Hypothesis ruling

- **“On well-fit segments we are close”: only partially supported.** Height
  (`-0.347 m` bias), speed (`-0.150 m/s` bias), and apex (`-0.315 m`) are
  reasonably close on segment 5, but planar biases of `+1.338 m` and
  `+4.490 m` make the full 3D positions not close. Segment 1 is untestable.
- **“On fallback we are absurd”: supported for the dominant long fallback.**
  Segment 2 reaches `23.530 m` while PB reaches `2.831 m`; fallback height
  bias is `+7.923 m`. Not every short fallback is high—several are instead
  too low—so the defect is inconsistency, not a uniform upward scale error.
- **“They stay plausible via interpolation”: contradicted by the export.** PB
  stays in a plausible emitted range while marking only one frame
  interpolated and omitting 79 rally frames. The export does not support
  attributing the visual result to broad honest interpolation.

### Run1 vs run6 robustness

| Artifact | Numeric/semantic fields compared | Run1 vs run6 |
|---|---|---|
| `ball_track.json` | frames, bounces, fps, source | identical |
| `court_calibration.json` | points, intrinsics, extrinsics, homography | identical |
| `ball_track_arc_solved.json` | frames, segments, anchors, summary, status | identical |
| `ball_arc_render.json` | samples, segments, summary, solver status | identical |

Arc/render byte hashes differ because `generated_at` and input paths differ;
the compared numeric arrays and decisions are equal. Run6 does not contain
run1's physics-filled, net-plane, or virtual-world artifacts, so robustness is
claimed only for the available BALL/CAL/arc/render chain.

## 5. Interpolation and fallback policy

| Policy/coverage item | Count | Fraction |
|---|---:|---:|
| PB emitted 3D | 173/252 rally frames | `68.65%` |
| PB non-interpolated 3D | 172/252 | `68.25%` |
| PB `interpolated=true` | 1/252 rally frames | `0.40%` |
| PB `interpolated=true` among emitted 3D | 1/173 | `0.58%` |
| PB no emitted 3D | 79/252 | `31.35%` |
| Our BVP fallback, full clip | 263/300 | `87.67%` |
| Our BVP fallback, aligned rally | 229/252 | `90.87%` |
| Our BVP fallback among PB non-interpolated pairs | 150/172 | `87.21%` |

PB's emitted spans are `74..103`, `105..111`, `113..125`, `132..156`,
`158..170`, `172..202`, `204..216`, `218..225`, `227..249`, and
`251..260`. Of the 79 missing rally frames, 65 are leading/trailing
(`45..73` and `261..296`) and 14 are interior gaps. The sole interpolated
frame is 116, inside an otherwise emitted span.

Our fallback is not equivalent to PB's policy: ours emits solver-generated 3D
through weak/rejected segments, while PB primarily omits 3D where it does not
emit a trajectory.

## 6. Bounce pseudo-GT

Two bounce events have both a high-confidence PB `actions.bounce` peak, PB
downstream trajectory support, and an our bounce candidate within two frames.
The PB bounce pixel is back-projected through our calibrated `z=0` plane. PB's
landing is its `insights.json` trajectory endpoint transformed into our frame;
ours is `world_xy_at_ball_radius` from our bounce candidate.

| Event | Our/PB frame | PB bounce confidence | Pseudo-GT xy (m) | PB xy / error | Our xy / error |
|---|---|---:|---|---|---|
| 1 | `157 / 157` | `0.972` | `[-0.629,5.955]` | `[-0.605,6.331] / 0.377 m` | `[-0.706,5.659] / 0.306 m` |
| 2 | `216 / 218` | `0.873` | `[-0.732,5.068]` | `[-0.689,5.115] / 0.064 m` | `[-0.521,4.973] / 0.231 m` |
| **Mean** | — | — | — | **`0.220 m`** | **`0.269 m`** |

PB has a supported bounce near frame 104 that ours does not see, so it is not
included in the “both systems see” score. Later high `actions.bounce` peaks
after PB's 3D trajectory ends are also excluded. This pseudo-GT is not
independent: it uses PB's event pixel and our camera calibration.

## 7. Camera cross-check

PB's horizontal FOV is `1.212825 rad = 69.490 deg`; ours is
`1.208050 rad = 69.216 deg`. The difference is `0.004775 rad = 0.274 deg`.
The transformed camera centers differ by `0.0269 m`.

The export does not document its Euler convention. The convention used here
was inferred from the 12 court correspondences and produces a PB self-
reprojection RMS of `6.36 px`; it is therefore evidence-backed but not an
independently documented API contract.

| Projection check on 12 shared court points | median | RMS | p95 | max |
|---|---:|---:|---:|---:|
| PB model vs PB exported court observations | `3.88 px` | `6.36 px` | `11.93 px` | `13.88 px` |
| Our PnP model vs our reviewed observations | `4.45 px` | `7.87 px` | `13.76 px` | `14.19 px` |
| PB model vs our PnP model | `8.93 px` | `9.06 px` | `13.00 px` | `13.63 px` |
| PB model vs our reviewed observations | `8.15 px` | `8.85 px` | `12.27 px` | `12.59 px` |
| Our model vs PB exported observations | `8.32 px` | `10.65 px` | `18.08 px` | `18.20 px` |

The camera geometries agree closely at the image/court level. This does not
resolve monocular ball-depth ambiguity, and our calibration remains
`metric_confidence=low` with `reprojection_high` and
`single_view_planar_full_calibration` warnings.

## 8. Stats layer glance

| Quantity | PB | Ours | Comparability |
|---|---|---|---|
| analyzed time | rally `1500..9866 ms` = `8.366 s` | `300` frames at 30 fps, about `10.0 s` | different windows; PB rally maps to our frames `45..296` |
| player distances | `[9.41,9.50,6.21,22.20] ft` | `[8.375,4.735,4.962,1.554] m` for IDs 19..22 | IDs are not mapped |
| aggregate distance | `57.32 ft = 17.471 m` | `19.626 m` | ours includes about 1.63 s outside PB rally |
| shots/rally | 6 shots, 1 rally | excluded by our match-stats policy | not comparable |

The PB distance unit is inferred from the feet-valued court frame; the stats
file does not carry an explicit unit field on `total_distance_covered`.
Because windows and player identities differ, the movement totals are a
schema/scale glance only, not a system ranking.

## Limits and manager ruling

1. PB is not ground truth. This report measures agreement and physical
   plausibility only.
2. Only one of our two normal-fit segments overlaps PB's 3D, so no claim about
   both normal-fit segments is possible.
3. The 2D result is strong in the middle and poor in the tail; both facts must
   travel together.
4. PB's export supports omission, not broad flagged interpolation, as the main
   coverage policy.
5. The highest-impact owner-facing defect remains our fail-open fallback 3D.
   The cross-system comparison independently reproduces the prior diagnosis:
   segment 2's `23.530 m` apex is not supported by PB's emitted trajectory or
   our own 2D agreement.
6. No source or upstream artifact was edited. Outputs are confined to this
   lane directory.
