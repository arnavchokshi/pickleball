# Overlapping Court Calibration Goal

Date: 2026-07-03
Status: ran_not_verified
CAL status: not_cal3_verified=true, verified=false

## Goal

Build a fail-closed calibration path for shared tennis/pickleball courts where
white tennis lines overlap pickleball geometry. The requested target is a mean
court residual near 0.2 ft against reviewed labels, without silently promoting a
contaminated or non-baseline score.

## Source Ideas Taken Seriously

1. Isolate pickleball paint color with strict HSV masks instead of generic
   white-line detection.
2. Detect/crop around the pickleball net before line extraction to reduce
   far-side clutter.
3. Cluster Hough line candidates into court boundaries.
4. Train/fine-tune a ResNet50 keypoint regressor on LabelMe-style 12/15 point
   court annotations.
5. Use Levenberg-Marquardt optimization to refine geometry against image
   evidence.
6. Extend beyond the original list with joint camera/distortion optimization,
   point+line residuals, a MobileNetV3-small keypoint backbone, and a shadow
   normalization fallback.

## Research Notes

- OpenCV camera calibration estimates intrinsics, distortion, rotation, and
  translation by minimizing reprojection error. That matches the joint camera
  optimization direction, but robust calibration normally expects multiple
  calibration views. Our reviewed court labels are single-view court frames, so
  distortion estimates can overfit and are not proof of production calibration.
- Perspective-n-Point/Line work supports using both point and line constraints
  for pose. The current implementation is a practical LM point+line residual
  seam, not a full minimal PnL solver.
- TorchVision MobileNetV3-small is a real lightweight pretrained backbone. It
  is a better iPhone-facing baseline candidate than ResNet50, but it still needs
  real court-label training before claiming accuracy.
- ShadowFormer, SID, and DHAN are real pretrained shadow-removal research
  families with public code/weight instructions. Their released weights are not
  currently configured in this repo, so the new ML-shadow lane is implemented
  as an explicit TorchScript adapter that fails closed when no real weight file
  is provided.

References:
- OpenCV camera calibration docs:
  https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
- TorchVision MobileNetV3-small docs:
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
- PnP/PnL research page:
  https://alexandervakhitov.github.io/pnpl/
- ShadowFormer official implementation:
  https://github.com/guolanqing/shadowformer
- SID official implementation:
  https://github.com/cvlab-stonybrook/SID
- DHAN shadow-removal implementation:
  https://github.com/mducducd/Shadow-Removal

## Implemented

- `threed/racketsport/overlapping_court_calibration.py`
  - Strict HSV paint mask + clustered Hough segment extraction.
  - Optional net-aware near-side crop path.
  - LM homography refinement over reviewed 15-point labels.
  - Joint focal length, pose, and radial distortion LM fit.
  - Robust metric-plane camera LM fit that optimizes inverse court-space
    backprojection residual directly instead of only pixel reprojection error.
  - Point+line LM fit using reviewed keypoints plus supported Hough line
    observations, with sampled 2D segment pixels scored against projected court
    lines.
  - All-15 keypoint camera fit diagnostic using the three reviewed net
    landmarks as elevated points, scored back on the same 12 floor points.
  - Diagnostic-only pair-subset oracle over persistent line observations to
    quantify the best possible gain from line-quality selection without
    promoting label-leaking selection into production.
  - Diagnostic top-residual refit path that drops the largest full-label
    metric-plane residuals, refits on the remaining inliers, and reports
    inlier-only versus all-original-label residuals separately.
  - Extended drop-5 diagnostic progression for quantifying how much residual is
    concentrated in a small set of reviewed keypoints without promoting that
    label-derived subset selection.
  - Metric-plane outlier visual review packet renderer for high-residual
    keypoints. It writes diagnostic crops and a contact sheet without mutating
    reviewed labels.
  - Expected-line intersection diagnostics for top-residual exclusions. Each
    dropped keypoint now records its expected court-line pair, whether that
    intersection evidence is available, and the exact missing-line reason when
    unavailable.
  - Full-intrinsics top-residual line-intersection override diagnostic. It
    combines the free `fx/fy/cx/cy` camera model with the strict endpoint
    line-intersection replacement set, while keeping the score out of safe
    selection and original reviewed labels unchanged.
  - Full-intrinsics all-strict endpoint line-intersection diagnostic. It uses
    every canonical endpoint intersection with strict overlapping-segment
    support, without top-residual ranking, to test whether the sub-0.2 result
    survives a less label-selected line-evidence policy.
  - Line-quality telemetry and quality-gated full-intrinsics endpoint
    line-intersection sweep. It records candidate angle agreement,
    perpendicular distance, segment overlap, and optional model-projection
    proximity, then tests fixed quality profiles without residual-rank label
    selection.
  - Model-projected line-observation selector. It matches raw temporal
    Hough/LSD segments against the current camera model projection instead of
    reviewed line positions, and reports whether reviewed line matching was
    used.
  - Read-only neural court-keypoint checkpoint evidence scanner. It parses
    existing `court_keypoint_metrics.json` artifacts, resolves checkpoint
    paths, ranks real-label median pixel errors, and marks every candidate as
    diagnostic/non-promoted unless its reviewed-label gate passes.
  - Deterministic LAB luminance shadow-normalization fallback.
  - Fail-closed TorchScript pretrained shadow-removal adapter. It records model
    path, SHA-256, device, and candidate model family provenance when real
    weights are configured, and records `pretrained_model_used=false` when they
    are not.
- `threed/racketsport/court_finding_technology_benchmark.py`
  - Line-candidate benchmark adapters for raw Hough, HSV paint Hough, HSV+net
    crop Hough, shadow-normalized Hough, and pretrained-ML shadow-removal Hough.
- `threed/racketsport/court_detector_v2_model.py`
  - ResNet50 court keypoint regressor head.
  - MobileNetV3-small court keypoint regressor head.
  - Fail-closed MobileNetV3-small checkpoint evaluator for explicit
    `sigmoid_normalized_xy` court-regressor checkpoints. It computes reviewed
    label error/PCK when a compatible checkpoint exists and reports
    unavailable status when no checkpoint is present.
- `scripts/racketsport/evaluate_overlapping_court_calibration.py`
  - Reviewed-label metric report for homography LM, joint distortion camera
    fit, and point+line fit coverage.
- `scripts/racketsport/render_overlapping_court_outlier_review_packet.py`
  - Visual packet CLI for metric-plane outlier crops and support counts.
- `scripts/racketsport/render_overlapping_court_top_residual_refit_review_packet.py`
  - Visual packet CLI for the keypoints excluded by the diagnostic top-residual
    refit, including inlier-only and all-original-label residual context.
- `scripts/racketsport/evaluate_court_finding_technologies.py`
  - Reviewed-label line-support benchmark for detection technologies.

## Hard Label-Data Results

Reviewed label slice:
- 4 full 15-point court clips.
- 1 partial visible-label clip excluded from metric homography scoring but used
  for line-candidate support scoring.

Calibration report:
`runs/overlapping_court_calibration_20260703/neural_keypoint_evidence_report.json`

| Metric | Result |
| --- | ---: |
| Corner-seed mean residual | 0.518219 ft |
| LM homography mean residual | 0.414584 ft |
| LM improvement | 0.103635 ft |
| Clips under 0.2 ft target | 1 / 4 |
| Joint distortion reprojection RMSE mean | 4.591091 px |
| Joint distortion inverse court residual mean | 0.378245 ft |
| Robust metric-plane camera residual mean | 0.332284 ft |
| Full-intrinsics metric-plane residual mean | 0.270737 ft diagnostic-only |
| Full-intrinsics reprojection RMSE mean | 5.953761 px |
| Metric-plane global trimmed worst-8 residual mean | 0.196048 ft diagnostic-only |
| Metric-plane per-clip trimmed worst-3 residual mean | 0.183579 ft diagnostic-only |
| Top-3 residual refit inlier-only residual mean | 0.127569 ft diagnostic-only |
| Top-3 residual refit scored against all original labels | 0.354965 ft |
| Minimum top-residual drops for mean inlier target | 2 per clip diagnostic-only |
| Minimum top-residual drops for every clip inlier target | 4 per clip diagnostic-only |
| Top-4 residual refit inlier-only residual mean | 0.090304 ft diagnostic-only |
| Top-4 residual refit worst clip inlier residual | 0.158915 ft diagnostic-only |
| Top-4 residual refit review packet | 16 items across 4 clips |
| Top-4 residual refit expected-line status | 14 available, 2 missing `far_baseline` |
| Top-4 residual refit review packet line support | 7 model-closer, 5 reviewed-label-closer, 2 ambiguous/tie, 2 missing line intersection |
| Top-5 residual refit inlier-only residual mean | 0.080915 ft diagnostic-only |
| Top-5 residual refit worst clip inlier residual | 0.169655 ft diagnostic-only |
| Top-5 residual refit scored against all original labels | 0.366454 ft |
| Top-5 residual refit expected-line status | 18 available, 2 missing `far_baseline` |
| Top-residual strict line-intersection override residual | 0.236469 ft diagnostic-only |
| Top-residual strict line-intersection override candidates | 13 used |
| Top-residual strict override scored against original labels | 0.435706 ft |
| Full-intrinsics top-residual strict line override residual | 0.193404 ft diagnostic-only |
| Full-intrinsics top-residual strict line override candidates | 13 used |
| Full-intrinsics top-residual override scored against original labels | 0.408027 ft |
| Full-intrinsics all-strict endpoint line override residual | 0.230184 ft diagnostic-only |
| Full-intrinsics all-strict endpoint line override candidates | 30 used |
| Full-intrinsics all-strict override scored against original labels | 0.655275 ft |
| Full-intrinsics quality-gated endpoint line override best residual | 0.182784 ft diagnostic-only |
| Full-intrinsics quality-gated endpoint line override best profile | `tight_overlap35_dist12_angle8_model24` |
| Full-intrinsics quality-gated endpoint line override candidates | 25 used |
| Full-intrinsics quality-gated endpoint override worst clip | 0.236466 ft |
| Full-intrinsics quality-gated endpoint override scored against original labels | 0.494922 ft |
| Model-projected quality-gated endpoint override residual | 0.182784 ft diagnostic-only |
| Model-projected quality-gated endpoint override uses reviewed line positions | false |
| Top-residual relaxed all-available line override residual | 0.244716 ft diagnostic-only |
| Top-residual relaxed all-available line override candidates | 14 used |
| Top-residual relaxed override scored against original labels | 0.435871 ft |
| Metric-plane high-residual review candidates | 12 total, 10 line-intersection-supported |
| Outlier packet line support | 6 model-closer, 4 reviewed-label-closer, 2 missing line intersection |
| Endpoint-only line-intersection override residual | 0.243114 ft diagnostic-only |
| Endpoint-only override candidates | 8 used, 2 center keypoints skipped |
| Override fit scored against original reviewed labels | 0.379335 ft |
| All-15 net-constrained floor residual mean | 5.643976 ft |
| Point+line fit coverage | 4 / 4 clips |
| Point+line sampled segment pixels per clip | 69.75 mean, 9.0 per line observation |
| Point+line reprojection RMSE mean | 4.971532 px |
| Point+line inverse court residual mean | 0.438441 ft |
| Point+line temporal line observations | 5-8 per full clip |
| Best per-clip point+line weight-sweep residual mean | 0.376670 ft |
| Safe selected camera residual mean | 0.332284 ft |
| Safe selected source split | 4 metric-plane camera |
| Diagnostic pair-subset oracle residual mean | 0.361120 ft |
| Neural checkpoint metric artifacts scored | 19 |
| Neural real-label candidates | 17 |
| Neural checkpoint gate pass count | 0 / 19 |
| Best neural real-label median error | 27.538259 px |
| Best neural real-label p95 error | 262.527094 px |
| Best neural PCK at 5 px | 0.054444 |
| MobileNetV3 court-regressor checkpoint candidates | 4 |
| MobileNetV3 court-regressor scored candidates | 4 |
| Best MobileNetV3 held-out median error | 317.563592 px |
| Best MobileNetV3 held-out PCK at 5 px | 0.000000 |

Line-candidate report:
`runs/overlapping_court_calibration_20260703/current_line_finding/court_finding_technology_benchmark.json`

ML shadow-removal benchmark:
`runs/overlapping_court_calibration_20260703/ml_shadow_removal_benchmark/court_finding_technology_benchmark.json`

Metric-plane outlier review packet:
`runs/overlapping_court_calibration_20260703/metric_plane_outlier_review_packet/metric_plane_outlier_review_packet.json`

Contact sheet:
`runs/overlapping_court_calibration_20260703/metric_plane_outlier_review_packet/metric_plane_outlier_contact_sheet.jpg`

Top-residual refit review packet:
`runs/overlapping_court_calibration_20260703/top_residual_refit_review_packet/metric_plane_top_residual_refit_review_packet.json`

Top-residual refit contact sheet:
`runs/overlapping_court_calibration_20260703/top_residual_refit_review_packet/metric_plane_top_residual_refit_contact_sheet.jpg`

Drop-5 top-residual refit review packet:
`runs/overlapping_court_calibration_20260703/top_residual_refit_drop5_review_packet/metric_plane_top_residual_refit_review_packet.json`

Drop-5 top-residual refit contact sheet:
`runs/overlapping_court_calibration_20260703/top_residual_refit_drop5_review_packet/metric_plane_top_residual_refit_contact_sheet.jpg`

| Technology | Mean reviewed-line support |
| --- | ---: |
| raw OpenCV Hough | 0.8083 |
| strict HSV paint Hough | 0.0000 |
| strict HSV paint + net crop Hough | 0.0250 |
| shadow-normalized Hough | 0.7167 |
| pretrained-ML shadow-removal Hough | 0.0000, unavailable because no real TorchScript weights are configured |

MobileNetV3 keypoint backbone check:
- TorchVision MobileNetV3-small is wired as a lightweight court-keypoint
  regressor head with output shape `[1, 45]` for 15 `(x,y,visibility)`
  keypoints.
- Parameter count: 1,563,981.
- The evaluator is now test-covered with an explicit synthetic
  `sigmoid_normalized_xy` checkpoint and fail-closed missing-checkpoint path.
- A real CPU four-fold clip-holdout trial now trains MobileNetV3-small direct
  regressors on the current reviewed label rows. Artifacts live under
  `runs/overlapping_court_calibration_20260703/mobilenet_v3_direct_regressor/`.
- The CAL scanner uses each checkpoint's sibling holdout metrics report instead
  of rescoring trained checkpoints on all eval rows, preventing contaminated
  all-row MobileNetV3 scoring.
- Current MobileNetV3 direct-regression result: 4 scored holdout candidates;
  best median error `317.563592 px`, best PCK@5 `0.000000`. This is far worse
  than the existing heatmap checkpoint evidence and does not help CAL.

Existing neural checkpoint evidence:
- The report now scores existing `court_keypoint_metrics.json` artifacts instead
  of leaving the heatmap-checkpoint lane unmeasured.
- Best current checkpoint:
  `runs/cal_external_retrain_20260702T003120Z/internal/cfgC_tier1tier2_e800/court_keypoint_heatmap.pt`.
- Best current real-label metric: `27.538259 px` median, `61.445160 px` mean,
  `262.527094 px` p95, and `0.054444` PCK at 5 px across 120 real holdout rows.
- Gate status: failed. The gate threshold is `0.95` PCK at 5 px, and no scanned
  checkpoint passed.

## Decisions From Results

- Strict HSV paint masking does not work on the current label slice. It is a
  useful opt-in path only for courts whose pickleball lines are actually a
  separable saturated paint color.
- Net crop did not rescue HSV on this data. It raised mean line support from
  0.0000 to only 0.0250.
- Shadow normalization should not be default. It reduced support from 0.8083 to
  0.7167 on this label slice.
- The pretrained-ML shadow-removal adapter is now real and test-covered with a
  TorchScript image-to-image model path, but the current checkout has no
  configured pretrained shadow-removal weights. On the reviewed label slice it
  therefore fails closed with zero candidates and `pretrained_model_used=false`,
  rather than silently falling back to LAB normalization and overclaiming AI.
- LM homography helps but does not reach the requested 0.2 ft target globally.
- Joint distortion fitting is now implemented and test-covered, but a single
  planar court view is not enough evidence to trust distortion parameters as a
  production calibration answer.
- Robust metric-plane LM is the strongest current non-oracle calibration path.
  It improves mean residual from `0.378245 ft` to `0.332284 ft` by optimizing
  the actual floor-plane backprojection metric instead of only pixel
  reprojection. It is still not close enough to the requested `0.2 ft` target.
- A freer full-intrinsics metric-plane diagnostic (independent `fx/fy`, free
  principal point, pose, and radial distortion) improves the mean reviewed-label
  residual to `0.270737 ft`, but still misses `0.2 ft` and produces implausible
  intrinsics on some clips (for example very large `fy`). This says extra lens
  degrees of freedom explain some residual, but not enough to promote the model;
  the safe selected camera remains the fixed-center metric-plane fit at
  `0.332284 ft`.
- Residual diagnostics show the target miss is concentrated: trimming the worst
  8 of 48 floor-keypoint residuals gives `0.196048 ft`, and trimming the worst
  3 keypoints within each clip gives `0.183579 ft`. This is diagnostic-only and
  does not verify CAL, but it strongly points to reviewed-label/keypoint
  definition outliers rather than a uniformly bad camera solve.
- A deterministic top-residual refit diagnostic now drops the three largest
  residual keypoints per clip, refits on the remaining inliers, and reports both
  scores. The inlier-only score crosses the target at `0.127569 ft`, but the
  same refit scored against all original reviewed labels is `0.354965 ft`.
  This proves the current camera model can fit a consensus subset, but also
  proves the result must not be promoted until the excluded labels/evidence are
  human-reviewed or independently replaced.
- The top-residual progression shows that dropping 2 worst residual keypoints
  per clip crosses the mean inlier target, but dropping 4 per clip is the first
  setting where every clip is under `0.2 ft` inlier residual. The top-4 packet
  renders those 16 excluded keypoints. Expected-line intersection evidence is
  now available for 14 of those 16 items. The only missing evidence cases are
  Indoor `far_left_corner` and Indoor `far_right_corner`, both because
  `far_baseline` is not observed in the line-evidence set. Among packet items,
  line evidence is closer to the refit/model projection on 7 items, closer to
  the reviewed point on 5 items, ambiguous/tied on 2 items, and unavailable on
  2 items. This is now the highest-priority review queue because it directly
  corresponds to the all-clips inlier score crossing the target.
- A drop-5 diagnostic pushes inlier-only residual further to `0.080915 ft`
  mean with worst clip `0.169655 ft`, and 18 of 20 excluded keypoints have
  expected-line intersections available. But the same drop-5 refit scored
  against all original reviewed labels is `0.366454 ft`, still worse than the
  safe `0.332284 ft` metric-plane baseline. This strengthens the outlier-review
  case but is not a production score.
- Worst metric-plane keypoints by clip are now explicit in the report:
  Burlington `far_right_corner` (`0.978100 ft`), Indoor `far_nvz_left`
  (`2.056184 ft`), Outdoor `far_nvz_left` (`0.475651 ft`), and Wolverine
  `far_right_corner` (`0.961704 ft`).
- The report now emits outlier review candidates without mutating labels. Top
  candidates include reviewed pixel, model-projected pixel, and Hough/LSD
  line-intersection pixel when both supporting court lines are available. The
  current queue is 12 high-residual candidates, 10 with line-intersection
  support after adding centerline evidence.
- The visual packet renders all 12 current candidates. Line-intersection
  evidence is closer to the metric-plane projection on 6 candidates, closer to
  the reviewed point on 4 candidates, and unavailable on 2 candidates. This is
  enough to prioritize review, but not enough to auto-correct labels.
- A diagnostic endpoint-only line-intersection override refit improves the
  adjusted-label residual from the safe `0.332284 ft` baseline to `0.243114 ft`,
  but still misses the requested `0.2 ft` target. Scoring that same override fit
  against the original reviewed labels is worse (`0.379335 ft`), proving this
  must stay a human-review diagnostic and not a silent scoring change.
- A broader top-residual strict line-intersection override can use 13 endpoint
  candidates from the drop-4 review queue. It improves the adjusted-label
  residual slightly further to `0.236469 ft`, but still misses the `0.2 ft`
  target. Scoring the same fit against original reviewed labels is worse
  (`0.435706 ft`), so this also remains diagnostic-only and cannot become a
  baseline or hidden label correction.
- Combining that same strict top-residual line-intersection set with the
  full-intrinsics metric-plane camera produces the first non-trimmed diagnostic
  below the requested target: `0.193404 ft` mean residual on the temporary
  line-override observations. It still scores `0.408027 ft` against the
  original reviewed labels, so it is evidence that the residual problem is
  localizable to a small set of keypoint/line observations, not a CAL promotion.
  The safe selected camera remains the fixed-center metric-plane fit at
  `0.332284 ft`.
- The non-top-residual version was also tried: full-intrinsics fitting with all
  strict canonical endpoint line intersections. It uses 30 endpoint candidates
  across the 4 full clips, but scores only `0.230184 ft` on temporary override
  observations and `0.655275 ft` against original reviewed labels. This misses
  the target and is much worse than the 13-candidate top-residual diagnostic,
  so the next production-shaped step is line-quality selection, not blindly
  replacing every available endpoint with a line intersection.
- The production-shaped line-quality sweep now records angle agreement,
  perpendicular distance, segment overlap, and optional distance from the
  current full-intrinsics model projection for every line-intersection
  candidate, then tests five fixed quality profiles. The best profile,
  `tight_overlap35_dist12_angle8_model24`, improves the all-strict endpoint
  diagnostic from `0.230184 ft` to `0.182784 ft` and reduces the candidate set
  from 30 to 25. This crosses the temporary override target, but it still has a
  worst clip of `0.236466 ft` and scores `0.494922 ft` against original reviewed
  labels. This is useful evidence that model-proximity line identity matters,
  not a CAL promotion.
- The same fixed quality profiles now also run on a model-projected
  line-observation lane. That lane matches raw temporal Hough/LSD segments
  against the current full-intrinsics court projection instead of reviewed line
  positions, and reproduces the `0.182784 ft` temporary override residual with
  `uses_reviewed_line_positions_for_matching=false`. This removes one source of
  label contamination from the line selector, but it is still diagnostic because
  the camera seed and score are evaluated on reviewed labels.
- A relaxed all-available top-residual line override was also tested. It uses
  14 available intersections, including non-strict or centerline-supported
  evidence, but worsens the adjusted-label residual to `0.244716 ft` and
  original-label residual to `0.435871 ft`. More line intersections are
  therefore not automatically better; the strict endpoint support policy remains
  the better diagnostic bound on this label slice.
- Centerline collinear evidence makes Burlington `far_nvz_center` and
  `near_nvz_center` reviewable, but blindly feeding center keypoints into the
  override refit worsens the diagnostic residual. The default override strategy
  therefore uses endpoint intersections only and skips center keypoints.
- First candidates to inspect visually:
  Burlington `far_right_corner`: reviewed `[1850.629, 515.839]`, model
  `[1867.024, 518.358]`, line intersection `[1867.755, 517.905]`; Indoor
  `far_nvz_left`: reviewed `[693.706, 553.881]`, model `[695.958, 560.310]`,
  line intersection `[700.341, 561.000]`; Wolverine `far_right_corner`:
  reviewed `[653.427, 327.867]`, model `[648.921, 330.460]`, line intersection
  `[641.679, 336.101]`.
- The all-15 elevated net-landmark fit failed badly on the current labels
  (`5.643976 ft` floor residual). Treat the reviewed net landmarks as
  diagnostic/visual points for now, not metric camera constraints.
- Persistent temporal Hough/LSD fixed point+line observation coverage from 1/4
  to 4/4 full clips. The PnL path now samples 2D points along each selected
  segment and minimizes distance from those sampled segment pixels to the
  projected court line. That is closer to the requested raw segment-pixel
  objective, but the current default point+line weight worsened the inverse
  court residual to `0.438441 ft` versus the point-only distorted camera fit
  (`0.378245 ft`) and robust metric-plane fit (`0.332284 ft`).
- A reviewed-label diagnostic weight sweep prevents PnL from silently worsening
  the report, but the robust metric-plane camera now beats both the point+line
  sweep and the pair-subset oracle on this slice.
- The new pair-subset oracle shows that perfect label-backed line selection
  would improve mean residual only to `0.361120 ft`. That means the current
  persistent line observations contain some useful signal, but line selection
  alone is not enough to reach the requested `0.2 ft` target.
- MobileNetV3-small direct regression has now been tried on a non-leaky
  clip-holdout split. It fails badly on this tiny label slice: best held-out
  median error is `317.563592 px` and PCK@5 is `0.000000`. Existing heatmap
  checkpoints remain much stronger but still do not solve CAL: best median
  error is `27.538259 px`, and the best gate value is only `0.054444` PCK at
  5 px versus the `0.95` pass threshold.
- For AI shadow removal to become an accuracy candidate, configure real
  ShadowFormer/SID/DHAN-style weights through
  `PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT`, rerun the ML-shadow benchmark, and
  compare against raw Hough `0.8083` and LAB-shadow Hough `0.7167`.

## Next Todos

- Use the robust metric-plane camera fit as the current best non-oracle
  calibration candidate, but keep it behind reviewed-label gates until it clears
  the `0.2 ft` target across more clips.
- Review or re-annotate the worst residual keypoints first: Burlington
  `far_right_corner`, Indoor `far_nvz_left`, Outdoor `far_nvz_left`, and
  Wolverine `far_right_corner`. The current diagnostics indicate these outliers
  are the fastest path to proving or disproving the `0.2 ft` target.
- Human-review the generated outlier packet before changing any reviewed label
  JSON. If any label edits are accepted, rerun the full reviewed-label CAL
  report and compare the safe selected residual against `0.332284 ft` and the
  `0.2 ft` target.
- Treat the 13 full-intrinsics strict top-residual line-override candidates as
  the highest-value review queue, because they cross `0.2 ft` only when the
  temporary line evidence replaces reviewed points and fall back to `0.408027 ft`
  on the untouched reviewed labels.
- Do not promote the all-strict endpoint policy. It is less label-selected than
  the top-residual queue, but its `0.230184 ft` override score and `0.655275 ft`
  original-label score show that unfiltered line intersections are too noisy.
- Do not promote the quality-gated endpoint policy either. The best fixed
  model-proximity profile now crosses the temporary override target at
  `0.182784 ft`, but the selected profile is still chosen by reviewed-label
  outcome inside this diagnostic report and still scores `0.494922 ft` against
  original labels. The model-projected lane removes reviewed line-position
  matching, but the next improvement must predeclare one profile and validate it
  on held-out clips before any production PnL path uses these intersections.
- Do not use the reviewed net landmarks as elevated metric constraints until a
  separate review pass confirms they are labeled on the actual net top and not
  on visible net/floor intersections.
- Add a production-safe line-quality predictor before PnL. The reviewed-label
  pair oracle proves there is a small upside from better line selection, but
  its ceiling on this slice is still `0.361120 ft`, so the next selector must
  be measured against the robust metric-plane baseline, the oracle, and the
  distorted-camera baseline.
- Add line color consistency and geometry-template competition before feeding
  point+line LM, then rerun the weight sweep to see if the selected PnL model
  beats distorted-camera residual by more than noise.
- Add a centerline-specific quality gate before any center keypoint
  line-intersection can influence fitting. The current centerline evidence is
  useful for human review but not reliable enough for the correction-impact
  refit.
- Convert one official shadow-removal model family to the TorchScript adapter,
  set `PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT`, and rerun
  `opencv_hough_pretrained_shadow_removed` against the reviewed label slice
  before treating AI shadow removal as an accuracy candidate.
- Add a real PnL or solvePnP-with-line formulation after line observations are
  reliable enough to avoid optimizing against noise.
- Add LabelMe importer/exporter only if a real dual-line court annotation corpus
  exists or is created. Do not count synthetic or copied labels as independent
  human verification.
- Do not spend more local CPU time on direct MobileNetV3 regression with only
  the current 32 reviewed/static-copy rows. The non-leaky trial is already far
  below the heatmap checkpoint. A useful next neural attempt needs either a
  much larger real corpus or synthetic pretraining followed by a real held-out
  gate.
- Keep all overlapping-court helpers opt-in until they beat raw Hough and pass
  CAL guard thresholds on reviewed labels.
