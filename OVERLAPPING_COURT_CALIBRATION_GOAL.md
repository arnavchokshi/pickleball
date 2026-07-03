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
    observations.
  - All-15 keypoint camera fit diagnostic using the three reviewed net
    landmarks as elevated points, scored back on the same 12 floor points.
  - Diagnostic-only pair-subset oracle over persistent line observations to
    quantify the best possible gain from line-quality selection without
    promoting label-leaking selection into production.
  - Metric-plane outlier visual review packet renderer for high-residual
    keypoints. It writes diagnostic crops and a contact sheet without mutating
    reviewed labels.
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
- `scripts/racketsport/evaluate_overlapping_court_calibration.py`
  - Reviewed-label metric report for homography LM, joint distortion camera
    fit, and point+line fit coverage.
- `scripts/racketsport/render_overlapping_court_outlier_review_packet.py`
  - Visual packet CLI for metric-plane outlier crops and support counts.
- `scripts/racketsport/evaluate_court_finding_technologies.py`
  - Reviewed-label line-support benchmark for detection technologies.

## Hard Label-Data Results

Reviewed label slice:
- 4 full 15-point court clips.
- 1 partial visible-label clip excluded from metric homography scoring but used
  for line-candidate support scoring.

Calibration report:
`runs/overlapping_court_calibration_20260703/metric_plane_line_override_endpoint_report.json`

| Metric | Result |
| --- | ---: |
| Corner-seed mean residual | 0.518219 ft |
| LM homography mean residual | 0.414584 ft |
| LM improvement | 0.103635 ft |
| Clips under 0.2 ft target | 1 / 4 |
| Joint distortion reprojection RMSE mean | 4.591091 px |
| Joint distortion inverse court residual mean | 0.378245 ft |
| Robust metric-plane camera residual mean | 0.332284 ft |
| Metric-plane global trimmed worst-8 residual mean | 0.196048 ft diagnostic-only |
| Metric-plane per-clip trimmed worst-3 residual mean | 0.183579 ft diagnostic-only |
| Metric-plane high-residual review candidates | 12 total, 10 line-intersection-supported |
| Outlier packet line support | 6 model-closer, 4 reviewed-label-closer, 2 missing line intersection |
| Endpoint-only line-intersection override residual | 0.243114 ft diagnostic-only |
| Endpoint-only override candidates | 8 used, 2 center keypoints skipped |
| Override fit scored against original reviewed labels | 0.379335 ft |
| All-15 net-constrained floor residual mean | 5.643976 ft |
| Point+line fit coverage | 4 / 4 clips |
| Point+line reprojection RMSE mean | 4.617988 px |
| Point+line inverse court residual mean | 0.388882 ft |
| Point+line temporal line observations | 5-8 per full clip |
| Best per-clip point+line weight-sweep residual mean | 0.376467 ft |
| Safe selected camera residual mean | 0.332284 ft |
| Safe selected source split | 4 metric-plane camera |
| Diagnostic pair-subset oracle residual mean | 0.363960 ft |

Line-candidate report:
`runs/overlapping_court_calibration_20260703/current_line_finding/court_finding_technology_benchmark.json`

ML shadow-removal benchmark:
`runs/overlapping_court_calibration_20260703/ml_shadow_removal_benchmark/court_finding_technology_benchmark.json`

Metric-plane outlier review packet:
`runs/overlapping_court_calibration_20260703/metric_plane_outlier_review_packet/metric_plane_outlier_review_packet.json`

Contact sheet:
`runs/overlapping_court_calibration_20260703/metric_plane_outlier_review_packet/metric_plane_outlier_contact_sheet.jpg`

| Technology | Mean reviewed-line support |
| --- | ---: |
| raw OpenCV Hough | 0.8083 |
| strict HSV paint Hough | 0.0000 |
| strict HSV paint + net crop Hough | 0.0250 |
| shadow-normalized Hough | 0.7167 |
| pretrained-ML shadow-removal Hough | 0.0000, unavailable because no real TorchScript weights are configured |

Neural keypoint backbone check:
- Downloaded `MobileNet_V3_Small_Weights.IMAGENET1K_V1`.
- Adapted output shape: `[1, 45]` for 15 `(x,y,visibility)` keypoints.
- Parameter count: 1,563,981.
- No real label-data accuracy claim yet; the local reviewed real set is too
  small to train and honestly report a reliable held-out MobileNet improvement.

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
- Residual diagnostics show the target miss is concentrated: trimming the worst
  8 of 48 floor-keypoint residuals gives `0.196048 ft`, and trimming the worst
  3 keypoints within each clip gives `0.183579 ft`. This is diagnostic-only and
  does not verify CAL, but it strongly points to reviewed-label/keypoint
  definition outliers rather than a uniformly bad camera solve.
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
  to 4/4 full clips. That is real progress on PnL plumbing, but the current
  point+line weight still does not improve the inverse court residual
  (`0.382028 ft`) over the point-only distorted camera fit (`0.378245 ft`).
- A reviewed-label diagnostic weight sweep prevents PnL from silently worsening
  the report, but the robust metric-plane camera now beats both the point+line
  sweep and the pair-subset oracle on this slice.
- The new pair-subset oracle shows that perfect label-backed line selection
  would improve mean residual only to `0.366028 ft`. That means the current
  persistent line observations contain some useful signal, but line selection
  alone is not enough to reach the requested `0.2 ft` target.
- MobileNetV3-small is the right lightweight neural next step, but training
  needs a larger real court-keypoint corpus or a carefully separated synthetic
  pretrain plus real held-out gate.
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
- Do not use the reviewed net landmarks as elevated metric constraints until a
  separate review pass confirms they are labeled on the actual net top and not
  on visible net/floor intersections.
- Add a production-safe line-quality predictor before PnL. The reviewed-label
  pair oracle proves there is a small upside from better line selection, but
  its ceiling on this slice is still `0.366028 ft`, so the next selector must
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
- Train MobileNetV3-small only after there is a non-leaky train/holdout split.
- Keep all overlapping-court helpers opt-in until they beat raw Hough and pass
  CAL guard thresholds on reviewed labels.
