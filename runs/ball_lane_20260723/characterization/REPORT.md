# Ball 3D solver characterization — ball_lane_20260723_characterization

Measurement-only report for the CURRENT fail-closed physics-only ball arc solver. `VERIFIED=0` stays binding; nothing here is a promotion. Reprojection statistics are image-consistency only and are blind to metric depth error (no 3D ground truth exists yet).

Input manifest sha256: `93f55ae887c9617bdf462981af37addb775f851d4bb616de57aa525fd3ea2338` (see `manifest.json`).

## Pooled headline

| Metric | Value |
|---|---:|
| Clips measured / skipped | 3 / 1 |
| Rally frames | 2051 |
| Accepted 3D frames | 334 (16.3%) |
| Frame zero-return rate | 83.7% |
| Segments accepted / total | 12 / 30 |
| Segments returning zero accepted frames | 18 |

### Fail-closed reason taxonomy (pooled)

| Reason | Segments |
|---|---:|
| insufficient_inliers | 8 |
| max_reprojection_error_above_bound | 18 |
| outliers_exceed_inliers | 16 |
| spatial_sanity_violation | 5 |

### Anchor anatomy vs acceptance (pooled)

| Metric-anchor combo | Segments | Accepted |
|---|---:|---:|
| bounce_auto | 15 | 6 |
| bounce_auto+bounce_solver_proposed | 1 | 1 |
| bounce_auto+contact_wrist_seed | 9 | 3 |
| bounce_solver_proposed+contact_wrist_seed | 1 | 1 |
| contact_wrist_seed | 4 | 1 |

### Court-half split (frames with a solver world position)

| Half | Frames | Accepted | Accepted % |
|---|---:|---:|---:|
| y_negative | 367 | 130 | 35.4% |
| y_positive | 984 | 204 | 20.7% |

## Per-clip results

### burlington_gold_0300_low_steep_corner

| Metric | Value |
|---|---:|
| Solver status | ran |
| Accepted 3D coverage | 181 / 600 (30.2%) |
| Hidden frames | 12 |
| Fail-closed suppressed frames | 407 |
| Segments accepted / total | 5 / 13 |
| Camera side | negative_y (unverified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0-75 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/69 | None/1342.681532 | calibration_not_sha_verified | contact_wrist_seed | n/a |
| 1 | 75-132 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 10/33 | 10.678067/259.594571 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | n/a |
| 2 | 132-168 | fit | accepted (-) | 20/4 | 9.39988/72.971101 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | n/a |
| 3 | 168-218 | fit | accepted (-) | 44/7 | 13.513988/33.529323 | calibration_not_sha_verified | bounce_solver_proposed+contact_wrist_seed | pass |
| 4 | 218-266 | fit | accepted (-) | 43/0 | 6.208437/17.732113 | calibration_not_sha_verified | bounce_auto+bounce_solver_proposed | pass |
| 5 | 266-289 | fit | accepted (-) | 18/0 | 1.57074/3.38368 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | n/a |
| 6 | 289-334 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 2/39 | 1.618053/256.383406 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | fail |
| 7 | 334-347 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,spatial_sanity_violation) | 12/2 | 9.899622/53.590161 | calibration_not_sha_verified | bounce_auto | pass |
| 8 | 347-355 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 1/6 | 1.651146/780.463851 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | n/a |
| 9 | 355-423 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/64 | None/568.687683 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | pass |
| 10 | 423-497 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 18/48 | 10.847089/519.563039 | calibration_not_sha_verified | bounce_auto | pass |
| 11 | 497-519 | fit | accepted (-) | 6/0 | 5.595205/9.581789 | calibration_not_sha_verified | bounce_auto+contact_wrist_seed | n/a |
| 12 | 519-587 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 18/22 | 11.180367/83.332892 | calibration_not_sha_verified | contact_wrist_seed | n/a |

### indoor_doubles_fwuks_0500_long_mid_baseline

Skipped: `missing_artifacts` (missing: ball_track_arc_solved.json).

### outdoor_webcam_iynbd_1500_long_high_baseline

| Metric | Value |
|---|---:|
| Solver status | ran |
| Accepted 3D coverage | 98 / 1151 (8.5%) |
| Hidden frames | 678 |
| Fail-closed suppressed frames | 375 |
| Segments accepted / total | 4 / 7 |
| Camera side | negative_y (unverified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 303-321 | fit | accepted (-) | 3/16 | 12.557402/193.786885 | calibration_not_sha_verified | bounce_auto | pass |
| 1 | 321-426 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 5/77 | 12.95793/400.060151 | calibration_not_sha_verified | bounce_auto | pass |
| 2 | 426-596 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 1/123 | 12.700218/991.58344 | calibration_not_sha_verified | bounce_auto | pass |
| 3 | 596-605 | fit | accepted (-) | 2/6 | 5.82454/228.474589 | calibration_not_sha_verified | bounce_auto | pass |
| 4 | 605-707 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 5/76 | 12.764765/480.989569 | calibration_not_sha_verified | bounce_auto | pass |
| 5 | 707-746 | fit | accepted (-) | 40/0 | 3.994588/13.151291 | calibration_not_sha_verified | bounce_auto | pass |
| 6 | 746-774 | fit | accepted (-) | 28/1 | 3.062019/21.712268 | calibration_not_sha_verified | bounce_auto | pass |

### wolverine_mixed_0200_mid_steep_corner

| Metric | Value |
|---|---:|
| Solver status | ran |
| Accepted 3D coverage | 55 / 300 (18.3%) |
| Hidden frames | 10 |
| Fail-closed suppressed frames | 235 |
| Segments accepted / total | 3 / 10 |
| Camera side | negative_y (sha_verified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0-13 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 5/7 | 10.431983/1356.172899 | 30.943969/1307.575167/1348.66744 | bounce_auto | n/a |
| 1 | 13-36 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/20 | None/519.829685 | 311.276503/492.093866/585.842732 | bounce_auto+contact_wrist_seed | pass |
| 2 | 36-70 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 0/6 | None/213.278704 | 207.909161/269.622336/284.942865 | contact_wrist_seed | n/a |
| 3 | 70-84 | fit | accepted (-) | 12/0 | 2.791168/6.505183 | 2.149789/3.442162/6.505183 | contact_wrist_seed | pass |
| 4 | 84-104 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound) | 13/3 | 10.633439/91.796242 | 65.650223/83.800916/190.095521 | bounce_auto+contact_wrist_seed | n/a |
| 5 | 104-156 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 11/16 | 10.413056/193.444458 | 74.55246/168.494626/264.606695 | bounce_auto | fail |
| 6 | 156-217 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/58 | None/649.182147 | 358.058657/626.921561/715.141555 | bounce_auto | fail |
| 7 | 217-250 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 8/26 | 11.108728/73.405487 | 62.752616/96.847359/106.965636 | bounce_auto | fail |
| 8 | 250-272 | fit | accepted (-) | 22/1 | 5.804084/26.825665 | 5.087757/8.77641/26.825664 | bounce_auto | pass |
| 9 | 272-289 | fit | accepted (-) | 18/0 | 1.597502/4.33879 | 1.06099/2.414486/4.33879 | bounce_auto | pass |

---
Accepted = segment passes `arc_segment_fail_closed_v1` (min inliers 3, max reprojection 40.0 px) and the frame carries a solver world position. Physics-fill artifacts are render-only and reported separately, never counted.
