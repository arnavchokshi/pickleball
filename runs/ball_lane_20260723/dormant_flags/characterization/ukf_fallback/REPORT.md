# Ball 3D solver characterization — ball_lane_20260723_dormant_flags_ukf_fallback

Measurement-only report for the CURRENT fail-closed physics-only ball arc solver. `VERIFIED=0` stays binding; nothing here is a promotion. Reprojection statistics are image-consistency only and are blind to metric depth error (no 3D ground truth exists yet).

Input manifest sha256: `4d3a86cfaeb3f0f035424932ca0dcbc765c1b3a168ba9faa7117939e7b6735a3` (see `manifest.json`).

## Pooled headline

| Metric | Value |
|---|---:|
| Clips measured / skipped | 3 / 0 |
| Rally frames | 2051 |
| Accepted 3D frames | 492 (24.0%) |
| Frame zero-return rate | 76.0% |
| Segments accepted / total | 21 / 35 |
| Segments returning zero accepted frames | 19 |

### Fail-closed reason taxonomy (pooled)

| Reason | Segments |
|---|---:|
| insufficient_inliers | 4 |
| max_reprojection_error_above_bound | 11 |
| outliers_exceed_inliers | 7 |
| spatial_sanity_violation | 7 |

### Anchor anatomy vs acceptance (pooled)

| Metric-anchor combo | Segments | Accepted |
|---|---:|---:|
| bounce_auto | 13 | 7 |
| bounce_auto+bounce_solver_proposed | 3 | 3 |
| bounce_auto+contact_wrist_seed | 11 | 7 |
| bounce_auto+other_weak_ray_endpoint | 1 | 0 |
| bounce_solver_proposed+contact_wrist_seed | 4 | 3 |
| contact_wrist_seed | 2 | 1 |
| contact_wrist_seed+other_weak_ray_endpoint | 1 | 0 |

### Court-half split (frames with a solver world position)

| Half | Frames | Accepted | Accepted % |
|---|---:|---:|---:|
| y_negative | 264 | 135 | 51.1% |
| y_positive | 597 | 357 | 59.8% |

## Per-clip results

### burlington_gold_0300_low_steep_corner

| Metric | Value |
|---|---:|
| Solver status | degraded |
| Accepted 3D coverage | 256 / 600 (42.7%) |
| Hidden frames | 172 |
| Fail-closed suppressed frames | 172 |
| Segments accepted / total | 11 / 18 |
| Camera side | negative_y (sha_verified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0-75 | blocked:segment_budget_exceeded | accepted (-) | 0/0 | None/None | n/a | contact_wrist_seed | n/a |
| 1 | 75-91 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 2/14 | 10.678067/259.594571 | 110.732107/223.27316/263.989167 | bounce_auto+contact_wrist_seed | n/a |
| 2 | 92-107 | fit_bvp_fallback | rejected_fail_closed (spatial_sanity_violation) | 13/0 | 3.944336/6.65764 | 3.820186/6.414351/18.138067 | bounce_solver_proposed+contact_wrist_seed | n/a |
| 3 | 107-132 | fit | accepted (-) | 12/2 | 4.079306/62.147848 | 3.753827/7.798298/49.47478 | bounce_auto+contact_wrist_seed | n/a |
| 4 | 132-139 | fit | accepted (-) | 8/0 | 1.554418/1.971345 | 1.721813/1.956514/1.971345 | bounce_auto+contact_wrist_seed | n/a |
| 5 | 139-151 | fit | accepted (-) | 12/0 | 0.918769/1.759055 | 0.74983/1.442724/1.759055 | bounce_solver_proposed+contact_wrist_seed | n/a |
| 6 | 151-168 | fit | accepted (-) | 5/0 | 2.654028/3.747957 | 3.579107/5.006793/5.846016 | bounce_solver_proposed+contact_wrist_seed | n/a |
| 7 | 168-218 | fit | accepted (-) | 44/7 | 13.513988/33.529323 | 15.551746/17.944289/31.685296 | bounce_solver_proposed+contact_wrist_seed | pass |
| 8 | 218-266 | fit | accepted (-) | 43/0 | 6.208437/17.732113 | 4.51967/8.325968/17.732113 | bounce_auto+bounce_solver_proposed | pass |
| 9 | 266-289 | fit | accepted (-) | 18/0 | 1.57074/3.38368 | 1.232658/2.211014/3.38368 | bounce_auto+contact_wrist_seed | n/a |
| 10 | 289-334 | blocked:segment_budget_exceeded | accepted (-) | 0/0 | None/None | n/a | bounce_auto+contact_wrist_seed | pass |
| 11 | 334-347 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,spatial_sanity_violation) | 12/2 | 9.899622/53.590161 | 22.327524/32.490569/33.938464 | bounce_auto | pass |
| 12 | 347-355 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 1/6 | 1.651146/780.463851 | 453.220496/898.92654/962.227384 | bounce_auto+contact_wrist_seed | n/a |
| 13 | 355-423 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/64 | None/568.687683 | 581.792844/603.006938/629.947805 | bounce_auto+contact_wrist_seed | pass |
| 14 | 423-447 | fit | accepted (-) | 15/1 | 5.300534/510.792508 | 2.478403/11.840013/510.792508 | bounce_auto+contact_wrist_seed | n/a |
| 15 | 447-497 | fit | accepted (-) | 50/0 | 1.912784/9.596773 | 1.196083/1.992466/3.035397 | bounce_auto+contact_wrist_seed | n/a |
| 16 | 497-543 | fit_weak | rejected_fail_closed (max_reprojection_error_above_bound) | 25/1 | 21.660646/48.381242 | 21.481857/219.548327/276.910942 | bounce_auto+other_weak_ray_endpoint | n/a |
| 17 | 289-332 | fit_weak | rejected_fail_closed (max_reprojection_error_above_bound) | 38/1 | 2.158554/249.40644 | 1.127966/1.706311/2.258819 | contact_wrist_seed+other_weak_ray_endpoint | n/a |

### outdoor_webcam_iynbd_1500_long_high_baseline

| Metric | Value |
|---|---:|
| Solver status | degraded |
| Accepted 3D coverage | 183 / 1151 (15.9%) |
| Hidden frames | 948 |
| Fail-closed suppressed frames | 20 |
| Segments accepted / total | 7 / 8 |
| Camera side | negative_y (sha_verified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 303-321 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 3/16 | 12.557402/193.786885 | 75.179796/192.77226/234.691879 | bounce_auto | pass |
| 1 | 321-388 | fit_bvp_fallback | accepted (-) | 38/17 | 11.313115/34.16809 | 31.243726/53.131869/60.523847 | bounce_auto+bounce_solver_proposed | pass |
| 2 | 388-426 | fit | accepted (-) | 24/4 | 4.416748/68.221029 | 2.461134/33.246728/68.221029 | bounce_auto+bounce_solver_proposed | pass |
| 3 | 426-596 | blocked:segment_budget_exceeded | accepted (-) | 0/0 | None/None | n/a | bounce_auto | not_evaluated |
| 4 | 596-605 | fit | accepted (-) | 2/6 | 5.82454/228.474589 | 173.244151/210.11931/228.474589 | bounce_auto | pass |
| 5 | 605-707 | blocked:segment_budget_exceeded | accepted (-) | 0/0 | None/None | n/a | bounce_auto | not_evaluated |
| 6 | 707-746 | fit | accepted (-) | 40/0 | 3.994588/13.151291 | 2.823192/5.635824/6.821948 | bounce_auto | pass |
| 7 | 746-774 | fit | accepted (-) | 28/1 | 3.062019/21.712268 | 2.634698/5.459354/21.712268 | bounce_auto | pass |

### wolverine_mixed_0200_mid_steep_corner

| Metric | Value |
|---|---:|
| Solver status | degraded |
| Accepted 3D coverage | 53 / 300 (17.7%) |
| Hidden frames | 70 |
| Fail-closed suppressed frames | 177 |
| Segments accepted / total | 3 / 9 |
| Camera side | negative_y (sha_verified_calibration) |

| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0-13 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers,spatial_sanity_violation) | 5/7 | 10.431983/1356.172899 | 30.943969/1307.575167/1348.66744 | bounce_auto | n/a |
| 1 | 13-36 | fit_bvp_fallback | rejected_fail_closed (insufficient_inliers,max_reprojection_error_above_bound,outliers_exceed_inliers) | 0/20 | None/519.829685 | 311.276503/492.093866/585.842732 | bounce_auto+contact_wrist_seed | pass |
| 2 | 36-70 | fit_bvp_fallback | rejected_fail_closed (spatial_sanity_violation) | 6/0 | 4.841829/8.434995 | 3.822532/6.794268/8.434995 | contact_wrist_seed | n/a |
| 3 | 70-104 | fit | accepted (-) | 24/4 | 6.96772/31.508552 | 6.831874/19.118491/31.508552 | bounce_auto+contact_wrist_seed | n/a |
| 4 | 104-156 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,spatial_sanity_violation) | 17/10 | 10.399882/204.468403 | 13.804294/91.691422/204.468403 | bounce_auto | fail |
| 5 | 156-217 | blocked:segment_budget_exceeded | accepted (-) | 0/0 | None/None | n/a | bounce_auto | not_evaluated |
| 6 | 217-250 | fit_bvp_fallback | rejected_fail_closed (max_reprojection_error_above_bound,outliers_exceed_inliers) | 8/26 | 11.108728/73.405487 | 62.752616/96.847359/106.965636 | bounce_auto | fail |
| 7 | 250-272 | fit_bvp_fallback | rejected_fail_closed (spatial_sanity_violation) | 22/1 | 5.804084/26.825665 | 17.110643/21.389277/22.112596 | bounce_auto | pass |
| 8 | 272-289 | fit_bvp_fallback | accepted (-) | 18/0 | 1.597502/4.33879 | 4.003622/5.744151/6.055632 | bounce_auto | pass |

---
Accepted = segment passes `arc_segment_fail_closed_v1` (min inliers 3, max reprojection 40.0 px) and the frame carries a solver world position. Physics-fill artifacts are render-only and reported separately, never counted.
