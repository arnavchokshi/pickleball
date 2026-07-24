# Dormant-flag measurement study — ball_lane_20260723_dormant_flags

Flags-only measurement of the ball solver's dormant evidence sources (joint-anchor search, conservative UKF fallback) on the three locally re-solvable eval clips. MEASUREMENT ONLY: no defaults changed, nothing promoted, `VERIFIED=0` stays binding. Owner decision 2026-07-23: default-on requires future T1 metric proof.

## Honest caveats (read first)

- Measurement only. VERIFIED=0 stays binding; nothing here is a promotion or a default change.
- Reprojection residuals and fail-closed acceptance are image-consistency measurements only; they are blind to metric depth error. T1 metric-3D ground truth is required before any acceptance/promotion claim.
- UKF fallback outputs are a render-only sidecar: every sample is source=physics_interpolated, band=physics_predicted, trust_band=low_confidence by design. The characterization harness never reads the sidecar (it is not in discover_clip_inputs), so UKF samples cannot leak into accepted statistics; they are reported separately below and must never be counted as accepted/measured 3D.
- Joint-anchor search injects candidate_hypothesis anchors (marks_measured=false) through the unchanged production event selector and fail-closed gates; anchor adoption is not evidence of depth accuracy.
- The solver has a 5.0s per-segment wall-clock safety budget (SEGMENT_WALL_CLOCK_BUDGET_S); segments that exceed it abstain with typed segment_budget_exceeded. This budget is machine/load dependent, so segment counts near the budget boundary carry run-to-run noise; the determinism probe below bounds this for the burlington baseline.
- The 2026-07-05 archived solve predates several solver changes (LOO per-holdout BVP refit, BVP span protection v2, per-segment wall-clock budget, fail-closed emission changes); the baseline-vs-archived delta is code drift, not an input mismatch (input shas are pinned and verified).
- contact_windows/skeleton3d/rally_spans shas were not recorded by the archived 2026-07-05 chain manifest; the ball_f1 copies used here are the best available originals and their shas are pinned in this report's manifest (archived_sha_recorded=false).

## 1. Baseline vs archived 2026-07-05 solve (reproduction verdict)

**Verdict: `not_reproduced_code_drift_since_20260705`.** Input shas are pinned to the archived chain manifests and verified before solving; any delta is solver code drift between 2026-07-05 and this worktree, and it applies equally to every variant (all variants share the same current code).

| Clip | Status arch->base | Segments arch->base | Accepted segs arch->base | Accepted frames arch->base | Coverage arch->base |
|---|---|---|---|---|---|
| burlington_gold_0300_low_steep_corner | ran -> degraded | 13 -> 18 | 5 -> 11 | 181 -> 256 | 30.2% -> 42.7% |
| outdoor_webcam_iynbd_1500_long_high_baseline | ran -> degraded | 7 -> 8 | 4 -> 7 | 98 -> 183 | 8.5% -> 15.9% |
| wolverine_mixed_0200_mid_steep_corner | ran -> degraded | 10 -> 9 | 3 -> 3 | 55 -> 53 | 18.3% -> 17.7% |

Per-clip taxonomy delta (baseline minus archived):

- burlington_gold_0300_low_steep_corner: reasons {insufficient_inliers -1, max_reprojection_error_above_bound -2, outliers_exceed_inliers -4}; statuses {blocked:segment_budget_exceeded +2, fit +4, fit_bvp_fallback -3, fit_weak +2}
- outdoor_webcam_iynbd_1500_long_high_baseline: reasons {insufficient_inliers -1, max_reprojection_error_above_bound -2, outliers_exceed_inliers -2}; statuses {blocked:segment_budget_exceeded +2, fit_bvp_fallback -1}
- wolverine_mixed_0200_mid_steep_corner: reasons {insufficient_inliers -2, max_reprojection_error_above_bound -3, outliers_exceed_inliers -3, spatial_sanity_violation +2}; statuses {blocked:segment_budget_exceeded +1, fit -2}

## 2. Determinism probe (wall-clock budget noise bound)

Second sequential burlington baseline solve with identical pinned inputs: raw bytes **NOT byte-identical** (`45a65c2df0a0` vs `1cb98bd25830`); after stripping only `segments[*].degradation.elapsed_s` (wall-clock provenance in budget-abstained segments): **identical**. Same pinned inputs + pinned generated_at, sequential runs on the same machine. Raw bytes may differ only in the wall-clock elapsed_s recorded inside budget-abstained segments; normalized_identical=true means every measurement-relevant field (segments, verdicts, frames, anchors) was identical across reruns. If normalized_identical were false, variant deltas would carry segment-level budget noise.

## 3. Per-flag headline deltas vs baseline

| Variant | Clip | Segs | Accepted segs | Accepted frames | Coverage delta | Fail-closed reason deltas | Anchor anatomy deltas |
|---|---|---:|---:|---:|---:|---|---|
| +joint_anchor_search | burlington_gold_0300_low_steep_corner | -2 | -1 | +0 | +0.000000 | insufficient_inliers -1, max_reprojection_error_above_bound -1, outliers_exceed_inliers -1 | double_metric_anchor -1, single_sided_metric_anchor -1 |
| +joint_anchor_search | wolverine_mixed_0200_mid_steep_corner | +2 | +1 | +9 | +0.030000 | insufficient_inliers +1, max_reprojection_error_above_bound +1, outliers_exceed_inliers +1, spatial_sanity_violation +1 | double_metric_anchor +2 |
| +joint_anchor_search | outdoor_webcam_iynbd_1500_long_high_baseline | +0 | +0 | +0 | +0.000000 | - | - |
| +ukf_fallback | burlington_gold_0300_low_steep_corner | +0 | +0 | +0 | +0.000000 | - | - |
| +ukf_fallback | wolverine_mixed_0200_mid_steep_corner | +0 | +0 | +0 | +0.000000 | - | - |
| +ukf_fallback | outdoor_webcam_iynbd_1500_long_high_baseline | +0 | +0 | +0 | +0.000000 | - | - |
| +both | burlington_gold_0300_low_steep_corner | -2 | -1 | +0 | +0.000000 | insufficient_inliers -1, max_reprojection_error_above_bound -1, outliers_exceed_inliers -1 | double_metric_anchor -1, single_sided_metric_anchor -1 |
| +both | wolverine_mixed_0200_mid_steep_corner | +2 | +1 | +9 | +0.030000 | insufficient_inliers +1, max_reprojection_error_above_bound +1, outliers_exceed_inliers +1, spatial_sanity_violation +1 | double_metric_anchor +2 |
| +both | outdoor_webcam_iynbd_1500_long_high_baseline | +0 | +0 | +0 | +0.000000 | - | - |

Additive-flag identity of the arc-solved artifact (raw sha / normalized sha, where normalized strips only the wall-clock `degradation.elapsed_s` provenance):

- burlington_gold_0300_low_steep_corner: both_equals_joint_anchor_search: raw=False normalized=True, joint_anchor_search_equals_baseline: raw=False normalized=False, ukf_equals_baseline: raw=False normalized=True
- outdoor_webcam_iynbd_1500_long_high_baseline: both_equals_joint_anchor_search: raw=False normalized=True, joint_anchor_search_equals_baseline: raw=False normalized=False, ukf_equals_baseline: raw=False normalized=True
- wolverine_mixed_0200_mid_steep_corner: both_equals_joint_anchor_search: raw=False normalized=True, joint_anchor_search_equals_baseline: raw=False normalized=False, ukf_equals_baseline: raw=False normalized=True

## 4. Variant detail per clip

### archived_20260705

| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |
|---|---|---|---|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | ran | 13 (5) | 181 / 600 | 30.2% | 407 | 12 |
| wolverine_mixed_0200_mid_steep_corner | ran | 10 (3) | 55 / 300 | 18.3% | 235 | 10 |
| outdoor_webcam_iynbd_1500_long_high_baseline | ran | 7 (4) | 98 / 1151 | 8.5% | 375 | 678 |

Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):

- burlington_gold_0300_low_steep_corner: seg2 [132-168] fit rmse/max 9.39988/72.971101 px, raw p50/p90/max 9.875852/60.609101/81.535791; seg3 [168-218] fit rmse/max 13.513988/33.529323 px, raw p50/p90/max 15.551746/17.944289/31.685296; seg4 [218-266] fit rmse/max 6.208437/17.732113 px, raw p50/p90/max 4.51967/8.325968/17.732113; seg5 [266-289] fit rmse/max 1.57074/3.38368 px, raw p50/p90/max 1.232658/2.211014/3.38368; seg11 [497-519] fit rmse/max 5.595205/9.581789 px, raw p50/p90/max 4.665039/12.222434/16.183402
- wolverine_mixed_0200_mid_steep_corner: seg3 [70-84] fit rmse/max 2.791168/6.505183 px, raw p50/p90/max 2.149789/3.442162/6.505183; seg8 [250-272] fit rmse/max 5.804084/26.825665 px, raw p50/p90/max 5.087757/8.77641/26.825664; seg9 [272-289] fit rmse/max 1.597502/4.33879 px, raw p50/p90/max 1.06099/2.414486/4.33879
- outdoor_webcam_iynbd_1500_long_high_baseline: seg0 [303-321] fit rmse/max 12.557402/193.786885 px, raw p50/p90/max 56.846483/154.180503/193.786885; seg3 [596-605] fit rmse/max 5.82454/228.474589 px, raw p50/p90/max 173.244151/210.11931/228.474589; seg5 [707-746] fit rmse/max 3.994588/13.151291 px, raw p50/p90/max 2.823192/5.635824/6.821948; seg6 [746-774] fit rmse/max 3.062019/21.712268 px, raw p50/p90/max 2.634698/5.459354/21.712268

### baseline

| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |
|---|---|---|---|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | degraded | 18 (11) | 256 / 600 | 42.7% | 172 | 172 |
| wolverine_mixed_0200_mid_steep_corner | degraded | 9 (3) | 53 / 300 | 17.7% | 177 | 70 |
| outdoor_webcam_iynbd_1500_long_high_baseline | degraded | 8 (7) | 183 / 1151 | 15.9% | 20 | 948 |

Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):

- burlington_gold_0300_low_steep_corner: seg0 [0-75] fit rmse/max None/None px, raw n/a; seg3 [107-132] fit rmse/max 4.079306/62.147848 px, raw p50/p90/max 3.753827/7.798298/49.47478; seg4 [132-139] fit rmse/max 1.554418/1.971345 px, raw p50/p90/max 1.721813/1.956514/1.971345; seg5 [139-151] fit rmse/max 0.918769/1.759055 px, raw p50/p90/max 0.74983/1.442724/1.759055; seg6 [151-168] fit rmse/max 2.654028/3.747957 px, raw p50/p90/max 3.579107/5.006793/5.846016; seg7 [168-218] fit rmse/max 13.513988/33.529323 px, raw p50/p90/max 15.551746/17.944289/31.685296; seg8 [218-266] fit rmse/max 6.208437/17.732113 px, raw p50/p90/max 4.51967/8.325968/17.732113; seg9 [266-289] fit rmse/max 1.57074/3.38368 px, raw p50/p90/max 1.232658/2.211014/3.38368; seg10 [289-334] fit rmse/max None/None px, raw n/a; seg14 [423-447] fit rmse/max 5.300534/510.792508 px, raw p50/p90/max 2.478403/11.840013/510.792508; seg15 [447-497] fit rmse/max 1.912784/9.596773 px, raw p50/p90/max 1.196083/1.992466/3.035397
- wolverine_mixed_0200_mid_steep_corner: seg3 [70-104] fit rmse/max 6.96772/31.508552 px, raw p50/p90/max 6.831874/19.118491/31.508552; seg5 [156-217] fit rmse/max None/None px, raw n/a; seg8 [272-289] fit rmse/max 1.597502/4.33879 px, raw p50/p90/max 4.003622/5.744151/6.055632
- outdoor_webcam_iynbd_1500_long_high_baseline: seg1 [321-388] fit rmse/max 11.313115/34.16809 px, raw p50/p90/max 31.243726/53.131869/60.523847; seg2 [388-426] fit rmse/max 4.416748/68.221029 px, raw p50/p90/max 2.461134/33.246728/68.221029; seg3 [426-596] fit rmse/max None/None px, raw n/a; seg4 [596-605] fit rmse/max 5.82454/228.474589 px, raw p50/p90/max 173.244151/210.11931/228.474589; seg5 [605-707] fit rmse/max None/None px, raw n/a; seg6 [707-746] fit rmse/max 3.994588/13.151291 px, raw p50/p90/max 2.823192/5.635824/6.821948; seg7 [746-774] fit rmse/max 3.062019/21.712268 px, raw p50/p90/max 2.634698/5.459354/21.712268

### joint_anchor_search

| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |
|---|---|---|---|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | degraded | 16 (10) | 256 / 600 | 42.7% | 155 | 189 |
| wolverine_mixed_0200_mid_steep_corner | degraded | 11 (4) | 62 / 300 | 20.7% | 167 | 71 |
| outdoor_webcam_iynbd_1500_long_high_baseline | degraded | 8 (7) | 183 / 1151 | 15.9% | 20 | 948 |

Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):

- burlington_gold_0300_low_steep_corner: seg1 [107-132] fit rmse/max 4.079306/62.147848 px, raw p50/p90/max 3.753827/7.798298/49.47478; seg2 [132-139] fit rmse/max 1.554418/1.971345 px, raw p50/p90/max 1.721813/1.956514/1.971345; seg3 [139-151] fit rmse/max 0.918769/1.759055 px, raw p50/p90/max 0.74983/1.442724/1.759055; seg4 [151-168] fit rmse/max 2.654028/3.747957 px, raw p50/p90/max 3.579107/5.006793/5.846016; seg5 [168-218] fit rmse/max 13.513988/33.529323 px, raw p50/p90/max 15.551746/17.944289/31.685296; seg6 [218-266] fit rmse/max 6.208437/17.732113 px, raw p50/p90/max 4.51967/8.325968/17.732113; seg7 [266-289] fit rmse/max 1.57074/3.38368 px, raw p50/p90/max 1.232658/2.211014/3.38368; seg8 [289-334] fit rmse/max None/None px, raw n/a; seg12 [423-447] fit rmse/max 5.300534/510.792508 px, raw p50/p90/max 2.478403/11.840013/510.792508; seg13 [447-497] fit rmse/max 1.912784/9.596773 px, raw p50/p90/max 1.196083/1.992466/3.035397
- wolverine_mixed_0200_mid_steep_corner: seg1 [13-21] fit rmse/max 1.570114/2.317295 px, raw p50/p90/max 1.379504/2.256031/2.317295; seg5 [70-104] fit rmse/max 6.96772/31.508552 px, raw p50/p90/max 6.831874/19.118491/31.508552; seg7 [156-217] fit rmse/max None/None px, raw n/a; seg10 [272-289] fit rmse/max 1.597502/4.33879 px, raw p50/p90/max 4.003622/5.744151/6.055632
- outdoor_webcam_iynbd_1500_long_high_baseline: seg1 [321-388] fit rmse/max 11.313115/34.16809 px, raw p50/p90/max 31.243726/53.131869/60.523847; seg2 [388-426] fit rmse/max 4.416748/68.221029 px, raw p50/p90/max 2.461134/33.246728/68.221029; seg3 [426-596] fit rmse/max None/None px, raw n/a; seg4 [596-605] fit rmse/max 5.82454/228.474589 px, raw p50/p90/max 173.244151/210.11931/228.474589; seg5 [605-707] fit rmse/max None/None px, raw n/a; seg6 [707-746] fit rmse/max 3.994588/13.151291 px, raw p50/p90/max 2.823192/5.635824/6.821948; seg7 [746-774] fit rmse/max 3.062019/21.712268 px, raw p50/p90/max 2.634698/5.459354/21.712268

### ukf_fallback

| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |
|---|---|---|---|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | degraded | 18 (11) | 256 / 600 | 42.7% | 172 | 172 |
| wolverine_mixed_0200_mid_steep_corner | degraded | 9 (3) | 53 / 300 | 17.7% | 177 | 70 |
| outdoor_webcam_iynbd_1500_long_high_baseline | degraded | 8 (7) | 183 / 1151 | 15.9% | 20 | 948 |

Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):

- burlington_gold_0300_low_steep_corner: seg0 [0-75] fit rmse/max None/None px, raw n/a; seg3 [107-132] fit rmse/max 4.079306/62.147848 px, raw p50/p90/max 3.753827/7.798298/49.47478; seg4 [132-139] fit rmse/max 1.554418/1.971345 px, raw p50/p90/max 1.721813/1.956514/1.971345; seg5 [139-151] fit rmse/max 0.918769/1.759055 px, raw p50/p90/max 0.74983/1.442724/1.759055; seg6 [151-168] fit rmse/max 2.654028/3.747957 px, raw p50/p90/max 3.579107/5.006793/5.846016; seg7 [168-218] fit rmse/max 13.513988/33.529323 px, raw p50/p90/max 15.551746/17.944289/31.685296; seg8 [218-266] fit rmse/max 6.208437/17.732113 px, raw p50/p90/max 4.51967/8.325968/17.732113; seg9 [266-289] fit rmse/max 1.57074/3.38368 px, raw p50/p90/max 1.232658/2.211014/3.38368; seg10 [289-334] fit rmse/max None/None px, raw n/a; seg14 [423-447] fit rmse/max 5.300534/510.792508 px, raw p50/p90/max 2.478403/11.840013/510.792508; seg15 [447-497] fit rmse/max 1.912784/9.596773 px, raw p50/p90/max 1.196083/1.992466/3.035397
- wolverine_mixed_0200_mid_steep_corner: seg3 [70-104] fit rmse/max 6.96772/31.508552 px, raw p50/p90/max 6.831874/19.118491/31.508552; seg5 [156-217] fit rmse/max None/None px, raw n/a; seg8 [272-289] fit rmse/max 1.597502/4.33879 px, raw p50/p90/max 4.003622/5.744151/6.055632
- outdoor_webcam_iynbd_1500_long_high_baseline: seg1 [321-388] fit rmse/max 11.313115/34.16809 px, raw p50/p90/max 31.243726/53.131869/60.523847; seg2 [388-426] fit rmse/max 4.416748/68.221029 px, raw p50/p90/max 2.461134/33.246728/68.221029; seg3 [426-596] fit rmse/max None/None px, raw n/a; seg4 [596-605] fit rmse/max 5.82454/228.474589 px, raw p50/p90/max 173.244151/210.11931/228.474589; seg5 [605-707] fit rmse/max None/None px, raw n/a; seg6 [707-746] fit rmse/max 3.994588/13.151291 px, raw p50/p90/max 2.823192/5.635824/6.821948; seg7 [746-774] fit rmse/max 3.062019/21.712268 px, raw p50/p90/max 2.634698/5.459354/21.712268

### both

| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |
|---|---|---|---|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | degraded | 16 (10) | 256 / 600 | 42.7% | 155 | 189 |
| wolverine_mixed_0200_mid_steep_corner | degraded | 11 (4) | 62 / 300 | 20.7% | 167 | 71 |
| outdoor_webcam_iynbd_1500_long_high_baseline | degraded | 8 (7) | 183 / 1151 | 15.9% | 20 | 948 |

Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):

- burlington_gold_0300_low_steep_corner: seg1 [107-132] fit rmse/max 4.079306/62.147848 px, raw p50/p90/max 3.753827/7.798298/49.47478; seg2 [132-139] fit rmse/max 1.554418/1.971345 px, raw p50/p90/max 1.721813/1.956514/1.971345; seg3 [139-151] fit rmse/max 0.918769/1.759055 px, raw p50/p90/max 0.74983/1.442724/1.759055; seg4 [151-168] fit rmse/max 2.654028/3.747957 px, raw p50/p90/max 3.579107/5.006793/5.846016; seg5 [168-218] fit rmse/max 13.513988/33.529323 px, raw p50/p90/max 15.551746/17.944289/31.685296; seg6 [218-266] fit rmse/max 6.208437/17.732113 px, raw p50/p90/max 4.51967/8.325968/17.732113; seg7 [266-289] fit rmse/max 1.57074/3.38368 px, raw p50/p90/max 1.232658/2.211014/3.38368; seg8 [289-334] fit rmse/max None/None px, raw n/a; seg12 [423-447] fit rmse/max 5.300534/510.792508 px, raw p50/p90/max 2.478403/11.840013/510.792508; seg13 [447-497] fit rmse/max 1.912784/9.596773 px, raw p50/p90/max 1.196083/1.992466/3.035397
- wolverine_mixed_0200_mid_steep_corner: seg1 [13-21] fit rmse/max 1.570114/2.317295 px, raw p50/p90/max 1.379504/2.256031/2.317295; seg5 [70-104] fit rmse/max 6.96772/31.508552 px, raw p50/p90/max 6.831874/19.118491/31.508552; seg7 [156-217] fit rmse/max None/None px, raw n/a; seg10 [272-289] fit rmse/max 1.597502/4.33879 px, raw p50/p90/max 4.003622/5.744151/6.055632
- outdoor_webcam_iynbd_1500_long_high_baseline: seg1 [321-388] fit rmse/max 11.313115/34.16809 px, raw p50/p90/max 31.243726/53.131869/60.523847; seg2 [388-426] fit rmse/max 4.416748/68.221029 px, raw p50/p90/max 2.461134/33.246728/68.221029; seg3 [426-596] fit rmse/max None/None px, raw n/a; seg4 [596-605] fit rmse/max 5.82454/228.474589 px, raw p50/p90/max 173.244151/210.11931/228.474589; seg5 [605-707] fit rmse/max None/None px, raw n/a; seg6 [707-746] fit rmse/max 3.994588/13.151291 px, raw p50/p90/max 2.823192/5.635824/6.821948; seg7 [746-774] fit rmse/max 3.062019/21.712268 px, raw p50/p90/max 2.634698/5.459354/21.712268

## 5. Joint-anchor search sidecar (candidate-only, never measured)

| Clip | Variant | Fallback wins searched | Refused | Hypotheses | Submitted (rank-1) | Chosen by selector | Hypothesis anchors used (segments) |
|---|---|---:|---:|---:|---:|---:|---|
| burlington_gold_0300_low_steep_corner | +both | 4 | 1 | 9 | 4 | 1 | - |
| burlington_gold_0300_low_steep_corner | +joint_anchor_search | 4 | 1 | 9 | 4 | 1 | - |
| wolverine_mixed_0200_mid_steep_corner | +both | 6 | 1 | 23 | 6 | 1 | seg2(rejected_fail_closed), seg3(rejected_fail_closed) |
| wolverine_mixed_0200_mid_steep_corner | +joint_anchor_search | 6 | 1 | 23 | 6 | 1 | seg2(rejected_fail_closed), seg3(rejected_fail_closed) |
| outdoor_webcam_iynbd_1500_long_high_baseline | +both | 2 | 0 | 8 | 2 | 0 | - |
| outdoor_webcam_iynbd_1500_long_high_baseline | +joint_anchor_search | 2 | 0 | 8 | 2 | 0 | - |

Segments carrying a hypothesis anchor that were ACCEPTED: none. Where acceptance changed vs baseline while no accepted segment carries a hypothesis anchor, the mechanism is re-segmentation around the injected candidate boundary, not direct anchoring of an accepted arc. A selector-chosen anchor can also end up in no final segment's anchors_used when subsequent refinement drops or merges its window.

## 6. UKF fallback sidecar (reported separately, NEVER counted)

| Clip | Variant | Attempted gaps | Recovered gaps | Recovered samples | Refused gaps | All samples physics_interpolated low-confidence |
|---|---|---:|---:|---:|---:|---|
| burlington_gold_0300_low_steep_corner | +both | 6 | 0 | 0 | 6 | True |
| burlington_gold_0300_low_steep_corner | +ukf_fallback | 7 | 0 | 0 | 7 | True |
| wolverine_mixed_0200_mid_steep_corner | +both | 7 | 0 | 0 | 7 | True |
| wolverine_mixed_0200_mid_steep_corner | +ukf_fallback | 6 | 0 | 0 | 6 | True |
| outdoor_webcam_iynbd_1500_long_high_baseline | +both | 1 | 0 | 0 | 1 | True |
| outdoor_webcam_iynbd_1500_long_high_baseline | +ukf_fallback | 1 | 0 | 0 | 1 | True |

Bucketing verification: the harness never reads `ball_ukf_fallback.json` (not part of `discover_clip_inputs`), and the arc-solved artifact with the flag on is identical to baseline modulo the wall-clock `degradation.elapsed_s` provenance (normalized shas equal, section 3), so UKF samples cannot appear in any accepted statistic.

UKF refusal reasons:

- burlington_gold_0300_low_steep_corner (+both): gap_above_short_gap_ceiling +3, no_adjacent_accepted_fit_seed +3
- burlington_gold_0300_low_steep_corner (+ukf_fallback): gap_above_short_gap_ceiling +3, no_adjacent_accepted_fit_seed +4
- wolverine_mixed_0200_mid_steep_corner (+both): contact_proposal_inside_gap +1, gap_above_short_gap_ceiling +3, no_adjacent_accepted_fit_seed +3
- wolverine_mixed_0200_mid_steep_corner (+ukf_fallback): gap_above_short_gap_ceiling +2, no_adjacent_accepted_fit_seed +4
- outdoor_webcam_iynbd_1500_long_high_baseline (+both): no_adjacent_accepted_fit_seed +1
- outdoor_webcam_iynbd_1500_long_high_baseline (+ukf_fallback): no_adjacent_accepted_fit_seed +1

## 7. Skipped clips

- `indoor_doubles_fwuks_0500_long_mid_baseline`: no local re-solve inputs: no ball_track.json (tracking-stage output) exists for this clip on this machine and no solved artifacts exist anywhere locally (baseline lane report already listed it skipped: missing ball_track_arc_solved.json)

---
Accepted = segment passes `arc_segment_fail_closed_v1` and the frame carries a solver world position (existing harness definition, unchanged). Nothing in this study alters defaults, gates, or trust bands. VERIFIED=0.
