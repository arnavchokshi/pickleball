# BALL LoSO (Leave-One-Source-Out) Validation Report

Status: `TESTED-ON-REAL-DATA` | objective_result: `PASS`

BALL is not verified by this report. This is a scoring/analysis artifact over already-materialized predictions and already-reviewed labels; it runs no inference and trains nothing.

- CVAT root: `runs/lanes/w7_ballingest3_20260709/reviewed_corpus`
- Internal-val-only (legal LoSO fold) clip ids: `['burlington_gold_0300_low_steep_corner', 'wolverine_mixed_0200_mid_steep_corner']`
- Strict-holdout clip ids never scored by this script: `['outdoor_webcam_iynbd_1500_long_high_baseline', 'indoor_doubles_fwuks_0500_long_mid_baseline']`

## Candidates

| Candidate | Folds | Metric | Pooled/Mixed | LoSO-mean | LoSO-worst | Gap (pooled-mean) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| official_tennis_control | 40 | F1@20 | 0.5956 | 0.6029 | 0.0000 | -0.0072 |
| official_tennis_control | 40 | Recall@20 | 0.5826 | 0.5913 | 0.0000 | -0.0087 |
| official_tennis_control | 40 | Precision@20 | 0.6093 | 0.6328 | 0.0000 | -0.0235 |
| official_tennis_control | 40 | HiddenFP | 0.5946 | 0.6024 | 1.0000 | -0.0077 |

## Held-out comparisons (literals supplied via --heldout-metric)

_No held-out comparisons: fewer than 2 candidates supplied --heldout-metric for the same clip+metric._
