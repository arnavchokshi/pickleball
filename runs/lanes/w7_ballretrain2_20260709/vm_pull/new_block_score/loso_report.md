# BALL LoSO (Leave-One-Source-Out) Validation Report

Status: `TESTED-ON-REAL-DATA` | objective_result: `PASS`

BALL is not verified by this report. This is a scoring/analysis artifact over already-materialized predictions and already-reviewed labels; it runs no inference and trains nothing.

- CVAT root: `runs/lanes/w7_ballingest4_20260709/reviewed_corpus`
- Internal-val-only (legal LoSO fold) clip ids: `['burlington_gold_0300_low_steep_corner', 'wolverine_mixed_0200_mid_steep_corner']`
- Strict-holdout clip ids never scored by this script: `['outdoor_webcam_iynbd_1500_long_high_baseline', 'indoor_doubles_fwuks_0500_long_mid_baseline']`

## Candidates

| Candidate | Folds | Metric | Pooled/Mixed | LoSO-mean | LoSO-worst | Gap (pooled-mean) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| E3k_matched_seed_official_aug | 40 | F1@20 | 0.7485 | 0.7708 | 0.1250 | -0.0222 |
| E3k_matched_seed_official_aug | 40 | Recall@20 | 0.7739 | 0.7762 | 0.1250 | -0.0023 |
| E3k_matched_seed_official_aug | 40 | Precision@20 | 0.7248 | 0.7730 | 0.1250 | -0.0481 |
| E3k_matched_seed_official_aug | 40 | HiddenFP | 0.3360 | 0.3495 | 1.0000 | -0.0135 |
| official_tennis_control | 40 | F1@20 | 0.6487 | 0.6058 | 0.0000 | 0.0429 |
| official_tennis_control | 40 | Recall@20 | 0.6253 | 0.5918 | 0.0000 | 0.0335 |
| official_tennis_control | 40 | Precision@20 | 0.6740 | 0.6352 | 0.0000 | 0.0388 |
| official_tennis_control | 40 | HiddenFP | 0.5666 | 0.4919 | 1.0000 | 0.0747 |

## Held-out comparisons (literals supplied via --heldout-metric)

_No held-out comparisons: fewer than 2 candidates supplied --heldout-metric for the same clip+metric._
