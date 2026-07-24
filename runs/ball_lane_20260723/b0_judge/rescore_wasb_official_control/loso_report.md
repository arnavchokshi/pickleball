# BALL LoSO (Leave-One-Source-Out) Validation Report

Status: `TESTED-ON-REAL-DATA` | objective_result: `PASS`

BALL is not verified by this report. This is a scoring/analysis artifact over already-materialized predictions and already-reviewed labels; it runs no inference and trains nothing.

- CVAT root: `None`
- Internal-val-only (legal LoSO fold) clip ids: `['burlington_gold_0300_low_steep_corner', 'wolverine_mixed_0200_mid_steep_corner']`
- Strict-holdout clip ids never scored by this script: `['outdoor_webcam_iynbd_1500_long_high_baseline', 'indoor_doubles_fwuks_0500_long_mid_baseline']`

## Candidates

| Candidate | Folds | Metric | Pooled/Mixed | LoSO-mean | LoSO-worst | Gap (pooled-mean) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| wasb_official_control | 2 | F1@20 | 0.5670 | 0.5164 | 0.2933 | 0.0506 |
| wasb_official_control | 2 | Recall@20 | 0.5851 | 0.5271 | 0.3667 | 0.0580 |
| wasb_official_control | 2 | Precision@20 | 0.5500 | 0.5222 | 0.2444 | 0.0278 |
| wasb_official_control | 2 | HiddenFP | 0.4932 | 0.4906 | 0.6757 | 0.0025 |

## Held-out comparisons (literals supplied via --heldout-metric)

_No held-out comparisons: fewer than 2 candidates supplied --heldout-metric for the same clip+metric._
