# BALL LoSO (Leave-One-Source-Out) Validation Report

Status: `TESTED-ON-REAL-DATA` | objective_result: `PASS`

BALL is not verified by this report. This is a scoring/analysis artifact over already-materialized predictions and already-reviewed labels; it runs no inference and trains nothing.

- CVAT root: `runs/lanes/w6_labelingest_20260708/reviewed_corpus`
- Internal-val-only (legal LoSO fold) clip ids: `['burlington_gold_0300_low_steep_corner', 'wolverine_mixed_0200_mid_steep_corner']`
- Strict-holdout clip ids never scored by this script: `['outdoor_webcam_iynbd_1500_long_high_baseline', 'indoor_doubles_fwuks_0500_long_mid_baseline']`

## Candidates

| Candidate | Folds | Metric | Pooled/Mixed | LoSO-mean | LoSO-worst | Gap (pooled-mean) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A_seed_official_aug | 20 | F1@20 | 0.6142 | 0.6435 | 0.3750 | -0.0293 |
| A_seed_official_aug | 20 | Recall@20 | 0.6525 | 0.6694 | 0.3333 | -0.0169 |
| A_seed_official_aug | 20 | Precision@20 | 0.5802 | 0.6270 | 0.3750 | -0.0468 |
| A_seed_official_aug | 20 | HiddenFP | 0.2483 | 0.3247 | 1.0000 | -0.0764 |
| E3k_seed_official_aug | 20 | F1@20 | 0.5207 | 0.5575 | 0.0000 | -0.0367 |
| E3k_seed_official_aug | 20 | Recall@20 | 0.5704 | 0.5828 | 0.0000 | -0.0125 |
| E3k_seed_official_aug | 20 | Precision@20 | 0.4791 | 0.5499 | 0.0000 | -0.0708 |
| E3k_seed_official_aug | 20 | HiddenFP | 0.3804 | 0.3906 | 1.0000 | -0.0102 |
| official_tennis_control | 20 | F1@20 | 0.3611 | 0.3035 | 0.0000 | 0.0576 |
| official_tennis_control | 20 | Recall@20 | 0.3812 | 0.3080 | 0.0000 | 0.0733 |
| official_tennis_control | 20 | Precision@20 | 0.3430 | 0.3193 | 0.0000 | 0.0237 |
| official_tennis_control | 20 | HiddenFP | 0.5968 | 0.5847 | 1.0000 | 0.0121 |

## Held-out comparisons (literals supplied via --heldout-metric)

_No held-out comparisons: fewer than 2 candidates supplied --heldout-metric for the same clip+metric._
