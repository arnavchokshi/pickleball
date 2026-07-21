# Person Track GT Scoring

- status: `scored_existing_tracks_only`
- IoU threshold: `0.5`
- track sources: `1`
- track files: `1`
- inference: not run

Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.

## Source Decisions

| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Mean HOTA | Worst HOTA | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | `do_not_promote` | 1 | 0.5590 | 0.5590 | 0.6223 | 0.6223 | 4 | 1094 | 907 | 0.4566 | 0.4566 | n/a | missing_gt_detections | missing_required_clips:burlington_gold_0300_low_steep_corner,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner, indoor_doubles_fwuks_0500_long_mid_baseline:idf1_below_0.85, indoor_doubles_fwuks_0500_long_mid_baseline:id_switches_present, indoor_doubles_fwuks_0500_long_mid_baseline:spectator_or_background_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:off_court_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:four_player_coverage_below_0.95 |

## Source Decisions (Gate v2)

Gate v2 promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero **true** spectator/background false positives (near-miss localization FPs on real players no longer count against this axis), zero off-court false-positive frames, four-player coverage >= 0.95, and a near-miss false-positive rate <= 0.10 (localization-quality target, non-strict).

| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | `do_not_promote` | 395 | 354 | 0.1250 | 345 | 907 | 0.4566 | missing_required_clips:burlington_gold_0300_low_steep_corner,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner, indoor_doubles_fwuks_0500_long_mid_baseline:idf1_below_0.85, indoor_doubles_fwuks_0500_long_mid_baseline:id_switches_present, indoor_doubles_fwuks_0500_long_mid_baseline:true_spectator_or_background_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:off_court_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:four_player_coverage_below_0.95, indoor_doubles_fwuks_0500_long_mid_baseline:near_miss_false_positive_rate_above_0.10 |

## Source Decisions (Gate v2.1)

Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= 0.85, zero ID switches, zero true spectator/background false positives, four-player coverage >= 0.95, near-miss rate <= 0.10), **except the off-court axis is narrowed from any world point outside the court lines to only points more than 1.0m beyond them** (`far_off_court_false_positive_frames == 0`). Excursions within the 1.0m apron are reported as `apron_off_court_excursion_*` diagnostics and are never gate-blocking. See the module docstring (`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. **PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in `runs/manager/heldout_eval_ledger.md`.**

| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | `do_not_promote` | 395 | 0.1250 | 390 | 750 | 0.4566 | missing_required_clips:burlington_gold_0300_low_steep_corner,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner, indoor_doubles_fwuks_0500_long_mid_baseline:idf1_below_0.85, indoor_doubles_fwuks_0500_long_mid_baseline:id_switches_present, indoor_doubles_fwuks_0500_long_mid_baseline:true_spectator_or_background_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:far_off_court_false_positives_present, indoor_doubles_fwuks_0500_long_mid_baseline:four_player_coverage_below_0.95, indoor_doubles_fwuks_0500_long_mid_baseline:near_miss_false_positive_rate_above_0.10 |

## Clip Scores

| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 0.5590 | 0.6223 | 0.3886 | 0.9966 | 0.1894 | 4 | 1094 | 1641 | 907 | 0.4566 | 310/679 | n/a | 4 | missing_gt_detections | `/home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection/indoor_doubles_fwuks_0500_long_mid_baseline/indoor_doubles_fwuks_0500_long_mid_baseline/tracks.json` |

## Temporal Coverage Diagnostics

| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 0-899 | 0-899 | 0 | 0 | 0 |

## Identity Switch Events

Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.

| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 38 | 4 | 3 | 2 | 36 | 2 | 0.5580 |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 39 | 4 | 2 | 3 | 38 | 1 | 0.5054 |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 197 | 4 | 3 | 2 | 185 | 12 | 0.5039 |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval1_selection` | indoor_doubles_fwuks_0500_long_mid_baseline | 205 | 4 | 2 | 3 | 197 | 8 | 0.5053 |
