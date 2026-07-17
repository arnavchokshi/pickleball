# Person Track GT Scoring

- status: `scored_existing_tracks_only`
- IoU threshold: `0.5`
- track sources: `1`
- track files: `2`
- inference: not run

Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.

## Source Decisions

| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Mean HOTA | Worst HOTA | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | `do_not_promote` | 2 | 0.8236 | 0.7640 | 0.8208 | 0.7525 | 2 | 336 | 99 | 0.7642 | 0.7117 | 31.0685 | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |

## Source Decisions (Gate v2)

Gate v2 promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero **true** spectator/background false positives (near-miss localization FPs on real players no longer count against this axis), zero off-court false-positive frames, four-player coverage >= 0.95, and a near-miss false-positive rate <= 0.10 (localization-quality target, non-strict).

| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | `do_not_promote` | 9 | 327 | 0.0973 | 0 | 99 | 0.7117 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |

## Source Decisions (Gate v2.1)

Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= 0.85, zero ID switches, zero true spectator/background false positives, four-player coverage >= 0.95, near-miss rate <= 0.10), **except the off-court axis is narrowed from any world point outside the court lines to only points more than 1.0m beyond them** (`far_off_court_false_positive_frames == 0`). Excursions within the 1.0m apron are reported as `apron_off_court_excursion_*` diagnostics and are never gate-blocking. See the module docstring (`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. **PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in `runs/manager/heldout_eval_ledger.md`.**

| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | `do_not_promote` | 9 | 0.0973 | 234 | 0 | 0.7117 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |

## Clip Scores

| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0.8831 | 0.8892 | 0.7906 | 1.0000 | 0.7746 | 0 | 184 | 357 | 29 | 0.7117 | 427/600 | 32.3344 | 4 | missing_gt_detections | `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/burlington_gold_0300_low_steep_corner/arm0a_repro/tracks.json` |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0.7640 | 0.7525 | 0.7271 | 0.7789 | 0.6908 | 2 | 152 | 217 | 70 | 0.8167 | 245/300 | 29.8026 | 4 | missing_gt_detections | `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/wolverine_mixed_0200_mid_steep_corner/arm0a_repro/tracks.json` |

## Temporal Coverage Diagnostics

| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |

## Identity Switch Events

Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.

| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 55 | 2 | 4 | 3 | 47 | 8 | 0.5182 |
| `runs/lanes/trk_rfdetr_prod_20260716/gate1/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 58 | 1 | 3 | 1 | 42 | 16 | 0.5877 |
