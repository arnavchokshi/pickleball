# Person Track GT Scoring

- status: `scored_existing_tracks_only`
- IoU threshold: `0.5`
- track sources: `5`
- track files: `10`
- inference: not run

Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.

## Source Decisions

| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Mean HOTA | Worst HOTA | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `detbench_out/scored/arm0a_repro` | `do_not_promote` | 2 | 0.8673 | 0.8516 | 0.8752 | 0.8611 | 0 | 315 | 96 | 0.7358 | 0.7117 | 31.0685 | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm0b_feeder` | `do_not_promote` | 2 | 0.8594 | 0.8192 | 0.8573 | 0.8104 | 2 | 363 | 66 | 0.9500 | 0.9267 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm1_rfdetr_l` | `do_not_promote` | 2 | 0.8579 | 0.7955 | 0.8590 | 0.7947 | 1 | 330 | 75 | 0.8617 | 0.7267 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | `do_not_promote` | 2 | 0.8102 | 0.7433 | 0.8159 | 0.7479 | 2 | 257 | 54 | 0.5258 | 0.3900 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm3_dfine_l` | `do_not_promote` | 2 | 0.8525 | 0.8157 | 0.8539 | 0.8130 | 1 | 269 | 55 | 0.6433 | 0.5767 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |

## Source Decisions (Gate v2)

Gate v2 promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero **true** spectator/background false positives (near-miss localization FPs on real players no longer count against this axis), zero off-court false-positive frames, four-player coverage >= 0.95, and a near-miss false-positive rate <= 0.10 (localization-quality target, non-strict).

| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `detbench_out/scored/arm0a_repro` | `do_not_promote` | 0 | 315 | 0.0942 | 0 | 96 | 0.7117 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `detbench_out/scored/arm0b_feeder` | `do_not_promote` | 19 | 344 | 0.0966 | 0 | 66 | 0.9267 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm1_rfdetr_l` | `do_not_promote` | 16 | 314 | 0.0896 | 0 | 75 | 0.7267 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | `do_not_promote` | 0 | 257 | 0.0822 | 0 | 54 | 0.3900 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm3_dfine_l` | `do_not_promote` | 0 | 269 | 0.0817 | 0 | 55 | 0.5767 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |

## Source Decisions (Gate v2.1)

Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= 0.85, zero ID switches, zero true spectator/background false positives, four-player coverage >= 0.95, near-miss rate <= 0.10), **except the off-court axis is narrowed from any world point outside the court lines to only points more than 1.0m beyond them** (`far_off_court_false_positive_frames == 0`). Excursions within the 1.0m apron are reported as `apron_off_court_excursion_*` diagnostics and are never gate-blocking. See the module docstring (`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. **PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in `runs/manager/heldout_eval_ledger.md`.**

| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `detbench_out/scored/arm0a_repro` | `do_not_promote` | 0 | 0.0942 | 234 | 0 | 0.7117 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `detbench_out/scored/arm0b_feeder` | `do_not_promote` | 19 | 0.0966 | 248 | 0 | 0.9267 | wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm1_rfdetr_l` | `do_not_promote` | 16 | 0.0896 | 230 | 0 | 0.7267 | wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | `do_not_promote` | 0 | 0.0822 | 247 | 0 | 0.3900 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `detbench_out/scored/arm3_dfine_l` | `do_not_promote` | 0 | 0.0817 | 186 | 0 | 0.5767 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |

## Clip Scores

| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `detbench_out/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0.8831 | 0.8892 | 0.7906 | 1.0000 | 0.7746 | 0 | 184 | 357 | 29 | 0.7117 | 427/600 | 32.3344 | 4 | missing_gt_detections | `detbench_out/scored/burlington_gold_0300_low_steep_corner/arm0a_repro/tracks.json` |
| `detbench_out/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0.8516 | 0.8611 | 0.7415 | 1.0000 | 0.7133 | 0 | 131 | 213 | 67 | 0.7600 | 228/300 | 29.8026 | 4 | missing_gt_detections | `detbench_out/scored/wolverine_mixed_0200_mid_steep_corner/arm0a_repro/tracks.json` |
| `detbench_out/scored/arm0b_feeder` | burlington_gold_0300_low_steep_corner | 0.8997 | 0.9042 | 0.8176 | 1.0000 | 0.8000 | 0 | 232 | 248 | 31 | 0.9733 | 584/600 | n/a | 4 | missing_gt_detections | `detbench_out/scored/burlington_gold_0300_low_steep_corner/arm0b_feeder/tracks.json` |
| `detbench_out/scored/arm0b_feeder` | wolverine_mixed_0200_mid_steep_corner | 0.8192 | 0.8104 | 0.7866 | 0.8349 | 0.7617 | 2 | 131 | 153 | 35 | 0.9267 | 278/300 | n/a | 4 | missing_gt_detections | `detbench_out/scored/wolverine_mixed_0200_mid_steep_corner/arm0b_feeder/tracks.json` |
| `detbench_out/scored/arm1_rfdetr_l` | burlington_gold_0300_low_steep_corner | 0.9204 | 0.9233 | 0.8525 | 1.0000 | 0.8408 | 0 | 190 | 192 | 30 | 0.9967 | 598/600 | n/a | 4 | missing_gt_detections | `detbench_out/scored/burlington_gold_0300_low_steep_corner/arm1_rfdetr_l/tracks.json` |
| `detbench_out/scored/arm1_rfdetr_l` | wolverine_mixed_0200_mid_steep_corner | 0.7955 | 0.7947 | 0.7224 | 0.8743 | 0.6892 | 1 | 140 | 232 | 45 | 0.7267 | 218/300 | n/a | 4 | missing_gt_detections | `detbench_out/scored/wolverine_mixed_0200_mid_steep_corner/arm1_rfdetr_l/tracks.json` |
| `detbench_out/scored/arm2_rfdetr_seg_l` | burlington_gold_0300_low_steep_corner | 0.8771 | 0.8838 | 0.7811 | 1.0000 | 0.7646 | 0 | 181 | 384 | 32 | 0.6617 | 397/600 | n/a | 4 | missing_gt_detections | `detbench_out/scored/burlington_gold_0300_low_steep_corner/arm2_rfdetr_seg_l/tracks.json` |
| `detbench_out/scored/arm2_rfdetr_seg_l` | wolverine_mixed_0200_mid_steep_corner | 0.7433 | 0.7479 | 0.6701 | 0.8349 | 0.6475 | 2 | 76 | 345 | 22 | 0.3900 | 117/300 | n/a | 4 | missing_gt_detections | `detbench_out/scored/wolverine_mixed_0200_mid_steep_corner/arm2_rfdetr_seg_l/tracks.json` |
| `detbench_out/scored/arm3_dfine_l` | burlington_gold_0300_low_steep_corner | 0.8893 | 0.8948 | 0.8007 | 1.0000 | 0.7867 | 0 | 169 | 343 | 30 | 0.7100 | 426/600 | n/a | 4 | missing_gt_detections | `detbench_out/scored/burlington_gold_0300_low_steep_corner/arm3_dfine_l/tracks.json` |
| `detbench_out/scored/arm3_dfine_l` | wolverine_mixed_0200_mid_steep_corner | 0.8157 | 0.8130 | 0.7446 | 0.8876 | 0.7225 | 1 | 100 | 232 | 25 | 0.5767 | 173/300 | n/a | 4 | missing_gt_detections | `detbench_out/scored/wolverine_mixed_0200_mid_steep_corner/arm3_dfine_l/tracks.json` |

## Temporal Coverage Diagnostics

| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `detbench_out/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `detbench_out/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `detbench_out/scored/arm0b_feeder` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `detbench_out/scored/arm0b_feeder` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `detbench_out/scored/arm1_rfdetr_l` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `detbench_out/scored/arm1_rfdetr_l` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `detbench_out/scored/arm3_dfine_l` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `detbench_out/scored/arm3_dfine_l` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |

## Identity Switch Events

Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.

| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `detbench_out/scored/arm0b_feeder` | wolverine_mixed_0200_mid_steep_corner | 56 | 2 | 4 | 3 | 46 | 10 | 0.5915 |
| `detbench_out/scored/arm0b_feeder` | wolverine_mixed_0200_mid_steep_corner | 63 | 4 | 3 | 4 | 38 | 25 | 0.6497 |
| `detbench_out/scored/arm1_rfdetr_l` | wolverine_mixed_0200_mid_steep_corner | 55 | 2 | 4 | 3 | 49 | 6 | 0.5042 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | wolverine_mixed_0200_mid_steep_corner | 103 | 2 | 4 | 3 | 42 | 61 | 0.5245 |
| `detbench_out/scored/arm2_rfdetr_seg_l` | wolverine_mixed_0200_mid_steep_corner | 121 | 4 | 3 | 4 | 42 | 79 | 0.5369 |
| `detbench_out/scored/arm3_dfine_l` | wolverine_mixed_0200_mid_steep_corner | 58 | 1 | 4 | 1 | 42 | 16 | 0.5552 |
