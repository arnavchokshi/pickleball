# Person Track GT Scoring

- status: `scored_existing_tracks_only`
- IoU threshold: `0.5`
- track sources: `3`
- track files: `6`
- inference: not run

Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.

## Source Decisions

| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Mean HOTA | Worst HOTA | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `rfdetrflip_out/scored/arm0a_repro` | `do_not_promote` | 2 | 0.8673 | 0.8516 | 0.8752 | 0.8611 | 0 | 315 | 96 | 0.7358 | 0.7117 | 31.0685 | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `rfdetrflip_out/scored/rfdetr_l_p` | `do_not_promote` | 2 | 0.8628 | 0.8036 | 0.8635 | 0.8021 | 1 | 328 | 74 | 0.8583 | 0.7233 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |
| `rfdetrflip_out/scored/yolo_mech018_960` | `do_not_promote` | 2 | 0.8673 | 0.8516 | 0.8752 | 0.8611 | 0 | 315 | 96 | 0.7358 | 0.7117 | n/a | missing_gt_detections | burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95 |

## Source Decisions (Gate v2)

Gate v2 promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero **true** spectator/background false positives (near-miss localization FPs on real players no longer count against this axis), zero off-court false-positive frames, four-player coverage >= 0.95, and a near-miss false-positive rate <= 0.10 (localization-quality target, non-strict).

| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rfdetrflip_out/scored/arm0a_repro` | `do_not_promote` | 0 | 315 | 0.0942 | 0 | 96 | 0.7117 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `rfdetrflip_out/scored/rfdetr_l_p` | `do_not_promote` | 4 | 324 | 0.0922 | 0 | 74 | 0.7233 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `rfdetrflip_out/scored/yolo_mech018_960` | `do_not_promote` | 0 | 315 | 0.0942 | 0 | 96 | 0.7117 | burlington_gold_0300_low_steep_corner:off_court_false_positives_present, burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:off_court_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |

## Source Decisions (Gate v2.1)

Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= 0.85, zero ID switches, zero true spectator/background false positives, four-player coverage >= 0.95, near-miss rate <= 0.10), **except the off-court axis is narrowed from any world point outside the court lines to only points more than 1.0m beyond them** (`far_off_court_false_positive_frames == 0`). Excursions within the 1.0m apron are reported as `apron_off_court_excursion_*` diagnostics and are never gate-blocking. See the module docstring (`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. **PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in `runs/manager/heldout_eval_ledger.md`.**

| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rfdetrflip_out/scored/arm0a_repro` | `do_not_promote` | 0 | 0.0942 | 234 | 0 | 0.7117 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `rfdetrflip_out/scored/rfdetr_l_p` | `do_not_promote` | 4 | 0.0922 | 229 | 0 | 0.7233 | wolverine_mixed_0200_mid_steep_corner:idf1_below_0.85, wolverine_mixed_0200_mid_steep_corner:id_switches_present, wolverine_mixed_0200_mid_steep_corner:true_spectator_or_background_false_positives_present, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |
| `rfdetrflip_out/scored/yolo_mech018_960` | `do_not_promote` | 0 | 0.0942 | 234 | 0 | 0.7117 | burlington_gold_0300_low_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:four_player_coverage_below_0.95, wolverine_mixed_0200_mid_steep_corner:near_miss_false_positive_rate_above_0.10 |

## Clip Scores

| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `rfdetrflip_out/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0.8831 | 0.8892 | 0.7906 | 1.0000 | 0.7746 | 0 | 184 | 357 | 29 | 0.7117 | 427/600 | 32.3344 | 4 | missing_gt_detections | `rfdetrflip_out/scored/burlington_gold_0300_low_steep_corner/arm0a_repro/tracks.json` |
| `rfdetrflip_out/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0.8516 | 0.8611 | 0.7415 | 1.0000 | 0.7133 | 0 | 131 | 213 | 67 | 0.7600 | 228/300 | 29.8026 | 4 | missing_gt_detections | `rfdetrflip_out/scored/wolverine_mixed_0200_mid_steep_corner/arm0a_repro/tracks.json` |
| `rfdetrflip_out/scored/rfdetr_l_p` | burlington_gold_0300_low_steep_corner | 0.9220 | 0.9248 | 0.8553 | 1.0000 | 0.8442 | 0 | 185 | 189 | 30 | 0.9933 | 596/600 | n/a | 4 | missing_gt_detections | `rfdetrflip_out/scored/burlington_gold_0300_low_steep_corner/rfdetr_l_p/tracks.json` |
| `rfdetrflip_out/scored/rfdetr_l_p` | wolverine_mixed_0200_mid_steep_corner | 0.8036 | 0.8021 | 0.7252 | 0.8872 | 0.6917 | 1 | 143 | 226 | 44 | 0.7233 | 217/300 | n/a | 4 | missing_gt_detections | `rfdetrflip_out/scored/wolverine_mixed_0200_mid_steep_corner/rfdetr_l_p/tracks.json` |
| `rfdetrflip_out/scored/yolo_mech018_960` | burlington_gold_0300_low_steep_corner | 0.8831 | 0.8892 | 0.7906 | 1.0000 | 0.7746 | 0 | 184 | 357 | 29 | 0.7117 | 427/600 | n/a | 4 | missing_gt_detections | `rfdetrflip_out/scored/burlington_gold_0300_low_steep_corner/yolo_mech018_960/tracks.json` |
| `rfdetrflip_out/scored/yolo_mech018_960` | wolverine_mixed_0200_mid_steep_corner | 0.8516 | 0.8611 | 0.7415 | 1.0000 | 0.7133 | 0 | 131 | 213 | 67 | 0.7600 | 228/300 | n/a | 4 | missing_gt_detections | `rfdetrflip_out/scored/wolverine_mixed_0200_mid_steep_corner/yolo_mech018_960/tracks.json` |

## Temporal Coverage Diagnostics

| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `rfdetrflip_out/scored/arm0a_repro` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `rfdetrflip_out/scored/arm0a_repro` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `rfdetrflip_out/scored/rfdetr_l_p` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `rfdetrflip_out/scored/rfdetr_l_p` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |
| `rfdetrflip_out/scored/yolo_mech018_960` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |
| `rfdetrflip_out/scored/yolo_mech018_960` | wolverine_mixed_0200_mid_steep_corner | 0-299 | 0-299 | 0 | 0 | 0 |

## Identity Switch Events

Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.

| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rfdetrflip_out/scored/rfdetr_l_p` | wolverine_mixed_0200_mid_steep_corner | 58 | 1 | 4 | 1 | 42 | 16 | 0.5963 |
