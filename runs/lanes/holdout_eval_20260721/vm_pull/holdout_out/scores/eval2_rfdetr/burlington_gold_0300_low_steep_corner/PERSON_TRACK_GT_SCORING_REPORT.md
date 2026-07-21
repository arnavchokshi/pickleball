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
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr` | `do_not_promote` | 1 | 0.9233 | 0.9233 | 0.9260 | 0.9260 | 0 | 182 | 30 | 0.9933 | 0.9933 | n/a | missing_gt_detections | missing_required_clips:indoor_doubles_fwuks_0500_long_mid_baseline,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner, burlington_gold_0300_low_steep_corner:spectator_or_background_false_positives_present, burlington_gold_0300_low_steep_corner:off_court_false_positives_present |

## Source Decisions (Gate v2)

Gate v2 promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero **true** spectator/background false positives (near-miss localization FPs on real players no longer count against this axis), zero off-court false-positive frames, four-player coverage >= 0.95, and a near-miss false-positive rate <= 0.10 (localization-quality target, non-strict).

| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr` | `do_not_promote` | 0 | 182 | 0.0760 | 0 | 30 | 0.9933 | missing_required_clips:indoor_doubles_fwuks_0500_long_mid_baseline,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner, burlington_gold_0300_low_steep_corner:off_court_false_positives_present |

## Source Decisions (Gate v2.1)

Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= 0.85, zero ID switches, zero true spectator/background false positives, four-player coverage >= 0.95, near-miss rate <= 0.10), **except the off-court axis is narrowed from any world point outside the court lines to only points more than 1.0m beyond them** (`far_off_court_false_positive_frames == 0`). Excursions within the 1.0m apron are reported as `apron_off_court_excursion_*` diagnostics and are never gate-blocking. See the module docstring (`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. **PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in `runs/manager/heldout_eval_ledger.md`.**

| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr` | `do_not_promote` | 0 | 0.0760 | 71 | 0 | 0.9933 | missing_required_clips:indoor_doubles_fwuks_0500_long_mid_baseline,outdoor_webcam_iynbd_1500_long_high_baseline,wolverine_mixed_0200_mid_steep_corner |

## Clip Scores

| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr` | burlington_gold_0300_low_steep_corner | 0.9233 | 0.9260 | 0.8575 | 1.0000 | 0.8467 | 0 | 182 | 186 | 30 | 0.9933 | 596/600 | n/a | 4 | missing_gt_detections | `/home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr/burlington_gold_0300_low_steep_corner/burlington_gold_0300_low_steep_corner/tracks.json` |

## Temporal Coverage Diagnostics

| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `//home/arnavchokshi/holdout_eval_20260721/repo/holdout_out/score_roots/eval2_rfdetr` | burlington_gold_0300_low_steep_corner | 0-599 | 0-599 | 0 | 0 | 0 |

## Identity Switch Events

Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.

| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
