# CVAT Exports

Canonical export type: `CVAT for video 1.1`.

Use one zip per source video. Keep these stable filenames:

- `01_burlington_gold_0300_low_steep_corner_cvat_for_video_1.1.zip`
- `02_wolverine_mixed_0200_mid_steep_corner_cvat_for_video_1.1.zip`
- `03_outdoor_webcam_iynbd_1500_long_high_baseline_cvat_for_video_1.1.zip`
- `04_indoor_doubles_fwuks_0500_long_mid_baseline_cvat_for_video_1.1.zip`

Current active set: videos 1, 2, 3, and 4. Video 4 (`indoor_doubles_fwuks_0500_long_mid_baseline`)
landed 2026-07-02 and is imported at `runs/cvat_imports/2026_06_30/indoor_doubles_fwuks_0500_long_mid_baseline/`.
Outdoor and Indoor are strict held-out eval clips (`threed/racketsport/eval_guard.py`,
`runs/manager/heldout_eval_ledger.md`), and current YOLO/TrackNet training exporters fail closed
if either clip appears in training or validation-during-fitting inputs. Older generated
`runs/cvat_imports/2026_06_30/gate_inputs/` and `yolo_datasets/` artifacts are historical
detector-label evidence, not the recommended current training/export scope.

Video 3 is intentionally capped: keep source frames `0..1150` only. The raw
CVAT export has 1800 frames, but frame 1151 onward is discarded for the
reviewed import/eval artifacts. Use:

`cvat_upload/03_outdoor_webcam_iynbd_1500_long_high_baseline_frames_0000_1150.mp4`

as the source video when recreating video 3 reviewed import/eval artifacts.

Imported artifacts go under:

`runs/cvat_imports/2026_06_30/<clip_id>/`

Each imported clip directory should contain:

- `reviewed_boxes.json`: reviewed `player`, `paddle`, and `ball` boxes from CVAT. CVAT `ball` ellipses are converted to detector bounding boxes.
- `person_ground_truth.json`: player-only ground truth for tracking evaluation.
- `import_summary.json`: compact counts for quick review.

The consolidated manifest and generated YOLO dataset paths are in:

`runs/cvat_imports/2026_06_30/manifest.json`
