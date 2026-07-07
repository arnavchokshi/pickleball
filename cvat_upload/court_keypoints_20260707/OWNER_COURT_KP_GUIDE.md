# Owner Court-Keypoint Guide - 2026-07-07

Task: label one PNG frame per legal non-held-out harvest source using the metric-15 court convention.

Import command:

```bash
.venv/bin/python cvat_upload/court_keypoints_20260707/import_court_kp_tasks.py
```

This creates one CVAT image task, `racketsport_metric15_court_keypoints_20260707_6frames`, with 6 frames. One task is used to keep owner import/export to one click path; each PNG filename and the manifest preserve source, clip, and frame provenance.

Important count note: the current harvest card has 8 downloaded sources total, but `pwxNwFfYQlQ` and `vQhtz8l6VqU` are held-out proposals and are excluded here. This package therefore contains 6 legal frames, not 8.

Convention citation: the exact metric-15 names/order are defined in `threed/racketsport/schemas/__init__.py:30-46`; world positions and descriptions are in `threed/racketsport/court_keypoint_net.py:161-177`; the full metric-15 loader is `threed/racketsport/court_calibration_metric15.py:111-150`.

## Points

Place each point at the center of the painted court-line intersection or net-top location named below. Use the target pickleball court, not adjacent courts. The full metric-15 convention has no per-point occlusion attribute; the selected frames are intended to be labelable at all 15 points. If a point is genuinely unplaceable, note that in the export handoff rather than inventing an occlusion value.

| Order | CVAT label | Meaning |
|---:|---|---|
| 0 | `near_left_corner` | near baseline left sideline corner |
| 1 | `near_baseline_center` | near baseline at centerline |
| 2 | `near_right_corner` | near baseline right sideline corner |
| 3 | `far_right_corner` | far baseline right sideline corner |
| 4 | `far_baseline_center` | far baseline at centerline |
| 5 | `far_left_corner` | far baseline left sideline corner |
| 6 | `near_nvz_left` | near NVZ line at left sideline |
| 7 | `near_nvz_center` | near NVZ line at centerline |
| 8 | `near_nvz_right` | near NVZ line at right sideline |
| 9 | `net_left_sideline` | net top at left sideline |
| 10 | `net_center` | net top at centerline |
| 11 | `net_right_sideline` | net top at right sideline |
| 12 | `far_nvz_left` | far NVZ line at left sideline |
| 13 | `far_nvz_center` | far NVZ line at centerline |
| 14 | `far_nvz_right` | far NVZ line at right sideline |

Net labels are on the visible top of the net at the left sideline, centerline, and right sideline. Baseline/NVZ labels are on the court plane.

## Selected Frames

| Source | Clip | Absolute source frame | PNG |
|---|---|---:|---|
| `HyUqT7zFiwk` | `HyUqT7zFiwk_rally_0001` | 10195 | `HyUqT7zFiwk__HyUqT7zFiwk_rally_0001__abs_010195.png` |
| `73VurrTKCZ8` | `73VurrTKCZ8_rally_0002` | 3808 | `73VurrTKCZ8__73VurrTKCZ8_rally_0002__abs_003808.png` |
| `wBu8bC4OfUY` | `wBu8bC4OfUY_rally_0001` | 10248 | `wBu8bC4OfUY__wBu8bC4OfUY_rally_0001__abs_010248.png` |
| `Ezz6HDNHlnk` | `Ezz6HDNHlnk_rally_0004` | 10677 | `Ezz6HDNHlnk__Ezz6HDNHlnk_rally_0004__abs_010677.png` |
| `zwCtH_i1_S4` | `zwCtH_i1_S4_rally_0001` | 3636 | `zwCtH_i1_S4__zwCtH_i1_S4_rally_0001__abs_003636.png` |
| `_L0HVmAlCQI` | `_L0HVmAlCQI_rally_0001` | 509 | `_L0HVmAlCQI___L0HVmAlCQI_rally_0001__abs_000509.png` |

Expected owner time: about 10 minutes total for the 6 frames.

## Export

After labeling, export the task from CVAT as `CVAT for images 1.1` and place the ZIP under:

```text
cvat_upload/exports/court_keypoints_20260707/
```

Recommended filename:

```text
cvat_upload/exports/court_keypoints_20260707/court_keypoints_metric15_20260707_annotations.zip
```

Do not export a CVAT task backup for the training handoff; use annotations/dataset export so the XML can be parsed and filed like the harvest-review exports.
