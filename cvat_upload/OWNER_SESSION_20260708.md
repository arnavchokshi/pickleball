# Owner Session 2026-07-08

Use these tasks in order. Each ball session is 640 frames, about 2.7 hours at the measured 240 frames/hour. The packages already carry one editable ball prelabel per frame.

## First Command

```bash
cd /Users/arnavchokshi/Desktop/pickleball
open -a Docker
cd /Users/arnavchokshi/cvat_labelfactory/cvat_src && docker compose up -d
cd /Users/arnavchokshi/Desktop/pickleball
runs/lanes/w3_labelfactory_20260707/venv/bin/python runs/lanes/w5_labelpack_20260708/import_w5_labelpack_tasks.py
```

Then open http://localhost:8080 and label the tasks named below.

## Ball Convention

- Label the ball box around the visible ball or visible blur streak.
- BlurBall convention: for motion-blurred balls, put the box center on the blur-streak center, not the leading edge.
- `clear`: ball is visible and localizable without material occlusion.
- `partial`: ball is localizable but partly occluded or blurred.
- `full`: ball is expected in-frame but fully hidden.
- `out_of_frame`: ball is outside the image bounds.
- Keep one ball object per frame. Drag/correct the prelabel if it is close; delete/recreate only when it is misleading.
- **Prelabel on a NON-ball while the real ball IS visible**: drag the box onto the real ball
  (or delete + redraw). Never leave clear/partial on a non-ball — those coords train the model.
- **Ball NOT in frame but a box exists**: either DELETE the box (frame becomes a reviewed-absent
  negative — valid and valuable) OR leave it anywhere and set visibility_level=out_of_frame
  (coords are ignored for full/out_of_frame). Use the visibility_level ATTRIBUTE dropdown, not
  CVAT's native outside/occluded toggles.
- **Background ball (real ball, not the game ball — fence line, adjacent court)**: ALWAYS delete
  that box. One game-ball box max per frame (the importer hard-errors on 2+), and static
  background balls are the distractor-lock failure class that broke a prior model.
- **When in doubt, delete.** False positives poison training far more than missed positives;
  reviewed-empty frames are useful negatives.

## Session Order

### ball_session_01 - 640 frames

- CVAT task: `w5_ball_sst_ball_session_01_20260708`
- Images: `cvat_upload/w5_labelpack_20260708/packages/ball_session_01_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip`
- Prelabels: `cvat_upload/w5_labelpack_20260708/packages/ball_session_01_640f_73VurrTKCZ8_Ezz6HDNHlnk_prelabels_cvat1_1.zip`
- Source classes: 73VurrTKCZ8=outdoor_day_multicam, Ezz6HDNHlnk=outdoor_night_fenced
- Error mix: `{"large-offset": 320, "student-only": 160, "teacher-only": 160}`
- Unlocks: First label this: front-loads both available outdoor queue sources and all three disagreement classes.

### ball_session_02 - 640 frames

- CVAT task: `w5_ball_sst_ball_session_02_20260708`
- Images: `cvat_upload/w5_labelpack_20260708/packages/ball_session_02_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip`
- Prelabels: `cvat_upload/w5_labelpack_20260708/packages/ball_session_02_640f_73VurrTKCZ8_Ezz6HDNHlnk_prelabels_cvat1_1.zip`
- Source classes: 73VurrTKCZ8=outdoor_day_multicam, Ezz6HDNHlnk=outdoor_night_fenced
- Error mix: `{"large-offset": 320, "student-only": 160, "teacher-only": 160}`
- Unlocks: Adds reviewed outdoor disagreement frames for BALL P0-4 and keeps both current queue sources represented.

### ball_session_03 - 640 frames

- CVAT task: `w5_ball_sst_ball_session_03_20260708`
- Images: `cvat_upload/w5_labelpack_20260708/packages/ball_session_03_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip`
- Prelabels: `cvat_upload/w5_labelpack_20260708/packages/ball_session_03_640f_73VurrTKCZ8_Ezz6HDNHlnk_prelabels_cvat1_1.zip`
- Source classes: 73VurrTKCZ8=outdoor_day_multicam, Ezz6HDNHlnk=outdoor_night_fenced
- Error mix: `{"large-offset": 349, "student-only": 160, "teacher-only": 131}`
- Unlocks: Adds reviewed outdoor disagreement frames for BALL P0-4 and keeps both current queue sources represented.

### ball_session_04 - 640 frames

- CVAT task: `w5_ball_sst_ball_session_04_20260708`
- Images: `cvat_upload/w5_labelpack_20260708/packages/ball_session_04_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip`
- Prelabels: `cvat_upload/w5_labelpack_20260708/packages/ball_session_04_640f_73VurrTKCZ8_Ezz6HDNHlnk_prelabels_cvat1_1.zip`
- Source classes: 73VurrTKCZ8=outdoor_day_multicam, Ezz6HDNHlnk=outdoor_night_fenced
- Error mix: `{"large-offset": 400, "student-only": 160, "teacher-only": 80}`
- Unlocks: Adds reviewed outdoor disagreement frames for BALL P0-4 and keeps both current queue sources represented.

## Court-KP Mini Task

- CVAT task: `w5_court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_20260708`
- Images: `cvat_upload/w5_labelpack_20260708/packages/court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_images.zip`
- Prelabels: `cvat_upload/w5_labelpack_20260708/packages/court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_prelabels_cvat1_1.zip`
- On `relabel_net_far_side` frames, correct the net and far-side points first.
- On `replacement_full_metric15` frames, label all 15 metric court keypoints if the original frame remains ambiguous.
- Focus points: `far_right_corner, far_baseline_center, far_left_corner, net_left_sideline, net_center, net_right_sideline, far_nvz_left, far_nvz_center, far_nvz_right`

## Export

For each finished task: Actions -> Export task dataset -> `CVAT for images 1.1`. Save the zip under:

```text
cvat_upload/exports/w5_labelpack_20260708/
```

Recommended filenames: `<task_name>_annotations.zip`.

## Package Check

Protected material check passed before handoff:

```text
NO_MATCH pattern=pwxNwFfYQlQ packages=10
NO_MATCH pattern=vQhtz8l6VqU packages=10
NO_MATCH pattern=outdoor_webcam_iynbd_1500_long_high_baseline packages=10
NO_MATCH pattern=indoor_doubles_fwuks_0500_long_mid_baseline packages=10
NO_MATCH pattern=03_outdoor_webcam_iynbd packages=10
NO_MATCH pattern=04_indoor_doubles_fwuks packages=10
```

## Localhost Quickstart (live-verified 2026-07-08)

CVAT is running now at **http://localhost:8080** with all 5 tasks below loaded and
prelabels rendering (verified via API frame/annotation counts + a browser screenshot). If
it's ever down after a reboot, run the "First Command" block above -- `docker compose up -d`
brings back the *same* instance (all prior tasks/annotations are on disk in Docker volumes,
nothing was recreated).

- **Login**: username `admin`, password in `data/credentials/cvat_local.txt` (gitignored,
  chmod 600 -- not printed here). The CVAT login form is two-step: type the username, click
  Next, then enter the password.
- **Label in this order**: `w5_ball_sst_ball_session_01_20260708` (640f) ->
  `_02_` (640f) -> `_03_` (640f) -> `_04_` (640f) ->
  `w5_court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_20260708` (4f).
- **The 3-sentence reminder**: get the ball box tight around the true ball position/blur
  streak first. Then set `visibility_level` (clear / partial / full / out_of_frame) to match
  what's actually visible in that frame. For motion blur, center the box on the blur streak,
  not the leading edge.
- **Export when a session is done**: task page -> Actions -> Export task dataset -> format
  `CVAT for images 1.1` -> save the zip under `cvat_upload/exports/w5_labelpack_20260708/`
  (e.g. `w5_session_01_ball_annotations.zip`, matching the Export section above).
