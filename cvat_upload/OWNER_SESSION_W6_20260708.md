# Owner Session W6 2026-07-08

Use these tasks in order. Full sessions are 640 frames, about 2.7 hours at the measured 240 frames/hour. The packages already carry one editable ball prelabel per frame.

## First Command

```bash
cd /Users/arnavchokshi/Desktop/pickleball
open -a Docker
cd /Users/arnavchokshi/cvat_labelfactory/cvat_src && docker compose up -d
cd /Users/arnavchokshi/Desktop/pickleball
runs/lanes/w3_labelfactory_20260707/venv/bin/python runs/lanes/w6_labelpack_20260708/import_w6_labelpack_tasks.py --dry-run
# The separate import lane removes --dry-run when it is ready to create tasks.
```

Then open http://localhost:8080 and label the tasks named below after the import lane creates them.

## Ball Convention

- Label the ball box around the visible ball or visible blur streak.
- BlurBall convention: for motion-blurred balls, put the box center on the blur-streak center, not the leading edge.
- `clear`: ball is visible and localizable without material occlusion.
- `partial`: ball is localizable but partly occluded or blurred.
- `full`: ball is expected in-frame but fully hidden.
- `out_of_frame`: ball is outside the image bounds.
- Keep one ball object per frame. Drag/correct the prelabel if it is close; delete/recreate only when it is misleading.
- **Prelabel on a NON-ball while the real ball IS visible**: drag the box onto the real ball (or delete + redraw). Never leave clear/partial on a non-ball.
- **Ball NOT in frame but a box exists**: either DELETE the box or set visibility_level=out_of_frame. Use the visibility_level ATTRIBUTE dropdown.
- **Background ball**: ALWAYS delete that box. One game-ball box max per frame.
- **When in doubt, delete.** False positives poison training far more than missed positives; reviewed-empty frames are useful negatives.

## Session Order

### ball_session_01 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_01_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_01_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_01_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 223, "student-only": 160, "teacher-only": 257}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_02 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_02_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_02_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_02_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 160, "student-only": 150, "teacher-only": 330}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_03 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_03_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_03_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_03_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 160, "student-only": 120, "teacher-only": 360}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_04 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_04_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_04_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_04_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 160, "student-only": 120, "teacher-only": 360}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_05 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_05_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_05_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_05_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 81, "student-only": 120, "teacher-only": 439}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_06 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_06_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_06_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_06_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 94, "teacher-only": 466}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_07 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_07_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_07_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_07_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_08 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_08_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_08_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_08_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_09 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_09_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_09_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_09_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_10 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_10_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_10_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_10_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_11 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_11_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_11_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_11_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_12 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_12_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_12_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_12_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_13 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_13_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_13_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_13_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_14 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_14_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_14_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_14_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_15 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_15_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_15_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_15_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 80, "teacher-only": 480}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_16 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_16_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_16_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_16_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 53, "teacher-only": 507}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_17 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_17_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_17_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_17_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_18 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_18_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_18_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_18_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_19 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_19_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_19_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_19_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_20 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_20_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_20_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_20_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_21 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_21_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_21_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_21_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_22 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_22_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_22_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_22_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_23 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_23_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_23_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_23_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_24 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_24_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_24_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_24_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_25 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_25_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_25_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_25_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_26 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_26_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_26_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_26_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_27 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_27_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_27_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_27_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_28 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_28_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_28_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_28_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_29 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_29_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_29_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_29_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_30 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_30_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_30_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_30_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_31 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_31_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_31_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_31_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_32 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_32_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_32_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_32_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_33 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_33_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_33_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_33_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_34 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_34_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_34_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_34_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_35 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_35_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_35_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_35_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_36 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_36_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_36_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_36_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_37 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_37_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_37_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_37_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_38 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_38_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_38_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_38_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_39 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_39_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_39_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_39_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_40 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_40_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_40_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_40_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_41 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_41_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_41_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_41_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_42 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_42_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_42_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_42_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_43 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_43_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_43_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_43_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_44 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_44_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_44_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_44_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_45 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_45_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_45_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_45_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_46 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_46_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_46_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_46_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_47 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_47_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_47_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_47_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_48 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_48_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_48_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_48_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_49 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_49_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_49_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_49_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_50 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_50_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_50_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_50_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 160, "_L0HVmAlCQI": 160, "wBu8bC4OfUY": 160, "zwCtH_i1_S4": 160}`
- Error mix: `{"large-offset": 80, "student-only": 40, "teacher-only": 520}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_51 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_51_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_51_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_51_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, _L0HVmAlCQI=outdoor_night_tennis_overlay, wBu8bC4OfUY=outdoor_night_tennis_overlay, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 207, "_L0HVmAlCQI": 73, "wBu8bC4OfUY": 154, "zwCtH_i1_S4": 206}`
- Error mix: `{"large-offset": 104, "student-only": 51, "teacher-only": 485}`
- Unlocks: Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage.

### ball_session_52 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_52_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_52_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_52_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level, zwCtH_i1_S4=outdoor_day_broadcast_overlay
- Source counts: `{"HyUqT7zFiwk": 634, "zwCtH_i1_S4": 6}`
- Error mix: `{"large-offset": 261, "student-only": 159, "teacher-only": 220}`
- Unlocks: Keeps the still-active Phase-B sources mixed before any single-source tail work.

### ball_session_53 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_53_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_53_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_53_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_54 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_54_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_54_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_54_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_55 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_55_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_55_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_55_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_56 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_56_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_56_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_56_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_57 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_57_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_57_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_57_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_58 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_58_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_58_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_58_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_59 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_59_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_59_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_59_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_60 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_60_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_60_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_60_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_61 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_61_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_61_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_61_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_62 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_62_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_62_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_62_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_63 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_63_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_63_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_63_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_64 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_64_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_64_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_64_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_65 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_65_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_65_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_65_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_66 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_66_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_66_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_66_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_67 - 640 frames

- CVAT task: `w6_ball_sst_ball_session_67_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_67_640f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_67_640f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 640}`
- Error mix: `{"student-only": 160, "teacher-only": 480}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

### ball_session_68 - 350 frames

- CVAT task: `w6_ball_sst_ball_session_68_20260708`
- Images: `cvat_upload/w6_labelpack_20260708/packages/ball_session_68_350f_w6_images.zip`
- Prelabels: `cvat_upload/w6_labelpack_20260708/packages/ball_session_68_350f_w6_prelabels_cvat1_1.zip`
- Source classes: HyUqT7zFiwk=indoor_court_level
- Source counts: `{"HyUqT7zFiwk": 350}`
- Error mix: `{"student-only": 78, "teacher-only": 272}`
- Unlocks: Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source.

## Export

For each finished task: Actions -> Export task dataset -> `CVAT for images 1.1`. Save the zip under:

```text
cvat_upload/exports/w6_labelpack_20260708/
```

Recommended filenames: `<task_name>_annotations.zip`.

## Package Check

Protected material check passed before handoff:

```text
NO_MATCH pattern=pwxNwFfYQlQ packages=136
NO_MATCH pattern=vQhtz8l6VqU packages=136
NO_MATCH pattern=outdoor_webcam_iynbd_1500_long_high_baseline packages=136
NO_MATCH pattern=indoor_doubles_fwuks_0500_long_mid_baseline packages=136
NO_MATCH pattern=03_outdoor_webcam_iynbd packages=136
NO_MATCH pattern=04_indoor_doubles_fwuks packages=136
```
