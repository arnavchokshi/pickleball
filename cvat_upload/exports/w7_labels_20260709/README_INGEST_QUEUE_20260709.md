# Ingest queue note — 2026-07-09 (Fable, owner-confirmed)

job_21 (= w6 ball session_03, task 21) is INGESTED (corpus 1750 -> 2388).

**MISSED BATCH — must be in the NEXT ingest:**
`cvat_upload/exports/w6_labelpack_20260708/w6_ball_sst_ball_session_02_20260708_annotations.zip`
(= w6 ball session_02, task 20 / job_20; owner-exported 2026-07-09, validated: 640 frames,
584 boxes, 56 reviewed-absent negatives, 0 multi-ball frames. The w7 Downloads sweep that
secured job_21 predates this file's arrival and never consumed it.)

Expected: +~635 rows -> corpus ~3,020 => CROSSES THE 3,000 CHECKPOINT (owner ruling:
3k-checkpoint retrain eval fires). Owner also confirmed session_03 finished — no further
export needed for it (a redundant Fable snapshot of identical content was removed).

Also staged, court (separate pipeline):
- `cvat_upload/exports/w5_labelpack_20260708/w5_court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_20260708_annotations.zip` (complete, 4f x 15kp)
- `cvat_upload/exports/court_keypoints_20260707/court_keypoints_metric15_20260707_partial_goodangles_20260709_annotations.zip` (3 usable frames; see PARTIAL_EXPORT_NOTES)
