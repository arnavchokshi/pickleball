# Task 13 (metric15 court keypoints) — partial export, owner-ruled 2026-07-09

Export: `court_keypoints_metric15_20260707_partial_goodangles_20260709_annotations.zip`
(CVAT task 13 `racketsport_metric15_court_keypoints_20260707_6frames`, server-side export
2026-07-09; task intentionally left "in progress" — owner is DONE with it, do not re-queue.)

OWNER RULING (2026-07-09): most of these 6 frames show courts that are mostly NOT fully
visible, at camera angles too low to be expected/allowed from product users. Only the
good-angle frames were filled, and ONLY those are usable.

## Usable (FULL 15 keypoints) — ingest these 3 only
- frame 0 `73VurrTKCZ8__73VurrTKCZ8_rally_0002__abs_003808.png`
- frame 2 `HyUqT7zFiwk__HyUqT7zFiwk_rally_0001__abs_010195.png`
- frame 5 `zwCtH_i1_S4__zwCtH_i1_S4_rally_0001__abs_003636.png`

## Rejected for angle/visibility (partial points = abandoned, NOT ground truth)
- frame 1 `Ezz6HDNHlnk__...__abs_010677.png` (1 pt)
- frame 3 `_L0HVmAlCQI__...__abs_000509.png` (7 pts)
- frame 4 `wBu8bC4OfUY__...__abs_010248.png` (4 pts)
Do NOT train or eval on these partial frames.

## Overlap with the w5 relabel task (task 18) — task 18 wins
Task 18 export (`../w5_labelpack_20260708/w5_court_kp_relabel_..._annotations.zip`,
complete: 4 frames x full 15 kp) re-labels the same
`HyUqT7zFiwk abs_010195` and `zwCtH_i1_S4 abs_003636` frames. On any conflict, prefer
the task-18 (w5 relabel) coordinates — they are the newer pass.

## Product signal (input-quality gate)
The rejection reason is an OWNER ANGLE POLICY data point: courts not fully visible /
camera too low = below the acceptance bar for user-submitted clips. Relevant to the
input-quality first-class gate from the W6 North Star refresh.
