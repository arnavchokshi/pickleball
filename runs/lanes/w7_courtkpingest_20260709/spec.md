# LANE w7_courtkpingest_20260709 — ingest owner COURT-KP RELABEL export + owner-gate re-score (class-G wake)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. PROTECTED-DATA CHECK FIRST: the export must contain ONLY HyUqT7zFiwk / zwCtH_i1_S4 material (harvest sources, internal) — if ANY Outdoor/Indoor protected pattern appears (pwxNwFfYQlQ, vQhtz8l6VqU, outdoor_webcam_iynbd, indoor_doubles_fwuks), STOP immediately and report; do not ingest those rows. Do NOT touch: scripts/racketsport/process_video.py + configs/racketsport/best_stack.json (tierprov lane owns NOW), scripts/racketsport/import_w6_labelpack_tasks.py + web/replay vite config (micro-debt lane), any court model training. NO board/doc edits (manager books results). Artifacts under runs/lanes/w7_courtkpingest_20260709/ only.

## OBJECTIVE
The owner just exported cvat_upload/exports/w5_labelpack_20260708/w5_court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_20260708_annotations.zip — the long-requested court-keypoint RELABELS for the two harvest sources. Do:
1. Inspect + convert the export with the standing court-kp handling (read how cvat_upload/exports/court_keypoints_20260707/ was ingested — reuse that converter path; if the relabel export's format needs a converter tweak, that converter + its tests are yours to edit).
2. Reconcile vs the previous court-kp GT for these two sources: rows added/changed/dropped, per-source counts, any frames where the relabel MOVED a keypoint by >5px (distribution summary) — this is the owner-correction signal.
3. Re-score the court keypoint owner gate (scripts/racketsport/evaluate_court_keypoint_owner_gate.py) for BOTH sources against the CORRECTED GT for every banked court-model candidate the harness already knows (at minimum the current promoted court calibration default + the CALV1 candidates it can reach from banked artifacts — read the CLI's --help and the calv1 lane dirs for what is scoreable WITHOUT a GPU or training). Produce before-vs-after tables (old GT vs corrected GT) per candidate.
4. Rebuild deterministic corpus/GT manifests (md5) so downstream lanes can pin this GT revision.

## SELF-VERIFICATION
Converter tests (if touched) + evaluate CLI reference test + a determinism rerun (byte-identical manifest). MPLBACKEND=Agg. Fix what you introduce; prove pre-existing at HEAD.

## REPORT
Self-write runs/lanes/w7_courtkpingest_20260709/report.json (lane_report.schema.json structure): protected-scan result FIRST, ingest accounting, relabel-delta distribution, before/after gate tables per candidate w/ exact metric keys, BEST-STACK DELTA (expected none — GT/scoring only; if the corrected GT flips which court candidate looks best, SAY SO as a PENDING-evidence note, do not touch the manifest), honest_issues, next.
