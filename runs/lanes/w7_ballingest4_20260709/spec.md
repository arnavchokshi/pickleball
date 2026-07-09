# LANE w7_ballingest4_20260709 — ingest owner ball-label batch (ball_session_02 — found un-ingested in exports during watchdog outage; completes the 3k corpus)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. PROTECTED SCAN FIRST on the export: refuse pwxNwFfYQlQ / vQhtz8l6VqU / outdoor_webcam_iynbd / indoor_doubles_fwuks material. OWNER RULINGS BINDING (cvat_upload/exports/w6_labelpack_20260708/SESSION_NOTES_20260709.md): w6-session visibility_level = UNINFORMATIVE (box-position-only; count-but-never-analyze), multi-ball double-boxed frames = skip-and-account, no owner rework. Other lanes' dirs read-only; base corpora read-only. Artifacts under runs/lanes/w7_ballingest4_20260709/ only.

## OBJECTIVE
Ingest cvat_upload/exports/w6_labelpack_20260708/w6_ball_sst_ball_session_02_20260708_annotations.zip (annotations.xml, CVAT images 1.1) with the STANDING converter (scripts/racketsport/ingest_owner_ball_labels.py):
1. Protected scan (NO_MATCH proof + export md5).
2. Full accounting: images, rows added, skips by reason, per-source/class/visibility counts (visibility = accounting only per ruling).
3. Deterministic NEXT corpus revision: base = the 2,388-row corpus (runs/lanes/w7_ballingest3_20260709/reviewed_corpus/, md5 0ae65f014ce26b2ddf8573427c60853d, read-only) + new rows -> runs/lanes/w7_ballingest4_20260709/reviewed_corpus/ + byte-identical-rerun manifest + NEW md5. REPORT THE TOTAL ROW COUNT PROMINENTLY (the owner believes we cross 3,000 — the number decides whether the 3k checkpoint fires tonight).
4. LoSO fold manifest rebuild (all-disjoint proof, OUTDOOR fold present).
5. Refresh the prepared GPU re-score block (38-clip pattern from ballingest2, W7_EXTRA_CANDIDATES hook) against the new corpus paths — prepare-only.

## SELF-VERIFICATION
Converter tests + determinism rerun proof; MPLBACKEND=Agg focused suite; fix what you introduce.

## REPORT
Self-write runs/lanes/w7_ballingest4_20260709/report.json (lane_report.schema.json structure): protected row FIRST, TOTAL CORPUS N headline, accounting, md5s, LoSO proof, BEST-STACK DELTA none, honest_issues, next.
