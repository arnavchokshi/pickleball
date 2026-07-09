# LANE w7_ballingest2_20260709 — ingest owner ball-label export (class-G wake #2, w6-labelpack session-01)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. PROTECTED SCAN FIRST: refuse any pwxNwFfYQlQ / vQhtz8l6VqU / outdoor_webcam_iynbd / indoor_doubles_fwuks material (expect NO_MATCH; the w6 pack is harvest/owner sources). Do NOT touch: process_video.py + best_stack.json (tierprov), gate_check_body_decode.py (harness lane), import_w6_labelpack_tasks.py (landed this wave — read-only). The RUNNING ball retrain lane is pinned to corpus md5 37a5d43ab537a15bd12d382bb882a5fe — you build the NEXT corpus revision; do NOT modify runs/lanes/w6_labelingest_20260708/reviewed_corpus in place (other lanes' dirs are read-only evidence). Artifacts under runs/lanes/w7_ballingest2_20260709/ only.

## OBJECTIVE
Ingest cvat_upload/exports/w6_labelpack_20260708/w6_ball_sst_ball_session_01_20260708_annotations.zip with the STANDING wave-6 converter (scripts/racketsport/ingest_owner_ball_labels.py — CVAT images 1.1 path, blur-streak-center convention):
1. Protected-pattern scan (report NO_MATCH proof).
2. Full ingest accounting: images in export, rows added, skipped-with-reason (multi-box / duplicates / conflicts), per-source + per-class + per-visibility counts.
3. Deterministic NEW corpus build: base = the 1,121-row corpus (read-only from w6_labelingest lane dir) + new rows -> runs/lanes/w7_ballingest2_20260709/reviewed_corpus/ with byte-identical-on-rerun manifest + NEW md5. Report total rows (1121 -> N).
4. Rebuild LoSO fold manifest on the new corpus (all-disjoint proof; OUTDOOR fold present).
5. PREPARE-ONLY the GPU re-score command block for the new corpus (control + current candidates incl. any w7 retrain winners once known — parameterize the checkpoint list; bash -n clean). NO GPU dispatch — the manager sequences scoring after the running retrain lane reports.

## SELF-VERIFICATION
Converter tests + your accounting tests if you extend anything; determinism rerun proof; MPLBACKEND=Agg focused suite. Fix what you introduce; prove pre-existing at HEAD.

## REPORT
Self-write runs/lanes/w7_ballingest2_20260709/report.json (lane_report.schema.json structure): protected-scan row FIRST, accounting rows, corpus delta (1121->N w/ new md5), LoSO proof, BEST-STACK DELTA (none expected — data only), honest_issues, next.
