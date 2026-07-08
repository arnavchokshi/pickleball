# LANE w6_labelpack_20260708 — package Phase-B 24 clips into owner CVAT labeling sessions (wave-6 queue #7)

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes from this lane. Working-tree changes only in your OWNED FILES. The manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or any runs/manager/ board — put your proposed dated bullet TEXT in your report; the manager books it.
- Protected eval data: Outdoor/Indoor held-out material must NEVER appear in any package. You MUST re-run the programmatic protected-material check (precedent: runs/lanes/w5_labelpack_20260708/protected_programmatic_check.txt) over every package you produce, with at MINIMUM these patterns, all required NO_MATCH: `pwxNwFfYQlQ`, `vQhtz8l6VqU`, `outdoor_webcam_iynbd_1500_long_high_baseline`, `indoor_doubles_fwuks_0500_long_mid_baseline`, `03_outdoor_webcam_iynbd`, `04_indoor_doubles_fwuks`. Any MATCH = STOP, report, do not ship the package.
- Honest reporting: HONEST ISSUES unsoftened. PASS with unexplained test failures = rejected lane.
- Use .venv/bin/python. Any new CLI ships its scaffold-index/direct-CLI reference test same-lane. If any new doc/zip trips the doc-allowlist/truthful-capabilities guardrail tests, register it same-lane (precedent commit 5b8234748 registered the owner runbook + local-only labelpack zips).
- Artifacts go under runs/lanes/w6_labelpack_20260708/ and cvat_upload/w6_labelpack_20260708/ ONLY. Other lanes' run dirs are READ-ONLY evidence.

## FILE OWNERSHIP (exclusive this wave)
- OWNED: cvat_upload/w6_labelpack_20260708/** (new), runs/lanes/w6_labelpack_20260708/**, optionally ONE new owner-doc file cvat_upload/OWNER_SESSION_W6_20260708.md.
- READ-ONLY evidence: runs/lanes/w5_closeproof_20260708/phase_b_predictions/** (the 24 clip prediction dirs), runs/lanes/w5_labelpack_20260708/** (precedent scripts + manifests), cvat_upload/w5_labelpack_20260708/** (precedent packages/manifests), data/** as needed for frames.
- DO NOT TOUCH: process_video.py, scripts/racketsport/remote_body_dispatch.py, threed/racketsport/ball_arc_solver.py, CAPABILITIES.md, scripts/racketsport/train_ball_stage2.py, web/replay/**, ios/** (other lanes / other sessions own these).

## OBJECTIVE
Phase-B predictions landed for 24 new rally clips (runs/lanes/w5_closeproof_20260708/phase_b_predictions/, md5-verified, 24 dirs). Package these clips' teacher/student disagreement rows into CVAT-importable labeling sessions exactly per the w5_labelpack precedent, so the owner's labeling queue extends from 16/40 to 40/40 clips and covers all 6 non-heldout sources. The owner labels ~240 frames/hr; sessions are 640 frames each (~2.7h per session).

## EVIDENCE TO READ FIRST
1. runs/lanes/w5_labelpack_20260708/build_labelpack.py + package_manifest.json + import_w5_labelpack_tasks.py + protected_programmatic_check.txt — the proven pipeline you are extending (copy/adapt INTO your lane dir; do not edit the w5 lane's files).
2. cvat_upload/w5_labelpack_20260708/package_manifest.json + validation_report.json — the manifest/validation shape to reproduce.
3. cvat_upload/OWNER_SESSION_20260708.md — the owner-doc format (session order, error-mix lines, ball convention block, export instructions).
4. runs/lanes/w5_closeproof_20260708/phase_b_predictions/ — the 24 new clips' predictions (your input).

## THE DESIGN (pinned)
- Rebuild/extend the disagreement queue over the 24 new clips using the same disagreement classes as w5 (large-offset / student-only / teacher-only) with the same selection logic build_labelpack.py used.
- Package into 640-frame sessions with per-session error-mix balancing per the w5 precedent. Session ORDER must maximize source diversity: front-load sessions that introduce sources NOT yet covered by the w5 sessions (w5 covered 73VurrTKCZ8=outdoor_day_multicam + Ezz6HDNHlnk=outdoor_night_fenced; there are 6 non-heldout sources total — the NEW sources go first).
- Each session ships images.zip + prelabels_cvat1_1.zip (one editable ball prelabel per frame) under cvat_upload/w6_labelpack_20260708/packages/.
- Produce: package_manifest.json + validation_report.json (same schema as w5), an import script import_w6_labelpack_tasks.py (adapted from w5, pointing at the SAME local CVAT instance/venv the w5 one used), and cvat_upload/OWNER_SESSION_W6_20260708.md (session order, per-session unlock lines, export path cvat_upload/exports/w6_labelpack_20260708/).
- Package ALL disagreement rows available from the 24 clips into sessions (expect several sessions; report the exact count). If total rows exceed ~8 sessions, still package all, ordered strictly by (new-source coverage first, then disagreement-severity).

## ACCEPTANCE (measured, all required)
1. 24/24 Phase-B clips represented in the manifest (or an explicit per-clip exclusion reason, e.g. zero disagreement rows).
2. All 6 non-heldout sources appear in the packaged queue (union of w5 + w6 sessions); report the per-source frame counts.
3. Programmatic protected-material check: NO_MATCH for every pattern listed in HARD RULES across all packages — write protected_programmatic_check.txt in your lane dir.
4. validation_report.json: every packaged frame's prelabel parses and references an existing image entry (same validation the w5 pack ran); zero validation errors.
5. Import script: dry-run mode proof only (do NOT import into CVAT from this lane — a separate local lane does the import; CVAT/network is outside your sandbox). Assert the script's task-name scheme is w6_ball_sst_<session>_20260708 and collides with nothing in the w5 manifest.
6. Tests: run the doc-consistency / truthful-capabilities / scaffold-index test files (the blast radius of new docs+CLI under cvat_upload/); if you touched NO repo source files, the full wide suite is not required — say so explicitly in the report. Register anything the guardrail tests flag, same-lane.

## KILL / STOP CRITERIA
- Any protected-material MATCH -> STOP with the evidence line; do not ship.
- If phase_b_predictions lack a field the w5 queue-builder needs (schema drift) -> build what is buildable, report the exact missing field + affected clips as HONEST ISSUES; do not fabricate rows.

## REPORT (schema-enforced)
objective_result PASS/PARTIAL/BLOCKED vs the 6 acceptance items; full_suite line (which test files ran, passed/failed, failures proven pre-existing or NOT); session table (name, frames, sources, error mix); per-source coverage; HONEST ISSUES; proposed BUILD_CHECKLIST bullet text; NEXT (the import-lane command + owner-doc pointer for the manager).
