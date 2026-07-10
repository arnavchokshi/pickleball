# LANE w7_auditstratum_20260709 — build the uniform-random SCRATCH-LABEL audit task (owner labeling ask #1)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. PROTECTED SOURCES EXCLUDED ABSOLUTELY: no frames from pwxNwFfYQlQ / vQhtz8l6VqU / outdoor_webcam_iynbd / indoor_doubles_fwuks. NO PRELABELS anywhere in the package — the entire point is scratch labeling (no prelabel zips, no seeded annotations, nothing prefilled). Do NOT touch other lanes' files. Artifacts under runs/lanes/w7_auditstratum_20260709/ + a CVAT package dir under cvat_upload/w7_audit_stratum_20260709/.

## OBJECTIVE (RULINGS R2b structural fix; fires the post-3k labeling pivot)
Build ONE CVAT images-task package of ~350 frames sampled UNIFORMLY AT RANDOM for the owner to label from scratch:
1. SAMPLING FRAME UNIVERSE: all frames of all rally videos across the 6 legal harvest sources (data/online_harvest_20260706/rallies/**) — uniform over the frame universe (weight by video length), seeded RNG (seed=20260709) for reproducibility. ~350 frames. Record the exact (source, rally, frame_idx) manifest + md5. EXCLUDE frames already in the reviewed corpus (runs/lanes/w7_ballingest4_20260709/reviewed_corpus/ manifest) so every row is NEW information; report how many collisions were resampled.
2. EXTRACT frames at native resolution (ffmpeg, the labelpack machinery in runs/lanes/w6_labelpack_20260708/build_labelpack.py is the proven extractor pattern — reuse its frame-naming convention so ingest works unchanged: <source>__<rally>__abs_<frameidx>.png).
3. PACKAGE: one CVAT-images task zip (images only, no annotations) + a TASK_README.md for the owner (what this set is, why no prefills, the same ball/absent conventions as always, visibility dropdown OPTIONAL per their w6 ruling). Name: w7_audit_stratum_uniform350.
4. IMPORT PREP: emit the exact import command using scripts/racketsport/import_w6_labelpack_tasks.py (the idempotent importer) with --dry-run first — DO NOT run the import (no localhost in your sandbox; the manager runs it).
5. INGEST COMPATIBILITY: verify (by construction + a unit check) that a future export of this task flows through ingest_owner_ball_labels.py unchanged, and that rows will carry provenance class "scratch" trivially (no prelabels exist to match against).

## SELF-VERIFICATION
Determinism (rerun sampling = identical manifest), extraction spot-checks (N=5 frames decode + dimensions), package zip integrity, focused tests for anything you touch. Disk: frames ~350 x ~1-2MB ≈ <1GB — fine.

## REPORT
Self-write runs/lanes/w7_auditstratum_20260709/report.json (lane_report.schema.json structure): sampling manifest md5, per-source distribution, collision count, package path + size, the exact manager import command, honest_issues, next.
