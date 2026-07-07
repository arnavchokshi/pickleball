# LANE w3_courtkp_prep_20260707 — court-keypoint labeling task set (1 frame × 8 harvest sources, metric-15pt)

## HARD RULES
- You are Codex lane `w3_courtkp_prep_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- You own NO repo source files. Writes ONLY under runs/lanes/w3_courtkp_prep_20260707/ and cvat_upload/court_keypoints_20260707/ (the task-set output). No git/branch/commit/push; commit_manifest.md in lane dir. No network; do NOT talk to the local CVAT instance (you produce packages + an import script; a human/Sonnet runs the import).
- Held-out: pwxNwFfYQlQ / vQhtz8l6VqU sources are EXCLUDED from frame selection (owner queue says 8 sources — the harvest has 8 non-held-out sources; verify from the harvest manifest/corpus card and report the exact source list).
- Honest reporting. `.venv/bin/python`. Read first: BUILD_CHECKLIST last ~5 bullets; the labelfactory lane scripts (runs/lanes/w3_labelfactory_20260707/create_project_and_tasks.py + build_and_import_prelabels.py) as the proven CVAT-import reference; the metric-15pt calibration convention (find it: the manual/metric-15pt calibration path in the court code + its schema — cite where the 15 points are defined).

## CONTEXT
Owner-queued (next owner session, ~10 min of owner time): a court-keypoint task set to feed P4 court auto-find viewpoint diversity. One frame per harvest source video, labeled with our metric-15pt convention.

## OBJECTIVE
1. **Frame selection**: for each of the 8 non-held-out harvest sources, select ONE frame from its rally clips with the court maximally visible (simple defensible heuristic — e.g., from the longest rally clip, a frame with high court-line visibility / no near-camera occlusion; document the heuristic; export the frame as PNG for preview). Record clip + absolute frame index + preview path per source.
2. **Label spec**: a CVAT label schema for the 15 metric points (names/order EXACTLY per our metric-15pt convention — cite the source-of-truth file), point-type labels, with per-point occlusion attribute if our convention has one (report what the convention supports; do not invent).
3. **Task packages**: CVAT-importable packages (one task per source, single frame each — or one 8-frame task if that is materially simpler for the owner; you choose, justify in one sentence) + `import_court_kp_tasks.py` script (modeled on the labelfactory scripts, reading the same credentials file) that a human/Sonnet runs against localhost CVAT in one command.
4. **Owner instructions**: OWNER_COURT_KP_GUIDE.md in the task-set dir: what each of the 15 points is (with the convention's diagram/description restated), expected ~10 min total, export instructions mirroring the harvest-review flow (export to cvat_upload/exports/court_keypoints_<date>/ from the UI or the documented CLI).
5. Validation: packages schema-checked with the repo's CVAT tooling where applicable; frame indices verified in-range; held-out assert; preview PNGs render (8/8).

## ACCEPTANCE
8/8 sources × 1 frame selected w/ previews; label spec matches the metric-15pt convention with citation; packages + one-command import script + owner guide complete; held-out clean; no source edits.

## STRUCTURED REPORT
objective_result; acceptance table; the 8 selections (source/clip/frame); convention citation; HONEST ISSUES; NEXT; commit_manifest path; BUILD_CHECKLIST bullet DRAFT.
