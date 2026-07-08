# LANE w6_labelingest_20260708 — owner-label ingest + LoSO OUTDOOR fold (wave-6 queue #1, CRITICAL PATH)
# STATUS: STAGED-HOT — dispatch ONLY when >=1 owner export zip exists under cvat_upload/exports/w5_labelpack_20260708/ (or a w6 export dir). Manager fires this.

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes. Working-tree changes only in your OWNED FILES. Manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or runs/manager/ boards — proposed bullet text goes in your report.
- Protected eval data: Outdoor/Indoor HELD-OUT labels are NEVER touched (no ledger row exists). The owner CVAT exports you ingest are TRAINING-side outdoor sources (73VurrTKCZ8 / Ezz6HDNHlnk etc.), NOT the held-out clips — verify by construction: assert no ingested frame maps to any held-out source id; the w5_labelpack protected patterns (pwxNwFfYQlQ, vQhtz8l6VqU, outdoor_webcam_iynbd*, indoor_doubles_fwuks*) must be absent from every ingested row. Any hit = STOP.
- Honest reporting; PASS with full_suite.failed>0 not proven pre-existing = rejected.
- .venv/bin/python; MPLBACKEND=Agg on wide runs.
- Artifacts under runs/lanes/w6_labelingest_20260708/ ONLY. Other lanes' run dirs READ-ONLY.
- HELD-OUT SHOT DISCIPLINE (verbatim from the wave-6 boot prompt): no held-out anything from this lane. This lane produces INTERNAL-val machinery only.

## FILE OWNERSHIP
- OWNED: the reviewed-corpus build outputs (a NEW dated corpus dir — do not overwrite cvat_upload/exports/harvest_review_20260707/), any ingest-converter script (new, under scripts/racketsport/ or your lane dir per house pattern), LoSO fold config/manifest additions, their tests, runs/lanes/w6_labelingest_20260708/**.
- READ-ONLY: cvat_upload/exports/** (owner exports; never modify), runs/lanes/w5_ballretrain_20260707/** (the corpus/LoSO/bridge-scoring precedent — your measurement recipe), cvat_upload/w5_labelpack_20260708/package_manifest.json (frame->clip->source mapping).
- DO NOT TOUCH: process_video.py, remote_body_dispatch.py, ball_arc_solver.py, CAPABILITIES.md, web/replay/**, ios/**.

## OBJECTIVE
Ingest the owner's exported CVAT session annotations into the reviewed corpus, rebuild it, add the OUTDOOR fold to LoSO internal-val, and prepare re-scoring of the wave-5 aligned candidates (control / stage1_official / seed_official from w5_ballretrain) through the bridge in OFFICIAL mode + LoSO-mean with the new fold. This is what makes candidate selection inversion-resistant. Current corpus: 486 reviewed rows (cvat_upload/exports/harvest_review_20260707/, 268 pos + 218 reviewed-absent per w5_ballretrain spec).

## THE DESIGN (pinned)
1. Convert each export zip (CVAT for images 1.1) to reviewed rows in the SAME schema the 486-row corpus uses (locate the exact consumer in the w5_ballretrain seed pipeline and match it field-for-field; visibility_level clear/partial/full/out_of_frame semantics preserved; BlurBall convention noted in provenance).
2. Rebuild the reviewed corpus as a NEW dated corpus (union of the 486 + new rows), with per-source + per-session counts and md5 manifest.
3. LoSO: add the OUTDOOR fold(s) so leave-one-source-out covers the newly-labeled outdoor sources; emit the fold manifest; unit test that no fold's val rows appear in its train rows (the leave-one-source-out VALIDATION bug class from the deep review is the thing this kills).
4. Prepare (do NOT run) the GPU re-score: the exact commands to score the three banked wave-5 candidates through the bridge OFFICIAL + LoSO-mean with the new fold, per the w5_ballretrain recipe (control row MANDATORY). Report them for the manager's GPU errand.
5. Throughput bookkeeping: report total reviewed frames after ingest vs the >=10-20k held-out-shot bar (owner does ~240 frames/hr).

## ACCEPTANCE
1. Every export zip ingested with zero dropped frames unaccounted (per-zip row counts reconcile with CVAT task sizes; explicit skip reasons otherwise).
2. Protected-pattern scan of ingested rows: all NO_MATCH.
3. Corpus rebuild deterministic (two runs byte-identical manifest) + counts table (per source, per class, per visibility level).
4. LoSO fold-disjointness test green; fold manifest lists the OUTDOOR fold.
5. Tests for any new converter CLI (scaffold-index reference test same-lane); FULL wide suite green if repo source touched.
6. The GPU re-score command block, with the control row named.

## KILL / STOP CRITERIA
- Export schema surprises (CVAT version drift, missing visibility attribute) -> ingest what is clean, report the exact drift with a sample; never guess label semantics.
- If ingested outdoor rows < 300 (a tiny partial session), still complete the corpus + folds but flag SMALL-N in the report — the manager decides whether re-scoring is worth the GPU errand yet.

## REPORT (schema-enforced)
objective_result; full_suite; counts tables; fold manifest path; GPU re-score command block; HONEST ISSUES; proposed BUILD_CHECKLIST bullet; NEXT.
