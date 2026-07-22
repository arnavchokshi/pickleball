# LANE person_p1_roboflow_20260721 — P1: audit + export the eligible Roboflow person subset (CPU, runs for real today)

## HARD RULES
- No branches/commits. Read NORTH_STAR_ROADMAP.md §2 (TRK/DATA rows) and
  runs/regroup_20260721/EXACT_PLAN.md §3.4 P1 + §2.2 first.
- QUARANTINES (binding): exclude Roboflow `testing-esifc/pickle-ball-labeling-mff1d`
  (BY-NC-SA 4.0); exclude ALL adjacent-sport person rows (15,469, tennis-dominated); the
  11,459 person boxes on the 4 protected clips are eval-only and untouchable; do NOT weaken
  the existing CVAT exporter's protected-data refusal — this is a NEW Roboflow-index exporter.
- Honest reporting; WIDE test suite (MPLBACKEND=Agg, full tests/racketsport), exact counts.
- Artifacts under runs/lanes/person_p1_roboflow_20260721/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/export_roboflow_person_yolo_dataset.py (new)
- tests/racketsport/test_export_roboflow_person_yolo_dataset.py (new)
Nothing else.

## OBJECTIVE (EXACT_PLAN P1, verbatim contract — this lane RUNS it for real)
  .venv/bin/python scripts/racketsport/export_roboflow_person_yolo_dataset.py \
    --index data/roboflow_universe_20260706/aggregated/subset_indexes/person_index.json \
    --bucket core_pickleball \
    --exclude-source testing-esifc/pickle-ball-labeling-mff1d \
    --val-source pickleball-od8al/pickleball-version2 \
    --test-source hemel/pickleball-cedmo \
    --group-forks --source-balanced \
    --audit-samples-per-source 15 \
    --protected-root eval_clips/ball \
    --out runs/lanes/person_p1_roboflow_20260721/roboflow_person
Starting eligible inventory to verify (report actual): 15,312 images / 47,044 boxes /
14 CC BY 4.0 sources. Requirements:
- keep fork families together; whole-source train/val/test splits (named val/test sources);
- never pull adjacent-sport rows or the NC source;
- emit YOLO data.yaml + images/labels layout (symlink or copy — state which and why);
- stage 15 audit samples per source with drawn boxes into an easily-reviewable page/dir for
  the human precision/recall card (the >=98% box precision / >=95% recall judgment belongs to
  the human reviewer — stage it, do NOT self-certify);
- EXHAUSTIVE protected-frame collision check (pHash + embedding or pHash at multiple scales)
  of every exported image against every protected frame; required result ZERO collisions;
- automatable retention check: after exclusions, >=5,000 images across >=8 train source
  groups, else verdict PERSON_RF_POOL_TOO_THIN.

## ACCEPTANCE NUMBERS (report all)
Exact train/val/test image+box counts by source; fork-family grouping table; collision
count (must be 0); retained-source count; audit-pack path + per-source sample counts.
P(fail) is priced at 50% — an honest PERSON_RF_POOL_TOO_THIN or annotation-quality negative
is a valid lane outcome; do not massage exclusions to pass.

## DATA CONTRACT
- Input: person_index.json sha256
  4dbf5c8e7ca328b2a05743f525b0f6e4cbf3b50c5c074da8a436a2e83b358135. Ledger rows pending
  steward bootstrap. CPU only, no GPU. Effort cap ~10h.
- Utilization delta: roboflow person subset unused -> export-staged (or rejected/thin verdict).
- End-of-lane number: the count table + collision count + retention verdict.

## CROSS-SIGNAL
Consumes: Roboflow aggregation indexes, protected-frame hashes. Feeds: PERSON P2 YOLO26m
data-domain arm, P0 human compare card (as its candidate-side), data-steward ledger.

## BEST-STACK DELTA
None — dataset export. Any model change is P2's, and P2 is gated on P0's human judge.

## MANDATORY STRUCTURED REPORT
objective_result; full_suite counts; HONEST ISSUES; artifacts; the acceptance-numbers table.
