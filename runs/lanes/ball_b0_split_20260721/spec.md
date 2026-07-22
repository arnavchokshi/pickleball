# LANE ball_b0_split_20260721 — B0: source-held scratch judge splitter + lineage reconciliation

## HARD RULES
- No branches/commits. Read NORTH_STAR_ROADMAP.md §2 and runs/regroup_20260721/EXACT_PLAN.md
  §3.2 B0 + §2.2 quarantines first. 4 protected clips (eval_clips/ball/*) EVAL-ONLY — this lane
  may hash/pHash them for collision checks but never train-stage them.
- Honest reporting; WIDE test suite (MPLBACKEND=Agg, full tests/racketsport, no -x), exact counts.
- Artifacts under runs/lanes/ball_b0_split_20260721/. Other lanes' run dirs READ-ONLY.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/build_ball_regroup_split.py (new)
- tests/racketsport/test_build_ball_regroup_split.py (new)
Nothing else.

## OBJECTIVE (EXACT_PLAN B0, verbatim contract)
Deterministic splitter:
  .venv/bin/python scripts/racketsport/build_ball_regroup_split.py \
    --reviewed-root runs/lanes/w7_ballingest4_20260709/reviewed_corpus \
    --scratch-package cvat_upload/w7_audit_stratum_20260709/package_manifest.json \
    --scratch-export cvat_upload/exports/w7_audit_stratum_20260709/w7_audit_stratum_uniform350_annotations.zip \
    --holdout-source HyUqT7zFiwk --holdout-source Ezz6HDNHlnk \
    --out runs/lanes/ball_b0_split_20260721/split
Must: compare every final label to its original prelabel/package lineage and emit per-row
lineage class `scratch` | `corrected_prelabel` | `confirmed_prelabel`; correct the old "LoSO"
semantics from per-clip to PARENT-SOURCE grouping; whole-source split with holdouts
HyUqT7zFiwk (expect 100 scratch val rows, indoor court-level) and Ezz6HDNHlnk (expect 67,
outdoor night/fenced); training = all 3,026 reviewed rows minus every row from those two
sources (expect 2,066 old rows) plus remaining 183 scratch rows from four sources; zero rows
from the four protected eval clips. confirmed_prelabel rows train-only at explicit low weight,
NEVER eval; every evaluation row must be `scratch`.

## CHECKS (the B0 gate — implement as hard assertions in the tool's output)
- 350/350 scratch-package images reconciled; train/val source intersection EMPTY;
- protected collision count ZERO against EVERY protected frame (all frames of all 4
  eval_clips/ball clips + the 2 court-keypoint additions), not the old 35-frame sample;
- every eval row lineage==scratch; metrics fields reported separately for HyU and Ezz.
- Any violation -> exit nonzero with verdict string BALL_NO_CLEAN_JUDGE.

## OPERATIONAL REALITY
The owner is labeling task 87 NOW; the scratch export zip may appear at
cvat_upload/exports/w7_audit_stratum_20260709/ during your run. Sequence:
1. Build + test the splitter against a SYNTHETIC fixture export zip (construct a small CVAT
   images-format fixture in tests; cover scratch vs corrected vs confirmed lineage cases,
   holdout separation, protected-collision refusal).
2. Run the REAL lineage reconciliation now (reviewed_corpus + package manifests exist):
   produce exact per-source and per-lineage-class counts for the 3,026-row corpus
   (expect per-source 73VurrTKCZ8=397, Ezz6HDNHlnk=400, HyUqT7zFiwk=560, _L0HVmAlCQI=554,
   wBu8bC4OfUY=555, zwCtH_i1_S4=560 — verify, report actual).
3. Before finishing, check ONCE whether the real export zip exists; if yes run the full
   splitter for real and report the real split counts; if no, report SPLITTER_READY_AWAITING_EXPORT.

## DATA CONTRACT
- Inputs: w7_ballingest4 ingest_report.json sha256
  c4200032e86f912d68adfefdb27118e48b2ed0673fee6122a21df638c601968d; w7_audit_stratum
  package_manifest.json sha256 a04fd956ac56c16130643a79344c21298dec6a3e69f507e0788e996a927a9a55.
  Ledger rows pending steward bootstrap.
- Utilization delta: 3,026 reviewed rows -> lineage-classified; 350 scratch frames -> judge
  candidates. No GPU. Effort cap ~6h.
- End-of-lane number: exact lineage counts by class and source; split counts if export landed.

## CROSS-SIGNAL
Consumes: CVAT prelabel lineage, package manifests. Feeds: BALL B2 matched A/B (the judge),
ball_loso_validation.py parent-source mode, data-steward ledger.

## BEST-STACK DELTA
None — data/judge infrastructure.

## MANDATORY STRUCTURED REPORT
objective_result vs checks; full_suite counts; HONEST ISSUES; artifacts; lineage-count table;
whether real export was ingested.
