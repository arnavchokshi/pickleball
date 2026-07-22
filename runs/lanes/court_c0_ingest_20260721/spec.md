# LANE court_c0_ingest_20260721 — C0: CVAT court-image ingest adapter (fixtures now, real export when tasks 88-91 land)

## HARD RULES
- No branches/commits. Read NORTH_STAR_ROADMAP.md §2 (CAL row) and runs/regroup_20260721/
  EXACT_PLAN.md §3.3 C0 + §2.2 first. `IYnbdRs1Jdk` derivatives are PERMANENTLY DENIED
  (same family as strict-protected outdoor_webcam_iynbd). eval_clips/ball protected frames:
  pHash-compare only, never staged.
- Honest reporting; WIDE test suite (MPLBACKEND=Agg, full tests/racketsport), exact counts.
- Artifacts under runs/lanes/court_c0_ingest_20260721/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/ingest_cvat_court_images.py (new)
- tests/racketsport/test_ingest_cvat_court_images.py (new)
Nothing else. Do NOT touch train_court_model_v2.py in this lane.

## OBJECTIVE (EXACT_PLAN C0, verbatim contract)
  .venv/bin/python scripts/racketsport/ingest_cvat_court_images.py \
    --package-manifest cvat_upload/court_diversity_20260712/package_manifest.json \
    --cvat-export cvat_upload/exports/court_diversity_20260712/annotations.zip \
    --deny-source IYnbdRs1Jdk \
    --protected-root eval_clips/ball \
    --out runs/lanes/court_c0_ingest_20260721/real_court_diversity
Emits the trainer's existing `<source>/labels/court_keypoints.json` format (read the format
from the existing court trainer/eval code paths and match it exactly; document which file
defines it). Behavior:
- Deny the 3 IYnbdRs1Jdk frames BY SOURCE ID BEFORE any label read.
- Dense pHash of every remaining image against every protected frame; any hit -> reject + count.
- Freeze the pre-label-inspection holdout partition (8 source groups, verbatim):
  outdoor 1or-bXVM80M, 4qSoA-jwpVM, C5YUQlqZqBY, q3575jnmjJQ;
  indoor A9H6EWfXht0, Se7M6ZKaC4Y, a_HzWrwK6vM, wv3aPJrDwK4.
  The other 19 source IDs are train candidates. Group by original source/channel/venue family;
  NO frame-random split.
- Output exact counts: reviewed / usable / protected-denied / train / holdout / rejected.
- Gate: require >=60 usable train rows across >=15 train source groups AND >=2 valid rows in
  ALL EIGHT holdout groups; else exit nonzero with verdict COURT_DIVERSITY_ROWS_INSUFFICIENT.

## OPERATIONAL REALITY
Tasks 88-91 are being labeled by the owner soon (task 87 first); the export zip does not exist
yet. Build + test on a synthetic CVAT image-task fixture (cover: deny-before-read, pHash
rejection, split freezing, count gate both pass and fail). Before finishing, check ONCE for
the real export; if present, run for real and report real counts; else report
ADAPTER_READY_AWAITING_EXPORT.

## DATA CONTRACT
- Inputs: court_diversity package_manifest.json sha256
  c0243e9146152c5c46b5d0aebca9d571bfd39b6e90b34227d4024d09eabcdd7e. Ledger rows pending
  steward bootstrap. No GPU. Effort cap ~8h.
- End-of-lane number: fixture-gate proof + (if export landed) exact usable/protected/rejected counts.

## CROSS-SIGNAL
Consumes: owner CVAT court labels, source-family lineage from online_harvest_20260712.
Feeds: COURT C1 preview challenger (train_court_model_v2 --real-root), classical-line
comparison harness, data-steward ledger.

## BEST-STACK DELTA
None — data ingest infrastructure. Learned court remains preview-only per plan §2.3.

## MANDATORY STRUCTURED REPORT
objective_result; full_suite counts; HONEST ISSUES; artifacts; the count table; the file that
defines the court_keypoints.json format you matched.
