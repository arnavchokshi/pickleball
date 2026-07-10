# COURT-DATA-2b — emit the partial-row external corpus (unlock the 8 large datasets)

## HARD RULES
- No branches/commits/pushes. No network. Read runs/lanes/court_wave_20260709/DESIGN_RULING.md,
  runs/lanes/court_loader1_20260709/HANDOFF.md (the EXACT schema — consume verbatim), and
  runs/lanes/court_data2_20260709/{HANDOFF.md,keypoint_mappings.json,COURT_DATASET_AUDIT.md} first.
- Protected eval clips: do NOT open any eval_clips/**/labels/* file (the previous lane's hash
  guard artifacts under runs/lanes/court_data2_20260709/ carry the needed hashes — REUSE them).
- data/roboflow_universe_20260706/** READ-ONLY. Outputs in runs/lanes/court_data2b_20260709/.
- Honest reporting; wide suite at end (MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport).

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/build_real_court_corpus.py + tests/racketsport/test_build_real_court_corpus.py (extend)
- runs/lanes/court_data2b_20260709/**
Do NOT touch train_court_keypoint_heatmap.py (just landed masked-loader support — consume only).

## MISSION
Extend build_real_court_corpus.py with a partial-row emission mode producing a SECOND corpus at
runs/lanes/court_data2b_20260709/real_court_corpus_partial/:
1. Include every audited dataset whose mapping verdict is direct or partial-planar per
   keypoint_mappings.json (Xuann/Testworkspace/Stump/Necromancer/Nigh/Syncz 12-pt;
   n-do-tran/ping-pong 12-pt + 2 net endpoints; chetan full-15 stays too, emitted with its
   direct rows). Unmapped keypoints = JSON null exactly per the LOADER-1 handoff schema.
   Items use status "reviewed_external_dataset" (chetan rows INCLUDED in this corpus also get
   external status — owner-independent bucket stays clean).
2. Keep all guards from the parent lane: exact-dup removal across datasets, eval/harvest hash
   guard REUSED from parent artifacts, license fail-closed quarantine, held-out source denylist.
3. Per-dataset source_group provenance; emit a split_proposal.json BY DATASET: >=2 datasets as
   validation (choose for viewpoint diversity; document why), rest train.
4. Emission-correctness overlays: render 5 overlays PER INCLUDED DATASET from the EMITTED
   corpus rows (not the audit mappings) proving null-masking + coordinates survived emission.
5. Loader-contract proof: import the updated loader from train_court_keypoint_heatmap.py,
   load the partial corpus root, report row count, labeled-keypoint histogram, 0 schema errors.
   Also prove one partial row round-trips through court_keypoint_label_rows with nulls intact.
## ACCEPTANCE
- A1: partial corpus >= 800 rows from >= 5 datasets (honest counts if reality differs; report per-dataset).
- A2: labeled-keypoint histogram reported (12/14/15 buckets); ZERO fabricated points (spot-proof: net
  channels null for 12-pt datasets in 5 random sampled rows printed in report).
- A3: loader proof green with the NEW loader; overlays rendered per dataset.
- A4: split_proposal.json exists with >=2 val datasets + rationale.
- A5: focused tests green; wide suite failures==0 or proven pre-existing; write scope clean.
## BEST-STACK DELTA
Expected (c) NO stack delta (data prep). State explicitly.
## REPORT
Structured report per schema + HANDOFF.md with corpus root, row counts, split, and the exact
--real-root invocation the training lane should use.
