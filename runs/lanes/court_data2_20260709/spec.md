# COURT-DATA-2 — Roboflow real court-dataset audit + unified real-court corpus

## HARD RULES
- No branches, no commits, no pushes. No network (everything is already local).
- Read NORTH_STAR_ROADMAP.md (CAL rows, §2.3) first.
- Protected eval clips (eval_clips/**) EVAL-ONLY: do not read their labels; they must not
  enter the corpus. Held-out harvest sources pwxNwFfYQlQ / vQhtz8l6VqU excluded everywhere.
- data/roboflow_universe_20260706/** is READ-ONLY. All outputs under runs/lanes/court_data2_20260709/
  except the one new script + test you own.
- Honest reporting; run WIDE suite (MPLBACKEND=Agg, .venv/bin/python -m pytest tests/racketsport)
  at the end; every new CLI ships a direct-CLI reference test same-lane.

## FILE OWNERSHIP (exclusive)
- NEW scripts/racketsport/build_real_court_corpus.py
- NEW tests/racketsport/test_build_real_court_corpus.py
- runs/lanes/court_data2_20260709/**
Do NOT touch any other repo file. (calibrate_harvest_courts.py and the projector belong to the
concurrent court_data1 lane; trainer/gate files belong to a later lane.)

## CONTEXT
data/roboflow_universe_20260706/ holds 67 locally-downloaded Roboflow Universe datasets
(manifest.json has license_as_recorded per project). At least ~14 are pickleball-COURT
datasets, e.g.: xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10 (683/195/156),
stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1 (296),
testworkspace-i8nb1__pickle-court-keypoints__v2 (53/15/8),
pickleball-dl6zm__pickleball-courts-emwra-w8dsr__v1 (84/24/11), gideons__pickleball-court__v1
(84/24/12), plus court dirs under n-do-tran, chetan-rajagiri, luiss-workspace, acmai,
hughs-workspace-plw3g, pickleball-ball-detection (court-keypoints-syncz), necromancer,
ping-pong-paddle-ai, nigh-workspace. Our canonical target = the 15 named PICKLEBALL_KEYPOINTS
in threed/racketsport/court_keypoint_net.py:348-364 (near/far baseline corners+centers x3 each,
net line x3, NVZ lines x6). Real labels the trainer accepts: <clip>/labels/court_keypoints.json
rows with label_status='reviewed' (see scripts/racketsport/train_court_keypoint_heatmap.py
load_real_court_keypoint_labels for the exact schema).

## MISSION
1. AUDIT every court-related dataset dir (search ALL 67 for court/keypoint/line themes, not
   just the named ones): for each -> {dir, task type (keypoints/bbox/segmentation), annotation
   schema (names+counts of categories/keypoints), image counts per split, resolution stats,
   license_as_recorded, viewpoint character (sample 5 images: elevated/low/steep/broadcast/
   portrait), overlap-suspects (identical images across datasets — hash-check a sample),
   USABILITY VERDICT for 15-kp training (direct-map / partial-map / corners-only / unusable)}.
2. KEYPOINT MAPPING: for each usable dataset define an explicit mapping table from its
   category/keypoint names to our 15 canonical names (or the subset it supports). Ambiguous
   orderings (which corner is near-left?) must be resolved by geometric heuristics + visual
   spot-check of >=5 rendered overlays per dataset (save overlays to lane dir). Document
   per-dataset confidence in the mapping. NO GUESSED MAPPINGS: a dataset with unresolvable
   semantics gets verdict 'unusable_ambiguous'.
3. BUILD scripts/racketsport/build_real_court_corpus.py: reads the audited datasets + mapping
   tables (checked into the lane dir as JSON), emits a unified corpus at
   runs/lanes/court_data2_20260709/real_court_corpus/ in the EXACT layout
   train_court_keypoint_heatmap.py accepts as --real-root (one pseudo-clip dir per dataset
   split with labels/court_keypoints.json + labels/court_keypoint_frames/*.png symlinks or
   relative refs — follow the loader's contract precisely; if the loader requires real image
   files, use SYMLINKS not copies, disk is tight). label_status for these rows: 'reviewed'
   only if the dataset is human-annotated; document your choice. Include per-row provenance
   {dataset, split, original_image, license}.
4. DEDUP + LEAKAGE GUARD: exact-dup image hashes removed across datasets; keep a
   near-dup note (same court/different frame is FINE and expected within a dataset —
   record dataset-level source grouping so the TRAINING lane can split BY DATASET).
   Verify none of the corpus images equal any eval_clips or harvest GT frame (hash check
   against those image files WITHOUT reading their label jsons).
5. LICENSE CARD: per-dataset license, attribution string, commercial-use verdict
   (CC BY 4.0 = OK w/ attribution; BY-NC-* = research-only -> EXCLUDE from the default
   corpus, keep in a quarantined list). Aggregate card at lane dir.
6. STATS REPORT: final corpus rows by dataset, keypoint coverage histogram (how many rows
   have all 15 vs subsets), viewpoint distribution summary, split proposal (train/val BY
   DATASET with >=2 datasets held out for val).

## ACCEPTANCE (numbers)
- A1: audit table covers ALL court-related dirs found (>=10 expected); each has a usability verdict.
- A2: >=500 unified corpus rows from >=3 distinct datasets with >=8 of our 15 keypoints mapped
  (if reality falls short, report honest counts — do not force mappings).
- A3: mapping overlays rendered (>=5/dataset) and each usable dataset's mapping confidence stated.
- A4: loader-contract proof: run train_court_keypoint_heatmap.py's load path (import the loader
  function directly in a test or tiny harness) against the corpus root and show it loads N rows
  with 0 schema errors. This is the lane's decisive check.
- A5: license card complete; any non-commercial dataset excluded from default corpus.
- A6: focused tests green; wide suite failures==0 or proven pre-existing; zero writes outside owned files.

## BEST-STACK DELTA (mandatory in report)
Expected (c) NO stack delta (data prep). State explicitly.

## REPORT
Structured report per output schema + HANDOFF.md bullet with corpus size/composition for the
training lane.
