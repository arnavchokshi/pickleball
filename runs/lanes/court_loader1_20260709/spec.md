# COURT-LOADER-1 — masked per-keypoint supervision + honest external label status

## HARD RULES
- No branches/commits/pushes. Read NORTH_STAR_ROADMAP.md CAL rows + runs/lanes/court_wave_20260709/DESIGN_RULING.md (R5) first.
- Protected eval clips EVAL-ONLY (do not read labels; tests use synthetic fixtures).
- DEFAULT BEHAVIOR BYTE-IDENTICAL: existing full-15 'reviewed' rows must load, train, and
  score EXACTLY as before (prove with a regression test that existing fixture rows produce
  identical loss tensors pre/post change).
- Honest reporting; WIDE suite (MPLBACKEND=Agg, .venv/bin/python -m pytest tests/racketsport) at end.
- Artifacts under runs/lanes/court_loader1_20260709/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/train_court_keypoint_heatmap.py
- tests/racketsport/test_train_court_keypoint_heatmap.py
- scripts/racketsport/evaluate_court_keypoint_owner_gate.py + tests/racketsport/test_evaluate_court_keypoint_owner_gate.py (ONLY if gate-side handling is required; keep changes minimal)
Do NOT touch: build_real_court_corpus.py (concurrent lane), calibrate_charuco_device.py, project_court_pseudo_labels.py, any BODY/decode file, threed/racketsport/court_keypoint_net.py.

## CONTEXT
Ruling R2/R5: we will train on external real datasets (Roboflow pickleball court keypoints)
where rows label only a SUBSET of the 15 canonical keypoints (typically the 12 floor points;
the 3 net-top points are off-plane and often unlabeled). Current loader
(court_keypoint_label_rows/_court_keypoint_label_row_from_item) hard-requires all 15 names,
and the loss supervises every channel. We need:

1. **Schema extension (additive):** a row item MAY carry per-keypoint entries of the form
   {"x":..,"y":..} (as today) OR null / {"labeled": false} for unlabeled points. Alternatively
   the item may carry "labeled_keypoints": [names...]. Choose ONE explicit schema, document it
   in the module docstring, and validate fail-loud on anything else. Unlabeled != occluded:
   an occluded-but-known point stays labeled with its coordinates; unlabeled means NO supervision.
2. **New label_status 'reviewed_external_dataset'** added to ACCEPTED_ITEM_STATUSES: human-
   annotated third-party datasets. Training accepts it; every summary/report that counts
   labels_independent_human_frames must count external rows SEPARATELY (new field, e.g.
   labels_external_dataset_frame_count). The owner gate's 'independent' buckets must NOT
   include external rows.
3. **Loss masking:** per-row per-keypoint mask tensor; unlabeled channels contribute ZERO to
   heatmap loss (both peak and background terms) and are excluded from metric aggregation.
   Visibility-head targets (if trained) are also masked for unlabeled points.
4. **Eval masking:** evaluate_checkpoint_against_real_labels skips unlabeled keypoints in
   per-row error lists (count reflects only labeled points); reports keypoint_count per row.
   The owner gate path (all-15 rows) must be provably unchanged.
5. **Tests:** (a) regression: existing full-15 fixture -> identical loss values pre/post
   (hardcode expected values or compute via the old path in-test); (b) masked row -> zero
   gradient/loss contribution on unlabeled channels (construct a case where the unlabeled
   channel's prediction is garbage and assert loss unaffected); (c) loader rejects malformed
   partial schemas fail-loud; (d) external status counted separately in the training summary;
   (e) direct-CLI smoke: a 2-epoch tiny training run on a fixture mixing full-15 'reviewed'
   + partial 'reviewed_external_dataset' rows completes and writes a checkpoint.

## ACCEPTANCE
- A1: all 5 test classes above green; existing test file still green.
- A2: byte-identical default proof (regression test) explicitly named in report.
- A3: wide suite failures==0 or proven pre-existing (list them with evidence).
- A4: module docstring documents the partial schema + status semantics.
- A5: zero writes outside owned files + lane dir.

## BEST-STACK DELTA (mandatory in report)
Expected (c) NO stack delta (training infrastructure; no default model/policy change). State explicitly.

## REPORT
Structured report per schema; HANDOFF.md bullet stating the exact partial-row schema chosen
(the corpus-builder lane consumes it verbatim).
