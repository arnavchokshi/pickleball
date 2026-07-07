# LANE w3_cvatsparse_20260707 — sparse-review CVAT import/benchmark support + teacher decision table

## HARD RULES
- You are Codex lane `w3_cvatsparse_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branch/commit/push; commit_manifest.md in your lane dir.
- OWNED files (edit only these + new tests + your lane dir + runs/cvat_imports/harvest_review_20260707/ re-registration):
  `threed/racketsport/cvat_video.py`, `threed/racketsport/ball_cvat_benchmark.py`, `threed/racketsport/ball_cvat_bounce_labels.py` (only if the import path requires), `scripts/racketsport/import_cvat_video_annotations.py`, the corresponding tests (tests/racketsport/test_cvat_video_import.py, test_ball_cvat_benchmark*.py etc.).
- FENCED (do not edit): everything the phasefix/verify touched (foot_contact.py, placement.py, foot_lock_solver.py, footlock.py, body_grounding_refine.py, worldhmr.py, body_grounding_quality.py), frame_rating.py + body_mesh_readiness.py (meshfallback lane STILL RUNNING), process_video.py, camera_motion files, remote_body_dispatch.py, roboflow_corpus.py, ball_tracknet_cvat_dataset.py, train_ball_pretrain.py.
- Protected data: 4 eval clips EVAL-ONLY; held-out IDs nowhere. The harvest review labels are NOT held-out (roles in runs/cvat_imports/harvest_review_20260707/set_manifest.json).
- Honest reporting; WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --ignore=tests/racketsport/test_court_finding_technology_benchmark.py` (NOTE the test_ prefix — this is the correct filename) + focused suites; classify residuals REAL / PRE-EXISTING / SANDBOX-SUSPECT / CROSS-LANE-SUSPECT (meshfallback + phaseverify run concurrently).
- importorskip("torch"); new CLI surface ⇒ direct-CLI reference test; no new root .md. `.venv/bin/python`.
- Read FIRST: runs/lanes/w3_reviewimport_20260707/{REPORT.md,baseline_adapter_gap.json,report.json} + its lane-local harvest_review_import.py (the normalization it had to do = your bug list); runs/cvat_imports/harvest_review_20260707/set_manifest.json; cvat_upload/exports/harvest_review_20260707/README.md.

## CONTEXT
The owner's harvest review set is SPARSE: ~81 sampled frames per clip (absolute source-frame indices; frame_step tasks + one cloud pair), ~268 ball boxes across 6 clips, registered at runs/cvat_imports/harvest_review_20260707/. The existing import/benchmark path assumes densely-reviewed tasks: it treated all 70,229 unsampled frames as reviewed hidden negatives (expected: 218 sampled hidden frames), so every candidate score is invalid. The reviewimport lane worked around import bugs with lane-local XML normalization — the repo path must now handle these natively.

## THE DESIGN (manager-ruled scoring semantics — these are the rules, implement exactly)
1. **Sparse review contract**: imported review artifacts carry `reviewed_frame_indices` (absolute source-frame indices actually shown to the labeler — derive from task metadata/frame_step/selection JSON; for this set the authoritative sources are the export XML meta + runs/lanes/wave2_integration_20260706/harvest_ingest/cvat_review_selection.json; reconcile and record which was used). A frame not in the set is UNREVIEWED: it says NOTHING about ball presence.
2. **Benchmark on reviewed frames only**: recall = matched human boxes / human boxes (on reviewed frames, excluding out_of_frame-tagged); precision = matched candidate detections ON REVIEWED FRAMES / candidate detections ON REVIEWED FRAMES; detections on unreviewed frames are EXCLUDED from precision and never counted as FP; hidden (reviewed, no ball) frames contribute candidate-FPs normally. Report per-clip and pooled {recall, precision, F1} at the benchmark's existing px-matching convention (state it). `partial` visibility counts as a positive.
3. **Import-path bugs, fixed natively** (from the reviewimport findings): (a) `outside=1` boxes with clear/partial visibility import cleanly (semantics: track-end marker, not a label error); (b) project-wrapped exports (cloud CVAT shape, the wBu8bC4OfUY case) parse via proper meta/task lookup; (c) frame_count inference for absolute-frame sampled exports. Each bug gets a regression test built from a minimized fixture derived from the REAL failing export shapes.
4. **Re-register natively**: rerun the import through the FIXED repo path over cvat_upload/exports/harvest_review_20260707/ (same README rules — verify remap/dedupe/drop as the reviewimport lane did), replacing the lane-normalized artifacts in runs/cvat_imports/harvest_review_20260707/ with natively-imported ones; counts must match the registered 47/47/33/41/57/43 = 268 exactly; keep set_manifest.json's roles/provenance (update import_provenance to native + your lane id).
5. **THE DECISION TABLE (the headline deliverable)**: score against the human labels, per clip + pooled: (i) raw WASB sidecars (the 6 surviving prelabel dirs); (ii) each of the 3 teacher sets at runs/lanes/w3_teachertune_20260707/teacher_sets/{strict,moderate,permissive}/ (they carry approx=true imputed points — score twice: including and excluding approx points; state which the SST seeding decision should use and why in one paragraph). Output runs/lanes/w3_cvatsparse_20260707/teacher_decision_table.{json,md}: rows = raw/strict/moderate/permissive×(with/without approx), cols = per-clip + pooled recall/precision/F1.

## ACCEPTANCE
- Sparse semantics implemented + unit-tested (incl. a synthetic case proving unreviewed-frame detections don't hit precision, and hidden reviewed frames do).
- 3 import bugs fixed w/ regression tests from real-shape fixtures.
- Native re-registration: 6/6, counts == 268 breakdown exactly, schema-valid, manifest roles preserved.
- Decision table complete (4 candidates × 2 approx modes × 7 metrics-cells minimum), reproducible via a single documented command.
- WIDE suite green-or-classified per HARD RULES.

## STRUCTURED REPORT
objective_result; acceptance table; THE DECISION TABLE inline (compact md); count reconciliation; changes file:line; full_suite + classification; HONEST ISSUES; NEXT; commit_manifest; BUILD_CHECKLIST bullet DRAFT.
