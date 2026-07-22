# FIX ROUND for court_c0_ingest_20260721 — address ultra-review REJECT (runs/lanes/court_c0_ingest_20260721_review/review.json)

Same HARD RULES + FILE OWNERSHIP as spec.md (ingest_cvat_court_images.py + its test only).
Read the review JSON first; implement its exact_fixes for C0-01..C0-04 plus any non-blocking items:

- C0-01 (CRITICAL): production mode must hard-require the permanent deny set {IYnbdRs1Jdk}; assert the
  pinned manifest contains exactly 3 IYnbdRs1Jdk rows and all are excluded; reject wrong-valid/extra
  deny configurations; adversarial tests.
- C0-02 (CRITICAL): implement true source/channel/venue FAMILY grouping (not source_id).
  ORCHESTRATOR RULING 2026-07-21 (fail-closed family policy, per EXACT_PLAN §2.2 split rule): any
  train-candidate source whose channel/venue family intersects a FROZEN holdout family is EXCLUDED
  before label read with state QUARANTINED_FAMILY_COLLISION (e.g., 3sC53GlvW_s vs holdout
  4qSoA-jwpVM, shared channel PPA Tour). The frozen 8 holdout groups stay exactly as-is. Recompute
  gate counts over FAMILIES; re-verify >=60 rows / >=15 train family groups still holds after
  exclusion and report the new counts. Shared-family adversarial fixture required.
- C0-03 (HIGH): geometric validity gate — reject duplicate/collinear/crossed/near-zero-area or
  non-invertible four-corner configurations; failure tests for each.
- C0-04 (HIGH): emit artifacts the EXACT C1 command consumes: source_split.json must satisfy
  train_court_model_v2.py --real-split-proposal (train_datasets list, verify against its parser at
  line ~719-737); corpus paths must be relocation-safe (relative to the corpus root); test through
  the real C1 load path, not just the low-level loader.

Acceptance: all review exact_fixes implemented + tested; focused suite green; wide suite no NEW
failures vs the known 31-32 environmental set; report the post-exclusion family/gate counts.
Report to report_fix1.json (schema-valid lane report).
