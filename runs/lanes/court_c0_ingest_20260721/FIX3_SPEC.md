# FIX ROUND 3 for court_c0_ingest_20260721 — single blocker: true relocation semantics
(runs/lanes/court_c0_ingest_20260721_review/review_r3.json probe 2)

Surgical scope ONLY. Ownership: ingest_cvat_court_images.py + its test, PLUS the already-granted
minimal fence extension into the C1 trainer loader path for frame_dir/--real-root rebasing.

Required: emit frame_dir RELATIVE TO THE CORPUS ROOT, and make the real C1 loader rebase such
paths against the corpus root supplied via --real-root (backward-compatible: existing absolute or
repo-relative corpora must keep loading; add a regression test for that). The R3 reviewer's probe
must pass: COPY the emitted corpus to a different directory, run the real C1 code path from repo
root with --real-root pointed at the copy, and require 60/60 rows to resolve + one image to
materialize from the COPY (assert zero resolved paths point at the original location).
Do not touch geometry/deny/shard/alias code (all confirmed PASS).
Report to report_fix3.json. Suite: failures must remain exactly the known 30.
