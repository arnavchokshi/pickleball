# FIX ROUND 2 for court_c0_ingest_20260721 — close R2 blockers (runs/lanes/court_c0_ingest_20260721_review/review_r2.json)

Ownership: ingest_cvat_court_images.py + its test. ONE bounded fence extension is granted below.

1. C0-03 geometry (FAIL): add scale-aware minimum edge/area and normalized homography-stability
   checks. The reviewer's sliver probe (1280x720, quad height 0.08px, area 96px^2 > configured
   92.16 minimum) must be REJECTED; add subpixel-sliver adversarial tests (their two probe configs
   verbatim + variants).
2. C0-04 relocation (FAIL): emit frame_dir paths that resolve correctly when the EXACT C1 command
   runs from the repository root (repo-root-relative paths preferred; the corpus lives under
   runs/lanes/... on any machine). Add a test that loads through the REAL C1 trainer code path with
   CWD = repo root (no cd-into-corpus masking). FENCE EXTENSION (only if strictly unavoidable): a
   minimal patch to the trainer's frame_dir/path_base resolution, backward-compatible, with its own
   regression test — justify in the report if used.
3. Production manifest shard key (FAIL): read the pinned manifest's actual `shards` key (the real
   manifest c0243e91... has four shards under `shards`, not `task_shards`); the production CLI must
   accept the real manifest TODAY (probe it); fixture updated to the real key.
4. PPA alias — ORCHESTRATOR RULING 2026-07-21: "PPA Tour" and "PPA Tour Asia" are ONE
   organizational family for collision purposes (conservative-direction merge). Encode as an
   explicit, hash-pinned alias map citing this ruling; recompute family counts (expected: train
   66/17-18 families — must stay >=15; the 8 frozen holdout SOURCE groups unchanged).
5. Refresh the JUnit artifacts to match actual final runs (reviewer found stale focused/wide XML).
No NEW wide-suite failures beyond the known environmental set. Report to report_fix2.json.
