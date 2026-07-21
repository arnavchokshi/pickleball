# ultra_review_w3a_20260721 — focused review of the two product-bug fixes (fast, tight scope)

REVIEW-ONLY, gpt-5.6-sol ultra. Small ops+consumer diff from w3a_product_bugs_20260721 (report.json). Verify before commit:
1. scripts/fleet/lane_vm_startup.sh: pipeline-vs-training compute-mode parameterization correct and default-safe (a pipeline VM MUST NOT get EXCLUSIVE_PROCESS; a training VM still can; no breakage of the rail/watchdog parts of the script).
2. threed/racketsport/rally_metrics.py: the track_world_xy consumer fix reads the ACTUAL exported tracks.json fields (world_xy per frame), legacy fallback sane, absence → typed degraded reason (never a crash, never fabricated positions); the fixture test uses the real pulled tracks.json shape.
3. No fence violations (only those 2 files + tests changed by this lane — the tree also carries OTHER live lanes' edits; attribute correctly, don't blame w3a for them).
OUTPUT: verdict COMMIT_OK / COMMIT_WITH_FIXES (exact) / DO_NOT_COMMIT + file:line. One line.
