# LANE w7_microdebt_20260709 — two disjoint micro-debt items from the wave-7 boot prompt (#10)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. Do NOT touch: scripts/racketsport/process_video.py, configs/racketsport/best_stack.json (tierprov lane owns them NOW), threed/racketsport (P2-2 landed files), web/replay src (landed). Artifacts under runs/lanes/w7_microdebt_20260709/ only.

## ITEM 1 — import_w6_labelpack_tasks.py idempotence guard
scripts/racketsport/import_w6_labelpack_tasks.py (the CVAT task importer) must be safe to re-run: detect already-imported tasks (by name/fingerprint against the CVAT API or its own import ledger — read the script to pick the mechanism it already half-has) and skip-with-account rather than duplicate. Booked as a required guard before any re-run. Include a unit test with a mocked CVAT client proving second-run = zero new tasks + explicit skip accounting.

## ITEM 2 — Vite allow-root degradation from /tmp worktrees
Wave-6 booked: running the replay viewer from /tmp worktrees degrades because Vite refuses roots outside the project (allow-root/fs.allow). Fix in web/replay Vite CONFIG ONLY (vite.config.*): declare the fs.allow entries needed so a worktree-served viewer works, WITHOUT weakening production build behavior (config-scoped, dev-server-only). Do not touch web/replay/src/**. Keep the change minimal + commented.

## SELF-VERIFICATION
Item 1: its new test + the existing importer tests. Item 2: npm run typecheck + npm run build still green (config-only change). Fix what you introduce; prove pre-existing at HEAD.

## REPORT
Self-write runs/lanes/w7_microdebt_20260709/report.json (lane_report.schema.json structure): one acceptance row per item, changes file:line, suite numbers, BEST-STACK DELTA (expected none — importer guard + dev config; state it), honest_issues, next.
