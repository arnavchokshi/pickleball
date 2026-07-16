# Lane event_head_ext_20260716 — bounded extension: full-train mode + anchor-candidate CLI (owner acceleration)

Codex implementation lane, DinkVision repo /Users/arnavchokshi/Desktop/pickleball. VERIFIED=0.
The event_head scaffold (lane event_head_scaffold_20260716) is BUILT and manager-ruled CPU-smoke
GREEN — read runs/lanes/event_head_scaffold_20260716/spec.md for its contract, then read the
existing code: threed/racketsport/event_head/{datasets,model,matcher}.py and
scripts/racketsport/{train,eval,build}_event_head*.py. Your job is EXACTLY the two extensions in
runs/lanes/event_head_scaffold_20260716/extension_brief.md (read it verbatim — it is the spec for
WHAT to build, including the frozen Track A anchor JSON schema). This file adds the operating
constraints:

## HARD RULES (deltas)
- NO commits/branches. Fence (exhaustive): scripts/racketsport/train_event_head.py,
  scripts/racketsport/build_event_head_anchor_candidates.py (NEW),
  tests/racketsport/test_event_head_training.py, tests/racketsport/test_event_head_anchor_candidates.py
  (NEW), threed/racketsport/event_head/** (shared helpers if needed),
  scripts/racketsport/list_scaffold_tools.py (ONLY registration entries for the new CLI),
  runs/lanes/event_head_ext_20260716/**. NOTHING else.
- A CONCURRENT resumed codex session is running the full wide suite right now (bystander — it edits
  nothing). Do NOT kill python/pytest processes you did not start; do NOT run the wide suite or
  any tests/racketsport-wide command. Focused verification ONLY:
  (a) MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_event_head_training.py
      tests/racketsport/test_event_head_anchor_candidates.py tests/racketsport/test_event_head_model.py -q
  (b) .venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
  (c) the two CPU proof runs from the extension brief (tiny --full run; anchor CLI on the tiny
      fixture video) — real unpiped exit codes echoed.
- --smoke behavior of train_event_head.py must stay byte-compatible (existing tests must pass
  unmodified except additive asserts).
- Protected-data rules unchanged (no Tier-A, no protected seed, no owner media in any train path).
- Artifacts under runs/lanes/event_head_ext_20260716/. Structured report.json (schema at
  docs/racketsport/lane_report.schema.json): per-extension acceptance w/ exit codes, honest issues,
  BEST-STACK DELTA (c) no stack delta expected.
- Speed matters (owner directive: GPU pretrain waits on you): keep it bounded, no scope creep,
  no refactors of working scaffold code beyond what the extensions require.
