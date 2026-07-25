# MERGE_NOTES — lane `oneworld_wire_20260724`

Branch `fusion-wire-20260724`, forked from `9d318ad`. For the integration owner
who serializes `scripts/racketsport/process_video.py`.

> Placed under `runs/lanes/` on purpose. A root-level `MERGE_NOTES.md` would
> break `tests/racketsport/test_truthful_capabilities.py::test_markdown_doc_inventory_stays_small_and_explicit`,
> which asserts an exact allowlist of root Markdown files. `runs/` is ignored by
> that test, and AGENTS.md says dated status and handoffs belong under `runs/`.

## Files touched (6 source files + 3 lane docs)

| File | Change | Conflict risk |
|---|---|---|
| `scripts/racketsport/process_video.py` | Defect A capability rename (2 tuples) + Defect B stage wiring (stack key block, graph node, `enabled_by` literal, `authoritative_stage_names` kwarg, 3 identity dicts, `PipelineOptions` fields, `_stage_identity_options`, `_stage_explicit_inputs`, `_stage_one_world`, 2 CLI flags, `resolved_best_stack_config_from_options`, `best_stack_overrides_from_options`, 1 import) | **HIGH by policy** — single-owner file. **LOW in practice**: not in the main checkout's dirty set as of `a83ca9b`. |
| `server/bundle_policy.py` | `_mandatory_requirements()`: `fusion` -> `composited_world`, lockstep with the worker | **LOW** — not dirty in main |
| `configs/racketsport/best_stack.json` | +1 entry `world.one_world_v1_fusion`; `updated` 2026-07-23 -> 2026-07-24. **Revision stays 15.** | **LOW** — not dirty in main |
| `RUNBOOK.md` | Stage 19 `composited_world` note; new optional `one_world` stage paragraph; documents the pre-existing undocumented `player_selection` stage | **HIGHEST** — `RUNBOOK.md` **is** modified-uncommitted in the main checkout right now |
| `tests/racketsport/test_capability_naming_honesty.py` | new file (7 tests) | none |
| `tests/racketsport/test_one_world_stage.py` | new file (21 tests) | none |
| `runs/lanes/oneworld_wire_20260724/{REPORT.md,report.json,MERGE_NOTES.md,evidence/*}` | lane evidence | none |

Not touched: `threed/racketsport/one_world_v1.py`, `threed/racketsport/virtual_world.py`,
`threed/racketsport/placement.py`, any court file, `NORTH_STAR_ROADMAP.md`,
`scripts/racketsport/list_scaffold_tools.py`, and the three scaffold one_world CLIs.

## Concurrent-work assessment (checked read-only, never written to)

The main checkout moved during this lane. Last observed at `60631f1`, 3 commits
ahead of my base `9d318ad`, with 37 dirty paths.

- Commits `9d318ad..60631f1` (`a83ca9b` remote-BODY/CUDA isolation, `244c9c1`
  BODY foot-phase propagation, `60631f1` refined skeletons for world assembly)
  touch `scripts/racketsport/remote_body_dispatch.py`,
  `threed/racketsport/placement.py`, `threed/racketsport/virtual_world.py`, and
  three tests. **Zero overlap** with the files this branch changes.
- `60631f1` is worth a second look despite not conflicting textually: it changes
  `virtual_world.py`, i.e. the compositor whose output this branch renames to
  `composited_world` and which `one_world` consumes as `virtual_world.json`. It
  changes what the compositor writes, not the capability contract or the fusion
  wiring, so the rename and the stage remain correct — but re-run the
  blast-radius command after merging rather than assuming.
- Main's uncommitted set includes `RUNBOOK.md`, `NORTH_STAR_ROADMAP.md`,
  `README.md`, `scripts/racketsport/audit_data_utilization.py`,
  `scripts/racketsport/audit_storage_policy.py`,
  `scripts/racketsport/list_scaffold_tools.py`,
  `tests/racketsport/test_truthful_capabilities.py`, and several other tests.
- **Only `RUNBOOK.md` overlaps this branch.** My edits are confined to the
  numbered "Stage Order" list (items 4/19 and two inserted optional-stage
  paragraphs). If the concurrent edit is elsewhere in the file, the merge is
  textual and trivial.
- The other agent appears to be repairing several of the failures this lane
  reports as pre-existing (`audit_data_utilization.py`, `audit_storage_policy.py`,
  `test_truthful_capabilities.py`). If that work lands first, the pre-existing
  failure list in `REPORT.md` section 4.2 will shrink independently of this branch;
  re-run before re-attributing.

## Merge-order recommendation

1. Land the other agent's in-flight repo-structure repairs first — they touch the
   audit tooling and the truthful-capabilities test that this branch's failure
   attribution depends on, and none of them touch `process_video.py`.
2. Then land this branch. Re-run the blast-radius command in `REPORT.md`
   section 4.2 after merging and expect the pre-existing failure list to be a
   subset of the four recorded there.

## Behavioural review checklist

- Default-OFF is load-bearing. `authoritative_stage_names` gained a
  `one_world: bool = False` keyword; the default is what preserves every existing
  call site. Do not "tidy" it to match `player_selection`'s `= True`.
- `resolved_best_stack_config_from_options` adds the new key **only when
  enabled**. Unconditionally adding it would change every other stage's
  content-addressed fingerprint and invalidate all existing generations.
- Nothing may be added to `RUN_IDENTITY_DEPENDENCIES` that depends on
  `one_world`. A test enforces this; it is what keeps the preview fusion from
  becoming product authority.
- The capability rename must stay lockstep across `process_video.py` and
  `server/bundle_policy.py`. A one-sided rename silently downgrades every
  complete bundle to `partial` via `gate_reported_status`.

## Standing-rules reminder

`VERIFIED=0`. This branch is integration progress, not capability progress. The
new best_stack entry is `PENDING` / `do_not_promote` and must not be promoted on
the strength of green tests.

## Incident affecting another lane (not a merge blocker)

While interrupting my own time-boxed wide test runs, an over-broad process match
also sent SIGINT to the concurrent `ball-lane-20260723` worktree's test run,
killing it at 43:05 (it had reached 71 failed / 3937 passed / 30 skipped /
28 errors). No files were modified and no repo state was corrupted — the cost is
that lane's wall-clock time. That lane should re-run its suite. Recorded in
`report.json` -> `incident`.
