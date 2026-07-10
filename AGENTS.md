# Agent Instructions

- Stay on `main` unless the user explicitly asks for a branch.
- Never state that a capability is complete, verified, or current without fresh evidence from code, tests, artifacts, device state, or live infrastructure.
- Preserve unrelated dirty worktree changes. Do not revert files you did not intentionally change.
- Start with `NORTH_STAR_ROADMAP.md`, then this file and the relevant `RUNBOOK.md` section.
- Treat `VERIFIED=0` as binding until a documented gate passes. Scoped passes, smoke tests, internal-val runs, browser loads, and copied fixtures are not promotions.
- The main pipeline entrypoint is `scripts/racketsport/process_video.py`. The old `threed/racketsport/pipeline_cli.py` contract CLI is legacy plumbing unless a task explicitly targets it.
- `NORTH_STAR_ROADMAP.md` is the only product/current-truth/future-plan authority. Do not create another checklist, master plan, wave roadmap, capability matrix, or blueprint.
- Put dated status, experiments, and handoffs under `runs/`. Volatile coordination belongs in `runs/manager/inflight_lanes.md` and `runs/manager/gpu_fleet.md`, never a root append log.
- Baseline first, score every candidate with the same frozen gate, and keep raw observations immutable. Visual plausibility and optimizer residuals are not accuracy proof.
- One integration owner serializes `scripts/racketsport/process_video.py`; file-fenced CAL/TRK/BALL/BODY/RKT lanes may run in parallel.

## Agent Navigation

| Need | Start here |
|---|---|
| Product goal, current truth, exact next order, and no-retry decisions | `NORTH_STAR_ROADMAP.md`. |
| Repo rules and doc map | This file, then `README.md`. |
| Current pipeline run behavior | `RUNBOOK.md`, then `scripts/racketsport/process_video.py`. |
| Selected runtime/model defaults | `configs/racketsport/best_stack.json`, `models/MANIFEST.json`, and `configs/`. |
| Pipeline CLIs and orchestration helpers | `scripts/racketsport/`. |
| Stage implementations and shared library code | `threed/racketsport/`. |
| Python verification | `tests/racketsport/`. |
| Native capture/live/upload/replay surfaces | `ios/`. |
| Web replay viewer | `web/replay/`. |
| Public schemas and manifests | `docs/racketsport/`; JSON schemas/manifests only, not narrative docs. |
| Generated evidence | `runs/`; Generated evidence only, not a source of current truth unless rerun or explicitly cited. |
| Historical plans and decisions | `runs/archive/root_docs_20260709/INDEX.md`; context only, never current sequencing. |

Before adding a new CLI, removing code, or claiming the tree is clean, run the
small repo-structure checks that match the current documentation policy:

```bash
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .
python3 scripts/racketsport/audit_storage_policy.py --root . --json
```
