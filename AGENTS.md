# Agent Instructions

- Stay on `main` unless the user explicitly asks for a branch.
- Never state that a capability is complete, verified, or current without fresh evidence from code, tests, artifacts, device state, or live infrastructure.
- Preserve unrelated dirty worktree changes. Do not revert files you did not intentionally change.
- Start with `README.md`, then `MASTER_PLAN.md`, `RUNBOOK.md`, `CAPABILITIES.md`, `BUILD_CHECKLIST.md`, and `TECH_STACK.md`.
- Treat `VERIFIED=0` as binding until a documented gate passes. Scoped passes, smoke tests, internal-val runs, browser loads, and copied fixtures are not promotions.
- The main pipeline entrypoint is `scripts/racketsport/process_video.py`. The old `threed/racketsport/pipeline_cli.py` contract CLI is legacy plumbing unless a task explicitly targets it.
- Keep narrative docs small. If a new document is needed, first check whether it belongs in an existing canonical doc or as a generated run artifact under `runs/`.

## Agent Navigation

| Need | Start here |
|---|---|
| Repo rules and doc map | `README.md`, then this file. |
| Product goal, current truth, and no-retry decisions | `MASTER_PLAN.md`. |
| Current pipeline run behavior | `RUNBOOK.md`, then `scripts/racketsport/process_video.py`. |
| Capability truth and promotion gates | `CAPABILITIES.md` and `BUILD_CHECKLIST.md`. |
| Runtime/model/code ownership | `TECH_STACK.md`, `models/MANIFEST.json`, and `configs/`. |
| Pipeline CLIs and orchestration helpers | `scripts/racketsport/`. |
| Stage implementations and shared library code | `threed/racketsport/`. |
| Python verification | `tests/racketsport/`. |
| Native capture/live/upload/replay surfaces | `ios/`. |
| Web replay viewer | `web/replay/`. |
| Public schemas and manifests | `docs/racketsport/`; JSON schemas/manifests only, not narrative docs. |
| Generated evidence | `runs/`; Generated evidence only, not a source of current truth unless rerun or explicitly cited. |

Before adding a new CLI, removing code, or claiming the tree is clean, run the
small repo-structure checks that match the current documentation policy:

```bash
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .
python3 scripts/racketsport/audit_storage_policy.py --root . --json
```
