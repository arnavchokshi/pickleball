# Pickleball Video-to-3D Pipeline

This repo is the active implementation for the Sway Body pickleball pipeline:
record one video, provide fast on-device guidance, upload to an offline GPU
pipeline, and generate a trust-banded 3D replay plus coaching artifacts.

## Read This First

1. `AGENTS.md` - repo-specific operating rules for future agents.
2. `MASTER_PLAN.md` - final product goal, current truth, open gates, and what not to retry.
3. `RUNBOOK.md` - how the current `process_video.py` pipeline actually runs, stage by stage.
4. `CAPABILITIES.md` - canonical live/server tier split and capability truth matrix.
5. `BUILD_CHECKLIST.md` - small operational board for what remains.
6. `TECH_STACK.md` - model/runtime registry and where code lives.
7. `BALL_TRACKING_PIPELINE.md` - focused ball-stage contract used by ball code comments.

## Current Truth

`VERIFIED=0`. A scoped `process_video.py` run can produce a scrubber bundle, but
CAL, TRK, BALL, BODY, RKT, iOS live tier, replay, and full E2E remain unverified
until their explicit gates pass on the required real labels, device runs, or
runtime evidence. Do not promote a row because a smoke test, copied fixture,
internal-val run, browser load, or schema validation passed.

## Repo Map

| Path | Purpose |
|---|---|
| `scripts/racketsport/process_video.py` | Main one-command video-to-replay pipeline. |
| `threed/racketsport/` | Python stage implementations, schemas, gates, and artifact builders. |
| `tests/racketsport/` | Python verification for stage contracts and truth-doc invariants. |
| `ios/` | Native Swift capture/live-tier/upload/replay app modules. |
| `web/replay/` | Review viewer for `replay_viewer_manifest.json` bundles. |
| `models/MANIFEST.json` | Checkpoint/model manifest. Treat it as the weight registry. |
| `configs/` | Runtime/tracker/remote configuration. |
| `eval_clips/` | Small committed eval/label assets. Protected owner clips are eval-only unless code explicitly permits internal-val use. |
| `docs/racketsport/` | JSON schemas and manifests only. Narrative docs were collapsed into the root docs above. |
| `runs/` | Generated evidence and experiment outputs. Usually ignored; cite paths carefully and never assume old runs are current. |

## Storage Policy

Keep source, schemas, small eval labels, and canonical docs in git. Keep
checkpoints under `models/checkpoints/`, not loose in the repo root. Keep
generated experiment evidence under `runs/` and cite exact run paths when needed.
The only tracked files over 5 MB should be the explicit `models_coreml/` app
candidate packages and the Indoor CVAT/eval video mirror: the CVAT upload source
and eval clip source intentionally share one Git blob so upload/import and local
eval paths stay stable without duplicating repository object storage.
Run `python3 scripts/racketsport/audit_storage_policy.py --root . --json` before
adding or removing large files. That command also names the active large source/fixture exceptions
that are not generated evidence: the iOS bundled
Reality replay USDZ, the iOS bundled WorldFixture `virtual_world.json`, and the
web replay `solid_mesh_real_window_000/body_mesh_faces.json` fixture. Those files
are referenced by active Swift/Vitest tests; do not make extra copies under new
fixture or run paths. The same audit fails on generated cache/build leftovers
such as `__pycache__`, `.pytest_cache`, `ios/.build`, and `web/replay/dist`;
remove those after local tests/builds instead of leaving them in the workspace.
Do not add new long-lived research/status Markdown under `docs/racketsport/`;
use the canonical root docs or a run-local report instead.

## Verification Commands

Use focused tests for touched code, then broader checks when changing shared
contracts:

```bash
.venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py -q
.venv/bin/python -m pytest tests/racketsport/test_process_video.py tests/racketsport/test_pipeline_contracts.py -q
swift test --package-path ios
npm test -- --run --dir web/replay
```

Run only commands you can actually support in the current environment, and report
the exact command and result. If a command is skipped, say why.

## iOS Visual Gate

Before any manager/device deploy of the native app, run the deterministic
`DinkVisionScreenshotWalker` UI test and inspect the captured screenshots. The
current runner notes and extraction command live at
`runs/lanes/ios_uifix_20260707/SCREENSHOT_WALKER.md`. No device deploy should be
treated as visually cleared without a manager walker pass.
