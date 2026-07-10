# Pickleball Video-to-3D Pipeline

This repo is the active implementation for the Sway Body pickleball pipeline:
record one video, provide fast on-device guidance, upload to an offline GPU
pipeline, and generate a trust-banded 3D replay plus coaching artifacts.

## Read This First

1. `NORTH_STAR_ROADMAP.md` - the sole product vision, current truth, ordered
   future plan, gates, owner asks, and active agent queue.
2. `AGENTS.md` - durable repository rules and code navigation.
3. `RUNBOOK.md` - how the current `process_video.py` pipeline actually runs.
4. `BALL_TRACKING_PIPELINE.md` - focused numbered BALL contract used by code.

Historical plans, checklists, capability tables, blueprints, check-ins, and goal
documents are indexed under `runs/archive/root_docs_20260709/`. They are
evidence/context, not current instructions.

## Current Truth

`VERIFIED=0`. A scoped `process_video.py` run can produce a scrubber bundle, but
CAL, TRK, BALL, BODY, RKT, iOS live tier, replay, and full E2E remain unverified
until their explicit gates pass on the required real labels, device runs, or
runtime evidence. Do not promote a row because a smoke test, copied fixture,
internal-val run, browser load, or schema validation passed.

The 2026-07-09 deep review also found P0 product-correctness blockers in the
Swift/Python sidecar, production upload call, source/cache identity,
raw/undistorted coordinates, partial/complete propagation, artifact packaging,
and same-run BODY→event/arc feedback. Read the North Star and
`runs/CV_PIPELINE_DEEP_REVIEW_20260709.md` before launching another isolated
model campaign.

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
| `docs/racketsport/` | JSON schemas and manifests only. Product direction lives only in the North Star. |
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
update the North Star only for current product/direction changes, otherwise use
a run-local report.

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
