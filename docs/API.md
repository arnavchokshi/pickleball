# API & Callable Surface

How to call every part of this system: the HTTP server API, the pipeline CLI,
the programmatic / single-stage entry points, the replay viewer, and the tests.
Every route/flag/command below was read from source (`server/`, `scripts/`,
`threed/`, `web/replay/`, `tests/`).

`VERIFIED=0`: being callable is not being verified. See `CAPABILITIES.md`.

## Contents
1. [Run the whole pipeline (CLI)](#1-run-the-whole-pipeline-cli)
2. [Call one stage (programmatic / `--stage`)](#2-call-one-stage-programmatic--stage)
3. [HTTP server API](#3-http-server-api)
4. [Replay viewer](#4-replay-viewer)
5. [iOS app](#5-ios-app)
6. [Tests — what exists and how to run parts](#6-tests--what-exists-and-how-to-run-parts)

---

## 1. Run the whole pipeline (CLI)

The production entry point is `scripts/racketsport/process_video.py`. Its
commands, stage order, full flag list, remote-BODY options, outputs, and
troubleshooting live in **`RUNBOOK.md`** — that is the authoritative reference;
this doc does not duplicate the flag table.

```bash
.venv/bin/python scripts/racketsport/process_video.py --video <path> [flags]
.venv/bin/python scripts/racketsport/process_video.py --help
.venv/bin/python scripts/racketsport/doctor.py --json   # env/model/remote preflight
```

`--video` is the only required flag. Exit code 0 for summary status `complete`
or `partial`, 1 for `failed`. `--json` prints the full `PIPELINE_SUMMARY` block.

The module that implements each stage is indexed in
[`threed/racketsport/README.md`](../threed/racketsport/README.md).

---

## 2. Call one stage (programmatic / `--stage`)

### The spine: `threed/racketsport/orchestrator.py`

`run_pipeline(*, clip, inputs_dir, run_dir, stage="e2e", ...)` runs the ordered
stage closure up to and including `stage`. CLI wrapper:

```bash
.venv/bin/python -m threed.racketsport.orchestrator \
  --clip <id> --inputs <dir> --out <dir> --stage <stage>
```

`--stage` values (from `PIPELINE_STAGE_CONTRACTS` in `pipeline_contracts.py`):
`calibration`, `tracking`, `pose`, `body`, `physics`, `ball_events`, `racket`,
`metrics`, `shot_drill`, `copy`, `replay`, `e2e`. By default every dependency
stage is re-derived each call (`reuse_existing_stage_artifacts=False`).

### The public-contract CLI: `threed/racketsport/pipeline_cli.py`

Legacy plumbing — validates/copies schema-checked stage artifacts and only falls
through to the real spine for `artifact_or_spine` stages with no valid artifact.
**Run as a module** (relative imports):

```bash
.venv/bin/python -m threed.racketsport.pipeline_cli --video <path> --stage <name>
.venv/bin/python -m threed.racketsport.pipeline_cli --list-stages --json
```

Stages: `capture_sidecar`, `court_calibration`, `tracks`, `ball_track`,
`contact_windows`, `player_ground` (NOT IMPLEMENTED), `racket_pose`, `metrics`
(NOT IMPLEMENTED), `replay`. Flags: `--stage` xor `--from`/`--to`, `--force`
(real execution, skips artifact/fixture copy), `--allow-fixture-fallback`
(**copies frozen sample artifacts — not a real run**), `--tier`, `--json`.

### Run an individual builder/tool

Most stage artifacts also have a standalone CLI under `scripts/racketsport/`
(`build_*`, `apply_*`, `run_*`, `evaluate_*`, `render_*`, `export_*`,
`validate_*`). Each is independently `--help`-documented. There are ~257 of
them; discover them with the catalog generator rather than grepping:

```bash
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
```

---

## 3. HTTP server API

FastAPI app: `server.render_app:app` (`create_app()` in `server/render_app.py`).

```bash
.venv/bin/uvicorn server.render_app:app --reload --port 8000     # local dev
```

`create_app()` builds one of **two mutually-exclusive route sets**, gated by
`PICKLEBALL_ACCOUNTS_ENABLED` (env). `render.yaml` pins it to `"0"` in
production (accounts OFF until Atlas/S3 secrets exist), so **§3.1 legacy is what
ships today**; the §3.3 accounts surface is built + tested but not yet reachable
from the shipped UI.

### 3.0 Always-on routes (both modes)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/health` | none | Liveness + runner mode (probes Mongo/S3 when accounts on). Never raises. |
| POST | `/api/court/predict` | none | Multipart video upload → `predict_court_layout_from_video` (template seed, or real `court_detector_v2` if `PICKLEBALL_COURT_PREDICTOR_MODE=detector`). Returns an **unverified** `court_layout_prediction`. |
| POST | `/api/court/reviews` | none | Save a human-reviewed court calibration (full artifact or raw fields). 400 on missing fields. |

### 3.1 Legacy single-user routes (`PICKLEBALL_ACCOUNTS_ENABLED` = `0`/unset — current prod)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/jobs` (202) | Multipart `video` (+ optional `clip`, `max_frames`, sidecar/court files). Creates a `JobStore` record and dispatches `GpuRunner.run()` (see §3.2). |
| GET | `/api/jobs/{job_id}` | Poll status/progress; `result` block once `complete`. 404 on unknown/unsafe id. |
| GET | `/api/jobs/{job_id}/manifest` | Stream `replay_viewer_manifest.json`. |
| GET | `/api/jobs/{job_id}/artifacts/{path}` | Stream any artifact under the job dir (path-traversal guarded). |

### 3.2 GPU execution backends (`server/gpu_runner.py`, `runner_from_env()`)
Precedence: `PICKLEBALL_GPU_WORKER_URL` → `HttpGpuRunner` > `PICKLEBALL_GPU_SSH_HOST` +
`PICKLEBALL_GPU_SSH_KEY_PATH` → `SshGpuRunner` (rsyncs code+inputs to a GCP host,
runs `process_video.py` remotely, rsyncs results back) > `PICKLEBALL_ALLOW_LOCAL_PIPELINE=1`
→ `LocalPipelineRunner` (in-process, **disabled by default**) > else
`UnconfiguredGpuRunner` (`.run()` always raises `MissingGpuRunnerConfig`).

### 3.3 Accounts routes (`PICKLEBALL_ACCOUNTS_ENABLED=1`, INFRA-1)
Requires `PICKLEBALL_JWT_SECRET`, `PICKLEBALL_MONGODB_URI`, `PICKLEBALL_S3_BUCKET`
or startup raises. Modules: `server/routes/{auth,clips,jobs_v2,account,stripe_webhook}.py`.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` (201) | invite code | 5/hour. `{email, password, invite_code}`; 403 bad code, 409 duplicate. |
| POST | `/api/auth/login` | none | 10/min. Sets httpOnly refresh cookie + 15-min HS256 access token. |
| POST | `/api/auth/refresh` | refresh cookie | Rotates refresh token; replay of a rotated token revokes the session chain (theft detection). |
| POST | `/api/auth/logout` (204) | refresh cookie | Revoke chain + clear cookie. |
| POST | `/api/clips` (201) | Bearer JWT | Create clip doc + presigned S3 multipart PUT. |
| POST | `/api/clips/{id}/complete` | Bearer JWT | Complete the S3 multipart upload. |
| GET | `/api/clips` | Bearer JWT | List caller's clips. |
| POST | `/api/jobs` (202) | Bearer JWT | `{clip_id, max_frames?}` — Mongo job doc, pulls raw bytes from S3, same `GpuRunner` as §3.2. Replaces the legacy `POST /api/jobs` in this mode. |
| GET | `/api/jobs/{id}` | Bearer JWT | Mongo-backed status; 404 if not owned. |
| DELETE | `/api/account` | Bearer JWT | **Stub 501** (delete-cascade ships in INFRA-5). |
| POST | `/api/stripe/webhook` | none | **Stub 503** (scaffold; live at P7-3). |

Tests: `.venv/bin/python -m pytest tests/render_service -q`

---

## 4. Replay viewer

`web/replay/` (Vite/React/Three.js). The viewer consumes
`replay_viewer_manifest.json` (not `PIPELINE_SUMMARY.json`).

```bash
cd web/replay && npm install
npm run dev            # vite --host 127.0.0.1  (:5173)
npm run build          # -> dist/
npm test               # vitest run
npm run typecheck      # tsc --noEmit over an explicit file list in the script
```

Open a bundle: `http://127.0.0.1:5173/?manifest=/@fs/ABSOLUTE/path/to/replay_viewer_manifest.json`
(add `&view=courtmap` for court-map mode). Headless honesty check:
`scripts/racketsport/verify_process_video_viewer.py --manifest <run>/replay_viewer_manifest.json --out-dir <run>/viewer_verify`.
Frontend API client: `web/replay/src/uploadApi.ts` (calls only `POST /api/jobs`,
`/api/court/predict`, `/api/court/reviews` — the §3.3 accounts routes have no UI caller yet).

---

## 5. iOS app

Swift package `ios/Package.swift` (7 library targets: `PickleballCore`,
`Capture`, `Calibration`, `FastTier`, `Guidance`, `Upload`, `Replay`) plus the
hosted app under `ios/App/` (Xcode project `ios/Pickleball.xcodeproj`).

```bash
swift test --package-path ios          # the 7 SwiftPM library modules
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball \
  -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO test          # App/AppTests/AppUITests (hosted XCTest)
```

Before any device deploy, run the `DinkVisionScreenshotWalker` UI test and
inspect screenshots (see README §iOS Visual Gate). More: `ios/README.md`.

---

## 6. Tests — what exists and how to run parts

Runner: `.venv/bin/python -m pytest`. **Always prefix `MPLBACKEND=Agg`** (many
body/court/overlay tests touch matplotlib). Markers: `h100`, `integration`
(both only on `test_real_tracking_runner.py`, skipped by default). No
`conftest.py`; fixtures are plain imports from
`tests/racketsport/{calibration_fixtures,json_schema_assertions}.py` +
`tests/racketsport/fixtures/`. Full suite ≈ 3,100 tests, ~8–10 min.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q                 # full suite
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q -m "not h100 and not integration"
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -k ball -q         # one domain (substring)
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_worldhmr.py::test_name -q  # one test
.venv/bin/python -m pytest tests/render_service -q                             # HTTP server
npm test -- --run --dir web/replay                                            # viewer
```

Categories (representative files):

| Category | What it checks | Examples |
|---|---|---|
| Stage-contract / plumbing | orchestrator stage graph, readiness gating, artifact→schema mapping | `test_pipeline_contracts.py`, `test_orchestrator_spine.py`, `test_process_video.py` (largest) |
| Truth-doc / repo-invariant | docs stay small/accurate, dead-code + storage hygiene, eval-clip protection | `test_truthful_capabilities.py`, `test_dead_code_audit.py`, `test_storage_policy_audit.py`, `test_eval_guard.py`, `test_cli_help.py` |
| Schema validation | real artifacts vs `docs/racketsport/*_schema.json` | `test_schemas.py` (central), 267 files use `json_schema_assertions` inline |
| Eval gates | promotion pass/fail vs real thresholds | `test_*_gate.py` (25 files), `test_eval_*_gates.py`, `test_eval_noop_gate_hardening.py` |
| Domain (by prefix) | per-pillar behavior | `test_ball_*` (51), `test_court_*` (35), `test_body_*` (19), `test_racket_*`, `test_shot_*` |

**Run after almost any change** (the structural minimum, priority order): the
eval-guard/ledger tests if you touched eval clips (`test_eval_guard.py`,
`test_append_lock.py`); `test_pipeline_contracts.py` for stage/artifact changes;
`test_process_video.py` for CLI changes; `test_schemas.py` for artifact-shape
changes; the four repo self-consistency gates
(`test_truthful_capabilities.py`, `test_dead_code_audit.py`,
`test_storage_policy_audit.py`, `test_scaffold_tool_index.py`) if you
added/removed a doc/script/artifact; `test_cli_help.py` if you added/renamed a
CLI; and the domain gate file matching what you touched.
