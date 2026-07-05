# Process Video Runbook

Last updated: 2026-07-03.

`scripts/racketsport/process_video.py` is the current one-command pipeline for a
video-to-scrubber bundle. It is the entrypoint future agents should start from
unless the task explicitly targets older contract plumbing.

## Minimal Commands

Use a trusted calibration seed when possible:

```bash
python3 scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --court-calibration eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json \
  --out runs/process_video_wolverine_current \
  --verify-viewer
```

Manual four-corner seed path:

```bash
python3 scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --court-corners path/to/court_corners.json \
  --out runs/process_video_<clip>
```

Reuse precomputed tracks or ball track when evaluating downstream stages:

```bash
python3 scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --court-calibration path/to/court_calibration.json \
  --tracks path/to/tracks.json \
  --ball-track path/to/ball_track.json \
  --out runs/process_video_<clip> \
  --force
```

CPU/skeleton-only smoke:

```bash
python3 scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --court-calibration path/to/court_calibration.json \
  --tracks path/to/tracks.json \
  --ball-track path/to/ball_track.json \
  --no-gpu \
  --out runs/process_video_smoke
```

## Inputs

| Input | Flag | Notes |
|---|---|---|
| Source video | `--video` | Required. `--clip` defaults to the video stem. |
| Manual corner seed | `--court-corners` | `court_corners.json` with declared `image_size`. |
| Capture sidecar | `--capture-sidecar` | Pre-built sidecar, e.g. ARKit/manual capture metadata. |
| Court keypoints | `--court-keypoints` | Optional no-tap/metric path paired with a capture sidecar. |
| Solved calibration | `--court-calibration` | Preferred when available. Validated before use; explicit flag wins. |
| Tracks reuse | `--tracks` | Reuses a valid `tracks.json`; otherwise live tracking may run. |
| Ball reuse | `--ball-track` | Reuses a valid `ball_track.json`; otherwise WASB path may run. |
| Arc/contact sidecars | `--events-selected`, `--ball-track-arc-solved` | Optional ball-aware mesh scheduling inputs. |

`--allow-auto-court-corners-preview` can seed preview taps from line detection
when no trusted calibration exists. It is unverified by definition and must not
be called CAL promotion.

## Stage Order

`process_video.py` writes `PIPELINE_SUMMARY.json` even on partial runs. The stage
order is:

1. **ingest** - validate video and build/consume capture sidecar.
2. **calibration** - create or consume `court_calibration.json`, court zones, net
   plane, and court evidence. Calibration is the only hard dependency.
3. **tracking** - run or reuse person tracks, optionally with raw-pool global
   association.
4. **rally_gating** - optional loose rally-span gating before downstream work.
5. **frames** - materialize BODY frames from tracks and planned mesh windows.
6. **ball** - run/reuse ball track, bounce, and court in/out artifacts.
7. **events** - fuse ball/audio/wrist cues into `contact_windows.json` and
   `frame_compute_plan.json`.
8. **body** - dispatch Fast SAM-3D-Body to the configured remote GPU path by
   default, run local BODY only with `--body-local`, or skip with `--no-gpu`.
9. **grounding** - render-honest BODY grounding refinement when inputs exist.
10. **world** - write `virtual_world.json` and `trust_bands.json`.
11. **confidence** - write `confidence_gated_world.json` unless
    `--no-confidence-gate` is set.
12. **manifest** - write `replay_viewer_manifest.json` and optional point scene.
13. **verify** - optional `--verify-viewer` headless web viewer check.

## Important Flags

| Flag | Effect |
|---|---|
| `--force` | Recompute stages even when valid artifacts already exist. |
| `--max-frames` | Cap frames for smoke runs only. Do not use capped runs as promotion evidence. |
| `--no-global-association` | Skip raw-pool global association after loose-pool tracking. |
| `--global-association-profile` | Explicit internal-val tuning profile. Defaults are not universal proof. |
| `--allow-auto-ball-track` | Opt into clip-id discovery of old ball tracks under `runs/`; preview reuse only. |
| `--skip-ball` | Omit ball stage. Downstream ball/event outputs will be absent or degraded. |
| `--skip-audio` | Omit audio onsets from contact fusion. |
| `--rally-gating` | Opt into loose rally-span gating and preserve pre-gating copies. |
| `--mesh-coverage-mode ball_aware` | Default mesh scheduling policy. Uses physically validated ball/contact/proximity triggers, not low-confidence wrist cues alone. |
| `--no-gpu` | Skip live tracking/pose/BODY unless reuse artifacts are supplied. |
| `--body-local` | Run BODY in-process on a GPU host instead of remote dispatch. |
| `--no-grounding-refine` | Skip BODY grounding refinement. |
| `--no-confidence-gate` | Point viewer at raw `virtual_world.json`. |
| `--no-scene-points` | Skip point GLB scene generation. |
| `--verify-viewer` | Start the replay verifier against the produced manifest. |
| `--no-ball-arc` | Skip the default ball 3D arc stage (auto-bounce anchors -> arc solver -> flight-sanity gate). |
| `--ball-candidates` | Reuse existing `ball_candidates.json` top-K detector sidecars (repeatable). Emitted by default when ball inference runs. |
| `--no-ball-candidates` | Disable default top-K candidate sidecar emission during ball inference. |
| `--json` | Print the full summary JSON instead of a human table. |

## Remote BODY Runtime

Fast SAM-3D-Body runs through `scripts/racketsport/remote_body_dispatch.py`;
remote dispatch is the default BODY path unless `--body-local` or `--no-gpu` is set.
The remote path is operational plumbing plus model execution. A completed remote
run can still be `partial`, skeleton-only, or gated by downstream trust bands, so
these flags are not promotion evidence by themselves.

| Flag | Effect |
|---|---|
| `--remote-host` | SSH target for the remote A100 BODY worker. |
| `--remote-ssh-key` | SSH key used for the remote worker connection. |
| `--remote-repo` | Repo path on the remote worker; it must match the committed pipeline layout. |
| `--remote-python` | Python used for remote repo-side orchestration. |
| `--remote-fast-sam-python` | Python used for the Fast SAM-3D-Body environment. |
| `--remote-fast-sam-root` | Fast SAM-3D-Body checkout/root on the remote worker. |
| `--remote-lock-wait-timeout-s` | Maximum wait for the shared remote GPU lock. |
| `--remote-command-timeout-s` | Maximum wall-clock time for the remote BODY command after the lock is held. |
| `--sam3d-body-input-size-px` | BODY input-size benchmark/runtime knob. Changing it creates a new run condition. |
| `--sam3d-crop-bucket-sizes` | Cross-frame crop bucket sizes used by the SAM3D body-mode path. |
| `--no-sam3d-torch-compile` | Disable `torch.compile` for the SAM3D body-mode decoder path. |
| `--sam3d-compile-warmup-buckets` | Bucket sizes warmed before timing or running compiled SAM3D decode. |
| `--serialize-tier2-mesh-vertices` | Debug/storage-heavy override that serializes tier-2 mesh vertices instead of joints-only tier-2 output. |

Shared GPU lock wait and remote command timeout are separate budgets. Raising
one does not raise the other, and either timeout can make the BODY stage partial.
Keep `--serialize-tier2-mesh-vertices` off unless inspecting tier-2 mesh frames;
it increases artifact size and is not needed for normal replay bundles.

## Outputs

The run directory contains stage artifacts and a top-level `PIPELINE_SUMMARY.json`.
Common outputs include:

- `capture_sidecar.json`
- `court_calibration.json`
- `tracks.json`
- `ball_track.json`
- `ball_track_arc_solved.json` (default ball 3D arc, render-only, self-kill gated)
- `ball_flight_sanity.json` and `ball_bounce_candidates.json` (default ball-arc stage)
- `contact_windows.json`
- `frame_compute_plan.json`
- `smpl_motion.json`
- `body_mesh.json`
- `virtual_world.json`
- `confidence_gated_world.json`
- `trust_bands.json`
- `replay_viewer_manifest.json`

Treat each artifact according to its trust band. Existence and schema validity
are not accuracy proof.

## Status Interpretation

`process_video.py` returns success for `complete` and `partial` summary status.
That means the bundle is inspectable; it does not mean the system is verified.
Promotion still depends on the gates in `MASTER_PLAN.md` and `CAPABILITIES.md`.

## Legacy Contract CLI

`threed/racketsport/pipeline_cli.py` still exists for public contract/schema
plumbing and fixture-copy tests. It is not the current full offline pipeline.
If you use `--allow-fixture-fallback`, you are copying old sample artifacts, not
running models.

## Focused Verification

```bash
.venv/bin/python -m pytest tests/racketsport/test_process_video.py -q
.venv/bin/python -m pytest tests/racketsport/test_pipeline_contracts.py -q
python3 scripts/racketsport/process_video.py --help
```

Before claiming the repo is clean or adding another tool, also run the hygiene
checks that guard this simplified layout:

```bash
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .
python3 scripts/racketsport/audit_storage_policy.py --root . --json
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
.venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py tests/racketsport/test_scaffold_tool_index.py tests/racketsport/test_dead_code_audit.py tests/racketsport/test_storage_policy_audit.py -q
```

These commands are reference and storage checks only. They do not prove semantic
reachability, model quality, device behavior, or E2E verification.

For viewer changes:

```bash
npm test -- --run --dir web/replay
python3 scripts/racketsport/verify_process_video_viewer.py --manifest <run>/replay_viewer_manifest.json --out-dir <run>/viewer_verify
```
