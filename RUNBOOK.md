# Process Video Runbook

Last updated: 2026-07-16.

`scripts/racketsport/process_video.py` is the current one-command pipeline for a
video-to-scrubber bundle. It is the entrypoint future agents should start from
unless the task explicitly targets older contract plumbing.

## Minimal Commands

Use a trusted calibration seed when possible:

```bash
.venv/bin/python scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --court-calibration eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json \
  --out runs/process_video_wolverine_current \
  --verify-viewer
```

For a real BODY run, also supply an explicit current `--remote-host <host>` or
choose `--body-local`. The remote host intentionally has no universal default;
without either choice BODY may be blocked and the command can still return a
partial bundle.

### Content-addressed run identity

North Star task `NS-01.3 Content-addressed DAG` is engineering-complete. Stage
reuse now requires exact source/code/model/config/input/upstream identities;
unfingerprinted or stale artifacts rebuild unless they carry an explicit
migration attestation. Generated stage directories are published as immutable,
transactional generations. `--force` is a bounded rebuild override, never the
authority for artifact identity or correctness. Continue to use a stable
source-specific `--clip` and a lane/run-specific `--out` for clear provenance;
legacy run directories need migration attestation rather than name-based trust.

Swift-encoded version-1 sidecars now share the Python strict contract for live
recordings, missing-sensor captures, and camera-roll imports. The checked-in
cross-language golden fixtures must remain green and unknown keys remain
rejected. This is contract proof, not current physical-device proof; a fresh
device sidecar and upload trace are still required for the product exit gate.

Use `--clip` explicitly for eval clips. The committed eval videos are usually
named `source.mp4`; if `--clip` is omitted, the clip id becomes `source`, which
silently misses clip-keyed tracker tuning and default raw-pool profiles.

Manual four-corner seed path:

```bash
.venv/bin/python scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --clip <globally_unique_source_clip_id> \
  --court-corners path/to/court_corners.json \
  --out runs/process_video_<clip>_<fresh_run_id>
```

Use a lane/run-specific `<fresh_run_id>` for inspectable provenance. Identity
safety no longer depends on the directory being new.

Reuse precomputed tracks or ball track when evaluating downstream stages:

```bash
.venv/bin/python scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --clip <globally_unique_source_clip_id> \
  --court-calibration path/to/court_calibration.json \
  --tracks path/to/tracks.json \
  --ball-track path/to/ball_track.json \
  --out runs/process_video_<clip>_<fresh_run_id>
```

CPU/skeleton-only smoke:

```bash
.venv/bin/python scripts/racketsport/process_video.py \
  --video path/to/clip.mp4 \
  --clip <stable_clip_id> \
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
| Stable clip id | `--clip` | Set explicitly for eval clips whose video file is `source.mp4`; clip-keyed tuning uses this id. |
| Run directory | `--out` | Defaults to `runs/process_video_<clip>/`; use a lane/run-specific path. |
| Manual corner seed | `--court-corners` | `court_corners.json` with declared `image_size`. |
| Capture sidecar | `--capture-sidecar` | Pre-built sidecar, e.g. ARKit/manual capture metadata. |
| Court keypoints | `--court-keypoints` | Optional no-tap/metric path paired with a capture sidecar. |
| Solved calibration | `--court-calibration` | Explicit `--court-calibration` wins. Otherwise explicit `--court-corners`/`--capture-sidecar` inputs win; only when none are supplied does the runner auto-discover `<video_dir>/labels/court_calibration_metric15pt.json`. |
| Tracks reuse | `--tracks` | Reuses a valid `tracks.json` only when no existing clip-local track wins; use a new run directory. |
| Ball reuse | `--ball-track` | Reuses a valid `ball_track.json` only when no existing clip-local ball track wins; use a new run directory. |
| Arc/contact sidecars | `--events-selected`, `--ball-track-arc-solved` | Optional ball-aware mesh scheduling inputs. |

## Environment gotchas

Use `.venv/bin/python` for repo Python CLIs and tests. On Apple Silicon, a
shell that looks like it has PyTorch installed can still be the wrong Anaconda
or Homebrew interpreter, which changes MPS behavior and package visibility.

Set `MPLBACKEND=Agg` before pytest or other matplotlib-touching checks:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q
```

Long pipeline/debug commands can run for minutes in zsh. Do not assume a command
is hung only because it has been quiet; check stage logs, remote stdout, or the
process table before killing it.

`--allow-auto-court-corners-preview` can seed preview taps from line detection
when no trusted calibration exists. It is unverified by definition and must not
be called CAL promotion.

## Stage Order

After argument parsing and option construction succeed, `process_video.py`
writes `PIPELINE_SUMMARY.json` even on partial runs. Pre-flight argument/path
failures can exit before any run directory or `PIPELINE_SUMMARY.json` exists.
The default serial path has 23 stage outcomes. Optional `rally_gating` makes
24, and optional viewer verification also makes 24; enabling both makes 25. The
order below is projected from `AUTHORITATIVE_STAGE_GRAPH` in
`scripts/racketsport/process_video.py`; overlap mode uses the same graph and
only moves `frames` ahead of the four overlapped BALL/event stages:

1. **ingest** - validate video and build/consume capture sidecar.
2. **calibration** - create or consume `court_calibration.json`, court zones, net
   plane, and court evidence. Calibration is the only hard dependency.
3. **input_quality** - write the advisory/strict input-quality report. This is
   currently after calibration; the target DAG splits media checks before
   calibration from court-visibility checks after it.
4. **tracking** - run or reuse person tracks, optionally with raw-pool global
   association.
5. **camera_motion** - optional/auto preview camera-motion compensation before
   placement homography projection.
6. **placement** - project tracks into court/world placement when inputs exist.
Optional insertion after stage 6: **rally_gating**. On a cold run it occurs
before fresh BALL/audio and does not yet trim all downstream decoding.

7. **ball** - run/reuse ball track, bounce, and court in/out artifacts.
8. **ball_arc** - default 3D ball chain: auto-bounce anchors, arc solver, and
   flight-sanity gate. Use `--no-ball-arc` to skip it.
9. **events** - fuse ball/audio/wrist cues into `contact_windows.json` and
    `frame_compute_plan.json`. On a cold serial run BODY does not exist yet, so
    wrist cues are explicitly blocked.
10. **ball_fill** - render-honest fill from accepted ball arc/contact evidence.
11. **frames** - materialize BODY frames from tracks and the event/mesh plan.
12. **body** - dispatch SAM-3D-Body to the configured remote GPU path by
   default, run local BODY only with `--body-local`, or skip with `--no-gpu`.
   RTMW/RTMW3D/RTMPose are retired; the pipeline is SAM-3D-Body only for
   offline body joints/mesh.
13. **placement_refine** - currently always skipped by the R3 same-pass safety
    rule; BODY foot pixels require a true second pass before a fresh BODY run.
14. **grounding_refine** - render-honest BODY grounding
    refinement when inputs exist.
15. **placement_trajectory_refine** - default-OFF preview stage after grounding.
    Enable it with `--placement-trajectory-refine` (or a future enabled
    `body.placement_trajectory_refine` best-stack value) to write the separate
    `placement_trajectory_refined.json` artifact from current TRK footpoints,
    BODY placement, and planted-foot windows. Missing BODY/plant evidence is a
    typed skip/degrade; malformed inputs fail loudly. Raw tracks, placement,
    skeleton, phase, and grounding artifacts remain immutable. The output stays
    preview-band, `do_not_promote`, and `VERIFIED=0`.
16. **paddle_pose** - write a fail-closed, render-only estimated paddle artifact
    when BODY wrist/palm evidence exists.
17. **events_refined** - build the separately versioned post-BODY
    `contact_windows_refined_v1.json` from dependency-current BODY, paddle,
    ball, timebase, and audio evidence. Raw `contact_windows.json` remains
    immutable; content hashes gate reuse.
18. **ball_arc_refined** - re-run the render-only arc chain from current
    refined contacts when its dependency hashes changed. Segment guard
    timeouts remain typed degraded outcomes and this stage records their real
    wall time.
19. **world** - compose `virtual_world.json` and `trust_bands.json` from the
    already-finished refined artifacts; refinement time is not folded into
    this stage.
20. **confidence_gate** - write `confidence_gated_world.json` unless
    `--no-confidence-gate` is set.
21. **match_stats** - default-on fail-open placement/court movement stats.
22. **coaching_facts** - build deterministic rally metrics and coaching facts,
    then run the zero-fabrication audit before the manifest. This stage fails
    closed: on audit failure the facts are excluded and the bundle stays
    `partial`.
23. **manifest** - write `replay_viewer_manifest.json` and optional point scene,
    linking only finished current stats and audited coaching facts.
Optional final stage: **verify** via `--verify-viewer`. It is stage 24 without
rally gating and stage 25 with rally gating.

`match_stats` and audited deterministic `coaching_facts` run before `manifest`.

The target architecture adds an explicit post-BODY event/arc/placement refine
pass, then global fusion, stats and coaching, and builds the manifest last. Do
not simulate that target by leaving stale artifacts in the clip directory.

## Important Flags

| Flag | Effect |
|---|---|
| `--clip` | Stable clip id. Set it explicitly when the source file is named `source.mp4`; otherwise clip-keyed defaults use `source`. |
| `--sport {pickleball,tennis}` | Sport/rules hint for downstream court/event semantics. Pickleball is the v1 product target. |
| `--max-players {2,4}` | Expected player count for tracking/BODY selection. Use 4 for doubles clips. |
| `--court-proposals-preview` | Write fail-closed court proposals/correction task when no trusted calibration seed exists; preview only. |
| `--force` | Requests bounded regeneration/cleanup. Exact content identity remains authoritative; `--force` does not make mismatched artifacts current or prove correctness. |
| `--max-frames` | Cap frames for smoke runs only. Do not use capped runs as promotion evidence. |
| `--device` | Device hint for tracking/ReID/pose code that supports it, e.g. `cuda:0`, `mps`, or `cpu`. |
| `--no-global-association` | Skip raw-pool global association after loose-pool tracking. |
| `--global-association-profile` | Explicit internal-val tuning profile. Defaults are not universal proof. |
| `--reid-model` | OSNet ReID checkpoint used by global association. Treat path changes as a new run condition. |
| `--allow-auto-ball-track` | Opt into clip-id discovery of old ball tracks under `runs/`; preview reuse only. |
| `--skip-ball` | Omit ball stage. Downstream ball/event outputs will be absent or degraded. |
| `--skip-audio` | Omit audio onsets from contact fusion. |
| `--rally-gating` | Opt into loose rally-span gating and preserve pre-gating copies. |
| `--placement-keypoints-2d` | Optional native/body 2D keypoints for pre-BODY placement. |
| `--camera-motion` | Reuse a `camera_motion.json` artifact for placement compensation. |
| `--enable-camera-motion` | Force camera-motion estimation on, bypassing the auto decision. |
| `--disable-camera-motion` / `--skip-camera-motion` | Force camera-motion estimation off; this wins over `--enable-camera-motion`. |
| `--camera-motion-estimator {hardened,legacy}` | Select the camera-motion estimator profile. The hardened profile is the default. |
| `--camera-motion-flow-backend {lk,raft-small}` | Select optical flow backend. `raft-small` is flag-gated and does not download weights. |
| `--no-camera-motion-person-mask` | Disable person masking in the camera-motion estimator for ablations/debug. |
| `--no-placement-undistort` | Disable placement-stage pixel undistortion before homography projection. |
| `--mesh-coverage-mode ball_aware` | Default mesh scheduling policy. Uses physically validated ball/contact/proximity triggers, not low-confidence wrist cues alone. |
| `--target-mesh-frame-budget` | Tier-1 deep-mesh frame budget. Use 0 for no cap only in controlled runs. |
| `--ball-proximity-m` | Ball-aware scheduling distance threshold for player-to-arc-solved-ball proximity. |
| `--high-confidence-swing-floor` | Minimum contact-window confidence for swing-triggered ball-aware mesh scheduling. |
| `--body-schedule {serial,overlap}` | BODY scheduling mode. `serial` is the conservative default; `overlap` overlaps CPU pipeline work with remote BODY dispatch and needs fresh run-specific proof. |
| `--no-gpu` | Skip live tracking/pose/BODY unless reuse artifacts are supplied. |
| `--body-local` | Run BODY in-process on a GPU host instead of remote dispatch. |
| `--fetch-body-monoliths` | Opt into fetching/writing the large `smpl_motion.json` and `body_mesh.json` monoliths. Default runs fetch `body_mesh_index/` for replay review instead. |
| `--no-grounding-refine` | Skip BODY grounding refinement. |
| `--placement-trajectory-refine` | Opt into the preview-band placement trajectory artifact after grounding; the existing rev-13 best-stack value is default OFF. |
| `--no-confidence-gate` | Point viewer at raw `virtual_world.json`. |
| `--no-scene-points` | Skip point GLB scene generation. |
| `--confidence-calibration-curves` | Confidence-curve artifact for trust-band calibration; omitted runs use the default only when present. |
| `--manifest` | Override the output replay manifest path/shape consumed by viewer tooling. |
| `--tracker-config` | Tracker runtime config override. Treat as a new run condition. |
| `--verify-viewer` | Start the replay verifier against the produced manifest. |
| `--vite-allow-root` | Root directory the local Vite replay server may serve for off-root manifests/assets. |
| `--no-ball-arc` | Skip the default ball 3D arc stage (auto-bounce anchors -> arc solver -> flight-sanity gate). |
| `--ball-candidates` | Reuse existing `ball_candidates.json` top-K detector sidecars (repeatable). Emitted by default when ball inference runs. |
| `--no-ball-candidates` | Disable default top-K candidate sidecar emission during ball inference. |
| `--wasb-checkpoint` | Explicit WASB checkpoint path for ball inference. Missing or stale runtime config should fail closed, not silently promote. |
| `--wasb-repo` | Explicit WASB repo/runtime path for ball inference. |
| `--json` | Print the full summary JSON instead of a human table. |

## Remote BODY Runtime

The active SAM-3D-Body implementation runs through
`scripts/racketsport/remote_body_dispatch.py`; remote dispatch is the default BODY
path unless `--body-local` or `--no-gpu` is set. The legacy
`--remote-fast-sam-*` flag spellings below are compatibility names for runtime
plumbing; they do not select the separately benchmarked Fast-SAM-3D-Body
challenger, which remains rejected. A completed remote run can still be
`partial`, skeleton-only, or gated by downstream trust bands, so these flags are
not promotion evidence by themselves.

There is no persistent/default remote host. The reviewer-snapshot H100 was a
lane-specific transient worker, not a reusable endpoint. Before using any
remote BODY command, recheck SSH connectivity, disk, `nvidia-smi`, version
stamps, and the shared GPU lock instead of assuming a named VM is live.

There is no RTMW fallback in this path. If SAM-3D-Body is unavailable, BODY is
missing/partial rather than replaced by legacy pose output.

| Flag | Effect |
|---|---|
| `--remote-host` | SSH target for the remote A100 BODY worker. |
| `--remote-ssh-key` | SSH key used for the remote worker connection. |
| `--remote-repo` | Repo path on the remote worker; it must match the committed pipeline layout. |
| `--remote-python` | Python used for remote repo-side orchestration. |
| `--remote-fast-sam-python` | Compatibility-named flag for the Python used by the active SAM-3D-Body remote environment. |
| `--remote-fast-sam-root` | Compatibility-named flag for the active SAM-3D-Body checkout/root on the remote worker. |
| `--remote-lock-wait-timeout-s` | Maximum wait for the shared remote GPU lock. |
| `--remote-command-timeout-s` | Maximum wall-clock time for the remote BODY command after the lock is held. |
| `--sam3d-body-input-size-px` | BODY input-size benchmark/runtime knob. Changing it creates a new run condition. |
| `--sam3d-crop-bucket-sizes` | Cross-frame crop bucket sizes used by the SAM3D body-mode path. |
| `--no-sam3d-wrist-bone-lock` | Disable the default wrist-bone lock. Use only for controlled ablations. |
| `--no-sam3d-torch-compile` | Disable `torch.compile` for the SAM3D body-mode decoder path. |
| `--sam3d-compile-warmup-buckets` | Bucket sizes warmed before timing or running compiled SAM3D decode. |
| `--serialize-tier2-mesh-vertices` | Debug/storage-heavy override that serializes tier-2 mesh vertices instead of joints-only tier-2 output. |

Standalone `remote_body_dispatch.py` debug flags:

| Flag | Effect |
|---|---|
| `--verify-version-stamp` | Remote-side/internal check that the VM repo files match the local BODY version stamp before trusting VM timings or outputs. |
| `--sync-remote-code` | Sync the remote checkout to local HEAD via git bundle and verify the version stamp; it does not run BODY. |
| `--allow-dirty` | Permit dirty tracked runtime files in the version-stamp metadata for local development only; still records the dirty state. |
| `--known-hosts-file` | Pinned SSH known_hosts file. Use `scripts/fleet/refresh_remote_host.sh` when fleet IPs are recycled. |
| `--transport {tar_batch,rsync}` | Remote BODY input/output transport. `tar_batch` is the hardened default; `rsync` is the fallback. |

Shared GPU lock wait and remote command timeout are separate budgets. Raising
one does not raise the other, and either timeout can make the BODY stage partial.
Keep `--serialize-tier2-mesh-vertices` off unless inspecting tier-2 mesh frames;
it increases artifact size and is not needed for normal replay bundles.

## Outputs

The run directory contains stage artifacts and a top-level `PIPELINE_SUMMARY.json`.
Common outputs include:

- `capture_sidecar.json`
- `court_calibration.json`
- `input_quality.json`
- `tracks.json`
- `camera_motion.json` when camera-motion estimation or explicit reuse is active
- `ball_track.json`
- `ball_track_arc_solved.json` (default ball 3D arc, render-only, self-kill gated)
- `ball_flight_sanity.json` and `ball_bounce_candidates.json` (default ball-arc stage)
- `contact_windows.json`
- `frame_compute_plan.json`
- `body_mesh_index/body_mesh_index.json` and chunked mesh files
- `body_stage_phase_timing.json`
- `remote_body_stdout.log` when remote BODY dispatch runs; check it before guessing at VM failures
- `body_full_clip_gate.json`
- `smpl_motion.json` only when `--fetch-body-monoliths` is requested
- `body_mesh.json` only when `--fetch-body-monoliths` is requested
- `racket_pose_estimate.json` when BODY wrist/palm evidence supports the
  estimated-preview paddle stage
- `virtual_world.json`
- `confidence_gated_world.json`
- `trust_bands.json`
- `replay_viewer_manifest.json`
- `match_stats.json` when the BODY+COURT consumer stage succeeds
- `rally_metrics.json`, `coaching_card_facts.json`, and
  `coaching_fact_audit.json` when deterministic fact generation and the
  zero-fabrication audit succeed

`match_stats` and audited deterministic `coaching_facts` run before `manifest`.
The manifest links stats only from a finished current stage and links coaching
facts only after the audit passes. Wording/reference/user-facing product
coaching remains unverified; no language generation runs in this path.

Treat each artifact according to its trust band. Existence and schema validity
are not accuracy proof.

## View your replay in a browser

The viewer takes `replay_viewer_manifest.json`, not `PIPELINE_SUMMARY.json`.
Start the local Vite app from `web/replay`:

```bash
cd web/replay
npm install
npm run dev -- --host 127.0.0.1
```

Open the produced manifest with an absolute `/@fs/` URL:

```text
http://127.0.0.1:5173/?manifest=/@fs/absolute/path/to/replay_viewer_manifest.json
```

If the manifest references files outside the repo root, create/verify the bundle
with `--vite-allow-root <directory>` or start Vite with a matching
`server.fs.allow` policy. The viewer flow is also documented in
`web/replay/README.md`.

## When a run fails - where to look

Start with the top-level and clip-local `PIPELINE_SUMMARY.json` when they exist.
Each stage records `stage`, `status`, `wall_seconds`, notes, artifacts, and
trust badges. `complete` or `partial` means an inspectable bundle exists; it is
not a VERIFIED claim.

If the command failed before options/path validation finished, there may be no
run directory and no `PIPELINE_SUMMARY.json`. In that case, fix the CLI/path
error first instead of searching for missing stage artifacts.

Common traps and fixes:

| Symptom | Check/Fix |
|---|---|
| Browser loads the wrong file | Use `replay_viewer_manifest.json`, not `PIPELINE_SUMMARY.json`. |
| Large BODY monoliths are missing | Default remote BODY fetches `body_mesh_index/`; rerun with `--fetch-body-monoliths` only when `smpl_motion.json` or `body_mesh.json` is required. |
| Remote BODY failed or looks silent | Inspect `remote_body_stdout.log`, then SSH/disk/`nvidia-smi`/lock state. |
| VM timings or outputs look suspicious after a sync/rebuild | Run `scripts/racketsport/remote_body_dispatch.py --verify-version-stamp` before trusting VM numbers; use `--sync-remote-code` when the VM checkout is stale. |
| Recycled fleet IP causes host-key failures | Refresh the pinned known_hosts entry with `scripts/fleet/refresh_remote_host.sh`. |
| Storage audit fails immediately after tests/builds | Rerun with `--ignore-generated-artifacts` or remove fresh generated caches. |

## Status Interpretation

`process_video.py` exit 0 includes `complete` or `partial` by contract. Consumers
must inspect summary status, missing/degraded capabilities, and manifest URLs;
exit 0 alone does not mean ready or verified. The server status core has scoped
proof through the runner, worker, database, and API path; app-side propagation
is code-wired, but the owner-gated physical end-to-end trace remains open.
Promotion still depends on the gates in `NORTH_STAR_ROADMAP.md`.

## Legacy Contract CLI

The duplicate `threed/racketsport/pipeline_cli.py` runner was removed. The sole
runtime entrypoint is `scripts/racketsport/process_video.py`. Its former public
artifact tier/schema metadata and `--public-contracts` readiness validation now
live as data-only contracts in `threed/racketsport/pipeline_contracts.py`,
invoked through `scripts/racketsport/validate_pipeline_artifacts.py`.

## Focused Verification

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_process_video.py -q
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_pipeline_contracts.py -q
.venv/bin/python scripts/racketsport/process_video.py --help
.venv/bin/python scripts/racketsport/doctor.py --json
```

Before claiming the repo is clean or adding another tool, also run the hygiene
checks that guard this simplified layout:

```bash
.venv/bin/python scripts/racketsport/audit_storage_policy.py --root . --ignore-generated-artifacts --json
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py tests/racketsport/test_scaffold_tool_index.py tests/racketsport/test_dead_code_audit.py tests/racketsport/test_storage_policy_audit.py -q
```

These commands are reference and storage checks only. They do not prove semantic
reachability, model quality, device behavior, or E2E verification.

For viewer changes:

```bash
npm test -- --run --dir web/replay
.venv/bin/python scripts/racketsport/verify_process_video_viewer.py --manifest <run>/replay_viewer_manifest.json --out-dir <run>/viewer_verify
```
