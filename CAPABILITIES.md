# Capability Truth Matrix

Last updated: 2026-07-05.

This file records what the current repo can honestly claim. It is not a run log
and not a place for old experiment narratives. `BUILD_CHECKLIST.md` owns the
small operational board; `RUNBOOK.md` owns commands.

`VERIFIED=0`. Scoped passes remain scoped until the named gate passes.
CAPABILITIES.md is canonical on doc conflicts. The July 2026 GPU state is
reset-pending during winddown; do not treat any named A100 VM as currently
available without a fresh runtime check.

## Canonical Tier Split

This section is the single source of truth for live/server placement. `TIER_MAP.md`
is only a short mirror.

**ON-DEVICE LIVE / fast tier:**

- AVFoundation capture controls and capture-quality guidance.
- ARKit/manual court seed and cached setup metadata.
- Lightweight person detect/track, court map, 2D pose, ball preview, contact cues.
- One conservative live cue plus upload priors.

**SERVER OFFLINE / deep tier:**

- Calibration refinement and deep person tracking.
- Ball/event processing, reviewed gates, and confidence-banded line calls.
- Fast SAM-3D-Body mesh, grounding, foot-lock, and physics refinement. This is
  SAM-3D-Body only: RTMW/RTMW3D/RTMPose are retired because optimized
  Fast SAM-3D-Body gives better pickleball-joint accuracy at equal-or-better
  speed.
- Paddle 6DoF once true-corner/reference GT exists.
- Metrics, replay bake, LLM coaching copy, and week-over-week analysis.

**Borderline rules:**

- Camera-space mesh preview is `server-fast`, not phone-real-time.
- LiDAR is a near-field bonus only. It is not required for v1.
- Offline outputs may be shown as preview only when upstream gates have not passed.

## Capability Matrix

| stage | named tech (registry) | actually invoked? | correct variant+weight? | wired into spine? | gate type (accuracy/presence/none) | gate run on real labels? | honest status |
|---|---|---|---|---|---|---|---|
| calibration | manual/metric sidecar, ARKit target, OpenCV solvePnP, court detector candidates | manual/metric paths can feed `process_video.py`; auto/no-tap solver has not passed; Wave A court-autofind is worktree-patch evidence only | metric sidecar is data, not a model; candidates remain unpromoted | yes for sidecar/metric; preview detector can seed taps only | held-out court PCK/reprojection | yes for prior owner-gate attempts; failed; Wave A aggregate 213.3px misses 200px hard bar | SCAFFOLD/PREVIEW, not VERIFIED |
| tracking | YOLO26m, BoT-SORT/ReID, OSNet, raw-pool association | runner/tooling exist; pre-registered gate runs still fail required coverage/identity/spectator constraints | manifest-checked where runtime uses weights | yes, with explicit reuse and association modes | IDF1, ID switches, spectator/background FP, off-court FP, coverage | yes; no candidate promoted | IN-PROGRESS, not VERIFIED |
| ball | WASB, TrackNetV3 family, audio onset, bounce/in-out, event fusion, default 3D arc chain | runtime/tooling and reviewed-label utilities exist; default 3D chain landed but held-out product F1 0.6969 is below the 0.7248 standing zero-shot bar | checkpoints/candidates are contextual until a gate passes | yes, with reuse, confidence-banded display, and fail-closed arc sanity | reviewed ball F1, bounce/contact timing, in/out agreement | partial reviewed labels exist; acceptance gates have not passed; training campaign concluded negative | SCAFFOLD, not VERIFIED |
| body | Fast SAM-3D-Body, body-frame scheduler, grounding refinement, mesh index export | remote/local BODY paths, mesh index artifacts, speed improvements, and visual smoothing exist for scoped clips | runtime configuration is explicit; candidate-label review is not GT; payload-collapse accuracy isolation is not clean yet | yes, writes BODY/world contracts when inputs exist | independent-GT world-MPJPE | external/candidate paths have not produced a promotion; visual metrics are scoped review evidence | SCAFFOLD, not VERIFIED |
| foot/physics | foot-lock, render-honest ball fill, placement chain | scoped Wolverine internal-val chain exists | deterministic artifacts, no flagship learned dynamics lock | partial, consumed into world artifacts when present | slide/penetration plus protected replay validation | internal-val only | INTERNAL-VAL DONE, not VERIFIED |
| racket | detector/segmenter candidates, PnP-IPPE, fused wrist/palm paddle estimator, future reference-pose stack | phase-1 final_v3 fused estimator exists for render-only review; true reference GT still missing | no production detector or pose model approved | fail-closed render-only candidate path only | face-angle/contact error versus true reference GT | no true-corner/reference GT yet; internal-val only | SCAFFOLD, not VERIFIED |
| metrics | biomech primitives, confidence calibration, report checks | local primitives only | n/a | partial/no user-facing authority | reviewed metric correctness | no | SCAFFOLD, not VERIFIED |
| shot/drill | pose/ball event features and classifier candidates | scaffold/eval utilities exist | no pickleball-approved model | not an authority path | reviewed shot taxonomy accuracy | no | SCAFFOLD, not VERIFIED |
| replay | Three.js review viewer, GLB/USDZ/replay manifest, mesh index viewer path, trust bands | scoped web/native artifact paths load review bundles; current mesh-index visual artifacts are viewer-consumable | scoped artifacts only | yes for review surface; production native path still gated | structural, visual, FPS, native/web playback | scoped visual checks only | SCAFFOLD/SCOPED PASS, not VERIFIED |
| e2e | `scripts/racketsport/process_video.py` | can write complete/partial scoped bundles with fail-closed summaries | no full-stack approved variant set | yes, as the glue entrypoint | artifact completeness plus every component quality gate | no full clean gate-passing clip yet | SCAFFOLD/SCOPED PASS, not VERIFIED |

## Current Spine Reality

`scripts/racketsport/process_video.py` chains ingest, calibration, tracking,
placement, rally gating, frames, ball, the default ball-arc stage, events,
ball fill, BODY, placement refinement, grounding refinement, world, confidence
gating, manifest generation, and optional viewer verification. It reuses valid
artifacts unless `--force` is set, degrades missing optional stages into
trust-banded gaps, and writes `PIPELINE_SUMMARY.json` for both complete and
partial runs.

BODY dispatch is remote by default unless `--body-local` or `--no-gpu` is used,
but the winddown state is reset-pending and any remote host must be rechecked
before use. Current BODY outputs fetch `body_mesh_index/` for replay by default;
large `smpl_motion.json` and `body_mesh.json` monoliths are opt-in with
`--fetch-body-monoliths`.

The important honesty boundary is that `complete` means "a bundle was produced,"
not "the underlying CV has passed product gates."

## Protected Data Policy

- Outdoor and Indoor are strict held-out clips.
- Burlington and Wolverine are internal-val only when explicitly opted in.
- Roboflow/public corpora may be training data only after provenance/dedup checks.
- Candidate predictions, copied labels, and box-derived paddle candidates are not
  independent ground truth.

## Promotion Vocabulary

| Word | Meaning |
|---|---|
| `VERIFIED` | The documented acceptance gate passed on required current evidence. No row currently qualifies. |
| `SCOPED PASS` | A specific command/artifact/browser/device slice passed. It must name its scope. |
| `INTERNAL-VAL DONE` | A useful internal-val result. It does not promote held-out or product claims. |
| `SCAFFOLD` | Code/contracts/tests exist, but quality gates or runtime proof are missing. |
| `PREVIEW` | Usable for visual inspection or UX only; not authoritative. |
