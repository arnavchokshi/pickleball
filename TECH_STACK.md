# Technical Registry

Last updated: 2026-07-05.

`models/MANIFEST.json` is the checkpoint registry. This document explains where
the main technologies live and how to treat them. It is not promotion evidence.

## System Split

| Tier | Responsibility | Main code |
|---|---|---|
| iOS live tier | capture controls, setup guidance, lightweight preview cues, upload priors, native playback shell | `ios/` |
| Python offline tier | calibration, tracking, ball/events, BODY, world, replay manifest, gates | `scripts/racketsport/`, `threed/racketsport/` |
| Web review tier | inspect `replay_viewer_manifest.json` and trust-banded worlds | `web/replay/` |

## Stage Registry

| Stage | Primary tech/candidate | Current code surfaces | Truth boundary |
|---|---|---|---|
| Capture/iOS | AVFoundation, ARKit/manual sidecar, CoreMotion, Core ML/Vision candidates | `ios/Capture`, `ios/Calibration`, `ios/FastTier`, `ios/Guidance`, `ios/Upload`, `ios/Replay` | Swift tests and device slices are scoped unless a full physical capture/live/replay gate is run. |
| Calibration | manual/metric sidecar, OpenCV solvePnP, court detector candidates | `threed/racketsport/court_*`, `scripts/racketsport/evaluate_court_*`, `scripts/racketsport/process_video.py` | Tap/metric seed can run; no no-tap solver is verified. |
| Person tracking | YOLO26m, BoT-SORT/ReID, OSNet, raw-pool/global association | `threed/racketsport/orchestrator.py`, `offline_person_authority.py`, `raw_pool_person_authority.py`, `person_track_gt_scoring.py` | Gate failures remain failures even if one clip looks strong. |
| Ball/events | WASB/TrackNet family, default 3D arc chain, audio onset, bounce/in-out, physics fill, event fusion | `threed/racketsport/ball_*`, `event_fusion.py`, `rally_gating.py`, `scripts/racketsport/build_*ball*`, `process_video.py` | Confidence-gated preview only until reviewed F1/contact/in-out gates pass; default arc chain remains fail-closed and not promoted. |
| BODY | Fast SAM-3D-Body, frame scheduling, remote A100 dispatch, mesh index export, grounding refinement | `scripts/racketsport/remote_body_dispatch.py`, `run_sam3dbody_*`, `threed/racketsport/body_*`, `process_video_body_frames.py` | Candidate-label review is not independent GT; speed/visual artifacts are scoped evidence only. |
| Foot/physics | deterministic foot-lock/grounding, render-honest ball fill, placement chain | `threed/racketsport/foot_*`, `body_grounding_refine.py`, `ball_physics_fill.py`, `virtual_world.py` | Internal-val pass is not product verification. |
| Paddle | detector/segmenter candidates, PnP-IPPE, fused wrist/palm estimator, future reference-pose stack | `threed/racketsport/racket_*`, `paddle_proxy.py`, `paddle_pose_fused.py`, `scripts/racketsport/rkt_*`, `scripts/racketsport/build_paddle_pose_fused.py` | Fused paddle output is render-only until true reference GT exists; box-derived candidates cannot become verified 6DoF claims. |
| Metrics/shot/report | biomech primitives, shot candidates, report faithfulness checks | `threed/racketsport/*metrics*`, `shot_taxonomy.py`, `report_*` | No user-facing authority until reviewed labels and faithfulness gates pass. |
| Replay/world | `virtual_world.json`, confidence/trust bands, GLB/USDZ/replay manifest, Three.js/RealityKit targets | `threed/racketsport/virtual_world.py`, `replay_*`, `web/replay`, `ios/Replay` | Viewer load and schema validation are scoped checks, not full replay verification. |

## Runtime Notes

- The default full pipeline entrypoint is `scripts/racketsport/process_video.py`.
- BODY dispatch is remote A100 by default unless `--body-local` or `--no-gpu` is used,
  but the July 2026 GPU state is reset-pending during winddown. Recheck SSH,
  disk, `nvidia-smi`, and the GPU lock before treating any host as available.
- RTMW/RTMW3D/RTMPose are retired. The offline BODY stack is SAM-3D-Body only:
  Fast SAM-3D-Body replaced the RTMW family because its accuracy is better on
  the pickleball body/joint surface and the optimized path is equal-or-better
  speed.
- `--mesh-coverage-mode ball_aware` is the default scheduling policy for tier-1
  mesh frames. It should use validated ball/contact/proximity triggers, not raw
  low-confidence wrist cues alone.
- `--fetch-body-monoliths` is opt-in. Default BODY fetches replay-consumable
  `body_mesh_index/` artifacts instead of large `smpl_motion.json` and
  `body_mesh.json` monoliths.
- `--body-schedule=overlap` is an optimization mode, not a promotion claim;
  the conservative schedule remains `serial` unless a run proves overlap for
  that exact condition.
- The default ball chain runs auto-bounce anchors, arc solve, and flight sanity
  unless `--no-ball-arc` is set. WASB inference still needs explicit/current
  repo and checkpoint configuration when live detection is required.
- Court auto-find Wave A is patch/worktree evidence only; the code on main must
  not be described as having a promoted no-tap solver from that work.
- `--allow-auto-court-corners-preview` and `--allow-auto-ball-track` are preview
  convenience paths. They must not become promotion evidence.
- Protected eval policy belongs in code and tests, not only docs.

## Where To Add New Code

Before adding a new CLI, use `scripts/racketsport/list_scaffold_tools.py`:
`python scripts/racketsport/list_scaffold_tools.py --root .`. It is the
category/workstream map for every checked-in CLI. New tools should land with a
direct CLI-reference test and should not remain in the `unknown` category.
Before deleting or calling Python source dead, run
`python scripts/racketsport/audit_dead_code.py --root . --json`. It reports
dead-code candidate files that lack an exact import, path reference, CLI
reference, or matching focused test in the current tracked-plus-untracked
worktree. Treat that report as a candidate finder, not semantic proof that code
is safe to delete without reading it.

| Need | Preferred location |
|---|---|
| New CLI for an existing stage | `scripts/racketsport/` plus `tests/racketsport/test_<stage>.py` |
| Shared stage logic | `threed/racketsport/<stage>.py` |
| Artifact schema | `threed/racketsport/schemas/__init__.py` and matching JSON schema under `docs/racketsport/` when public tooling needs it |
| iOS live/capture behavior | The matching `ios/<Module>/Sources/Pickleball<Module>/` package |
| Web replay behavior | `web/replay/src/` with Vitest coverage |

## Verification Surfaces

| Surface | Command |
|---|---|
| Python docs/truth invariants | `.venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py -q` |
| Process pipeline contracts | `.venv/bin/python -m pytest tests/racketsport/test_process_video.py tests/racketsport/test_pipeline_contracts.py -q` |
| iOS packages | `swift test --package-path ios` |
| Web replay | `npm test -- --run --dir web/replay` |

Run broader suites when changing shared schemas, manifest contracts, or pipeline
stage behavior.
