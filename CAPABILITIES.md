# Racket-Sport Capability Truth Matrix

Last audited: 2026-06-28.

This file is the single source of truth for what the current pipeline actually invokes. A smoke, probe, precomputed artifact adapter, CPU scaffold, or presence-count check is not `DONE` for a row that names a specific model or algorithm, and it never counts as `VERIFIED`.

Corrected checklist counts after the 2026-06-27 truthfulness reset:

| status | count |
|---|---:|
| DONE | 23 |
| SCAFFOLD | 16 |
| PROTOTYPE-GATE | 3 |
| IN-PROGRESS | 2 |
| VERIFIED | 0 |

| stage | named tech (registry) | actually invoked? | correct variant+weight? | wired into spine? | gate type (accuracy/presence/none) | gate run on real labels? | honest status |
|---|---|---|---|---|---|---|---|
| calibration | manual/ARKit seed plus OpenCV solvePnP, reprojection gate | yes, manual-sidecar calibration path | n/a | yes | accuracy | prototype reviewed-corner runs only; no full DATA-1 gate | DONE, not VERIFIED |
| tracking | YOLO26m plus BoT-SORT-ReID, court filter, N-lock | real YOLO26m BoT-SORT-ReID runner is registered; precomputed detections remain explicit manual mode | yolo26m manifest sha256 verified before model load; H100 smoke/gate still required | yes, default real runner plus explicit precomputed runner | spine contract plus future IDF1/spectator gate | no | IN-PROGRESS, not VERIFIED |
| body | Fast SAM-3D-Body, camera-to-court world-SMPL, SAT-HMR/Multi-HMR preview | registered BODY runner exists; local prototype run fails before model load because H100 runtime/checkpoints are absent | code verifies Fast SAM-3D-Body, MHR, MoGe, and yolo26m manifest sha256s before load; local run could not verify files because `/workspace/checkpoints` is absent | yes, registered `BodyStageRunner` writes BODY contracts only from runtime outputs | presence_check in current evaluator; target is world-MPJPE accuracy | no | SCAFFOLD, not VERIFIED |
| foot/physics | Z=0 foot-lock, CCD-IK, PhysPT, MuJoCo-MJX/PHC/PULSE | CPU primitive/scaffold only | no model/runtime path in spine | no | presence_check in current evaluator | no | SCAFFOLD |
| ball | TrackNetV3, audio pop detector, event fusion, 3D ball physics | TrackNetV3 smoke plus no-click TrackNet/VballNet fusion and local-trajectory review filter exist for accepted-four 10-second windows | TrackNetV3 checkpoint was smoke-verified; VballNet ONNX/Keras weights are adjacent verifier models, not approved pickleball checkpoints | no BALL StageRunner yet | presence_check in current evaluator plus held-out prototype benchmark, not an acceptance gate | sparse click benchmark only, no full representative BALL gate | SCAFFOLD |
| racket | RTMDet plus SAM2, GigaPose/FoundPose, PnP-IPPE, UKF SE(3) | CPU container/scaffold only | no | no | presence_check in current evaluator | no | SCAFFOLD |
| metrics | biomechanical metrics, confidence calibration, Protocol A/B/C velocity checks | CPU metric primitives only | n/a | no | presence_check in current evaluator | no | SCAFFOLD |
| report/copy | latest Claude copy with facts-in/copy-out faithfulness gate | deterministic template/checker only | no Anthropic/API model call | no | none for LLM generation; local faithfulness checks only | no | SCAFFOLD |
| replay | OpenUSD/USDZ plus GLB export with smplx/pygltflib and compression | manifest assembly only | no real GLB/USDZ generation path | no | presence_check/artifact reference check | no | SCAFFOLD |
| e2e | full pipeline from calibration through replay | calibration plus tracking; BODY runner is registered but fails closed without H100 runtime/checkpoints | no full-stack variant/gate approval | partial; BODY registered, downstream runners still absent | none; artifact completeness only | no | BLOCKED/FAILED at BODY runtime on local |

## Current Spine Reality

`threed/racketsport/orchestrator.py` currently registers real tracking by default and keeps a manual precomputed mode:

| runner | real_model | source_mode | notes |
|---|---:|---|---|
| `ManualCalibrationRunner` | false | `manual_sidecar` | Writes calibration, zones, and net artifacts. Uses real geometry/solvePnP where available, but still depends on reviewed manual sidecar inputs for prototype runs. |
| `RealYOLO26BoTSORTReIDTrackingRunner` | true | `yolo26m_botsort_reid` | Verifies the manifest `yolo26m` checkpoint sha256 before `YOLO(...)`, then invokes Ultralytics `model.track(..., tracker=botsort_reid.yaml)` with ReID enabled and writes `tracks.json`. |
| `PrecomputedTrackingRunner` | false | `precomputed_detections` | Explicit manual/prototype mode only (`--tracking-mode precomputed`). Converts existing `detections.json` into `tracks.json`. It does not invoke YOLO26m, BoT-SORT, or ReID. |
| `BodyStageRunner` | true | `fast_sam_3d_body` | Verifies `fast_sam_3d_body_dinov3`, `sam_3d_body_mhr_model`, `moge_2_vitl_normal`, and `yolo26m` sha256s before loading, invokes Fast SAM-3D-Body on tracked player boxes, converts camera outputs through `court_calibration.json`, and writes `smpl_motion.json` plus `skeleton3d.json`. |

`run_pipeline --stage e2e` still stops honestly at the first missing or unavailable real runtime. Real tracking fails loudly if the source video, manifest checkpoint, checksum, tracker config, or Ultralytics runtime is missing. BODY now fails loudly if the H100 Fast SAM-3D-Body repo, required checkpoints, or checksum verification is missing. The furthest locally verified prototype vertical slice remains calibration plus precomputed tracking; BODY real inference still needs an H100 runtime smoke and labeled world-MPJPE gate before promotion.

## Gate Labels

Current evaluator gates for body, foot/physics, ball/events, racket, metrics, replay, and e2e are presence or artifact-completeness checks. They must be described as `presence_check` or artifact checks, not accuracy gates, and they do not count toward `VERIFIED`.

The target accuracy gates remain:

| area | real gate required before VERIFIED |
|---|---|
| body | world-MPJPE target, roughly 50-70 mm on labeled evaluation data |
| ball/events | ball F1 at least 0.90 and contact timing within plus/minus 2 frames |
| racket | face-angle error at most 5 degrees against ArUco/GT |
| foot/physics | foot-slide at most 3 mm and zero penetration |
| metrics | metric accuracy against manual review plus confidence calibration |
| replay/e2e | generated assets load correctly and end-to-end SLA/quality gates pass on real clips |

## Real Smokes That Exist But Do Not Promote Status

- YOLO26m plus BoT-SORT-ReID is now a registered tracking `StageRunner`, but the real H100 clip smoke and labeled IDF1/spectator-rejection gate have not passed yet.
- TrackNetV3 has real H100 smoke windows and an adapter that can run the official predictor when the external runtime exists, but BALL lacks full-clip contact-window generation and label F1/timing gates.
- The current strict no-click ball review artifact is `ball_track_fusion_temporal_vball100_localtraj.json` under each accepted clip's `tracknet_smoke_0000_0010/` directory. It improved the four-clip benchmark from 8 teleports to 0 and hidden false positives from 0.425 to 0.294 versus the looser fusion track, with hit recall dropping from 0.563 to 0.509. This is useful for prototype review but does not promote BALL.
- Fast SAM-3D-Body has a Burlington frame probe with mesh/keypoint tensors, and BODY now has a registered checksum-gated StageRunner plus camera-to-court conversion code. It has not yet produced real BODY contract artifacts on H100 through the spine in this reviewed state; the local-only run lacks `/opt/fast-sam-3d-body` and `/workspace/checkpoints/body4d/...`, while the H100 path still needs a fresh BODY spine smoke and world-MPJPE gate before promotion.
