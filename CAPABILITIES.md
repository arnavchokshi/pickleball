# Racket-Sport Capability Truth Matrix

Last audited: 2026-06-27.

This file is the single source of truth for what the current pipeline actually invokes. A smoke, probe, precomputed artifact adapter, CPU scaffold, or presence-count check is not `DONE` for a row that names a specific model or algorithm, and it never counts as `VERIFIED`.

Corrected checklist counts after the 2026-06-27 truthfulness reset:

| status | count |
|---|---:|
| DONE | 23 |
| SCAFFOLD | 17 |
| PROTOTYPE-GATE | 3 |
| IN-PROGRESS | 1 |
| VERIFIED | 0 |

| stage | named tech (registry) | actually invoked? | correct variant+weight? | wired into spine? | gate type (accuracy/presence/none) | gate run on real labels? | honest status |
|---|---|---|---|---|---|---|---|
| calibration | manual/ARKit seed plus OpenCV solvePnP, reprojection gate | yes, manual-sidecar calibration path | n/a | yes | accuracy | prototype reviewed-corner runs only; no full DATA-1 gate | DONE, not VERIFIED |
| tracking | YOLO26m plus BoT-SORT-ReID, court filter, N-lock | precomputed detections only | no; no YOLO26m/BoT-SORT spine invocation | yes, scaffold runner | none for spine; future IDF1/spectator gate | no | SCAFFOLD |
| body | Fast SAM-3D-Body, camera-to-court world-SMPL, SAT-HMR/Multi-HMR preview | smoke/probe only | no; Burlington probe used YOLO11n detector, not registry YOLO26m | no | presence_check in current evaluator | no | SCAFFOLD |
| foot/physics | Z=0 foot-lock, CCD-IK, PhysPT, MuJoCo-MJX/PHC/PULSE | CPU primitive/scaffold only | no model/runtime path in spine | no | presence_check in current evaluator | no | SCAFFOLD |
| ball | TrackNetV3, audio pop detector, event fusion, 3D ball physics | smoke/probe only | TrackNetV3 checkpoint was smoke-verified, but no full spine/gate path | no | presence_check in current evaluator | no | SCAFFOLD |
| racket | RTMDet plus SAM2, GigaPose/FoundPose, PnP-IPPE, UKF SE(3) | CPU container/scaffold only | no | no | presence_check in current evaluator | no | SCAFFOLD |
| metrics | biomechanical metrics, confidence calibration, Protocol A/B/C velocity checks | CPU metric primitives only | n/a | no | presence_check in current evaluator | no | SCAFFOLD |
| report/copy | latest Claude copy with facts-in/copy-out faithfulness gate | deterministic template/checker only | no Anthropic/API model call | no | none for LLM generation; local faithfulness checks only | no | SCAFFOLD |
| replay | OpenUSD/USDZ plus GLB export with smplx/pygltflib and compression | manifest assembly only | no real GLB/USDZ generation path | no | presence_check/artifact reference check | no | SCAFFOLD |
| e2e | full pipeline from calibration through replay | calibration plus precomputed tracking only | no | partial; body and downstream runners absent | none; artifact completeness only | no | BLOCKED past tracking |

## Current Spine Reality

`threed/racketsport/orchestrator.py` currently registers two default runners:

| runner | real_model | source_mode | notes |
|---|---:|---|---|
| `ManualCalibrationRunner` | false | `manual_sidecar` | Writes calibration, zones, and net artifacts. Uses real geometry/solvePnP where available, but still depends on reviewed manual sidecar inputs for prototype runs. |
| `PrecomputedTrackingRunner` | false | `precomputed_detections` | Converts existing `detections.json` into `tracks.json`. It does not invoke YOLO26m, BoT-SORT, or ReID. |

`run_pipeline --stage e2e` therefore stops honestly at the first missing downstream runner. The furthest current prototype vertical slice is calibration plus precomputed tracking, then `blocked` at body.

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

- YOLO26m teacher outputs exist in prototype label/autolabel artifacts, but YOLO26m plus BoT-SORT-ReID is not a registered tracking `StageRunner`.
- TrackNetV3 has real H100 smoke windows and an adapter that can run the official predictor when the external runtime exists, but BALL lacks full-clip contact-window generation and label F1/timing gates.
- Fast SAM-3D-Body has a Burlington frame probe with mesh/keypoint tensors, but BODY lacks the camera-to-court world-SMPL conversion and does not write BODY contract artifacts through the spine.
