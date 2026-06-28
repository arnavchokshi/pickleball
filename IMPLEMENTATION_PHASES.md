# Pickleball/Tennis CV Pipeline: Codex Implementation Plan

**Audience:** the Codex coding agent.
**Status:** target build spec plus current implementation delta. Follow the build order and gates exactly, but do not read target phase text as proof that the feature already exists.
**Companion docs:** `SWAY_BODY_PICKLEBALL_MVP.md` (product), `TECH_STACK.md` (technology rationale), `ACCURACY_AND_TRAINING.md` (**source of truth for datasets, training/fine-tune/auto-label recipes, filtering parameters, and the numeric validation protocols A/B/C/D referenced by the gates below**), `BUILD_CHECKLIST.md` (**the operational checklist + multi-agent coordination protocol — Codex drives the build from there; each checklist task maps to a phase here**). World-grounded body, foot-skate elimination, physics refinement, racket 6DoF, and the 3D replay renderer are built and gated in Phases 3/4/6/10 below. **`TECH_STACK.md §2.3 Model Registry` is the canonical source for exact model variants + weights (offline vs live, candidates, fallbacks, licenses) — this doc must not state any variant that contradicts it; on any naming discrepancy the registry wins.**
**Date:** 2026-06-26.

> **Round-4 architecture (licensing lifted — research/personal use):** we are not selling now, so the most accurate models are used regardless of license; license is recorded as *informational* for future commercialization, not as a blocker. This **flips the prior skeleton-first decision: SMPL/SMPL-X mesh is now CORE** (world-grounded body for the physics-accurate 3D replay + foot-skate elimination); the fast positional skeleton is demoted to a **preview/triggering overlay**. New subsystems: **foot-skate elimination + physics refinement**, **racket 6DoF tracking**, and a **3D replay renderer** — all detailed in Phases 4/6/10 below.

> **Accuracy doctrine (from `ACCURACY_AND_TRAINING.md`):** heavy/multi-view/physics/audio models are *teachers* used at training/labeling time to manufacture ground truth; we distill into fast single-camera *students* that ship. Calibrated court geometry is a free accuracy multiplier (reprojection loss, metric root depth, foot-on-ground constraint) and — critically — it gives world-grounded HMR the known camera + ground plane those methods normally have to estimate (their biggest weakness). Validate, then claim — every metric ships only after passing its validation protocol; until then it is gated.

> **Round-5 — single-camera scope + iOS client/server split.** The product is **single-camera**: one rear iPhone camera on a tripod. **Multi-camera is a FUTURE product capability** — the *only* multi-camera use today is **training-time** (the 2-camera shoot manufactures 3D ground truth to distill a single-camera model; see Data & Training Infra). Do not build any product feature that requires two cameras at inference. This is a **native iOS Swift app + GPU server**:
> - **iPhone (client)** — captures (AVFoundation, locked exposure/focus, high fps), runs an **on-device fast-tier preview + capture guidance** (Apple Vision + a small Core ML model on the Neural Engine), and **plays the 3D replay** (RealityKit/USDZ). Produces the **capture sidecar** (camera intrinsics + ARKit pose + court-plane seed + on-device pose prior + optional LiDAR depth) that the server consumes.
> - **Server (H100)** — the heavy **deep tier**: Fast SAM-3D-Body → our world-grounding → foot-lock → physics → racket 6DoF → render-bake (USDZ + GLB).
> The **Client Track (iOS App — Phases C1–C3)** is defined right before Phase 0 (it sets the input contract); the server **Pipeline Track is Phases 0–11**. Both tracks appear in the One-Page Build Order Summary.

---

## 0. How To Use This Doc

### 0.0 Current implementation reality (2026-06-28 audit)

This file describes the system we are building. The implementation is not there yet. The current checklist truth is `VERIFIED=0`: no phase has passed its documented real-clip acceptance gate. The rows below are the current state to use before reading the target phase details.

| area | current evidence | next gate that actually promotes it |
|---|---|---|
| repo/history | `main` and `origin/main` were at `22fba00`; strict last-12-hour commit window had 0 commits, while the relevant wider burst had 43 commits; local dirty tree was much larger than pushed main | separate pushed code, local dirty code, and ignored `runs/` artifacts before claiming status |
| iOS client | Swift package/app target builds and Swift tests pass; IOS-1 has partial AVFoundation preview/recording runtime code, but sidecar writing, live-frame feed, fps/shutter/luminance measurement, codec validation, and physical-device gates are missing; IOS-2..IOS-6 remain scaffold-level | implement the missing AVFoundation/ARKit/Vision/CoreML/RealityKit/Upload runtime paths on a device and pass C1-C3 gates |
| court calibration | sidecar/manual PnP, regulation zones/net, overlay/evidence plumbing, fail-closed court-line evidence, and pipeline semantic blockers for unready court evidence exist | implement/train no-tap solver, then pass Phase 1 gates on real labeled clips |
| tracking | YOLO26m plus BoT-SORT-ReID runner is registered and fail-closed; precomputed/manual mode remains explicit. Canonical accepted-four tracks are still too sparse for BODY scheduling (Burlington 2/180 four-player frames, Wolverine 1/91, Outdoor 1/1786, Indoor 0/900). H100 source-30 strict candidates improved Outdoor to 864/1800 and Indoor to 424/900 four-player coverage and audit as `canonical_candidate_not_gate_verified`; widened-margin diagnostics audit as `diagnostic_only`. | run the real H100 labeled gate for IDF1, spectator rejection, ID switches, and throughput before TRK promotion |
| body | BodyStageRunner exists and accepted-four scheduled-frame H100 smoke produced mesh/joint artifacts for Burlington, Wolverine, and Indoor; `body_gate_report.json` is still `blocked` because reviewed world-MPJPE labels and a full-clip BODY gate are missing, and Outdoor has no scheduled BODY output. Draft/not-ground-truth BODY labels now fail closed instead of measuring MPJPE. | canonical-safe BODY scheduling, reviewed/equivalent world-joint labels, full-clip BODY gate, then world-MPJPE and foot/NVZ gates |
| ball/events | strict no-click ball review tracks and cue artifacts exist; canonical accepted-four contact windows were promoted from saved human review inputs, not complete machine cue triplets. `contact_timing_review_alignment.json` now proves 34/34 saved review contacts match promoted windows at 0-frame delta, but `ball_verified=false`. | audio/wrist/ball cue coverage, label F1, machine contact timing, and +/-2-frame timing gates |
| racket | PnP/IPPE primitives, preview overlays, true-corner review crop sheets, and fail-closed audits exist; all accepted-four candidates are still box-derived, have 0 true-corner labels and 0 reference/GT labels, and cannot promote to canonical `racket_pose.json` | real paddle GT/assets or human true-corner labels, true detector/keypoint/CAD pose runtime, face-angle/contact gates |
| report/visualization | report model and confidence primitives exist; visualization modules emit CPU payloads/metadata, not final rendered delivery | rendered court/ghost/overlay assets, SSE/tiered delivery, and copy faithfulness gates |
| replay | static review GLBs/manifests, `replay_readiness_report.json/html`, and a web parser/summary exist; all four accepted clips are review-visual-ready but production replay and metrics remain blocked. Static GLBs are now classified as `review_static_glb_export` and rejected by production validation. | production animated GLB/USDZ export, real Three.js/R3F viewer, RealityKit playback, visual/perf gates after BODY/BALL/RKT gates are trustworthy |
| data/eval | validators, manifests, and hardening exist; real data assets and several numeric gates are missing | populate reviewed datasets and replace presence checks with the documented numeric phase gates |

### 0.1 Phase-gate discipline (mandatory)

This pipeline is built in ordered phases. **Do not start phase N+1 until every acceptance gate in phase N passes on the GPU test clips.** Each phase ends with a `REPORT.md` written to `runs/phaseN/REPORT.md` containing the measured numbers against each gate, marked PASS/FAIL. If a gate fails, fix or escalate; do not silently proceed. A failed gate that cannot be met is a product decision, not a coding decision — write `BLOCKED` with the measured number and the reason.

### 0.2 Repo assumptions

- Work in this `pickleball` repo on `main`; do not treat the older `sam4dbody` tree as the live codebase for this build. Historical `sam4dbody`/WP references are useful background only, not current file ownership.
- Reuse the current local modules and patterns:
  - `threed/racketsport/court_calibration.py`, `court_templates.py`, `court_zones.py`, `net_plane.py`, and `court_line_evidence.py` for calibration geometry and semantic evidence.
  - `threed/racketsport/orchestrator.py`, `pipeline_contracts.py`, and `scripts/racketsport/validate_pipeline_artifacts.py` for fail-closed stage wiring and artifact readiness.
  - `threed/racketsport/eval/` and `tests/racketsport/` for phase evaluators and regression tests.
  - `scripts/gpu-eval-run.sh`, `scripts/gpu-train-lock.sh`, and `models/MANIFEST.json` for shared-H100 coordination and model inventory.
- All server code lives under `threed/racketsport/` and tests under `tests/racketsport/`. The 3D replay **viewer** is a separate web frontend (Three.js / React Three Fiber) under `web/replay/`. Module layout (create files as phases require them):

```
threed/racketsport/
  court_templates.py      court_calibration.py     court_zones.py
  net_plane.py            capture_quality.py       drift_guard.py
  person_fast.py          track_lock.py            doubles_id.py
  hmr_fast.py             hmr_deep.py              worldhmr.py
  skeleton3d.py           footlock.py              physics_refine.py
  body_mesh_readiness.py
  ball_tracknet.py        ball_tap_track.py        ball_physics3d.py
  audio_pop.py            event_fusion.py          contact_windows.py
  racket6dof.py           movement_metrics.py      biomech.py
  insight_rules.py        confidence.py            shot_classifier.py
  drill_verify.py         habit_model.py           report_model.py
  llm_copy.py             replay_readiness.py      replay_export.py
  virtual_world.py        pipeline_contracts.py    orchestrator.py
  review_packet.py        replay_viewer_manifest.py
  schemas/                eval/
web/replay/   # Three.js + React Three Fiber web-share viewer (Phase 10)
ios/          # native Swift app (Client Track C1–C3)
  Capture/Sources/PickleballCapture/              # AVFoundation capture target/module scaffold
  Calibration/Sources/PickleballCalibration/      # ARKit/manual-tap calibration target/module scaffold
  FastTier/Sources/PickleballFastTier/            # on-device preview/guidance target/module scaffold
  Guidance/Sources/PickleballGuidance/            # capture-quality guidance target/module scaffold
  Replay/Sources/PickleballReplay/                # RealityKit/USDZ replay target/module scaffold
  Upload/Sources/PickleballUpload/                # upload target/module scaffold
  App/                                            # Pickleball iOS app target
models_coreml/  # YOLO (+ any small student) exported to Core ML (coremltools 9.0, INT8/ANE)
```

### 0.3 Environment

- One H100-class GPU. Python 3.11, CUDA 12.x, PyTorch ≥2.4 (TensorRT + `torch.compile`), `onnxruntime-gpu`, `opencv-python`, `kornia`, `ffmpeg` with NVDEC, `librosa`, `xgboost`, `mmcv`/`mmpose`/`mmdet` (OpenMMLab), `ultralytics` (YOLO26 — AGPL, license noted for future commercialization), `scipy`, `filterpy` (Kalman/UKF), `smplx` + SAM-3D-Body/Fast SAM-3D-Body (MHR→SMPL), `trimesh`/`pyrender` (mesh), `mujoco` + `mujoco-mjx` (physics sim), `pydantic` (schemas), `anthropic` (LLM copy). Web viewer: Node + `three` (r171+) + `@react-three/fiber`/`drei`/`rapier`, `@gltf-transform/cli`.
- **H100 performance substrate** (the runtime everything rides on; full rationale in `TECH_STACK.md §H100 runtime & serving`): **TensorRT 10.x** (frozen-model engines), **NVIDIA DALI** + **NVDEC** (GPU video decode — fixes CPU-decode starvation, the #1 bottleneck), **BF16 AMP** (all training; prefer over FP16 on H100), **`torch.compile`** (`max-autotune`, for fast-iterating code), **CUDA graphs/streams** (fixed-shape + decode→infer→postproc overlap), **Triton Inference Server** (ensemble DAG serving the multi-model pipeline on one box — skip Ray/Modal/Bento), **DCGM** (`dcgmi`) for GPU telemetry. Deep-tier delivery: **FastAPI + SSE**, **Redis** (Pub/Sub), **Celery** (GPU worker queue), **APNs** (push). iOS upload: **TUSKit** (resumable).
- **Benchmark Fast SAM-3D-Body per-player-frame FIRST (Phase 0).** It is the pipeline's pacing item; its measured VRAM + FPS set the MIG geometry (`BUILD_CHECKLIST.md §1.5`) and the deep-tier latency budget. Do not size anything else before this number exists.
- Pin everything in `requirements-racketsport.txt` (+ `web/replay/package.json`). One env, reproducible. Record `pip freeze` into each `runs/phaseN/REPORT.md`.
- **iOS client toolchain (Client Track):** Xcode 16+ / Swift 6, deployment target iOS 18+ (RealityView virtual camera) with iOS 17 fallbacks (Vision 3D pose, `videoRotationAngle`); frameworks **AVFoundation, ARKit, Vision, CoreML, RealityKit, CoreMotion**. **Model export:** `coremltools` 9.0 (PyTorch→Core ML, INT8/W8A8 for the ANE) to produce `models_coreml/`. **Replay bake (server, Python):** OpenUSD `pxr` (USDZ/UsdSkel) + `pygltflib` + `smplx` (GLB) — author USDZ and GLB **directly from the reconstruction**, do not round-trip one into the other.

### 0.4 Model checkpoints to fetch (Phase 0)

License is **informational** (for future commercialization), not a blocker. Flag bleeding-edge items **[VERIFY]** = confirm repo/license/benchmarks resolve before relying on them; an established fallback is named.

| Model | Use | License (info) | Source / note |
|---|---|---|---|
| **YOLO26m** (offline) / **YOLO26n** Core ML (live) | person detection | AGPL | ultralytics — real (Jan 2026), NMS-free, +1.4–2.8 mAP over YOLO11 at equal latency. Offline default **26m** (accurate enough for ≤4 big players — tracking-limited, not detection-limited); benchmark **26l/26x** only if far-court recall is low. Live **26n** on the ANE (Core ML INT8); **tune with official defaults** |
| **BoT-SORT-ReID** | tracker (PRIMARY) | MIT | native to ultralytics (`tracker=botsort.yaml`); appearance resolves doubles crossings |
| ByteTrack | tracker (simpler fallback) | MIT | github ifzhang/ByteTrack — fine for well-separated players |
| RF-DETR-L / RTMDet | detector (Apache alt for future commercial) | Apache-2.0 | roboflow/rf-detr; OpenMMLab `mmdet` |
| RTMPose-m / RTMW-l | 2D whole-body pose | Apache-2.0 | OpenMMLab `mmpose` — **stays (already optimal)** |
| **Fast SAM-3D-Body** (`facebook/sam-3d-body-dinov3` + `yangtiming/Fast-SAM-3D-Body`) | **per-frame mesh backbone (DEEP tier PRIMARY)** | SAM License ⚠️ **verify-commercial** | best per-frame mesh (3DPW PA-MPJPE 30.4 mm); MHR→SMPL via built-in MLP; ~15 FPS/person (RTX 5090 — ⚠️ benchmark on H100). **This is the body core.** License-safe fallback if commercializing: **SAT-HMR (Apache)** → NLF → HMR2.0 |
| SAM-3D-Body (original) | per-frame mesh (superseded by Fast variant) | SAM License ⚠️ verify-commercial | facebookresearch/MHR; 0.8 FPS — use Fast version |
| GVHMR / WHAM / TRAM | **optional world-trajectory + foot-velocity sanity-check only** | ZJU-NC / MIT / MIT | NOT the primary mesh — their per-frame backbone is HMR2.0 (worse than SAM-3D-Body); we already solve world-grounding via calibration |
| **SAT-HMR / Multi-HMR 2** | fast multi-person camera-space SMPL (FAST tier) + license-safe body fallback | Apache (SAT-HMR) / Naver ⚠️ (Multi-HMR 2) | **SAT-HMR** `sat_644_3dpw.pth` (Apache, 24 FPS, assumes 60° FOV — verify per court cam) is the license-safe pick; **Multi-HMR 2** (no FOV assumption) is the accuracy candidate but Naver-proprietary; fallback HMR2.0/4D-Humans |
| RTMW3D-l / MotionBERT | 3D skeleton (preview/triggering overlay only) | Apache-2.0 | OpenMMLab / Walter0807 |
| **PhysPT** | physics-aware motion refinement (no engine at inference) | MIT | github zhangy76/PhysPT — default plausibility pass |
| PHC / PULSE + SMPLOlympics | imitation controller on MuJoCo (flagship deep replay) | BSD-3 / open | github ZhengyiLuo/PHC, /PULSE; SMPLOlympics tennis envs |
| MuJoCo + MJX | physics engine | Apache-2.0 | google-deepmind/mujoco, mjx |
| MultiPhys | inter-player non-penetration (doubles) | CC BY-NC-SA | github nicolasugrinovic/multiphys |
| UnderPressure | learned foot-contact classifier (option) | research | github InterDigitalInc/UnderPressure |
| RacketVision (RTMDet+RTMPose) | racket detect + top/bottom/handle keypoints | check repo | github OrcustD/RacketVision (2D only; we add 6DoF) |
| GigaPose / FoundPose | coarse 6DoF object-pose init for racket | open | nv-nguyen/gigapose; FoundPose (arXiv 2311.18809) |
| TrackNetV3 (now) / TrackNetV5 [VERIFY] | ball detector | MIT / check | github qaz812345/TrackNetV3; V5 arXiv 2512.02789 (on release) |
| SAM2 | paddle silhouette segmentation | Apache-2.0 | facebookresearch |
| TennisCourtDetector | court-keypoint base (fine-tune → pickleball) | none stated | github yastrebksv/TennisCourtDetector |
| BioPose (NeurIK) | anatomically-valid joint limits (option) | research | arXiv 2501.07800 |

**Datasets to download (full table with sizes/licenses/URLs in `ACCURACY_AND_TRAINING.md §9`):** body fine-tune order — **BEDLAM2 → AthletePose3D → CalTennis [VERIFY] → RICH (contact labels) → AMASS**; also ASPset-510 (CC0), SportsPose, H3WB, Human3.6M, EMDB(2) (world-trajectory eval). Ball — InAPickle (18k) + ball-tracking (11k) + other pickleball Roboflow sets (~50k), TrackNet tennis (36.9k), OpenTTGames (120 fps), BlurBall TT, badminton Shuttlecock. Court — TennisCourtDetector (8.8k) + pickleball court-keypoint Roboflow sets. Audio aug — MUSAN/ESC-50/FSD50K/DEMAND. Racket — RacketVision (tennis) + Roboflow paddle + ~50k synthetic paddle-CAD (BlenderProc) + ~5k ArUco-GT real frames. Record each in `models/MANIFEST.json` with license + sha256.

**Licensing posture (informational; canonical in `TECH_STACK.md §2.3`):** licensing is **lifted for research/personal use now** — use the most accurate model regardless of license. The only commercial-risk items are **Fast SAM-3D-Body (SAM License — ⚠️ verify before commercial)** and **Multi-HMR 2 (Naver-proprietary)**; the license-safe body swap is **SAT-HMR (Apache)**. Other non-commercial models (SMPL/SMPL-X, AMASS, BEDLAM2, MultiPhys, AthletePose3D; GVHMR only as an optional cross-check) and AGPL (YOLO26) are **used freely now**. Record each license in `MANIFEST.json` so a future commercial pivot knows the swaps (YOLO26→RF-DETR/RTMDet (Apache); SAM-3D-Body→SAT-HMR (Apache); SMPL→a commercial body model). Do not let license block accuracy today.

**On-device (iOS) frameworks — no server checkpoint, run on the Neural Engine (Client Track):**

| Framework / model | Use (on-device) | Notes / limits |
|---|---|---|
| **Apple Vision** `VNDetectHumanBodyPoseRequest` | 2D pose, ≤4 players, fast-tier preview | 19 joints, multi-person, ~60 FPS on ANE — **preview/guidance only** |
| **Apple Vision** `VNDetectHumanBodyPose3DRequest` | coarse 3D pose preview / capture guidance | **17 joints, single-person, no fingers/foot-orientation; metric scale needs LiDAR; ~5–25° joint error** — NOT a coaching metric |
| **Apple Vision** `VNDetectHumanHandPoseRequest` | grip cue (21 landmarks) | the only on-device finger path (Vision 3D body has no hands) |
| **Apple Vision** `VNGeneratePersonSegmentationRequest` | framing/silhouette in capture UI | fast quality tier is real-time |
| **YOLO (Core ML)** | ball/racket fast detection (preview) | export via coremltools 9.0, INT8/ANE; keep all loaded on-device models **< 500 MB** total; use `EnumeratedShapes` to stay on ANE |
| **ARKit** `ARWorldTrackingConfiguration` | camera intrinsics + 6DoF pose + horizontal court-plane (calibration seed) | cannot run with a high-fps `AVCaptureSession` simultaneously → ARKit *setup pass*, then switch to AVFoundation recording |
| **ARKit** `sceneDepth` (LiDAR, 12 Pro+) | near-camera depth assist (foot-contact, near court plane) | **~5 m range, fails in sun** → Tier-A bonus within range only; useless for far court/ball |
| **RealityKit + USDZ / OpenUSD** | native in-app 3D replay (RealityView virtual camera, iOS 18+) | **SceneKit is soft-deprecated (WWDC25) — use RealityKit**; AR Quick Look = one skeleton/scene (RealityView handles doubles) |

The server deep tier (Fast SAM-3D-Body + physics + racket + render-bake) is **the source of truth**; on-device models are preview/guidance only.

### 0.5 Reporting format (per phase)

`runs/phaseN/REPORT.md`: env hash, clips used, per-gate measured value, PASS/FAIL, runtime/GPU-time per clip, artifact paths. Also emit `runs/phaseN/metrics.json` (machine-readable) for the regression dashboard (§ Cross-Cutting).

### 0.6 Parallelization & running tests on the ONE shared H100

Codex (lead) hands work to subagents that build **in parallel**, but **all agents share a single H100 (GCP A3) for testing/training.** GPU is the contended resource — the full protocol (MIG geometry, `flock` lease/queue, VRAM budget, per-task `[GPU]`/`[CPU]` tags) lives in `BUILD_CHECKLIST.md §1.5` (follow it; not duplicated here). The rules that shape this plan:

- **Every task is tagged `[CPU/IO]` or `[GPU]`** in `BUILD_CHECKLIST.md`. **`[CPU/IO]` tasks NEVER touch the GPU lease and run fully in parallel** — iOS app (C1–C3), `web/replay/` viewer, all `schemas/`, dataset download/convert, module scaffolding, CPU post-processing (e.g. MuJoCo runs on CPU), report/viz assembly, doc/CI wiring. Maximize parallelism here.
- **`[GPU]` tasks serialize through the GPU lease/queue** — model training/fine-tune, heavy inference, and **every phase Acceptance-Gate test run** (they execute models on test clips). Acquire the lease → run → release. Light eval can run in parallel **MIG slices**; full-GPU training takes the exclusive lease. See `BUILD_CHECKLIST.md §1.5` (helper `scripts/gpu-eval-run.sh` is defined in Phase 0).
- **Running tests is mandatory and is itself a GPU job.** A phase is `VERIFIED` only after its Acceptance-Gate commands are **actually run on the shared H100** (via the lease), the measured numbers are written to `runs/phaseN/REPORT.md` + `metrics.json`, and they meet the gate. Never mark a gate passed without the recorded run.
- **What can proceed concurrently vs what must wait:** the Client Track (C1–C3, mostly `[CPU/IO]` + on-device, no server GPU) and the Data & Training Infra track run alongside the server pipeline from the start. Within the server pipeline, **phase-gate order still holds** (Phase N+1 waits for Phase N `VERIFIED`), but the *coding* of a later phase's `[CPU/IO]` parts (schemas, scaffolding, CLI wiring) may begin early — only its `[GPU]` gate test waits for upstream artifacts. **Benchmark Fast SAM-3D-Body per-player-frame in Phase 0** before sizing anything else — it paces the pipeline and decides the MIG geometry (eval `2×3g.40gb` vs training `1×7g.80gb`).

### 0.7 Model variant selection (benchmark, don't assume — task EVAL-0)

The exact model variants in `TECH_STACK.md §2.3 Model Registry` are **defaults + candidates, not final**. ⚠️ **Every speed number in this repo is a T4/A100/RTX extrapolation — there are no published H100 latencies.** Before a stage's variant is treated as final, **task EVAL-0** (runs early, alongside Phase 0–3; a `[GPU]` job via the §1.5 lease) benchmarks the registry's candidate variants and locks the winner in `models/MANIFEST.json`.

Procedure per tradeoff stage:
1. Run each candidate variant on the **real test-clip set** (the varied-camera matrix) on the H100.
2. Measure **accuracy vs the phase Acceptance Gate** AND **latency (ms/frame, batch=1 and batch=4) + VRAM** on the actual H100.
3. **Render side-by-side COMPARISON VIDEOS** — each candidate's actual output overlaid on the **same ≥3 real clips** (detection boxes / pose / SMPL mesh / ball track / foot-lock before-after / racket 6DoF / replay), labeled with name + gate metric + latency + VRAM → `runs/eval0/<stage>/compare/*.mp4` + `runs/eval0/<stage>/variant_selection.md`.
4. **HUMAN APPROVAL GATE (`BUILD_CHECKLIST.md §1.6`):** do **not** silently finalize. The agent may **auto-finalize only if one variant is obviously better** (passes the gate AND Pareto-dominant on accuracy+speed, OR all others fail the gate, OR wins by a large clear margin with no meaningful speed/VRAM cost) — logging it as `auto-finalized (obvious)`. **Otherwise set the task `PENDING-APPROVAL`, present the comparison videos, and STOP** until the human picks. Dependents that need the final variant must not proceed until approved.
5. **Tier rule:** **LIVE = the lightest variant that gives a usable preview** (benchmark on a real iPhone, not the H100); **OFFLINE always recomputes with the accurate variant** — the live output is a preview + a server *prior*, never the final result.
6. On approval (or obvious auto-pick), record in `models/MANIFEST.json`: chosen variant, weights id + sha256 + license, measured H100/on-device numbers (replace the estimates), and the approval (`approved by <human> on <date>` or `auto-finalized (obvious)`).

Stages to benchmark (candidates in §2.3): person detector (YOLO26 m/l/x; live 26n vs 11n) · 2D pose (RTMW-l vs -x; RTMPose m/l/x crops) · 3D mesh (Fast SAM-3D-Body vs SAT-HMR vs Multi-HMR 2) · ball (TrackNetV3 vs V4-tennis) · racket (SAM2 small/base-plus/large; FoundationPose vs GigaPose) · audio (BEATs vs AST vs small CNN) · shot (PoseConv3D vs BST). Each produces comparison videos + a `variant_selection.md`; the human-approved (or obvious) pick is locked and every later phase uses it.

---

## Test-Clip Dataset Spec (collect before Phase 1)

Store under `data/testclips/`. Every clip is a single static-camera capture per the **capture spec** (`ACCURACY_AND_TRAINING.md §1`): **Landscape, 1080p, HDR off, locked exposure/focus, shutter ≥1/500 s** (indoor 1/100 or 1/120 to kill flicker), tripod ≥1.2–1.5 m, all 4 corners visible. **Variable camera pose is a core product requirement — the test set must stress it.** Bias coverage toward the real-world mode: **baseline-corner (~45%) and side-fence (~30%) viewpoints get the most clips** (~75% of production input).

### Required clip matrix (minimum 24 clips)

| Axis | Required coverage |
|---|---|
| Camera height | low (~tripod 1.2 m), mid (~1.7 m), high (~3 m / fence-top) |
| Camera angle | shallow baseline, steep corner, side fence, near-overhead-ish — **weight toward baseline-corner + side-fence** |
| Play type | ≥10 doubles, ≥6 singles/drill, ≥4 "messy real-world" (spectators, partial occlusion) |
| Environment | ≥8 indoor, ≥8 outdoor (sun/shadow), mixed court colors (blue/green/red) |
| Ball/clothing | ≥2 ball colors (yellow/green/neon), varied shirt colors incl. ≥1 same-color-partners clip |
| Frame rate | **default 60 fps**; ≥6 clips at **120 fps** (swing-speed + contact-timing + racket-6DoF tests); ≥2 clips at **240 fps** self-downsampled to 120/60 (velocity fps-floor **Protocol C**); a few 30 fps clips only to confirm graceful degradation |
| Length | 12 short (60–120 s) for metric tests; 4 long (15–20 min) for throughput tests |
| Racket GT (subset) | ≥3 clips with **ArUco markers on the paddle back** for racket-6DoF ground truth (markers removed for inference) |

> Frame-rate rationale (`ACCURACY_AND_TRAINING.md §1, §5b`; racket-fps in Phase 6): 60 fps is the product floor; **120 fps is required for any absolute swing-speed number and for racket 6DoF** (a ~1000°/s swing rotates ~33°/frame at 30 fps and the SE(3) filter diverges); 240 fps clips exist only as a Protocol-C downsample reference. Audio anchors sub-frame contact timing regardless of video fps.

### Labels (store as `data/testclips/<clip>/labels/`)

- `court_corners.json` — pixel coords of 4 corners + NVZ/centerline intersections.
- `players.json` — per sampled frame (every 30th): bboxes + identity + side/role.
- `feet_nvz.json` — ≥30 frames/clip near the kitchen: support-foot pixel + "in/out/near NVZ".
- `ball.json` — ball pixel coords on ≥200 sampled frames/short clip (incl. blur/occlusion).
- `events.json` — manual contact/bounce/net-crossing timestamps (frame-accurate; use 60 fps clips for the tight timing gate).
- `racket_pose.json` — for ArUco-GT clips: per-frame paddle 6DoF + face-normal (ground truth).
- `foot_contact.json` — for ≥3 clips: per-frame per-foot on-ground labels (foot-skate gate reference).
- `coach_habits.json` — coach-scored habit labels per short clip.
- `manual_metrics.json` — reviewer measurements (NVZ margin ft, inter-player distance ft, knee angle deg, X-factor deg).

A clip is "test-ready" only when its label files exist. Phase gates reference these by path.

---

# Client Track — iOS App (Phases C1–C3)

The native Swift app. It runs **in parallel** with the server Pipeline Track (Phases 0–11). It owns three things: **capture** (the clean frames + the calibration sidecar that the server consumes), the **on-device fast-tier preview + capture guidance**, and **replay viewing**. Dependency edges: **C1 defines the capture sidecar contract consumed by server Phase 1**; **C3 consumes the server Phase 10 USDZ**. Device tiers: **Tier A** = Pro + rear LiDAR (12 Pro→16 Pro) gets ProRes + depth assist; **Tier B** = standard iPhone (vision-only, the baseline — fully viable); **Fallback** = older/non-ARKit-3D devices (locked capture + manual court-corner taps).

## Phase C1 — iOS Capture & Calibration

**WHAT:** capture the cleanest possible frames and produce a per-clip **calibration sidecar** so the server never has to estimate camera geometry. **HOW it's used:** the locked, high-fps clip + sidecar are uploaded; server Phase 1 *seeds* its solve from the sidecar (and Phase 3 world-grounding uses the known camera). Clean, photometrically-stable frames are the single biggest free accuracy win.

**Current delta:** the Swift package, Pickleball app target, shared contracts, and build/test scaffolds exist. IOS-1 includes partial AVFoundation preview/recording runtime code, but it has not written the real sidecar, provided the server live-frame feed, measured fps/shutter/luminance gates, validated HEVC/ProRes on device, enforced landscape at record time, or passed physical-device recording tests. The current app/runtime accepts portrait metadata and shows orientation guidance; hard portrait rejection, the ARKit setup pass, manual tap UI, background upload, and physical-device validation remain unbuilt.

**Build (`ios/Capture` / `PickleballCapture`, `ios/Calibration` / `PickleballCalibration`, `ios/Upload` / `PickleballUpload`):**
- `AVCaptureSession` (rear wide cam). **Before recording, lock everything** (`device.lockForConfiguration()`): `setExposureModeCustom(duration: 1/500–1/1000 s, iso: clampedISO)` (clamp to `activeFormat.minISO…maxISO`), `setFocusModeLocked(lensPosition: courtFocus)`, `setWhiteBalanceModeLocked`. This stops auto-systems pumping brightness/sharpness/color mid-rally (what makes ball + pose stable frame-to-frame).
- **Format/fps policy:** choose `activeFormat` + `activeVideoMin/MaxFrameDuration`: **default 1080p @ 120 fps**; **240 fps (≈720p binned)** for a ball-physics/swing deep-dive; **4K @ 60 fps + ProRes 422 LT** on Tier A. 60 fps is the floor. One mode per session (high-fps and iOS-26 cinematic 30 fps are mutually exclusive — **do not use cinematic**).
- **Landscape target:** `AVCaptureConnection.videoRotationAngle` (iOS 17+; replaces deprecated `videoOrientation`) and locked UI orientation are the intended runtime path. The current scaffold supports portrait/landscape metadata and the app shows portrait guidance; hard record-time portrait rejection is still TODO.
- **Record:** `AVAssetWriter` → HEVC (10-bit) default; **ProRes 422 LT** on Tier A for max-quality reconstruction. `AVCaptureVideoDataOutput` provides live frames to the C2 fast tier in parallel.
- **Hands-free control:** iOS 26 `AVCaptureEventInteraction` / SwiftUI `.onCameraCaptureEvent` → start/stop from the Action/Volume button or AirPods stem (no touching the tripod).
- **ARKit setup/calibration pass** (`PickleballCalibration`): briefly run `ARWorldTrackingConfiguration` with `planeDetection = .horizontal` to grab `ARFrame.camera.intrinsics` + `.transform` (6DoF pose) + an `ARPlaneAnchor` for the court floor. **Note: you cannot run ARKit and a high-fps `AVCaptureSession` at peak rates simultaneously** — do the ARKit pass during setup, persist the result, then switch to the AVFoundation recording session. Write the **capture sidecar** (`capture_sidecar.json`): intrinsics, camera pose, court-plane transform, device tier, fps/format, gravity vector (CoreMotion).
- **Manual fallback:** a court-corner tap UI (4 corners) when ARKit plane is unreliable (blank/low-contrast courts, low light) → still produces the sidecar's correspondences.
- **LiDAR (Tier A only):** capture `sceneDepth` (`ARDepthData.depthMap` + `confidenceMap`) for the near court / near player; attach as optional sidecar frames. **Honest limit: ~5 m range, fails in sun** — it helps near-player foot-contact and the near court plane only, never the far player, far baseline, or the ball. Never a dependency.
- **Upload (`PickleballUpload`):** trim to the rally window on-device (cut upload size). **Send the tiny `capture_sidecar.json` + on-device pose track FIRST** (~50 KB — the pose JSON is ~166× smaller than the video) so the **server starts the SMPL fit while the video is still uploading**. Then stream the clip + LiDAR depth (Tier A) via **resumable chunked upload — TUSKit / iOS 17 native background upload (`URLSession` background task, 5–10 MB chunks, file-backed)** so it survives app suspension.

**Test procedure (on a test device + a Mac harness):**
- Record a locked-exposure clip; verify per-frame mean luminance variance is flat (no auto-exposure pumping) vs an unlocked control.
- Confirm `capture_sidecar.json` validates (schema) and contains intrinsics + pose + court-plane (or manual taps).
- Confirm recorded clips carry the requested fps/format, HEVC/ProRes as configured, and that the production capture path either records landscape or blocks with a capture-quality warning. Current scaffolds do not yet satisfy the hard landscape-block gate.

**Acceptance gates:**
- Locked-exposure clip: frame-to-frame luminance std **< 2%** of range across a 60 s clip (vs the unlocked control which will be higher).
- Sidecar present + schema-valid on **100%** of captures; ARKit intrinsics within plausible range, or manual-tap correspondences present.
- Landscape enforced; requested fps achieved within **±2 fps**; shutter ≤ 1/500 s honored when light allows (else a capture-quality warning is raised in C2).

**Risks:** low light forces slow shutter → C2 warns "add light / lower fps." ARKit plane fails on blank courts → manual-tap fallback. LiDAR oversold → treat as Tier-A bonus only.

## Phase C2 — On-Device Fast Tier & Capture Guidance

**WHAT:** an instant, on-device preview and real-time camera-placement coaching, all on the Neural Engine. **HOW it's used:** gives the user a "<10 s, I feel seen" result and steers them to a good camera setup *before* recording — but it is **preview/guidance only; the server deep tier is the source of truth** (Apple's on-device 3D pose is 17-joint, single-person, ~5–25° error).

**Current delta:** the module boundary and contracts exist, but no real Vision/Core ML runtime path or on-device latency gate has passed.

**Build (`ios/FastTier` / `PickleballFastTier`):**
- **On-device CV (all ANE):** Apple Vision `VNDetectHumanBodyPoseRequest` (2D, ≤4 players), `VNDetectHumanBodyPose3DRequest` (coarse 3D, single-person — for the framed player), `VNDetectHumanHandPoseRequest` (grip cue), `VNGeneratePersonSegmentationRequest` (framing); **YOLO Core ML** (`models_coreml/`, INT8) for ball/racket boxes when Vision is insufficient.
- **Instant preview:** overlay the 2D/coarse-3D pose + court overlay (from the C1 sidecar) + one priority cue on the live/just-recorded clip.
- **Capture-quality guidance (before record):** court-corner visibility (Vision line/corner check → "move back / raise tripod"); ARKit `trackingState == .normal` + plane covers near court ("hold steady, finding ground"); exposure clipping (`exposureTargetOffset` + histogram → "too dark / sun in frame"); motion-blur risk (chosen shutter vs light → "not enough light for a sharp ball; lower fps or add light"); level/shake (CoreMotion gravity + accelerometer variance → "tilt down 5° / steady the tripod"). Mirrors the server `capture_quality` grades.
- Emit the on-device pose track + a capture-quality grade into the upload payload (C1).

**Test procedure:** run on a physical test device (A17/M-class); measure Vision request latency; force each guidance condition (dark frame, tilted phone, occluded corner) and confirm the right flag fires.

**Acceptance gates:**
- On-device 2D pose **≥ 30 FPS** (target 60) on the test device; 3D coarse pose runs without dropping the preview below 30 FPS.
- Each capture-guidance flag (framing, tracking, exposure, blur, level, shake) fires correctly on a scripted bad-setup clip and stays silent on a good-setup clip.
- The fast-tier preview renders **< 10 s** after recording stops.

**Risks:** Apple 3D pose mistaken for a metric → label it "preview" in code + UI; never feed it to the report. Multi-person 3D unavailable on-device → 2D only for ≤4 players; real 3D is server-side.

## Phase C3 — Native 3D Replay Viewer (RealityKit)

**WHAT:** play the server-baked physics-accurate replay natively, free-viewpoint. **HOW it's used:** the phone downloads the baked **USDZ** (from server Phase 10) and plays it; a **web GLB share link** (also from Phase 10) covers cross-platform sharing.

**Current delta:** the replay module scaffolding exists, but no server-baked USDZ, RealityKit animation playback, attached racket/ball path, or device FPS gate has passed.

**Build (`ios/Replay` / `PickleballReplay`):**
- **RealityKit + `RealityView`** in **`virtual` camera mode** (iOS 18+) for pure non-AR 3D playback; orbit/pan/zoom via `DragGesture`/`MagnifyGesture` → **free-viewpoint**. (SceneKit is soft-deprecated — do not use.)
- Load the baked **USDZ** (`Entity(named:)`) and play its skeletal animation (`playAnimation(...)`); racket/ball pinned to joints via the WWDC25 attach API. Timeline scrub/loop; camera presets + free orbit.
- **Doubles:** use the in-app `RealityView` path (handles multiple skeletons); reserve AR Quick Look (single-skeleton limit) for single-player AR.
- Share: open the **web GLB link** (Phase 10) for non-iOS recipients.

**Test procedure:** load a server-baked USDZ on a test device; verify skeletal animation plays, free-viewpoint orbit works, racket/ball stay attached, and the rendered motion shows no foot-skate/penetration (the Phase-4 gates hold through to render).

**Acceptance gates:**
- USDZ replay plays at **≥ 30 FPS** with up to 4 players on iPhone 14+; free-viewpoint orbit smooth.
- No visible foot-skate or floor/inter-player penetration in the rendered replay (regression vs server Phase-4/10 numbers).
- Racket + ball render physically consistent (racket meets ball at contact; ball bounces on the court plane).

**Risks:** SMPL-X pose-corrective blend shapes may not reproduce under RealityKit LBS → **bake the corrected vertex deformation server-side** (Phase 10) so the phone just plays. Heavy mesh on low-end devices → server provides a decimated LOD.

---

## Phase 0 — Environment, Scaffolding, Ingest

**Goal:** reproducible env, repo scaffold, all checkpoints downloaded and smoke-tested, NVDEC decode working, test clips ingested.

**Current delta:** repo/env scaffolding, manifest/checksum inventory, CPU ffmpeg ingest/QC, capture-quality scoring, decode benchmark helpers, and several local/H100 smokes exist. Representative Phase 0 acceptance is not complete: true NVDEC/DALI decode on real 1080p/4K/120+ clips, all-model forward-pass smoke, Triton serving, and the multi-slot GPU lease proof still need current recorded gates.

**Build:**
- `requirements-racketsport.txt` + `web/replay/package.json`; `scripts/racketsport/setup_env.sh` (installs deps, downloads checkpoints to `models/`, verifies hashes).
- `threed/racketsport/__init__.py` + empty module stubs from § 0.2.
- `threed/racketsport/io_decode.py` — `decode_clip(path, fps_out=None) -> FrameSource` via `ffmpeg`/PyNvVideoCodec NVDEC; yields frames + audio; supports frame-range and stride.
- `threed/racketsport/schemas/` — `pydantic` models for every artifact JSON (schemas at end). One `validate(path)` per schema.
- `scripts/racketsport/ingest_testclips.py` — walks `data/testclips/`, decodes each, writes `runs/phase0/<clip>/frames_meta.json` (resolution, fps, frame count, duration, audio sample rate, decode FPS).
- **`scripts/gpu-eval-run.sh`** — the MIG-slot `flock` lease helper that **all `[GPU]` tasks/tests use** (referenced by `BUILD_CHECKLIST.md §1.5`): scan `/run/gpu-lease/slots/slot*.lock` with `flock -n`; on a free slot `export CUDA_VISIBLE_DEVICES=$(cat slotN.uuid)`, write a heartbeat, run `"$@"`, release; if none free, block on `flock` (FIFO). Plus `scripts/gpu-train-lock.sh` (exclusive `full-gpu.lock` for training mode). Set up the MIG geometry (`eval = 2×3g.40gb` default).
- **`scripts/racketsport/benchmark_sam3dbody.py`** — **run FIRST:** measure Fast SAM-3D-Body per-player-frame FPS + peak VRAM on the H100 (B=1 and batched crops). Its numbers set the MIG geometry and deep-tier latency budget.
- **Perf substrate scaffold:** minimal **Triton** ensemble config skeleton (`serving/triton/`) for the eval-serving path. TensorRT export utilities, `torch.compile` warmup, INT8 calibration hooks, and DALI/NVDEC decode remain pending implementation rather than completed code.

**Models:** all of § 0.4 (download + load smoke test only; the body models Fast SAM-3D-Body (primary) + Multi-HMR 2/SAT-HMR (fast) + GVHMR/WHAM (cross-check) load + 1 forward pass).

**Deliverables:** `runs/phase0/REPORT.md`, decoded metadata for every clip, `models/MANIFEST.json` (name, license, sha256, path).

**Test procedure:**
```bash
bash scripts/racketsport/setup_env.sh
python -m pytest tests/racketsport/test_schemas.py tests/racketsport/test_io_decode.py -q
python scripts/racketsport/ingest_testclips.py --root data/testclips --out runs/phase0
python scripts/racketsport/smoke_models.py --manifest models/MANIFEST.json   # loads each model, 1 forward pass, prints FPS
```

**Acceptance gates:**
- All schema + io_decode unit tests pass.
- Every checkpoint loads and runs one forward pass; `smoke_models.py` prints per-model FPS on the H100. Bleeding-edge **[VERIFY]** models that fail to resolve → record the named fallback in `MANIFEST.json` and proceed.
- NVDEC decode **≥ 8× real-time** on 1080p (20-min clip decodes < 2.5 min); **≥ 3× real-time** on 4K.
- 100% of test clips ingested with valid `frames_meta.json`.
- **SAM-3D-Body benchmark recorded** (per-player-frame FPS + peak VRAM, B=1 and batched) in `runs/phase0/REPORT.md`; MIG geometry chosen from it (eval `2×3g.40gb` default).
- **GPU lease smoke test:** `scripts/gpu-eval-run.sh` launches **2 concurrent eval jobs on separate MIG slices without OOM** and serializes a 3rd via the queue (proves the shared-H100 protocol works before parallel agents rely on it).

**Risks:** NVDEC build issues → PyNvVideoCodec fallback; slow 4K → downscale to 1080p at decode (record the choice).

---

## Phase 1 — Court Calibration, Camera Pose, Net Plane

**Goal:** per-clip, viewpoint-agnostic court coordinate frame robust to highly variable camera height/angle. Output a validated homography + camera pose + net plane. **This is the keystone — it is the known camera + ground plane that the world-grounded body (Phase 3) and physics (Phase 4) depend on.**

**Current delta:** CAL has regulation templates, zones, net plane, sidecar/manual PnP, multi-frame/reprojection primitives, overlay/evidence tooling, fail-closed `court_line_evidence.json` plumbing, and pipeline readiness blockers when `auto_calibration_ready=false`. Burlington is retired for court calibration because fisheye curvature bends court lines, though it remains useful for player-ID, ball, BODY smoke, paddle/replay, and other non-court QA. CAL does **not** yet have the trained no-tap keypoint/line solver or real Phase-1 dataset coverage; the current training entry point is the scaffold `scripts/racketsport/train_court_keypoint_heatmap.py`.

**Build:**
- `court_templates.py` — `PICKLEBALL = CourtTemplate(width_ft=20, length_ft=44, nvz_ft=7, net_center_in=34, net_post_in=36, line_in=2)`; `TENNIS_SINGLES`, `TENNIS_DOUBLES` with service boxes. World origin at net center, X across court width, Y toward the far baseline, **Z up (court plane Z=0)**, meters for artifact coordinates.
- `intrinsics.py` — current `get_intrinsics(clip_or_path)` returns measured intrinsics from `capture_sidecar.json`. The tiered future resolver is still pending: cached **ChArUco** per `phone-model+zoom` (RMS <0.3 px), **EXIF** focal guess, then **GeoCalib** per-clip from empty court frames. Do not self-calibrate distortion from court lines alone (degenerate).
- `sidecar.py` — current `load_capture_sidecar(clip_or_path)` resolves and parses `capture_sidecar.json`; the manual/sidecar calibration path consumes its intrinsics and optional manual taps. Live ARKit 6DoF pose/court-plane capture, gravity disambiguation, and LiDAR depth refinement are still iOS/server integration work and must be re-validated with reprojection before they can replace manual taps.
- `court_calibration.py`:
  - `solve_camera_pose(...)` via **`cv2.solvePnP` (full 6-DOF) as PRIMARY** (Acc@5 ~0.71 vs ~0.59 homography; essential at shallow angles). Seed P3P/homography → `SOLVEPNP_ITERATIVE` LM. Homography retained only for near-overhead + as a PnP seed.
  - `solve_multiframe(...)` — aggregate correspondences across **20–40 static frames** + one joint bundle-adjust / `cv2.calibrateCamera(CALIB_USE_INTRINSIC_GUESS)` (−23–57% error, free).
  - `refine_lines_subpixel(...)` — sub-pixel line fit; analytic intersections incl. **out-of-image** (recovers occluded corners). Optional PnL point+line cost (α≈0.6).
  - `reprojection_error(...) -> {median_px, p95_px}`.
  - `manual_tap_calibration(...)` — MVP fallback (taps feed the same PnP solver).
- `court_keypoint_net.py` + `scripts/racketsport/train_court_keypoint_heatmap.py` — current CPU scaffold validates court keypoint taxonomy/training config and heatmap decoding. Future work is to fine-tune a licensed court-keypoint model after synthetic court renders across 50–500 viewpoints, then ~200–500 hand-labeled pickleball frames → sub-pixel keypoints → `solve_multiframe`. Replaces manual taps only after it passes the gate.
- `net_plane.py` — `net_plane_from_template(calibration)`: raise net-line endpoints to 36 in (posts), center 34 in; return vertical plane + sag in world. The physical net remains regulation geometry, but `court_line_evidence.json` records observed top-net pixel evidence as a trust cross-check; overlays and downstream metrics fail closed when the projected top net is not independently supported.
- `capture_quality.py` — `score_capture(...) -> {grade, reasons[]}`: flags shallow angle, extreme distortion, small court coverage. Drives confidence gating + "raise/move camera" hint.
- `drift_guard.py` — `verify(H, frame_t)` every N frames via reprojection + ORB/optical-flow warp; flag re-cal on breach.
- `court_zones.py` — NVZ/kitchen, transition, baseline, service boxes as world polygons; `classify_point(world_xy) -> zone`.
- `scripts/racketsport/calibrate.py` — low-level sidecar/manual calibration CLI → `court_calibration.json`, `court_zones.json`, `net_plane.json`.
- `scripts/racketsport/build_court_line_evidence.py` and the default orchestrator calibration stage — sample video/frames to write `court_line_evidence.json` for kitchen/NVZ lines, center service lines, and trusted top-net evidence; video-backed runs stop at calibration when evidence is not ready.

**Models:** none (classical CV); `kornia` for differentiable homography if needed.

**Deliverables:** `court_calibration.json`, `court_zones.json`, `net_plane.json`, `court_line_evidence.json`, overlay MP4 per clip. `court_line_evidence.json` is mandatory for readiness, even when it is fail-closed.

**Test procedure:**
```bash
python -m threed.racketsport.orchestrator \
  --clip <clip> \
  --inputs data/testclips/<clip> \
  --out runs/phase1/<clip> \
  --stage calibration
python -m threed.racketsport.eval.calib_eval --root runs/phase1 --labels data/testclips --out runs/phase1/metrics.json
```
Measure per clip, **bucketed by camera height × angle**: median/p95 reprojection (px + ft), semantic kitchen/centerline/top-net residuals, `auto_calibration_ready`, overlay IoU, recovered-height plausibility, drift stability on long clips.

**Acceptance gates:**
- Overlay matches on **≥ 90% of clips** (projected-corner reprojection **median < 8 px AND p95 < 15 px** @1080p).
- **No pose bucket** (low/mid/high × shallow/steep/side) below **80%** pass — robustness uniform across viewpoints.
- Feet-to-world error per viewpoint budget (`ACCURACY_AND_TRAINING.md §3`): high-corner **lat <0.2 ft / depth <0.5 ft**; mid-sideline **lat <0.3 ft / depth <0.8 ft**; low-shallow may exceed depth — `capture_quality` marks `warn/poor` and far-court NVZ is confidence-gated downstream.
- `solve_multiframe` beats single-frame on reprojection (regression check).
- Net-plane projection within **15 px** of the visible net line (cross-check).
- `court_line_evidence.json` reports accepted/missing NVZ, centerline, and top-net IDs; video-backed runs do not proceed to tracking when `auto_calibration_ready=false`.
- Drift guard catches an injected 20-px bump within **N+1 frames**, **0** false triggers on a static clip.

**Risks:** occluded corners → homography reconstructs them; wide-FOV distortion → device-profile intrinsic calibration; shallow angles inflate depth → `capture_quality` flags, never hides.

---

## Phase 2 — Person Detection, Tracking, Doubles ID (Fast Tier)

**Goal:** cheap, fast, ID-stable tracking of ≤4 on-court players over 20-min clips, using court geometry as the primary ID-stability engine.

**Current delta:** tracking primitives, a registered real YOLO26m plus BoT-SORT-ReID runner, and a canonical-safety promotion audit exist, with explicit precomputed/manual mode retained. TRK is still `IN-PROGRESS` because the real H100 labeled gate for IDF1, spectator rejection, ID-switches, and throughput has not passed. The latest accepted-four audit found the canonical root `tracks.json` artifacts are too sparse for BODY scheduling: Burlington 2/180 four-player frames, Wolverine 1/91, Outdoor 1/1786, Indoor 0/900. Fresh H100 YOLO26m source-30 strict candidates improved Outdoor to 864/1800 and Indoor to 424/900 four-player frames and audit as `canonical_candidate_not_gate_verified`; widened-margin diagnostics audit as `diagnostic_only` because they include off-court frames, so they must not promote.

**Build:**
- `person_fast.py` — `detect(frames_batch) -> dets` using **YOLO26m** (TensorRT FP16, batch 32–64), person class only. (YOLO26 is real — Ultralytics Jan 2026, +1.4–2.8 mAP over YOLO11 at equal GPU latency, NMS-free end-to-end; **tune with official defaults** — the NMS-free/DFL-free changes need it. RF-DETR-L/RTMDet retained as Apache fallbacks for a future commercial pivot.) For ≤4 large players on a known court we are tracking-limited, not detection-limited — even YOLO26n suffices.
- `track_lock.py`:
  - Base tracker **BoT-SORT-ReID** (MIT, native to ultralytics `tracker=botsort.yaml`) — appearance features resolve doubles crossings. **ByteTrack** is the simpler fallback for well-separated players.
  - `court_polygon_filter(dets, calibration)` — `cv2.pointPolygonTest` on foot point; drop off-court **before** tracking.
  - `ground_plane_association` — project feet to world; associate in court-meters; reject "teleports."
  - `n_lock(N)` — exactly N tracklets on first clean frame; candidate queue; never spawn (N+1)th without match.
  - `hsv_color_cue(crop)` — upper-body HSV histogram + Bhattacharyya.
  - Optional `osnet_reid` — only when `outfits_ambiguous()`.
- `doubles_id.py` — IDs by court position + side prior; `coach_anchor(...)` 1-tap lock; `detect_stack(...)`; `recover_swap()` (position → color → embedding).
- `rally_segment.py` — motion+audio activity detector → rally vs dead-time spans (full pipeline only on rallies; ~1 fps state-keeping in dead time).
- `scripts/racketsport/track.py` — CLI → `tracks.json` (+ overlay MP4).

**Models:** YOLO26m + BoT-SORT-ReID (ByteTrack fallback); OSNet-x0.5 (optional). No pose here.

**Deliverables:** `tracks.json` (per-frame boxes + world positions + IDs + side/role + confidence), rally segmentation, overlay MP4.

**Test procedure:**
```bash
python scripts/racketsport/track.py --detections runs/phase2/<clip>/detections.json --calibration runs/phase1/<clip>/court_calibration.json --out runs/phase2/<clip>/tracks.json --max-players 4 --id-strategy role_lock
python -m threed.racketsport.eval.track_eval --root runs/phase2 --labels data/testclips --out runs/phase2/metrics.json
python scripts/racketsport/benchmark_person_trackers.py --clip <20min_clip> --video data/testclips/<20min_clip>/source.mp4 --calibration runs/phase1/<20min_clip>/court_calibration.json --out-root runs/phase2/bench --candidate yolo26m_botsort_reid=models/checkpoints/yolo26m.pt,configs/racketsport/botsort_reid.yaml --device 0 --max-players 4
```
Measure: IDF1 / ID-switches vs `players.json`; spectator rejection; side/role accuracy; swap-recovery latency; throughput; the same-color-partners clip. Presence/coverage diagnostics such as four-player-frame rate are useful for BODY scheduling, but they are not substitutes for labeled IDF1/spectator gates.

**Acceptance gates:**
- **≥ 90%** identity/side stability on 2-player clips after confirmation; **≥ 85%** doubles after `coach_anchor`.
- **0** permanent IDs on off-court people (100% spectator rejection).
- ID-switches **≤ 2 per minute** on doubles (excluding substitutions).
- Throughput: 20-min 1080p track in **< 90 s** on H100 (batched); 4K **< 3 min**.
- Same-color-partners clip stability **≥ 80%** (geometry carries ID when color fails).

**Risks:** net crossings → ground-plane association holds (boxes overlap, world positions don't). Color+geometry both fail → escalate to OSNet, log it, never silently swap.

---

## Phase 3 — 3D Body: Mesh-Core (Fast SAM-3D-Body backbone + our world-grounding) + Fast Preview

**Goal:** the core body engine. **SMPL/SMPL-X mesh is the primary representation** (drives the physics replay, biomechanics, and foot-skate elimination). The per-frame mesh backbone is **Fast SAM-3D-Body** — the best per-frame mesh available (3DPW PA-MPJPE 30.4 mm, beats HMR2.0 by 15+ mm; user-confirmed; SAM License — ⚠️ verify before commercial, Apache fallback **SAT-HMR**; see `TECH_STACK.md §2.3`) — and **we add world-grounding ourselves** using the known camera + court plane. Two tiers: a **fast multi-person camera-space mesh** for live preview, and a **deep per-frame SAM-3D-Body + world-grounding + (Phase-4) foot-lock + physics** for the replay/metrics. **GVHMR/WHAM are demoted to optional world-trajectory/foot-velocity sanity-checks — NOT the primary mesh** (their per-frame backbone is HMR2.0, weaker; we already solve world-grounding via calibration). The positional skeleton is only a preview/triggering overlay.

**Current delta:** BodyStageRunner wiring, checksum-gated runtime loading, camera-to-court conversion, scheduled-frame BODY execution manifests, and `body_mesh_readiness.json`/`body_gate_report.json` audits exist. Accepted-four contact windows were promoted from human review inputs and request limited world-mesh frames on Burlington, Wolverine, and Indoor. Current BODY gate truth: Burlington 3 scheduled BODY frames / 9 mesh player-frames, Wolverine 4 / 12, Indoor 2 / 6 pass mesh-smoke only; Outdoor remains 0 / 0 with no BODY output. The report now fail-closes draft/not-ground-truth BODY world labels and lists expected label/full-clip gate paths. BODY remains scaffold because no multi-frame H100 BODY accuracy gate has passed, no reviewed world-MPJPE labels/equivalent evaluator exists, and no full-clip BODY coverage gate exists.

**Build:**
- `hmr_fast.py` / iOS `PickleballFastTier` — fast-tier preview metadata, Apple Vision/Core ML seams, and eventual top-down `RTMW-l`/whole-body keypoint adapters for the ≤4 player crops from Phase 2. **Keypoints MUST include bilateral hips + shoulders** (X-factor) **and hands/fingers** (racket grip + face path). Whole-body (133-kpt).
- `worldhmr.py` — `reconstruct_world(frames, tracks, calibration) -> SmplMotion`:
  - **DEEP (primary): Fast SAM-3D-Body** per player crop → best per-frame mesh (MHR→SMPL via its built-in MLP). Then **world-grounding is OURS**: project the per-frame mesh to world via the known K,[R|t] + court plane Z=0 (metric root depth from court PnP), then temporal smoothing (windowed Gaussian / Savitzky-Golay / SmoothNet). This replaces any dependence on a world-HMR model's SLAM. Precedent: arXiv 2512.21573 (World-Coordinate retargeting via SAM 3D Body). Foot-lock + physics follow in Phase 4.
  - **Optional cross-check only: GVHMR / WHAM / TRAM** — run (GVHMR without its SLAM, our camera injected) purely to sanity-check world trajectory + foot-velocity against our grounded output. **Do NOT use them as the mesh source** — their per-frame mesh is HMR2.0-based (worse than SAM-3D-Body) and we already have ground-truth camera geometry.
  - **FAST (server post-upload quick pass): Multi-HMR 2** (20 FPS, no FOV assumption — preferred given variable cameras) **or SAT-HMR** (24 FPS, assumes 60° FOV — verify per court cam) — multi-person camera-space SMPL for a quick server preview while the deep tier runs. **[VERIFY]**; fallback = HMR2.0/4D-Humans, or fit SMPL to RTMW3D. ~7–8 mm quality penalty vs SAM-3D-Body is fine for a preview. (The *instant, during-capture* preview is on-device **Phase C2 / Apple Vision** — this server fast pass is the post-upload one.)
  - Multi-person: **no single robust monocular ≤4-person world pipeline** — run **per-player** after Phase-2 detection; the court plane gives per-player metric scale + root depth.
- `worldhmr.py` emits `skeleton3d.json` as a **preview/triggering overlay only**: RTMW3D-l or RTMPose→MotionBERT lift, world coords. Used for the <10 s fast preview and for sub-frame contact triggering; NOT the source of truth for the replay (that is world-grounded SMPL).
- `skeleton3d.py` owns semantic joint adapters for preview skeleton payloads, including the fail-closed SAM3D/MHR-70 generic-joint map needed before wrist-driven shot-transfer fallback can trust `skeleton3d.json`.
- `worldhmr.py` + `footlock.py` + `body_mesh_readiness.py` own court geometry as free accuracy (`ACCURACY_AND_TRAINING.md §5`): (1) **reprojection-consistency loss** at fine-tune; (2) **metric root depth** from court PnP; (3) **hard foot-on-ground** seed (full foot-lock is Phase 4). Foot-contact MLP on `[foot_y_world, |v_foot_world|, ankle_z]` (bootstrap AMASS zero-velocity, warm-start UnderPressure).
- **Body fine-tune — biggest accuracy lever** (`scripts/racketsport/finetune_pose.py`; `ACCURACY_AND_TRAINING.md §5`). Generic models hit ~214 mm MPJPE on athletic motion vs ~65 mm fine-tuned. Ladder:
  1. **Fine-tune order: BEDLAM2 → AthletePose3D → CalTennis [VERIFY] → RICH (contact) → AMASS (priors)** (backbone LR 0.1× head). Eval on EMDB2 (world trajectory) + CalTennis multi-view + AthletePose3D.
  2. **+ ground-plane constraints** → lower world MPJPE + drift.
  3. **+ pseudo-label our own footage + distill** (planned): heavy oracle = the 2-camera multi-view triangulation (best 3D GT) + SAM-3D-Body, over our clips → filter by reprojection <8 px through our known camera → active-learn ~500 hardest frames → re-fine-tune. Plus multi-view→monocular distillation (Data & Training Infra).
  4. **+ conditional synthetic** viewpoint augmentation if angle-specific failures remain.
- **Keypoint/param target:** SMPL-X params (body + hands + feet); H3WB-133 for the preview skeleton (lift only 17 body joints temporally, MotionBERT 243-frame).
- `person_calibration.py` — `lock_body_model(player, sessions)`: fix per-player SMPL shape (β) after a few sessions → faster + more accurate (better-with-use moat). **Cache betas in Postgres/Redis; reuse as a fixed prior → SMPLify-fit iterations drop ~100→~20 on every repeat session** (first clip slow, every later one fast).
- **Performance (the deep tier is the pipeline's pacing item — optimize here most):**
  - **Server-side SMPL-fit uses the uploaded on-device 2D pose track as a PRIOR** (from the C1 sidecar `ondevice_pose_track`) → cuts mesh-fit iterations **50–80%**; **start fitting while the video is still uploading** (the ~50 KB pose JSON arrives first).
  - **Event-triggered compute:** run deep Fast SAM-3D-Body **only on rally/contact spans** (from Phase-2 `rally_spans` + Phase-5 `contact_windows`); skip the 40–60% dead time → ~2× throughput. Coast/interpolate through low-information frames; densify around contact.
  - **GPU placement:** **Fast SAM-3D-Body at FP16/BF16, run SEQUENTIALLY per player** (mesh model too big to batch at B=4; **do NOT INT8 it — causes mesh artifacts**); **batch the ≤4 player crops at B=4 for YOLO26/RTMW** (TensorRT INT8 for those); **CUDA-stream overlap** decode→detect→pose→mesh. SAM-3D-Body ~15 FPS/person, ~4–6 FPS for 4 crops (⚠️ RTX-5090 estimate — **re-benchmark on the H100 first, §0.7**; it paces the pipeline) — offline-feasible on rally spans.
- Registered `body` stage in `threed.racketsport.orchestrator` — clip + tracks + calibration → `smpl_motion.json` (+ `skeleton3d.json` preview) when the Fast SAM-3D-Body runtime/checkpoints are available. There is no standalone BODY CLI in the current repo.

**Models:** Fast SAM-3D-Body (primary per-frame mesh, deep) + our world-grounding; GVHMR/WHAM/TRAM (optional trajectory sanity-check only); Multi-HMR 2 / SAT-HMR (fast tier); RTMW/RTMW3D/MotionBERT (preview). Fine-tuned checkpoints under `models/finetuned/`.

**Deliverables:** `smpl_motion.json` (per player/frame: SMPL-X params + world root + per-joint confidence + foot-contact flags), `skeleton3d.json` (preview), fine-tuned checkpoint + eval card.

**Test procedure:**
```bash
python scripts/racketsport/finetune_pose.py --order bedlam2,athletepose3d,caltennis,rich,amass --out models/finetuned
python -m threed.racketsport.orchestrator --clip <clip> --inputs data/testclips/<clip> --out runs/phase3/<clip> --stage body --tracking-mode precomputed
python -m threed.racketsport.eval.body_eval --root runs/phase3 --labels data/testclips --out runs/phase3/metrics.json
```
Measure: **world MPJPE on EMDB2 + CalTennis**; per-frame MPJPE on racket-motion val; foot/NVZ agreement vs `feet_nvz.json`; knee/elbow angle error + X-factor vs `manual_metrics.json`; spacing error; FPS (fast vs deep). The local `body_gate_report.json` additionally requires clip-level BODY world-joint labels at `labels_root/<clip>/body_world_joints.json` or `labels_root/<clip>/labels/body_world_joints.json` plus `body_full_clip_gate.json` before mesh-smoke can become BODY verification.
**Velocity/kinematics (conflict RESOLVED — `ACCURACY_AND_TRAINING.md §5b`):** R²0.96 = ball-speed model from 2D in-plane speed; r0.11–0.28 = derivative of noisy lifted-3D. `velocity.py` computes the reliable way: **project wrist/ankle through court homography to metric world plane (Type D)** → RTS/Kalman → central difference → zero-phase Butterworth `filtfilt` (6 Hz body/timing, 8–10 Hz swing) or Savitzky-Golay (order 2, win 9); One-Euro (β≈0.007) real-time. **Never** contact-frame peak from pose (use the ball). Run Protocols A (ball-speed GT: r≥0.70 Tier-1 / r≥0.55 Tier-2), B (timing: MAE ≤2 frames), C (240→120/60 downsample: peak underestimate ≤15%, r≥0.90).

**Acceptance gates:**
- Per-frame mesh quality matches Fast SAM-3D-Body's published range (3DPW PA-MPJPE ~30 mm); world MPJPE on EMDB2/CalTennis within range after our grounding; racket-motion per-frame MPJPE **< 80 mm** (target ~65 mm) after fine-tune.
- Foot/NVZ in/out/near agreement **≥ 80%** on confident frames.
- Sagittal large-joint angle error (knee, elbow, trunk) **median ≤ 10°** vs manual.
- Inter-player spacing **within 12 in**; NVZ margin **within 6 in** on confident frames.
- X-factor only when both shoulders+hips confident; error **≤ 12°** vs manual on quasi-static frames (gate on fast-rotation).
- FAST-tier camera-space mesh (Multi-HMR 2 / SAT-HMR) **≥ 20 FPS** (≤4 players, H100 batched) for live preview; skeleton preview **≥ 60 effective FPS**. Deep-tier Fast SAM-3D-Body ~15 FPS/person (~4–6 FPS for 4 batched crops) — offline-feasible for ~20-min clips.
- Velocity: Protocols A/B/C run; split-step timing passes Protocol B; each swing-speed metric gets a Tier-1/2/gated decision; Tier-3 quantities confirmed not surfaced.

**Risks:** monocular depth worse at shallow angles → ground constraint + capture-quality gating. Per-player world placement consistency across ≤4 players → validate depth ordering; foot-lock (Phase 4) further stabilizes.

---

## Phase 4 — Foot-Skate Elimination & Physics Refinement

**Goal:** make the reconstructed motion **physically plausible and foot-skate-free** — the watchable-replay requirement. The known ground plane (Z=0) lets us beat published foot-slide numbers. (This phase **replaces** the prior SAM-3D-Body "contact micro-mesh burst" idea — mesh is now core, not a burst.)

**Current delta:** only CPU primitives/scaffolds and evaluator hardening exist. There is no trusted `smpl_motion.json` stream from Phase 3, no real foot-contact labels/gate pass, and no PhysPT/MuJoCo refinement path through the pipeline.

**Build:**
- `footlock.py` — the decisive skate killer (the modern form of "Reducing Footskate with Ground Contact Constraints"). Per foot (toe+heel), per player, per frame:
  - `contact = (height_above_Z0 < τ_h≈2–3 cm) AND (world_speed < τ_v≈1 cm/frame) AND (pose_conf > τ_c)` with **hysteresis** (separate on/off thresholds → no flicker). Swap the threshold trigger for the UnderPressure learned classifier if fast lunges prove noisy.
  - On confident contact: **snap stance toe/heel to Z=0** (kill penetration/float), **hold its (x,y)** (zero world velocity), **CCD/Jacobian IK** so the leg meets the locked target while the upper body keeps its motion, **blend** in/out over a few frames at contact edges.
  - Light temporal smoothing (One-Euro live / Butterworth offline).
- `physics_refine.py` module — CPU scaffold for plausibility (balance, momentum, non-penetration):
  - **Default: PhysPT** (MIT, no engine at inference) over the foot-locked motion → −68.7% foot-slide, −83.8% accel error, penetration handling, emits GRF/torques.
  - **Flagship: PHC/PULSE on MuJoCo+MJX** — drive a simulated SMPL humanoid to *track* the foot-locked kinematics as reference (physically valid by construction); start from **SMPLOlympics** tennis env.
  - **Doubles: MultiPhys** — inter-player non-penetration so players don't clip.
  - **BioPose NeurIK** (option) for anatomically-valid joint limits before physics.
- Output updates `smpl_motion.json` with `foot_contact`, `skate_free=true`, and (flagship) per-frame GRF.

**Models:** PhysPT (MIT, default); PHC/PULSE + MuJoCo/MJX (flagship); MultiPhys (doubles); UnderPressure / BioPose (options).

**Deliverables:** foot-locked + physics-refined `smpl_motion.json` plus `physics_refinement.json`; physics QA overlay (foot trails, contact flags); GRF track (flagship).

**Test procedure:**
```bash
python scripts/racketsport/build_virtual_world.py --court-calibration runs/phase1/<clip>/court_calibration.json --tracks runs/phase2/<clip>/tracks.json --smpl-motion runs/phase4/<clip>/smpl_motion.json --skeleton3d runs/phase3/<clip>/skeleton3d.json --out runs/phase4/<clip>/virtual_world.json
python scripts/racketsport/build_physics_refinement_from_world.py --clip <clip> --virtual-world runs/phase4/<clip>/virtual_world.json --out runs/phase4/<clip>/physics_refinement.json
python -m threed.racketsport.eval.physics_eval --root runs/phase4 --labels data/testclips --out runs/phase4/metrics.json
```
Measure: **foot-slide** (stance-foot world displacement during detected contact) vs `foot_contact.json`; floor-penetration (min foot-vertex Z); inter-player penetration (mesh intersection); acceleration jitter; contact-detection precision/recall.

**Acceptance gates:**
- **Foot-slide ≤ 3 mm** during contact windows (we snap to an exact plane → beats published ~3.0–4.4 mm).
- **0 floor penetration** (min foot-vertex Z ≥ 0, hard constraint).
- **No visible inter-player penetration** on doubles clips (MultiPhys).
- Foot-contact detection **precision ≥ 0.9 / recall ≥ 0.85** vs `foot_contact.json`.
- PhysPT pass reduces acceleration jitter measurably vs Phase-3 input (regression check).

**Risks:** noisy contact in fast lunges → learned classifier (UnderPressure). MuJoCo imitation is offline/heavy → PhysPT is the default; sim is the flagship-only path. Keep the foot-lock step (cheap) as the always-on guarantee.

---

## Phase 5 — Ball Tracking + Audio Events + Event Fusion + 3D Ball Physics

**Goal:** 2D ball track + 3D ball trajectory + contact/bounce/net events accurate enough to drive adaptive-compute triggers, timing metrics, and the physics-consistent replay. Not line calls.

**Current delta:** TrackNetV3 smoke/adapters, strict no-click review tracks, ball inflection/audio/wrist cue artifacts, saved human review inputs, promoted accepted-four `contact_windows.json`, fail-closed contact-window builders, and `contact_timing_review_alignment.json` exist. The alignment report confirms 34/34 saved human-review contact timestamps match promoted windows at 0-frame delta. BALL is still scaffold: the registered prototype runner consumes precomputed review tracks rather than running TrackNet, the promoted contacts are human-review prototype artifacts rather than label/F1 gates, audio/wrist coverage is incomplete, and machine cue-fusion contact timing has not passed.

**Build:**
- `ball_tap_track.py` — manual/semi-auto tap-track fallback (trust floor).
- `ball_tracknet.py` — **build on TrackNetV3 (MIT, public) now; upgrade to TrackNetV5 [VERIFY] on release** (best F1 ~0.986, −74% FN under occlusion). Heatmap, multi-frame. **Transfer hierarchy:** badminton → tennis+TT (~155k) → pickleball. **Data:** pickleball Roboflow sets (InAPickle 18k + ball-tracking 11k ≈ 50k), bbox-center→(x,y), visibility=2; ~2–3k own frames using **BlurBall midpoint** convention; **TOTNet visibility-weighted loss + occlusion aug** + background-subtraction concat. **Aug:** motion-blur synthesis, H.264/JPEG artifact, color jitter hue ±30° (ball colors), copy-paste, fps sim (geometric transforms identical across the stack).
- `ball_physics3d.py` — `lift_ball_3d(track2d, calibration)`: EKF `[x,y,vx,vy]` w/ gravity → RANSAC parabola (reject >5 px off-arc, fill gaps) → **3D uplift via physics ODE** `dv/dt = g − ½ρCd·A|v|v/m + ½ρCl·A(ω×v)/m` with **z=0-at-bounce constraint** (resolves single-camera depth) → **Magnus spin from trajectory curvature** → bounce = vertical-velocity sign change (sub-frame via parabola) → net crossing = trajectory ∩ net plane. **Pickleball aero:** topspin generates *more* lift than backspin (perforations), ground COR ≈ 0.62–0.66, paddle-ball COR ≤ 0.43–0.44.
- `audio_pop.py` — **two-stage** (`ACCURACY_AND_TRAINING.md §7`): (1) **onset/peak detector** → sub-frame timestamp (~0.09–4 ms); (2) small **2D-CNN on 64-mel spectrogram** (FFT 512, hop 256, 100 ms window, 10 kHz high-pass) → event type. **Never let the classifier window drive timing.** Collect **2–5k pickleball "pops" @44.1 kHz WAV** (never AAC); negatives 2–3×. Augment: noise SNR 0–20 dB (MUSAN/ESC-50/FSD50K/DEMAND) + RIR (pyroomacoustics) + SpecAugment + mixup → ~20k. **Mandatory distance-delay correction:** shift audio back by `d/343 s` (`d` from homography) before fusion.
- `event_fusion.py` — `fuse(ball_track, smpl_motion, audio) -> events`: audio = WHEN, ball/visual = WHICH event, pose+ball = WHICH player. Contact = audio peak ∧ wrist-velocity peak ∧ ball-trajectory inflection; bounce = sign flip; net crossing = ∩ net plane. **Doubles attribution:** ball-to-wrist proximity (court coords) + pre-contact trajectory vector (single-mic DOA does NOT work). Emit `contact_windows.json` (drive the racket phase + replay).
- **Auto-label loop** (planned; current helpers include `scripts/racketsport/build_audio_onsets.py`, `scripts/racketsport/build_ball_inflections.py`, and `scripts/racketsport/build_contact_windows_from_cues.py`): audio onset → keep where ball-inflection agrees ±2–3 frames → human-verify ~10–15% → Mean-Teacher (DCASE'24 baseline) → active learning (~1/3 labels).
- **Performance:** TrackNet → **TensorRT FP16 + CUDA graph** (it runs every frame at high rate — graph capture kills per-kernel launch overhead); audio CNN → **TensorRT INT8** on a parallel CUDA stream; ball tracking **overlaps body inference via CUDA streams** (decode→ball/detect→pose→mesh). Audio is high-rate regardless of video fps — it carries sub-frame contact timing for free.

**Models:** TrackNetV3 (now) / TrackNetV5 (on release); two-stage audio CNN (ours).

**Deliverables:** `ball_track.json` (2D + 3D + spin + bounces), `contact_windows.json`, events overlay MP4.

**Test procedure:**
```bash
python scripts/racketsport/prepare_tracknetv3_finetune_dataset.py --help
python scripts/racketsport/validate_ball_audio_dataset.py --help
# TrackNet/audio training CLIs are planned, not present yet.
python -m threed.racketsport.orchestrator --clip <clip> --inputs data/testclips/<clip> --out runs/phase5/<clip> --stage ball_events --tracking-mode precomputed --ball-source runs/phase5/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json
python -m threed.racketsport.eval.ball_event_eval --root runs/phase5 --labels data/testclips --out runs/phase5/metrics.json
```
Measure: ball P/R/F1 vs `ball.json` (incl. blur/occlusion subset); false positives; contact-timing vs `events.json` (60 fps clips); bounce/net accuracy; 3D-trajectory physics-fit residual.

**Acceptance gates:**
- Ball F1 **≥ 0.90** (target 0.90–0.95) on the held-out **2k-frame** set; blur/occlusion recall **≥ 0.75**; false positives **<5%** after physics filter.
- Contact-event timing **≤ ±2 frames, ≤ ±4 ms with audio** (after distance-delay correction); **≤ ±66 ms** pose-only fallback.
- Bounce within **±2 frames**; net-crossing **≥ 85%** where applicable.
- 3D ball trajectory obeys the physics model (fit residual within tolerance); plausible bounce on the court plane.
- If ball F1 < 0.85 → **ship tap-track** for MVP, record the decision (ball is a timing/context layer, not the promise).

**Risks:** small public pickleball ball data → our 2–3k labels are the bottleneck. Pickleball "pop" acoustics distinct/unstudied → retrain audio, validate on real clips.

---

## Phase 6 — Racket / Paddle 6DoF Tracking

**Goal:** track each player's paddle/racket in full 6DoF (position + orientation) and render it in the 3D world. This should unlock real contact-point-on-face + face-angle: markerless wrist pronation is typically too noisy, while the target gate for a large planar paddle with known dimensions is **~3–5°**. The current repo has not proven that accuracy because no ArUco/GT face-angle evaluation has passed. (See `TECH_STACK.md` §(r).)

**Current delta:** PnP/IPPE geometry, fail-closed candidate handling, review overlays, preview paddle poses, promotion audits, true-corner review artifacts, and crop sheets exist. RKT remains scaffold: no true paddle detector/mask/keypoint/CAD pose runtime has run, no canonical accepted-four `racket_pose.json` exists, and no ArUco/GT face-angle gate has passed. The current accepted-four candidate sets are all box-derived (Burlington 42, Wolverine 36, Outdoor 397, Indoor 280), have 0 reviewed true-corner labels and 0 reference/GT labels, and `RacketStageRunner` rejects them before canonical pose promotion.

**Build:**
- `racket6dof.py` module plus registered `racket` stage:
  - **Detect:** RTMDet box + **SAM2** silhouette of the paddle.
  - **Keypoints:** RTMPose **top/bottom/handle only** (PCK 92–99% reliable; **NOT** side keypoints, 65–80%) + convex-hull face corners from the silhouette.
  - **Coarse 6DoF:** GigaPose or FoundPose, seeded with a **hand-grip prior** — grip frame from wrist + middle-finger MCP (Phase 3 hand keypoints), attach the paddle CAD rigidly → narrows SO(3) search, survives grip occlusion.
  - **Refine:** **PnP-IPPE** on the known 3D paddle corners (resolve IPPE's 2-fold planar ambiguity via gravity/temporal prior).
  - **Temporal:** **UKF on SE(3)**, constant-ω model; blur streak = rotation cue (direction = ω axis, length ∝ speed); Kalman-predict through ≤3 occluded frames; B-spline the swing arc.
  - **Physics-validate:** given racket 6DoF + incoming ball velocity, predict rebound; reject poses whose predicted ball path disagrees with the observed `ball_track` (uses Phase-5 ball + aero).
  - **Outputs:** `contact_point_on_face` (intersect ball trajectory ∩ paddle plane) + `face_normal` at each `contact_window`.
- Training (`ACCURACY_AND_TRAINING.md`): fine-tune RTMPose on **RacketVision (tennis) + Roboflow pickleball** box sets; render **~50k synthetic frames from a paddle CAD via BlenderProc** (domain randomization + synthetic blur → free 6DoF + corner GT); collect **~5k real frames with ArUco markers** on the paddle back for real 6DoF GT (remove markers for inference).
- Render hook: paddle mesh as a child of the wrist bone at the estimated 6DoF (feeds Phase 10).

**Models:** RTMDet + SAM2 + RTMPose (paddle); GigaPose/FoundPose (coarse); OpenCV PnP-IPPE; UKF (filterpy).

**Deliverables:** `racket_pose.json` (per player/frame: 6DoF pose + confidence; per contact: contact-point-on-face + face-normal).

**Test procedure:**
```bash
# Racket keypoint/model training CLI is planned, not present yet.
python scripts/racketsport/validate_racket_dataset.py --help
python -m threed.racketsport.orchestrator --clip <clip> --inputs data/testclips/<clip> --out runs/phase6/<clip> --stage racket --tracking-mode precomputed
python -m threed.racketsport.eval.racket_eval --root runs/phase6 --labels data/testclips --out runs/phase6/metrics.json   # vs ArUco-GT clips
```
Measure: face-angle error vs `racket_pose.json` (ArUco GT); contact-point-on-face error; 6DoF tracking continuity through swings (≥120 fps clips); occlusion recovery. Before this test is meaningful, accepted clips need reviewed true paddle-face corners in top-left, top-right, bottom-right, bottom-left order, or CAD/reference/ArUco evidence. Do not copy the yellow draft box corners from crop sheets into true-corner labels.

**Acceptance gates:**
- **Face-angle error ≤ 5°** vs ArUco GT on the ≥120 fps clips (target 3–5°).
- **Contact-point-on-face within ±1–3 cm** (ball-3D-limited).
- 6DoF track continuous through ≥90% of swing frames; recovers within ≤3 frames after grip occlusion.
- Physics-validation rejects implausible poses (regression: predicted vs observed ball rebound residual within tolerance).

**Risks:** at 30 fps the SE(3) filter diverges → require ≥60 fps (120 for face-angle claims). Side keypoints unreliable → use top/bottom/handle + silhouette only. No public pickleball-paddle pose dataset → synthetic + ArUco GT are the path.

---

## Phase 7 — Biomechanics Metrics + Rule-Based Insight Engine + Confidence Gating

**Goal:** turn physics-refined body + racket + events + court into defensible coaching metrics and rule-based habit flags, with honest confidence gating.

**Current delta:** rule/confidence/report primitives exist, but the trusted upstream inputs do not. Metrics that depend on BODY, BALL, FOOT, or RKT cannot be promoted until those phases pass their real gates.

**Build:**
- `movement_metrics.py` / `biomech.py` — foot/NVZ margin, court-zone occupancy, inter-player spacing, base-of-support, CoM (segmental mass-weighted, De Leva), reach outside base, recovery time, knee/elbow/trunk angles, shoulder-hip X-factor, contact point relative to body, contact height, split-step timing, weight transfer. **Now also racket-derived:** paddle-face angle at contact + contact-point-on-face (from Phase 6 — newly claimable). Each metric returns `{value, units, confidence, frames_used}`.
- `confidence.py` — central gating: pose + event + capture-quality + racket-6DoF confidence on every metric. **Hard-gate** any velocity metric that failed Phase-3 tests and any face-angle when racket 6DoF confidence is low; present values as **ranges**, never false precision. The "no charge if we can't trust it" plumbing.
- `insight_rules.py` — rule-based thresholds = source of truth. PB: kitchen-foot, transition-stuck, partner-gap, overreach, late-split, arm-led, **paddle-face-at-contact** (now defensible via racket 6DoF). Tennis: serve landing balance, toss/contact consistency, serve+1 readiness, backhand spacing. Each rule: required signals, threshold band, flag + clip ref + metric.
- `habit_model.py` — rank flags into habits; the "one leak at a time" selector picks the single highest-leverage habit; before/after hooks.
- Current module primitives (`movement_metrics.py`, `biomech.py`, `confidence.py`, `insight_rules.py`) feed `racket_sport_metrics.json` and draft `habit_report.json`; a standalone metrics CLI is still pending.

**Deliverables:** `racket_sport_metrics.json`, draft `habit_report.json`, per-metric confidence.

**Test procedure:**
```bash
# Generate or supply racket_sport_metrics.json and habit_report.json from the current module primitives.
python -m threed.racketsport.eval.metric_eval --root runs/phase7 --labels data/testclips --out runs/phase7/metrics.json
```
Measure: metric accuracy vs `manual_metrics.json`; rule-flag agreement vs `coach_habits.json`; confidence calibration; confirm gated metrics never surfaced.

**Acceptance gates:**
- Kitchen/NVZ foot **within 6 in**; spacing **within 12 in** on confident frames.
- Paddle-face angle **within 5°** vs ArUco GT (now claimable via racket 6DoF; gated when racket conf low).
- Rule-flag precision vs coach labels **≥ 75%** on confident spans; **habit dismiss rate < 20%**.
- Confidence calibration monotonic (uncertain bucket has higher error than confident).
- **0** ungated transverse-rotation values from *pose alone* (face-angle only via the racket path).

**Risks:** over-claiming is the central failure mode → gating is a gate, not a nicety.

---

## Phase 8 — Shot Classification + Drill Verification

**Goal:** label shot types and verify drill reps — ML-for-label-only, rules elsewhere.

**Current delta:** shot/drill modules, dataset validators, an event-level shot scorer, a SAM-3D-Body/MHR70 semantic joint adapter, a DATA-5 reviewed-label builder, a transfer/heuristic review baseline, and a CPU centroid trainable sanity baseline exist, but SHOT is still scaffold-level. There is no populated reviewed pickleball shot dataset, approved PoseConv3D/BST pickleball classifier, or promotion-grade macro-F1/rep-count gate result. External transfer checks on 2026-06-28 show the current baseline is not claimable: THETIS 60-clip family accuracy is 51.7% overall / 66.7% top-2, with `serve` 0/15 and `overhead` 0/5; OpenSportsLab 100-event broad family accuracy is 81.0%, but `swing` is 81/81 and `serve` is 0/19. These are review signals only and do not promote SHOT-1.

**Build:**
- `shot_classifier.py`, `shot_transfer_baseline.py`, `shot_trainable_baseline.py` — build the shot-label path. The current CPU centroid baseline is an end-to-end sanity check over DATA-5 `features.*`, not a production classifier. The real classifier remains hierarchical: Stage 0: `serve`, `overhead_candidate`, `normal_hit`, `unknown`. Stage 1: `fh_shot`, `bh_shot`, `serve`, `overhead`. Stage 2 PB taxonomy: `fh_drive`, `bh_drive`, `dink`, `lob`, `third_shot_drop`, `reset_block`. Start with **PoseConv3D/PoseC3D** for pose-window robustness, then compare **BST-style** pose+ball/player fusion after the ball/court tensors are clean. **No published pickleball classifier -> collect + label** (`data/pb_shots/`), seed with ball-trajectory + contact-height + body/racket-velocity heuristics for weak labels, then train.
- `drill_verify.py` — rep counting via wrist-velocity peak + confirmed contact; per-rep quality gate; state machine ready→windup→contact→follow-through; output `{reps, clean_reps, per_rep_quality}`.
- `scripts/racketsport/generate_shot_classifications.py`, `scripts/racketsport/train_shots.py`, `scripts/racketsport/build_shot_dataset.py` — current SHOT/DATA-5 CLIs. Drill verification currently exists as `threed/racketsport/drill_verify.py`; a standalone drill CLI is still pending.

**Models:** PoseConv3D/PoseC3D first, BST-style fusion second, both trained/fine-tuned on our data after external tennis/badminton transfer checks.

**Deliverables:** shot labels in `racket_sport_metrics.json`; `drill_report.json`.

**Test procedure:**
```bash
python scripts/racketsport/build_shot_dataset.py --truth-events <reviewed-events.json> --contact-windows <contact_windows.json> --out-dir data/pb_shots/<clip> --dataset-id pb_shots_v1 --clip-id <clip> --split train --fps 60
python scripts/racketsport/train_shots.py --manifest data/pb_shots/<clip>/shot_dataset_manifest.json --out runs/phase8/shot_trainable_baseline.json
python scripts/racketsport/generate_shot_classifications.py --run-dir runs/phase8/<clip> --clip-id <clip> --out-json runs/phase8/<clip>/shot_classifications.json
# Drill CLI pending; use threed.racketsport.drill_verify primitives until a script wrapper exists.
python -m threed.racketsport.eval.shot_drill_eval --root runs/phase8 --labels data/testclips --out runs/phase8/metrics.json
```
Measure: shot-class macro-F1 + top-2 accuracy; rep-count error vs manual.

**Acceptance gates:**
- Shot macro-F1 **≥ 0.65**, top-2 **≥ 0.85** (BST hits ~0.70/0.93 on badminton).
- Drill rep-count error **≤ ±1 rep** (within ±2 on ≥ 90% of drills).

**Risks:** dataset collection is the long pole — start labeling during earlier phases.

---

## Phase 9 — LLM Coaching Copy, Report Artifacts, Visualization, Tiered Delivery

**Goal:** the user-facing outputs and conversion-core UX. Fast result < 10 s; premium async after.

**Current delta:** structured report artifacts and deterministic checks exist. `viz_*` modules currently emit CPU payloads/metadata, not final rendered court/ghost/overlay assets, and FastAPI/SSE/Celery/APNs/tiered delivery has not been built or validated.

**Build:**
- `llm_copy.py` — LLM **for copy only** (latest Claude — Opus/Sonnet). Input = structured biomechanical facts (angles, flags, confidence, clip refs); output = "one prescriptive external cue + plain-language why + drill" per habit. **The LLM never invents facts.** Pattern after CoachMe/BioCoach; enforce with a schema (copy references only provided facts; reject hallucinated numbers).
- `report_model.py` — assemble `habit_report.json` + `coach_report.json` + share payload (`coverage`, `skipped_reason_counts`, per-habit fields).
- Visualization (conversion core, render fast):
  - `viz_courtmap.py` — top-down court map + shot/position heatmap + player paths.
  - `viz_ghost.py` — self-vs-self before/after aligned at the contact frame.
  - `viz_overlay.py` — 2D pose overlay on the player's video + **auto-telestration** (knee-angle arc, contact-point marker, base-of-support box, **paddle-face indicator** from Phase 6).
  - Premium async: links to the Phase-10 3D replay.
- `orchestrator.py` + `pipeline_contracts.py` — the two tiers; **progressive disclosure**: 0–10 s (skeleton/SMPL overlay + court map + 1 priority metric), 10–60 s (full per-shot metrics + heatmap), 2–10 min async (LLM summary + 3D replay).
- **Progressive rally-by-rally streaming (`stream_api.py`) — never make the user wait for the whole clip.** A **Celery GPU worker** processes rallies, publishing each result to a **Redis Pub/Sub** channel; a **FastAPI SSE endpoint** (`EventSourceResponse`) relays to iOS (the same pattern as LLM token streaming; iOS consumes via `URLSession`/`EventSource`). Event sequence: `job_accepted → rally_total:N → rally_done:{i, metrics, replay_url}` (×N, as each finishes) `→ habits_partial` (as detected) `→ complete`. Fire an **APNs push** on `complete` so a backgrounded user is notified. **Optimistic UI:** the app renders the N-rally checklist instantly from on-device segmentation and fills/▸-enables each row the moment its SSE event lands — each rally is interactive before `complete`. (Internal GPU-worker→aggregator hop may use gRPC streaming.)

**Models:** Claude (Opus/Sonnet) via `anthropic`.

**Deliverables:** `habit_report.json`, `coach_report.json`, court map / ghost / overlay renders, fast-tier preview payload.

**Test procedure:**
```bash
python scripts/racketsport/build_report_artifacts.py --metrics runs/phase7/<clip>/racket_sport_metrics.json --out-dir runs/phase9/<clip>
python -m threed.racketsport.eval.copy_faithfulness --root runs/phase9 --labels data/testclips --out runs/phase9/metrics.json
# coach usefulness: 5 reports → 5 coaches, record Y/N "would you use this in a lesson"
```
Measure: LLM-copy faithfulness (no invented facts); fast-tier latency; coach usefulness.

**Acceptance gates:**
- LLM copy faithfulness **100%** (zero invented numbers; any hallucination is a hard fail).
- Fast-tier first-useful-screen **< 10 s** after upload on a 90-s clip; **< 30 s** for full per-shot metrics.
- **Streaming:** first `rally_done` SSE event reaches the client **before the full clip finishes** (perceived latency = time-to-first-rally, not time-to-whole-clip); SSE reconnect survives a dropped connection; APNs push fires on `complete`.
- Coach usefulness **≥ 3/5** would use a report in a lesson.

**Risks:** LLM latency → keep the < 10 s fast tier rule-based + LLM-free; LLM only in the async premium layer.

---

## Phase 10 — 3D Replay Renderer

**Goal:** a watchable, physics-accurate, free-viewpoint 3D replay of the game — players (SMPL-X meshes), court, net, ball, and rackets — shareable on web + app. (See `TECH_STACK.md` §(s).)

**Current delta:** virtual-world assembly, preview/static review artifacts, static review GLB export, GLB structure checks, `replay_readiness_report.json/html`, and a web parser/summary scaffold exist. All four accepted clips are review-visual-ready and each references two valid static GLBs, but those GLBs are classified as `review_static_glb_export` and production validation rejects them. Production replay and metrics readiness remain blocked by missing skeletal animation/skinning/production compression, BODY accuracy, preview-only paddle pose, approximate court-plane ball projection, missing DATA-1 labels, and Outdoor missing BODY mesh. Production animated GLB/USDZ export, compression, CDN delivery, a real Three.js/R3F viewer, and RealityKit playback remain unbuilt.

**Build:**
- `virtual_world.py` + `replay_export.py` — assemble the scene in the **calibrated metric world frame** (court Z=0): procedural court + net mesh; per-player SMPL-X mesh placed by world foot-contact; ball + racket transforms. All from `smpl_motion.json` (foot-locked + physics-refined), `ball_track.json`, `racket_pose.json`.
- `replay_export.py` — **bake physics server-side into skeletal-animation tracks** (foot-locked body θ @30 fps, ball gravity/bounce/Magnus, racket-ball contact). **Author the same scene into BOTH formats directly from the reconstruction** (do NOT round-trip one into the other — `usd_from_gltf` is archived and bloats animation):
  - **USDZ (native iOS replay, C3):** OpenUSD `pxr` — build a `UsdSkelSkeleton` (SMPL rest pose), bind via `UsdSkelBindingAPI`, write per-frame `UsdSkelAnimation`, package with `UsdUtils.CreateNewARKitLayer`. **Bake character + animation in a single USDZ** (a separate-file animation fails with "invalid bind path"). **Bake SMPL-X pose-corrective blend-shape deformation into the animation** so RealityKit's LBS reproduces it faithfully.
  - **GLB (web share):** `pygltflib` + `smplx` — map SMPL joints to a glTF `skin`, export per-frame joint quaternions as **true skeletal animation channels** (not per-frame morph targets, which bloat at 30 fps). Compress with **MeshOpt + Draco + KTX2** (`@gltf-transform/cli`) → **~8–12 MB per 10-s rally**.
  - Shared court+net mesh cached once; per-player skinned mesh (β) + animation + ball + racket per point. Upload both to CDN.
  - **Progressive streaming so a shared link opens fast:** embed the **lowest-LOD skeleton in the GLB root for an instant first render (<500 ms)**, fetch higher-fidelity mesh LODs as separate buffers on demand (`EXT_meshopt_compression` + `KHR_mesh_quantization`); **pre-generate every per-rally GLB at server compute time so it is CDN-warm before the user taps**; long-TTL CDN (CloudFront/Cloudflare) keyed by rally id; preload the shared court mesh first. **HLS-segmented server-rendered video** as the low-bandwidth / non-3D fallback.
- `web/replay/` — **Three.js r171+ + React Three Fiber**, WebGPU renderer with WebGL2 fallback. Loads shared court (cached) + lazy-streams the point GLB; `OrbitControls` for **free-viewpoint**; timeline scrub/loop; camera presets. **Rapier.js in a Web Worker** client-side only for an optional "what-if" mode. Gaussian splatting deferred to v2.
- Native iOS playback is **Phase C3** (RealityKit + USDZ); this phase produces the USDZ it consumes.
- Avatar: render SMPL-X skinned mesh directly (bake β once, drive θ per frame); jersey/skin texture; optional source-video projection for likeness.

**Models:** none new (consumes prior artifacts); SMPL-X body model + `smplx`/Meshcapade `smplcodec`; OpenUSD `pxr` (USDZ) + `pygltflib` (GLB).

**Deliverables:** per-point **GLB (web) + USDZ (native iOS)** on CDN; the `web/replay/` viewer; a shareable replay link consumed by both the web viewer and the Phase-C3 RealityKit viewer.

**Test procedure:**
```bash
python scripts/racketsport/build_replay_review_export.py --virtual-world runs/phase10/<clip>/virtual_world.json --out-dir runs/phase10/<clip>/replay_review --scene-out runs/phase10/<clip>/replay_scene.json
npx @gltf-transform/cli inspect runs/phase10/<clip>/replay_review/points/point_001_review.glb
# Production animated GLB/USDZ export is pending; current command emits static review GLBs only.
# Serve web/replay locally and load the review manifest/scene.
python -m threed.racketsport.eval.replay_eval --root runs/phase10 --labels data/testclips --out runs/phase10/metrics.json
```
Measure: GLB size per rally; viewer cold-load time; free-viewpoint works; foot-skate/penetration visible in-render (regression vs Phase 4 numbers); coach "looks right" review. Static review GLBs and `virtual_world_paddle_preview.html` are visual QA only; they do not satisfy the production replay gate.

**Acceptance gates:**
- Per-10-s-rally GLB **≤ 12 MB**; web viewer cold-load **< 6 s** on a typical connection; holds **≥ 30 FPS** with 4 players on mid-range mobile.
- **USDZ** exports validate and **play in RealityKit (Phase C3)** with skeletal animation + attached ball/racket — single baked file, pose-corrective deformation reproduced.
- Free-viewpoint orbit works; the rendered replay shows **no visible foot-skate or floor/inter-player penetration** (Phase-4 gates hold through to render).
- Ball/racket render physically consistent (bounce on court plane; racket meets ball at contact).
- Coach review: **≥ 3/5** say the replay "looks right" / is useful.

**Risks:** per-player monocular placement consistency → validate depth ordering; if mesh is too heavy on low-end mobile, decimate to ~2k verts / stick-figure context view. WebGPU support gaps → WebGL2 fallback.

---

## Phase 11 — End-to-End Integration, Performance, Final Acceptance

**Goal:** one orchestrated run from upload → fast tier → deep tier → 3D replay on real 20-min clips, optimized, with capture-quality gating and the full artifact set.

**Current delta:** the orchestrator is intentionally fail-closed and can run partial stages, and `pipeline_readiness_e2e.json`, `body_gate_report.json`, `paddle_true_corner_review.json`, and `replay_readiness_report.json` report artifact plus semantic readiness blockers. This is `SCAFFOLD`, not `PROTOTYPE-GATE`: the integrated system is blocked/failed after partial runtime-dependent stages because canonical-safe tracking, BODY accuracy, true paddle pose, production replay export, and final performance/accuracy gates are missing.

**Build:**
- Wire `orchestrator.py` end-to-end: ingest → calibrate → track (fast) → body3d (fast multi-person mesh preview + deep Fast SAM-3D-Body + our world-grounding) → foot-lock + physics → ball/events → racket 6DoF → metrics/insights → report/viz → replay export. **Adaptive compute:** cheap baseline on all frames; escalate deep SAM-3D-Body + physics + racket on rally/contact spans; second pass re-runs heavier models only on low-confidence spans. Honor the single-GPU lock (`test_wp39_single_run_lock` pattern) + scoreboard (`test_wp36_scoreboard` pattern: per-phase timing + GPU cost).
- Performance (full stack — see `TECH_STACK.md §H100 runtime & serving`): **NVDEC/DALI GPU decode** (the #1 bottleneck fix); **TensorRT INT8/FP16 engines** (YOLO26/RTMPose/RTMW/TrackNet/audio; Fast SAM-3D-Body FP16/BF16, never INT8); **CUDA-stream overlap** decode→detect→pose→mesh + **B=4 crop batching** (SAM-3D-Body sequential per player); **on-device pose prior** to cut server SMPL-fit 50–80%; **event-triggered compute** (deep SAM-3D-Body + physics only on rally/contact spans, skip 40–60% dead time); cached betas; temporal subsample+interpolate on dead-time; mesh decimation for render. Serve the multi-model DAG via **Triton ensemble**. Top-4 levers = decode + TRT + stream-overlap + crop-batching.
- `capture_quality` gating wired into the report + replay.
- Long-clip handling: rally segmentation, deep tier on high-value/coach-selected spans only, tiered artifact retention (sparse proxies + habit clips by default; full mesh/replay GLBs only for selected points).

**Test procedure:**
```bash
python -m threed.racketsport.orchestrator --clip data/testclips/<20min_clip> --players 4 --out runs/phase11/<clip> --full
python -m threed.racketsport.eval.e2e_eval --root runs/phase11 --labels data/testclips --out runs/phase11/metrics.json
python -m pytest tests/racketsport -q   # full regression suite
```
Measure: end-to-end fast-tier latency; full deep-tier wall-clock + GPU-cost per 20-min clip; replay export time; artifact completeness; regression suite green.

**Acceptance gates:**
- Fast-tier preview **< 10 s** after a short upload; 20-min fast-tier full timeline **< 3 min**.
- **Throughput after NVDEC + TRT + stream-overlap: the per-rally deep pipeline runs faster than real-time** (the GPU, not CPU decode, is the binding resource — verify via the scoreboard); first `rally_done` streams to the client before the clip finishes.
- Deep-tier full report + replay (on selected spans) p95 within the agreed beta SLA; scoreboard shows GPU cost per report supporting **≥ 70% gross margin** at target pricing.
- Readiness gate: report/replay "ready" only when all required artifacts exist (WP47 pattern).
- Full `tests/racketsport` suite green; all e2e accuracy gates (foot/NVZ, spacing, contact timing, foot-slide ≤3 mm, racket face ≤5°, copy faithfulness) pass on the integrated run.

**Risks:** queue/storage on long clips → rally segmentation + tiered retention mandatory. Deep SAM-3D-Body + physics are the heaviest steps → run on spans, not whole clips; PhysPT (cheap) before the MuJoCo sim (flagship-only).

---

## Data & Training Infrastructure (runs alongside all phases)

Full detail in `ACCURACY_AND_TRAINING.md §11–§14`. Build in parallel — it is what makes the per-phase accuracy gates reachable.

- **Unified 2-camera capture shoot (keystone, ~$80, ~10 h):** 2 phones ~90° apart (cam A baseline-corner = product viewpoint; cam B side-fence = multi-view training signal), auto-calibrated from court keypoints, 1080p/60 (a 120 fps subset for racket/contact), 5–8 players, indoor+outdoor. **One session auto-labels court, person, pose, ball, audio events, and (via the 2nd view) 3D pose at once.**
- **Auto-labeling + distillation + active-learning:** heavy teachers (DINO/YOLO-L person, **multi-view triangulation (2-cam shoot) + SAM-3D-Body** pose — GVHMR/WHAM only as trajectory cross-check, TrackNet full-res ball, audio onset) → **confidence filter** (court reproj <τ, pose conf + temporal smoothness, ball physics-consistent parabolas) → **physics check** (ballistic arc, joint limits) → versioned pseudo-label DB → **active learning** (uncertainty + embedding diversity) → CVAT human verify → **distill into the fast students** that ship. Tools: FiftyOne, CVAT, Roboflow, Label Studio.
- **Multi-view→monocular distillation (the accuracy moat) — TRAINING-TIME ONLY:** the 2-camera shoot exists purely to manufacture 3D ground truth and a **two-view consistency loss** ("Two Views Are Better Than One": −43.6% MPJPE on SportsPose). **The shipped product is strictly single-camera.** A live 2nd-camera/true-triangulation mode is a **FUTURE** capability — do not build it into the v1 product or any phase gate.
- **Synthetic data** (ROI: ball ★★★★★ > court ★★★★ ≈ pose ★★★★ > racket ★★★★ > person ★★): balls along physics trajectories, courts across 50–500 viewpoints, AMASS poses on court backgrounds, **paddle CAD across viewpoints/blur**; post-process every synthetic frame with our real pathologies (H.264/RS/flicker); mix ~30–40%.
- **MLOps / corrections flywheel (the durable moat):** DVC + Roboflow Versions + W&B. Every in-app user/coach correction (wrong shot label, dragged ball/foot/racket) logs clip + predicted-vs-corrected into a corrections queue → active-learning prioritization → next training batch → fewer future corrections. **Retrain triggered** when a correction-rate spikes >3%; **scheduled** every 4–6 weeks.

## Cross-Cutting Requirements

- **Validation dataset & eval harness:** `threed/racketsport/eval/` holds one evaluator per phase, each reading `data/testclips/*/labels/` and writing `metrics.json`. Validation Protocols A/B/C/D (`ACCURACY_AND_TRAINING.md §10`) and the physics/racket gates (Phases 4, 6, 10 acceptance gates) define the numeric gates.
- **Regression dashboard + CI:** current regression coverage is `scripts/racketsport/check_eval_regression.py`, `scripts/racketsport/summarize_eval_runs.py`, and tests under `tests/racketsport/`. The tracked GitHub workflow primarily runs checker tests on relevant file changes; real baseline-vs-current metric comparison is currently a manual `workflow_dispatch` path, not a universal merge blocker.
- **CI:** `tests/racketsport/` owns the current regression coverage; readiness-gate tests check artifact plus semantic blockers, while scoreboard/perf tests should be added only when real per-phase timing/cost artifacts exist.
- **Artifact schema registry:** every JSON artifact has a `pydantic` schema in `threed/racketsport/schemas/` with `schema_version`; `validate()` runs in CI.
- **Confidence plumbing ("no charge if we can't trust it"):** confidence + coverage propagate per-frame pose → per-metric → per-habit → report; report exposes `coverage.overall` + `skipped_reason_counts`; below-threshold reports flagged comp-able. Wired in Phase 7, surfaced in Phase 9, honored in the Phase 10 replay (gray/omit low-confidence).

---

## Artifact JSON Schemas (authoritative)

```jsonc
// capture_sidecar.json  (produced by iOS Phase C1; consumed by server Phase 1)
{
  "schema_version": 1, "device_tier": "A_lidar|B_standard|fallback",
  "device_model": "iPhone16,2", "fps": 120, "format": "hevc|prores422lt",
  "resolution": [1920, 1080], "orientation": "landscape",
  "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": true},
  "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "dist": [..], "source": "arkit|charuco|exif|geocalib"},
  "arkit_camera_pose": {"R": [[...]], "t": [..]},        // null if fallback
  "court_plane": {"point": [X,Y,Z], "normal": [..]},      // ARKit plane OR from manual taps
  "manual_court_taps": [[x,y]],                            // present when ARKit plane unreliable
  "gravity": [gx, gy, gz],                                // CoreMotion
  "lidar_depth_refs": ["depth/000123.bin"],              // Tier A only; near-court assist
  "ondevice_pose_track": "ondevice_pose.json",          // C2 prior to speed up server fit
  "capture_quality": {"grade": "good|warn|poor", "reasons": []}
}

// court_calibration.json
{
  "schema_version": 1, "sport": "pickleball",
  "homography": [[...],[...],[...]],
  "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "dist": [..]},
  "extrinsics": {"R": [[...]], "t": [..], "camera_height_m": 0.0},
  "reprojection_error_px": {"median": 0.0, "p95": 0.0},
  "capture_quality": {"grade": "good|warn|poor", "reasons": []},
  "image_pts": [[x,y]], "world_pts": [[X,Y,Z]]
}

// court_line_evidence.json
{
  "schema_version": 1, "sport": "pickleball", "source": "auto_hough_template_video",
  "line_observations": [
    {"line_id": "near_nvz|far_nvz|near_centerline|far_centerline", "image_segment": [[x,y],[x,y]],
     "confidence": 0.0, "frame_indexes": [0], "residual_px": {"mean": 0.0, "p95": 0.0},
     "visible_fraction": 0.0, "source": "auto_hough_template"}
  ],
  "keypoint_observations": [],
  "net_observations": [
    {"net_id": "top_net", "image_points": [[x,y],[x,y],[x,y]], "confidence": 0.0,
     "frame_indexes": [0], "residual_px": {"mean": 0.0, "p95": 0.0}, "source": "auto_hough_net_top"}
  ],
  "aggregate": {
    "accepted_line_ids": [], "rejected_line_ids": [], "missing_required_line_ids": [],
    "missing_required_net_ids": [], "mean_residual_px": 0.0, "p95_residual_px": 0.0,
    "temporal_stability_px": 0.0, "auto_calibration_ready": false, "reasons": []
  }
}

// court_zones.json
{ "schema_version": 1, "zones": {"nvz_near": [[X,Y]], "nvz_far": [...],
  "transition_near": [...], "baseline_near": [...], "service_boxes": [...] } }

// net_plane.json
{ "schema_version": 1, "plane": {"point": [X,Y,Z], "normal": [..]},
  "endpoints": [[X,Y,Z],[X,Y,Z]], "center_height_in": 34, "post_height_in": 36 }

// tracks.json
{ "schema_version": 1, "fps": 30, "players": [
  {"id": 1, "side": "near", "role": "left",
   "frames": [{"t": 0.0, "bbox": [x,y,w,h], "world_xy": [X,Y], "conf": 0.0}]} ],
  "rally_spans": [{"t0": 0.0, "t1": 0.0}] }

// smpl_motion.json  (PRIMARY body artifact — world-grounded, foot-locked, physics-refined)
{ "schema_version": 1, "model": "smplx", "fps": 30, "world_frame": "court_Z0",
  "players": [
   {"id": 1, "betas": [..],
    "frames": [{"t": 0.0, "global_orient": [..], "body_pose": [..],
      "left_hand_pose": [..], "right_hand_pose": [..],
      "transl_world": [X,Y,Z], "joints_world": [[X,Y,Z]], "joint_conf": [..],
      "foot_contact": {"left": true, "right": false},
      "grf": [[Fx,Fy,Fz]]}],   // grf optional (flagship physics)
    "skate_free": true, "physics": "footlock|physpt|mujoco_phc"} ] }

// skeleton3d.json  (preview/triggering overlay only)
{ "schema_version": 1, "joint_names": [...], "preview_only": true,
  "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[X,Y,Z]], "joint_conf": [..]}]}] }

// ball_track.json
{ "schema_version": 1, "fps": 30, "source": "tracknet|tap",
  "frames": [{"t": 0.0, "xy": [x,y], "conf": 0.0, "visible": true,
    "world_xyz": [X,Y,Z], "spin_rpm": 0.0, "approx": true}],
  "bounces": [{"t": 0.0, "world_xy": [X,Y]}] }

// racket_pose.json
{ "schema_version": 1, "fps": 30, "players": [
  {"id": 1, "paddle_dims_in": {"w": 8, "h": 16},
   "frames": [{"t": 0.0, "pose_se3": {"R": [[...]], "t": [X,Y,Z]}, "conf": 0.0}],
   "contacts": [{"t": 0.0, "contact_point_face_cm": [u,v], "face_normal": [..], "conf": 0.0}]} ] }

// contact_windows.json
{ "schema_version": 1, "events": [
  {"type": "contact|bounce|net_cross", "t": 0.0, "frame": 0,
   "player_id": 1, "confidence": 0.0,
   "sources": {"audio": 0.0, "wrist_vel": 0.0, "ball_inflection": 0.0},
   "window": {"t0": 0.0, "t1": 0.0, "importance": 0.0}}] }

// racket_sport_metrics.json
{ "schema_version": 1, "players": [
  {"id": 1, "shots": [{"t": 0.0, "type": "dink", "type_conf": 0.0,
    "metrics": {"nvz_margin_ft": {"value": -0.6, "conf": 0.86, "frames": 5},
                "x_factor_deg": {"value": 31, "conf": 0.7, "gated": false},
                "paddle_face_deg": {"value": 4.0, "conf": 0.82, "gated": false, "source": "racket6dof"},
                "contact_point_face_cm": {"value": [1.2, -0.5], "conf": 0.8}}}]} ] }

// habit_report.json
{ "schema_version": 1, "sport": "pickleball",
  "coverage": {"overall": 0.82, "skipped_reason_counts": {"ball_uncertain": 4, "foot_occluded": 7}},
  "priority_habit_id": "kitchen_foot",
  "replay_ref": {"glb_url": "/replay/<match>?point=3"},
  "habits": [{"id": "kitchen_foot", "title": "Kitchen foot", "summary": "...",
    "confidence": 0.86, "clip_ref": {"t0_sec": 31.2, "t1_sec": 41.2},
    "court_metric": {"nvz_margin_ft": -0.6}, "body_metric": {"support_state": "right_contact"},
    "cue": "...", "drill": {"name": "...", "duration_min": 6},
    "trend_vs_last": {"nvz_margin_ft": +0.4}}] }

// replay_scene.json  (manifest for the GLB export)
{ "schema_version": 1, "world_frame": "court_Z0", "fps": 30,
  "court_glb": "court_pickleball.glb", "players": [1,2,3,4],
  "points": [{"id": 3, "t0": 31.2, "t1": 41.2, "glb_url": "point_3.glb", "size_mb": 9.4}] }

// drill_report.json
{ "schema_version": 1, "drill": "kitchen_dinks", "reps": 10, "clean_reps": 7,
  "per_rep": [{"t": 0.0, "quality": "clean|fault", "reasons": []}] }
```

---

## One-Page Build Order Summary

**Parallelization (see `BUILD_CHECKLIST.md §1.5`):** the **Client Track (C1–C3)** and **Data & Training Infra** run **fully in parallel** with the Server Pipeline Track from day one (mostly `[CPU/IO]` + on-device). Within the server track, phase-gate order holds for `[GPU]` gate tests, but each phase's `[CPU/IO]` coding (schemas, scaffolding, CLI wiring, viz) can start early. **All `[GPU]` work — training, heavy inference, and every Acceptance-Gate test — serializes through the single-H100 `flock` lease (MIG eval slices in parallel; full-GPU lease for training).** A phase is `VERIFIED` only after its gate test is actually run on the shared H100 and the numbers are recorded.

**Client Track (iOS app) — parallel to the server pipeline:**

| Phase | Builds | Hard gate to advance |
|---|---|---|
| C1 | iOS capture (locked exposure/focus, fps policy, landscape, HEVC/ProRes) + ARKit calibration sidecar | locked-luminance std <2%; sidecar schema-valid 100%; fps within ±2 |
| C2 | on-device fast tier (Apple Vision + YOLO Core ML) + capture guidance | on-device 2D pose ≥30 FPS; all guidance flags fire; preview <10 s |
| C3 | native RealityKit + USDZ replay viewer (free-viewpoint) | USDZ plays ≥30 FPS, 4 players; no visible skate/penetration |

**Server Pipeline Track (Phases 0–11):**

| Phase | Builds | Hard gate to advance |
|---|---|---|
| 0 | env, scaffold, NVDEC, checkpoints | decode ≥8× RT; all models load (or fallback recorded) |
| 1 | court calib + pose + net plane | overlay ≥90%, uniform across camera poses; feet <0.5 ft |
| 2 | detect/track/doubles ID (YOLO26m+BoT-SORT-ReID) | ≥90% 2p / ≥85% doubles ID; 20-min <90 s |
| 3 | 3D body mesh-core (Fast SAM-3D-Body deep + our grounding; Multi-HMR2/SAT-HMR fast) | per-frame PA-MPJPE ~30 mm; foot/NVZ ≥80%; fast mesh ≥20 FPS; velocity decided |
| 4 | foot-skate elimination + physics | **foot-slide ≤3 mm; 0 penetration**; contact P≥0.9/R≥0.85 |
| 5 | ball (TrackNet) + audio + events + 3D physics | ball F1 ≥0.90 (FP<5%); contact ≤±2 frames / ±4 ms; 3D obeys physics |
| 6 | racket 6DoF | **face-angle ≤5°; contact-point ±1–3 cm** |
| 7 | metrics + rules + confidence gating | NVZ ±6 in; rule precision ≥75%; rotation gated unless via racket |
| 8 | shot classification + drill reps | shot F1 ≥0.65/top-2 ≥0.85; reps ±1 |
| 9 | LLM copy + reports + viz + tiers | copy 100% faithful; <10 s fast tier; ≥3/5 coaches |
| 10 | 3D replay renderer | GLB ≤12 MB/10 s; <6 s load; no visible skate/penetration; ≥3/5 "looks right" |
| 11 | e2e integration + perf | <10 s preview; SLA + ≥70% margin; suite green |
