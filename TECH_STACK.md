# Sway Body — Pickleball/Tennis CV Technology Stack

**Status:** Target/default technical reference for implementation; approved runtime locks live in `models/MANIFEST.json` after the EVAL-0 gate.
**Date:** 2026-06-26
**Scope:** Every model/technology in the pipeline, the decision behind it, and the speed/accuracy/license/UI tradeoff it sits on.
**Companion docs:** `SWAY_BODY_PICKLEBALL_MVP.md` (product/market), `IMPLEMENTATION_PHASES.md` (Codex build + test plan), `ACCURACY_AND_TRAINING.md` (**source of truth for datasets, training/fine-tuning/auto-labeling recipes, filtering parameters, and validation protocols**), `BUILD_CHECKLIST.md` (the operational checklist + multi-agent coordination protocol). World-grounded mesh, foot-skate elimination, physics refinement, racket 6DoF, and the 3D replay renderer are detailed per-layer below ((e),(p)–(s)) and built in `IMPLEMENTATION_PHASES.md` Phases 3/4/6/10.

This document states decisions, not options. Where a number is unverified in source research it is flagged `[UNVERIFIED]`. The live implementation is in this `pickleball` repo while reusing proven `sam4dbody` runtime ideas and assets where appropriate; it does not start greenfield.

> **Round-4 revision (2026-06-26):** the product is **not being sold yet** (research/personal use), so **licensing is no longer a build blocker** — we use the most accurate model per stage regardless of license (license recorded in §5 for future commercialization). This **flips the core body decision: a full body MESH is now CORE** (world-grounded, physics-refined, foot-skate-free) and the fast skeleton is demoted to a **preview/triggering overlay**. New goals — a **physics-accurate 3D replay** and **no foot-skating** — add four layers: world-grounded body + foot-lock (p), physics refinement (q), **racket 6DoF** (r), and the **3D replay renderer** (s). Detailed in (p)–(s) below and built in `IMPLEMENTATION_PHASES.md` Phases 4/6/10.
>
> **Round-4b correction (verified — supersedes the GVHMR-primary call):** the per-frame mesh backbone is **Fast SAM-3D-Body**, not GVHMR. Reality-check (and the user's hands-on results) confirm **SAM-3D-Body is the best per-frame mesh available** (3DPW PA-MPJPE **30.4 mm**, beats HMR2.0 by 15+ mm) — usable now under the licensing-lifted policy; for future commercial use the SAM License must be verified, with **SAT-HMR (Apache)** as the license-safe fallback. **GVHMR/WHAM are HMR2.0-backbone pipelines: their per-frame mesh is *worse* than SAM-3D-Body; their only advantage (world trajectory + foot-contact) is something we already solve via our known camera + court plane.** So they are **demoted to optional world-trajectory/foot-velocity sanity-checks**, not the primary mesh. **World-grounding is ours**: project per-frame SAM-3D-Body output to world via known K,[R|t] + court Z=0 → temporal smooth → foot-lock → physics (precedent: arXiv 2512.21573). **PromptHMR is dropped** (full-image hurts distant players; needs manual prompts; worst translation error). **Detector flips to YOLO26m + BoT-SORT-ReID** (YOLO26 is real — Ultralytics Jan 2026, +1.4–2.8 mAP over YOLO11 at equal GPU latency). Sections below reflect 4b.
>
> **Round-5 (handoff): SINGLE CAMERA is the product; native iOS Swift client + GPU server.** Multi-camera is **future**; multi-view is a **training-time-only** instrument that ships a single-camera model (§4). The product is a **native iOS Swift app** (capture + on-device fast-tier preview + capture guidance + native replay viewer) talking to a **GPU server** (the deep tier). This adds a client/server split (§2.1), four client-side layers — iOS capture (t), iOS calibration via ARKit (u), on-device fast tier via Apple Vision + Core ML (v) — and makes rendering **native RealityKit/USDZ in-app + Three.js/GLB on the web** (s). Apple's on-device 3D pose is **preview/guidance only** (17 joints, no fingers, single-person, ~5–25° error); the **server SAM-3D-Body mesh stays the source of truth**. **LiDAR (Pro devices) is a near-camera (~5 m) bonus only** — it cannot reach a 6–12 m court or the ball; vision-first is the baseline.

---

## 1. Design Philosophy

### The core identity: an adaptive compute budget

Every competitor runs **one fixed model at a fixed frame rate over the whole video**. That is the structural weakness we attack. Sway Body treats the pipeline as a **budget allocator**: a cheap baseline runs over 100% of frames, and expensive compute is spent surgically in exactly three places —

Current implementation status: `frame_compute_plan.json` and
`body_compute_execution.json` now express that budget in the fail-closed spine,
and `BodyStageRunner` consumes planned deep-mesh windows when they exist. The
accepted-four prototype clips now have canonical `contact_windows.json` files
promoted from explicit human review inputs, with `player_id` still untrusted/null.
Current BODY scheduling is limited and fail-closed: Burlington schedules 3 frames
/ 9 player-frames, Wolverine 4 / 12, Outdoor webcam 0 / 0 due player coverage,
and Indoor doubles 2 / 6. These are scheduling artifacts only; they do not make
BODY or BALL verified.

1. **Where it matters biomechanically** — the swing/contact window, the split-step, the balance-loss moment.
2. **Where the user sees and pays** — the avatar "wow" frame, the shareable highlight, the premium replay.
3. **Where the model is uncertain** — low-confidence spans get *escalated* to a heavier model, never bluffed.

This is why we can be **faster AND more accurate than the market simultaneously** — it is a different cost curve, not a point on the same one.

### Four principles that fall out of this

| Principle | What it means | Why |
|---|---|---|
| **Mesh-core, skeleton as fast preview** *(round-4)* | A fast multi-person mesh/skeleton runs live for preview/metrics/contact-triggering; the **world-grounded body mesh — Fast SAM-3D-Body per frame, grounded to the court plane by us — is the core representation** for the physics-accurate replay, foot-lock, and biomechanics. | The replay needs a volumetric body (can't drive a rigged avatar from a stick figure); foot-skate is solved by foot-locking a strong per-frame mesh to the known ground plane; SMPL/SMPL-X params are the animation/physics lingua franca. SAM-3D-Body is the best per-frame mesh (PA-MPJPE 30.4 mm) and runs fast enough (Fast variant ~15 FPS/person) for the deep tier; SAT-HMR/Multi-HMR 2 cover the fast tier. |
| **Every layer is an accuracy/speed/UI toggle** | Each stage has a cheap setting and a rich setting; the product (and the pricing tier) chooses where to spend. | Lets one codebase serve a free "Instant" tier and a paid "Deep" tier off the same pipeline. |
| **Single static camera, highly variable height/angle** | Nothing about camera pose is hardcoded. Everything geometric is solved per-clip from the court itself. | Users place one tripod at whatever height/angle they can. Robustness to viewpoint is a first-class requirement, not a nice-to-have. |
| **Confidence-gating is a trust feature** | Low-confidence metrics are grayed/omitted and not charged for; we never surface a number we can't defend. | The #1 documented churn driver in AI sports apps is wrong outputs, not missing features. |

### Capture spec (the input contract — biggest free accuracy win)

The **native app configures capture via AVFoundation; the user only frames it** — user-facing ask is three things: **Landscape · stable tripod · all four corners in view (good light)**. Target product capture sets: **landscape, 1080p/60 fps** (120 for swing-speed/racket), **HDR off**, **locked exposure + focus + white balance**, **shutter ≥1/500 s** (1/1000 s ideal; 1/100 or 1/120 s indoor to kill flicker), tripod target **≥1.2–1.5 m**. Current repo reality: the Swift app/package has partial capture runtime, orientation guidance, landscape defaults, and code-level readiness/start-recording rejection for portrait policies, but physical-device capture gates, sidecar writing, and fps/codec/luminance proof are not yet verified. **Locking the auto-systems before recording is the single biggest free accuracy win** — it stops exposure/focus/WB from pumping mid-rally and destabilizing ball/pose detection.

- **Frame rate:** **60 fps floor; 120 fps preferred and the default when light permits** (trivial via AVFoundation on iPhone 13+) — **120 fps is required for racket 6DoF (r)** (a ~1000°/s swing rotates ~33°/frame at 30 fps and the SE(3) filter diverges) and for absolute swing-speed. A **240 fps (≈720p binned, rear-only) "ball-physics deep-dive"** is optional; **don't make 240 fps the default** (resolution loss hurts pose/ball more than the fps helps). Audio anchors contact timing sub-frame regardless of fps.
- **iOS mechanism (layer (t)):** AVFoundation `setExposureModeCustom(duration:1/1000…1/500, iso:)` + `setFocusModeLocked` + `setWhiteBalanceModeLocked`; pick the high-fps `activeFormat`; **landscape via `videoRotationAngle`** (iOS 17+); record **HEVC** (ProRes on Pro) via `AVAssetWriter`; live frames via `AVCaptureVideoDataOutput`. This is the target runtime path; current checked-in code has partial capture scaffolding plus code-level landscape defaults/recording guards, and still needs physical-device verification plus lock/fps/codec/luminance gates. Full pathology-to-mitigation table in `ACCURACY_AND_TRAINING.md` §2.

### How this differs from the dance (`sam4dbody`) pipeline

The dance pipeline runs one heavy path uniformly: `video → 2D identity tracking → SAM2 masks → SAM-3D-Body MHR mesh → smoothed floor-anchored outputs → S3 + callback`. That was correct for dance (many people, chaotic motion, stage-centric output, latency-tolerant). It is wrong for racket sports.

| Dimension | Dance pipeline | Sway Body pickleball pipeline |
|---|---|---|
| Compute allocation | One heavy model, uniform over all frames | Adaptive budget: cheap baseline + event-triggered heavy bursts |
| People finding | Heavy tracker (many dancers, high ID-swap risk) | Cheap detector + court-geometry association (≤4 players, fixed court) |
| Body representation | SAM-3D-Body mesh on everything, camera-space, uniform | Same strong backbone (Fast SAM-3D-Body) but tiered: fast multi-person preview (SAT-HMR/Multi-HMR 2) + per-frame SAM-3D-Body grounded-to-court + foot-lock + physics on replay spans |
| Scene context | Stage floor projection | Court calibration, zones (NVZ), net plane, ball, audio |
| Latency model | Single long async job | Two-speed: fast tier (<10s feel) + deep tier (async, notified) |
| Geometry | Stage-centric | Per-clip court solve robust to any camera height/angle |
| New cheap sensor | — | Audio "pop" drives contact timing + event triggers |

---

## 2. End-to-End Pipeline

```
                          INGEST (NvDEC hardware decode, trim, QC, capture-quality score)
                                              |
        ┌─────────────────────────────────── FAST TIER ───────────────────────────────────┐
        │  (target: usable result <10s; "Instant" product tier)                            │
        │                                                                                   │
        │  court calibration (tap/auto homography + solvePnP + reproj gate) → court + zones │
        │  YOLO26m person detect → BoT-SORT-ReID + HSV + court-polygon → ≤4 locked IDs       │
        │  TrackNetV3 ball + audio "pop" → rally segmentation + contact/bounce events        │
        │  RTMW-l 2D → Multi-HMR 2 / SAT-HMR camera-space mesh (~20-24fps); skeleton lift for │
        │     sub-frame contact triggering; feet snapped to court plane                      │
        │                                                                                   │
        │  → PREVIEW: court map + player paths + 1 priority metric + live mesh/overlay       │
        └───────────────────────────────────────────────┬───────────────────────────────────┘
                                                         │  events define WHERE to spend
        ┌──────────────────── DEEP / REPLAY TIER (async, notified) ───────────────────────────┐
        │                                                                                     │
        │  Fast SAM-3D-Body per player (best per-frame mesh) → grounded to world via known    │
        │     K,[R|t]+court Z=0 (our grounding; GVHMR/WHAM = optional trajectory x-check only)  │
        │     → FOOT-SKATE KILLER: contact vs court Z=0 + zero-velocity + CCD-IK (~2-3mm)      │
        │     → PHYSICS REFINEMENT: PhysPT (default) / PHC+PULSE on MuJoCo+MJX (flagship)      │
        │        + MultiPhys (inter-player non-penetration, doubles)                           │
        │  RACKET 6DoF per player: detect → kpts(top/bottom/handle) → PnP-IPPE → UKF →         │
        │        physics-validate → contact-point + face-angle (~3-5°)                         │
        │  BALL 3D physics: TrackNet → z=0-bounce uplift + Magnus → contact impulse            │
        │                                                                                     │
        │  biomechanics metrics → rule-based insight engine (+confidence gating)              │
        │     → shot classification (scaffold now; PoseConv3D default vs BST candidate)        │
        │                                                                                     │
        │  → 3D REPLAY (Three.js+R3F, SMPL-X GLB, physics baked→glTF, free-viewpoint)         │
        │  → REPORT: top habits, clips, self-vs-self ghost, drill, trend                       │
        └─────────────────────────────────────────────────────────────────────────────────────┘
```

Low-confidence escalation runs across both tiers: any span the fast tier flags uncertain is re-run at higher fidelity in the deep tier rather than reported.

---

## 2.1 iOS Client / Server Architecture (native Swift app)

The product is a **native iOS Swift app** + a **GPU server**. The phone does what it is uniquely good at (controlled capture, free camera geometry from ARKit, instant on-device preview/guidance on the Neural Engine, and native 3D playback); the server does the heavy, accuracy-critical reconstruction. **Single camera only for v1.**

| Task | On-device (iPhone) | Server (H100) | Why there |
|---|---|---|---|
| Capture (locked exposure/focus/WB, fps/format) | ✅ AVFoundation (t) | — | hardware control only the device has |
| Camera intrinsics + 6DoF pose + court-plane | ✅ ARKit setup pass → sidecar (u) | refine with solvePnP (b) | ARKit gives calibration **for free**; server refines |
| Fast-tier preview pose + ball/racket | ✅ Apple Vision + YOLO Core ML (v) | — | instant, on the Neural Engine, free |
| Capture-quality guidance (framing/light/level) | ✅ Vision + ARKit + CoreMotion (v) | — | must be real-time, before recording |
| Person detect/track (deep) | — | ✅ YOLO26m + BoT-SORT (c) | accuracy + batch throughput |
| **3D body mesh (source of truth)** | — | ✅ **Fast SAM-3D-Body + our grounding (e/p)** | ViT-H-class; can't run real-time on-device |
| Foot-lock, physics, racket 6DoF, ball 3D | — | ✅ (p)(q)(r)(h) | heavy, global, deterministic |
| Metrics, insights, LLM copy | — | ✅ (k)(l) | server-side, async |
| **Render bake → animated scene** | — | ✅ one bake → **USDZ + GLB** | author once, play everywhere |
| Replay playback | ✅ RealityKit/USDZ (s) | — | native free-viewpoint viewer |
| Web share playback | (link) | served GLB | Three.js cross-platform (s) |

**Upload payload:** trimmed HEVC/ProRes clip + sidecar (ARKit intrinsics/pose/court-plane + on-device pose track as a **server prior** + LiDAR depth frames if Tier A). **Latency model:** on-device preview is instant; the deep-tier replay is async (seconds–minutes) and swapped in when ready.

### Device tiers

- **Tier A — Pro + LiDAR (iPhone 12 Pro → 16 Pro):** ProRes to server, ARKit calibration, **LiDAR `sceneDepth` as a near-camera (~5 m) bonus** — near-player foot-ground contact and near court-plane refinement only. Best accuracy.
- **Tier B — standard iPhone (no LiDAR, 13–16 non-Pro):** HEVC, 1080p120/240, ARKit world-tracking + plane detection for calibration (no depth). **Vision-only — this is the product baseline and fully viable.**
- **Fallback (older / no ARKit-3D):** AVFoundation locked capture + HEVC + manual court-corner-tap calibration.

**LiDAR honesty:** effective range ~5 m, fails in direct sun, 256×192 @ 60 Hz → **useless for the far player, far baseline, and the ball in flight** (court is 6–12 m away). Treat depth as a Tier-A bonus, never a dependency; **ball 3D comes from multi-frame triangulation + physics on the server**, not from any depth sensor.

---

## 2.2 Live vs Offline / Performance Architecture

§2.1 splits by *task placement*; this splits by *latency and how results reach the user*. Design target: **show something useful instantly, stream the accurate version as it computes, never make the user wait for the whole clip** (edge-first hybrid — the SwingVision pattern).

### Three perceived tiers
| Tier | Budget | What the user sees | Where |
|---|---|---|---|
| **Instant** | **<1 s** | live cues during play (framing OK, swing/split-step detected, late-contact hint), live skeleton/court overlay | on-device, Neural Engine |
| **Preview** | **<10 s, no upload wait** | coarse on-device replay + summary the moment recording stops (split-steps, swing count, rough contact timing) | on-device |
| **Deep / async** | **streamed + push** | accurate biomechanics rally-by-rally, habits as detected, progressive 3D replay; APNs push on completion | server (H100) → SSE/CDN |

### LIVE-vs-OFFLINE assignment (latency budget)
Live budgets are per-frame (33 ms = 30 fps); offline budgets are wall-clock within the rally job.

| Task | Where | Budget | Why |
|---|---|---|---|
| Framing / camera-OK check | LIVE on-device | <16 ms/f | instant during setup (YOLO-nano + court-corner check) |
| 2D body pose (≤4 players) | LIVE on-device | 5–20 ms/f | `VNDetectHumanBodyPose` 30 fps on A14+ |
| Player + ball detect/track | LIVE on-device | 8–15 ms/f | YOLO-nano Core ML ~85 FPS A17 + Kalman |
| Hand/grip coarse pose, segmentation overlay | LIVE on-device | 8–20 ms/f | Vision hand pose + fast segmentation |
| **Coarse live cues** (split-step/swing/late-contact "detected") | LIVE on-device | <33 ms–200 ms | thresholds on 2D pose velocity + ball proximity — the live-coaching payload |
| On-device rally segmentation (skip-logic) | LIVE on-device | ~0 added | ball-gap >1 s = dead time → tags spans so the server skips them |
| Coarse monocular 3D preview | LIVE on-device (optional) | 15–25 ms/f | `VNDetectHumanBodyPose3D` — preview only, noisy |
| **Accurate SMPL-X mesh, foot-lock, physics, racket 6DoF, ball 3D** | OFFLINE server | rally job | needs GPU + temporal/global solve |
| ±2° biomechanics, frame-accurate contact, temporal ID | OFFLINE server | rally job | smoothing/mesh-fit impossible from raw on-device pose |
| Shot classification, habit detection | OFFLINE server | per-rally, streamed | aggregates across the rally |
| Personalized betas (cached) | OFFLINE server, once | ~0 after | fit once, reuse forever |
| Replay render (GLB/USDZ) | OFFLINE server → CDN | per-rally | pre-generated, streamed |

The honest line: **on-device = coarse-enough geometry for "what just happened" guidance; server = noise-filtered, mesh-fit, sub-degree "exactly why" truth.**

### Progressive delivery (report)
Stream each rally's result the instant it's computed — **perceived latency = time-to-first-rally, not whole-clip.**
- **Transport: SSE** (iOS↔server) — the same one-way push behind LLM token streams (stateless, HTTP, auto-reconnect, trivial in FastAPI). WebSocket only if we later add live bidirectional drill feedback.
- **Internals:** GPU **Celery** worker → **Redis Pub/Sub** → FastAPI SSE endpoint → iOS (server-internal gRPC streaming for protobuf efficiency).
- **Events:** `job_accepted` → `rally_total:N` → `rally_done:{i, metrics, replay_url}` ×N → `habits_partial` → `complete`; fire an **APNs push** on `complete`.
- **UX:** optimistic rally checklist ("3/14 analyzed…"), each row tappable the instant it lands; don't gate the UI on `complete`.

### Replay streaming
Skeletal replay is tiny (<1 MB/min), so the win is per-rally streaming + CDN, not heavy compression. **Per-rally GLB on a CDN, pre-warmed** at compute time; **progressive LOD** — low-res skeleton in the GLB root renders <500 ms, higher mesh LODs fetched on demand; **EXT_meshopt_compression + KHR_mesh_quantization** (faster decode than Draco, 60–80% smaller). Native: RealityKit async `Entity(named:)` load-low-then-swap. Web: `<model-viewer>`/Three.js on the same CDN GLBs. HLS-segmented server-rendered video as the low-bandwidth/non-3D fallback. (Renderer detail in (s).)

### Top latency-reduction levers (ranked)
1. **Upload the on-device 2D pose track as a server prior — and start fitting before the video finishes uploading.** ~**166× smaller** than video (≈1.8 MB vs 300 MB for 5 min) and **50–80% fewer SMPL-fit iterations** (good 2D init drives convergence); send the ~50 KB pose file on record-stop so the server fits *while* the video uploads in the background. The single biggest lever (and a privacy win — skeleton ≪ raw video).
2. **Event-triggered compute — process rally spans only.** Recreational pickleball is 40–60% dead time; on-device rally tags let the server skip it → ~2× throughput.
3. **Cache the personalized body model (betas) across sessions** — convergence drops ~100→~20 iterations on every repeat session.
4. **Cold-start avoidance** — keep `min_containers=1` during active hours, bake model weights into the image (not runtime-fetched), heartbeat to prevent eviction (cold start can be ~75× slower).
5. **Trim on-device + chunked/resumable upload** (TUSKit / iOS 17 background `URLSession`, write to disk first) — survives app suspension.
6. **Rally-by-rally streaming** (above) — collapses *perceived* latency to time-to-first-rally.

---

## 2.3 Model Registry — Target Defaults And Candidates

This is the registry of target defaults, candidates, and fallbacks for model/variant/weight decisions, split by **OFFLINE (server H100, accuracy-first)** and **LIVE (on-device iOS, real-time preview/guidance only)** tier. It is not proof that every listed model is currently invoked; approved locks move to `models/MANIFEST.json` after EVAL-0.

**Selection rule (how a variant becomes final):** the values below are **defaults + candidates**, not final. Codex benchmarks the candidates on real clips (offline on the H100 via the `BUILD_CHECKLIST.md §1.5` GPU lease; live on a test device), **renders side-by-side comparison videos, and the human approves the pick before it is locked — unless one variant is obviously better, in which case Codex may auto-finalize and log it** (the full gate is `BUILD_CHECKLIST.md §1.6`; procedure in `IMPLEMENTATION_PHASES.md §0.7 "Model variant selection"`). It locks the approved variant whose accuracy meets the phase gate. Bias: pick the most accurate variant whose latency is acceptable for its tier — on the H100 the gap between detector/pose sizes is sub-millisecond, so lean to the bigger one *only where it buys accuracy we need* (for ≤4 large players we are tracking-limited, not detection-limited, so the mid detector is plenty). **LIVE = lightest variant that gives a usable preview; OFFLINE always recomputes with the accurate variant** — the live output is a preview + a server *prior*, never the final result.

> ⚠️ **Every speed number in this repo is a T4/A100/RTX extrapolation — there are NO published H100 single-image latencies.** Treat all FPS/ms as estimates until Codex re-benchmarks on the real H100 + on-device. Benchmark **Fast SAM-3D-Body first** — it is the pacing item and sets the MIG geometry.

| # | Stage | OFFLINE default (weights · license) | LIVE default | Candidates to benchmark | Fallback |
|---|---|---|---|---|---|
| 1 | Person detect | **YOLO26m** (`ultralytics yolo26m.pt` · AGPL) — accurate enough for ≤4 big players | **YOLO26n** Core ML INT8 (ANE) | 26m / 26l / 26x (offline if far-court recall low) | offline → YOLO11m; live → YOLO11n (if 26 Core ML misbehaves) |
| 2 | Tracker | **BoT-SORT + ReID** (`osnet_ain_x1_0_msmt17` · MIT/Apache) | — (server) | OSNet x0.25 / x1.0 / ain_x1.0 | ByteTrack (no ReID) |
| 3 | 2D pose (server) | **RTMW-l @384** (mmpose · Apache) | — | RTMW-x (≈+0.1 mAP), RTMPose-x | RTMW-l |
| 3b | 2D pose (per-crop) | **RTMPose-m @384** (Apache) | — | RTMPose-l / -x | RTMPose-m |
| 4 | 3D body backbone | **Fast SAM-3D-Body** (`facebook/sam-3d-body-dinov3` + `yangtiming/Fast-SAM-3D-Body` · SAM License ⚠️ verify-commercial) | — (always server) | Fast SAM-3D-Body / SAT-HMR / Multi-HMR 2 | **SAT-HMR (Apache — license-safe)** → NLF → HMR2.0 |
| 4b | 3D mesh (server preview) | **SAT-HMR** (`sat_644_3dpw.pth` · Apache) | — | SAT-HMR vs Multi-HMR 2 (Naver ⚠️) | HMR2.0b (MIT) |
| 5 | Ball | **TrackNetV3** (`qaz812345/TrackNetV3`, badminton→fine-tune PB · MIT) | — | TrackNetV4 tennis weights; TrackNetV5 (no code yet) | manual tap-track |
| 6 | Court keypoints | **TennisCourtDetector** (`yastrebksv`, no-LICENSE ⚠️) → fine-tune + add ~4 kitchen-line kpts | (computed once/setup, reused) | yastrebksv init vs scratch | manual 4-tap homography |
| 7 | Racket 6DoF | **SAM 3 concept detection/tracking when approved, DINO-X/Grounded-SAM-2 fallback, FoundationPose/GigaPose/FoundPose CAD or reference-pose scoring, plus PnP-IPPE/UKF** (research/runtime probes still required) | — | SAM 3 vs DINO-X/Grounded-SAM-2; FoundationPose vs GigaPose vs FoundPose | true-corner IPPE gate; no box-derived promotion |
| 8 | Audio contact | **BEATs** (or AST) fine-tuned (MIT — no PB pretrain) | small CNN on mel-spec (on-device) | BEATs vs AST vs small CNN | AST |
| 9 | Shot classifier | **PoseConv3D** (pyskl SlowOnly-R50) adapted (Apache); BST as ball-aware option | — | PoseConv3D vs BST | train own |
| 10 | Physics | **PhysPT** (MIT) + **PHC+/PULSE** on **MuJoCo 3.9/MJX** | — | PhysPT vs PHC+/PULSE | PHC |
| 11 | Live pose | — (server uses #3) | **Apple Vision**: `VNDetectHumanBodyPose` (2D) + `VNDetectHumanBodyPose3D` (coarse, single-person) + `VNDetectHumanHandPose` + `VNGeneratePersonSegmentation`; iOS18 holistic | YOLO11n-pose Core ML | — |
| 12 | Render | **OpenUSD (pxr) → USDZ** + **GLB** | RealityKit 4 (USDZ) / Three.js (GLB) | — | AR Quick Look |

Shot classifier status (2026-06-28): no trained SHOT-1 model is approved. The current transfer/external checks are not gate-verified: THETIS 60-clip family accuracy is 51.7% overall / 66.7% top-2, with serve and overhead at 0%; OpenSportsLab 100-event broad accuracy is 81.0%, but it is 81/81 on swing and 0/19 on serve. Treat serve, overhead, and exact pickleball shot labels as blocked classes until per-class eval passes.

**License posture (research/personal use now; flagged for future commercialization):** SAM-3D-Body (SAM License) and Multi-HMR 2 (Naver) are the only commercial-risk items — if commercializing, switch the 3D backbone to **SAT-HMR (Apache)**. Everything else is Apache/MIT or trained by us. TennisCourtDetector has no LICENSE file (use as init only; our fine-tuned weights are ours).

**Weights to mirror (fragile hosting):** PhysPT (Dropbox), BEATs (OneDrive), TrackNetV3 / TennisCourtDetector / SAT-HMR (Google Drive). Record every checkpoint in `models/MANIFEST.json` with source URL + sha256 + license.

---

## 3. Per-Layer Technology Choices

> The per-layer detail below explains WHAT/WHY/HOW-tested. The default/candidate variant names for each layer are centralized in the **§2.3 Model Registry** above, and approved runtime locks belong in `models/MANIFEST.json` after EVAL-0.

### (a) Ingest / decode — **NvDEC hardware decode**

- **Chosen:** NVIDIA NvDEC (via PyAV/torchaudio/DALI) for GPU-side H.264/HEVC decode; trim dead time pre-decode; compute a **capture-quality score** per clip (camera height/angle estimate, court coverage, resolution, blur).
- **Why:** On 20-min 1080p/4K clips the bottleneck is decode, not the models. NvDEC keeps frames on-GPU and frees CPU. 4K is downsized to model input (≈640px) right after decode, so model cost ≈ 1080p.
- **Toggle:** decode full clip (rich) ↔ decode only active-rally spans found by a coarse pass (cheap).
- **Differs from dance:** dance assumed short clips and CPU decode was acceptable; racket sports require GPU decode + dead-time skipping for ~20-min ingest.

### (b) Court calibration + net plane — **semantic court evidence + solvePnP, with manual fallback**

- **Chosen (MVP):** the server attempts semantic court evidence first, writes `court_line_evidence.json`, and fail-closes video-backed runs when NVZ/centerline/top-net evidence is not ready. Trusted manual/ARKit sidecars still seed `cv2.solvePnP` using known court dimensions (20×44 ft, 7 ft NVZ), gated by reprojection and semantic residual checks. Manual 4–6 tap remains the universal fallback until the no-tap solver passes CAL-3.
- **iOS seed (round-5, layer (u)):** the sidecar schema and Swift calibration package support ARKit intrinsics + 6DoF pose + a detected horizontal court-plane as a calibration seed, but the live ARKit setup pass is not yet wired or device-verified. Until that runtime path passes, manual/reviewed sidecar taps remain the only trusted calibration seed.
- **Auto-assist (Phase 2):** classical Hough + RANSAC + court-template fit (~6 ms/frame) pre-places the tap points; user corrects. Later, a fine-tuned court-keypoint net.
- **Rejected:** TennisCourtDetector (0.961 acc, 1.83 px median, but **license unstated** and tennis keypoint topology ≠ pickleball, no NVZ line); PnLCalib (CC BY 4.0 but 164–439 ms/frame and soccer-trained). No well-licensed pickleball court-keypoint model exists today — so we do not depend on one for MVP.
- **Variable camera handling:** because calibration is solved per-clip from the court itself, **any height/angle works** as long as enough court is visible; the reprojection gate + capture-quality score catch bad setups. Drift/bump handled by re-verifying the homography every N frames and warping via optical flow on static court lines; full re-detect on threshold breach.
- **Accuracy levers** (detail in `ACCURACY_AND_TRAINING.md` §3): prefer **`solvePnP` full 6-DOF pose over homography-only** (~2×, essential at shallow angles); **average correspondences over 20–40 static frames** (−23–57% error, free); sub-pixel line refinement + line-based (PnL) optimization; **ChArUco intrinsics cached per phone-model → EXIF → GeoCalib** fallback (don't self-calibrate distortion from court lines — degenerate); **train our own court-keypoint net** (fine-tune TennisCourtDetector + synthetic renders across 50–500 viewpoints). Feet-to-world error ranges from ~3–5 cm (high corner) to ~0.6 m far-court depth (low/shallow) → capture-quality score gates far-court NVZ calls.
- **Toggle:** one-time manual tap (cheap) ↔ auto-detect + continuous re-solve (rich).
- **License:** OpenCV (BSD), kornia (Apache). Clean.
- **Current repo modules:** `threed/racketsport/court_calibration.py`, `threed/racketsport/court_auto_evidence.py`, `threed/racketsport/calibration_overlay.py`, `threed/racketsport/court_zones.py`, and `threed/racketsport/net_plane.py`.

### (c) Person detection + tracking + doubles ID — **YOLO26m + BoT-SORT-ReID + court geometry** *(round-4b: YOLO26 verified real)*

- **Chosen (round-4b):** **YOLO26m** detector (Ultralytics, released Jan 2026; NMS-free end-to-end; **+1.4–2.8 mAP over YOLO11 at equal GPU latency**; AGPL-3.0 — fine, not selling yet) + **BoT-SORT-ReID** (its appearance features resolve identity through doubles crossings; one-line Ultralytics flag) + **HSV upper-body color histogram**. The **real ID engine is still geometry**: a court-polygon filter rejects spectators before the tracker sees them, and **ground-plane association** rejects "teleport" swaps when players cross at the net. **N-lock** guarantees ≤4 permanent IDs; a **one-tap coach anchor** labels the 4 players in frame 1. Doubles sides/roles assigned by court position + side priors.
- **Why YOLO26:** the user flagged it, and verification confirms it's real and genuinely better (NMS-free, more accurate at the same speed). Tune with **official training defaults** (the NMS-free/DFL-free changes need it). For 2–4 large players on a known court we are *tracking-limited, not detection-limited*, so even YOLO26n suffices — this is an upgrade, not the bottleneck. **Avoid YOLO12** (still research-grade).
- **Runners-up / fallbacks:** **ByteTrack** (MIT) as the simpler motion-only tracker if BoT-SORT is overkill; **RF-DETR-L** (Apache, +3.4 mAP) or **RTMDet** (Apache) as permissive detectors for a future commercial build. OSNet-x0.5 ReID only if outfits clash.
- **Speed:** 20 min @ 30 fps = 36,000 frames. YOLO26m ≈ 4.7 ms/frame (T4 TRT); on H100 with **batch 32–64 + rally-segmentation**, the fast-tier detect+track pass for a 20-min clip is **<1 min** `[H100 figure extrapolated]`.
- **Toggle:** nano detector + motion tracker (cheap) ↔ + ReID model (rich, only if needed).
- **Pose is OFF in this layer** — pose runs downstream; for ≤4 players geometry resolves IDs without it.
- **Differs from dance:** dance justified a heavy tracker; here a nano detector + free spatial priors replace it.
- **Current repo modules:** `threed/racketsport/person_mot.py`, `threed/racketsport/track_lock.py`, `threed/racketsport/doubles_id.py`, and `threed/racketsport/person_tracking_benchmark.py`.

### (d) 2D pose — **RTMPose / RTMW (top-down on ≤4 crops)**

- **Chosen:** **RTMPose-m/l** or **RTMW** whole-body (Apache-2.0, MMPose). Top-down on the ≤4 player crops. RTMPose-l ≈ 289 FPS (GTX 1660 Ti, TRT FP16); far higher on H100.
- **Why:** Apache license (commercial-clean), real-time, whole-body keypoints including hands (needed for paddle-face) and bilateral hips/shoulders (needed for X-factor).
- **Rejected:** ViTPose-H (79.1% AP but ~20 FPS A100 — too slow for the baseline); one-stage RTMO only pays off at large N.
- **Toggle:** RTMPose-m (cheap) ↔ RTMW-l whole-body 133-kpt (rich).
- **License:** Apache-2.0.

### (e) 3D body — **round-4b: Fast SAM-3D-Body per-frame mesh is the CORE; world-grounding is ours** *(corrects the GVHMR-primary call)*

Two tiers (our world-grounding/foot-lock detailed in (p); built in `IMPLEMENTATION_PHASES.md` Phases 3–4):

- **DEEP/REPLAY tier — core: Fast SAM-3D-Body per-frame mesh.** It is the **best per-frame mesh available** (3DPW PA-MPJPE **30.4 mm** for the Fast variant / 33.8 mm original; EMDB 38.2 mm — beats HMR2.0 by 15+ mm and NLF on 3DPW), **user-confirmed in practice**, native body+hands+feet (MHR rig). (License: SAM License — research/personal use now; ⚠️ verify before commercial; SAT-HMR/Apache fallback — see §2.3.) MHR→SMPL via its built-in MLP (the Fast variant made this ~10,000× cheaper). Then **we add world-grounding ourselves** (see (p)): project per-frame output to world via our known K,[R|t] + court plane Z=0 → temporal smoothing → foot-lock → physics. This drives the replay, the rigged avatar, and biomechanics. Refine with BioPose (anatomically valid joints).
- **Current contract reality:** `BodyStageRunner` now preserves both `joints_world` and optional `mesh_vertices_world` in `smpl_motion.json` when Fast SAM-3D-Body returns vertices. Scheduled-frame H100 BODY smokes have produced real BODY contract artifacts for Burlington, Wolverine, and Indoor; Outdoor remains unscheduled/missing. BODY is still not promoted until full coverage plus the real world-MPJPE gate pass.
- **FAST tier — preview (two flavors).** **(1) On-device, instant (layer (v)):** Apple **Vision** runs on the Neural Engine *on the phone* for the live capture overlay + guidance — `VNDetectHumanBodyPoseRequest` (2D, ≤4 players), `VNDetectHumanBodyPose3DRequest` (coarse single-person), hand pose, segmentation; **preview/guidance ONLY, not a coaching metric** (17 joints, no fingers, single-person, ~5–25° error). **(2) Server fast tier (<10 s after upload):** **Multi-HMR 2** (~20 FPS, no FOV assumption — preferred for our variable cameras) or **SAT-HMR** (~24 FPS, assumes ~60° FOV — verify) for a multi-person camera-space mesh on all ≤4 players in one pass (~7–8 mm worse than SAM-3D-Body, fine for a preview); **RTMPose-2D → MotionBERT** skeleton lift only for sub-frame contact triggering. Feet **hard-constrained to the solved court ground plane**.
- **GVHMR / WHAM — demoted to optional references only.** They are HMR2.0-backbone pipelines: their *per-frame mesh is worse* than SAM-3D-Body, and their one advantage (world trajectory + foot-contact) is exactly what we already get from our known camera + court plane. Use GVHMR/WHAM at most as an offline world-trajectory/foot-velocity sanity-check, **not** as the mesh source. **PromptHMR: dropped** (full-image hurts distant players, needs manual prompts, worst translation error — user got bad results).
- **Two non-negotiables (unchanged):**
  1. **Fine-tune on racket-sport motion.** Generic models hit ~214 mm MPJPE on athletic motion, dropping to ~65 mm after sport fine-tuning (AthletePose3D). Mandatory.
  2. **Joint set must include bilateral hips + shoulders** (X-factor; single mid-pelvis fails) **and hands** (the racket attaches at the grip; see (r)).
- **Why SAM-3D-Body over a world-HMR model:** real accuracy beats benchmark-paper extrapolation. SAM-3D-Body's per-frame mesh is the strongest published, the user confirms it, and we hold the world-geometry the world-HMR methods spend their model capacity estimating. Precedent for exactly this pattern (strong per-frame SAM-3D-Body + known camera → world coordinates): arXiv 2512.21573.
- **Accuracy levers** (detail in `ACCURACY_AND_TRAINING.md` §5): (1) **sports-domain fine-tune** (AthletePose3D + ASPset-510 + AthleticsPose + SportsPose / CalTennis) → **~70% in-domain MPJPE drop (measured 214→65 mm)** — mandatory; (2) **pseudo-label our own footage** (oracle filtered through our known camera, reprojection <8 px) and **distill into the fast-tier model**; (3) **court-geometry constraints** — reprojection loss, metric root depth, hard foot-on-ground (Z=0) constraint that kills foot-slide and nails kitchen-foot; (4) temporal smoothing (Butterworth 6 Hz body / 8–10 Hz wrist offline, One-Euro live). Expected MPJPE path: ~150–200 generic → ~30–50 (SAM-3D-Body + sport fine-tune) → tighter with grounding/temporal.
- **Toggle:** Multi-HMR 2 / SAT-HMR preview (cheapest) ↔ Fast SAM-3D-Body per-frame (core) ↔ + world-grounding + temporal + physics (richest, see (p)/(q)).
- **License:** SAM-3D-Body (SAM License — research-OK now; ⚠️ verify-commercial → **SAT-HMR (Apache)** is the license-safe fallback); Multi-HMR 2 (Naver proprietary); MotionBERT/RTMPose (Apache). See §2.3.
- **Differs from dance:** same strong per-frame backbone (SAM-3D-Body), but tiered + world-grounded + foot-locked + physics-refined instead of one uniform camera-space pass.

### (f) Mesh surface generation for rendering — **LBS from SAM-3D-Body MHR / SMPL params**

- **Chosen:** the body **surface** comes from the regression network in (e) — **Fast SAM-3D-Body** outputs the MHR mesh directly (and SMPL via its MLP). Driving it per frame is **cheap linear blend skinning** (>7,000 meshes/s, ~0.14 ms/mesh): the expensive part is the regression network, not the mesh, so the rigged GLB avatar is essentially free once we have params. Rendered as a `SkinnedMesh` GLB (see (s)).
- **Why:** SAM-3D-Body is the best per-frame mesh, so the avatar/render path uses the **same backbone as measurement** — no separate model.
- **Toggle:** no mesh (Instant preview uses camera-space mesh/skeleton) ↔ full world-grounded mesh replay (Deep tier).
- **License:** SAM License — research/personal use now; ⚠️ verify before commercial; **SAT-HMR (Apache)** is the license-safe fallback (see §2.3).

### (g) Twist / pronation / paddle-face — **hand-keypoints / racket-pose / optional IMU, contact window only**

- **Chosen:** estimate axial twist (forearm pronation, paddle-face angle) from **hand-landmark geometry** (already in whole-body keypoints), optionally a **racket 6DoF object-pose** detector or a **forearm IMU**, run **only in the contact window**.
- **Current 2026 paddle stack:** prefer **SAM 3** concept detection/segmentation/tracking when checkpoints and license are approved; fall back to **DINO-X/Grounded-SAM-2** for open-vocabulary detection plus video masks. For 6DoF, use **FoundationPose** when CAD/reference-image onboarding is available, with **GigaPose/FoundPose** as RGB/CAD alternatives, then keep **PnP-IPPE + reprojection/ambiguity + UKF + rebound** as fail-closed geometry gates. See `docs/racketsport/archive/paddle_pose_research_2026_06_28.md`.
- **Why:** axial roll is **unobservable from two joint endpoints** (20–45° error keypoint-only) — and a mesh does **not** fix this either. The fix is an explicit twist signal: a purpose-trained hand+IMU model reached ~4.65–5.61° MAE.
- **Decision:** **confidence-gate or omit** paddle-face/pronation when no extra signal is available. We do not claim a pronation number we can't defend. This is a trust win, not a gap.
- **Toggle:** omit (default) ↔ hand-keypoint estimate ↔ + racket-pose/IMU (richest).
- **License:** MediaPipe hands (Apache); racket-pose detector trained in-house.

### (h) Ball tracking — **TrackNetV3 → V5/BlurBall**

- **Chosen (MVP):** **TrackNetV3** (MIT; heatmap, 8-frame temporal input, built-in trajectory rectification for occlusion; F1 0.986, ~25 FPS) **fine-tuned from TrackNetV2 weights** on ~2,000–3,000 hand-labeled pickleball frames + converted Roboflow data (~10k total). Manual **tap-track** is the Phase-0 fallback and the cheapest pricing tier.
- **Upgrade path:** **TrackNetV5** (F1 0.986 badminton / 0.988 tennis, 38 FPS T4; repo not yet released) or **BlurBall** (F1 97.17 TT; emits blur length/angle = a free velocity proxy); **TOTNet** for heavy occlusion offline.
- **Rejected:** YOLO/RT-DETR single-frame detectors — a blurred ball reads as background; heatmap+temporal beats detect+track for this object. SAHI slicing — only helps for <10 px broadcast balls, not single-court cams.
- **3D ball:** context only (not line calls) via physics + court homography + "Z=0 at bounce" (TT3D-style); single-cam ball 3D is meter-level (Acc@1m ≈ 66%). **Line-call precision requires a 2nd camera — out of scope.**
- **Accuracy levers** (detail in `ACCURACY_AND_TRAINING.md` §6): **transfer hierarchy** badminton weights → tennis+TT fine-tune (~155k frames) → pickleball (~50k Roboflow images already exist, convert bbox→(x,y)); **BlurBall midpoint labeling** (+1.2% F1); **TOTNet visibility-weighted loss + occlusion augmentation** (occluded-frame acc 0.63→0.80); background-subtraction concat; augmentation tuned to our inputs (motion-blur synthesis, color jitter for ball colors, H.264 artifact injection, copy-paste); **physics post** (EKF+gravity → RANSAC parabola → sign-change bounce → homography net-crossing) cuts false positives <5%. Expected F1 ~0.90–0.95 after full pipeline (validate on a 2k-frame held-out set).
- **Toggle:** manual tap (cheapest) ↔ TrackNetV3 ↔ TrackNetV5/BlurBall + physics 3D (richest).
- **License:** MIT (TrackNet V2/V3).
- **New vs dance:** entirely new layer.

### (i) Audio "pop" + event fusion — **the cheap sensor competitors ignore**

- **Chosen:** fuse three signals for the **contact frame**: **audio "pop"** (XGBoost on MFCC/STFT; 98% CV / ~96% real-world, ~4 ms precision) + **wrist-velocity peak** + **ball-trajectory inflection** → contact timing **<1 frame (≤33 ms)**. Net crossing via homography; bounce via vertical-velocity sign flip. **Events drive the adaptive-compute triggers** (they define where mesh/twist bursts fire) and rally segmentation.
- **Caveat:** pickleball's polymer-ball acoustics differ from tennis and are unstudied — **the audio model must be retrained on our clips.**
- **Why:** audio is near-zero GPU cost and gives contact timing, shot counting, rally segmentation, and clean-vs-mishit ("pop" vs "thunk"). Almost no competitor exploits it.
- **Accuracy levers** (detail in `ACCURACY_AND_TRAINING.md` §7): **two-stage audio model** — an onset/peak detector for the *timestamp* (~0.09–4 ms) + a small CNN on mel-spectrogram for the *event type*; never let the classifier window drive timing. **Distance-delay correction is mandatory** (shift audio back `d/343 s`, up to ~30 ms at 10 m, via homography distance). Collect 2–5k pickleball "pops" at **44.1 kHz WAV** (never AAC — energy to 8–12 kHz); augment with noise (MUSAN/ESC-50/FSD50K/DEMAND) + RIR (pyroomacoustics) + SpecAugment. Doubles attribution via ball-to-wrist proximity + trajectory (single-mic DOA does not work). Auto-label via audio-onset↔ball-inflection agreement → Mean-Teacher semi-supervision.
- **Toggle:** trajectory heuristic only (cheap) ↔ + audio (cheap, always on) ↔ + learned event head (rich).
- **License:** in-house model; librosa (ISC) / torchaudio (BSD).

### (j) Shot classification — **PoseConv3D first, BST after fusion tensors are clean**

- **Chosen:** train **PoseConv3D/PoseC3D** first because it is built for pose-window action recognition and should handle noisy transferred pose better than the current single-frame heuristic. Compare **BST-style** pose+ball/player cross-attention after ball trajectory, court zones, and player tensors are trustworthy. **ML produces the shot LABEL only** — never a coaching fact.
- **Reality:** no published pickleball classifier matches our taxonomy. Current code supports `serve`, `fh_shot`, `bh_shot`, `fh_drive`, `bh_drive`, `dink`, `lob`, `overhead`, `third_shot_drop`, `reset_block`, but the baseline is not trained and fails serve/overhead in external checks. We must collect and label `data/pb_shots/`.
- **Toggle:** abstract side fallback (`fh_shot`/`bh_shot`) for review coverage ↔ PoseConv3D family classifier ↔ BST-style multimodal exact classifier.
- **License:** model in-house; backbones Apache/MIT.

### (k) Biomechanics metrics computation — **deterministic from 3D skeleton + court + events**

- **Chosen:** compute coaching metrics directly from the skeleton, court frame, and event timestamps. Defensible set: foot/NVZ position, court zones, spacing, balance (segmental mass-weighted CoM, ~20–30 mm error — sufficient; mesh CoM not materially better), base of support, reach, recovery time, knee/elbow bend, **shoulder-hip separation (X-factor, <1° MAE with bilateral keypoints)**, contact point/height, swing path, split-step timing, weight transfer, trunk tilt.
- **Confidence-gated/deferred:** forearm pronation/paddle-face (needs twist signal), and **velocity/acceleration** metrics (see §4 — conflicting evidence, validate empirically before trusting).
- **Toggle:** core positional metrics (cheap) ↔ + event-synced contact-window metrics ↔ + twist metrics (rich).
- **Current repo modules:** `threed/racketsport/movement_metrics.py`, `threed/racketsport/biomech.py`, `threed/racketsport/contact_windows.py`, `threed/racketsport/virtual_world.py`, and `threed/racketsport/physics_world_refinement.py`.

### (l) Insight engine — **rules = truth; ML = label; LLM = copy only**

- **Chosen:** a **rule-based threshold engine is the source of truth** (BioCoach-style ±5–10° violation bands → explainable flags). ML supplies only the shot label. **Confidence gating** suppresses/hedges any tip built on low-confidence or low-accuracy-DOF signals. An **LLM (latest Claude — Opus/Sonnet) phrases copy only**, fed structured biomechanical facts — it **never invents a fact** (CoachMe/BioCoach pattern: fine-tuned/structured copy beat raw GPT-4o by +31–58% on quality, +89.5% bio-accuracy vs a VLM baseline).
- **Feedback unit:** one **prescriptive external cue** + annotated overlay + plain-language "why + what to do" + a self-monitorable drill.
- **Toggle:** rules + template copy (cheap) ↔ + LLM copy ↔ + pro/self DTW reference comparison (rich).
- **License:** rules in-house; Claude via API.

### (m) Visualization / UX — **court map + 1 metric + self-vs-self is the conversion core; 3D is premium async**

- **Chosen conversion core (delivered <10s):** top-down **court map / heatmap** + **one outcome-linked priority metric** (+3–4 context) + **self-vs-self before/after**. This is what mid-level players pay to retain.
- **Premium async (notified):** 2D pose overlay on own video, **auto-telestration** (auto-drawn angle arcs / contact markers / base-of-support box), physics-accurate 3D mesh replay, LLM coaching summary, self-vs-self 3D ghost aligned at contact.
- **Why this split:** the UX research is unambiguous — court map/heatmap/before-after are the "aha"+retention drivers; full 3D replay and kinematic charts are premium proof layers, not the first conversion surface. Nielsen limits: <1s in-flow, 1–10s spinner, >10s disengage. PB Vision's silent multi-minute wait is the friction gap we attack with progressive disclosure (0–10s preview / 10–60s full metrics / 2–10 min report).
- **Toggle:** metric cards + 2D overlay (cheap, free tier) ↔ physics-accurate 3D mesh replay + ghost (premium).

### (n) Storage / artifacts — **tiered retention**

- **Chosen:** keep sparse fast-tier artifacts + habit clips + downsampled previews by default; archive full mesh/skeleton only for selected key spans; chunked object layout; re-process from source on demand. Artifacts: `court_calibration.json`, `court_zones.json`, `net_plane.json`, `tracks.json`, `smpl_motion.json` (`joints_world` plus optional `mesh_vertices_world`), `ball_track.json`, `contact_windows.json`, `racket_pose.json`, `frame_compute_plan.json`, `virtual_world.json`, `racket_sport_metrics.json`, `habit_report.json`, `coach_report.json`, `corrections.json`.
- **Why:** 20-min clips at 60 FPS make full-fidelity retention for every frame wasteful; we keep what's shown and re-derive the rest.
- **Toggle:** previews only (cheap) ↔ full per-span mesh archive (rich).
- **Reuse:** current repo readiness plumbing is `threed/racketsport/pipeline_contracts.py`, `scripts/racketsport/validate_pipeline_artifacts.py`, and `threed/racketsport/replay_readiness.py`. S3 archive/backend callback production gates are not implemented yet.

### (o) Drill verification — **wrist-velocity peaks + contact events**

- **Chosen:** count reps via wrist-velocity peak detection + confirmed contact event (peak + contact = 1 rep); a state machine (ready→windup→contact→follow-through) with filtering; per-rep quality gate from the metric engine. Accuracy MAE ~1.1–1.4 reps; within ±2 reps 91–95%.
- **Toggle:** rep count only (cheap) ↔ + per-rep quality scoring (rich).

### (p) World-grounding + foot-skate elimination — **SAM-3D-Body per-frame → our court-plane grounding → contact-vs-Z=0 + CCD-IK** *(round-4b)*

- **Chosen:** take the **Fast SAM-3D-Body** per-frame mesh (e) and **ground it ourselves** — place the root in world via our known K,[R|t] + court plane Z=0 (no SLAM, no estimated gravity; precedent arXiv 2512.21573), temporally smooth, then the **foot-skate killer**: detect contact against the **known court plane** (foot height-above-Z=0 < ~2–3 cm AND world-speed < ~1 cm/frame AND pose-confidence, with hysteresis) → **snap stance toe/heel to Z=0 + zero world velocity → CCD-IK** so the leg meets the locked foot while the upper body keeps its motion. UnderPressure learned contact classifier if thresholds prove noisy in fast lunges. **GVHMR/WHAM are an optional cross-check on the world trajectory only — not the mesh.**
- **Why we beat the papers:** SAM-3D-Body gives a better per-frame mesh than the world-HMR methods, and our plane is *exact*, so snapping to it gives **~2–3 mm foot-slide or below** (vs 3.0–4.4 mm for methods that estimate the ground) and zero floor penetration. Build/test in `IMPLEMENTATION_PHASES.md` Phases 3–4.
- **Toggle:** foot-lock + IK (always, cheap, kills visible skating) ↔ + physics refinement (q).

### (q) Physics refinement — **PhysPT (default) / PHC+PULSE on MuJoCo+MJX (flagship)** *(round-4, new)*

- **Chosen default:** **PhysPT** (CVPR 2024, MIT) — physics-aware transformer post-processor, **no engine at inference**; −68.7% foot-slide, −83.8% accel error, penetration handling, emits GRF/torques. ~80–90% of the visual benefit at low engineering cost.
- **Flagship:** imitation in **MuJoCo + MJX** (Apache-2.0) with a **PHC / PULSE** controller (BSD-3) tracking the foot-locked motion → physically valid by construction; start from **SMPLOlympics** tennis/TT environments. Add **MultiPhys** for inter-player non-penetration in doubles.
- **Why:** foot-lock (p) removes *visible* skating; physics adds *physical correctness* — balance, momentum, contact forces, no clipping. Build/test in `IMPLEMENTATION_PHASES.md` Phase 4.
- **Toggle:** none ↔ PhysPT ↔ full MuJoCo sim (+ MultiPhys for doubles).
- **Engine:** MuJoCo + MJX (Apache-2.0); MuJoCo-Warp/Genesis only if batch-processing many clips.

### (r) Racket / paddle 6DoF — **detect → keypoints → PnP-IPPE → UKF → physics-validate** *(round-4, new — flips paddle-face from gated to claimable)*

- **Chosen:** detect/track the paddle with **SAM 3** concept prompts when runtime/checkpoints are approved; fall back to **DINO-X/Grounded-SAM-2** for open-vocabulary boxes/masks. Derive true face corners/keypoints/masks, score/refine pose with **FoundationPose/GigaPose/FoundPose** when CAD/reference inputs exist, add a **hand-grip prior** from wrist+middle-finger MCP, then promote only through **PnP-IPPE on the known paddle dimensions** → **UKF on SE(3)** (blur streak as a rotation cue; predict through grip occlusion) → **physics-validate** against the observed ball rebound → **contact-point-on-face (±1–3 cm) + face-normal (~3–5°)**.
- **Current contract reality:** the local code has deterministic PnP-IPPE, reprojection diagnostics, ambiguity reporting, SE(3) smoothing, rebound checks, camera-space-to-`court_Z0` pose conversion, and a fail-closed `RacketStageRunner` that writes `racket_pose.json` only from schema-validated explicit four-corner paddle candidates. It still lacks the trained detector/keypoint/mask/CAD runner, so racket remains scaffold until ArUco/GT gates pass.
- **2026-06-28 research refresh:** the canonical acceptance path remains true paddle-face corners plus fail-closed IPPE because planar paddles can produce two visually similar pose hypotheses. For ordinary single-RGB uploads, GigaPose/FoundPose are the first CAD/RGB coarse-pose scorers to probe; FoundationPose remains a strong candidate for Tier-A depth/ARKit/pseudo-depth-capable paths or if its RGB/runtime path proves practical on our H100 setup. SAM2/Grounded-SAM2 should be treated as mask/video-tracking candidate generators, not as pose solvers.
- **Why this changes the target envelope:** the paddle is large, rigid, planar, and dimensioned, so its pose/face-normal should be recoverable **~5–15× better than wrist-derived pronation** (3–5° vs 20–35°). This makes paddle-face/contact-point claimable only after true paddle evidence and RKT gates pass; current box-derived preview candidates stay gated. Render the paddle mesh as a child of the wrist bone. Build/test in `IMPLEMENTATION_PHASES.md` Phase 6.
- **Toggle:** none ↔ wrist-proxy contact point (cheap) ↔ full racket 6DoF (rich, claimable face angle).
- **Training/license:** RTMPose fine-tuned on RacketVision (tennis) + Roboflow PB + ~50k synthetic (BlenderProc paddle CAD) + ArUco real GT; OpenCV PnP (BSD), FoundationPose/GigaPose/FoundPose (research/runtime terms must be checked before commercialization).

### (s) 3D replay rendering — **native RealityKit/USDZ in-app + Three.js/GLB on web, from one server bake** *(round-4, +round-5 native path)*

- **Author once, on the server:** the deep tier bakes the physics-resolved motion into **one animated scene** and exports **both formats** directly from the reconstruction (do **not** round-trip one into the other): **USDZ** via OpenUSD `pxr` (`UsdSkelSkeleton` + per-frame `UsdSkelAnimation`, packaged with `UsdUtils.CreateNewARKitLayer`) and **GLB** via `pygltflib` + `smplx` (joints → glTF `skin`, per-frame quaternions as animation channels). **True skeletal animation, not per-frame morph targets** (morph-per-frame bloats at 30 fps). SMPL-X pose-corrective blend shapes are baked server-side so the client just plays.
- **Native in-app (round-5): RealityKit + USDZ.** Use **`RealityView` virtual-camera mode (iOS 18+)** for a pure non-AR free-viewpoint replay (orbit/pan/zoom via `DragGesture`/`MagnifyGesture`); `Entity(named:)` + `playAnimation`. **SceneKit is soft-deprecated (WWDC25) — do not use it.** Gotchas to design around: bake character+animation in a **single** USDZ (separate-clip binding fails); **AR Quick Look supports one skeleton per scene** → use the RealityView path for doubles (multiple entities) and reserve AR Quick Look for single-player. A rigged 6,890-vtx SMPL ×2 players + court is well within iPhone 14+ (~20–80 MB / 10 s).
- **Web share: Three.js (r171+) + React Three Fiber**, WebGPU with WebGL2 fallback; **GLB** `SkinnedMesh`; procedural court+net mesh; everything in the calibrated world frame (court Z=0) so **free-viewpoint + occlusion are free** (Z-buffer + OrbitControls). **Physics baked → glTF animation tracks** (deterministic, scrubbable); **Rapier.js** client-side only for an optional "what-if" mode. Per-point GLB **~8–12 MB / 10-s rally** (MeshOpt+Draco+KTX2), CDN-streamed → instant share links.
- **Rejected for v1:** Gaussian splatting (single-person research-grade, floaters off-camera, loses crisp court lines) — deferred to v2 as a skinned-Gaussian avatar upgrade; Unity/Unreal WebGL (8–30 MB / 8–30 s cold load). Build/test in `IMPLEMENTATION_PHASES.md` Phase 10.
- **Toggle:** metric cards + 2D overlay ↔ 3D mesh replay ↔ physics replay + free-viewpoint (premium); native USDZ (in-app) / GLB (web share).
- **License:** RealityKit/OpenUSD/USDZ (Apple SDK / Apache-modified — free for iOS apps), Three.js/R3F (MIT), Rapier.js (Apache/MIT), pygltflib (BSD/MIT), smplx (research).

### (t) iOS capture — **AVFoundation, locked, high-fps** *(round-5, client)*

- **Chosen target:** `AVCaptureSession` + rear wide camera; before recording, `lockForConfiguration()` then **`setExposureModeCustom(duration:1/1000…1/500, iso:clamped)` + `setFocusModeLocked` + `setWhiteBalanceModeLocked`**; select the high-fps `activeFormat` (`activeVideoMin/MaxFrameDuration`); **landscape via `AVCaptureConnection.videoRotationAngle`** (iOS 17+); record **HEVC** (`AVAssetWriter`; **ProRes 422 LT** on Pro) while `AVCaptureVideoDataOutput` feeds the on-device fast tier (v). Current repo has partial capture runtime and Swift tests, but physical-device verification and hard capture-quality gates remain pending. iOS 26 `AVCaptureEventInteraction` / `.onCameraCaptureEvent` lets the user start/stop from the Action/Volume button **without touching the tripod**.
- **What it does / why:** delivers the cleanest possible frames — locking the auto-systems stops brightness/focus pumping mid-rally (the biggest free accuracy win); high fps + fast shutter freeze the ball; ProRes gives the server near-lossless frames on Pro.
- **How tested before promotion:** record a clip; assert exposure/ISO/WB constant across frames (no drift), fps matches the requested format, orientation is landscape or blocked with a capture-quality warning, and the file decodes server-side. Current tests do not yet include this physical-device gate. Skip iOS 26 cinematic mode (30 fps cap — incompatible with high-fps).
- **License:** Apple SDK (free for iOS apps).

### (u) iOS calibration — **ARKit (camera pose + intrinsics + court-plane); LiDAR bonus on Pro** *(round-5, client)*

- **Chosen target:** a short **`ARWorldTrackingConfiguration`** setup pass (`planeDetection = .horizontal`) captures **camera intrinsics + 6DoF pose + a horizontal court-floor plane**, written as **sidecar metadata** that seeds the server's court calibration (b). Current repo status is schema/package scaffold only; no live ARKit setup pass has been device-verified. **You cannot run a full ARSession and a high-fps `AVCaptureSession` at peak rates simultaneously** — do the ARKit pass during setup, then switch to the AVFoundation recording session and ship the calibration in the sidecar. **Manual court-corner tap is the universal fallback.** On **Tier A (LiDAR)** also attach `sceneDepth` frames within ~5 m (near-player foot-contact / near court-plane only).
- **What it does / why:** gives metric calibration **for free** (no checkerboard), bootstrapping the homography + scale; the detected plane is a strong seed even on low-contrast courts where it must be refined server-side.
- **How tested before promotion:** verify the sidecar contains plausible intrinsics + a horizontal plane at the court; reproject court template through ARKit pose vs solvePnP refinement and confirm the gate passes; confirm graceful fallback to manual taps when `trackingState` is `.limited`. This remains pending for the live app.
- **License:** Apple SDK (free).

### (v) On-device fast tier — **Apple Vision + YOLO Core ML on the Neural Engine** *(round-5, client; preview/guidance ONLY)*

- **Chosen target:** Apple **Vision** on the ANE — `VNDetectHumanBodyPoseRequest` (2D, ≤4 players), `VNDetectHumanBodyPose3DRequest` (coarse single-person 3D), `VNDetectHumanHandPoseRequest` (grip), `VNGeneratePersonSegmentationRequest` (framing) — plus a small **YOLO (Core ML, INT8) for ball/racket** (one-line `model.export(format='coreml')`; ~60–85 FPS on A17 ANE). Current repo has FastTier data structures/decoders and local CoreML packages, but app installation/copy/bundling and device latency checks are not yet verified. This tier should drive the **live capture overlay + capture-quality guidance** and produce an on-device pose track uploaded as a **server prior** after those gates pass.
- **Honest limit / why server stays truth:** Apple's 3D pose is **17 joints, no fingers, single-person, needs LiDAR for metric scale, ~5–25° error** — excellent for *preview + guidance*, **not** a coaching metric. The server **Fast SAM-3D-Body** mesh (e) is the source of truth.
- **How tested:** on-device pose runs **≥30 FPS** on a target device; guidance flags fire correctly on staged bad setups (portrait, dark, shaky, corners cropped); the uploaded pose prior measurably speeds/stabilizes the server fit. Keep all simultaneously-loaded on-device models **<~500 MB** (ANE budget); use `coremltools.optimize` W8A8 + `EnumeratedShapes` to stay on the ANE.
- **License:** Apple Vision SDK (free); Core ML; coremltools (BSD-3); YOLO weights as in (c).

---

## 4. Defensible Accuracy Envelope

We make only claims the markerless-3D literature supports, and we gate the rest.

| Signal | Status | Evidence / error |
|---|---|---|
| Foot/ankle court position, zones, spacing | **CLAIM** | cm-level after homography; feet hard-constrained to court plane |
| Sagittal/large-joint angles (knee, elbow flexion, trunk tilt) | **CLAIM** | markerless 3–10° vs mocap; a human coach's eye is only ~12° |
| Shoulder-hip separation (X-factor) | **CLAIM** | <1° MAE with **bilateral** hip+shoulder keypoints |
| Center of mass / balance / weight transfer (relative) | **CLAIM** | segmental CoM ~20–30 mm; below the 50–100 mm shifts coaching flags |
| Contact point/height relative to body | **CLAIM** | positional, reliable |
| Contact/bounce/net **timing** | **CLAIM** | <1 frame with audio fusion |
| Velocity — wrist/elbow swing speed (ball-speed predictor), horizontal CoM velocity, split-step timing, tempo | **CLAIM (Tier 1, after validation)** | ±15–25% relative for swing speed; split-step timing ±17 ms@60 / ±8 ms@120. Compute via court-homography→metric in-plane speed (the free lever). Validate via Protocol A/B (`ACCURACY_AND_TRAINING.md` §10). |
| Velocity — swing-speed *index*, 3D angular velocity | **CLAIM (Tier 2, relative/estimated only)** | present as "faster/slower than baseline," not absolute m/s |
| Velocity — 3D angular **acceleration**, depth-axis velocity, contact-frame peak from pose | **NEVER (Tier 3)** | use the **ball** for contact-frame speed; depth-axis derivative is catastrophically noisy |
| *Velocity conflict, resolved* | — | The R²≈0.96 result was a model predicting **ball speed from 2D in-plane endpoint speed** (raw wrist↔ball only r≈0.50–0.70); the r≈0.11–0.28 result was the derivative of **noisy lifted-3D positions**. Different quantities — not a contradiction. |
| Transverse/axial joint rotation (from pose) | **GATE/OMIT** | markerless error 3–57° from pose alone |
| **Paddle-face angle + contact-point-on-face — *with racket 6DoF tracking (r)*** | **CLAIM (round-4 flip)** | **~3–5° face angle, ±1–3 cm contact point** via PnP on the known paddle (vs 20–35° wrist-derived). Without racket tracking, still gate/omit. |
| Foot-slide / floor penetration in the 3D replay | **CLAIM** | **≤3 mm foot-slide, 0 penetration** via contact-vs-Z=0 lock + CCD-IK (p) |
| Ball 3D position (single camera) | **CONTEXT ONLY** | meter-level (Acc@1m ≈ 66%); not line-call grade |
| Line calls (in/out) | **OUT OF SCOPE** | requires 2nd camera + sub-pixel calibration |

Low or shallow camera angles compress depth and worsen foot-contact and ball-depth accuracy → the **capture-quality score** warns the user and **confidence-gates** affected metrics.

### Accuracy comes from training, not a bigger inference model

We do not buy accuracy by running a heavier model at inference — that only costs speed. We buy it at **training time** and ship a fast student. Two theses (full plan in `ACCURACY_AND_TRAINING.md`):

- **Heavy teacher → distill to fast student.** Slow/heavy/physics/audio models (SMPLer-X/WHAM pose oracle, TrackNetV3/available ball candidates now, TrackNetV5 full-res when released, audio onset) auto-label our own footage; a confidence + physics filter keeps only trustworthy labels; active learning routes the hard frames to humans; we distill into the fast single-camera models that run in production. The product's inference path stays cheap; the accuracy compounds offline.
- **Multi-view-to-train-monocular (the moat) — training-time only.** Use **2 phones 90° apart at training/labeling time only** to manufacture 3D ground truth + a cross-view consistency loss ("Two Views Are Better Than One": −43.6% MPJPE on SportsPose), then **ship a single-camera model** (zero user friction). The v1 *product* is single-camera; a live 2nd-camera capture mode for true 3D triangulation is a **future** power feature, not part of v1.
- **The corrections flywheel.** Every in-app coach/user correction becomes a training label → fewer future corrections. One-shot competitors can't compound this.

---

## 5. Licensing Summary

> **Round-4 reframe:** licensing is **no longer a build gate** (not selling yet). This table is **informational metadata for a future commercial build** — the "Commercial OK?" column tells you what would need swapping *if* we commercialize, with the permissive fallback named. Today we run the most accurate option per stage.

**Note on the round-4b body decision:** the **core body backbone (SAM-3D-Body) is usable now** (SAM License — research/personal use); for future commercial use, verify the SAM License or fall back to **SAT-HMR (Apache)**. Licensing also matters for **datasets** (AMASS/BEDLAM2/AthletePose3D/RICH/CalTennis — research/NC), **YOLO26** (AGPL), and the optional GVHMR/WHAM trajectory references. Other round-4 components (research/personal use — fine now): **PhysPT** (MIT), **PHC/PULSE** (BSD-3), **MuJoCo/MJX** (Apache), **MultiPhys** (CC BY-NC-SA), **Three.js/R3F** (MIT), **Rapier.js** (Apache), **FoundationPose/GigaPose/FoundPose** (research/runtime terms need review), **OpenCV PnP** (BSD).

| Component | License | Commercial OK? | Note |
|---|---|---|---|
| **SAM-3D-Body / Fast SAM-3D-Body** | SAM License | **VERIFY** | Usable for current research/personal work. Before commercialization, verify the SAM License terms or switch the body backbone to SAT-HMR (Apache). |
| Multi-HMR 2 / SAT-HMR (fast-tier preview) | Research | **NO** | Preview only; commercial-swap → SAM-3D-Body per-crop. |
| GVHMR / WHAM / TRAM (optional trajectory x-check) | Research / MIT | NO / YES | Demoted to optional world-trajectory sanity-check, not the mesh. |
| PromptHMR | Research | — | **Dropped** (bad on automated sports; user-confirmed). |
| SMPL / SMPL-X (avatar/anim params) | Model license | **NO** | From SAM-3D-Body MHR→SMPL; commercial build keeps MHR. |
| **Ultralytics YOLO26** | **AGPL-3.0** | **CAVEAT** | Primary detector now; copyleft → swap to RF-DETR-L/RTMDet (Apache) for commercial. |
| BoT-SORT / ByteTrack / OC-SORT | MIT | **YES** | Trackers (BoT-SORT-ReID primary). |
| RTMPose / RTMW / RTMW3D (MMPose) | Apache-2.0 | **YES** | 2D pose + skeleton-lift. |
| MotionBERT | Apache-2.0 | **YES** | 2D→3D lift (preview triggering). |
| TrackNet V2 / V3 (→V5 on release) | MIT | **YES** | Ball tracking. |
| WASB / BlurBall / TOTNet | MIT / open | **YES** | Ball upgrades. |
| PhysPT / PHC / PULSE / MuJoCo+MJX / MultiPhys | MIT / BSD-3 / Apache / CC-BY-NC-SA | mostly **YES** | Physics (MultiPhys NC → use only doubles, or swap). |
| FoundationPose / GigaPose / FoundPose / OpenCV PnP-IPPE | research/runtime-specific / BSD | PnP **YES** | Racket 6DoF. |
| Three.js / React Three Fiber / Rapier.js | MIT / Apache | **YES** | 3D replay renderer (web). |
| **Apple Vision / Core ML / ARKit / RealityKit / AVFoundation** | Apple SDK | **YES** | Free for iOS apps; on-device capture/calibration/fast-tier/native replay (t)(u)(v)(s). |
| coremltools / OpenUSD (USDZ) / pygltflib | BSD-3 / Apache-mod / BSD-MIT | **YES** | Model conversion + render-bake to USDZ/GLB. |
| TennisCourtDetector | **Unstated** | **AVOID** | License risk; train our own court net. |
| OpenCV / kornia / librosa / torchaudio | BSD / Apache / ISC / BSD | **YES** | Utilities. |
| Claude (LLM copy) | Anthropic API | **YES** | Commercial API. |

**Rule (round-4b):** use the **most accurate** component per stage now (research/personal use). For a future paid build, verify SAM-3D-Body's commercial terms and plan likely swaps for YOLO26→RF-DETR-L/RTMDet, NC datasets→licensed/synthetic, MultiPhys→a permissive equivalent, and SAM-3D-Body→SAT-HMR if the SAM License is not acceptable.

**Verify-before-commit (bleeding-edge, some possibly future-dated in source research):** Multi-HMR 2, SAT-HMR, WATCH, OnlineHMR, HTD-Refine, PhysHMR, CalTennis, BEDLAM2, TT4D — confirm repo/license/benchmarks resolve before relying on them. **Proven core (build on now):** SAM-3D-Body + Fast SAM-3D-Body, YOLO26, BoT-SORT/ByteTrack, RTMPose/RTMW, MotionBERT, TrackNetV3, PhysPT, PHC/PULSE, MuJoCo+MJX, FoundationPose/GigaPose/FoundPose, OpenCV PnP, Three.js. Established fallbacks: SAM-3D-Body per-crop (for Multi-HMR 2/SAT-HMR preview), PhysPT (for PHC), RF-DETR-L/RTMDet (for YOLO26).

---

## 6. Hardware & Performance Budget

**Hardware:** one H100-class GPU (matches the single-active-run lock already in `pod_agent`).

| Stage | Throughput (target) | Notes |
|---|---|---|
| NvDEC decode | real-time+ | GPU-side; bottleneck for raw frames, not models |
| Fast-tier detect + track (20-min clip) | **<1 min** | batch 32–64 + rally segmentation `[H100 extrapolated]` |
| Fast-tier preview mesh (Multi-HMR 2 / SAT-HMR) | ~20–24 FPS | all ≤4 players in one pass; ~7–8 mm worse than SAM-3D-Body (fine for preview) |
| **Deep-tier per-frame mesh (Fast SAM-3D-Body)** | **~15 FPS/person; ~4–6 FPS for 4 batched crops** | the accuracy core; run on rally spans, not dead time |
| Ball (TrackNetV3) | ~25 FPS native; faster batched | run on rally spans only |
| Audio event detection | real-time, ~CPU | near-zero GPU |
| LBS mesh generation (for avatar) | >7,000 meshes/s | ~0.14 ms/mesh — free |

**Deep-tier cost reality:** a 20-min clip at 30 fps × 4 players ≈ 144k crops; at ~4–6 FPS that's an overnight offline job (acceptable — the replay is async). Cut it by running SAM-3D-Body only on **rally spans** (skip dead time) and at the densest rate only on **contact micro-spans**, interpolating params (SLERP rotations, linear translation) between, with the fast-tier preview guiding in-between frames.

**Engineering levers:** Fast SAM-3D-Body itself (TensorRT + CUDA graphs, MLP replacing iterative MHR→SMPL fitting) is 7.9–10.9× over the original; FP16; batched ≤4 person crops; GPU-native preprocessing; rally-span + contact-span gating; temporal subsample + interpolate.

### H100 runtime — ranked speed levers

The pipeline's #1 risk is the **GPU starving while the CPU decodes video** — fix that before touching kernels. (How tested: profile with `nvidia-smi`/DCGM + per-stage timers; a lever "passes" if GPU utilization rises and wall-clock drops on a 20-min test clip.)

| # | Lever | Expected gain | How |
|---|---|---|---|
| 1 | **GPU decode: NVDEC via NVIDIA DALI** (replace CPU FFmpeg/OpenCV) | 37–72% E2E cut; frees all CPU cores | DALI `fn.decoders.video` in `mixed`/`gpu` mode → frames stay on-GPU, no H2D copy |
| 2 | **TensorRT 10.x engines** for frozen models (YOLO26, TrackNet, RTMW, audio CNN) | 2–5× (up to 6× Torch-TRT) | ONNX → `trtexec`; layer fusion + autotune + CUDA-graph capture; cache per fixed input shape |
| 3 | **INT8/FP8 quantization** (inference CV models) | ~2× over FP16, <1% mAP | TRT INT8 PTQ (~1000-frame calib). **Keep SAM-3D-Body at FP16/BF16** — INT8 causes visible mesh artifacts |
| 4 | **BF16 AMP** for all training/fine-tuning | ~2× vs FP32 | `torch.autocast(bfloat16)` — prefer BF16 over FP16 on H100 (wider range, no loss-scaling) |
| 5 | **torch.compile** (`mode="max-autotune"`) for fast-iterating code | 1.5–3× | one line; use instead of TRT for training loops / research variants (no 20-min rebuild) |
| 6 | **FP8 H100 Transformer Engine** — large transformer blocks only | 1.3–1.7× over BF16 | only worth it for big attention modules (≳1B params); skip the small CNNs |
| 7 | **CUDA-stream overlap** decode→infer→postproc | 30–50% wall-clock | 3-stage pipeline across CUDA streams; H100 copy engines hide H2D behind compute |
| 8 | **CUDA graphs** (fixed-shape inference) | 10–30% latency | capture per model at B=1/B=4; biggest for TrackNet (high frame rate) + audio CNN; free via TRT |
| 9 | **Batch the ≤4 player crops (B=4)** | removes 3 redundant launches/stage | YOLO crop-infer + RTMW pose; **SAM-3D-Body runs SEQUENTIAL per player** (too big to batch at B=4) |

**Top-4 = decode (1) + TRT (2) + stream-overlap (7) + crop-batching (9)** ≈ 80% of the win. **SAM-3D-Body mesh-per-player-frame is the pacing item — benchmark it in isolation first**; it sets pipeline throughput and the MIG geometry in `BUILD_CHECKLIST.md §1.5`.

### Serving stack — **Triton ensemble**

For one H100 with a fixed multi-model DAG, **NVIDIA Triton Inference Server** ("Dynamo Triton") is the right abstraction — native multi-model DAG, shared GPU memory, zero-copy tensor passing between stages, per-stage dynamic batching. **Skip Ray Serve / Modal / BentoML** (their distribution/serverless layer buys nothing on a single box; revisit Ray only if we go multi-node). If the pipeline is one tightly-controlled Python process you can orchestrate TRT engines + CUDA streams directly and add Triton when you want clean model versioning + concurrent eval-agent serving. **Training/fine-tune path:** torch.compile + BF16 AMP (+ TE-FP8 on large transformer blocks). **Sharing the one H100 across parallel build/test agents** (MIG geometry + `flock` lease/queue + VRAM budget) is specified in **`BUILD_CHECKLIST.md §1.5`** — not duplicated here.

---

## 7. The Toggle Map

| Pipeline layer | Cheap / fast setting | Rich / accurate setting | When we lean in |
|---|---|---|---|
| Capture (iOS) | 1080p60 HEVC, locked | 1080p120 (240≈720p ball-physics) / ProRes on Pro | 120 fps for racket/swing-speed; ProRes on Pro |
| Calibration source | manual court-corner tap | ARKit intrinsics+plane sidecar (+LiDAR depth Tier A) | ARKit when available; tap fallback |
| Compute location | on-device preview (Apple Vision) | server deep tier (SAM-3D-Body + physics) | preview on phone; truth on server |
| Frame sampling | uniform low FPS | adaptive dense at events | always adaptive |
| Decode coverage | rally spans only | full clip | long uploads → spans |
| GPU decode | CPU (FFmpeg/OpenCV) | **NVDEC + DALI (GPU)** | benchmark per real clip set; current 1080p H.264 H100 sample favored CPU decode |
| Runtime | PyTorch eager | **TensorRT INT8/FP16 + CUDA graphs** (SAM-3D-Body stays FP16/BF16) | frozen models in production |
| Stage execution | serial | **CUDA-stream pipelined** (decode→infer→postproc overlap) | always when GPU-bound |
| Result delivery | wait for full clip | **stream rally-by-rally (SSE) + APNs push** | always |
| Court calibration | one-time manual tap | auto-detect + continuous re-solve | bumped / varied cameras |
| Detect + track | YOLO26m + ByteTrack | + BoT-SORT-ReID / OSNet | doubles crossings / outfits clash |
| 2D pose | RTMPose-m | RTMW-l whole-body | hands/X-factor needed |
| **3D body** | **preview mesh (Multi-HMR 2 / SAT-HMR) + skeleton lift** | **Fast SAM-3D-Body per-frame → our court-plane grounding + temporal** | **fast = preview; deep/replay = SAM-3D-Body grounded (default)** |
| **Physics** | none | foot-lock (CCD-IK) → PhysPT → full MuJoCo sim | replay tier; MuJoCo for flagship + doubles |
| **Racket** | none / wrist-proxy | full 6DoF (PnP-IPPE + UKF) | when face-angle/contact-point wanted |
| Twist/paddle-face | omit (gated) | racket 6DoF target (~3–5° only after RKT gates) / hand-kpt | racket tracking on |
| Ball | manual tap / TrackNetV3 | TrackNetV5/BlurBall + physics 3D | timing/depth matters |
| Events | trajectory heuristic | + audio + learned head | audio always on (cheap) |
| Shot class | abstract FH/BH review fallback | PoseConv3D family classifier → BST-style multimodal exact classifier | premium reports |
| Insights | rules + template copy | + LLM copy + DTW reference | premium reports |
| Visualization / render | court map + metric + 2D overlay | 3D mesh replay → physics replay + free-viewpoint (native RealityKit/USDZ in-app + Three.js/GLB web) | premium / share moments |
| Cameras | **1 phone (v1 product)** | 2 phones (true 3D) — **FUTURE; training-instrument only for now** | future line-call tier |

The bolded rows are the round-4 architectural identity: a fast skeleton/camera-space preview, a world-grounded SMPL mesh core that is foot-locked and physics-refined for the replay, and the racket tracked as a 6DoF object — all on one phone at any camera height.

---

## 8. Why This Beats the Market

| Differentiator | How it works | Competitor weakness it attacks |
|---|---|---|
| **Adaptive compute budget** | cheap baseline + event-triggered heavy bursts | All run one fixed model uniformly → forced to choose slow-or-shallow |
| **Court-geometry as a free accuracy booster** | feet snapped to solved ground plane; impossible poses rejected | SportAI/Sportsbox/PB Vision lack per-clip calibrated geometry feeding pose |
| **Multi-signal fusion (pose+ball+court+audio)** | each cross-checks the others; joint physics solve | Competitors solve ball and body separately |
| **Audio as a cheap sensor** | contact timing 4 ms, mishit detection, rally segmentation, ~0 GPU | Almost no competitor exploits audio |
| **Fast-first <10s + premium async** | court map + 1 metric + skeleton in seconds; deep report later | **PB Vision's silent multi-minute wait** |
| **Body habits, not just stats** | the "why," tied to specific joints/footwork | **PB Vision / SwingVision = stats only**; zero body-mechanics requests on PB Vision's roadmap |
| **True 3D measurement, honestly gated** | inches/degrees where defensible; gate the rest | **SwingVantage admits "heuristic estimates"; SwingVision does no body mechanics** |
| **Confidence-as-trust ("no charge if we can't trust it")** | gray/omit low-confidence; never bluff | Churn driver #1 is wrong AI outputs (SwingVision scoring complaints) |
| **Doubles "team body" layer** | partner spacing/middle-gap/split-step sync in court coords | No incumbent models 2-body geometry |
| **Self-vs-self 3D ghost** | your best rep vs failed rep, aligned at contact | Pro-comparison demoralizes; self-comparison retains |
| **Personalized body calibration** | lock bone lengths/shape over sessions → faster + more accurate | One-shot competitors can't compound |
| **Learns from corrections** | coach/user corrections fine-tune per club | One-shot models don't improve with use |
| **Graceful 1→2 camera scaling** *(future)* | v1 is single-camera; later a 2nd phone → true 3D ball + line-call tier | Single-cam-locked or expensive fixed rigs |
| **Free ARKit calibration (native iOS)** | camera intrinsics + pose + court-plane from an ARKit setup pass — no checkerboard, no manual taps on most devices | Web/Android-first competitors hand-calibrate or skip metric geometry |
| **Instant on-device preview + capture guidance** | Apple Vision on the Neural Engine shows pose/court the moment you point the phone, and coaches the camera setup before recording | Upload-then-wait competitors give no real-time feedback |
| **Native in-app 3D replay (RealityKit/USDZ)** | free-viewpoint mesh replay plays natively on the phone, plus a Three.js/GLB web share link from the same bake | Sportsbox shows fixed angles; nobody ships a native free-viewpoint physics replay |
| **Movement screening for 50+** | non-diagnostic balance/asymmetry from 3D | Wide-open lane; peer-reviewed fall-prevention demand, no incumbent |
| **Physics-accurate, foot-skate-free 3D replay** | world-grounded SMPL + contact-vs-Z=0 lock + physics refinement (≤3 mm slide) | Competitors show 2D/skeleton overlays or sliding avatars; none deliver a watchable physics replay |
| **Racket tracked as a 6DoF object** | PnP on known paddle → face angle ~3–5°, contact-point ±1–3 cm | Everyone else uses the hand center as a contact proxy (e.g. LATTE-MV); nobody estimates racket pose |
| **Free-viewpoint replay on one phone** | calibrated world frame + Three.js mesh scene → rotate to any angle | Free-viewpoint elsewhere needs multi-camera rigs (Hawk-Eye/TRACAB) |
| **3D avatar that's a real SMPL-X mesh, generated for free** | LBS from already-computed SMPL-X params | Sportsbox's "avatar" is a skeleton rig — we render a true volumetric mesh at LBS cost |

**One line:** the fastest, most accurate, most plain-spoken coach in the player's pocket — court map + one priority metric + one prescriptive fix + self-vs-self proof in under 10 seconds, with mesh/3D as the async premium flourish, all on a single phone at any camera height.
