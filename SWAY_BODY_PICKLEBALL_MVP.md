# Sway Body — Pickleball & Tennis MVP

**Status:** Target product + technical spec with current implementation caveats
**Date:** 2026-06-26
**Repos in scope:** this `pickleball` repo is the live CV/client implementation; `sam4dbody` is an upstream reference/runtime source, `sway` remains the product integration surface, and `ios/` is the native Swift app.
**Platform target:** **native iOS Swift app** (capture, on-device preview/guidance, replay viewer) + **GPU server** (heavy 3D reconstruction). **Current repo status:** Swift package/app scaffolds plus partial AVFoundation preview/recording; ARKit/Vision/CoreML/RealityKit/upload and physical-device gates remain scaffold-level until `BUILD_CHECKLIST.md` promotes them. **Single static camera is the product focus; multi-camera is future; multi-view is a training-time-only technique that ships a single-camera model.**
**Companion docs:** `TECH_STACK.md` (every model + why), `IMPLEMENTATION_PHASES.md` (phase-by-phase build + test gates), `ACCURACY_AND_TRAINING.md` (datasets, training, validation), `BUILD_CHECKLIST.md` (the operational checklist + multi-agent coordination protocol Codex drives). Physics, foot-skate elimination, racket 6DoF, and the 3D replay renderer live in `TECH_STACK.md` §(p)–(s) and `IMPLEMENTATION_PHASES.md`.
**Thesis:** Be the fastest, most accurate, most plain-spoken movement coach in a racket player's pocket. Turn one court video into the 2–3 body habits costing points, the proof, one drill, and week-over-week improvement — in under 10 seconds for the first useful screen.

**Canonical tier split:** `CAPABILITIES.md` section "Canonical Tier Split" is the source of truth. ON-DEVICE LIVE / fast tier runs on the iPhone ANE during play; SERVER OFFLINE / deep tier is async GPU. Camera-space mesh preview is `server-fast`, not phone-real-time. LiDAR is a near-field (~5 m) bonus only.

---

## 0. Executive decision

Sway should not enter pickleball/tennis as another line-calling, scoring, or stat app — that market is saturated (SwingVision, PB Vision, DinkAI). The wedge is the question none of them answer: **not "what happened," but "why, and exactly what to change with your body."** PB Vision's public roadmap contains zero requests for body-mechanics/form/footwork analysis — only more stat correlations. That is the white space.

> **Round-4 update — licensing lifted; mesh is now core.** We are not selling the product yet, so license is no longer a build blocker (license info is kept as metadata for future commercialization, not a gate). This **supersedes the earlier "skeleton-first" decision**: SMPL/SMPL-X mesh is now the core body representation, we add a **physics-accurate, foot-skate-free 3D replay**, and we track the **racket as a first-class 6DoF object**. Full design in `TECH_STACK.md` §(p)–(s) and `IMPLEMENTATION_PHASES.md` Phases 4/6/10.

Six decisions define the build:

1. **Full-body mesh is the SERVER OFFLINE core target representation; the phone-live tier is 2D guidance.** A watchable, physics-accurate 3D replay needs a volumetric body (you can't drive a rigged avatar or run physics on a stick figure), and SMPL(-X) params are the animation/physics lingua franca. **The per-frame mesh backbone target is Fast SAM-3D-Body** — selected from research benchmarks and hands-on checks as the best current candidate (3DPW PA-MPJPE 30.4 mm, beats HMR2.0 by 15+ mm) and it keeps the existing `sam4dbody` investment. Current repo evidence is scheduled-frame BODY smoke, not BODY verification. (License: SAM License — fine for research/personal use now; verify before commercial, with **SAT-HMR/Apache** as the license-safe fallback.) We add **world-grounding ourselves** (project per-frame to world via the known camera + court plane → foot-lock → physics), rather than switching to a world-HMR model whose per-frame mesh is weaker. SAT-HMR/Multi-HMR 2 are `server-fast` camera-space preview candidates, not phone-real-time.

2. **A physics-accurate, foot-skate-free 3D replay is a core deliverable target.** We reconstruct the played game in one metric world frame — players (SMPL-X), court, net, ball, rackets — that obeys physics: **no foot sliding, no floor/player penetration, ball obeys gravity/bounce/Magnus.** Current code has CPU review scaffolds and static review GLBs only; the hard physics/replay gates still have to pass before this is a shipped claim.

3. **Track the racket as a 6DoF object.** A paddle is large, rigid, planar, and dimensioned, so its pose/face-normal should be recoverable to **~3–5°** via PnP on known geometry — **5–15× better than inferring face angle from the small, twist-ambiguous wrist (20–35°).** This makes paddle-face/contact-point a validated target only after true corner/CAD/reference evidence and RKT gates pass; current box-derived candidates stay preview-only.

4. **Adaptive compute budget / tiered pipeline is still the architectural identity.** ON-DEVICE LIVE / fast tier is court calibration seed/cache, person lock, 2D pose/joints, the risky ~288p CoreML ball heatmap tracker, cheap line/contact cues, court map, one priority cue, and capture guidance. SERVER OFFLINE / deep tier is mesh, world grounding, foot-lock/physics, paddle 6DoF, full biomech, 3D replay, LLM copy, and week-over-week. Every pipeline layer is an explicit accuracy/speed/UI toggle; spend heavy compute where it matters, where the user sees it, and where confidence is low.

5. **Accuracy honesty is a trust feature.** Gate/gray low-confidence metrics, show ranges not false precision, never charge for a result we can't trust. Note the update: **paddle-face becomes a gated target through true racket tracking** (wrist-only pronation stays gated). Foot-slide, penetration, and racket face-angle all require hard validation gates before product claims.

6. **Coaching value still leads the product.** The 3D replay is the proof layer and a differentiator, but the conversion core remains court map + one priority habit + self-vs-self + one drill, delivered fast (§4, §5.6).

**Capture constraint (core, not optional):** one static tripod camera (**single-camera is the product focus; multi-camera is future**), but **height and angle vary a lot**. Nothing geometric is hardcoded — we solve calibration per-clip from the court and world-ground the body off the *solved* plane (viewpoint-agnostic, and now also the anchor for physics + ball depth). The **target native iOS app** makes this a strength: AVFoundation locks exposure/focus/WB and runs a high shutter (1/500–1/1000 s) for sharp frames, and an **ARKit setup pass should supply camera intrinsics + 6DoF pose + the court-floor plane** (sidecar) to seed calibration once the device gate passes. Current repo reality is basic estimated sidecar writing plus calibration/manual-tap contracts; trusted ARKit/manual sidecar capture is not yet verified. Low/shallow angles compress depth → per-clip capture-quality score + confidence gating. Capture 60 fps (120 for swing-speed/racket); see §10. **LiDAR (Pro devices) is a near-camera (~5 m) bonus only** — it helps near-player foot-contact, never the far court or the ball; vision-first is the baseline.

**What this is NOT in v1:** auto scoring, DUPR/UTR replacement, medical diagnosis, consumer-only App Store launch before coach/club pilots, or server-deep inference pretending to be phone-real-time. Cheap on-device ball IN/OUT + net-crossing guidance is in the live tier, but final claims still gate on the documented BALL/court evidence. (6DoF paddle tracking and the 3D replay are now **in** scope as SERVER OFFLINE deep-tier targets.)

---

## 1. The company

**Sway Body is a 3D movement coach for racket sports** — body truth + court truth + coach workflow.

| World | Existing products | What's missing |
|---|---|---|
| Match analytics | SwingVision, PB Vision | why the body caused the miss |
| Coach video workflow | Onform, CoachNow, PlaySight | automatic body/court intelligence |
| 3D biomechanics | Sportsbox, SportAI | pickleball-native mechanics, doubles context, club workflow, the 50+ screening lane |

**One-liner:** Upload one clip. Get the two body habits costing you points, the exact clips, one drill, and proof you improved next week.

---

## 2. Market evidence

- **Pickleball:** SFIA reports 24.3M U.S. participants in 2025 (171.8% growth 2022–2025). Demand is fitness, competition, community, and staying on court — not only "rank me."
- **Tennis:** USTA's 2026 report (2025 data) says 27.3M U.S. players, 14.5M core (10+ times/year). The best tennis wedge is a high-value technical loop (serve lab, recovery footwork), not crowded match analytics.
- **Durability:** An AOAO 10-year study estimated 77,963 pickleball injuries 2013–2022; 91% in players 50+. A 2025 RCT on senior fall prevention found 25% (trained) vs 44% (control) fall rates and significant reduction in fear of falling; 42% of 50+ players report a pickleball-related fall. Generic screens (single-leg squat) do **not** predict pickleball falls — sport-specific movement analysis does. This is a wide-open premium lane (framed as movement screening, never diagnosis).

**Willingness to pay (verified anchors):** consumer sweet spot ~$10–15/mo (PB Vision $19.99/$49.99/mo; SwingVision ~$15/mo; tennis biomechanics apps Tennis AI 2.0 €9.99/mo, OnCourtAI £8.99/mo). One private lesson ($50–150/hr) is the value yardstick. Coach/club pricing runs 5–10× consumer (Sportsbox coach ~$79.99/mo; OnForm $20–60/mo with 3D gated to the top tier — confirming 3D is an upsell, not the core).

---

## 3. Competitor map

| Competitor | Position | Attack gap |
|---|---|---|
| **SwingVision** | AI scoring, stats, highlights, line calls | Does NOT do body mechanics/form/footwork (confirmed). Own the "why." |
| **PB Vision** | Strongest PB analytics; DUPR ratings from one match | Zero body-mechanics demand on roadmap. Own correction, mechanics, partner geometry, habits. |
| **DinkAI** | AI coach app; claims footwork/paddle-angle | Only ~5 ratings; unproven 2D. Be deeper and credibly 3D. |
| **Onform / CoachNow** | Coach video workflow, telestration | Add automated body/court intelligence to a familiar workflow. |
| **SportAI** ⚠️ | 3D **skeleton** (not mesh), near real-time, TIME Best Invention 2025, partnering with Shark for pickleball | Direct 3D threat. Beat on pickleball-native mechanics, doubles team-body, confidence honesty, and the 50+ lane — move fast. |
| **Sportsbox AI** ⚠️ | 3D skeleton-rig "avatar," golf-dominant, expanding | Direct 3D threat. Its avatar is a dressed-up skeleton; we match it via cheap LBS and win on racket-sport context and speed. |

**Key competitive truth:** the 3D avatar is a marketing/demo asset and a coach tool — it is *not* what converts amateurs to payers. SportAI and Sportsbox both prove skeleton-grade 3D is enough to win awards and customers.

---

## 4. What customers actually want (ranked, evidence-backed)

The dominant unmet need across Reddit, coaching sites, PB Vision's roadmap, and app reviews:

1. **"Tell me WHY I lose points / what I'm doing wrong"** — root cause, not stats. The central wedge. 3D answers it directly.
2. **"I'm stuck at my rating and don't know how to break through"** (3.5/4.0 plateaus). Subtle mechanics players can't feel.
3. **"Give me a plan/drills for MY weaknesses, not a dashboard."**
4. **"Fix my footwork / split-step / recovery."** Pure body-tracking output competitors don't touch.
5. **"Diagnose my contact point & paddle face."** 3D resolves the "heuristic estimate" weakness 2D tools (SwingVantage) openly admit to — and with **racket 6DoF tracking** we target true contact-point-on-face and face angle after RKT validation; the current repo keeps this gated (see §13).
6. **"Make my third-shot drop / reset / dink consistent."** PB-native; body cause (weight transfer, paddle angle) is ours.
7. **"Show me what I really look like."** The perception-gap "aha."
8. **"Am I actually getting better?"** Progress proof — uniquely answerable by tracking a mechanic in degrees/inches/ms over time.
9. **"Translate biomechanics into plain do-this-not-that language."** The arXiv "Talking Tennis" gap — LLM-for-copy on top of 3D.
10. **Coaches: save my review/telestration time, let me coach remotely and charge more.**
11. **Doubles: self-diagnose positioning, spacing, stacking, middle-ball, poaching.** Only possible if all players are tracked in court coordinates — our advantage.
12. **"Keep me on the court"** — balance, asymmetry, fall risk (50+). Wide open.

**What they will NOT durably pay for:** a 3D skeleton/avatar by itself, 20-metric menus, kinematic-sequence charts without interpretation, voiceovers. Metric overload is a documented stressor — the ceiling is ~1 priority metric + 3–4 context.

---

## 5. Technical strategy (the heart)

> **Round-4/5 note:** mesh is now core for SERVER OFFLINE, not for phone-real-time. The ON-DEVICE LIVE tier stays light: court/person/2D pose/ball/contact/cue/guidance. The `server-fast` preview may use camera-space mesh, while the deep tier owns world-grounded mesh + physics. Full body/physics/racket/replay design: `TECH_STACK.md` §(e)–(s) and `IMPLEMENTATION_PHASES.md` Phases 3–6, 10.

### 5.1 Why this is not a dance port

The dance pipeline (`video → 2D tracking → SAM2 masks → SAM-3D-Body mesh → smoothed floor-anchored output`) ran one heavy path uniformly. Racket sports invert several assumptions:

| Dimension | Dance | Racket sports | Consequence |
|---|---|---|---|
| People | many, chaotic | ≤4, on a fixed court | **cheaper, faster** person finding |
| Scene context | stage floor | court lines, zones, net, ball | **new** geometry + ball/event layers |
| Body model | one heavy mesh path, uniform | phone-live 2D guidance + `server-fast` camera-space SMPL preview + world-grounded SMPL with physics on replay spans | adaptive compute, not uniform |
| Grounding | floor projection | world-grounded off the *solved* court plane (no SLAM needed) + foot-lock to Z=0 | drift-free, skate-free |
| Motion speed | moderate | very fast swings | high temporal density + racket 6DoF at contact |
| Clip length | short | up to ~20 min | segmentation, dead-time skip, tiered storage |
| Camera | controlled | variable height/angle, single | per-clip calibration; known plane = physics + ball-depth anchor |

Net: **cheaper/faster on people and court; the phone-live path is 2D/court/ball guidance, the server path is mesh-core but tiered — `server-fast` camera-space mesh for preview, world-grounded mesh + foot-lock + physics for the replay; heavy racket/physics compute concentrated at contact spans.**

### 5.2 Adaptive compute budget

Cheap baseline everywhere; spend the expensive budget only where it pays:

- **Biomechanically critical events** — fire heavy compute in a ~±0.3 s window around each detected contact (the swing), scaled by shot importance (serve/put-away/error → bigger window; throwaway dink → skeleton only).
- **Where the user sees/pays** — render the mesh avatar (cheap LBS from existing pose params) only for the moments shown in premium reports.
- **Where confidence is low** — a two-pass cascade escalates only uncertain or borderline spans to heavier models. Everything else stays cheap.

### 5.3 Event-triggered heavy compute (the flagship technique)

The body baseline is now world-grounded SMPL mesh, but the *heaviest* compute is still concentrated at the moments that matter:

- Detect contact via **audio "pop" + wrist-velocity peak + ball-trajectory inflection** (~4 ms precision, near-free).
- In the contact window, for the hitting player, after RKT readiness/gates pass, run **racket 6DoF estimation** (PnP on the known paddle geometry → face-normal + contact-point) and the densest mesh/physics refinement of the swing.
- Run the **full physics refinement** (PhysPT, or MuJoCo imitation for the flagship replay) on rally spans; coast through dead time.
- **Foot-lock everywhere** (cheap, runs on all confident-contact frames) so the whole replay is skate-free, not just the highlights.

The replay is mesh throughout; the racket, densest swing fidelity, and full physics-sim are spent at the contact micro-moments that define technique.

### 5.4 Tiered delivery (phone-first, server-deep second)

| Tier | Latency | Work | User value |
|---|---|---|---|
| **ON-DEVICE LIVE / fast** | <10 s | cached court calibration seed, CoreML YOLO26n/s person detect+track+N-lock, 2D pose/joints, risky ~288p CoreML ball heatmap, homography+bounce IN/OUT/net-crossing, mic+wrist contact timing, court map, one priority cue, capture guidance | "I feel seen" during play/setup without waiting for upload |
| **SERVER OFFLINE / deep** | async, notified | Fast SAM-3D-Body / SAT-HMR mesh, WHAM/TRAM/GVHMR/OnlineHMR + court-plane hard constraint, foot-lock + physics, paddle 6DoF, full biomech, 3D replay render, LLM coaching copy, week-over-week | the report + watchable replay people pay for |

The tier toggle *is* the paywall: live guidance is cheap and immediate; deep reconstruction/replay/report is async and premium. Price tracks compute.

### 5.5 Locked model choices

The **target/default model + variant + weight candidates for every stage (offline vs live tier) live in `TECH_STACK.md` §2.3 Model Registry**. Final approved runtime locks belong in `models/MANIFEST.json` after the on-clip benchmark (EVAL-0) **with side-by-side comparison videos and human approval — auto-finalized only when one option is obviously better** (`BUILD_CHECKLIST.md §1.6`). Summary:

| Layer | Choice | License | Why |
|---|---|---|---|
| Court calibration | auto court-keypoint net + manual tap fallback; **`solvePnP` full 6-DOF** + multi-frame averaging; ChArUco/GeoCalib intrinsics; net plane from regulation geometry (34"/36") | OpenCV/own | viewpoint-agnostic; PnP ~2× homography at shallow angles |
| Detect + track | **YOLO26m + BoT-SORT-ReID** + HSV color; court-polygon filter + ground-plane association; N-lock + coach 1-tap anchor | AGPL (OK now) | YOLO26 real (Jan 2026), +1.4–2.8 mAP vs YOLO11 same latency, NMS-free; court geometry does the ID work |
| 2D pose | RTMW-l (whole-body) | Apache | 130 FPS, first open >70 COCO-WholeBody; stays |
| **3D body — server-fast preview** | camera-space mesh — **SAT-HMR (24 FPS) / Multi-HMR 2 (20 FPS, no FOV assumption)** | research | server preview only; not phone-real-time |
| **3D body — deep/replay (core)** | **Fast SAM-3D-Body** per crop (best per-frame mesh, 30.4 mm PA-MPJPE) + **our own world-grounding** (known camera + court plane); GVHMR/WHAM optional trajectory cross-check | SAM (research-OK; ⚠️ verify-commercial; SAT-HMR/Apache fallback) | best per-frame mesh, user-confirmed; we add grounding via calibration |
| Foot-skate + physics | contact vs known Z=0 + zero-velocity + CCD-IK (≤3 mm); **PhysPT** (MIT) default / **PHC+PULSE on MuJoCo+MJX** flagship + MultiPhys (doubles) | MIT/Apache | physically plausible, skate-free replay |
| **Racket 6DoF** | Target stack: SAM 3 concept detection/tracking when approved; DINO-X/Grounded-SAM-2 fallback; FoundationPose when CAD/reference onboarding exists, with GigaPose/FoundPose as alternatives; PnP-IPPE + UKF/rebound as fail-closed gates | OpenCV/own plus approval-gated model licenses | Current runtime is blocked; face-angle/contact-point claims stay gated until RKT validation and ArUco/GT evaluation pass |
| Ball | ON-DEVICE LIVE: distilled/quantized CoreML heatmap tracker (~288p, top risk); SERVER OFFLINE: TrackNetV3→V5/PB-MAT + physics ODE 3D uplift (z=0 bounce + Magnus) | MIT/own | known plane resolves single-cam depth; live cue fails closed until device spike passes |
| Events | ON-DEVICE LIVE: mic onset + wrist-vel peak; SERVER OFFLINE: audio "pop" CNN + wrist-vel peak + ball inflection; net crossing via homography | — | contact timing; drives event triggers |
| Shot class | scaffold/transfer baseline now; PoseConv3D family classifier first, BST-style pose+ball fusion after tensors are clean | open | current external eval misses serve/overhead; not a product claim until reviewed labels and SHOT gates pass |
| Insights | rule-based thresholds (source of truth) + confidence gating + LLM-for-copy (latest Claude) | — | never invents facts |
| 3D replay render | Target production path: **Three.js + R3F** (WebGPU/WebGL2), SMPL-X GLB, physics baked → glTF tracks, world frame (court Z=0). Current repo has a review-only Three.js/R3F viewer and static QA exports, not the shipped animated replay. | MIT | free-viewpoint, ~8–12 MB/10 s, shareable; splats deferred to v2 |
| Visualization | court map/heatmap + 1 priority metric + self-vs-self (conversion core); 3D physics replay premium async | — | matches what converts; progressive disclosure |

### 5.6 Accuracy & training strategy

Accuracy comes from **training, not a bigger inference model.** Full detail in `ACCURACY_AND_TRAINING.md`; the load-bearing decisions:

- **Heavy at training, light at inference.** Use slow/heavy/multi-view/physics/audio models as *teachers* to auto-label our footage, then **distill into fast single-camera students** that ship.
- **The moat — multi-view-to-train-monocular (training-time only).** Record with **2 phones 90° apart at training time only** to manufacture 3D ground truth + a cross-view consistency loss ("Two Views Are Better Than One": −43.6% MPJPE on SportsPose), then **ship a single-camera model** (zero user friction). A live 2nd-camera mode is a *future* power feature, not part of v1.
- **Calibrated court geometry is a free accuracy multiplier** — reprojection loss, metric root depth, and a hard foot-on-ground (Y=0) constraint turn the hardest single-view problems into solvable ones, and convert unreliable 3D-depth velocity (r≈0.28) into reliable metric in-plane velocity (r≈0.70+).
- **Biggest measured win:** sports-domain fine-tune (ASPset-510 + AthletePose3D + AthleticsPose + SportsPose) → **~70% MPJPE reduction (214→65 mm)**.
- **Velocity conflict resolved:** the "R²≈0.96 vs r≈0.11" disagreement was a measurement-type mismatch (a ball-speed model from 2D in-plane speed vs the derivative of noisy lifted-3D positions). See the claim tiers in §13.
- **Corrections flywheel:** every user/coach correction feeds active-learning → retrain → the product gets more accurate per club over time (compounding moat).

### 5.7 Live vs offline & performance (fastest possible results)

Edge-first hybrid target; current repo status is scaffold/partial for the iPhone-side gates listed in `BUILD_CHECKLIST.md`.

- **ON-DEVICE LIVE (<10 s) — target on-device:** live court map, person lock, 2D pose/joints, ball/contact cues, one priority cue, and capture-quality guidance via Apple Vision/CoreML on the Neural Engine. The ~288p CoreML ball heatmap tracker is the top live-tier risk.
- **SERVER OFFLINE (async, streamed) — target server H100 path or explicitly accepted replacement gate hardware:** accurate SMPL-X mesh, foot-lock, physics, racket, metrics, insights, replay — returned **rally-by-rally over SSE** (a "3/14 analyzed" checklist that fills progressively), with an APNs push when complete. Current repo has partial pipeline/artifact scaffolds; the streaming service and production replay delivery are not yet verified.

**Top latency levers:** (1) upload the on-device 2D pose track as a **server prior** — ~166× smaller than video, 50–80% fewer mesh-fit iterations, and the server starts fitting *while the video uploads*; (2) **event-triggered compute** — skip 40–60% dead time (≈2× throughput); (3) **cache the personalized body model** (betas) so repeat sessions converge ~5× faster; (4) warm GPU worker (no cold start); (5) resumable chunked upload. **Target H100 throughput** comes from NVDEC/DALI GPU decode + TensorRT engines + CUDA-stream overlap + B=4 crop batching (SAM-3D-Body runs sequentially per player and is the pacing item to benchmark first), served via Triton; current repo evidence still treats true NVDEC/DALI and Triton serving as pending gates. Multi-agent build/test on the single shared H100 is governed by the GPU lease/MIG protocol in `BUILD_CHECKLIST.md §1.5`.

---

## 6. Why we beat the market

| Differentiator | Attacks |
|---|---|
| **Adaptive compute budget** (faster *and* better) | Everyone runs one fixed model uniformly |
| **Court geometry as a free accuracy booster** (feet hard-constrained to the solved plane) | SportAI/Sportsbox lack calibrated court feedback into pose |
| **Multi-signal fusion** (pose+ball+court+audio cross-check each other) | Competitors solve each in isolation |
| **Audio as a cheap sensor** (contact, rally segmentation, clean-vs-mishit) | Almost no one exploits the "pop" |
| **Personalized body calibration** (locks bone lengths → better & faster with use) | One-shot models can't compound |
| **Confidence as trust** ("no charge if we can't trust it"; gate, don't fake) | Attacks the #1 AI-sports complaint: wrong outputs |
| **Fast-first <10 s + premium async** | PB Vision's silent multi-minute wait |
| **Doubles team-body layer** (middle gap, partner split-step sync) | No incumbent tracks all players in court coords for this |
| **Self-vs-self ghost** (your best rep vs your failed one) | Players trust their own footage; avoids demoralizing pro comparison |
| **"One leak at a time"** | Counters documented metric-overload fatigue |
| **Learns from corrections** (per coach/club) | One-shot competitors can't |
| **Native iOS 3D replay + free ARKit calibration** (RealityKit/USDZ in-app, intrinsics+court-plane from ARKit) | Web-only/manual-calibration competitors |
| **Graceful 1→2 camera scaling** *(future)* — unlock true 3D/line-calls with a 2nd phone | Single fixed product elsewhere |
| **Physics-accurate, foot-skate-free 3D replay** (free-viewpoint, shareable) | Sportsbox shows 6 preset angles of a single body; we render the whole game, physical and rotatable |
| **Target after RKT gate: racket tracked in 3D** (true contact-point-on-face + face angle) | Everyone else uses the hand center as a contact proxy |
| **Known ground plane = physics anchor** (foot-lock to exact Z=0, single-cam ball depth) | In-the-wild HMR methods fight drift/scale; we solved it via calibration |

---

## 7. Product strategy

**Primary wedge:** 3D body habits for coaches and serious players. Every session returns: the habit → the clip → 3D/court proof → one drill → trend vs last session → confidence + correction controls.

**First beachhead — pickleball doubles:** dominates rec play; competitors under-serve partner geometry; kitchen/transition habits are highly visible; 50+ players care about staying on court; clubs/coaches can sell assessment days.

**Second beachhead — tennis serve lab:** high-value, coachable, works before perfect ball tracking, and players already pay for serve lessons.

**Positioning:** not "AI line calls and full stats." Say: *"The body habits costing you points."*

---

## 8. Target users

- **Primary — club coach / teaching pro:** make a student's problem visible fast, assign one drill, prove progress, increase retention. Pays because it saves review time, makes lessons premium, and creates follow-up.
- **Secondary — serious 3.5–4.5 player:** know why the rating is stuck; one fix; clear progress without 40 stats.
- **Tertiary — clubs/facilities:** paid clinics/assessment days; monetize cameras/court time.
- **Anti-persona (v1):** casual player who only wants highlights or line calls.

---

## 9. Novel product ideas

The build-now differentiators first, then the longer-horizon set:

1. **Physics-accurate 3D replay** — watch the whole rally reconstructed (players, court, net, ball, rackets) from any angle, foot-skate-free and physics-consistent; the marquee "wow" + proof layer (see `IMPLEMENTATION_PHASES.md` Phase 10).
2. **Target after RKT gate: racket in 3D / true contact point** — show exactly where on the paddle face the ball hit and the face angle at contact, rendered on the 3D paddle after validation passes.
3. **Self-vs-self ghost**, aligned at contact — now mesh-on-mesh in the 3D replay; the marquee shareable.
4. **"One leak at a time"** — fix one habit, track only it, then advance.
5. **Doubles Chemistry Score** — partner gap over time, split-step sync, middle-ball ownership, stack timing.
6. **Court Advantage Meter** — label each moment attacking/neutral/defending/exposed; grade whether body+court state improved the rally.
7. **Stay-On-Court report** — non-medical movement screen (overreach, backpedaling, knee collapse, asymmetry) for the 50+ lane.
8. **Auto-telestration** — auto-draw the knee-angle arc, contact marker, base-of-support box (the minutes coaches pay to save).
9. **Drill Verification Mode** — 45 s upload, scored reps (10 clean dinks, 10 balanced serves) via wrist-velocity peaks + contact events.
10. **Lowlight reels** (coaching) + **highlight reels** (shareable growth loop).
11. **"Why your rating is stuck"** — explain the bottleneck, don't replace DUPR/UTR.
12. **Tennis Serve Lab** — trophy position, leg drive, trunk tilt, landing balance, foot fault, serve+1 readiness.
13. **Paddle/Racket Fit Lab** — connect specs (weight, swingweight, balance) to movement patterns; profile + body proxy, no 6DoF CV.
14. **Coach Revenue OS** — roster, auto 3-habit report, telestration, homework, before/after, share/PDF, credits. Likely the fastest path to $1M ARR.
15. **Club Assessment Day** — 20 players in a 2-hour block, each a 1-page report, coach gets clinic buckets.
16. **Facility camera body layer** — body layer on existing video; import path from PB Vision/facility clips.
17. **PT / movement-professional pack** — non-diagnostic screen, asymmetry trend, referral-friendly (needs legal/copy review).

---

## 10. MVP scope

### Capture
Single static iPhone (or facility camera). **The native app configures capture; the user only frames it — so the user-facing ask is three things: Landscape · stable tripod · all four corners in view (good light).** The app sets the rest via AVFoundation: landscape, 1080p/**60 fps** (120 fps for swing-speed/racket; not 240, which drops to 720p), **HDR off, locked exposure/focus/WB, shutter ≥1/500 s** (indoor 1/100 or 1/120 to kill flicker). Audio anchors sub-frame contact timing, so 60 fps is sufficient for most metrics. **Variable height/angle handled per-clip** (see §5, §13); the app auto-handles rolling shutter, compression, AF/AE drift, flicker, and shadows (`ACCURACY_AND_TRAINING.md` §2). Tripod ≥1.2–1.5 m, all 4 corners visible. 60–180 s clips for v1 product focus; ~20-min ingest is an architecture requirement. Pre-upload trim + naming; player/side confirmation; optional coach focus tag; failed QC = no credit charged. Upload UX (multipart/resume, facility import, coach batch, when fast vs deep tier starts) is an explicit design task.

### Court setup
- **Pickleball:** 20×44 ft template; 7 ft NVZ lines; baseline/sidelines/centerline; net plane (34" center / 36" sidelines). Target product default after CAL-3 is no user taps: upload/recording should trigger automatic court evidence for outline, kitchen/NVZ, center service lines, and trusted top net. Current server code attempts semantic court evidence, but still needs a trusted sidecar/manual seed or fails closed before tracking; manual corner review/taps remain the fallback/debug label path when automatic evidence or calibration seed trust fails.
- **Tennis:** singles/doubles template, service boxes, baseline, sidelines, net plane. Serve lab needs exact service-box/net geometry.

### Body insights (≤3 per session, defensible only)
| Insight | PB use | Tennis use | Min. signal | Claim status |
|---|---|---|---|---|
| Kitchen/NVZ foot | foot faults, approach timing | net approach | foot keypoint vs court plane | target after CAL/BODY foot gate |
| Split-step / ready timing | late readiness | return/serve+1 | ankle vel vs opponent-contact timestamp | target after event-timing gate |
| Overreach / balance | unstable dinks/resets | stretched ground strokes | base of support + CoM proxy | target after BODY confidence gate |
| Doubles spacing | middle gap, partner exposure | doubles shape | player positions in court frame | target after TRK/CAL gates |
| Transition-zone recovery | stuck between lines | recovery to neutral | zone occupancy over time | target after TRK/CAL/event gates |
| Shoulder–hip separation (X-factor) | drive/serve coil | serve/forehand coil | bilateral shoulder & hip keypoints | target trend after BODY labels |
| Knee bend / trunk tilt | load, posture | trophy/landing | sagittal joint angles | target after angle-validation gate |
| Contact point vs body | early/late | early/late | wrist vs front hip | target after event/BODY gate |
| Paddle-face / contact-point | drive/dink face control | serve/groundstroke face | **racket 6DoF (PnP on known paddle)** | target after true paddle evidence + RKT gate; current box-derived preview stays gated |

### Outputs
Session card with 3 habits; per-habit clip; **court map / top-down with player paths (conversion core)**; **self-vs-self before/after**; physics-accurate 3D mesh replay (premium); confidence score + skipped-span count; one drill per habit; week-over-week tracker; coach share link/PDF; manual correction (wrong player/span/habit, exclude timeline).

### In scope as of round-4
**Racket 6DoF tracking** and a **physics-accurate, foot-skate-free 3D replay** (players + court + net + ball + rackets, free-viewpoint, shareable) — see `IMPLEMENTATION_PHASES.md` Phases 4/6/10.

### Out of scope (v1)
Official/challenge-grade line calling, auto scoring, full-match auto-analysis UX on day one (but ~20-min ingest + tiered processing are architecture requirements), live in-rally coaching beyond the single priority cue, Apple Watch challenges, medical diagnosis, consumer-only launch before pilots. Photoreal Gaussian-splat avatars are deferred to v2.

---

## 11. Native court / net / ball / paddle tracking

### Court / floor
Stable per-clip court coordinate frame. MVP server path: `court_templates.py` geometry, `court_calibration.json`, `court_zones.json`, `net_plane.json`, trusted sidecar/manual-review calibration when available, automatic `court_line_evidence.json` on every run, reprojection sanity check, tripod-bump guard (re-verify every N frames + optical-flow warp), court zones (NVZ, transition, baseline, service boxes). Auto-detect is no longer optional for product flow: video-backed runs must attempt semantic court evidence and fail closed if sport-specific line evidence (pickleball kitchen/NVZ and centerlines; tennis service lines) plus top-net evidence is not trustworthy. The fully no-tap calibration solve is still gated behind the CAL-3 trained/heuristic keypoint-line solver; manual taps are fallback labels, not the main user flow.

### Net
Static net plane from the court template + regulation heights. Observed top-net pixels are used as evidence/trust checks, not as blind height estimation; projected top-net overlays and downstream net logic are low confidence when the observed top net is missing or inconsistent. Defer net-cord/contact classification.

### Ball
**MVP goal: timing/context plus cheap live line guidance, not official line adjudication.**
- **Phase A:** manual tap-track on key frames → anchor split-step, contact proxy, advantage state.
- **Phase B:** TrackNetV3 fine-tuned on ~2–3k labeled PB frames (+ converted Roboflow data) + Kalman/physics smoothing; output confidence; gray uncertain spans.
- **Phase C:** on-device cached-court homography+bounce IN/OUT and net-crossing guidance, plus server rough 3D via physics + "Z=0 at bounce." Official/challenge-grade line adjudication remains gated and likely needs a second camera.

### Paddle / racket — target tracked 6DoF object after RKT gate (round-4)
**Target: track the racket, don't proxy it.** Detect (RTMDet/SAM2-class or approved fallback) → top/bottom/handle keypoints + true face corners → **PnP-IPPE on the known paddle dimensions** → UKF on SE(3) → physics-validate against ball rebound → contact-point-on-face (±1–3 cm) + face-normal (~3–5°). Render the paddle in the 3D replay attached to the wrist bone only after the RKT/replay gates pass. Current repo reality is fail-closed PnP/review scaffolding: box-derived candidates are rejected before canonical `racket_pose.json`, and no detector/keypoint/CAD/GT gate has passed. Keep the spec database (weight, balance, swingweight) for the Fit Lab. Use side-edge keypoints only as a weak cue (RacketVision shows they degrade under occlusion/blur). Full pipeline + training in `IMPLEMENTATION_PHASES.md` Phase 6 and `TECH_STACK.md` §(r).

---

## 12. Architecture

See `TECH_STACK.md` for the full stack/toggle map and `IMPLEMENTATION_PHASES.md` for build steps.

### Client / server split (single camera target)
The table below is the target architecture. Current checked-in iOS code is scaffold/partial unless a gate in `BUILD_CHECKLIST.md` says otherwise.

| Target on **iPhone (Swift)** | Runs on **GPU server** |
|---|---|
| AVFoundation locked capture (exposure/focus/WB, fps/format, landscape, HEVC/ProRes) | Court calibration refine (seeded by ARKit sidecar) + net plane |
| ARKit setup pass → camera intrinsics + 6DoF pose + court plane (sidecar) | Person detect/track/ID (YOLO26m + BoT-SORT) |
| ON-DEVICE LIVE fast tier: CoreML YOLO26n/s person detect+track+N-lock, 2D pose/joints, ~288p CoreML ball heatmap, homography+bounce line cues, mic+wrist contact timing, court map + one cue + capture guidance | Fast SAM-3D-Body / SAT-HMR mesh → WHAM/TRAM/GVHMR/OnlineHMR cross-checks with court-plane hard constraint → foot-lock → physics |
| Upload: trimmed clip + sidecar + on-device pose/person/ball/contact priors (+ LiDAR depth bonus when present) | Paddle 6DoF, full biomech, ball 3D physics, metrics, insights, LLM copy, week-over-week |
| Replay viewer: RealityKit/USDZ (free-viewpoint) | Render-bake one scene → **USDZ** (native) + **GLB** (web) |

On-device is **preview/guidance only**; the server is the source of truth. Device tiers: Pro phones may add LiDAR as a near-field (~5 m) bonus, standard phones are the vision-only baseline, and fallback devices use manual court-tap calibration.

### Module layout (server: new code under `threed/racketsport/`)
```text
threed/racketsport/
  court_templates.py      court_calibration.py     court_zones.py
  net_plane.py            capture_quality.py       drift_guard.py
  person_fast.py          track_lock.py            doubles_id.py
  hmr_fast.py             hmr_deep.py              worldhmr.py
  footlock.py             physics_refine.py        body_mesh_readiness.py
  ball_tracknet.py        ball_tap_track.py        ball_physics3d.py
  audio_pop.py            event_fusion.py          contact_windows.py
  racket6dof.py           movement_metrics.py      biomech.py
  insight_rules.py        confidence.py            shot_classifier.py
  drill_verify.py         habit_model.py           report_model.py
  llm_copy.py             replay_readiness.py      replay_export.py
  virtual_world.py        pipeline_contracts.py    orchestrator.py
  review_packet.py        replay_viewer_manifest.py
  schemas/                eval/
ios/                   # native Swift: Capture / Calibration / FastTier / Guidance / Upload / Replay
web/replay/            # React + Three.js/R3F replay viewer (web share)
```

### Tiered data flow
```text
record/play setup
  |
  +-- ON-DEVICE LIVE / FAST TIER (<10 s, iPhone ANE)
  |     ARKit intrinsics+pose+floor-plane seed, cached
  |     CoreML YOLO26n/s person detect+track+N-lock + court-polygon filter
  |     Apple Vision or CoreML RTMPose-m 2D pose/joints
  |     distilled/quantized CoreML ball heatmap (~288p) -- top live risk
  |     homography+bounce IN/OUT and net-crossing on cached court, known Z=0
  |     mic onset + wrist-velocity peak contact timing
  |     -> live court map + 1 priority cue + capture-quality guidance
  |
  +-- SERVER OFFLINE / DEEP TIER (async GPU after upload)
        Fast SAM-3D-Body / SAT-HMR mesh
        WHAM/TRAM/GVHMR/OnlineHMR + court-plane hard constraint
        foot-lock + physics
        paddle 6DoF
        full biomech
        3D replay render
        LLM coaching copy
        week-over-week
```

### Artifacts
`court_calibration.json`, `court_zones.json`, `net_plane.json`, `ball_track.json`, `contact_windows.json`, `racket_pose.json`, `foot_contact.json`, `physics_refinement.json`, `racket_sport_metrics.json`, `habit_report.json`, `coach_report.json`, `corrections.json`, `capture_quality.json`, `replay_scene.json` plus referenced `.glb` assets. Schemas in `IMPLEMENTATION_PHASES.md`.

---

## 13. Defensible accuracy envelope

This table is the target claim envelope, not current shipped capability. Claim a row only after the matching `CAPABILITIES.md`/`BUILD_CHECKLIST.md` gate passes; current repo truth remains `VERIFIED = 0`.

| Signal class | Status | Action |
|---|---|---|
| Positions (feet, zones, spacing) post-calibration | cm-level | target claim after gate |
| Sagittal/large-joint angles (knee, elbow, trunk) | ±3–10° (beats a coach's ~12° eye) | target claim after gate |
| Shoulder–hip separation (X-factor) | reliable as a trend with bilateral keypoints | target claim as trend after gate |
| Contact timing (with audio) | <1 frame | target claim after gate |
| **Paddle-face angle + contact-point — via tracked racket** | target ~3–5° face / ±1–3 cm contact after RKT gate | **claim only after racket-6DoF validation (≤5° vs ArUco GT)** |
| Foot-slide during contact / floor & inter-player penetration | ≤3 mm / zero (physics-constrained) | **target claim after hard gates** |
| Wrist-only pronation / transverse-axial rotation (no racket) | 20–57° markerless error | **gate — superseded by the tracked racket above** |
| Velocity — wrist/elbow swing speed (ball-speed predictor), horizontal CoM velocity, split-step timing/tempo | reliable in-plane via court homography (±15–25%; timing ±17 ms@60 / ±8 ms@120) | **Tier 1 — claim with number after Protocol A/B passes** |
| Velocity — swing-speed index, 3D angular velocity | relative/trend only | **Tier 2 — present as "relative/estimated"** |
| Velocity — 3D angular acceleration, depth-axis velocity, contact-frame peak from pose | not recoverable markerless | **Tier 3 — never claim (use the ball for contact-frame speed)** |
| 3D ball height/depth (single cam) | meter-level | context only; live IN/OUT guidance uses homography+bounce on the court plane, not meter-level 3D |

Present uncertainty as ranges ("likely 52–61%"), bin confidence (confident/lean/uncertain), and gate any metric behind a minimum sample. Low/shallow camera angles lower the capture-quality score and widen gates.

---

## 14. Target feasibility verdict

This is a product/research target view, not current capability status. Current invoked pipeline truth lives in `CAPABILITIES.md` and mutable task counts live in `BUILD_CHECKLIST.md`.

| Claim | Target feasibility | Current truth source | Why |
|---|---|---|---|
| Short-clip person tracking (≤4) | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | court geometry does the ID work; cheap detector suffices |
| On-device live court/person/2D pose/ball/contact guidance | High target, ball tracker is the top risk | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | court and 2D pose are cheap; the ~288p CoreML ball heatmap must prove device FPS/thermal/recall |
| Court calibration (variable camera) | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | per-clip homography + solvePnP; reprojection-gated |
| NVZ/kitchen foot, spacing, balance | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | world-grounded mesh + foot-lock to court plane |
| Split-step / contact timing | Medium-high | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | needs ball/audio event; audio makes it strong |
| World-grounded mesh + foot-skate-free replay | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | Fast SAM-3D-Body + our world-grounding (known camera) + foot-lock to known Z=0 plane (≤3 mm) |
| Physics-accurate 3D replay (Three.js, free-viewpoint) | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | mesh + baked physics -> glTF; engineering, not research |
| Racket 6DoF (face angle / contact-point) | High target after gate | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | PnP on known paddle geometry; claim only after RKT gate and ArUco/GT evaluation pass |
| Ball automation | Medium | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | on-device homography+bounce line guidance is in scope; official/challenge-grade line adjudication likely needs 2 cams |
| Coach reports / UX | High | See `CAPABILITIES.md` + `BUILD_CHECKLIST.md` | mostly product/backend |
| $1M ARR | Plausible | market thesis, not pipeline capability | coaches/clubs first |

---

## 15. Build plan (summary)

Full detail with per-phase test gates is in `IMPLEMENTATION_PHASES.md` (see its One-Page Build Order Summary). Current priority sequence: **env/data/scaffolding → iOS capture/calibration sidecar → ON-DEVICE LIVE court/person/2D pose guidance → on-device ball heatmap spike + cached-court IN/OUT/net/contact cues → server calibration/tracking support → server ball/audio/event gates → SERVER OFFLINE mesh/world-grounding/foot-lock/physics → paddle 6DoF → metrics + rule insights + confidence → 3D replay render/viewer → shot classification + drill verification → LLM copy + report + server-progressive delivery → end-to-end on 20-min clips + perf + final acceptance.** SHOT remains scaffold/transfer-only until reviewed labels, trained model, macro-F1/top-2, and per-class serve/overhead gates pass.

**Gate philosophy:** do not advance until a phase's numeric acceptance gates pass on real GPU test clips spanning deliberately varied camera heights/angles.

---

## 16. Pricing and $1M ARR

Pure B2C at $19/mo needs ~4,400 retained payers — slow. Better: B2B2C.

| Segment | Price | Count | ARR |
|---|---:|---:|---:|
| Clubs | $299–399/mo | 75–100 | ~$270–479k |
| Coaches | $99–119/mo | 500 | ~$594–714k |
| Players | $19/mo | 750–1,000 | ~$171–228k |

Packaging: **Player** ($19/mo, 4 analyses, tracker, reports, paddle profile); **Coach** ($99–149/mo, roster, 25–50 analyses, reports/share/corrections); **Club** ($299–499/mo platform + usage credits + assessment tooling). Credit rules: failed QC = no charge; low-confidence comped/discounted; long clips cost more; ON-DEVICE LIVE guidance is cheap, SERVER OFFLINE deep reconstruction/replay/report costs more — price tracks compute.

---

## 17. Success metrics

**Product:** coach says report is lesson-useful ≥70%; player states one fix ≥70%; 2nd session within 30 days ≥40%; habit dismiss rate on confident spans <15%; failed QC after guidance <20%; ≥5 min review saved/session.
**Technical:** court reprojection passes on ≥90% of clips across varied angles; on-device person lock/2D pose/court guidance passes physical-device latency gates; on-device ball heatmap spike either passes or line/contact cues stay disabled fail-closed; NVZ foot within 6 in on confident frames; spacing within 12 in; ID stability ≥90% on 2-player clips after confirmation; contact timing within ±2 frames with audio; **ON-DEVICE LIVE first screen <10 s**; server deep-tier p95 under agreed async SLA.
**Business:** coach WTP ≥$79/mo; club ≥$299/mo or event fee; pilot→paid ≥30%; gross margin ≥70% at target pricing.

---

## 18. Quick tests to run ASAP

Run before any large push. (Codex phase gates expand these.)

1. **Person tracking sanity** — 10 clips (5 doubles, 3 singles/drill, 2 messy), baseline-corner + side-fence cameras. Pass: 2 main players stable enough for court metrics on 8/10.
2. **Court calibration across varied camera angles** — tap 4 corners; overlay court lines; report reprojection error on clips spanning low/high/steep/shallow setups. Pass: overlay visibly matches on 8/10 across ≥4 distinct viewpoints.
3. **Foot-to-NVZ from 3D skeleton** — Pass: reviewer agrees with in/out/near NVZ on confident frames ≥80%.
4. **Velocity-metric reality check** — compute wrist/joint velocities on labeled swings and correlate with manual/known values. Decides whether velocity claims are defensible or must be gated. (Resolves the R²0.96-vs-r0.11 conflict.)
5. **Contact timing via audio + pose + ball** — Pass: contact within ±2 frames vs manual on selected moments.
6. **On-device ball detector spike** — distilled/quantized CoreML heatmap tracker at ~288p on a physical iPhone; measure FPS/latency, thermal behavior, precision/recall, false positives from lines/shoes/paddles, temporal continuity, and cached-court line/contact cue stability. Pass: enough recall for live guidance; fail → disable line/contact cues and keep server/tap-track paths.
7. **Coach usefulness** — 5 manually assisted reports to 5 coaches. Pass: ≥3/5 would use it in a lesson.
8. **Runtime/cost** — one ON-DEVICE LIVE run on iPhone + one SERVER OFFLINE burst run on accepted GPU hardware; record latency, thermal/GPU time, cost. Pass: live tier <10 s; deep tier acceptable async.

---

## 19. Validation dataset

**Pickleball:** current DATA-1 sizing is governed by `IMPLEMENTATION_PHASES.md` Test-Clip Spec: minimum 24 clips with short and long slices, varied camera heights/angles, high-FPS/ArUco subsets, indoor/outdoor, doubles/singles/drill, and messy real-world coverage. The older MVP target was 30 short clips; treat that as superseded by the Test-Clip Spec matrix. Labels: court corners/lines, player boxes/IDs (sampled), support foot near NVZ (sampled), ball location (selected frames), contact/bounce timestamps, coach-scored habits.
**Tennis:** 20 serve + 10 rally/recovery clips, singles & doubles. Labels: calibration, serve-contact timestamp, landing foot/balance, recovery position.
**Acceptance philosophy:** practical truth (manual overlays, foot/NVZ review, coach usefulness, confidence honesty, before/after) — not perfect 3D ground truth first.

---

## 20. Product UX

**Player flow:** select sport → clip type → setup/record with ON-DEVICE LIVE court/person/2D pose/ball/contact guidance → trim/upload → confirm side/identity → optional paddle profile → submit → server-progressive results → notified when deep report ready → review 3 habits → corrections → repeat next week. Manual court review appears only when auto court evidence fails and the user/report should not be charged as a trusted automatic result.
**Coach flow:** create student/session → upload/import → confirm focus → review auto habits → dismiss/edit spans → add note → send report → assign drill → track next session.
**Club flow:** create assessment event → batch upload → assign names → generate reports → bucket by clinic need → send follow-up.
**Design rule:** first screen is the court map + the priority habit, never a raw metric dashboard:
```text
Habit 1: Late split-step
Habit 2: Overreaching at the kitchen
Habit 3: Partner gap opens after your drop
```
Each habit opens into proof. Feedback unit = **one prescriptive external cue + annotated overlay + plain-language "why + what to do" + a self-monitorable drill.**

---

## 21. Insight rules (MVP)

Rule-based thresholds are the source of truth; LLM only phrases. Each rule states metric, required signals, and confidence gate.

**Pickleball:** Kitchen foot (foot vs kitchen plane; needs calibration + foot confidence) · Transition stuck (zone occupancy + recovery time; needs zones + event timestamp) · Partner gap (inter-player distance; needs stable IDs) · Overreach (shoulder/hip/hand vs support base; needs joints + contact state) · Late split (split proxy vs ball/opponent timestamp; needs event time) · Arm-led stroke (hip-rotation vs elbow-extension peak timing; gate velocity until validated).
**Tennis:** Serve landing balance · Toss/contact consistency · Serve+1 readiness · Backhand spacing.

---

## 22. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---:|---|
| Velocity metrics not defensible | High | Phase-3 empirical validation; gate until proven |
| Paddle-face from wrist overclaimed | High | use the tracked racket (PnP), not the wrist; validate ≤5° vs ArUco GT |
| Foot-skating / unphysical replay | High | foot-lock to known Z=0 + zero-velocity + physics refine (PhysPT/MuJoCo); ≤3 mm gate |
| Racket pose fails (blur/occlusion/edge-on) | Medium-high | reliable top/bottom/handle keypoints only; hand-grip prior; UKF predict through occlusion; physics-validate |
| Variable camera angle breaks geometry | High | per-clip calibration + capture-quality score + confidence gating |
| On-device ball automation unreliable | High | top live-tier spike; disable line/contact cues fail-closed; tap-track/server fallback; do not promote until physical-device evidence passes |
| Server deep/replay tier too slow/costly | High | `server-fast` camera-space preview only after live tier; physics offline; PhysPT (no engine) before full MuJoCo sim; tiered pricing |
| Long uploads blow queue/storage | High | rally segmentation, dead-time skip, tiered retention, chunked upload |
| Doubles ID swaps | Medium-high | court-polygon + ground-plane association + coach anchor + correction UI |
| Coaches undervalue 3D | Medium | run 5 manual reports before full build |
| Bleeding-edge models unverified | Medium | named fallbacks (Fast SAM-3D-Body→original→NLF; SAT-HMR↔Multi-HMR 2; YOLO26→YOLO11; PhysPT→PHC); verify repo/license before committing |
| Future commercialization re-blocks licenses | Low-now | research/personal use now; if commercializing, swap 3D backbone to **SAT-HMR (Apache)** and detector to RTMDet/RT-DETR; license info retained per stage in `TECH_STACK.md §2.3` |
| Competitors (SportAI/Sportsbox) add PB mechanics | Medium-high | move fast on PB-native mechanics, doubles team-body, 50+ lane, confidence honesty |
| Overbuild before proof | High | 7-day falsification sprint first |

---

## 23. Work packages

WP-PB-00 validation set · 01 iOS capture/calibration sidecar · 02 ON-DEVICE LIVE court/person/2D pose guidance · 03 on-device ball heatmap + line/contact spike · 04 server court calibration/tracking support · 05 server ball + audio + event fusion · 06 SERVER OFFLINE mesh/world-grounding + foot/body metrics · 07 foot-skate killer + physics refine · 08 racket 6DoF · 09 habit rules + confidence gating · 10 report artifacts + LLM copy · 11 3D replay render/viewers · 12 shot classification + drill verification · 13 coach web review · 14 paddle profile/Fit Lab · 15 club assessment flow.

---

## 24. Initial Build Order Snapshot

This was the product-strategy build order at the time of the MVP memo. The
current operational order and blockers live in `BUILD_CHECKLIST.md`, especially
the Critical next build order and task table. Use that checklist, not this
snapshot, to decide what to build next.

1. iOS capture/calibration setup: ARKit intrinsics+pose+floor-plane seed, cached; manual fallback and sidecar packaging.
2. ON-DEVICE LIVE person detect+track+N-lock plus 2D pose/joints and court map/capture guidance on a physical iPhone.
3. ON-DEVICE LIVE ball heatmap spike at ~288p plus cached-court IN/OUT/net-crossing and mic+wrist contact timing; disable cues fail-closed if the spike fails.
4. Server court calibration/tracking/ball support gates, then Fast SAM-3D-Body + our world-grounding (known camera + court plane) on a 60–90 s doubles clip.
5. Foot-skate killer (contact vs Z=0 + zero-velocity + CCD-IK) → ≤3 mm; foot/NVZ + spacing + balance with confidence.
6. Three.js/RealityKit 3D replay viewer (court + SMPL-X mesh + baked motion; free-viewpoint, share link) after upstream deep-tier gates are trustworthy.
5. Audio + ball events for contact windows; racket 6DoF on the hitting player.
6. Court map + 3 habit cards + self-vs-self + one shareable coach report (LLM copy from facts).

Do not start with: official/challenge-grade line adjudication, full-match upload, DUPR integration, App Store consumer funnel, full MuJoCo physics sim (use PhysPT first), Gaussian-splat avatars (v2). The on-device ~288p ball heatmap spike is a priority because live line/contact guidance depends on it.

---

## 25. Final recommendation

The company is **Sway Body: the 3D movement coach for racket sports** — built on a world-grounded mesh core, fast preview overlays, an adaptive compute budget that spends deepest fidelity only where it matters, court geometry as a free accuracy moat, multi-signal fusion, and confidence honesty as a trust feature. Lead with pickleball doubles body IQ, kitchen/transition habits, doubles team-body, and the 50+ stay-on-court screen; second with the tennis serve lab. Sell to coaches and clubs first. If the quick tests pass, this is a credible $1M+ ARR direction; if they fail, the fallback (coach workflow + manually assisted reports) is still valuable while the CV layer matures.

---

## Sources

Market/product: SFIA pickleball participation; USTA 2026 report; AOAO injury study; senior fall-prevention RCT (PMC12469448); PB Vision roadmap & help docs; SwingVision/DinkAI App Store; Sportsbox AI; SportAI (TIME Best Invention 2025); Onform/CoachNow; "Talking Tennis" (arXiv 2510.03921). WTP: PB Vision/SwingVision/Sportsbox/OnForm pricing pages; Tennis AI 2.0; OnCourtAI; RevenueCat State of Subscription Apps 2025.
CV/biomechanics: TrackNetV2/V3/V5, BlurBall/TOTNet, WASB; TennisCourtDetector, PnLCalib, GeoCalib; **YOLO26m + BoT-SORT-ReID** (YOLO11/RTMDet/RT-DETR fallbacks), RTMPose/RTMW/RTMW3D, MotionBERT, MeTRAbs. 3D body (per-frame mesh): **Fast SAM-3D-Body (primary backbone), SAM-3D-Body, NLF**; `server-fast` **SAT-HMR / Multi-HMR 2** preview; **world-grounding done by us** via known camera + court plane (GVHMR/WHAM/TRAM = optional trajectory cross-check only; PromptHMR rejected); SMPLer-X/SMPLest-X (hands), BioPose (joint limits). Physics: **PhysPT, PHC/PULSE, MuJoCo+MJX, MultiPhys, UnderPressure, SMPLOlympics**. Racket: **RacketVision, GigaPose, FoundPose, FoundationPose, HOISDF, OpenCV PnP-IPPE**. Datasets: AMASS, BEDLAM2, AthletePose3D, CalTennis, RICH, EMDB, Human3.6M. Render: **Three.js, React Three Fiber, Rapier.js**. Shot/insight: BST, PoseConv3D/ST-GCN, CoachMe/BioCoach. Markerless validation: Theia3D, OpenCap, Pose2Sim. Licensing is informational only (research/personal use now); verify bleeding-edge repos before committing. Full per-claim URLs in `TECH_STACK.md` and `ACCURACY_AND_TRAINING.md`.
```
