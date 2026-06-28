# Sway Body — Accuracy & Training Playbook

**Purpose:** the concrete plan to push every pipeline stage to the highest accuracy achievable at our near-real-time speeds, given that we **have a strong training GPU and can download/collect/label datasets**. Datasets to download, training/fine-tuning/auto-labeling recipes, filtering parameters, the user capture spec, and the validation protocols that gate each phase.

**Companion docs:** `SWAY_BODY_PICKLEBALL_MVP.md` (product/strategy), `TECH_STACK.md` (model choices + why), `IMPLEMENTATION_PHASES.md` (Codex build + test gates; foot-skate elimination, physics, racket-6DoF, and 3D replay are built in Phases 4/6/10), `BUILD_CHECKLIST.md` (operational checklist + multi-agent coordination protocol). This doc is the source of truth for *accuracy*; the phase plan's training/validation steps reference it.

> **Round-4 update (mesh is now core; body backbone = Fast SAM-3D-Body).** The earlier "skeleton-first" choice is **superseded: SMPL/SMPL-X mesh is now the core body representation**, with the fast skeleton demoted to a preview/triggering overlay. The per-frame mesh backbone is **Fast SAM-3D-Body** — the most accurate per-frame mesh available (3DPW PA-MPJPE ~30.4 mm, beats HMR2.0 by 15+ mm), user-confirmed in practice, and it keeps the existing `sam4dbody` investment. (License: SAM License — **fine for research/personal use now; verify before commercial**; the license-safe Apache fallback is **SAT-HMR** — see the round-7 note and `TECH_STACK.md §2.3`.) **World-grounding/foot-contact is built by us** off the known court plane — we do **not** adopt GVHMR/WHAM as the primary mesh (their per-frame backbone is HMR2.0, which is weaker; demoted to optional trajectory/foot-velocity sanity-checks). **Licensing is still lifted** (not selling yet → AMASS/BEDLAM/AthletePose3D/CalTennis usable for *training*; YOLO AGPL fine), retained as informational metadata for future commercialization. See `TECH_STACK.md` §(e),(p)–(s) and `IMPLEMENTATION_PHASES.md` Phases 3/4/6/10 for the full body/physics/racket/replay design; this doc covers the *training and accuracy* angle.

> **Round-5 update (single-camera product; native iOS app).** **Single camera is the product focus; multi-camera is a FUTURE feature** and multi-view (§12) is a **training-time-only** technique that still ships a single-camera model. The capture surface is a **native iOS Swift app**, which is itself an accuracy advantage: the app locks exposure/focus/WB and sets the frame rate via **AVFoundation** (no reliance on the user toggling settings), an **ARKit** setup pass supplies camera **intrinsics + 6DoF pose + a court-floor plane** as a sidecar that *seeds* calibration (§3) for free, and the **on-device Apple Vision pose track is uploaded as a prior** that speeds/stabilizes the server-side Fast SAM-3D-Body fit. On Pro devices, near-camera **LiDAR depth** is *extra* supervision within ~5 m (§3, §5c) — vision-first stays the baseline.

> **Round-7 update (canonical Model Registry + tested variant selection).** **`TECH_STACK.md` §2.3 "Model Registry" is the single source of truth for the exact model variant + weights at every stage** — this doc must not contradict it. The registry's values are **defaults + candidates, not final**: each stage's candidate variants are **benchmarked on real clips, rendered as side-by-side comparison videos, and the human approves the pick before it is locked** in `models/MANIFEST.json` — **unless one variant is obviously better, in which case Codex may auto-finalize and log it** (gate: `BUILD_CHECKLIST.md §1.6`). **LIVE = lightest variant that gives a usable preview; OFFLINE always recomputes with the accurate variant** (the live output is a preview + a server *prior*, never the final result). ⚠️ **Every speed number in this doc is a T4/A100/RTX extrapolation — there are no published H100 latencies; re-benchmark on the real H100 + on-device before treating any FPS/ms as fact, and benchmark Fast SAM-3D-Body first (the pacing item).** **License posture:** SAM-3D-Body (SAM License) and Multi-HMR 2 (Naver) are the only commercial-risk items — fine for research/personal use now; if commercializing, switch the 3D backbone to **SAT-HMR (Apache)**.

---

## 0. Governing thesis

1. **Heavy at training, light at inference.** Use slow/heavy/multi-view/physics/audio models as *teachers* at training-and-labeling time to manufacture ground truth, then **distill into fast single-camera students** that ship. The product runs one cheap path; the accuracy comes from offline training.
2. **Calibrated court geometry is a free accuracy multiplier.** We solve camera intrinsics + pose per clip — and on iOS the **ARKit setup pass hands us intrinsics + 6DoF pose + a court-floor plane directly** (no checkerboard), which we refine server-side (§3). So we get the ground plane "for free," converting the hardest single-view problems (absolute depth, foot contact, metric velocity) into constrained, solvable ones. This is our structural advantage over SportAI/Sportsbox, who estimate geometry and eat the error.
3. **Validate, then claim.** Every metric ships only after passing a numeric validation protocol on our own GPU test clips (Section 10). Until then it is gated, presented as a relative/qualitative signal, or omitted. Accuracy honesty is a product feature.
4. **Optimize for the real input,** not a lab clip: single static phone, variable height/angle, H.264, fast motion, doubles, indoor/outdoor — see Section 2.

---

## 1. The capture spec (the biggest free accuracy win)

Zero ML cost, largest immediate quality lift. **In the native iOS app the *app* configures capture — the user only places the tripod.** The user-facing ask is **three things: Landscape · stable tripod · all four corners in frame.** Everything else (exposure/focus/WB lock, frame rate, format, HDR-off) the app sets via AVFoundation.

| Parameter | App sets (AVFoundation) | Why |
|---|---|---|
| Orientation | **Landscape** enforced (`videoRotationAngle`, iOS 17+) | Portrait crops the court to an unusable slice |
| Resolution / fps | **1080p / 120 fps default** (`activeFormat` + `activeVideoMin/MaxFrameDuration`); 60 fps floor on constrained devices | 120 fps makes swing-speed numbers always available (§5b) and halves motion blur; 4K adds storage for ~0 CV gain at 8–15 m |
| Exposure | **Locked, custom** (`setExposureModeCustom`, 1/500–1/1000 s shutter, clamped ISO); HDR off | Stops AE pumping (false motion in ball-tracker stacks); fast shutter freezes the ball |
| Focus | **Locked** (`setFocusModeLocked` on court distance) | No AF hunting on fast motion; stable intrinsics |
| White balance | **Locked** (`setWhiteBalanceModeLocked`) | Consistent ball/court color for segmentation |
| Codec | **HEVC** default; **ProRes 422 LT** on Pro (13 Pro+) | ProRes = near-lossless frames for best server reconstruction |
| Calibration sidecar | **ARKit** setup pass → intrinsics + 6DoF pose + court-floor plane | Free calibration seed (§3); no checkerboard |
| On-device prior | **Apple Vision** pose track uploaded with the clip | Speeds/stabilizes the server Fast SAM-3D-Body fit |
| Placement (user) | tripod ≥1.2–1.5 m, all 4 corners visible, stable | Higher angle → less foot occlusion, better depth; app's capture-quality guidance coaches this in real time |

**Special modes:**
- **240 fps "impact deep-dive"** (≈720p binned, rear-only) for max ball-trajectory/impact temporal resolution — let the server super-resolve. (Default 120 fps already covers swing-speed numbers.)
- **LiDAR depth (Tier-A Pro devices):** near-camera (~5 m) depth as *extra* supervision (near-player foot-contact, near court-plane) — **not** court-spanning, **not** the ball, fails in direct sun. Vision-first is the baseline.
- **2-camera mode — FUTURE.** A second phone is a *training-time* instrument (§12) and a future power-user product toggle for true 3D triangulation; **not** part of the single-camera v1 product.

The app **enforces** Landscape + locked capture and **auto-handles** the rest (Section 2). Consumer-proven: PB Vision requires a stationary camera ≥4 ft with all corners visible; CalTennis used iPhones on $40 tripods at 1.65 m.

---

## 2. Real-input pathologies — handle or guide

| Pathology | Effect | User mitigation | Automatic mitigation |
|---|---|---|---|
| Rolling shutter (CMOS readout 10–30 ms) | ball-position bias; racket skew on fast swings → velocity error | keep EIS/OIS on; **120 fps** shrinks readout fraction | RS-warp augmentation in training; prefer low/global-shutter at high fps; RS correction only on slow-mo offline clips (too slow for hot path) |
| H.264/HEVC artifacts | low-bitrate CTUs can erase a ~30–40 px ball; pose precision loss | record highest-bitrate / ProRes | train on H.264-degraded augmentations; light artifact-reduction pre-pass **before ball detection only** |
| HDR / auto-exposure | brightness jumps = false motion in TrackNet stacks | **HDR off, lock exposure** | temporal histogram matching to a reference frame |
| Autofocus hunting | periodic blur during rallies | **lock focus** | Laplacian-variance blur detector → down-weight frames |
| White-balance shift | color-based ball/court cues drift | manual WB if available | per-frame color normalization; don't rely on absolute hue |
| Indoor LED/fluorescent flicker (120 Hz banding) | banding mimics motion/edges | shutter 1/100 s (50 Hz) or 1/120 s (60 Hz) | banding detector + flicker augmentation |
| Outdoor shadows/glare | shadow lines mimic court lines → court-keypoint false positives | sun behind camera | CLAHE per-frame; strong shadow augmentation |
| Portrait orientation | court unrecoverably cropped | **landscape nag** | orientation detector → block/warn at upload |

**Use-case distribution to optimize for** (engineering estimate — refine from telemetry): baseline-corner ~45% (the mode), side-fence ~30% (best for doubles), elevated ~15%, dual ~10%. **Gate quality bars and active-learning sampling toward baseline-corner + side-fence (~75% of reality).** Dead time between points (~10–20 s) is free signal: use it for background-subtraction calibration, court re-fit, and rally segmentation.

---

## 3. Court calibration accuracy

**Decisive stack:** start from the **ARKit sidecar** (camera intrinsics + 6DoF pose + horizontal court-floor plane captured on-device) → auto-detect court keypoints → distortion-correct → aggregate over 20–40 static frames → **`solvePnP` (full 6-DOF), not homography-only** → reprojection gate + capture-quality score. The ARKit plane + pose give a strong initialization that the server refines; manual court-corner tap is the fallback when ARKit tracking is `.limited` (blank/low-contrast courts, low light).

**Current implementation note (2026-06-28):** the default server calibration stage now attempts automatic semantic court evidence for every available upload video/frame and writes `court_line_evidence.json` with kitchen/NVZ, centerline, and top-net readiness. Video-backed runs stop before tracking when that evidence is not ready. A random video without a trusted calibration seed still fails closed instead of fabricating `court_calibration.json`; the trained/heuristic no-tap solver must pass the gates below before this becomes verified automatic calibration.

Ranked levers:

| # | Lever | Expected gain | Effort |
|---|---|---|---|
| 1 | **`solvePnP` full pose vs homography-only** | Acc@5 0.71 vs 0.59; ~2× and essential at shallow angles | Low |
| 2 | **Multi-frame averaging (20–40 static frames)** | −23–57% alignment error, free | Low |
| 3 | Sub-pixel line/intersection refinement (intersect refined lines, incl. out-of-image → recovers occluded corners) | ~3–5× keypoint precision | Med |
| 4 | Line-based (PnL) point+line joint optimization | +5.7–8.3% over points-only | Med |
| 5 | Train our own court-keypoint detector | removes manual taps; ~2 px median err | High |
| 6 | One-time ChArUco intrinsics per phone-model | removes 5–20 px edge distortion → 0.5–2 ft feet error | Low/model |

**Intrinsics/distortion (tiered):** **ARKit `ARFrame.camera.intrinsics`** (on-device, per-clip, free — primary on iOS) → cached **ChArUco per `phone-model + zoom`** (refines distortion, target RMS <0.3 px) → **EXIF** focal bootstrap → **GeoCalib** per-clip (unknown phone, works on empty court frames). Do **not** self-calibrate radial distortion from court lines alone — degenerate.

**Auto court-keypoint detector:** fine-tune **TennisCourtDetector** (pickleball geometry is a strict subset of tennis; only the NVZ/kitchen line + centerline are new) → pickleball. Architecture: HRNetV2-W48 heatmap (highest accuracy) or TrackNet-style (lighter). Loss: MSE + Adaptive Wing + a quadrilateral/rectangle-consistency term. Recipe: pretrain on **synthetic court renders across 50–500 viewpoints** (height 1–4 m, tilt 10–80°, focal 28–90 mm-eq, randomized colors/shadows/glare/occluded corners — labels are free since geometry is known) → train on tennis (8.8k + Roboflow) → fine-tune on ~200–500 hand-labeled pickleball frames spanning our viewpoint range. Sub-pixel heatmap decode (DSNT/parabolic).

**Feet/ball-to-world error budget** (validate on our clips; geometric estimates flagged):

| Viewpoint | Lateral (X) | Depth (Y) |
|---|---|---|
| High corner ~3–4 m, 45–60° | ~3–5 cm | ~12 cm |
| Mid sideline ~2 m, 30° | ~6 cm | ~20 cm |
| Low baseline ~1.5 m, 20° | ~10 cm | **~60 cm far court** |

NVZ/kitchen calls run along the depth axis → most viewpoint-sensitive. The **capture-quality score** (tilt, keypoint count/spread, heatmap confidence, post-solve reprojection RMSE, depth-compression ratio → green/yellow/red) steers users to better setups and confidence-gates far-court foot calls. **Drift:** monitor per-frame reprojection RMSE + optical-flow consistency; optical-flow warp through micro-vibration, full re-detect on confirmed bump.

---

## 4. Person detection / tracking accuracy

Court geometry does most of the work here, so accuracy effort is small (we're tracking-limited, not detection-limited, for 2–4 large players). **Detector (offline/server) = YOLO26m + BoT-SORT-ReID (`osnet_ain_x1_0`)** (YOLO26 is real — Ultralytics, Jan 2026, NMS-free, +1.4–2.8 mAP over YOLO11 at equal GPU latency; tune with official defaults). The **live on-device** detector is **YOLO26n** (Core ML, fallback YOLO11n) for preview/guidance only — the offline tier recomputes with YOLO26m. ByteTrack = simpler fallback; RF-DETR-L (Apache) = runner-up; candidates 26l/26x if far-court small-player recall is weak (see Registry §2.3). Levers: fine-tune the detector on ~500 auto-labeled court frames (verify ~50); **court-polygon foot-point filter** (rejects spectators before the tracker); **ground-plane association** (reject "teleports"/net-crossing swaps in court meters, not pixels); N-lock (court-position init + side priors) + a one-tap coach anchor to lock the 4 players. BoT-SORT's ReID resolves identity through doubles crossings; OSNet only if outfits clash.

---

## 5. 3D body accuracy (keystone) — mesh-core

**Round-4b architecture (supersedes skeleton-first AND the GVHMR-primary call):** the core body representation is now **world-grounded SMPL/SMPL-X mesh**, because a physics-accurate watchable replay and foot-skate elimination both require a volumetric, contact-aware body (`TECH_STACK.md` §(e),(p); `IMPLEMENTATION_PHASES.md` Phase 3). Mapping:
- **Deep/replay tier — per-frame backbone: Fast SAM-3D-Body** (best per-frame mesh available — 3DPW PA-MPJPE **30.4 mm**, beats HMR2.0 by 15+ mm; user-confirmed; SAM License — research-OK now, verify before commercial, SAT-HMR/Apache is the license-safe fallback; ~15 FPS/person, ~4–6 FPS for 4 batched crops — speeds are estimates, benchmark on H100). MHR→SMPL via its built-in MLP. **World-grounding is OURS, not GVHMR's:** project per-frame output to world via known K,[R|t] + court Z=0 → temporal smooth → foot-lock (§5c) → physics (precedent arXiv 2512.21573). **GVHMR/WHAM are demoted to optional world-trajectory / foot-velocity sanity-check references only** — their per-frame mesh is HMR2.0-based (weaker), and the world-grounding they add is exactly what our calibration already gives us. **PromptHMR is dropped** (full-image hurts distant players; needs manual prompts; worst translation error on CalTennis).
- **Fast tier — SAT-HMR (~24 FPS) or Multi-HMR 2 (~20 FPS)** camera-space multi-person SMPL — **prefer Multi-HMR 2** (no fixed-FOV assumption; SAT-HMR assumes ~60° FOV which our variable cameras violate). ~7–8 mm penalty vs SAM-3D-Body, acceptable for a preview. The positional skeleton (RTMW3D/MotionBERT) is now only a preview overlay + sub-frame contact trigger.
- Multi-person: no single robust monocular ≤4-person world pipeline → run **per-player** SAM-3D-Body on tracker crops, then foot-lock each (court plane gives per-player metric scale).

**Why our grounding beats the field:** on the CalTennis real-tennis benchmark *every* monocular method shows **0.9–3.6 m translation error** (WHAM worst at ~2.66 m) because they estimate depth from appearance. We don't — we project through known K,[R|t] onto the court plane, so our world translation error is bounded by calibration (sub-foot per §3), not by monocular depth guessing. This is precisely the failure mode our setup already eliminates.

The accuracy levers below still apply — and now feed the SAM-3D-Body/SMPL mesh, not a bare skeleton. The three highest-leverage moves, in order:

1. **Sports-domain fine-tune → ~70% in-domain MPJPE reduction (measured: 214→65 mm).** Fine-tune the lifter (backbone LR 0.1× head) + the 2D stage on **ASPset-510 (CC0)** + **AthletePose3D** + **AthleticsPose** + **SportsPose**. Convert to H36M-17 (lifter) / COCO-WholeBody (per-frame). This single step is the biggest win.
2. **Pseudo-label our own footage and distill** — the only way to cover paddle/wrist kinematics (no public 3D racket dataset exists). Run a heavy oracle (SMPLer-X / WHAM, or the 2-camera multi-view rig of Section 7) over 5–10 h of footage → **filter by reprojection error <8 px through our known camera** → active-learn the ~500 hardest frames (swing peaks, kitchen approaches) for human label → re-fine-tune the fast model.
3. **Exploit the calibrated court + ground plane:**
   - **Reprojection loss** `‖π(K,[R|t],J₃D) − J₂D‖` (start 0.1× pose loss) — forces 3D to agree with the 2D observation under the exact known camera.
   - **Metric root depth** from court PnP + ground-plane homography — fixes the single-view scale/depth ambiguity that dominates absolute error.
   - **Hard foot-on-ground constraint** — court is Y=0; when foot height <2 cm and the contact classifier fires, snap foot Y to plane and zero its world velocity. Kills foot-slide and nails kitchen-foot (residual ≈ ankle keypoint accuracy, ±3–5 cm).

**Keypoint set:** train per-frame toward **COCO-WholeBody / H3WB-133** (body-17 incl. **bilateral hips+shoulders** for X-factor, **hands** for paddle-face, **feet** for kitchen/weight-transfer). **Lift only the 17 body joints temporally** (MotionBERT 243-frame window — validated); run a dedicated hand estimator (RTMPose-Hand / MediaPipe Hands) in the contact window for paddle-face. Do not lift 133 joints temporally (unvalidated).

**Foot-contact:** small MLP on `[foot_y_world, |v_foot_world|, ankle_z]` over a 5-frame window; bootstrap labels from AMASS zero-velocity threshold, warm-start from UnderPressure; combine with the Y=0 hard constraint.

**Synthetic variable-viewpoint augmentation (conditional — only if testing shows angle-specific failures):** render AMASS racket-adjacent motions (reach/lunge/throw, optionally seeded with SMPLify-X fits of our clips) in Blender/BEDLAM on synthetic courts; domain-randomize camera pitch [5°,45°]/yaw [0°,30°]/FOV, court colors, clothing, HDRI, 1–4 players; **post-process every synthetic frame with our real pathologies (H.264, RS, flicker)**; mix ~20–30% synthetic.

**Expected MPJPE progression** (validate per stage; italics extrapolated): baseline ~150–200 mm → **+sports fine-tune ~50–70 mm (measured)** → +ground-plane/reprojection/root ~40–60 mm → +pseudo-label own footage ~30–45 mm → +synthetic ~25–40 mm → +temporal smoothing ~22–38 mm → +SMPLify bundle (offline highlight bursts only) ~15–25 mm.

---

## 5b. Velocity / kinematics — conflict resolved, claim discipline

The R²≈0.96 vs r≈0.11–0.28 "conflict" is a **measurement-type mismatch, not a contradiction:**
- R²≈0.96 = a model predicting **ball speed** from **2D in-plane endpoint speed** (raw wrist↔ball is only r≈0.50–0.70). Build on this.
- r≈0.11–0.28 = derivative of **noisy lifted-3D joint positions** vs mocap. Don't build absolute claims on this.

**Velocity taxonomy:** (A) 2D resultant endpoint speed = reliable in-plane; (B) derivative of lifted-3D = unreliable; (C) 3D angular velocity = marginal (accel hopeless); (D) **calibrated 2D→world-plane speed = reliable and metric — build this.**

**Pipeline (offline default):** confidence-gate keypoints → outlier-reject (median, win 3–5) → **court-homography projection to metric world plane** → **RTS/Kalman smoother** → central difference → **4th-order zero-phase Butterworth (`filtfilt`)** at **6 Hz body/timing, 8–10 Hz swing**; or **Savitzky-Golay (order 2, win 9)** which differentiates + smooths and preserves peaks better. **Real-time:** One-Euro filter (β≈0.007, f_c_min≈1.0 Hz) — adaptive cutoff preserves the swing peak. Set cutoffs by residual analysis (Winter's method), higher for wrist than trunk.

**Hard limit:** ball-racket contact lasts ~2–8 ms (<1 frame even at 60 fps); any 6–10 Hz filter erases it. **You cannot measure contact-frame peak speed from pose — use the ball for that.** The recoverable signal is the pre-contact velocity ramp (200–300 ms) at 8–10 Hz / 120 fps.

**Frame rate:** **require 120 fps for any absolute swing-speed number;** 60 fps floor for timing/tempo/weight-transfer; fast shutter ≥1/500 s regardless.

**Biggest lever (free):** court-homography → metric in-plane velocity converts the unreliable Type-B (r≈0.28) into Type-D (r≈0.70+) for near-court events.

**Claim tiers:**

| Tier | Metrics | Action |
|---|---|---|
| **1 — claim with number** (after validation) | wrist/elbow swing speed as ball-speed predictor (±15–25%), horizontal CoM velocity (weight transfer), split-step timing (±17 ms@60 / ±8 ms@120), tempo/swing duration | ship after Protocol A/B pass |
| **2 — qualifier** ("relative/estimated") | swing-speed *index* (within-session ranking), 3D angular velocity (trend only) | ship as relative |
| **3 — never** | 3D angular **acceleration**, depth-axis velocity, contact-frame peak from pose | gate out of app |

---

## 5c. Foot-skate & physics accuracy

Full design/build in `IMPLEMENTATION_PHASES.md` Phase 4 (`TECH_STACK.md` §(p),(q)). Accuracy summary:

- **Foot-skate killer (do regardless of physics tier):** contact = (foot height above Z=0 < 2–3 cm) AND (world speed < 1 cm/frame) AND (pose-confidence), with on/off **hysteresis**; on contact, **snap stance toe/heel to the exact Z=0 court plane + zero its world velocity → CCD-IK** for the leg, blend at edges. Because the plane is *exact* (our calibration), this reaches **≤3 mm foot-slide and zero floor penetration — beating published world-HMR (3.0–4.4 mm)** that has to estimate the ground. Near real-time.
- **Foot-contact classifier** (if thresholds prove noisy in fast lunges): small MLP on `[foot_y_world, |v_foot_world|, ankle_z]`, trained on **RICH** (vertex-contact labels) + **UnderPressure** (vGRF→contact, ~95% vs ~81% thresholds). On **Tier-A (Pro + LiDAR)** devices, near-camera depth (≤~5 m, indoor/shade) gives a direct foot-to-floor distance for the *near* player — use it as extra supervision / a confidence boost for near-court contact; it does **not** reach the far player and **fails in direct sun**, so vision + the Z=0 constraint remain the baseline.
- **Physics refinement:** **PhysPT** (MIT, no engine at inference — −68.7% skate, −83.8% accel) as the default plausibility post-processor; **PHC/PULSE on MuJoCo+MJX** (from SMPLOlympics tennis envs) for the flagship deep replay; **MultiPhys** for inter-player non-penetration in doubles. Output is physically valid by construction (can't float/skate/penetrate).

## 5d. Racket 6DoF accuracy (paddle-face flips to claimable)

Full design/build in `IMPLEMENTATION_PHASES.md` Phase 6 (`TECH_STACK.md` §(r)). **The user's hypothesis is validated: tracking the paddle directly beats inferring face angle from the wrist by ~5–15×.**

- **Monocular PnP-IPPE + known paddle dimensions + clean face corners → face-angle ~3–5°**, vs **20–35°** for markerless wrist pronation (axial twist is geometrically unobservable from wrist keypoints). **Contact-point-on-face ±1–3 cm** (dominated by *ball* 3D localization, not paddle pose).
- **Pipeline:** RTMDet-m + SAM2.1-hiera-base-plus silhouette → RTMPose-m **top/bottom/handle** keypoints (side keypoints unreliable, skip) + convex-hull corners → coarse pose (GigaPose/FoundPose + **hand-grip prior** from wrist+middle-finger MCP) → **PnP-IPPE** on known 3D corners → **UKF on SE(3)** (blur streak = rotation cue; predict through occlusion) → physics-validate vs ball rebound.
- **Training:** fine-tune RTMPose on RacketVision (tennis) + Roboflow pickleball + **~50k synthetic paddle-CAD frames (BlenderProc, free 6DoF+corner GT)**; **~5k real frames with ArUco markers on the paddle back** for real 6DoF ground truth (train, remove markers for inference). ≥60 fps (120 preferred — at 30 fps a ~1000°/s swing rotates ~33°/frame and the SE(3) filter diverges).

**Defensible-envelope change:** **wrist-only pronation stays Tier-3 (never claim)**, but **racket-tracked paddle-face angle and contact-point are now Tier-1 (claim with number, after validation against ArUco GT).** This is novel — current systems (e.g. LATTE-MV) use the hand center as the contact proxy *because nobody estimates racket pose*.

---

## 6. Ball tracking accuracy

**Build-now baseline: TrackNetV3** (MIT, public, F1 ~0.986 badminton, heatmap + 8-frame + trajectory rectification). **Upgrade to TrackNetV5 when its repo releases** (best F1, ~114 FPS T4, −74% false negatives vs V4 under occlusion) — do not block on it. **Transfer hierarchy:** badminton weights → fine-tune on tennis + table-tennis (~155k frames) → fine-tune on pickleball.

Ranked levers: TrackNetV5 upgrade · multi-dataset pretrain → pickleball fine-tune · **BlurBall midpoint labeling** (+1.2% F1; label ball at blur-streak center) · **TOTNet visibility-weighted loss + occlusion augmentation** (occluded-frame acc 0.63→0.80) · background-subtraction concat (compute per-clip median) · trajectory rectification/inpainting · 60→120 fps · physics post-processing.

**Augmentation tuned to our inputs (ordered):** motion-blur synthesis (kernel length 3–20 px along ball direction) · background concat · mixup (α 0.5, preserves trajectory) · **color jitter hue ±30°** (yellow/green/orange balls) · **JPEG/H.264 compression-artifact injection** · copy-paste ball patches · fps simulation · indoor/outdoor color cast. **Geometric transforms must be identical across the 8-frame sequence; color transforms may vary per frame.**

**Physics post-processing (CPU, real-time):** EKF `[x,y,vx,vy]` with gravity → RANSAC parabola fit (reject >5 px off-arc, fill gaps) → **bounce = vertical-velocity sign change** (sub-frame via parabola interpolation → ±5–15 ms even at 30 fps) → **net crossing via homography** → rough 3D via physics + "Z=0 at bounce." Cuts false positives to <5%.

**Skip SAHI** (incompatible with multi-frame heatmaps) and super-resolution (overkill at 1080p). **Throughput:** NVDEC GPU decode + batch 8–16 → ~12 min per 20-min 120fps clip on a 4090-class GPU. **Expected after full pipeline:** F1 ~0.90–0.95 (extrapolated — validate on a 2k-frame held-out set).

---

## 7. Contact / audio events accuracy

**Audio is the sub-frame timing anchor (~0.09–4 ms, beats any video fps).** Two-stage: (1) **onset/peak detector** → timestamp; (2) small **CNN on mel-spectrogram** → event type. Never let the classifier window drive timing.

**Mandatory distance-delay correction:** shift audio back by `d/343 s` (up to ~30 ms / a full frame at 10 m), with `d` from homography player position. Skipping this injects a full-frame bias.

**Collect our own dataset (none exists):** 2,000–5,000 verified pickleball "pops" at **44.1 kHz WAV** (never AAC — energy to 8–12 kHz). Diversity: indoor/outdoor, distances 1–10+ m, paddle materials, power levels, doubles (<1 s successive hits). Negatives 2–3×: footsteps, wind, claps, ball-on-court bounce, paddle-tap, speech. Features: 64-mel, FFT 512, hop 256; 100 ms window centered on contact; 10 kHz high-pass. Model: 6-layer 2D-CNN (fast, ~0.96 F1) or BEATs/AST encoder for headroom. **Augmentation:** background noise SNR 0–20 dB (MUSAN/ESC-50/FSD50K/DEMAND) + RIR convolution (pyroomacoustics — free distance/reverb invariance) + SpecAugment + mixup + pitch/time/amplitude → ~20k from 2k.

**Multimodal fusion:** audio (WHEN) + visual event head T-DEED/ASTRM (WHICH event) + wrist-velocity peak + ball-trajectory inflection (confirm + WHICH player). Start audio-primary late fusion → upgrade to cross-attention (ASTRA / RacketVision). **Doubles attribution:** ball-to-wrist proximity (court coords) + pre-contact trajectory vector; **single-mic DOA does not work.**

**Auto-label loop:** audio onset → keep where ball-trajectory inflection agrees within ±2–3 frames → human-verify ~10–15% (disagreements + ambiguous) → **Mean-Teacher semi-supervision** (DCASE'24 Task 4 baseline, swap head to contact/bounce/background) → active learning (matches full supervision at 1/3 labels). **Visual pretrain:** OpenTTGames (120 fps, frame-accurate bounce/net, no audio). **Net crossing / serve:** homography net-line + ball-above-net-height (approximate, single-cam); serve contact = same model, phase-gated by court position + arm extension.

---

## 8. Shot classification

No published pickleball classifier exists → build the dataset from Phase-1 footage with **audio-snapped contact + pose-trajectory** labels (~2,000 clips, 8 classes: FH/BH drive, dink, lob, overhead, third-shot drop, serve, reset/block). Architecture: adapt **BST (Badminton Stroke Transformer)** — TCN→transformer with cross-attention between pose latent (Q) and ball-trajectory latent (K,V), 77% acc / 93% top-2 on ShuttleSet — or **PoseConv3D** (robust to pose noise) / ST-GCN. ML produces the shot **label only**; all coaching metrics stay rule-based + confidence-gated. **Drill rep counting:** wrist-velocity peak + confirmed contact event + Markov state machine (ready→windup→contact→follow-through); ~±1 rep.

---

## 9. Datasets to download (master list)

**Licensing is informational now, not a gate** (we are not selling yet — recorded for future commercialization). So AMASS / BEDLAM / BEDLAM2 / AthletePose3D / AthleticsPose / SMPL / SMPL-X / CalTennis (and GVHMR/WHAM, used only as trajectory references) are all **usable today**. Body-model **fine-tune/adapt order (round-4b): BEDLAM2 → AthletePose3D → CalTennis → RICH → AMASS** (priors), adapting the **Fast SAM-3D-Body** pipeline; eval world-MPJPE on EMDB2 + CalTennis. ASPset-510 (CC0) and SportsPose remain useful sports sources. **Verify-before-commit** (very recent, confirm repo/license/numbers): CalTennis, BEDLAM2, SAT-HMR, OnlineHMR, WATCH.

| Component | Dataset | Size | License | URL |
|---|---|---|---|---|
| 3D pose (sports) | **ASPset-510** | ~330k frames | **CC0** | archive.org/details/aspset510 |
| 3D pose (sports) | AthletePose3D | 1.3M frames, 60–120 fps | research | github.com/calvinyeungck/AthletePose3D |
| 3D pose (sports) | AthleticsPose | ~500k frames | see repo | github.com/SZucchini/AthleticsPose |
| 3D pose (sports) | SportsPose | 176k+ poses | see repo | github.com/ChristianIngwersen/SportsPose |
| 3D pose (base) | Human3.6M, MPI-INF-3DHP | 3.6M / 1.3M | research | vision.imar.ro/human3.6m · vcai.mpi-inf.mpg.de/3dhp-dataset |
| 3D whole-body target | H3WB | 100k images, 133 kpts | as H36M | github.com/wholebody3d/wholebody3d |
| Motion prior / synthetic seed | AMASS | 40+ h SMPL | research | amass.is.tue.mpg.de |
| Synthetic (clothed, base) | BEDLAM / **BEDLAM2** (has shoes → aids contact) | 8M+ frames | research | bedlam2.is.tuebingen.mpg.de · *verify BEDLAM2* |
| **Tennis 3D (domain match)** | **CalTennis** (11M frames, 51 h, 60 Hz, **multi-view GT**) | 11M frames | research (IRB) | *verify repo/license* |
| Foot/vertex **contact labels** | **RICH** | 90k SMPL-X + contact | research | virtualhumans.mpi-inf.mpg.de/RICH |
| Physics envs (tennis/TT) | **SMPLOlympics** (MuJoCo+SMPL) | — | research | github.com/SMPLOlympics/SMPLOlympics |
| Racket/paddle pose | RacketVision (tennis subset) + Roboflow paddle box sets + **synthetic paddle-CAD (BlenderProc, ~50k)** + **ArUco-GT real (~5k)** | ~55k+ | mixed / ours | github.com/OrcustD/RacketVision |
| In-the-wild eval (phone) | EMDB, 3DPW, AGORA, Fit3D | — | research | eth-ait.github.io/emdb · etc. |
| Ball (pickleball) | InAPickle_Core_Tracking | 18,055 | CC BY 4.0 | roboflow InAPickle |
| Ball (pickleball) | ball-tracking-1wfuq | 11,096 | CC BY 4.0 | roboflow hu-space |
| Ball (tennis, native) | TrackNet V1/V2 tennis | 36,962 frames | research | nol.cs.nctu.edu.tw |
| Ball (table tennis, 120 fps) | OpenTTGames | 55,582 frames | educational | lab.osai.ai |
| Ball (blur) | BlurBall TT | 64,119 frames | CVPR-W | github.com/cogsys-tuebingen/blurball |
| Ball (badminton, pretrain) | Shuttlecock Trajectory (TrackNetV3) | ~200k frames | research | github.com/qaz812345/TrackNetV3 |
| Ball/racket multi-sport | RacketVision | 435k frames | AAAI'26 (check) | github.com/OrcustD/RacketVision |
| Court | TennisCourtDetector | 8,841 imgs | **none stated** | github.com/yastrebksv/TennisCourtDetector |
| Court (pickleball, fine-tune) | Ping Pong Paddle AI court kpts, pickle-court-keypoints | ~300 each | CC BY 4.0 | roboflow |
| Audio noise (aug) | MUSAN, ESC-50, FSD50K, DEMAND | — | mixed | respective sites |
| Shot class (tennis) | THETIS; Tennis Player Actions | — | research | mendeley/edu |

---

## 10. Validation protocols (these gate the phases)

Use **ICC + Bland-Altman (bias + LoA)**, not just Pearson r.

| Component | Protocol | Pass gate |
|---|---|---|
| Court calibration | tap/auto-detect on a clip matrix spanning low/high/steep/shallow viewpoints; reprojection RMSE + visual overlay | overlay matches on ≥8/10 across ≥4 distinct viewpoints; median reprojection < threshold; feet-to-world within budget (§3) |
| 3D pose | hand-label ~50 frames/stage as truth set; MPJPE per stage; **Protocol D**: run pipeline on AthletePose3D public subset | each stage meets MPJPE target (§5); wrist-velocity r ≥0.50 vs AthletePose3D GT |
| **Velocity (A)** ball-speed GT | 30–50 swings/stroke @120 fps; radar or optical ball speed; wrist speed in [contact−5, contact−2] | **r≥0.70 → ship Tier-1 ±20%; r≥0.55 → Tier-2 relative** |
| **Velocity (B)** timing | 20 split-steps + 20 returns; 2 manual annotators | MAE ≤2 frames (≤33 ms@60 / ≤17 ms@120); tempo ICC ≥0.95 |
| **Velocity (C)** fps floor | record @240, downsample to 120/60, same filter | peak underestimate ≤15% and r≥0.90 vs 240 reference |
| Ball tracking | held-out 2,000+ diverse pickleball frames | P/R @10 px; false-positive <5% after physics filter; F1 target 0.90–0.95 |
| Contact/audio | labeled contacts; compare fused timestamp to manual | within ±2 frames (or ±4 ms with audio) |
| Foot/NVZ | manual in/out/near review on confident frames | reviewer agreement ≥80% |
| **Foot-skate** | per-foot world displacement during detected contact windows | **≤3 mm foot-slide** |
| **Penetration** | min foot-vertex Z; mesh-mesh intersection (doubles) | **0 floor penetration; no visible inter-player penetration** |
| **Racket face-angle** | vs ArUco-marker 6DoF GT | **≤5°** |
| **Racket contact-point-on-face** | vs multi-cam GT (ball-3D-limited) | **±1–3 cm** |
| **Replay realism** | coach "looks right" review of the 3D replay | qualitative pass before ship |

Test-set sizing for velocity: ≥3 players × ≥5 strokes × ≥10 reps = **≥150 clips**, slow (dink) to fast (smash).

---

## 11. Unified data-collection + auto-labeling plan

**Phase 0 — public seed (wks 1–4):** TennisCourtDetector + ~300 PB court labels; COCO person + ~500 PB frames; TrackNet ball weights → pickleball fine-tune; sports-pose fine-tune (Section 5, lever 1).

**Phase 1 — unified 2-camera shoot (wks 5–8, the keystone) — *training-data collection only, not the product capture path*:** **one calibrated 2-iPhone session auto-labels nearly every component at once.** 2 phones ~90° apart (cam A baseline-corner = primary/product viewpoint; cam B side-fence = multi-view training signal), auto-calibrated from court keypoints, 60 fps/1080p, locked exposure, ~10 h, 5–8 players, indoor+outdoor. Run Phase-0 heavy models → ~10k auto-verified frames across court/person/pose/ball/audio.

**Auto-labeling loop (all components):** heavy teachers (DINO/YOLO-L person, **Fast SAM-3D-Body** mesh (+ optional multi-view), TrackNetV5 full-res ball, audio onset) → **confidence filter** (court reproj <τ, pose conf + temporal smoothness, ball physics-consistent parabolas only) → **physics check** (ballistic arc, joint limits/continuity) → versioned pseudo-label DB → **active learning** (uncertainty + embedding-diversity) routes only hard/diverse frames to CVAT → **distill into fast students**. Audio (44 kHz, ~0.02 ms) snaps contacts to exact frames, bootstrapping shot labels.

**Prototype wave note (2026-06-27):** the current lowered `PROTOTYPE-GATE` calibration seed set is the four accepted static/user-realistic clips named in `BUILD_CHECKLIST.md §6`: `burlington_gold_0300_low_steep_corner`, `wolverine_mixed_0200_mid_steep_corner`, `outdoor_webcam_iynbd_1500_long_high_baseline`, and `indoor_doubles_fwuks_0500_long_mid_baseline`. Broadcast clips with moving cameras and 360 clips are excluded, and `side_view_game5_0100_high_side_fence` is `DEFERRED_REJECTED_SIDE_FISHEYE` for this wave. The reviewed court corners live under `runs/eval0/prototype_gate_h100_v2/review_bundle/corrections_accepted_static4/<clip>/court_corners.json`. Import them with `scripts/racketsport/import_cvat_labels.py`, then run `scripts/racketsport/build_calibration_from_review.py` to create sidecar/calibration artifacts. These corrected corners are `corrected_unverified` prototype seeds, not ground truth and not full DATA-1 readiness.

**TrackNet runtime note (2026-06-27):** TrackNetV3 and InpaintNet checkpoint files are verified on the H100, and a real 2-second Burlington smoke run now produces a schema-valid `ball_track.json` via `scripts/racketsport/run_tracknet_ball.py`. This is a runtime smoke only. TrackNet's official CSV exposes `Frame,Visibility,X,Y`, not calibrated probability, so confidence is recorded as visibility-derived `1.0/0.0` until a real heatmap-confidence adapter is built. Phase 5 still requires labeled ball/event ground truth, contact-window generation, and the documented F1/timing gates.

**Tools:** FiftyOne (dataset viz + active learning), CVAT (video keypoint/track annotation with interpolation), Roboflow (versioning + SAM/DINO auto-label + train CI), Label Studio (audio+video multimodal QA). Expected: ~50% fewer labeled samples to target accuracy; combined 3–5× fewer manual hours.

---

## 12. The moat — multi-view-to-train-monocular (training-time only)

**This is a TRAINING-TIME technique, not a product feature: the shipped product is single-camera.** Use 2 phones at training/labeling time only to manufacture 3D ground truth + a cross-view consistency signal; **ship a single-camera model (zero user friction).** Validated lineage: EpipolarPose / CanonPose (self-supervised, uncalibrated, cross-view); **"Two Views Are Better Than One" (CVPR'25-W): 2 synced views + Procrustes consistency loss at train time, no 3D labels, MPJPE 64.4→36.2 mm (−43.6%) on SportsPose, 90°-apart optimal**; CalTennis protocol (2–6 iPhones, $40 tripods, court-keypoint auto-calibration). Plan: 3–5 sessions, 2 phones 90° apart, auto-calibrated → train the monocular pose model with two-view consistency loss (target ~40% MPJPE reduction) → deploy single-camera. **CalTennis already ships multi-view GT (11M frames) we can train/eval against directly**, so our own 2-camera shoot is additive (pickleball-specific motions, our viewpoints) rather than the only source. **Product angle (FUTURE, not v1):** a live 2nd-camera toggle for true 3D triangulation (graceful 1→2 scaling via the same auto-calibration) is a future power-user feature — the v1 product is single-camera only.

---

## 13. Synthetic data — where it pays off

ROI ranking: **ball ★★★★★ > court ★★★★ ≈ pose ★★★★ > person ★★.** Ball: render a 6.7 cm holed pickleball along physics trajectories with randomized altitude/azimuth/court-texture/lighting/distance/spin and explicit motion blur (BlurBall) — free pixel-perfect labels for the rarest data. Court: render top-down → project to target extrinsics, vary line wear/shadow/tilt/distortion (10–50k in a weekend). Pose: BEDLAM pipeline + SportsPose/AthletePose3D motions on court backgrounds. **Always post-process synthetic frames with our real pathologies (H.264, RS, flicker, lens distortion); mix ~30–40% synthetic** (more regresses real-world accuracy).

---

## 14. MLOps / continuous-improvement flywheel

- **Versioning:** DVC (datasets-with-code) + Roboflow Versions (snapshots tied to runs) + W&B (experiments + model registry).
- **Eval harness:** held-out 500 clips across all 4 placements × indoor/outdoor × skill. Per-version metrics — Court PCK@10px; Person mAP@50/HOTA; Ball P/R@10px; Pose per-joint MPJPE; Velocity per Section 10; Shot per-class F1.
- **Regression CI:** GitHub Actions → DVC pull test set → infer → compare to baseline; **block merge if any component drops >2%.**
- **Corrections flywheel (the moat):** every in-app user/coach correction ("this was a lob, not a drive"; dragged ball/foot position) logs clip + predicted-vs-corrected into a corrections queue = high-value hard negatives → FiftyOne active-learning prioritization → CVAT verification → next training batch → fewer future corrections. **Retrain triggered** when a shot-class correction rate spikes >3%; **scheduled** full retrain every 4–6 weeks.

---

## 15. Priority order

1. **Capture spec + in-app enforcement** — zero ML cost, biggest immediate lift.
2. **Court calibration** (solvePnP + multi-frame + capture-quality score) — everything downstream depends on it.
3. **World-grounded SMPL mesh** — **Fast SAM-3D-Body** per player (best per-frame mesh) + our own grounding to the court plane; sports adapt (BEDLAM2 → AthletePose3D → CalTennis → RICH → AMASS). GVHMR only as a trajectory cross-check.
4. **Court-geometry constraints** (reprojection loss, metric root, foot-on-ground) — free accuracy.
5. **Foot-skate killer + physics** (contact-vs-Z=0 lock + CCD-IK → PhysPT) — the watchable, skate-free replay (`IMPLEMENTATION_PHASES.md` Phase 4).
6. **Ball (TrackNetV3→V5) + audio contact model** — enables event-triggered compute + scene physics.
7. **Racket 6DoF** — paddle-face + contact-point (flips paddle-face to claimable).
8. **Velocity pipeline + Protocol A/B/C validation** — turn on defensible velocity claims.
9. **Unified 2-camera shoot + auto-labeling loop; multi-view→monocular distillation** — manufacture ground truth; the accuracy moat.
10. **Shot classification + corrections flywheel** — durable, compounding moat.
