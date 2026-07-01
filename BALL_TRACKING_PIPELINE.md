# Ball-Tracking Pipeline — Master Design (Ball ONLY)

> Scope: this document covers **only the ball** — in‑air 2D tracking, bounce/landing,
> in/out calls, net‑crossing + contact events, 3D trajectory, spin, and speed.
> Players, pose, racket 6DoF, and coaching insights are **out of scope here**.
> This is the authoritative spec. A companion `BALL_TRACKING_GOAL_PROMPT.md`
> contains a ~4k‑character prompt to hand to a build agent.

---

## 0. Decisions locked (from requirements Q&A)

| Axis | Decision |
|---|---|
| Deliverable | Detailed spec (this doc) + 4k‑char agent prompt + plain summary |
| Runtime | **Hybrid** — fast on‑device (iPhone, live) **+** high‑accuracy offline (server) |
| Scope | Everything feasible: 2D track, bounce, in/out, events, 3D, spin — split across tiers |
| Camera | **Single phone on a tripod** (enforced capture protocol) |
| In/out bar | **Confident calls + explicit "too‑close‑to‑call" gray zone** (never fake precision) |
| Priority | **On‑device = max speed; offline = max accuracy** (both must stay reasonable) |
| Data | Public pickleball Roboflow (~50k) + transfer learning + optional self‑labeling |
| Licensing | Research/personal OK now (commercial caveats noted inline) |
| Capture control | **Yes** — a strict capture protocol is enforced |
| Kill‑list (top failures) | **Missed fast/blurry balls · ghost/false detections · jitter** |

**Non‑negotiable honesty rule:** single camera cannot produce officiating‑grade
line calls, especially on the **far** half of the court. We surface a call **only**
when the computed geometric margin exceeds the physically‑modeled uncertainty for
that specific bounce; otherwise we return `too_close_to_call`. This is the same bar
the leading pickleball product (PB Vision) respects — it reports in/out as a *stat*,
not as officiating.

---

## 1. Why this architecture (grounded in 2026 SOTA)

- **Detector family:** heatmap + temporal (TrackNet / WASB) — NOT single‑frame object
  detectors. A fast pickleball is small and motion‑blurred; to a YOLO/RT‑DETR box it
  reads as background. Every credible 2025–2026 result (TrackNetV3, TrackNetV4 motion
  attention, WASB, RacketVision) is heatmap+temporal.
- **Model facts we rely on:**
  - **WASB** — 1.5M params, >30 FPS, best AP across 5 sports; small enough for mobile.
  - **TrackNetV3** — F1 ~0.986 (badminton), 25 FPS base, ~75 FPS un‑ensembled,
    **~130 FPS with INT8 TensorRT**; ships trajectory rectification (InpaintNet).
  - **BlurBall** — emits blur length/angle → a free velocity proxy (helps fast balls).
  - **TrackNetV4** — plug‑in motion‑attention on V2/V3 (small gain, helps occlusion).
- **Bounce → in/out (single cam):** the proven recipe is
  `detector → trajectory → LEARNED bounce classifier (features: vertical vel/accel) →
  homography → in/out`, fused with an **audio "pop"** for sub‑frame timing. The
  z‑local‑minimum method is not reliable single‑camera and is replaced.
- **3D single cam:** meter‑level, context only — resolved per‑segment via a physics ODE
  anchored by the **"Z=0 at bounce"** constraint. Never used for line calls.

---

## 2. System overview

```
                 ┌──────────────────────────────────────────────┐
   iPhone (live)  │  TIER A — ON‑DEVICE (max speed)              │
                 │  capture → tiny heatmap tracker (CoreML/ANE) │
                 │  → 2D (x,y,conf) + Kalman smoothing          │
                 │  → live trail + rally on/off + QA feedback   │
                 └───────────────┬──────────────────────────────┘
                                 │ upload video + capture sidecar + rally spans
                                 ▼
                 ┌──────────────────────────────────────────────┐
   Server (GPU)   │  TIER B — OFFLINE (max accuracy)             │
                 │  decode → fine‑tuned TrackNetV3(+WASB verify) │
                 │  → heatmap‑confidence 2D track                │
                 │  → RANSAC + Kalman/RTS smoother + ghost kill  │
                 │  → court calib → learned bounce (+audio)      │
                 │  → in/out (uncertainty‑gated, gray zone)      │
                 │  → 3D physics uplift → spin, speed            │
                 │  → net‑cross + contact events                 │
                 └──────────────────────────────────────────────┘
```

Tier A gives instant feedback and **shrinks offline compute** (only rally spans get
the expensive pass). Tier B is the source of truth for every ball metric.

---

## 3. Capture protocol (ENFORCED — feeds everything)

These are hard requirements surfaced in‑app with a live "court lock" check.

| Rule | Value | Why |
|---|---|---|
| Mount | Tripod / fixed mount, **≥ 5 ft (1.5 m)**, higher is better | Camera motion breaks calibration; height cuts far‑court depth error |
| Position | Elevated **center or corner**, full court + all 4 corners + ~1 m margin | Homography needs corners; margin keeps out‑balls in frame |
| Resolution | **1080p minimum, 4K preferred** | 4K ~doubles far‑court ball pixels → better recall + localization |
| Frame rate | **60 fps default**; **120 fps "drive mode"** when lit | Halves motion blur; sharper bounce/contact timing |
| Shutter/exposure | Lock exposure; target **≤ 1/1000 s** (≥ 1/500 s floor) | Freezes the ball → the #1 fix for missed fast balls |
| HDR | **OFF** | HDR fusion smears fast motion |
| Video stabilization | **OFF** | Warps geometry frame‑to‑frame; corrupts homography |
| Focus / WB | **Locked** | Prevents hunting blur / color shifts mid‑rally |
| Ball & court | **Bright optic ball with high court contrast**; avoid yellow ball on tan/yellow court | Low contrast is the top real‑world accuracy killer |
| Audio | Record on‑device mic (mono ok) | Bounce/contact "pop" for sub‑frame timing |

Write these into a `capture_sidecar.json` (resolution, fps, exposure, intrinsics,
device, court‑lock pass/fail). Offline refuses or down‑grades confidence if the
protocol is violated (e.g., stabilization detected, corners missing).

---

## 4. TIER A — On‑device real‑time (max speed)

**Goal:** live ball overlay, rally detection, and capture QA at ≥ capture FPS. It is a
*preview and gating* tier — it is explicitly **not** trusted for in/out or 3D.

### 4.1 Model
- **Tiny heatmap tracker**, 3‑frame input, INT8, CoreML, running on the **ANE**:
  - Primary: **WASB‑lite (~1.5M)** or **ResTrackNetV2 (~1.2M)**.
  - Alternative: **distill TrackNetV3 → a nano student** (teacher = the offline model).
- Input **288×512** (drop to 256×256 only if FPS target missed). RGB 3‑frame stack.
- Output: single‑ball heatmap → arg‑max + local centroid refinement → `(x, y, peak)`.

### 4.2 On‑device rules
- `visible = peak ≥ τ_on` (default **0.50**).
- **Motion gate:** reject a detection if the pixel jump from the predicted position
  exceeds `base_jump_px + v_max_px_per_frame`. Derive `v_max_px_per_frame` from a rough
  live scale (ball ≤ **~25 m/s**), fallback default **120 px/frame @1080p/60fps**.
- **Smoother:** constant‑velocity **2D Kalman**; predict through gaps ≤ **3 frames**;
  reset on rally boundary.
- **Rally on/off:** rally = ball visible & moving for ≥ **5 consecutive frames**; end
  after **≥ 0.8 s** with no valid ball. Emit `rally_spans[]` with ±0.5 s padding.

### 4.3 On‑device outputs
- `live_ball_overlay`: `(t, x, y, conf, visible)` + a short trail.
- `rally_spans[]` (drives offline compute).
- `capture_quality[]`: flags like `ball_low_contrast`, `court_not_locked`,
  `too_dark`, `stabilization_on` → shown to the user *before* they rely on results.

### 4.4 Performance target
- **≥ 30 FPS (aim 60)** on A17 Pro / A18 ANE at 288×512; per‑frame latency **< 33 ms**.
- Battery/thermal: cap live inference to the capture FPS; never run the offline stack live.

---

## 5. TIER B — Offline server (max accuracy)

### 5.1 Decode & span selection
- **NVDEC** GPU decode; process **rally spans only** (from Tier A, or audio‑onset
  gating as fallback) with ±0.5 s padding. Downscale to model input after decode.

### 5.2 Detection (2D)
- **Primary:** **TrackNetV3 fine‑tuned on pickleball**, multi‑frame (3 fast / 8 accurate),
  **288×512 base**, **512×896 "hi‑res" pass for far‑court** frames.
- **Background subtraction:** per‑clip median frame concatenated as input (WASB/TrackNet
  style) — big recall win on static‑camera footage.
- **Trajectory rectification:** InpaintNet (TrackNetV3 companion) to recover occluded
  spans.
- **Verifier / ensemble:** run **WASB fine‑tuned** in parallel; fuse (see 5.4). Two
  independent architectures agreeing is the strongest ghost filter.
- **Confidence = heatmap peak value (0..1)** — do **not** collapse to 0/1. Carry
  `conf` through the whole pipeline (needed for gray‑zones and gap trust).
- **Blur handling (fast‑ball fix):** optional **BlurBall** head → blur length/angle
  gives a velocity prior; also feeds local‑search recovery in 5.4.

### 5.3 Court calibration
- **Auto:** court‑keypoint heatmap net → line/corner detection → **`solvePnP` full pose**
  (not homography‑only). **Manual 4‑tap fallback** in‑app.
- **Gate:** projected‑corner reprojection **median < 8 px AND p95 < 15 px @1080p**.
  Below gate → calibration `low_confidence`, all downstream ball‑world outputs degraded.
- **Net plane** from regulation geometry (34" center / 36" posts; 22 ft net span),
  never estimated from pixels.

### 5.4 2D track post‑processing (this is where the 3 kill‑list failures die)
Order of operations per rally span:
1. **Multi‑model consensus (ghost kill):** keep a detection if primary & verifier agree
   within **R px** (default **60 px @1080p**, scale with resolution). Disagreements fall
   back to the higher‑confidence, motion‑consistent point.
2. **Court + margin gating (ghost kill):** drop detections outside the court polygon +
   **0.5 m** margin (projected). Removes crowd/logo/second‑ball hits.
3. **Max‑speed gate (ghost kill):** reject links implying world speed > **30 m/s** or
   pixel jump > `base_jump_px(60) + v_max`.
4. **RANSAC parabola per sub‑segment (fast‑ball recovery + ghost kill):** segment the
   track at contacts/bounces; fit constant‑accel arcs; reject residual **> 5 px**; then
   **local‑search recover** missed fast frames *along* the predicted arc with a lowered
   heatmap threshold (τ = **0.25**).
5. **Kalman + RTS smoother (jitter kill):** constant‑acceleration state, forward Kalman
   then backward RTS smoothing → sub‑pixel, jitter‑free track with per‑frame covariance.
   Target **< 2 px** position std on straight segments. Fill gaps **≤ 6 frames**;
   longer gaps stay `visible=false`.

### 5.5 Bounce / landing detection
- **Learned bounce classifier** (start with **CatBoost/GBM**, upgrade to a 1D temporal
  CNN) over a sliding window (**~20 frames**) of features: image `y, vy, ay`, court‑space
  `x, y`, speed, trajectory curvature, arc‑residual, and audio‑pop proximity.
  Probability threshold **p_bounce ≥ 0.5**, min bounce separation **0.10 s**.
- **Audio fusion:** align to the nearest audio "pop" within **±40 ms**; correct for
  sound‑propagation delay (`distance_to_camera / 343 m/s`). Gives sub‑frame timing.
- **Bounce location:** at the bounce frame, project the **ball ground‑contact point**
  (bottom of ball = center + `r_px` down the image‑gravity vector; `r_px` from apparent
  size or prior), **not** the center → homography → court `(x, y)`. Removes the
  ball‑radius bias at oblique angles.

### 5.6 In/out calls (uncertainty‑gated, with gray zone)
- **Per‑bounce uncertainty** (this replaces the fixed 5 cm radius):
  ```
  σ_bounce = sqrt( σ_reproj²  +  σ_depth(region)²  +  σ_ballradius²  +  σ_localization² )
  ```
  where `σ_depth(region)` uses the viewpoint error budget (near court ~3–5 cm at a high
  corner; far court up to ~60 cm from a low baseline), `σ_reproj` from calibration,
  `σ_ballradius ≈ 2 cm`, `σ_localization` from the Kalman covariance at that frame.
- **Signed margin** = distance from bounce to the nearest relevant line (baseline,
  sideline, NVZ/kitchen, centerline on serve).
- **Rule:** `in` if `margin > σ_bounce`; `out` if `margin < −σ_bounce`; otherwise
  **`too_close_to_call`**. `confidence = |margin| / (|margin| + σ_bounce)`.
- **Region policy:** surface confident calls freely on the **near** half; on the **far**
  half only when `σ_bounce` is small; otherwise gray. Always report which line, the
  margin, the region, and the dominant uncertainty term.

### 5.7 3D trajectory, spin, speed (context only)
- **Per‑segment physics uplift:** between a contact and the next bounce (or bounce→bounce),
  fit a physics ODE with **gravity + quadratic drag (pickleball Cd, high due to holes) +
  Magnus**, with boundary constraints (launch at contact point, land at bounce point,
  **Z=0 at bounce**). This resolves single‑camera depth per segment. Output `world_xyz`
  per frame, **meter‑level, context only**.
- **Spin (`spin_rpm`, low confidence):** from in‑flight curvature (Magnus) + the bounce
  "kick" (Δ horizontal velocity across the bounce via a COR model: ground COR ≈ 0.62–0.66,
  paddle COR ≈ 0.43–0.44). Sign always; magnitude rough.
- **Speed:** world speed per shot (avg + peak) in mph/kph from the smoothed 3D track.

### 5.8 Events
- **Contact/hit:** fuse **audio pop + wrist‑velocity peak + ball‑trajectory inflection**;
  require **≥ 2 of 3** within a **±35 ms** window (audio dominates timing).
- **Net crossing:** trajectory ∩ net plane (homography); compare crossing height to the
  net‑top height at that `x`; emit `net_cross` (and `into_net` if below).

---

## 6. Consolidated rules & thresholds (quick reference)

| Parameter | Default | Tier |
|---|---|---|
| On‑device heatmap threshold `τ_on` | 0.50 | A |
| On‑device gap fill | ≤ 3 frames | A |
| Rally start / end | ≥5 frames moving / ≥0.8 s empty | A |
| Detector input (base / hi‑res) | 288×512 / 512×896 | B |
| Frames per stack (fast / accurate) | 3 / 8 | B |
| Heatmap threshold (primary / recovery) | 0.50 / 0.25 | B |
| Model‑consensus radius `R` | 60 px @1080p (scale w/ res) | B |
| Court gating margin | 0.5 m | B |
| Max world ball speed | 30 m/s | B |
| RANSAC arc residual | 5 px | B |
| Kalman/RTS gap fill | ≤ 6 frames | B |
| Jitter target (straight seg) | < 2 px std | B |
| Bounce classifier prob `p_bounce` | ≥ 0.50 | B |
| Bounce min separation | 0.10 s | B |
| Audio‑pop window | ±40 ms (speed‑of‑sound corrected) | B |
| Contact fusion window | ±35 ms, ≥2 of 3 cues | B |
| Calibration reproj gate | median<8 px, p95<15 px | B |
| Ball radius uncertainty | ~2 cm | B |

---

## 7. Data & training plan

- **Transfer hierarchy:** badminton/tennis pretrained weights → fine‑tune on **pickleball
  Roboflow (~50k; convert bbox‑center→(x,y), visibility flags)** → fine‑tune on **2–3k
  self‑labeled** frames using the **BlurBall midpoint** convention (label the blur‑streak
  center) → hold out **2k diverse frames** for validation.
- **Augmentation (tuned to phone footage):** motion‑blur synthesis (kernel 3–20 px along
  ball direction), per‑clip background concat, mixup (α 0.5), **hue jitter ±30°** (ball
  color variety), **JPEG/H.264 artifact injection**, copy‑paste ball patches, fps
  simulation, indoor/outdoor color cast. **Geometric transforms identical across the frame
  stack; color transforms may vary per frame.**
- **On‑device student:** knowledge‑distill from the offline teacher on the same data →
  INT8 quantize → CoreML export → verify accuracy + FPS on a real device.
- **Bounce classifier data:** label bounce frames on the same clips; features computed
  from the smoothed track + court projection + audio.

---

## 8. Model selection & licensing

| Component | Pick | License | Commercial note |
|---|---|---|---|
| Offline detector | TrackNetV3 (+ InpaintNet) | MIT | Safe |
| Verifier | WASB | Open (MIT‑style) | Safe |
| Blur/velocity | BlurBall (optional) | Open | Verify before commercial |
| Occlusion (offline) | TOTNet (optional) | Open | Verify |
| On‑device | WASB‑lite / distilled TrackNet nano | MIT/open | Safe |
| Bounce classifier | CatBoost / small temporal CNN | Apache/MIT | Safe |
| Court keypoints | custom heatmap net | your weights | Safe |

Research use is fine now. **For commercialization:** keep detectors MIT/Apache, avoid
AGPL (e.g., YOLO26) and SAM‑license assets in the shipped path.

---

## 9. Validation gates & metrics

**Detection (offline):** P/R @ τ (4 px strict, 10 px lenient); **F1 ≥ 0.90** (target
0.90–0.95); **blur/occlusion recall ≥ 0.75**; **false positives < 5%** after filters.
**Bounce:** timing within **±2 frames**; report bounce‑location error distribution.
**In/out:** on **confident (non‑gray)** calls, **≥ 95% agreement** with human review;
report gray‑zone rate and near/far split.
**Contact:** within **±2 frames** (**±4 ms** with audio).
**On‑device:** **≥ 30 FPS** on target device; ball recall vs the offline track ≥ 0.85.
**Jitter:** smoothed straight‑segment position std **< 2 px**.

---

## 10. Failure‑mode playbook (your top 3)

- **Missed fast/blurry balls:** short exposure (capture), 120 fps drive mode, motion‑blur
  augmentation, BlurBall velocity prior, and **local trajectory search with lowered
  threshold** to recover frames the detector skipped along the predicted arc.
- **Ghost / false detections:** two‑architecture consensus, court+margin gating,
  max‑speed gate, ballistic residual gate — a point must survive **all** to be kept.
- **Jitter:** Kalman + **RTS smoother** with sub‑pixel centroid refinement; never render
  raw arg‑max heatmap coordinates.

---

## 11. Output schema (ball artifacts)

```json
{
  "schema_version": 1,
  "fps": 60.0,
  "source": "tracknet",              // tracknet | wasb | fused | tap
  "frames": [
    {"t": 0.000, "xy": [960.2, 410.7], "conf": 0.87, "visible": true,
     "approx": false, "world_xyz": [1.20, -3.40, 1.05], "spin_rpm": 900}
  ],
  "bounces": [
    {"t": 3.21, "frame": 193, "world_xy": [2.98, 6.55], "contact_xy_img": [1201, 902],
     "court_call": "out", "kitchen_call": "non_nvz", "region": "far",
     "margin_m": 0.14, "uncertainty_m": 0.22, "confidence": 0.39,
     "call": "too_close_to_call", "nearest_line": "far_baseline"}
  ],
  "events": [
    {"type": "contact|bounce|net_cross|into_net", "t": 2.90, "frame": 174,
     "confidence": 0.8, "sources": {"audio":0.9,"wrist_vel":0.7,"ball_inflection":0.6}}
  ],
  "shots": [
    {"start_t": 2.90, "end_t": 3.21, "peak_speed_mph": 41.3, "avg_speed_mph": 33.0}
  ]
}
```

---

## 12. Build order (milestones)

1. **M0 Capture** — enforce protocol + `capture_sidecar.json` + court‑lock UX.
2. **M1 Offline detector** — fine‑tune TrackNetV3 on Roboflow+self‑label; hit F1 gate;
   keep heatmap confidence.
3. **M2 2D post** — RANSAC + Kalman/RTS + ghost gates + local‑search recovery (kill‑list).
4. **M3 Court calib** — keypoint net + solvePnP + reproj gate + manual fallback.
5. **M4 Bounce** — learned classifier + audio fusion + contact‑point projection.
6. **M5 In/out** — uncertainty model + gray zone + region policy.
7. **M6 3D/spin/speed + events** — physics uplift, Magnus/COR, net‑cross, contact fusion.
8. **M7 On‑device** — distill + INT8 + CoreML + on‑device FPS/recall gate + rally spans.
9. **M8 Verifier + validation** — add WASB fusion; run full metric suite; publish gates.
