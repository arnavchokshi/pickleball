# Court Tracking → Player Positioning — Master Build Spec

**Scope:** the court subsystem ONLY — automatically find the court in an iPhone-tripod video, recover its exact metric geometry, and use every court feature (corners, sidelines, baselines, centerline, kitchen/NVZ lines, net) to place players' feet and 3D bodies on the court at line‑call accuracy. Everything downstream (shots, rallies, coaching copy) consumes the artifacts defined here but is out of scope.

**Status:** design spec, v1. This is the source of truth for the implementing agent. The companion file `COURT_AGENT_GOAL_PROMPT.md` is the ~4k‑char kickoff prompt.

---

## 0. Locked decisions (from requirements)

| Decision | Choice | Consequence for design |
|---|---|---|
| Autonomy | **Fully automatic — zero manual taps** | No tap UI anywhere in the happy path. Manual corner tap exists only as a last‑resort recovery, never required. |
| Camera | **Fixed iPhone tripod, static** | Solve the court **once** at startup over N frames, freeze it, then verify cheaply per frame. |
| Source | **iPhone capture + ARKit sidecar** (intrinsics + 6DoF pose + horizontal floor plane) | We get **metric scale for free**. This is the single biggest reason cm‑grade is achievable without taps. |
| Positioning output | **Both feet (ankles) + full 3D body grounded on court plane** | Feet come from pose ankles projected to the plane, never bbox‑bottom. 3D body is root‑grounded on Z=0. |
| Accuracy target | **Line‑call grade (2–5 cm)** near court; confidence‑gated elsewhere | Everything is metric via ARKit plane; every call carries an uncertainty and can abstain ("too close to call"). |
| Latency | **As fast as possible at high accuracy** | On‑device CoreML/ANE; court solved once (<1 s), per‑frame court cost is a few ms (verify only). |
| Robustness | **multi‑court, occluded, low‑angle, and clean** | Explicit target‑court selection, occluded‑keypoint recovery, and viewpoint‑aware confidence gating. |
| Runtime | **On‑device first; hybrid escalation when the accuracy gate fails** | A clip that fails on‑device confidence gates is re‑processed server‑side with heavier models + bundle adjustment. |
| Constraints | None hard; prefer permissive licenses + low cost | Prefer MIT/Apache models and CoreML‑exportable nets. |

---

## 0.1 Execution discipline (MANDATORY — no fabrication)

This applies to every implementer/agent working from this spec. Violating it fails the task regardless of code written.

1. **Do it for real.** Actually download the real model weights, load them, and run real inference on real frames/clips. No stubs, mocks, hardcoded outputs, or "scaffold that intentionally does not load a model." If a component isn't built yet, it is `NOT IMPLEMENTED`, not "done."
2. **Never lie about status.** Mark something DONE only after running it and observing it work — and show the exact command + real output. If it is unimplemented, failing, skipped, partial, or uncertain, state that plainly.
3. **No fake success.** Synthetic/smoke/presence/"it imports" tests do NOT count as VERIFIED. `VERIFIED` requires passing the Section 9 gates on REAL reviewed labels. Never let a silent fallback hide a missing or broken model.
4. **Fail loudly.** If a download, dependency, or step fails, STOP and report the exact error; do not fabricate numbers or quietly substitute a dummy path.
5. **Cite what you used.** Every result must name the exact model + weights source (URL/repo/commit) and the command that produced it, so it is reproducible.
6. **Status vocabulary (use exactly):** `NOT IMPLEMENTED` · `IMPLEMENTED, UNTESTED` · `RUNS (smoke only)` · `VERIFIED (real labels)`. Only the last is "done."

---

## 1. Why this design reaches cm‑grade automatically (the core idea)

A monocular pixel homography from 4 tapped corners is only as good as the taps and gives **no metric scale** — it degrades badly at shallow angles and far court. We avoid that entirely:

1. **ARKit already hands us a metric world.** The setup pass provides camera intrinsics `K` (+ distortion), the 6DoF camera pose, and a **horizontal floor plane in metric world coordinates**. Any image pixel can be back‑projected as a ray and intersected with that floor plane to get a **metric 3D point on the ground** — no scale ambiguity.
2. **The court detector only has to answer "where on the floor is the court, and how is it rotated?"** — a 2D rigid placement on a plane we already know metrically. That is a vastly easier and more stable problem than full monocular calibration.
3. **We over‑constrain the solve with ALL court features** (15 keypoints + fitted lines), not 4 corners, and refine by minimizing reprojection with a point‑and‑line (PnL) cost. Redundancy + a known metric plane = tight, robust geometry.
4. **Static camera ⇒ temporal averaging.** We aggregate the solve over 20–40 startup frames with robust statistics, so per‑frame keypoint noise averages out.
5. **Every position is emitted with an uncertainty.** Line‑call assertions are only made when the uncertainty is below the line margin; otherwise we abstain. This makes "2–5 cm" an honest, gate‑enforced claim rather than a blanket promise.

---

## 2. Coordinate frames & conventions (single source of truth)

- **World (ARKit):** right‑handed, metric meters, gravity‑aligned. Origin = ARKit session origin. This is the metric anchor.
- **Court frame:** origin at **net center**, `x` across width (right positive), `y` along length (far positive), `z` up. Ground plane is `z = 0`. Units meters. (Centered frame — cleaner than a corner origin.)
- **Image:** pixels, origin top‑left, `+x` right, `+y` down, at the recorded video resolution. All keypoint/foot pixels are in **undistorted** image coordinates before any geometry.
- **Transforms:** `T_world_court` (rigid 4×4) places the court in the ARKit world. `K`, `dist` are the camera intrinsics. `T_world_cam` is the ARKit camera pose. Homography `H_court_image` maps court‑plane (z=0) points to image pixels and is derived, not primary.
- **Handedness/gravity:** the ARKit gravity vector must agree with the floor‑plane normal (dot ≥ 0.98); otherwise reject the plane.

---

## 3. Regulation pickleball court model (exact)

All lines are **2 in (0.0508 m)** wide; the **line is part of the area it bounds** except the NVZ line, which is part of the kitchen (stepping on it = fault).

| Quantity | Value (ft/in) | Value (m) |
|---|---|---|
| Court length (baseline–baseline) | 44 ft | 13.4112 |
| Court width (sideline–sideline) | 20 ft | 6.0960 |
| Non‑volley zone depth (net→NVZ line), each side | 7 ft | 2.1336 |
| Baseline → net, each side | 22 ft | 6.7056 |
| Service area length (NVZ line→baseline) | 15 ft | 4.5720 |
| Net height at center | 34 in | 0.8636 |
| Net height at posts | 36 in | 0.9144 |
| Net width (post to post) | 22 ft | 6.7056 |
| Post offset outside sideline | 1 ft | 0.3048 |
| Line width | 2 in | 0.0508 |

**Canonical 15‑keypoint schema** (court frame, z=0; half‑width `w`=3.048, half‑length `L`=6.7056, NVZ `k`=2.1336):

```
 0 near_left_corner      (-w, -L)
 1 near_baseline_center  ( 0, -L)
 2 near_right_corner     ( w, -L)
 3 far_right_corner      ( w,  L)
 4 far_baseline_center   ( 0,  L)
 5 far_left_corner       (-w,  L)
 6 near_nvz_left         (-w, -k)
 7 near_nvz_center       ( 0, -k)
 8 near_nvz_right        ( w, -k)
 9 net_left_sideline     (-w,  0)
10 net_center            ( 0,  0)
11 net_right_sideline    ( w,  0)
12 far_nvz_left          (-w,  k)
13 far_nvz_center        ( 0,  k)
14 far_nvz_right         ( w,  k)
```

Court zones (world polygons, built from the template): `court`, `near_kitchen`, `far_kitchen`, `near_left_service`, `near_right_service`, `far_left_service`, `far_right_service`. NVZ polygons include the NVZ line strip.

---

## 4. System overview

```
[A] iOS ARKit setup pass ──► capture_sidecar.json (K, dist, T_world_cam, floor_plane, gravity, quality)
        │
[B] Auto court keypoint detection (N startup frames) ──► court_keypoints.json (15 pts + confidences, per frame)
        │
[C] Metric calibration solve (ARKit plane + keypoints + lines) ──► court_calibration.json (T_world_court, H, reproj err, uncertainty)
        │
[D] Continuous verification / drift guard (per frame, cheap) ──► drift_status (ok | rebump → re-solve)
        │
[E] Person detect + track + court lock ──► tracks.json (per player, court_xy, side/role)
        │
[F] Pose → foot & 3D body grounding + foot-lock ──► player_ground.json (both feet court_xy + contact, 3D root)
        │
[G] Line/kitchen decision logic (uncertainty-gated) ──► calls.json (in | out | kitchen | too_close_to_call)
        │
[H] On-device confidence gate → escalate clip to server if below threshold
```

---

## 5. Stage rules (exact)

### Stage A — iOS ARKit capture pass (on‑device)

Purpose: obtain a metric, gravity‑aligned world + camera intrinsics + the floor plane, then record with locked settings.

Rules:
1. Run `ARWorldTrackingConfiguration` with `planeDetection = [.horizontal]` (and `sceneDepth`/LiDAR when available). Do **not** try to run a high‑fps `AVCaptureSession` simultaneously — do the ARKit pass first, then switch.
2. **Gate before proceeding:** require `trackingState == .normal` for ≥ 1.0 s, a horizontal plane whose extent covers ≥ 60% of the near half‑court region of the frame, and gravity∥plane‑normal (dot ≥ 0.98).
3. Capture `K` (`ARFrame.camera.intrinsics`), `distortion` (lensDistortionLookupTable → radial coeffs if exposed; else empty), `T_world_cam`, `gravity`, `floor_plane = {point, normal}` in world meters. If LiDAR present, attach a near‑court depth snapshot (≤ 5 m) for plane refinement only.
4. Switch to locked `AVCaptureSession`: lock exposure, focus, white balance; target ≥ 60 fps, 1080p+; HEVC. Write `capture_sidecar.json`. Record.
5. If the ARKit gate fails (blank/low‑contrast floor, low light, `.limited`): fall back to **auto detection without a metric plane** (Stage C degraded path) and mark `capture_quality` accordingly. Manual tap is only offered here as a final fallback.

Sidecar must include `intrinsics.source ∈ {arkit, arkit_lidar}` for the trusted metric path.

### Stage B — Automatic court keypoint detection (zero taps)

Model: **court‑keypoint network** predicting the 15‑keypoint schema as sub‑pixel heatmaps + per‑keypoint confidence. Recommended: a **YOLO‑pose (YOLO11‑pose class) or compact heatmap net (TrackNet/HRNet‑lite)** trained on pickleball (Section 10), exported to **CoreML int8** for on‑device. Input at 640×384 (letterboxed); decode at source resolution.

Rules:
1. Run on `N_startup = 30` frames sampled over the first ~1.5 s (static camera → these are near‑identical; sampling averages sensor noise and transient occlusions).
2. **Sub‑pixel decode:** per keypoint, take the heatmap argmax then parabolic/DSNT refine to sub‑pixel. Reject keypoints with peak confidence `< τ_kp = 0.35`.
3. **Undistort** every keypoint pixel with `K, dist` before any geometry.
4. **Multi‑court selection (target lock):** the detector may fire on multiple courts. Choose the target court by a score:
   `score = 0.5·(quad_area_norm) + 0.3·(centeredness) + 0.2·(player_containment)` where `player_containment` = fraction of person detections whose foot point falls inside the candidate court quad. Break ties by temporal stability across the N frames. Persist the winner; reject others.
5. **Occlusion / low‑angle recovery:** if a keypoint is missing/occluded in a frame, recover it by (a) fitting straight lines to the visible collinear keypoints (each court line has ≥3 known‑collinear points), intersecting lines to recover corners **even off‑image**; and (b) filling from other frames where it was visible (static camera → same pixel). A keypoint is accepted for the solve only if it is confident in ≥ 40% of the N frames OR recoverable by line intersection with residual `< 4 px`.
6. **Aggregate:** per keypoint, robust mean (median then trimmed mean) of accepted sub‑pixel positions across frames → `court_keypoints.json` with per‑keypoint aggregated confidence and inlier count.

### Stage C — Metric calibration solve (the accuracy engine)

Primary path (ARKit metric plane available):
1. For each aggregated keypoint pixel, back‑project a ray with `K` and intersect with `floor_plane` → metric 3D point on the ground `p_world_i`. Pair with its canonical court coordinate `p_court_i` (z=0).
2. Solve the **2D rigid transform on the plane** (rotation θ about plane normal + translation) that maps `p_court_i → p_world_i` via **Kabsch/Umeyama with RANSAC** (scale locked to 1 since both are metric; allow ±2% scale only as a sanity check, not a free parameter). Output `T_world_court`. Inlier threshold `3 cm`.
3. **PnP cross‑check:** with the same keypoints (known court 3D) + `K`, run `solvePnP (SQPnP or ITERATIVE, seeded by ARKit pose)` → camera extrinsics; convert to a court pose and compare to (2). If translation disagrees by > 8 cm or rotation by > 1.5°, lower confidence and prefer the ARKit‑plane solution (it has the metric prior).
4. **PnL refinement:** run Levenberg–Marquardt minimizing a joint cost over ALL features:
   `Σ_kp ρ(‖π(K, T, P_kp) − uv_kp‖²) + λ_line·Σ_line ρ(point‑to‑projected‑line distance for sampled line pixels)`
   with a robust kernel `ρ` (Huber, δ=2 px), `λ_line ≈ 0.5`. Lines: near/far baselines, both sidelines, near/far NVZ lines, centerline, net line — this is where corners + kitchen + centerline + net all contribute. Optimize `T_world_court` (and optionally a tiny intrinsics tweak if `source≠arkit`).
5. **Freeze** the calibration for the clip. Derive `H_court_image` for fast 2D overlay/filter use.
6. Compute and store: reprojection error (median, p95) over all keypoints; per‑keypoint residuals; a **spatial uncertainty field** (Section 7) so downstream can gate line calls.

Degraded path (no trusted ARKit plane): fall back to normalized‑DLT homography from ≥6 keypoints + RANSAC, then the same PnL LM refine using `K` from EXIF/ChArUco/GeoCalib tiers; mark `metric_confidence = low` and force downstream confidence gating / server escalation. Never emit line‑call assertions on this path.

**Hard rules:** always undistort first; never solve from only 4 points when more are available; never trust a solve whose gravity/plane check failed.

### Stage D — Continuous verification / drift guard (per frame, cheap)

Since the camera is static, per‑frame court cost must be ~ms:
1. Every `M_verify = 15` frames, re‑detect a **cheap subset** (4 corners + net_center) OR track the previous line intersections with KLT/optical flow; undistort; compute reprojection error against the frozen calibration.
2. If p95 reprojection error > `8 px` for ≥ 3 consecutive checks → declare a **tripod bump**; re‑run Stage B–C on the next N frames and mark the affected time span `recalibrating`.
3. Between checks, optical‑flow warp absorbs micro‑vibration. Log every drift event.

### Stage E — Person detection, tracking, court lock

1. Person detector (YOLO26/YOLO11 person) + tracker (**ByteTrack or BoT‑SORT‑ReID**). CoreML on‑device.
2. **Court‑polygon filter:** compute each detection's provisional ground point (Stage F gives the precise one; for filtering use ankle midpoint if pose available, else bbox‑bottom‑center) → project to court frame → keep only detections inside the target court polygon + `runoff_margin = 1.5 m`. Rejects spectators/adjacent‑court players.
3. **Ground‑plane association:** associate tracks in **court meters**, not pixels; reject frame‑to‑frame steps > `max_step = 2.0 m × Δframes` (teleport guard, e.g. across the net ID swaps).
4. Cap to `max_players ∈ {2,4}`; assign side (near/far by court `y`) and role (left/right by court `x`).

### Stage F — Pose → foot & 3D body grounding + foot‑lock (the payoff)

1. **2D pose** per player crop (RTMPose‑m / ViTPose server; a compact pose or Apple Vision body pose on‑device). Extract **ankle, heel, big‑toe** for each foot when available (else ankle only).
2. **Foot ground point (metric):** back‑project each ankle/heel/toe pixel and intersect with `floor_plane` (primary, metric) → court `xy`. Do **not** use bbox‑bottom. If ARKit plane absent, use `H_court_image` inverse (non‑metric fallback, gated).
3. **Foot‑contact detection:** `contact = (foot_height_above_plane < τ_h) AND (world_speed < τ_v) AND (pose_conf > τ_c)` with hysteresis: enter `τ_h=2.5 cm / τ_v=0.20 m·s⁻¹`, exit `τ_h=5.0 cm / τ_v=0.35 m·s⁻¹`, `τ_c=0.5`.
4. **Foot‑lock:** on confident contact, snap the stance foot to `z=0`, hold its `(x,y)` (zero world velocity), and CCD/Jacobian‑IK the leg — kills foot‑skate; the contact position is what line calls use.
5. **Full 3D body:** run the monocular 3D body model (Fast SAM‑3D‑Body class) per crop; **ground it ourselves** — set metric root depth from the foot‑on‑plane constraint + a reprojection loss under `K, T`; temporal smooth (RTS/Kalman). Output world joints + optional mesh.
6. Emit both feet's court `xy`, contact flags, per‑foot positional uncertainty, and 3D root/joints to `player_ground.json`.

### Stage G — Line & kitchen decision logic (uncertainty‑gated)

For any foot‑contact event, decide in/out relative to a court line or the kitchen:
1. Represent the relevant boundary as its **metric polygon including the 2‑in line strip**. The NVZ line belongs to the kitchen (touch = in‑kitchen).
2. Compute signed distance `d` from the foot contact point to the boundary (positive = inside kitchen / out of bounds as applicable).
3. Compute the point's **positional uncertainty radius** `σ_p` (Section 7).
4. Decision rule:
   - if `d > +σ_p·zσ` → **inside** (e.g. kitchen fault / ball out),
   - if `d < −σ_p·zσ` → **outside** (safe / in),
   - else → **too_close_to_call** (abstain), with `zσ = 2.0` (≈95%).
5. Only emit a hard call when `metric_confidence = high` and `capture_quality ≠ poor`. Every call stores `d`, `σ_p`, and the frames used.

### Stage H — On‑device vs hybrid escalation

Run everything on‑device. Escalate the clip (or just the uncertain spans) to the server when ANY of:
- court reprojection p95 > `5 px`, or metric_confidence ≠ high,
- keypoint inlier count < 10 of 15, or any required line unrecovered,
- capture_quality = poor, or a drift/recalibration event occurred,
- a line‑call is requested in a span that is `too_close_to_call` on‑device.
Server re‑runs with heavier pose/3D models + full multi‑frame bundle adjustment and returns an authoritative `court_calibration.json` + calls. On‑device results remain the low‑latency preview.

---

## 6. Parameters & thresholds (tune, but start here)

| Name | Value | Where |
|---|---|---|
| `N_startup` | 30 frames (~1.5 s) | B |
| `τ_kp` (keypoint conf) | 0.35 | B |
| keypoint accept coverage | ≥ 40% of N frames | B |
| Kabsch RANSAC inlier | 3 cm | C |
| PnP disagreement gate | 8 cm / 1.5° | C |
| Huber δ (reproj) | 2 px | C |
| `λ_line` | 0.5 | C |
| reproj gate (freeze) | median < 3 px, p95 < 8 px @1080p | C |
| reproj gate (line‑call grade) | p95 < 5 px | C/H |
| `M_verify` | every 15 frames | D |
| drift trip | p95 > 8 px ×3 checks | D |
| `runoff_margin` | 1.5 m | E |
| `max_step` | 2.0 m/frame | E |
| contact enter/exit | 2.5cm·0.20 / 5.0cm·0.35 | F |
| `τ_c` pose conf | 0.5 | F |
| call abstain band | `zσ = 2.0` | G |

---

## 7. Accuracy budget & uncertainty model

**Why 2–5 cm is real near court and honest elsewhere.** With a metric ARKit plane, the placed foot error is dominated by:
- ankle pixel error `e_px` (pose) × the **depth‑per‑pixel** at that court location (grows with distance and shallow angle),
- floor‑plane estimate error (ARKit horizontal plane, typically ~1–2 cm near, worse far),
- calibration residual (`T_world_court`).

Model per point: `σ_p ≈ sqrt( (e_px · gsd(x,y))² + σ_plane² + σ_calib² )`, where `gsd(x,y)` (ground sample distance, m/px) is computed analytically from `K, T` at the query court location. Store a `σ_p` map so downstream gating is location‑aware.

Expected (elevated tripod ~3–4 m, 30–60°):
- near court / kitchen: `σ_p ≈ 2–4 cm` → line‑call grade ✔
- mid court: `σ_p ≈ 4–8 cm`
- far baseline at low angle: `σ_p ≈ 15–60 cm` → abstain/confidence‑gate, or escalate.

This is why the product promise is: **line‑call in/out where `σ_p` permits, explicit "too close to call" otherwise** — never a fake precise number.

---

## 8. Artifacts / JSON schemas (contract)

`capture_sidecar.json`: `{ intrinsics{fx,fy,cx,cy,dist[],source}, image_size, camera_pose{R,t}, floor_plane{point,normal}, gravity, lidar_depth_ref?, fps, capture_quality{grade,reasons} }`

`court_keypoints.json`: `{ schema_version, frame_indexes[], keypoints[{name, uv[2], confidence, inlier_frames, recovered:bool}], target_court_score }`

`court_calibration.json`: `{ schema_version, sport, coordinate_frame:"court_netcenter_z_up_m", T_world_court[4x4], intrinsics, homography[3x3], reprojection_error_px{median,p95}, per_keypoint_residual_px[], metric_confidence:"high|med|low", gsd_model{...}, capture_quality, source, solved_over_frames }`

`net_plane.json`: regulation net plane derived from template (center 0.8636 m, posts 0.9144 m, half‑width 3.3528 m).

`tracks.json`: `{ fps, players[{ id, side, role, frames[{ t, bbox, court_xy[2], conf }] }] }`

`player_ground.json`: `{ fps, players[{ id, frames[{ t, feet[{side:L|R, court_xy[2], height_m, contact:bool, sigma_p_m}], root_world[3], joints_world?[], mesh_ref? }] }] }`

`calls.json`: `{ events[{ t, player_id, foot:L|R, boundary:"kitchen|sideline|baseline|centerline", decision:"in|out|kitchen|too_close_to_call", signed_dist_m, sigma_p_m, frames[] }] }`

`drift_log.json`: `{ checks[{frame, p95_px, tripped:bool}], recalibrations[{from_frame,to_frame,reason}] }`

---

## 9. Acceptance gates & evaluation protocol

Build is not "done" until, on a labeled clip matrix spanning {low/mid/high} × {shallow/steep/side} viewpoints, indoor+outdoor, single+multi‑court:

1. **Court found automatically on ≥ 95% of clips** with zero taps (target‑court correct on 100% of multi‑court clips).
2. **Reprojection:** median < 3 px, p95 < 8 px @1080p on ≥ 90% of clips; **no viewpoint bucket below 80%**.
3. **Keypoint PCK@5px ≥ 0.95** on the labeled court‑keypoint test set.
4. **Feet‑to‑world:** near/kitchen lateral < 3 cm, depth < 5 cm on confident frames (measured against ArUco/tape ground‑truth or LiDAR); far‑court beyond budget must be **abstained or escalated**, not wrong.
5. **Kitchen foot‑fault agreement ≥ 95%** vs human review on confident frames; report abstain rate.
6. **Drift guard:** detects an injected 20‑px bump within `M_verify+1` frames; **0** false trips on a static clip.
7. **Latency:** court solved once < 1 s over N frames; steady‑state per‑frame court overhead < 5 ms on device.
8. **Escalation correctness:** every clip that fails a metric gate on‑device is flagged for server; no silent low‑confidence line calls.

A model/algorithm is only `VERIFIED` when it passes the accuracy gate on **real reviewed labels** — presence/smoke/synthetic passes do not count.

---

## 10. Model choices & training recipe (my "best" mix)

- **Court keypoints (primary):** custom **YOLO‑pose (YOLO11‑pose)** with a 15‑keypoint head, OR a compact **heatmap net (TrackNet/HRNet‑lite)** with sub‑pixel decode. Rationale: YOLO‑pose exports cleanly to **CoreML int8** and lets us define the exact 15‑pt schema incl. kitchen + centerline; heatmap net gives slightly better sub‑pixel precision server‑side. Ship both: light on device, heavy on server.
  - **Init:** optionally warm‑start from **CourtKeyNet** (MIT) or **TennisCourtDetector** backbones; but they only give outer corners / tennis topology, so we retrain the head for 15 pickleball points.
  - **Data:** ~500–1000 labeled pickleball frames spanning our viewpoint range (indoor/outdoor, shallow/steep, single/multi‑court, occluded corners), plus **synthetic court renders** (50–500 randomized viewpoints; labels free since geometry is known) for pretraining. Heavy aug: color/shadow/glare/partial occlusion.
  - **Loss:** heatmap MSE + Adaptive‑Wing + a **quadrilateral/geometry consistency** term (enforce the known rectangle/line collinearity).
  - **Decode:** DSNT/parabolic sub‑pixel.
- **Person:** YOLO26/YOLO11 person + ByteTrack/BoT‑SORT‑ReID (CoreML).
- **Pose:** RTMPose‑m or ViTPose (server); compact pose / Apple Vision (device). Ankle precision is the accuracy bottleneck — prefer the strongest pose you can afford, and use heel+toe when available.
- **3D body:** Fast SAM‑3D‑Body class (server); grounded by our plane constraint, not the model's own depth.
- **Licensing:** prefer MIT/Apache (CourtKeyNet MIT; Ultralytics AGPL — use a permissive/own head or the enterprise route if shipping commercially).

---

## 11. Edge cases & failure handling

- **Blank/low‑contrast floor, low light, ARKit `.limited`:** degraded path + escalate; offer manual tap only as last resort.
- **Corners off‑frame (low angle):** recover by line intersection off‑image; if net + NVZ + one baseline visible, the rigid plane solve still succeeds.
- **Multiple courts / adjacent play:** target‑court score + player containment; hard‑reject other courts' players via court polygon.
- **Player standing on a line during startup:** multi‑frame aggregation + line‑fit recovery route around the occlusion.
- **Tripod bump mid‑clip:** drift guard → re‑solve; annotate the recalibrating span.
- **Fabricated intrinsics (no ARKit):** never allowed on the line‑call path; forces `metric_confidence=low` and server escalation.

---

## 12. Implementation milestones

1. **M1 — Geometry core (CPU, testable):** court template, 15‑pt schema, zones, net plane, ray‑plane intersection, Kabsch/Umeyama, PnP, PnL LM refine, undistort, uncertainty model. Unit‑tested against synthetic + a taped real court.
2. **M2 — Auto detection:** train/export court‑keypoint model; multi‑frame aggregation; target‑court selection; occluded‑keypoint recovery; freeze calibration; drift guard.
3. **M3 — Positioning:** pose ankles → plane projection; contact + foot‑lock; both‑feet court_xy; 3D grounding; `player_ground.json`.
4. **M4 — Decisions + gating:** line/kitchen logic with uncertainty abstain; on‑device confidence gate + server escalation.
5. **M5 — iOS integration:** ARKit setup pass, CoreML models, on‑device pipeline; hybrid handoff.
6. **M6 — Eval harness:** labeled clip matrix; all Section 9 gates automated; report per‑viewpoint.

---

## 13. Open risks

- On‑device pose ankle precision may cap far‑court accuracy → mitigated by confidence gating + server escalation.
- ARKit plane accuracy on textureless courts → LiDAR helps near; far court relies on keypoint geometry.
- Commercial licensing of Ultralytics YOLO → keep a permissive/own detection head option.
- 2–5 cm is only physically attainable where `σ_p` permits; the honest promise is "line‑call where confident, abstain otherwise."
