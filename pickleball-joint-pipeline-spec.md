# Pickleball Joint-Tracking Pipeline — Design Spec (v1)

**Scope:** *Human joint tracking only.* This document defines exactly how the system turns an uploaded video into per-player, per-frame, world-grounded 3D skeletons. Ball, court-line, net, and paddle detection are **out of scope** here except where they feed the joint pipeline (contact timing, court plane). This is a normative spec: **MUST / SHOULD / MAY** are used deliberately, and every rule has an ID (`JT-Rxx`) the team can cite.

---

## 0. Requirements contract (locked with product owner)

| # | Requirement | Decision |
|---|---|---|
| R1 | Primary output | **(a)** Metric 3D skeleton JSON per player/frame **+ (b)** a rotatable/scrubbable 3D replay viewer **+ (c)** a FULL-BODY 3D mesh (SMPL-X/MHR surface) for the hitter across each contact window, rendered as a solid surface in the viewer (not just skeleton points) |
| R2 | Joint set | **Body + feet (core) + hands (next), NO face.** Priority order body > feet > hands. Drop detail to gain meaningful speed, in that order. |
| R3 | Spatial frame | **Absolute metric world coordinates** in the court frame (Z=0 = ground); all players correctly spaced relative to each other and the court |
| R4 | Speed | **As fast as possible subject to a hard accuracy floor.** Accuracy is the constraint; speed is the objective. |
| R5 | On-device | **iPhone live preview is a bonus, not a guarantee.** If on-device can't hit the accuracy floor, fall back to offline. Offline server is the source of truth. |
| R6 | Players | **Singles (2) and doubles (4).** Variable 2–4. |
| R7 | Motion quality | **Balanced** — smooth + low-jitter, but must not lag fast swings/lunges noticeably |
| R8 | Accuracy floor | **"Clearly correct": ≤ 5 cm world error on limbs; wrists tighter (≤ 3 cm).** |
| R9 | Wrist priority | **Yes** — spend extra compute on the hitting arm/wrist (higher-res crop + contact mesh) |
| R10 | Licensing | Research-licensed models OK for now; **prefer commercially-usable (Apache/MIT) where quality is equal** (this is ultimately a product) |
| R11 | Camera | **Single fixed camera on a tripod, whole court visible** (static extrinsics for the whole clip) |

**Global success definition:** upload video → get, as fast as possible, a continuous, world-grounded 3D skeleton for every player across the whole clip (not just at contacts), accurate enough that the pose "clearly reads as correct," with wrists sharp at contact — rendered in a 3D viewer and exported as JSON.

---

## 1. Core architecture: two lanes (this is the key idea)

Never use a mesh model as the joint source for all players — mesh is the expensive tool, joints are the cheap one. The pipeline runs **two lanes**:

```
                      ┌─────────────────────────────────────────────┐
  video ─▶ calibrate ─▶ detect+track (all players, every frame)      │
          (once)      └───────────────┬─────────────────────────────┘
                                      │
        ┌─────────────────────────────┴──────────────────────────────┐
        ▼ LANE A — ALWAYS ON                       ▼ LANE B — EVENT ONLY
  fast 3D whole-body pose               deep mesh for the HITTER only,
  RTMW3D per player, every frame        Fast SAM-3D-Body, at contact windows
        │                                          │
        ▼                                          │
  temporal lift + smooth (MotionBERT + One-Euro)   │
        │                                          │
        ▼                                          ▼
  world-ground on court plane ◀── full-body MHR mesh (hitter) at contact
        │                            (rendered as a solid surface)
        ▼
  skeleton3d.json (metric world)  +  3D replay viewer
```

- **Lane A (always on, all 2–4 players, every frame):** produces the continuous skeleton. This *is* the baseline deliverable.
- **Lane B (event-triggered, hitter only, at contact):** a **full-body SMPL-X/MHR mesh** for the hitter, rendered as a solid surface across the contact window (R1c). The mesh's sharp wrist/hand joints also override Lane A joints for analysis (satisfies R9). During contact the hitter is shown as a mesh instead of a skeleton; everyone else stays a skeleton.

> **Why:** running Fast SAM-3D-Body on every player every frame is ~10× more expensive and still leaves gaps between contacts. The repo's current scheduler (`frame_rating.py`) already tiers frames into `deep_mesh` / `skeleton_preview` / `baseline` and even picks the target player — but **no runner exists behind `skeleton_preview`**, so today you only get joints inside deep-mesh windows. Lane A fills that hole.

---

## 2. Stage-by-stage exact rules

### Stage 0 — Calibration (once per clip; camera is static, R11)
- **JT-R01** The system MUST solve court→camera geometry **once** for the whole clip: intrinsics `K`, extrinsics `[R|t]` via `solvePnP` against the known court template, defining **world Z=0 = court plane**.
- **JT-R02** Reprojection error of the calibration MUST be reported; if median corner error > 3 px the clip is flagged `calibration_low_confidence` and grounding (Stage 5) downgrades to "relative-only" for that clip (no fabricated metric placement).
- **JT-R03** Because the camera is fixed, calibration MUST NOT be re-run per frame. (Fixes a current inefficiency where FOV/geometry is recomputed repeatedly.)

### Stage 1 — Detection + tracking + ID (all players, every frame)
- **JT-R04** Person detection: **YOLO** (repo already ships YOLO26m) restricted to `class=person`.
- **JT-R05** Tracking + stable IDs: **BoT-SORT-ReID** (already in repo). IDs MUST persist across brief occlusions (net crossings, kitchen crowding).
- **JT-R06** **Court-region gating:** a detection is a "player" only if its foot point lies within the court polygon (+ a configurable margin, default 1.5 m) for ≥ 5 consecutive frames. Bystanders/coaches outside the court MUST be dropped.
- **JT-R07** `expected_players ∈ {2, 4}` MUST be a per-clip input (singles/doubles). Players MUST be assigned to court **sides/quadrants** so IDs map to stable roles (e.g., `near-left`, `far-right`). Doubles → 4 quadrants; singles → 2 sides.
- **JT-R08** If detected player count ≠ `expected_players` for > 10 consecutive frames, flag `player_count_mismatch` (do not silently fabricate a missing player).

### Stage 2 — Lane A: fast per-frame 3D whole-body pose
- **JT-R09** Model: **RTMW3D-x** (MMPose, Apache-2.0) run top-down on each tracked box. Output: 133-keypoint whole-body 3D via SimCC (X/Y/Z), **root-relative metric** coordinates. Verified to run 4 people (public in-browser demo exists).
- **JT-R10** **Joint subset per R2:** keep **17 body + 6 feet + 42 hands = 65 joints; DROP the 68 face joints.** The always-computed "core" is the **23 body+feet joints**; hands are computed when the compute budget allows or for players near the ball (see JT-R21).
- **JT-R11** All active player crops for a frame MUST be **batched** into a single forward pass (not looped one-by-one).
- **JT-R12** Per-joint confidence MUST be preserved end-to-end. Joints below `conf=0.3` are marked low-confidence and are candidates for temporal fill (Stage 4), never emitted as high-confidence.
- **JT-R13** Lane A MUST run on **every tracked frame for every player** — this is the continuous skeleton. It replaces the empty `skeleton_preview` tier in `frame_rating.py`.

### Stage 3 — Lane B: contact-time FULL-BODY mesh (hitter only) (R1c, R9)
- **JT-R14** Contact windows come from the existing scheduler (`ContactWindows` → `deep_mesh_windows`). For each contact, the **hitting player only** is targeted.
- **JT-R15** For the hitter within `± contact_padding` (default 80 ms), run **Fast SAM-3D-Body** (MHR/SAM license — research-OK per R10). Its output dict provides the **full-body mesh**: `pred_vertices` (3D mesh vertices), `pred_keypoints_3d`, `body_pose_params`, `hand_pose_params`, `shape_params` — a complete **MHR (Momentum Human Rig, SMPL-X-style) body surface** including feet + hands. The Fast variant keeps on-par reconstruction fidelity vs base 3DB.
- **JT-R16** **Full-body mesh output + render:** transform `pred_vertices` to the world frame (`worldhmr` → `mesh_vertices_world`) and **export the mesh** (see §4) — per-frame world vertices + SMPL-X/MHR params + a reference to the **static MHR face topology** (triangle indices, loaded once from `mhr_model.pt` assets). In the viewer, the hitter is drawn as a **solid shaded mesh** across the contact window (not just skeleton), with a raised-cosine crossfade (skeleton↔mesh) at the window edges so there is no pop. All other players remain skeletons.
- **JT-R16b** The mesh's **wrist + hand joints also override Lane A joints** for that player over the window (this is where wrist accuracy, R9, is guaranteed) — the mesh is both a visualization and the joint source during contact.
- **JT-R16c** *(Optional, config flag `mesh_all_players_at_contact`)* the same full-body mesh MAY be produced for **all** active players during a contact window if compute allows; default is hitter-only.
- **JT-R17** If Lane B fails/degrades (mesh error, license disabled), the system MUST fall back to a **higher-resolution RTMW3D crop of the hitting arm** (skeleton only) and flag `mesh_unavailable` — never drop wrist quality silently and never fabricate a mesh.

### Stage 4 — Temporal lift, denoise & smoothing (R7 balanced, R8 accuracy)
- **JT-R18** **Body (17 joints):** feed the Lane A 2D/3D body sequence through **MotionBERT** (DSTformer, sliding window ≤ 243 frames) to produce temporally-coherent 3D. This delivers ~2.7–3.7 cm MPJPE and inherently reduces jitter (satisfies R8 body + R7 smoothness).
- **JT-R19** **Feet + hands (not covered by MotionBERT's H36M-17):** anchor RTMW3D's feet/hands onto the MotionBERT-smoothed body root/wrist, then apply a **One-Euro filter** per joint per axis. Default params (tune to fps): `mincutoff=1.0`, `beta=0.3` (raise `beta` if fast swings lag — R7).
- **JT-R20** **Bone-length constraint:** estimate each player's median bone lengths over a clean calibration window (first N confident frames), then enforce them per frame (project onto constant-length limbs). This removes limb "breathing" and improves world accuracy.
- **JT-R21** **Compute-adaptive detail (R4):** players **far from the ball** MAY drop to body+feet only (23 joints) and a lighter temporal stride; the player **near/at the ball** MUST get full hands + contact accent. This is how "as fast as possible" is achieved without hurting the shots that matter.

### Stage 5 — Absolute world grounding on the court plane (R3)
The camera is static and the court plane is known — this is our unfair advantage. Exact rules:
- **JT-R22** For each player/frame, pick the **support foot** = the ankle/foot keypoint that is (a) most confident, (b) lowest in world Z, and (c) lowest vertical velocity over a 5-frame window.
- **JT-R23** **Anchor XY:** back-project the support-foot **image** keypoint through `K,[R|t]` and intersect the world plane **Z=0** → the foot's metric world (X,Y). Rotate the player's root-relative skeleton into world orientation (camera→world rotation) and translate it so the support foot lands on that grounded point.
- **JT-R24** **Scale:** trust RTMW3D's metric depth as the primary scale; correct residual scale using the per-player height/limb-length prior (JT-R20). Do NOT let per-frame scale float freely.
- **JT-R25** **Foot-lock (anti-skate):** while a foot is in contact (height < 3 cm AND horizontal speed < threshold), **pin its world XY**; the rest of the body moves relative to it. This kills foot sliding.
- **JT-R26** **Airborne handling (jumps/lunges, both feet up):** hold the last grounded XY and integrate root translation from the metric-depth motion; re-anchor when a foot lands. Never re-derive XY from an airborne foot.
- **JT-R27** All emitted coordinates MUST be in the **court world frame, meters, Z-up, Z=0 = ground.**

### Stage 6 — Output (R1)
- **JT-R28** Emit `skeleton3d.json` (schema in §4): metric world joints per player, per frame, with per-joint confidence and provenance (`lane_a` / `lane_b_contact` / `interpolated`).
- **JT-R28b** Emit `body_mesh.json` for the hitter contact windows (schema in §4): per-frame `mesh_vertices_world`, SMPL-X/MHR params, a reference to the static MHR face topology, and provenance. This is the full-body mesh stream (R1c).
- **JT-R29** Provide a **3D replay viewer** (Three.js/web): renders the court plane, all 2–4 skeletons in correct relative spacing, and the ball if available; **renders the hitter as a solid shaded full-body mesh during contact windows** (loading the MHR faces + per-frame world vertices), crossfading skeleton↔mesh; supports orbit/rotate + frame scrub + play. IDs are color-coded and stable.
- **JT-R30** `skeleton3d.json` from Lane A MUST NOT be stamped `preview_only=true` (the current code marks all skeletons preview-only because they were a mesh byproduct — that flag is wrong for the real Lane A output).

---

## 3. Model choices & licenses

| Role | Model | Why | Speed | License |
|---|---|---|---|---|
| **Lane A pose (primary)** | **RTMW3D-x** | Real-time 3D whole-body, 133 kpts + metric root-rel depth; proven 4-person; direct 3D (no separate lifter needed for coarse output) | Real-time on GPU; 4-person in-browser demo exists | **Apache-2.0** ✅ |
| **Temporal refine** | **MotionBERT** | Smooths + lifts body-17 to ~2.7–3.7 cm MPJPE; ≤243-frame window; kills jitter (R7/R8) | Cheap, batched | Apache-2.0/MIT ✅ |
| **Detector** | YOLO26m (in repo) | Already integrated | Fast | check ✅ |
| **Tracking/ID** | BoT-SORT-ReID (in repo) | Already integrated | Fast | ✅ |
| **Lane B contact mesh** | Fast SAM-3D-Body (MHR) | **Full-body SMPL-X/MHR surface** (`pred_vertices` + body/hand/shape params) rendered at contact; sharp wrist/hand (~65 ms/frame on 5090) | Event-only | SAM/MHR (research-OK, R10) ⚠️ |
| **Strong alt / future** | NLF (Neural Localizer Fields) | 3D joints **+ SMPL shape** at 41–109 fps; could make Lane A lightly "meshed" everywhere and shrink Lane B | 79/410 fps (S) on 3090 | Research ⚠️ verify commercial |
| **Detector-free multi-person 2D (option)** | RTMO-l | 141 FPS V100, no detector, faster than top-down with ≥4 people | Very fast | Apache-2.0 ✅ |
| **On-device live (bonus)** | Apple Vision 3D **per-crop** / MediaPipe BlazePose / RTMW3D-m CoreML | Live preview only | On-device | ✅ |

**Licensing note (R10):** Lane A + refine + detector + tracker are all commercially-usable. Only the **contact accent** (Fast SAM-3D-Body) and the optional NLF path carry research licenses — both are isolated and swappable, so the product's core joint stream stays clean.

---

## 4. Output JSON schema (`skeleton3d.json`)

```json
{
  "schema_version": 1,
  "world_frame": "court_Z0_up",
  "units": "meters",
  "fps": 30.0,
  "expected_players": 4,
  "joint_names": ["pelvis", "left_hip", "...", "left_wrist", "right_wrist", "...feet...", "...hands..."],
  "joint_groups": {"body": [0, "..."], "feet": ["..."], "hands": ["..."]},
  "calibration": {"reproj_px_median": 1.8, "confidence": "ok"},
  "players": [
    {
      "id": 0,
      "court_role": "near_left",
      "bone_lengths_m": {"...": 0.0},
      "frames": [
        {
          "t": 0.0,
          "frame_idx": 0,
          "joints_world": [[x, y, z], "..."],
          "joint_conf": [0.97, "..."],
          "provenance": ["lane_a", "..."],
          "support_foot": "right",
          "foot_locked": true,
          "interpolated": false
        }
      ]
    }
  ]
}
```

- **JT-R31** Every joint MUST carry `provenance` so downstream (and QA) can see whether a value is Lane A, contact-mesh, or interpolated. No fabricated joint may be labeled `lane_a`.

### 4b. Full-body mesh stream (`body_mesh.json`) — hitter contact windows (R1c)

```json
{
  "schema_version": 1,
  "world_frame": "court_Z0_up",
  "units": "meters",
  "model": "mhr_smplx",
  "faces_ref": "assets/mhr_faces.npy",   // static triangle topology, loaded once
  "windows": [
    {
      "player_id": 0,
      "contact_t": 12.34,
      "frame_start": 370, "frame_end": 375,
      "frames": [
        {
          "t": 12.30, "frame_idx": 369,
          "mesh_vertices_world": [[x, y, z], "..."],   // per-frame world-space surface
          "smplx_params": {"global_orient": [], "body_pose": [], "left_hand_pose": [], "right_hand_pose": [], "betas": []},
          "blend_weight": 0.5,        // raised-cosine skeleton<->mesh crossfade
          "provenance": "lane_b_mesh"
        }
      ]
    }
  ]
}
```

- **JT-R31b** `mesh_vertices_world` MUST be in the same court world frame as `skeleton3d.json`, so the mesh and the other players' skeletons line up in the viewer. Faces are stored once (static MHR topology), not per-frame.

---

## 5. Accuracy & QA gates (hard floor per R8 — pipeline is "green" only if all pass)

| Gate | Metric | Threshold |
|---|---|---|
| G1 world limb accuracy | world-MPJPE on labeled frames (body+feet) | **≤ 50 mm** |
| G2 wrist accuracy | world error at wrists (contact frames) | **≤ 30 mm** |
| G3 foot slide | world XY drift of a foot during a contact/stance | **≤ 30 mm** over the stance |
| G4 jitter | MPJVE (per-joint velocity error) / accel spikes | below tuned threshold (no visible shake) |
| G5 ID stability | ID switches per rally | **0** target, ≤1 flagged |
| G6 coverage | frames where all `expected_players` have a skeleton | **≥ 98%** (gaps flagged, not fabricated) |
| G7 latency | wall-clock per minute of 4-player 1080p on target GPU | report; optimize under G1–G6 |
| G8 mesh at contact | hitter contact windows with a rendered full-body mesh (or explicit `mesh_unavailable`) | **100%** (mesh vertices align with skeleton world frame; no gaps or fabrication) |

- **JT-R32** These gates MUST be measured on a labeled validation set (a few clips with hand-annotated or mocap-referenced joints). "It renders" is NOT a pass — presence ≠ accuracy. (This is the biggest gap in the current repo: skeletons exist but nothing measures world-space joint error.)

---

## 6. Failure & edge-case rules (fail-closed, never fabricate)

- **JT-R33 Short occlusion (< 8 frames):** interpolate joints (constant-velocity/linear), mark `interpolated=true`, keep the ID.
- **JT-R34 Long occlusion (≥ 8 frames):** emit a gap (no joints) rather than fabricating; re-acquire via ReID.
- **JT-R35 Low-confidence joint:** fill from temporal neighbors if within window, else emit with low `joint_conf` — never upgrade confidence.
- **JT-R36 Two players overlapping (net/kitchen):** rely on ReID + court-side priors (JT-R07); if boxes merge, prefer splitting via instance mask; if unresolved, flag `identity_ambiguous` for that span.
- **JT-R37 Calibration low-confidence (JT-R02):** downgrade to relative-only output for the clip; do not emit false metric placement.

---

## 7. Speed budget (R4 — maximize under the accuracy floor)

Rough per-minute-of-video, 4 players @ 30 fps (≈ 7,200 person-crops), single modern datacenter GPU (L40S/A100):

- **Lane A (RTMW3D-x, batched):** the dominant cost; target **faster-than-real-time** with batching + TensorRT FP16. (Contrast: running Fast SAM-3D-Body on everyone every frame ≈ 12+ min/clip — Lane A is ~10× cheaper and continuous.)
- **MotionBERT refine:** small, batched over windows.
- **Lane B (contact mesh):** only a handful of windows/rally × 1 player → negligible fraction of total.
- **Recommended hardware:** **any ≥16 GB CUDA GPU; L40S or A100 recommended.** H100 is overkill for this workload.

> **Optimization levers (in priority order):** (1) batch all player crops per frame; (2) TensorRT/FP16 for RTMW3D; (3) drop hands + face for far-from-ball players (JT-R21); (4) temporal stride for low-motion players; (5) calibrate once, not per-frame.

---

## 8. On-device live preview (R5 — bonus, not the guarantee)

- **JT-R38** On-device is a **live "gist" preview only.** The offline server output is authoritative and is what must pass §5 gates.
- **JT-R39 Apple limitation:** `VNDetectHumanBodyPose3DRequest` returns **only the single most-prominent person** (17 joints, meters rel. camera, iOS 17+). For 2–4 players the app MUST **crop each tracked box and run the 3D request per crop**, or use a multi-person alternative (MediaPipe BlazePose 3D, or RTMW3D-m exported to CoreML).
- **JT-R40** The phone shows a rough real-time skeleton for immediate feedback (e.g., "in/out", stance); the server later returns the accurate, smoothed, world-grounded version. UX MUST make clear which is which.

---

## 9. Mapping to the existing repo (what to reuse vs build)

| Existing | Action |
|---|---|
| `frame_rating.py` (tier scheduler) | **Reuse.** It already tiers frames + picks the hitter. Implement the missing `skeleton_preview` tier as Lane A. |
| `body_compute.py` | **Reuse for Lane B scheduling.** Kill the `all_track_frames` fallback that meshes everyone every frame. |
| `worldhmr.py` (camera→world, foot-Z) | **Reuse** for Stage 5 grounding of the skeleton (not just the mesh). Fix the mesh/joint desync bug while here. |
| `skeleton3d.py` | **Reuse as a semantic remapper**, but stop marking Lane A output `preview_only` (JT-R30). |
| `BodyStageRunner` (Fast SAM-3D-Body) | **Keep — Lane B only** (contact, hitter). |
| **NEW: `PoseStageRunner` (RTMW3D)** | **Build.** `real_model=True`; runs Lane A on YOLO26 boxes for all players every frame; writes real `skeleton3d.json`. |
| **NEW: temporal refine (MotionBERT)** | **Build.** Stage 4. |
| `wrist_velocity_peaks.py` | **Rewire** to consume Lane A wrists → breaks the current circular dependency (contact needs wrists, wrists came from mesh, mesh scheduled by contact). |
| `models/MANIFEST.json` | **Add** RTMW3D-x + MotionBERT entries (RTMW 2D family already listed). |
| eval/ | **Add** world-MPJPE + foot-slide + ID-switch gates (§5). |

---

## 10. One-paragraph summary for the team

Build a **fast always-on 3D pose lane (RTMW3D-x)** that runs on every tracked player every frame, temporally smoothed by **MotionBERT + One-Euro + bone-length constraints**, and **grounded to the known court plane** for absolute metric world coordinates — that is the continuous 4-player skeleton and the product's core output. Reserve **Fast SAM-3D-Body for the hitter at contact only**, and there produce a **full-body SMPL-X/MHR mesh** (`pred_vertices` + params) that is rendered as a **solid surface** in the viewer — so at the moment of contact you see the hitter's whole body meshed instead of a skeleton, and the mesh's wrist/hand doubles as the joint source. Keep the existing tier scheduler (it already targets the hitter) but implement the empty `skeleton_preview` tier as this pose lane and delete the "mesh everyone every frame" fallback. Ship a **metric `skeleton3d.json` + `body_mesh.json` + a rotatable 3D replay viewer**, and gate every release on **world-MPJPE ≤ 5 cm (wrists ≤ 3 cm), zero ID switches, no foot slide, and a rendered mesh at every contact** — measured on labeled clips, because "it renders" is not "it's correct." On-device is a rough live preview only; offline is the accuracy guarantee.
