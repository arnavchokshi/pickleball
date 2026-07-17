# Static-first calibration: single-lock + cross-frame pooling + distortion fix

Lane: `static_cal_firstlock_20260717`
Status: **`VERIFIED=0`, preview-band, pre-registered.** No promotion. CPU-only, no GPU.
Owner directive (2026-07-17): "for the future we can just assume fully static
cameras that don't move." This is a v1 product assumption (North Star §1.1).
North Star anchors: §2.2 CAL row, NS-03.CAL, §5 queue rank 4, §2.3.

## 0. Why this exists (the leverage)

The placement pipeline consumes ONE court solution: homography (DLT) + PnP →
per-player 2D footpoint → undistort → inverse-homography unproject to metric
court-XY → pixel-noise covariance → inverse-covariance fusion of foot signals →
BODY grounding. So one good static lock lifts placement, ball-3D, in/out, BODY
grounding, paddle, and fusion at once. Under a static-camera assumption the
court geometry does not change frame-to-frame, so re-solving per frame is waste
and a per-frame solve is strictly noisier than one lock pooled over all frames.

This lane does NOT try to make automatic court-finding an authority path.
Learned/auto corner-finding is unsolved even at SOTA (owner-viewpoint PCK@5=0
for our learned candidates; TVCalib ~65-69% AC@5px even WITH GT segments; no
production system does auto single-MOVING-camera court tracking). The ≥0.95
auto-find gate (NS-03.CAL) stays on the books but v1 does not wait on it. Our
own line-evidence solve is 2.6px median — at/below pb.vision (M4 head-to-head:
ours 6.61px vs pb.vision 5.67px, <1px apart) — when seeded well. pb.vision's
edge is capture discipline (mandatory static mount, all four corners visible, no
cuts), not solver math. A static-camera assumption lets us buy that discipline
by construction and pool evidence across frames.

## 1. Scope and non-goals

In scope (CPU-only, new code around existing CAL seams; raw solves immutable):

1. A single-lock calibration path for static cameras.
2. Cross-frame court-line-evidence pooling to refine the single lock.
3. The zero-distortion 15-pt config fix (fit `k1`).
4. A static-consistency check that guards the fixed-camera assumption.

Explicit non-goals: no learned corner-finder retrain; no moving-camera / SfM /
DPVO / MegaSaM path; no change to the `metric_15pt_reviewed` authority
semantics (owner 15-pt review stays the only authority door); no new GPU work;
no promotion (`VERIFIED=0`). Raw per-frame observations remain immutable; the
lock and pooled evidence are separate artifacts with provenance/covariance.

## 2. Code seams (real, verified present)

- `threed/racketsport/court_calibration.py`
  - `homography_from_planar_points(world_pts, image_pts)` — DLT homography (line ~161).
  - `solve_camera_pose(world_pts, image_pts, intrinsics)` — PnP via `cv2.solvePnP`,
    already passes `intrinsics.dist` when present (line ~392).
  - `metric_calibration_from_sidecar_and_keypoints(...)` — the 15-pt metric solve
    (line ~600); undistorts keypoints via `undistort_pixels_for_intrinsics(...,
    calibration_intrinsics)` where `calibration_intrinsics =
    calibration_intrinsics_from_sidecar(sidecar)`. **This is where the
    zero-distortion defect lives: if `calibration_intrinsics.dist` is empty/zero,
    `undistort_applied` is False and `homography_from_planar_points` fits DISTORTED
    pixels on a k1=−0.28 camera.**
  - `calibration_intrinsics_from_sidecar` (line ~37), `undistort_pixels_for_intrinsics`
    (line ~85), `_dist_nonzero` (line ~844), `reprojection_error`/`passes_reprojection_gate`.
- `threed/racketsport/court_line_keypoints.py`
  - `detect_court_keypoints_from_image(...)` (line ~56); `keypoints_from_semantic_lines(...)`
    (line ~328); `_select_semantic_lines`, `_fit_line`, `_intersection`, `_refine_line_from_mask`.
    These already produce per-frame line groups + intersection keypoints — the pooling
    layer aggregates their outputs across frames.
- `scripts/racketsport/process_video.py`
  - `_stage_calibration(self)` (line ~1455) — writes `court_calibration.json`,
    `court_zones.json`, `net_plane.json`, `court_line_evidence.json`.
  - `_court_calibration_needs_correction(calibration, evidence)` (line ~6580) =
    `_calibration_is_unverified_or_estimated(...) and not _court_line_evidence_ready(...)`;
    `_court_line_evidence_ready` (line ~6611), `_missing_required_court_line_count` (line ~7150).

## 3. Design

### 3.1 Single-lock path (static reuse)

- Produce ONE authoritative court solve per clip, from either source:
  - **Owner 15-pt tap** → existing `metric_15pt_reviewed` seed (authority door,
    unchanged); OR
  - **Aggregated empty-court / low-occlusion auto-solve** → line-evidence solve
    over the clearest frames (permanently `line_evidence_solved_preview` band).
- Reuse that single solve for every frame instead of re-solving. New artifact
  `court_lock.json` (separate from raw per-frame `court_line_evidence.json`),
  carrying: coordinate space, distortion state, homography, extrinsics,
  residual, covariance, source class, and the frame set it was pooled over.
- `_stage_calibration` selects the lock (owner-tap > pooled auto-solve >
  profile/ARKit/known-court) and marks reuse provenance. Selection-OFF must be
  byte-identical to today's per-frame behavior.

### 3.2 Cross-frame line-evidence pooling

- Under the static assumption, a court line occluded by a player in frame *t* is
  visible at the SAME pixels in other frames. Run
  `detect_court_keypoints_from_image` / `_select_semantic_lines` over a sampled
  set of frames, then aggregate per canonical line with a robust estimator
  (per-pixel/per-parameter median or Theil-Sen on the fitted `_Line` params),
  weighting by per-frame support (`_mask_support_ratio`) and inverse occlusion.
- Refine the single lock from the pooled lines (intersections →
  `homography_from_planar_points`; `solve_camera_pose` for extrinsics). Pooling
  must be deterministic and reproducible; store the pooled evidence and the
  contributing frame indexes in `court_lock.json`.
- Guard: pooling never overwrites raw per-frame `court_line_evidence.json`; it is
  an additional refinement artifact with its own provenance.

### 3.3 Zero-distortion config fix (fit k1)

- The 15-pt metric solve must fit distortion (at least `k1`) rather than assume
  zero. Concretely, in `metric_calibration_from_sidecar_and_keypoints`, ensure
  `calibration_intrinsics.dist` carries the camera's distortion (estimate `k1`
  from the reviewed correspondences when the sidecar/profile lacks it) so
  `undistort_pixels_for_intrinsics` actually undistorts and the homography fits
  UNDISTORTED pixels. Expected effect on the pb.vision demo (k1=−0.28): reviewed
  15-pt residual drops from 19.16px toward the line-solve's 2.61px, so
  `metric_confidence` rises and the in/out stage stops abstaining.
- Keep raw reviewed taps immutable; the distortion-fitted solve is a new solve
  with declared distortion state, not an edit of the seed points. The two prior
  solves already AGREE on the camera (fx 719.3 vs 743.0, ~3%), so this is a
  config fix, not a re-derivation of intrinsics.

### 3.4 Static-consistency check (guard the assumption)

- Before trusting the single lock, verify the camera is actually static: measure
  cross-frame drift of pooled court-line parameters / keypoint intersections (or
  a sparse background-feature homography residual) across the clip.
- If drift exceeds a pre-registered threshold (proposed: median court-corner
  reprojection drift > ~2px, or a monotonic trend indicating pan/zoom), FLAG
  `camera_motion_suspected=true`, fall back to per-frame behavior for the moving
  spans, and surface it as a typed abstention — never silently reuse a stale
  lock on a moving clip. This makes the v1 static assumption falsifiable.

## 4. Acceptance (pre-registered, frozen before running)

All CPU-only, on the existing static clips + the pb.vision demo:

1. **Reuse fidelity:** reuse-lock reprojection residual within **~1px** of the
   per-frame solve on the static clips (pooled lock should be ≤ per-frame, never
   materially worse).
2. **In/out unblocks:** on the pb.vision demo, the distortion-fitted 15-pt solve
   raises `metric_confidence` above the in/out abstention threshold — **in/out no
   longer abstains** — with residual moving from 19.16px toward ~2.6px.
3. **No authority regression:** the `metric_15pt_reviewed` authority gate is
   unchanged; owner-tap seeds still solve at least as well; owner semantics
   (unmarked points = explicitly not-in-frame) preserved.
4. **Static-guard true-positive/negative:** the consistency check flags a
   synthetically panned clip and does NOT flag the genuinely static clips.
5. **Determinism / immutability:** independent rebuild byte-identical;
   selection-OFF byte-identical to today; raw per-frame evidence immutable.

Report `adopt` / `reject` / `partial` / `no-attempt` against these, save under
this lane dir with source/code/model/config versions. Preview-band only; a
smaller residual, a browser load, or test-green is NOT a promotion (North Star
§6.12). Promotion to the CAL gate still requires NS-02 independent labels
(owner-viewpoint PCK@5 ≥0.95, net-height ≤2cm).

## 5. Handoff required

- `court_lock.json` schema (space, distortion state, homography, extrinsics,
  residual, covariance, source class, pooled frame set, `camera_motion_suspected`).
- The distortion-fitted 15-pt solve wired behind the existing authority door,
  with a config flag; default-OFF until acceptance #1-#5 pass on the static clips.
- Static-consistency check output threaded into `_stage_calibration` as a typed
  abstention, not a hard failure.
- No edits to `player_global_association.py` / other lanes' files; CAL seams only.
