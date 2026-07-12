# Court-precision reproduction map — 2026-07-12

Status: research recommendation only. `VERIFIED=0`; no best-stack delta. The expected movements below are preregistration ranges, not measured results. Each rank must start from the sibling harness baseline and use an unchanged scorer. Outdoor/Indoor labels stay unread.

## Ranking rule

The ordering favors changes that (a) exploit known regulation geometry, (b) are testable without the owner, (c) improve both static precision and temporal lock, (d) preserve typed distortion/time coordinates, and (e) fail closed. Learned court regressors rank below geometry because the last wave proved real in-family learning at 2.8 px but catastrophic cross-source transfer; the 21M model reached about 290 px on held-out source families while the tiny CNN reached 13.5 px.

## 1. Replace both refinement no-ops with one guarded, distortion-aware point-and-line optimizer

- **Hypothesis:** The current seed is often close enough for local optimization, but no production-quality optimizer is actually applied. A robust point+line objective over an undistorted line-distance field can reduce residuals without another label campaign.
- **Exact wiring:** implement and test the objective in `threed/racketsport/court_proposal_optimizer.py::{score_homography_support,run_guarded_line_refinement,refine_homography_with_lines}`; reuse `threed/racketsport/overlapping_court_calibration.py::{fit_joint_camera_point_line_lm,fit_joint_distorted_camera_lm}` and `court_auto_evidence.projected_template_line_segments`. Keep floor paint and net-top evidence as separate 2D/3D residual families. Future integration owner gets a saved hunk for `scripts/racketsport/process_video.py::_stage_calibration` / `threed/racketsport/orchestrator.py::{ManualCalibrationRunner,ExternalCalibrationRunner}`; this window does not edit them.
- **Harness movement:** M1 median/p90 residual **20–50% lower** when the seed is inside the attraction basin; M3 net residual **10–30% lower** only when physical net evidence exists; M5 normal-to-line cm sensitivity **10–30% lower**. M2 must not regress more than 0.2 px on static clips.
- **Effort:** 6–10 Codex-lane-hours.
- **Kill:** reject if the optimizer returns the seed on >80% of eligible frames without a named reason, if two internal clips do not each improve coverage-weighted M1 p90 by at least 15%, if any fixed-visibility line worsens >1 px p90, if M2 static p95 rises >0.2 px, or if gains disappear under a second independent line extractor.
- **Labels:** NONE.
- **Fence interaction:** code can live entirely outside the forbidden integration spine. `court_calibration.py`, `court_calibration_metric15.py`, `process_video.py`, `orchestrator.py`, coordinates/timebase, BALL and BODY files remain handoff-only.
- **Research basis:** point+line nonlinear refinement is the central contribution of [PnLCalib](https://arxiv.org/abs/2404.08401); MAGSAC++ provides a scale-marginalized robust fitting precedent ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html)).

## 2. Build subpixel paint-center evidence, not integer Hough intersections

- **Hypothesis:** Court paint is a finite-width bright band. Fitting its two signed-gradient edges and using their centerline produces lower-bias, subpixel observations under blur/shadows than treating arbitrary white pixels or Hough endpoints as the line.
- **Exact wiring:** extend `threed/racketsport/court_line_bank.py` with a fixed candidate API returning centerline, covariance, support length and polarity; consume it in `court_auto_evidence.detect_image_line_segments`, `court_line_keypoints.detect_court_keypoints_from_image`, and rank-1 distance fields. A/B classical LSD, EDLines/ELSED if locally available, and DeepLSD as an optional evidence provider. Reuse `overlapping_court_calibration.shadow_removal_preprocess`, but require a raw-image arm.
- **Harness movement:** M1 evidence coverage **+10–25 percentage points**, median **0.5–2.0 px lower**, p90 **1–4 px lower**; M3 post/tape coverage improves directionally. No promised M5 change until rank 1 consumes covariance.
- **Effort:** 6–12 Codex-lane-hours.
- **Kill:** reject any provider that fails to improve M1 p90 by >=0.5 px on both internal clips, loses >5 points of fixed visibility coverage, selects the wrong paint family in a dual-line synthetic/metamorphic test, or exceeds 12 ms/frame CPU at 1080p for the classical L0 candidate. DeepLSD survives server-only if it improves >=1 px p90 over classical with no family-identity regression.
- **Labels:** NONE.
- **Fence interaction:** no forbidden spine edit. The integration owner later selects the evidence provider via a new explicit calibration sub-config; no best-stack/default flip from this lane.
- **Research basis:** DeepLSD explicitly combines learned attraction fields with accurate handcrafted detection and vanishing-point refinement ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Pautrat_DeepLSD_Line_Segment_Detection_and_Refinement_With_Deep_Image_Gradients_CVPR_2023_paper.html)); SOLD2 is self-supervised and occlusion-aware ([CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Pautrat_SOLD2_Self-Supervised_Occlusion-Aware_Line_Description_and_Detection_CVPR_2021_paper.pdf)).

## 3. Make camera motion a per-frame court lock with keyframe re-detection and drift resets

- **Hypothesis:** A correct reference calibration plus a current-pixel-to-reference transform is sufficient for placement only while optical flow stays valid. A stateful lock that predicts, measures, estimates covariance, and re-detects at evidence keyframes will make every frame explicit and stop drift.
- **Exact wiring:** build a new court-specific temporal module beside `threed/racketsport/camera_motion.py`; initialize at a trusted reference H, predict with the existing M transform, update using rank-2 line/keypoint observations, and retain top-K identity hypotheses. Because `camera_motion.frames[].M` maps current pixels to reference pixels, the per-frame court-to-image mapping is `inv(M_t) @ H_reference`; do not silently reverse it. Emit per-frame H/camera covariance, reset reason, source and missing state. Future integration handoffs target `process_video.py::_stage_camera_motion`, `_placement_camera_motion_path`, BALL-arc calibration consumption and BODY input packaging.
- **Harness movement:** on moving-camera frames M2 evidence-referenced p95 **40–70% lower** and long-horizon drift **50–80% lower**; on static frames added jitter **<=0.2 px p95**; M1 p90 stable or lower frame by frame.
- **Effort:** 10–18 Codex-lane-hours.
- **Kill:** reject if a known static clip jitters >0.2 px p95, if any 2-second moving window exceeds its existing static-calibration M1 p90, if reset latency exceeds 15 frames after evidence recovers, if covariance shrinks during missing evidence, or if a wrong symmetric hypothesis persists past two keyframes.
- **Labels:** NONE for internal diagnostics; gold-capture required before promotion.
- **Fence interaction:** this window writes no camera-motion or consumer code. All changes to process/orchestrator/BALL/BODY are serialized integration handoffs.
- **Research basis:** BroadTrack emphasizes direct field correspondences to reduce drift in moving-camera sports calibration ([WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.pdf)); BHITK explicitly models inter-frame motion and keypoint covariance in a two-stage filter ([paper](https://arxiv.org/abs/2311.10361)).

## 4. Ship profile-first guided confirmation for recurring courts and devices

- **Hypothesis:** The most reliable v1 path is not universal auto-find: reuse a known device/lens/court pose and verify it against current paint before asking the owner to adjust a small number of anchors.
- **Exact wiring:** reuse `profile_registry.py::{fingerprint_capture,rank_court_profile_matches}` and `court_profile_match.py::{verify_outer_court_lines,decide_court_profile_reuse}`. Keep the existing conjunction: fingerprint + paint color + at least 3 outer lines under 4.8 px median/12.3 px p95. A future integration handoff routes a successful profile into `process_video.py::_resolved_court_calibration_path`; failed verification opens the existing correction-task seam rather than falling back silently.
- **Harness movement:** accepted known-profile clips should meet M1 <=4.8 px median and <=12.3 px p95 by construction, while auto-coverage is reported separately; M2 static drift remains honest-absent unless per-frame transforms exist; M5 becomes stable across repeat sessions.
- **Effort:** 4–8 Codex-lane-hours plus UI/integration owner time.
- **Kill:** reject auto-reuse if any accepted clip fails the four-line bar, if two profiles pass without an owner tag, if device/lens/crop identity is incomplete, or if a profile refresh overwrites raw prior evidence.
- **Labels:** NONE for reuse; gold-capture for profile creation/validation.
- **Fence interaction:** profile libraries are available; all routing/UI changes are handoffs. No current forbidden file edit.

## 5. Add full-camera constraints only where identifiable: distortion, vanishing directions, and a physical net

- **Hypothesis:** Edge-of-frame placement errors need a camera model, not a free homography. Known intrinsics/distortion plus two orthogonal court line families and non-planar net posts can constrain pose; a single planar view cannot safely free every intrinsic.
- **Exact wiring:** diagnostic lane around `overlapping_court_calibration.fit_joint_camera_point_line_lm`, `fit_metric_plane_camera_lm`, `court_line_bank.cluster_line_family_directions`, `net_anchor_court` and `net_plane.project_net_plane`. Freeze fx/fy/cx/cy/distortion from native/profile metadata when available; optimize pose plus narrow distortion deltas. Keep 34-inch center sag and 36-inch post heights explicit rather than the current uniform net prior.
- **Harness movement:** M1 edge-line p90 **10–30% lower**, M3 post/tape residual **20–50% lower**, and M5 corner/near-baseline normal displacement **15–40% lower**. Central M1 may change little.
- **Effort:** 8–16 Codex-lane-hours.
- **Kill:** reject if parameter covariance/condition number says the solve is unidentifiable, if focal/principal point move beyond declared device uncertainty, if held-out line residual worsens, if net sag choice is not explicit, or if improved self-residual does not improve M5 on perturbed fixtures.
- **Labels:** NONE with trusted native/profile intrinsics; gold-capture for independent validation.
- **Fence interaction:** future calibration artifact changes belong to the integration/calibration owner. This lane only recommends a separate candidate artifact and handoff.
- **Research basis:** TVCalib optimizes camera pose/focal length from segment reprojection rather than only homography ([WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Theiner_TVCalib_Camera_Calibration_for_Sports_Field_Registration_in_Soccer_WACV_2023_paper.pdf)); ProCC warns that homography-only evaluation misses non-planar geometry and distortion ([CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Magera_A_Universal_Protocol_to_Benchmark_Camera_Calibration_for_Sports_CVPRW_2024_paper.html)).

## 6. Fuse native ARKit/gyro priors and model rolling shutter instead of correcting pixels late

- **Hypothesis:** For product-phone capture, synchronized VIO rotation and row exposure time can distinguish true camera motion from line-detector jitter and correct handheld/rolling-shutter warps that one homography cannot express.
- **Exact wiring:** consume the versioned sidecar’s per-frame AR camera transform, intrinsics, tracking state, gyro samples, crop/orientation and rolling-shutter row time. Predict row-dependent rays/transforms before rank-2 evidence update; retain raw and corrected timestamps. Future handoffs target the capture/timebase/coordinate owners and the rank-3 temporal module, never late destructive correction.
- **Harness movement:** native moving-capture M2 p95 **30–60% lower**, rolling-shutter top-vs-bottom line disagreement **20–50% lower**, and M1 p90 **0.5–2 px lower** on fast shakes; no expected benefit on stable imported clips.
- **Effort:** 10–20 Codex-lane-hours after NS-01 sidecar/timebase truth.
- **Kill:** stop if sensor/video alignment uncertainty exceeds 0.5 frame, tracking state is not normal, row time is missing, corrected evidence is not better than raw on a preregistered shake fixture, or static clips gain >0.2 px jitter.
- **Labels:** NONE for implementation; gold-capture for proof.
- **Fence interaction:** currently blocked from source edits by timebase/coordinate/capture ownership. Deliver only interface requirements.
- **Research basis:** Apple states ARKit combines image analysis and motion sensing through visual-inertial odometry and exposes tracking-quality state ([Apple documentation](https://developer.apple.com/documentation/arkit/understanding-world-tracking)); gyro-based phone stabilization can model rolling-shutter warping in real time ([Stanford report](https://aperture.stanford.edu/papers/stabilization/)).

## 7. Promote uncertainty plumbing before any officiating-style claim

- **Hypothesis:** A point estimate cannot support honest in/out. Calibration-parameter covariance, image-line covariance, temporal process noise and ball-localization uncertainty must propagate to the boundary-normal distance and trigger `too_close_to_call`.
- **Exact wiring:** extend the sibling harness M5 into a candidate covariance artifact using Jacobians or bootstrap/Monte Carlo over observed line/keypoint covariance; reuse `placement.homography_world_covariance` and `ball_line_calls.classify_bounce`. A future integration hunk adds covariance fields to calibration/per-frame-court artifacts and makes stage-6 in/out consume the same authority rather than its separate manual-corner solve.
- **Harness movement:** accuracy metrics need not improve. Acceptance is **100% covariance coverage** for emitted court positions, calibrated empirical coverage on synthetic/metamorphic perturbations, and a boundary-normal decision band in centimetres at every canonical line point. M5 must expand under worse evidence and farther extrapolation.
- **Effort:** 6–12 Codex-lane-hours.
- **Kill:** reject if covariance shrinks when evidence is removed/noised, empirical 95% coverage falls below 90% on controlled perturbations, correlation between corner errors is ignored, or any call inside the band is emitted as definite in/out.
- **Labels:** NONE for plumbing; gold-capture required to calibrate real coverage.
- **Fence interaction:** implement as a separate artifact first. BALL/calibration/schema integration is handoff-only.
- **Research basis:** Criminisi et al. derive homography covariance and propagate both correspondence and image-localization error into world measurements ([paper summary](https://www.sciencedirect.com/science/article/abs/pii/S0262885698001838)); the current ITF Gold criterion permits only 5 mm average and 10 mm maximum discrepancy, a standard far beyond an ungated monocular point estimate ([ITF procedures](https://www.itftennis.com/media/14686/elc-evaluation-procedures-v28-june-2025.pdf)).

## 8. Use the owner’s 100 diverse frames for a small semantic evidence model, not another direct court regressor

- **Hypothesis:** The diversity wall can be attacked by source-balanced supervision of paint masks/line-family heatmaps/visibility, while a geometric solver retains metric authority. A tiny/mid model should generalize better than the 21M family memorizer at this data scale.
- **Exact wiring:** after CVAT review, use `court_keypoint_labels.py`, `build_real_court_corpus.py`, `court_keypoint_lines.py` and a capacity-right-sized model. Split strictly by source/venue/channel; train semantic paint, visibility and uncertainty heads; pass their evidence to ranks 1–3. Do not train on protected clips, pb.vision, projected candidates, or safe-default-empty pseudo labels.
- **Harness movement:** on source-disjoint owner views, first survival target is PCK@5 **+0.30 absolute** and median <25 px versus frozen baseline; M1 evidence coverage **+15 points** without worse p90. Promotion still requires owner-viewpoint PCK@5 >=0.95 and the named CAL gates.
- **Effort:** 12–24 Codex-lane-hours plus owner labeling and one bounded GPU run.
- **Kill:** stop if any source group leaks across splits, fewer than 60 distinct legal views remain, a tiny model does not beat the 21M model on disjoint sources, PCK@5 gain is <0.30, median remains >=25 px, or M1 improvement comes from lower fixed visibility coverage.
- **Labels:** owner-100-frames.
- **Fence interaction:** no current calibration/default edit. A surviving checkpoint remains a proposal/evidence candidate in `best_stack.json` until independent gate and owner ruling.
- **Research basis:** KpSFR’s reported dataset contains 3,812 sequence images from 43 videos and uses keypoint-conditioned heatmaps plus DLT/RANSAC ([project page](https://ericsujw.github.io/KpSFR/)); this scale contrast is why 100 frames should supervise evidence and geometry, not promise a universal regressor.

## 9. Run zero/few-label foundation and test-time methods only as bounded evidence ablations

- **Hypothesis:** SAM2 masks or DINOv2 features may stabilize court-surface/paint extraction, and self-supervised alignment may improve the attraction basin, but none has court semantics or metric authority out of the box.
- **Exact wiring:** prompt SAM2 with the coarse court polygon/paint points and convert masks to line evidence; use DINOv2 only for frame/keyframe similarity or clustering; test AnyCalib as an intrinsics prior; test self-supervised shape alignment/ECC as a local residual after geometric initialization. Never regress court coordinates directly from these outputs and never use them as labels.
- **Harness movement:** no positive magnitude assumed. A provider survives only if M1 p90 improves >=1 px or coverage >=10 points on both internal clips at fixed visibility, M2 does not regress, and runtime fits its tier.
- **Effort:** 4–8 Codex-lane-hours per provider; cap the initial bake-off at three providers.
- **Kill:** immediate reject for wrong semantic masks, multi-line family confusion, confidence collapse, >2x server runtime without >=1 px gain, any self-training on its own predictions, or any attempt to call zero-shot output GT.
- **Labels:** NONE initially; owner-100 only for later calibration, not pseudo-GT.
- **Fence interaction:** standalone evidence artifacts only. No integration/default edits this window.
- **Research basis:** SAM2 is promptable video segmentation with streaming memory ([paper](https://arxiv.org/abs/2408.00714)); DINOv2’s general features depend on large diverse curated data ([paper](https://arxiv.org/abs/2304.07193)); self-supervised sports alignment reported best results after about 240 labeled fine-tuning images, not purely zero-shot authority ([WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Shi_Self-Supervised_Shape_Alignment_for_Sports_Field_Registration_WACV_2022_paper.html)); AnyCalib estimates model-agnostic intrinsics but not court pose ([paper](https://arxiv.org/abs/2503.12701)).

## Refinement defect diagnosis plan

1. Freeze four inputs: one good in-family row, one failed cross-source row, Wolverine and Burlington. Save raw predictions, confidence, visible floor/net split and source image size.
2. Instrument `refine_keypoint_xy_with_planar_homography` without changing behavior: candidate count, non-degenerate four-subset count, best inlier count, per-point residual, fallback reason, output delta and whether net-top points entered the planar fit.
3. Reproduce the exact-no-op cases. Expected diagnosis: fewer than eight predictions within 30 px. If not, test serialization/rounding and evaluator flag plumbing.
4. Reproduce the worse cases. Compare floor-12 only against all-15. The current code drops z from net-top points; a worse all-15/floor-12-safe result confirms model mismatch.
5. Separately invoke `court_proposal_optimizer.refine_homography_with_lines` and assert its explicit `optimizer_not_wired` result. Do not confuse this stub with the learned-keypoint postprocess.
6. Compare four candidates with the frozen harness: raw, floor-only confidence-weighted MAGSAC++, offline `fit_joint_camera_point_line_lm`, and the new guarded line-distance optimizer. Score fixed visibility, not only self-residual.
7. Add metamorphic tests: exact homography plus 1 px Gaussian noise must improve or remain within tolerance; one gross outlier must be rejected; three non-planar net tops must never enter a planar DLT; a wrong parallel line family must lose template competition; zero evidence must return the seed plus covariance inflation.
8. Only then create an integration handoff. A residual decrease is necessary but not sufficient; M1 fixed-visibility p90, M2 static jitter, M5 boundary-normal centimetres and downstream placement impact must all survive.

## Temporal court-lock design for moving cameras

State is a calibrated camera or normalized 8-DoF homography plus covariance, explicit identity hypothesis and reference generation. Prediction uses gyro/VIO when available, otherwise the existing current-to-reference motion transform and a process-noise model that grows with blur, low inlier ratio and missing frames. Measurement uses rank-2 paint centerlines, intersections, net posts/tape and optional coarse learned evidence. Update uses robust innovation gating; a measurement never silently snaps the state.

Every frame emits `measured`, `predicted`, `missing` or `reset`, plus H/camera parameters, covariance, evidence coverage, line-family identity, reference keyframe and reset reason. Re-detect on startup, every 0.5–1.0 seconds, after a scene cut, when covariance crosses the M5 product limit, or after line evidence recovers. Keep top-K symmetric/multi-line hypotheses until temporal evidence separates them. A hard cut creates a new generation; it never warps the old court through the cut.

Placement can already map current pixels into the reference frame through `camera_motion.json`. The immediate integration gap is broader: BALL 3D, the calibration artifact, replay overlays and any direct homography consumers need the same per-frame authority. Until those handoffs land, per-frame court lock is diagnostic and placement-scoped.

## L0 live on-device implication

L0 should be a court-lock health monitor, not officiating. Use low-resolution classical centerline evidence plus ARKit/VIO prediction at capture cadence, with a heavier keyframe detector at 1–2 Hz. Show framing/lock confidence and ask for adjustment between rallies; recording must never stall. Live output may say `lock_good`, `lock_degraded`, `camera_moved`, or `too_close_to_call`; it must not emit officiating-grade calls. SwingVision’s own setup guidance makes full-court calls dependent on mounting/framing and 60 fps ([official guide](https://swing.vision/hardware)), while certified ITF Gold systems are evaluated across venues, lighting, player occlusion and reliability—not merely per-frame latency ([ITF procedures](https://www.itftennis.com/media/14686/elc-evaluation-procedures-v28-june-2025.pdf)).

## Handoff order

1. Baseline and anti-gaming changes to the sibling harness.
2. Rank 1 defect instrumentation and rank 2 evidence A/B in parallel, without spine edits.
3. Adopt/reject ranks 1–2 on frozen Burlington/Wolverine diagnostics.
4. Build rank 3 temporal artifact using the surviving evidence path.
5. Wire rank 4 profile reuse for v1 guided confirmation.
6. Add ranks 5–7 only after coordinate/timebase/native sidecar contracts settle.
7. When owner labels return, run rank 8 once with source-grouped splits.
8. Treat rank 9 as a capped bake-off; idle compute is not evidence.

BEST-STACK DELTA: none. Every result remains diagnostic or candidate evidence until the North Star gate passes.
