# Precise court tracking research consult — 2026-07-12

Status: read-only research consult. `VERIFIED=0`. No code, config, documentation, model, manifest, or best-stack selection changed. External product claims below are attributed; they are not measurements of this repository. pb.vision remains a reference diagnostic only, never ground truth or training truth.

## Executive ruling

The highest-leverage move is not another universal court regressor. This repository already has most of the necessary geometry, evidence extraction, temporal motion, profile, and uncertainty pieces, but the production path stops at a seed or a score in several places:

1. the named proposal optimizer is an explicit stub;
2. the learned-keypoint homography postprocess often returns its input and can incorrectly flatten net-top points onto the floor;
3. the line-evidence path scores a fixed calibration but does not refine it;
4. camera motion is a reference-frame placement aid, not a per-frame calibrated court state;
5. profile reuse has conservative validation logic but is not routed into the default pipeline; and
6. manual in/out has a second, native-pixel four-corner homography authority that is separate from `court_calibration.json`.

The reproduction map therefore ranks: a guarded distortion-aware point+line optimizer; subpixel paint-center evidence; a keyframed temporal court lock; profile-first guided confirmation; identifiable camera/distortion/net constraints; native gyro/ARKit and rolling-shutter priors; uncertainty propagation; then a small semantic evidence model on the owner's 100 source-diverse frames. Foundation models are bounded evidence extractors only. Exact ranks, wiring seams, preregistered expected movement, effort and kill rules are in [PLAN.md](PLAN.md).

This conclusion is consistent with the banked negative, not an attempt to route around it: real labels worked in-family at about 2.8 px, while cross-source transfer failed (about 290 px for the 21M model and 13.5 px for the tiny CNN). The binding constraint is viewpoint/source diversity. The 100-frame owner-label package is the first legitimate attack on that constraint; it does not make a new learned regressor the best immediate precision mechanism.

## Scope, evidence and provenance

- Read first: `NORTH_STAR_ROADMAP.md`, `AGENTS.md`, relevant `RUNBOOK.md` sections, the court-wave close/ruling, train/data/ext reports, the harvest report, pb.vision forensics and scorecards, sprint plan, and sibling harness spec.
- Protected-label rule observed: no Outdoor/Indoor labels were opened or used. Burlington and Wolverine appear only as permitted internal diagnostics.
- Local source census covered the court/calibration modules under `threed/racketsport/`, the relevant scripts, `process_video.py`, `orchestrator.py`, selected defaults, and consumers. Structured results are in [census.json](census.json).
- The lane brief says `court_calibration.py` and `court_calibration_metric15.py` carry another lane's uncommitted edits. The initial file-scoped `git status`, unstaged diff and staged diff showed no delta for either path, so both were nevertheless treated as foreign/read-only. During final validation an unstaged `court_calibration_metric15.py` hunk from another lane appeared (coordinate helpers, extrinsics inversion and a coordinate-contract payload); `court_calibration.py` remained clean. HEAD concurrently advanced from `0e97c09fe1218b1a754222f879e40ae996e24657` to `a57476a39ffbdae6d137e958c6c8d81bf8e3c04f` through unrelated commits. This consult did not edit either file.
- Generated evidence was treated as a dated observation, never current truth unless directly inspected. No lane result changes `VERIFIED=0`.

## A. Repository census

### Current calibration routes

| Route | What happens now | Authority/artifacts | Precision implication |
|---|---|---|---|
| Manual four corners | `_stage_calibration` constructs/uses a capture sidecar, then `ManualCalibrationRunner` solves from the ordered four floor corners. | `court_calibration.json`, `court_zones.json`, `net_plane.json`, `court_line_evidence.json` | Operational and default-capable, but four taps do not independently validate distortion, net geometry, edge-line bias, or per-frame camera motion. |
| Native/ARKit metric keypoints | A sidecar plus `court_keypoints.json` enters `metric_calibration_from_sidecar_and_keypoints`; automatic evidence remains fail-closed. | Same main calibration family, with typed camera/intrinsic provenance | Best route to native intrinsics and metric constraints, provided coordinate/orientation/time contracts remain explicit. |
| Reviewed metric15 | An explicit `--court-calibration` or auto-discovered `labels/court_calibration_metric15pt.json` is validated by `ExternalCalibrationRunner`; the source must declare `metric_15pt_reviewed`. | Reviewed `court_calibration.json`-compatible artifact | Strongest reviewed single-view seed. Its non-planar net points must not be flattened into a floor homography. |
| Profile reuse | `profile_registry.py` fingerprints captures; `court_profile_match.py` verifies paint color plus at least three outer lines with 4.8 px median/12.3 px p95 limits. | Profile records/candidate match decision | Implemented but not called from `process_video.py` or `orchestrator.py`; high-value v1 path is currently dormant. |
| Auto-find preview | `--allow-auto-court-corners-preview` samples five frames, uses `court_line_keypoints`, picks the highest-confidence candidate, seeds four corners, and runs the manual solver. Server code enables the preview in its current path. | Preview-derived normal calibration artifacts | A coarse proposal is converted to calibration; sampling only five frames and ranking detector confidence is not a precision or temporal lock proof. |
| Proposal previews | Regulation-line proposals, detector v2 or net-anchor tooling can write proposal/correction artifacts. | Proposal artifacts, blockers and overlays | Fail-closed by design. The default pipeline can notice some pre-existing artifacts but does not make them calibration authority. |

### How calibration is consumed

The important coordinate chain is explicit but split across consumers:

1. Tracking reads `court_calibration.json`; `person_fast.py` projects the bbox-bottom foot point into `WORLD_XY_HOMOGRAPHY_M`. That world point also controls court membership/filtering.
2. `camera_motion.json` stores a per-frame matrix `M` mapping current-frame pixels to the reference frame. Placement applies `M`, then the inverse reference homography, and can undistort when the calibration declares an undistorted-pixel homography. It emits placement and rewrites track `world_xy`.
3. BODY grounding consumes the track world point and camera-motion-adjusted grounding pixels, placing output on `court_Z0`. It does not independently solve court pose.
4. BALL 3D reconstruction consumes `court_calibration.json` and writes the calibrated arc artifact (`ball_track_arc_solved.json`). Its accuracy therefore inherits calibration and observation error.
5. Manual BALL in/out is different: when the original `court_corners.json` is supplied, it constructs its own native-pixel four-corner homography and writes `ball_inout_summary.json`. Improving only `court_calibration.json` cannot improve that line-call authority until an integration owner unifies the paths.
6. Replay/world assembly consumes the calibrated track, BODY and BALL outputs in `court_Z0`; stats/zone facts use track `world_xy` and the regulation polygons in `court_zones.json`.

For a per-frame court-to-current-image mapping, the present direction contract implies `inv(M_t) @ H_reference`; treating `M_t` as the reverse mapping would silently corrupt placement. A future temporal artifact should save direction, pixel space, reference frame, uncertainty and reset reason on every frame, not rely on convention.

### Relevant module families

The structured census contains 50 relevant library modules and 29 scripts with `wired_default`, preview, standalone/offline, unwired, or dead-stub status. The main groups are:

- **Default authority and consumers:** `court_calibration.py`, `court_auto_evidence.py`, `court_line_evidence.py`, `court_templates.py`, `court_zones.py`, `placement.py`, `person_fast.py`, `net_plane.py`, `ball_physics3d.py`, `worldhmr.py`, and `virtual_world.py`.
- **Preview/proposal stack:** `court_line_keypoints.py`, `court_line_bank.py`, regulation proposals, `court_detector_v2*`, and `net_anchor_court.py`. These are reusable seed/evidence generators but are not accuracy authority.
- **Dormant precision machinery:** `overlapping_court_calibration.py` contains real LM homography, focal/pose/radial, point+line, metric-plane and diagnostic full-intrinsics solvers; `court_profile_match.py` and `profile_registry.py` contain guarded profile reuse; `court_motion_mode.py` is an unwired policy helper.
- **Banked learned/data paths:** the keypoint models, corpus builders, label-space diagnostics and gates remain useful for the owner-100 evidence-model route. The killed synthetic-only, pseudo-projection, tennis-transfer, legacy and 21M transfer recipes must not be repeated.
- **Downstream audit surfaces:** `court_cal_impact.py`, `person_court_membership.py`, placement covariance, BALL line calls and replay/world outputs can reveal whether lower pixel residual actually improves metric positions.

### Why refinement is currently a no-op or worse

There are distinct defects, so one blanket “turn on refinement” experiment would be uninformative:

1. `court_proposal_optimizer.score_homography_support` returns zero support and `run_guarded_line_refinement` returns the initial homography; its wrapper records `optimizer_not_wired`. This is an exact implementation stub.
2. `court_keypoint_net.refine_keypoint_xy_with_planar_homography` enumerates four-point subsets, needs at least eight inliers within 30 px, and otherwise returns the raw prediction. Exact equality is therefore an expected fallback, not evidence that optimization converged.
3. That keypoint refiner sends every selected metric point through only world `(x,y)`. The three net-top points have nonzero `z`; flattening them onto the court plane makes the model internally wrong and can increase held-out residual.
4. The keypoint refiner ignores prediction confidence, detected line support, distortion model and semantic point families. Its four-point proposal can be numerically valid but visibly wrong.
5. `court_auto_evidence` evaluates a fixed calibration. Reusing its nearest expected-line search as the optimization objective without frozen visibility and competing-line penalties would be circular and could reward the wrong parallel paint stripe.
6. The real offline LM routines are not default-wired. The most flexible single-plane intrinsic fit is explicitly diagnostic because one plane does not identify all camera parameters safely.
7. `camera_motion` can improve placement relative to one reference, but it is not a per-frame calibrated camera/court artifact, does not reset against semantic court evidence, and is not consumed by BALL 3D. Drift or a bad reference therefore persists asymmetrically.

The reproduction/diagnosis protocol in PLAN freezes good, failed-cross-source, Burlington and Wolverine inputs; logs subset degeneracy/inliers/fallback/output delta; splits floor and net evidence; runs identity/raw-floor-only/weighted robust/line-LM arms; and requires independent evidence improvement. That distinguishes a stub, no-inlier fallback, bad plane model, poor attraction basin and objective overfit.

## B. External research

### Production systems: what is public and what is not

| System | Public evidence | Engineering inference and limit |
|---|---|---|
| pb.vision | The vendor says a phone or facility camera tracks the court, players and shots and reconstructs play in 3D ([pb.vision](https://pb.vision/)). | Public material does not disclose a calibration algorithm or independent error protocol. The local Wolverine export's approximately 5.67 px court residual is a repository diagnostic only; it cannot be GT, a training label, or proof of vendor architecture. |
| SwingVision | The vendor advertises single-camera line calling and about 97% accuracy for close calls within 10 cm under ideal 60-fps conditions ([line-calling page](https://swing.vision/)); its court product repeats 97% close-call and >99% overall claims ([Swing Court](https://swing.vision/swing-court)). Apple reports its on-device pipeline uses the Neural Engine and that 1080p/60-fps analysis is intensive ([Apple developer story](https://developer.apple.com/news/?id=0pg4dthn)). | These are vendor/product claims, not an ITF-style public discrepancy dataset. Patent searches expose broad mobile video/object-tracking and planar object-tracking inventions ([US 10,467,478](https://patents.justia.com/patent/10467478), [EP 3,393,608](https://patents.google.com/patent/EP3393608B1)), but not enough detail to reproduce the present court-lock stack. |
| PlaySight | Its tennis product describes a permanently installed six-HD-camera system and kiosk with tracking, trajectory and line-calling features ([PlaySight](https://my.playsight.com/Home/WhatIsPlaySight)). | This is a useful architecture contrast: persistent multi-view installation and controlled court setup are materially different from one moving phone. No public line-error distribution was found. |
| Hawk-Eye / Foxtenn / Zenniz | The ITF's current classification page lists Hawk-Eye and Foxtenn as Gold and Zenniz as Silver in specified review/live settings ([ITF classified systems](https://www.itftennis.com/en/about-us/tennis-tech/classified-elc-systems/)). The 2025 procedure requires live response around 0.15 ± 0.05 s and, for Gold accuracy, average absolute discrepancy no more than 5 mm and maximum no more than 10 mm in the defined bounce set ([ITF evaluation procedure](https://www.itftennis.com/media/14686/elc-evaluation-procedures-v28-june-2025.pdf)). Hawk-Eye publicly describes optical tracking and camera calibration as core technologies but not the algorithms ([Hawk-Eye](https://www.hawkeyeinnovations.com/)). Zenniz says it combines 30 audio sensors with four cameras and claims about 7 mm error ([Zenniz technical overview](https://zenniz.com/smart-corner/how-does-zenniz-work)). | The ITF numbers are evaluation requirements, not proof that a monocular phone can reproduce these systems. Zenniz's 7 mm number is vendor-reported. I found no public Foxtenn implementation details sufficient to justify common camera-count/frame-rate lore, so none is asserted here. |
| Broadcast AR | Viz Arena advertises image-based camera tracking that keeps live graphics registered to a field ([Viz Arena](https://routing.vizrt.com/products/viz-arena-update/)). | This establishes commercial field-lock precedent, not officiating accuracy. Broadcast systems can exploit stable field markings, production hardware and re-detection; those ideas transfer, their claimed product capability does not establish our metric accuracy. |

The ITF history document is especially useful for evaluation design: its described reference setup used calibrated high-speed cameras at 2,000 Hz, with about 1 mm measurement capability and stated ±0.9 mm uncertainty ([ITF line-calling history](https://www.itftennis.com/media/12242/line-calling.pdf)). The lesson is not that our phone is close to Gold; it is that self-residuals and visual overlays are nowhere near an independent officiating reference.

### Sports-field registration research

- **TVCalib** optimizes pose and focal length from projected field-segment agreement and evaluates on SoccerNet and World Cup broadcast data; it treats segment reprojection as the geometric objective rather than a four-corner afterthought ([WACV 2023 paper](https://openaccess.thecvf.com/content/WACV2023/papers/Theiner_TVCalib_Camera_Calibration_for_Sports_Field_Registration_in_Soccer_WACV_2023_paper.pdf)). It supports our point+line and camera-model ranks, but soccer broadcast diversity is not pickleball transfer proof.
- **KpSFR** reports 3,812 images from 43 videos and combines keypoint-conditioned heatmaps/segmentation with DLT/RANSAC ([project and paper](https://ericsujw.github.io/KpSFR/)). This scale is an important warning: 100 owner frames can calibrate semantic evidence and uncertainty, but should not be sold as universal direct-regression coverage.
- **No Bells, Just Whistles** derives additional geometric keypoints, trains HRNet, and uses DLT/RANSAC across real sports datasets ([CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Gutierrez-Perez_No_Bells_Just_Whistles_Sports_Field_Registration_by_Leveraging_Geometric_CVPRW_2024_paper.html)). More geometrically meaningful evidence can help, but its reported cross-dataset setup does not erase our observed source-family wall.
- **PnLCalib** explicitly performs nonlinear calibration refinement from both points and lines on sports-field datasets ([paper](https://arxiv.org/abs/2404.08401)). It is the closest published analogue to the dormant point+line LM machinery already in this repo.
- **ProCC** argues that homography-only field evaluation misses camera geometry, distortion and non-planar structure ([CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Magera_A_Universal_Protocol_to_Benchmark_Camera_Calibration_for_Sports_CVPRW_2024_paper.html)). That directly motivates separating floor M1, net M3, distortion/pose identifiability and world sensitivity.
- **BroadTrack** uses direct field correspondences to reduce drift in moving broadcast-camera tracking ([WACV 2025 paper](https://openaccess.thecvf.com/content/WACV2025/papers/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.pdf)). **BHITK** combines inter-frame motion, keypoint uncertainty and sequential filtering ([paper](https://arxiv.org/abs/2311.10361)). Both support a predict/update/reset design rather than blind homography smoothing.
- A self-supervised sports shape-alignment method still reports its best sports result after roughly 240 labeled fine-tuning images ([WACV 2022 paper](https://openaccess.thecvf.com/content/WACV2022/html/Shi_Self-Supervised_Shape_Alignment_for_Sports_Field_Registration_WACV_2022_paper.html)). “Self-supervised” therefore does not justify claiming a zero-label universal court solver.
- Synthetic sports-camera work exists ([Chen and Little](https://arxiv.org/abs/1810.10658)), but our own two synthetic-only failures are the controlling local evidence. Synthetic data is at most a minority augmentation and metamorphic test source.

What appears to transfer across venues is a factorization: semantic field evidence, known regulation geometry, robust estimation, independent validation, and temporal re-acquisition. The public literature does not provide evidence that a 100-frame pickleball set can replace source diversity with one large end-to-end coordinate regressor.

### Precision machinery implementable without new labels

1. **Paint-center observations.** OpenCV's LSD API provides a classical line-segment baseline ([OpenCV LSD](https://docs.opencv.org/3.4/d1/dbd/classcv_1_1line__descriptor_1_1LSDDetector.html)). DeepLSD combines learned attraction fields with classical detection and vanishing-point refinement ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Pautrat_DeepLSD_Line_Segment_Detection_and_Refinement_With_Deep_Image_Gradients_CVPR_2023_paper.html)); SOLD2 learns occlusion-aware line detection/description without manual line labels ([CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Pautrat_SOLD2_Self-Supervised_Occlusion-Aware_Line_Description_and_Detection_CVPR_2021_paper.pdf)). For courts, the implementable precision step is still geometric: fit both signed-gradient edges of each finite-width paint band and use the center plus covariance. Every extractor must compete at fixed visibility against wrong parallel stripes and shadows.
2. **Robust fitting.** MAGSAC++ marginalizes over noise scale for robust estimation ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html)); GC-RANSAC adds spatial coherence to consensus ([paper](https://arxiv.org/abs/1706.00984)). Neither removes the need for semantic line identity, degeneracy checks and an independent accept metric.
3. **Direct alignment.** OpenCV's ECC family exposes intensity-based image alignment ([OpenCV motion documentation](https://docs.opencv.org/master/dc/d6b/group__video__track.html)). It is appropriate only inside a small attraction basin with masks for players/net/shadows; raw photometric improvement can lock onto non-court texture and must not be the sole accept score.
4. **Temporal lock.** The court state should predict from optical flow/VIO, update from semantic line evidence, track covariance, re-detect on keyframes, keep multiple symmetric hypotheses briefly, and reset when normalized innovation or independent line score fails. Smoothing a wrong homography is lower jitter but not higher accuracy.
5. **Phone priors and rolling shutter.** Apple describes ARKit world tracking as visual-inertial odometry with explicit tracking-quality state ([ARKit](https://developer.apple.com/documentation/arkit/understanding-world-tracking)). Gyroscope-based phone stabilization has demonstrated real-time rolling-shutter-aware warping ([Stanford report](https://aperture.stanford.edu/papers/stabilization/)); modern rolling-shutter correction work confirms that row-dependent warps are a real image-formation problem ([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_Learning_Adaptive_Warping_for_Real-World_Rolling_Shutter_Correction_CVPR_2022_paper.html)). This is only actionable after sensor/video timing, crop, orientation and row exposure are trustworthy.

### Uncertainty and honest in/out

Criminisi et al. derive homography covariance and propagate both correspondence and image-localization uncertainty into planar measurements ([paper record](https://www.sciencedirect.com/science/article/abs/pii/S0262885698001838)); the associated derivation shows first-order propagation into measured distances ([Oxford notes](https://www.robots.ox.ac.uk/~vgg/presentations/bmvc97/criminispaper/node6.html)). Control-point layout also affects calibration stability, so a four-corner ±1 px test is not a complete sensitivity model ([control-point study](https://arxiv.org/abs/1803.03025)).

For this stack, covariance must include line-center uncertainty, semantic correspondence ambiguity, camera/distortion parameters, temporal process/reset noise and ball-localization error. Propagate samples/Jacobians through the exact per-frame authority to the signed distance normal to each court boundary. A call becomes definite only outside a calibrated decision band; evidence loss must widen the band or yield absent/`too_close_to_call`, never increase confidence.

I found no current public Hawk-Eye document that publishes the requested operational “error ellipses.” A secondary human-factors paper describes multi-camera reconstruction and cites a roughly 3 mm system accuracy figure ([PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2602819/)), but that is not a covariance protocol. The stronger public benchmark is the ITF's direct reference discrepancy test, and it should not be misrepresented as vendor-published uncertainty ellipses.

### Few-label and foundation-model routes

- SAM 2 supplies promptable video segmentation with memory ([paper](https://arxiv.org/abs/2408.00714)); it could maintain a coarse court/paint region between keyframes. It does not know regulation-line identity or metric scale.
- DINOv2 learns general visual features from a large curated diverse corpus ([paper](https://arxiv.org/abs/2304.07193)); it is useful for frame diversity, keyframe retrieval or feature evidence, not direct court-coordinate authority.
- AnyCalib predicts model-agnostic single-view intrinsics/rays ([paper](https://arxiv.org/abs/2503.12701)); it can be an imported-video prior, but does not solve court pose and cannot override native/profile intrinsics.
- Test-time entropy minimization adapts normalization/affine parameters on unlabeled test batches in classification/segmentation ([TENT, ICLR 2021](https://iclr.cc/virtual/2021/spotlight/3479)). With one venue and no labels, confident wrong court families can self-reinforce; TTA belongs below fixed geometry and must be source-grouped, reversible and independently scored.

At the repository's present data scale, classical/learned line evidence feeding a constrained geometric solver is more defensible than a learned homography regressor. The owner-100 labels should supervise line family, visibility, paint masks and uncertainty with source-disjoint splits. SAM2/DINO/AnyCalib/TTA survive only if they improve fixed, non-circular evidence metrics on both permitted internal clips; their output is never GT or pseudo-label authority.

## C. Sibling harness critique: make the GT-free metrics hard to game

The sibling lane materialized its implementation and baseline while this consult was running, so the final critique covers the code and artifacts, not only the spec. Its current baseline is appropriately marked diagnostic/non-promotion and reports M2 absent for both clips because no true per-frame calibration exists. Current M1 is 5.33 px median/12.46 px p90 at 77.5% evidence coverage on Wolverine and 5.87 px/11.5 px at 83.3% on Burlington. M3 is 206.6 px median on Wolverine and 267.8 px on Burlington, which is a red flag for evidence semantics/model mismatch rather than a usable net-precision baseline. M5's current worst-direction local 1-px displacement has 12.59 cm maximum on Wolverine and 17.27 cm on Burlington. These are harness observations, not GT accuracy.

### M1 projected line residual

- The implementation searches an existing white mask only within ±14 px normal to each projected template sample. This is circular: a wrong H can select the nearest parallel paint stripe, and candidates outside the window become a coverage loss. Freeze candidate extraction independently, score all competing regulation/template identities, and report the margin to the runner-up.
- Court paint is a band. The current routine finds the closest supported run, then refines the brightest grayscale peak; that is not necessarily the paint center and its residual maximum is censored by the 14 px window. Fit both gradient edges, score signed normal distance to their center with width/covariance, and expose overflow rather than reporting a clipped maximum.
- The implementation does report missing evidence as lower coverage, which is good, but projected in-frame samples define “visible.” Freeze visibility independently of each candidate calibration so a method cannot move hard lines out of frame or into occlusion and improve its score.
- Report per named line, near/far side, edge/center and occlusion bucket, then coverage-weighted median/p90/max. Aggregate-only M1 is gameable by the easy near baseline.

### M2 temporal stability

- The implementation correctly refuses to infer temporal precision from static calibration/camera-motion alone, so both current baselines are honestly absent. When explicit per-frame solves exist, however, keypoint delta and distance from the clipwise median still reward a constant wrong homography and can penalize a correct moving-camera solution. Score each frame against independently detected evidence and track long-horizon template-point drift, innovation, coverage and reset latency.
- Separate static and intentionally moving clips. On static clips, added jitter is a negative. On moving clips, a zero-motion solution is a failure even if smooth.
- Report raw and filtered states. A filter must not win by hiding lag; include re-acquisition after occlusion/shake and covariance growth during missing evidence.

### M3 net consistency

- Net-top points are non-planar. The implementation does project 3D post-height endpoints, which is better than floor flattening, but it selects the first available evidence artifact and reduces its net to one observed segment. The 200–268 px current medians strongly suggest that source semantics, scaling, or segment identity must be audited before M3 is used to accept anything.
- Pickleball net height is not uniform: the center and posts differ. The current expected segment is a straight post-height line. Declare the center-sag curve and distinguish post, tape and mesh evidence. Missing or untrusted net evidence is absent, not a large but apparently comparable scalar.

### M4 pb.vision comparison

- The pb export does not supply semantic court-point identities. The implementation requires a unique nearest-neighbor assignment from the 12 pb points to our reviewed points; uniqueness prevents duplicates but does not establish semantic identity and still minimizes proximity. Freeze the mapping/protocol, source hashes, coordinate transforms, point exclusions and code version before scoring.
- Report agreement only. Do not call pb superiority/consistency “accuracy,” do not use it for training, and never optimize directly against the export.

### M5 ±1 px sensitivity

- The implementation evolved beyond the spec's “four-corner perturbation”: it projects ten canonical world locations, perturbs each resulting image location by axis-aligned ±1 px, and inverse-projects. That is a useful local image-to-world scale/Jacobian diagnostic, but it does not perturb or refit calibration observations and therefore is not calibration uncertainty. Perturb the actual point/line observations along estimated normal covariance, refit the whole calibration, and sample correlated bootstraps/Monte Carlo including family swaps.
- Report boundary-normal displacement in centimetres on a denser canonical grid, including both sides of every kitchen/baseline/sideline, corners, center and extrapolation regions. Include the worst Jacobian singular direction/eigenvector, not only image axes; Euclidean displacement is not the same as signed distance to the relevant call line.
- Propagate into representative player feet, BALL arc/bounce and line-call signed distance. A lower H parameter delta does not prove better placement.

### Global anti-gaming contract

Freeze frames, hashes, evidence extractor, visibility, line/template identities, crop/orientation/distortion space and scorer version before candidates run. Require an independent evidence provider for acceptance, exact missing/abstain counts, per-source results, metamorphic tests (crop/resize/known warp/blur/occlusion/dual-line distractor), and raw unfiltered artifacts. Keep pb agreement, self-residual, temporal smoothness, world sensitivity and eventual owner-reviewed PCK as separate axes. No weighted composite should hide a catastrophic axis, and no GT-free metric can promote CAL by itself.

## Temporal court-lock design for moving cameras

1. **State:** reference court-to-image geometry, identifiable camera/distortion subset, current-to-reference motion, covariance, semantic line identities, motion mode and lock state (`locked`, `coasting`, `reacquiring`, `absent`).
2. **Initialization:** only from reviewed/profile-verified calibration or a proposal that passes fixed independent evidence. Preserve top-K symmetric line-family hypotheses until separated by a margin.
3. **Prediction:** use masked optical flow; on native capture add time-aligned ARKit/gyro rotation and row-dependent rolling-shutter prediction. Grow covariance every frame, especially when features/evidence disappear.
4. **Measurement:** subpixel paint centers, intersections, vanishing directions and physical net observations, each with covariance and explicit floor/non-floor type.
5. **Update:** robust point+line solve in the correct pixel/distortion space. Gate by normalized innovation, condition/identifiability and independent held-out line score.
6. **Keyframes:** re-detect on a fixed cadence and immediately after innovation, occlusion, crop, exposure, tracking-state or motion-mode changes. Keyframes can reset drift; ordinary frames cannot silently redefine semantic identities.
7. **Output:** one immutable per-frame court artifact with H/camera state, coordinate direction, evidence coverage, covariance, keyframe/reset provenance and absent reasons. Placement, BODY, BALL 3D and in/out must eventually consume this same authority through integration-owner hunks.
8. **Fail closed:** no evidence means growing uncertainty and then absence, never a frozen “precise” calibration. Any close line call inside the propagated band abstains.

## L0 on-device implications

L0 should be profile-first and guided, not promise universal autonomous officiating. On-device classical paired-edge lines plus IMU prediction can maintain a cheap lock; a heavier semantic extractor runs only at initialization/keyframes or when confidence drops. Device/lens/crop/orientation/row-time fingerprinting is part of the calibration identity. The UI should show lock state and ask for a small correction when profile verification fails. A 60-fps smooth overlay is not evidence of millimetre accuracy; L0 promotion still needs the roadmap's owner and gold-capture gates.

## Ranked reproduction map and next handoff

[PLAN.md](PLAN.md) has nine ranks. The immediate no-owner sequence is ranks 1–3 as standalone candidate artifacts, with the harness frozen first; rank 4 can use existing profile logic; ranks 5–7 harden camera/uncertainty. Rank 8 waits for the staged 100-frame owner review. Rank 9 is a capped bake-off, not an open research program.

All integration edits are deliberately deferred. This window does not propose direct edits to `process_video.py`, `orchestrator.py`, `coordinates.py`, `timebase.py`, BALL files, or `court_calibration*.py`; PLAN names the future integration seams so an owner can serialize hunks. Best-stack delta: **none**.

## Honest limits

- Vendor accuracy and architecture statements from pb.vision, SwingVision and Zenniz are vendor claims; no independent raw evaluation set or reproducible calibration implementation was available.
- PlaySight, Hawk-Eye, Foxtenn and broadcast-AR public pages do not disclose enough algorithm detail to reproduce their court lock. No defensible Foxtenn camera-count/frame-rate claim is included.
- Patent searches reveal broad mobile video/object-tracking and planar-tracking claims, not a verified description of SwingVision's present production calibration.
- I found no official Hawk-Eye publication of operational error ellipses/covariance; the ITF discrepancy protocol and a secondary paper are reported separately.
- Several sports-registration papers are soccer/broadcast-centric. Their mechanisms are candidate precedents, not proof of pickleball/phone transfer.
- The sibling harness had only a spec when inspected, so no baseline or candidate movement was measured. Every expected magnitude in PLAN is a preregistered hypothesis and kill threshold.
- The 100-frame package is staged but not owner-reviewed. No route depending on those labels can start honestly yet.
- No gold capture proves boundary-normal centimetre accuracy, rolling-shutter correction, uncertainty coverage or in/out performance. `VERIFIED=0` remains binding.

## Lane compliance

- Code/config/docs changed: none.
- Writable scope used: `runs/lanes/court_research_sol_20260712/**` only.
- Test command: `no-code-lane`; artifact/schema/link/policy validation only.
- Artifacts: `REPORT.md`, `PLAN.md`, `census.json`.
- Best-stack delta: none (research).
