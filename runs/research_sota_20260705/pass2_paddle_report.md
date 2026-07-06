# PASS 2 — 6-DOF paddle/racket pose + hand-object from monocular video

2026-07-06. 41 agents.

# Pass-2 Deep Dive: 6-DOF Paddle/Racket Pose & Hand-Object Estimation — What's New

*Research date: 2026-07-06. Everything below is NEW relative to pass-1 (which already covered FoundationPose, MegaPose, SAM-6D, FoundPose, GigaPose, OnePose++, FreeZe, WiLoR, HaMeR, Hamba, HaWoR, HOLD, HOISDF, MOHO, RacketVision, TT4D, Diff-DOPE, IPPE, BlurBall, BOP, HANDAL, HOT3D, Grounding-DINO, SAM2, BundleSDF, LightGlue). Organized by pipeline sub-problem.*

## 1. Ball 3D trajectory (P0-adjacent: ball ensemble + physics arc solver)

**WASB** (arXiv:2311.05237, BMVC 2023) is not new to us — it's already vendored at `third_party/WASB-SBDT` and is one of our current ensemble members. What pass-2 surfaced concretely: it beats TrackNetV2/ResTrackNetV2/MonoTrack by **+7.8 to +16.8pp AP** across 5 sports (Tennis F1 93.6/AP 91.8, Badminton F1 95.1/AP 91.6, per Table 2), via **Hard-to-Localize Sample Mining (HLSM)** — retraining on frames the model misses/blurs with sharper real-valued Gaussian GT — plus simple 3-frame motion-consistency selection instead of Kalman/particle filtering. HLSM is a directly reusable fine-tuning recipe for retraining WASB on owner-captured in-domain data (our documented "next unlock").

**TT3D** (arXiv:2504.10035, CVPR-W 2025, github.com/cogsys-tuebingen/tt3d) reconstructs monocular 3D ball trajectory + spin from broadcast video by searching bounce states that minimize reprojection error of a physics-fit arc — same problem shape as our own arc solver — and it **literally reuses BlurBall**, which we already vendor. Worth reading closely as a cross-check on our own blurball integration, though table-tennis bounce physics need re-derivation for pickleball's net/paddle contacts.

**"Uplifting Table Tennis"** (arXiv:2511.20250, WACV 2026, Kienzle/Ludwig/Lorenz/Satoh/Lienhart) [VERIFIED verbatim abstract via adversarial check] proposes a two-stage pipeline: front-end perception trained on real 2D annotations only, back-end 2D→3D "uplifting" network trained **exclusively on physically-correct synthetic data**, explicitly re-engineered for robustness to missing detections and varying frame rate. This is a direct architectural template for closing our own sim-to-real gap without needing real 3D/spin ground truth — only real 2D labels (collectable) plus synthetic 3D data (generatable). Exact numeric tables were not extractable from the fetched PDF (image-heavy) — flagged as an open question.

**MuJoCo** (Apache-2.0, mujoco.org/github.com/google-deepmind/mujoco) is the physics engine TT4D (arXiv:2605.01234, 2026) uses to generate ~3M synthetic ball-trajectory points via a stitching algorithm sampling physically-plausible flight/bounce/hit segments — directly reproducible for pickleball-specific restitution/net/paddle-contact parameters, and a free, general-purpose (not sport-locked) way to manufacture in-domain-like training data.

**RacketVision** (already known from pass-1 via its TrajPred module) turned out to have two under-surfaced sub-benchmarks: **MS-TrackNetV3** ball tracking (mAP 71.1–83.1 across table tennis/tennis/badminton) and **RTMPose racket-keypoint** detection (handle keypoint PCK 92.6–97.9%, overall PCK@0.2 81.8–89.6%) — both MIT-licensed with public checkpoints (github.com/OrcustD/RacketVision). These are more directly integration-ready than TrajPred for our paddle-keypoint problem. See §2 below.

## 2. Paddle/racket 2D keypoints and 6-DOF pose

**RTMDet/RTMPose (OpenMMLab)**, repurposed by RacketVision as its racket-pose module (RTMDet-M box detection → RTMPose-M regressing 5 keypoints: top/bottom/handle/left/right), gives handle-keypoint PCK **92.6–97.9%** and overall PCK@0.2 of 81.8/89.6/88.5% for table tennis/tennis/badminton respectively. Output is 2D-only, no depth/rotation — complements rather than replaces our 6-DOF fused estimator (which already reports IoU 0.11→0.26/0.03→0.36, jitter 23-53→5 deg/frame), but is a cheap, open (Apache-2.0-family), checkpoint-ready front-end candidate for stabilizing box/keypoint inputs.

**Grounding DINO** (arXiv:2303.05499, github.com/IDEA-Research/GroundingDINO, Apache-2.0) was already known generically, but pass-2 connected a specific application: zero-shot text-prompted ("pickleball paddle") bounding boxes with **zero in-domain labels** (48.4–52.5 AP zero-shot on COCO, 26.1 mean AP on ODinW-35) — a direct, low-cost lever against our scarce-in-domain-data bottleneck for paddle box detection.

**ComPose** ("When to Trust Hands for Object Pose Tracking," arXiv:2605.23523, submitted 2026-05-22) fuses object-geometry cues with hand-motion cues via adaptive per-frame trust weighting, enforcing temporal consistency without an external smoothing pass — conceptually the exact problem we've already identified as our weak point ("boxes dominate, fingers rest-pose-weak"). Ablation: full-method ARE 19.88° vs. object-only 35.02°/hand-only 30.65° on DexYCB. **No code released**, evaluated only on small rigid YCB-style objects (DexYCB/HO-3D v2/OakInk-v1), 6 weeks old with no third-party reproduction — read the method as design input only.

**RGBTrack** (arXiv:2506.17119, IROS 2025, github.com/GreatenAnoymous/RGBTrack) is "FoundationPose without the depth camera": binary-search render-and-compare depth inference + XMem 2D tracking + Kalman filter + state machine + dynamic CAD-scale recovery, giving real-time frame-to-frame tracking instead of per-frame re-detection. Code is released — a strong candidate to stabilize our paddle track across a rally, but needs a simple CAD proxy (flat rectangle) for the paddle and no racket-specific validation exists.

**OnePoseViaGen** ("One View, Many Worlds," arXiv:2509.07978, CoRL 2025 Oral, github.com/GZWSAMA/OnePoseviaGen) builds a coarse 3D model from a **single reference photo** (no CAD), then uses text-guided generative synthesis (Trellis) to render large texture/viewpoint-randomized training sets for a render-and-compare pose estimator. This is close to an exact recipe for bootstrapping a paddle-specific pose head from ~1 real photo instead of hundreds of labeled captures — directly attacks our scarce-data constraint, code released.

**Image as an IMU** (arXiv:2503.17358, Oxford, code released) inverts single-frame motion blur into an instantaneous 6-DOF angular/translational velocity estimate via dense flow+depth regression and a small-motion least-squares fit, beating MASt3R/COLMAP baselines. Directly relevant to our fastest, most-blurred swing/contact frames where classical smoothing likely underestimates rotation — but untested on handheld-object (vs. camera) motion.

**Gaussian-splatting pose-tracking family** (GSGTrack arXiv:2412.02267 — RGB-only, silhouette-loss robust to noisy pose; GHOST arXiv:2603.18912 — category-agnostic hand+tool RGB reconstruction, speed-focused; GTR arXiv:2505.11905 — explicitly targets thin/symmetric/texture-poor objects; 6DOPE-GS arXiv:2412.01543 — 3.5Hz online tracking+splat with dynamic keyframe/confidence filtering, ~5x speedup vs. BundleSDF-class on HO3D/YCBInEOAT) is a cluster of 2024–2026 methods whose stated failure case — thin, symmetric, low-texture rigid objects — is exactly a paddle. None is racket-validated; worth a dedicated bake-off, not a single adoption.

Additional 2025–2026 signal-only items (no code or unproven on rackets): **SingRef6D** (arXiv:2509.21927, single-RGB-reference monocular pose, no code yet), **YOPO** (arXiv:2508.14965, single-stage monocular category-level pose, code released), **OneViewAll** (arXiv:2605.07023, symmetry-aware single-reference pose — directly targets planar/symmetric objects like a paddle face, no code confirmed), **Yolo-Key-6D** (arXiv:2603.03879, YOLO+keypoint+PnP, 63 FPS, no public repo found), **Flose** (arXiv:2602.19719, flow-matching pose denoiser addressing symmetry ambiguity), **PoseStreamer** (arXiv:2512.22979, RGB+event fast-motion tracking, model-free CAD bootstrap via BundleSDF), **GoTrack** (arXiv:2506.07155, lightweight CAD-conditioned refine+track loop), **DynamicPose** (arXiv:2508.11950, VIO+Kalman for fast-camera+fast-object — requires depth+IMU we lack).

## 3. Hand-object interaction / grip modeling

Pass-1 already knew HaMeR/Hamba/HaWoR/HOISDF/MOHO. New this pass: **AlignSDF** (arXiv:2207.12909, ECCV 2022) is a 2022-vintage hand+object SDF-shape method whose main modern relevance is as a **hard preprocessing dependency of HOISDF** ("follow the AlignSDF repo to generate the original SDF files" per HOISDF's own README) — a real integration-cost detail, not itself adoptable (targets fine shape reconstruction, not 6-DOF rigid pose). Verdict: skip as a standalone tool, but note the dependency cost if HOISDF is pursued.

Occlusion/generative hand-object reconstruction cluster (all recent, code status varies, none racket-validated): **ForeHOI** (arXiv:2602.06226, feed-forward occlusion-robust reconstruction, code released — targets exactly our hand-occludes-paddle pattern), **MagicHOI** (arXiv:2508.05506, diffusion 3D priors for limited-viewpoint monocular clips — matches our short single-viewpoint iPhone captures), **Follow My Hold** (arXiv:2508.18213, hand-as-geometric-scaffold contact/non-intersection constraints), **HORT** (arXiv:2503.21313, ICCV 2025, fast transformer point-cloud reconstruction), **UniHOPE** (arXiv:2503.13303, CVPR 2025, code released — hand-only/hand-object switcher, relevant to grip/no-grip transitions), **WHOLE** (arXiv:2602.22209, egocentric-only, lower relevance), **GenHOI** (arXiv:2603.19013, no code yet).

**gSDF** (arXiv:2304.11970, CVPR 2023, code released) improves on AlignSDF via kinematic-chain-guided SDF conditioning (17.6%/7.1% CD improvement on ObMan hand/object; 12.2%/14.4% on DexYCB) — same verdict as AlignSDF: skip, wrong output modality (dense SDF mesh vs. our 6-DOF rigid pose need).

## 4. Player mesh / body pose

**SAM 3D Body / Fast SAM 3D Body** (Meta, arXiv:2602.15989 + arXiv:2603.15603) is already partially vendored at `third_party/Fast-SAM-3D-Body`. It's a promptable full-body mesh model using a new Momentum Human Rig (MHR) that decouples skeleton from shape — a direct replacement/upgrade candidate for our currently-integrated SAT-HMR. Fast variant claims up to 10.9x speedup, ~65ms/frame on RTX 5090, "on-par, surpasses on LSPET" — but **no numeric MPJPE/PVE was retrievable** from the primary sources fetched (PDF exceeded fetch size), so accuracy vs. SAT-HMR is unverified; needs a real head-to-head on our own footage.

**4DHumans** (arXiv:2305.20091, ICCV 2023, MIT, code released) — MPJPE 70.0mm/PA-MPJPE 44.5mm on 3DPW — folds occlusion-robust tracking into the mesh stage itself (3D-space PHALP-based tracker), potentially reducing identity-swap/jitter issues our visual-polish work has been fighting, but is SMPL-only (no hand/finger detail) so doesn't help the paddle stage.

**CalTennis** (arXiv:2606.20542, 2026-06) — 11M+ frames/51hrs of tennis with 2-6 synchronized cameras + MOCAP ground truth, 10x larger than prior in-the-wild motion datasets — not racket-specific but a potential mining/fine-tuning source for validating our SAM-3D-Body player-mesh stage in racket-sport contexts.

**Metric3D v2** (arXiv:2404.15506, BSD-2-Clause, code released) — AbsRel 0.051-0.063, delta1 ~0.975-0.977 zero-shot metric depth — could supply a redundant metric-scale cross-check for court-plane homography, especially in tight paddle close-ups where court visibility is poor; not essential since we already get scale from court geometry.

## 5. Claim verification highlights [VERIFIED / corrected]

- **Uplifting Table Tennis abstract** [VERIFIED verbatim] — see §1.
- **PadelTracker100** (PMC12926558) [VERIFIED with correction]: confirmed ball bbox + 17-keypoint COCO player pose, zero racket annotations, inter-annotator IoU 0.826 (players)/0.677 (ball) — but the broader conclusion "no racket-sport has paddle-pose ground truth" is **wrong**: RacketVision already provides 2D racket keypoints for table tennis/tennis/badminton.
- **Pickleball smart-paddle IMU hardware** [VERIFIED with corrections]: Potenza SMARTx COREx4 is pre-order/sold-out, still "R&D phase," no confirmed ship date (official price $595.95, not the $495.99 circulating in blogs); UVA's PIKL is an unpublished one-off capstone with no accuracy data; **PIQ Robot is defunct** (company shut down March 2021) and never had a pickleball mode — drop as a candidate entirely.
- **"Reviewers note pickleball lacks sensor tech"** [corrected]: attributable to one reviewer (Matt, Matt's Pickleball), not a consensus — substance holds (no shipped IMU paddle exists) but framing was overstated.

## 6. Constraints check

Single-iPhone-RGB-only rules out: DynamicPose (needs depth+IMU), most Gaussian-splatting variants requiring RGB-D, PoseStreamer's event-camera mode. Scarce-in-domain-data is directly addressed by MuJoCo synthetic augmentation, OnePoseViaGen, Grounding DINO zero-shot bootstrapping, and the Uplifting-Table-Tennis two-stage sim2real template — these four form the strongest "now/soon" cluster. Licenses remain irrelevant per project brief (CC-BY-NC-ND, GPL-3.0 items like DynamicPose are usable but were excluded here on capability grounds, not licensing).


## Missed in pass 1

- **RacketVision's racket-pose and ball-tracking sub-benchmarks (RTMPose keypoints, MS-TrackNetV3)** — Pass 1 apparently surfaced RacketVision only via its TrajPred (ball+racket cross-attention trajectory forecasting) module. This pass found RacketVision also ships a fully separate, more directly relevant sub-benchmark: 2D racket pose (5 keypoints via RTMDet+RTMPose, PCK@0.2 81.8-89.6% overall, handle keypoint 92.6-97.9%) and a strong multi-sport ball tracker (MS-TrackNetV3, mAP 71.1-83.1). These are open MIT-licensed, checkpoint-downloadable, and structurally the closest published analog to our own paddle-keypoint and ball-detection problems -- more relevant to near-term integration than TrajPred itself. _Why:_ The racket-pose keypoint detector (>92% PCK on handle) is a ready-to-eval 2D front-end for stabilizing box/keypoint inputs into our fused wrist+box paddle estimator, at essentially zero integration cost since code+weights are already public -- this should have been flagged alongside TrajPred in pass 1 since it's part of the same paper/repo. https://arxiv.org/abs/2511.17045
- **WASB's HLSM (Hard-to-Localize Sample Mining) retraining recipe** — WASB (already vendored at third_party/WASB-SBDT) is one of our current ball-ensemble members, but the specific mechanism that gives it its accuracy edge -- Hard-to-Localize Sample Mining (mining frames the model misses/blurs, retraining with sharper real-valued Gaussian GT) plus simple 3-frame motion-consistency tracking instead of Kalman/particle filtering -- is a directly reusable recipe for retraining WASB on owner-captured in-domain pickleball footage, and should have been called out in pass 1 as the concrete 'next unlock' mechanism rather than just cataloging WASB as an existing dependency. _Why:_ Our own BALL chain state doc names 'owner in-domain data' as the next unlock; HLSM is the exact fine-tuning technique that would exploit that data once available, directly connecting a known dependency to our known blocker. https://arxiv.org/abs/2311.05237
- **HOISDF's hard dependency on AlignSDF's SDF-preprocessing tooling** — HOISDF (already known, github.com/amathislab/HOISDF) requires literally running AlignSDF's (arXiv:2207.12909, ECCV 2022) SDF-generation code as a preprocessing step ('follow the AlignSDF repo to generate the original SDF files, then use tools/pre_process_sdf.py' per HOISDF's own README) -- a concrete integration-cost detail not surfaced when HOISDF was first evaluated. _Why:_ If HOISDF is ever pursued for phase-2 hand-object fusion, this reveals an un-costed extra dependency (a 2022-vintage, unmaintained-looking preprocessing repo) that adds real integration friction beyond just running HOISDF itself. https://github.com/zerchen/AlignSDF
- **Grounding DINO's zero-shot bootstrap potential specifically for scarce-in-domain-data paddle box labeling** — Grounding DINO was already known generically, but its specific applicability -- generating zero-shot 'pickleball paddle'/'racket' bounding boxes with ZERO in-domain labeled training data (48.4-52.5 AP zero-shot on COCO, 26.1 mean AP on ODinW) -- as a direct bootstrap for our scarce-in-domain-data bottleneck on the paddle box detector was not connected to that specific project constraint in pass 1. _Why:_ This is a zero-cost (no labels needed), already-open (Apache-2.0), already-integrated-elsewhere (Grounded-SAM) tool that could immediately widen paddle box training/eval data without waiting on owner captures -- a direct lever on our stated scarce-data blocker that deserved explicit connection. https://arxiv.org/abs/2303.05499

## New adoptions

### [NOW] MuJoCo-based synthetic ball-trajectory augmentation (TT4D stitching-algorithm pattern) → Ball 3D trajectory stage / physics arc solver training data (ball chain 'next unlock = owner in-domain data' blocker)
- **what:** Reproduce TT4D's (arXiv:2605.01234) trick of simulating serve/rally/bounce/paddle-contact physics in MuJoCo (free, Apache-2.0, mujoco.org) to generate large synthetic 2D+3D ball-trajectory corpora (TT4D generated ~3M points this way) for pickleball-specific restitution/net/paddle-contact parameters.
- **evidence:** TT4D (arXiv:2605.01234) reports ~3M synthetic trajectory points and 2.35±1.03cm 3D position error on its synthetic benchmark using this augmentation; MuJoCo is open source (github.com/google-deepmind/mujoco)
- **expected_gain:** Directly unblocks the documented ball-chain wall by manufacturing in-domain-like training data without waiting on owner captures
- **confidence:** high — MuJoCo itself is mature, free, and the stitching idea is straightforward to re-derive for pickleball
- **url:** https://arxiv.org/abs/2605.01234

### [NOW] RacketVision RTMPose racket-keypoint front-end (evaluate as paddle keypoint stabilizer) → Paddle 6-DOF phase-1/phase-2 fused estimator — 2D keypoint input stabilization before box+wrist fusion
- **what:** Trial RacketVision's open MIT-licensed RTMDet+RTMPose racket-keypoint detector (5 keypoints: top/bottom/handle/left/right, handle PCK 92.6-97.9%) as a 2D front-end feeding our fused wrist+box paddle 6-DOF estimator
- **evidence:** RacketVision paper Table 4/5 (arXiv:2511.17045): handle keypoint PCK 92.6-97.9%, overall PCK@0.2 81.8-89.6% across table tennis/tennis/badminton
- **expected_gain:** Could reduce box/keypoint noise feeding the fusion stage; does not itself add 3D/6-DOF, so must remain a component swap not a stack replacement
- **confidence:** high — code/checkpoints already public, near-zero integration cost for a quick eval
- **url:** https://github.com/OrcustD/RacketVision

### [NOW] Grounding DINO zero-shot paddle/hand box bootstrap → Paddle 6-DOF box-proposal stage + SAM-3D-Body player-box input
- **what:** Use Grounding DINO (Apache-2.0, github.com/IDEA-Research/GroundingDINO) with text prompts ('pickleball paddle', 'hand') to generate zero-shot bounding boxes with no in-domain labels, as an initial or supplementary box source for the paddle/player stages
- **evidence:** 48.4-52.5 AP zero-shot on COCO, 26.1 mean AP on ODinW-35 (arXiv:2303.05499)
- **expected_gain:** Attacks scarce-in-domain-data bottleneck directly by removing the need for labeled paddle boxes to bootstrap detection
- **confidence:** medium-high — mature, well-documented, but no racket-specific accuracy numbers exist; needs empirical validation on our own clips against current 0.26 IoU / 5 deg/frame bars
- **url:** https://arxiv.org/abs/2303.05499

### [SOON] RGBTrack (depth-free FoundationPose + tracking) → Paddle 6-DOF phase-2 — frame-to-frame tracking stability instead of per-frame re-detection
- **what:** Depth-free, real-time 6D pose tracking built on FoundationPose: binary-search render-and-compare depth inference replacing real depth input, XMem 2D tracking + Kalman filter + state machine, dynamic CAD scale recovery for unknown-scale objects. Code released.
- **evidence:** arXiv:2506.17119, code at github.com/GreatenAnoymous/RGBTrack
- **expected_gain:** Could stabilize jitter across a rally by adding proper tracking (vs. per-frame fusion) without requiring depth hardware we don't have
- **confidence:** medium — real code, IROS 2025 acceptance, but no paddle/racket-specific validation, and would need a simple CAD proxy (flat rectangle) for our paddle
- **url:** https://arxiv.org/abs/2506.17119

### [SOON] OnePoseViaGen (single-reference-image synthetic pose training) → Paddle 6-DOF phase-2 — bootstrapping a dedicated pose/keypoint head from minimal real data
- **what:** Build a coarse 3D model from ONE reference photo of the actual paddle (no CAD needed), then use text-guided generative synthesis (Trellis) to render large texture/viewpoint-randomized training sets for a render-and-compare pose estimator. Code at github.com/GZWSAMA/OnePoseviaGen.
- **evidence:** arXiv:2509.07978, CoRL 2025 Oral, code released
- **expected_gain:** Could close the domain gap our racket-6DOF lane is fighting using just 1 real paddle photo instead of hundreds of labeled captures
- **confidence:** medium — CoRL 2025 Oral with code, but paddle is thin/planar (different shape profile than objects demoed) so needs a prototype run
- **url:** https://arxiv.org/abs/2509.07978

### [SOON] Image as an IMU (motion-blur-to-angular-velocity) → Paddle 6-DOF fast-swing/contact-frame refinement — fusing with existing wrist+box estimate during blur-heavy frames
- **what:** Invert paddle motion blur in a single frame into an instantaneous 6-DOF angular/translational velocity estimate via dense flow+depth regression and small-motion least-squares fit; code released (jerredchen.github.io/image-as-imu)
- **evidence:** arXiv:2503.17358, code released
- **expected_gain:** Could supply a cheap high-speed angular-velocity signal exactly where our current classical smoothing likely underestimates rotation (fast swing/contact)
- **confidence:** medium — beats MASt3R/COLMAP baselines on its own benchmarks, but untested on rigid handheld objects (vs. camera motion) or racket swings specifically
- **url:** https://arxiv.org/abs/2503.17358

### [SOON] SAM 3D Body / Fast SAM 3D Body head-to-head eval vs. SAT-HMR → Player mesh estimation stage (P2), currently held by SAT-HMR per visual-polish/racket-6dof memory
- **what:** Benchmark Meta's promptable full-body mesh model (already partially vendored at third_party/Fast-SAM-3D-Body) against our currently-integrated SAT-HMR on real captured footage for speed and accuracy
- **evidence:** Fast SAM 3D Body: up to 10.9x end-to-end speedup, ~65ms/frame on RTX 5090 (arXiv:2603.15603); base 3DB arXiv:2602.15989
- **expected_gain:** Possible major E2E speed win (current target 6-8 min/clip) plus better wrist/hand precision via MHR rig's skeleton/shape decoupling, IF accuracy holds on our footage — but re-tuning cost against existing smoothing/mesh-index work is nontrivial
- **confidence:** medium — Fast variant claims ~65ms/frame on RTX 5090 (real-time) and 'on-par, surpasses on LSPET' vs base 3DB, but no numeric MPJPE/PVE was retrievable from primary sources, so no verified accuracy comparison to SAT-HMR exists yet
- **url:** https://github.com/facebookresearch/sam-3d-body

### [SPIKE] ComPose (adaptive hand-vs-object-geometry trust gating) → Paddle 6-DOF phase-2 fusion module — directly targets our own documented weak point ('boxes dominate, fingers rest-pose-weak')
- **what:** Reads its full method (adaptive per-frame weighting between object-geometry cues and hand-motion cues, enforced temporal consistency with no external smoothing pass) as design input for our own paddle fusion logic, even without usable code
- **evidence:** ARE 19.5 deg (DexYCB), full-method ablation ARE 19.88 vs object-only 35.02/hand-only 30.65 (arXiv:2605.23523)
- **expected_gain:** Design-pattern only (read-and-borrow), not a drop-in tool; could inform a rewrite of our fusion gating logic
- **confidence:** low-medium — no code released, evaluated only on small rigid YCB-style objects (DexYCB/HO-3D v2/OakInk-v1), paper only 6 weeks old with no third-party reproduction
- **url:** https://arxiv.org/abs/2605.23523

### [SPIKE] Gaussian-splatting pose-tracking family (GSGTrack, GHOST, GTR, 6DOPE-GS) for paddle keyframe refinement → Paddle 6-DOF phase-2 — incremental live-splat keyframe refinement on top of the fused wrist+box estimate, using dynamic keyframe/confidence-filtering to reject motion-blur/occlusion frames
- **what:** Cluster of 2024-2026 RGB(-D) online Gaussian-splat pose tracking methods targeting thin/symmetric/texture-poor rigid objects (GSGTrack is RGB-only with silhouette-loss robustness to noisy pose; GHOST is category-agnostic hand+tool RGB reconstruction built for speed) — none CAD-dependent, several explicitly call out our exact failure mode (thin, symmetric, low-texture objects)
- **evidence:** GSGTrack arXiv:2412.02267, GHOST arXiv:2603.18912, GTR arXiv:2505.11905, 6DOPE-GS arXiv:2412.01543 (3.5Hz, ~5x speedup vs BundleSDF-class on HO3D/YCBInEOAT)
- **expected_gain:** Potential jitter/robustness win specifically on thin symmetric paddle geometry where box/keypoint approaches struggle, but exploratory-stage only
- **confidence:** low — promising conceptual fit but four+ competing variants, none racket/paddle-validated, needs a dedicated bake-off before committing engineering time
- **url:** https://arxiv.org/abs/2412.02267

### [SOON] TT3D / Uplifting Table Tennis two-stage sim2real blueprint for the ball-arc solver → Ball 3D trajectory / physics arc solver architecture (Phase A BVP arc safeguards lane)
- **what:** Adopt the architectural pattern (real-2D-supervised front-end perception + synthetic-only-3D back-end uplifting network, re-engineered for missing-detection/frame-rate robustness) as a template redesign for our physics arc solver, independent of any table-tennis-specific code reuse
- **evidence:** Uplifting Table Tennis abstract [VERIFIED verbatim] (arXiv:2511.20250, WACV 2026); TT3D arXiv:2504.10035 shares BlurBall dependency with our stack
- **expected_gain:** A validated design pattern for closing the sim-to-real gap without real 3D/spin ground truth — directly matches our own scarce-in-domain-data constraint
- **confidence:** medium-high on the architectural idea, low on direct code reuse (table-tennis-specific, no pickleball validation)
- **url:** https://arxiv.org/abs/2511.20250
