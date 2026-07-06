# PASS 2 — Automated coaching from 3D + sports video understanding + court calibration + events

2026-07-06. 39 agents.

## Pass-2 synthesis: pickleball coaching pipeline, new findings only

Scope note: pass-1 already had pb.vision, SwingVision, PlaySight, SportAI, Zenniz, PlayReplay, Talking Tennis, SportsGPT, MultiPly, PnLCalib, No-Bells-Just-Whistles, T3Set, TTNet, THETIS, MMPose, Sportsbox. Everything below is new material, organized by pipeline sub-problem (P0 capture/pre-processing, P1 2D ball ensemble, P2 3D ball trajectory/physics arc, P3 court calibration, P4 player mesh, P5 paddle 6-DOF, P6 event/shot detection, P7 grounded-LLM coaching).

### P1/P2 — 2D ball ensemble and physics-arc lifting

The closest published analog to our whole ball chain is **TT4D** (arXiv 2605.01234, submitted 2 May 2026), a monocular table-tennis 4D reconstruction pipeline built from 45,946 broadcast games (140+ hours, 211,534 points). Its "lift-first" pipeline inverts the usual order: a transformer lifts the *entire unsegmented* 2D ball track to 3D before any point/rally segmentation, specifically to survive occlusion and detection gaps — the same failure mode our BALL chain state doc flags as the current zero-shot wall. Camera calibration is PnP off table corners (no manual calibration target). As a downstream validation task, TT4D recovers racket 6-DOF pose/velocity at ball impact via **inverse optimal control from the reconstructed ball trajectory**, rather than direct racket tracking — reporting orientation error 26.4±4.4° and velocity error 0.58±0.40 m/s against 92 mocap-verified strokes (avg impact speed 3.72 m/s) [VERIFIED against the paper's own text]. No code/dataset link is confirmed yet (authors state "will release").

**Uplifting Table Tennis** (arXiv 2511.20250, WACV 2026, code at github.com/KieDani/UpliftingTableTennis) is TT4D's architectural ancestor and solves our exact data-scarcity problem for a sibling sport: a timestamp-keyed RoPE transformer lifts noisy 2D detections to 3D position+spin, trained *entirely on synthetic* physically-correct trajectories, validated against real video purely via 2D reprojection error (no 3D ground truth needed). Under stress ablations it holds up far better than a non-RoPE baseline: 5.56px reprojection error vs 24.15px under 10% missing detections, and spin accuracy 97.1% vs 67.7% under combined frame-drop + missing-detection stress. This is a concrete template — train the lift stage on a synthetic pickleball-physics generator, validate on real 2D tracks — for unlocking the BALL chain without needing in-domain 3D ground truth.

**RacketVision** (AAAI 2026 oral, arXiv 2511.17045, code public) benchmarks 1,672 clips/435k frames across tennis/badminton/table-tennis and shows cross-attention fusion of ball+racket features beats naive concatenation for trajectory prediction (best config: tennis mAP 81.9/MDE 1.96px, badminton 83.1/1.54px, table tennis 71.1/3.41px) — directly transferable to fusing our ball ensemble with the fused wrist+box paddle estimator.

Two cheap, low-risk infra items surfaced that we should audit rather than "discover": **RANSAC** outlier rejection ahead of arc curve-fitting, and confirming **Levenberg-Marquardt with a robust loss** (Ceres Solver, github.com/ceres-solver/ceres-solver, BSD) underlies both the arc solver and any camera-pose bundle adjustment — both are almost certainly already implicit via OpenCV/scipy but worth an explicit checklist pass. Also new: **SoccerNet-v3D** (arXiv 2504.10106, github.com/mguti97/SoccerNet-v3D) demonstrates a monocular scale-anchoring trick — back-solving 3D ball position from a single calibrated camera plus the ball's known real diameter, improving detector box IoU from 0.57 to 0.66 (size error 19.01%→7.27%) on their own re-annotated subset. Pickleball's fixed ball diameter (2.87–2.97in) makes this formula directly reusable as an independent depth cross-check, even though the soccer-trained weights don't transfer.

**Verified pickleball-specific physics**: TWU's aerodynamics study (Lindsey, May 2025) gives outdoor 40-hole-ball drag coefficient CD≈0.33 (SD 0.083) and indoor 26-hole CD≈0.45 (SD 0.092), with topspin producing asymmetrically larger lift than backspin (violates classical Magnus symmetry) [VERIFIED]. An independent UBC study (Steyn et al., arXiv 2501.00163, "Executing a Successful Third Shot Drop in Pickleball," submitted 30 Dec 2024 — correcting an earlier "Feb 2025" date) fits outdoor Cd=0.30±0.02 and Cl=0.195·S from 6 filmed trajectories [VERIFIED]. Adversarial check confirms these two outdoor-ball estimates **corroborate**, not contradict, each other (0.30 vs 0.33) — an earlier read flagging "disagreement" was wrong; the only real discrepancy is comparing an outdoor number to TWU's separate indoor-ball figure, which isn't a valid comparison. These are the first real physics-model coefficients specific to our exact ball, usable directly in the arc solver.

### P3 — Court calibration

New calibration candidates beyond the already-known PnLCalib/No-Bells-Just-Whistles lineage: **AnyCalib** (arXiv 2503.12701, ICCV 2025, code public) recovers intrinsics/distortion for an unknown lens model from a single image, useful for handling varying owner-iPhone lens/crop without a fixed camera-model assumption. A lightweight but valuable pattern from the **SoccerNet 2025 Game State Reconstruction challenge** (arXiv 2508.19182v1) and **BroadTrack** (arXiv 2412.01721, WACV 2025, code public): turn any single-frame court calibrator into a temporally-stable one by tracking keypoints via Lucas-Kanade optical flow and re-solving per-frame homography with RANSAC (GSR-13: up to 4% improvement) or propagating full calibration params via an explicit camera-motion model (BroadTrack: halves mean reprojection error, >15% Jaccard-index gain vs single-frame baselines) — directly applicable to our handheld (not tripod) capture, though validated on slow-panning broadcast cameras, not fast handheld motion.

Older/superseded items correctly triaged as skip: Homayounfar et al. 2017's MRF+branch-and-bound field localization (83% IOU on WorldCup) is beaten by every later method in the same lineage (Jiang 2020 89.8%, Citraro 2020 90.5%, KpSFR 2022 91.2%); the learned-error-optimization approach of Jiang et al. 2020 (sportsfield_release, non-commercial + patent-encumbered) is a design reference only, since it's soccer-template-specific.

### P4 — Player mesh / pose

The biggest new finding here is **JOSH** (ICLR 2026, correcting a prior placeholder arXiv ID to the real **arXiv 2501.02158**, code at github.com/genforce/JOSH), which jointly optimizes camera pose, scene geometry (MASt3R-style pointmaps), and human motion via foot-contact constraints, built directly on top of **TRAM/VIMO** (github.com/yufu-wang/tram, ECCV 2024) as its HMR backbone. [VERIFIED] On EMDB: WA-MPJPE100 68.9mm and W-MPJPE100 174.7mm, beating TRAM (76.4/222.4mm), WHAM (131.1mm), and SLAHMR (326.9mm). Full JOSH runs at only 0.8 FPS (optimization-heavy), but its distilled feed-forward variant **JOSH3R** hits 15.4 FPS — faster than WHAM (4.4 FPS) — while trading off scene-reconstruction quality. Tested on CUDA 12.8 + 24GB VRAM, i.e. A100-compatible. This directly targets the gap our "SAM-3D-Body per-frame + classical smoothing" approach patches ad hoc, but all reported numbers are on general in-the-wild single-human motion (EMDB/SLOPER4D/RICH) — no validation exists for 4 simultaneous players or fast athletic motion.

Also newly surfaced: **Vid2Avatar** (CVPR 2023, arXiv 2302.11566, code public) — single-person, per-video test-time neural-SDF optimization for clothed-avatar detail without external segmentation, and **SMPL-X** (arXiv 1904.05866, github.com/vchoutas/smplx) itself as a candidate upgrade path if our current body model lacks native hand articulation, which would matter for paddle-grip biomechanics. Both are complements/design references, not drop-in replacements, given per-video optimization costs.

The first published **pickleball-specific pose/kinematics study** was found: Edriss et al., J. Sports Sciences 2025 (tandfonline.com/doi/full/10.1080/02640414.2025.2524283) — MediaPipe on 14 players' dink shots, showing advanced players have greater femur flexion and continued post-contact wrist motion (both p<0.001). This validates that MediaPipe-class 2D pose already supports real pickleball coaching claims in the literature, and gives candidate coaching features.

### P6 — Event/shot detection

**Multi-Focus Temporal Shifting (MFS)** (arXiv 2507.07381 v3, code/data public) is a near-free upgrade (+1.3% params/+2.7 GFLOPs) for frame-exact hit/bounce/serve event spotting, beating prior SOTA on tennis (55.51% vs 48.82% mAP at δ=0) with 33% fewer FLOPs than a heavier baseline — a strong candidate to replace bespoke bounce-anchor logic. Shot-type taxonomies are also emerging for adjacent racket sports (BST transformer on skeleton sequences, arXiv 2502.21085; Extended OpenTTGames stroke/outcome taxonomy, arXiv 2512.19327) — directly portable label schemes for our own owner-captured in-domain labeling.

### P7 — Grounded-LLM coaching

Three new items map cleanly onto our "grounded-LLM coaching planned" stage and directly address the **scarce in-domain data** constraint. **CoachMe** (arXiv 2509.11698, ACL 2025, code released as MotionXperts/MotionExpert) aligns a learner's motion to a reference/expert motion via skeletal-graph attention, computes per-joint deviations, and domain-adapts from general pretraining to a new sport with limited data — beating GPT-4o by +31.6% (figure skating) / +58.3% (boxing) G-Eval on instruction quality. **ExpertEdit** (arXiv 2604.10466, Apr 2026, UT Austin) trains an expert-motion prior on **unpaired** expert clips (no matched novice/pro pairs needed) and edits a captured motion toward "what it should look like if performed expertly" — this is the single best match for our exact data situation (no paired pickleball footage, only a growing bank of owner captures). **BioCoach** (arXiv 2603.26938, Mar 2026) adds a degree-of-freedom selector that picks which joints/angles matter per skill before generating LLM feedback, reducing hallucination by grounding text in explicit numeric descriptors — directly reusable for choosing paddle/wrist/hip-rotation features per shot type. All three lack pickleball-specific validation and (for ExpertEdit/BioCoach) confirmed code release, so they're design patterns to prototype against, not drop-in modules.

### Market/competitive landscape (new)

**Owl AI** went live May 22 2026 as MLP's official AI officiating system, running entirely on existing broadcast cameras with no bespoke rig — a same-sport proof point that camera-only CV supports pro-level line/fault calls, though on multiple broadcast angles, not one phone. **PlayerU AI Coach** (launched June 2026) is a new consumer coaching app partnered with Kyle Yates; its underlying CV pipeline is undisclosed, so it's market-context only. **PadelTracker100** (Data in Brief, Feb 2026, DOI 10.1016/j.dib.2026.112546) [VERIFIED: IoU players 0.826/ball 0.677] is the closest annotated real-court analog to pickleball (small doubles court + net) but no pipeline/model paper exists yet for padel — a gap worth re-searching.

### Where our constraints bite

Single handheld iPhone: TRAM/JOSH/MultiPly's camera-SLAM machinery targets freely-moving cameras but is validated on general single-human motion, not 4-player fast athletic scenes or court-side semi-static shots — needs its own validation pass. Cloud A100: JOSH/JOSH3R's 24GB-VRAM footprint fits; MultiPly's ~24 GPU-hr/person test-time optimization does not. Scarce in-domain data: ExpertEdit/CoachMe's unpaired/low-data adaptation designs are the most directly relevant answer found this pass. Licenses irrelevant: PnLCalib (GPL-2.0), SMPL-X and sportsfield_release (non-commercial research license, patent-encumbered) are all moot for our private use but worth noting if code is ever redistributed.


## Missed in pass 1

- **TT4D: Pipeline and Dataset for Table Tennis 4D Reconstruction From Monocular Videos** — Single-camera racket-sport reconstruction system that lifts an entire unsegmented 2D ball track to 3D BEFORE segmenting into points/rallies, calibrates via PnP off table corners, and outputs joint ball position+spin, 3D player mesh, and racket 6-DOF/velocity at impact via inverse optimal control (not direct tracking) — built from 140+ hours / 211,534 points of broadcast footage. _Why:_ This is architecturally the single closest published system to our exact target stack (iPhone monocular -> 3D ball + player mesh + paddle 6-DOF for a fast racket sport). It should have surfaced in pass 1 given it directly names ball+mesh+racket reconstruction from monocular racket-sport video, which is precisely our project description. https://arxiv.org/abs/2605.01234
- **JOSH / JOSH3R (Joint Optimization for 4D Human-Scene Reconstruction in the Wild)** — ICLR 2026 poster (UCLA/genforce, arXiv 2501.02158) that jointly optimizes camera pose, dense scene geometry, and human motion via foot-contact constraints, beating TRAM/WHAM/SLAHMR on every EMDB metric, with a feed-forward distilled variant (JOSH3R) running at 15.4 FPS on A100-class hardware. _Why:_ Directly targets the exact gap our 'SAM-3D-Body per-frame + classical smoothing' approach patches ad hoc — temporally-consistent, scene/court-grounded player trajectories. Given it's public code, A100-compatible, and literally embeds a court/scene reconstruction module that could double as a court-calibration cross-check, this should have been caught as a player-mesh-stage candidate in pass 1. https://arxiv.org/abs/2501.02158
- **Owl AI x Major League Pickleball automated officiating** — A pickleball-specific, cloud-based CV officiating system (line calls + challenge review) that went live May 22 2026 running entirely on existing broadcast cameras, no bespoke multi-camera rig. _Why:_ This is a same-sport (pickleball) competitor/proof-point that camera-only CV can support pro-level calls, directly bearing on our camera-setup and court-calibration confidence — a pickleball-domain search should have surfaced this in the first pass given how narrow the domain is. https://www.sportsvideo.org/2026/05/22/owl-ai-and-major-league-pickleball-go-live-with-first-ever-ai-officiating-system-powered-by-broadcast-cameras-and-the-cloud/
- **ExpertEdit: Learning Skill-Aware Motion Editing from Expert Videos** — Trains a motion-editing model on UNPAIRED expert demonstration clips (no matched novice/pro pairs) via masked-motion-modeling, then edits a captured 3D motion toward higher skill to synthesize what a person's technique would look like if performed expertly (UT Austin, arXiv 2604.10466, Apr 2026). _Why:_ Directly answers our own documented constraint ('scarce in-domain data') for the coaching stage — it needs only a bank of expert clips, not paired novice/expert footage, which is exactly our data situation. This maps so precisely onto our stated bottleneck that it should have been found and flagged in pass 1. https://arxiv.org/abs/2604.10466

## New adoptions

### [NOW] RANSAC pre-filter for 2D ball detections before arc/curve fitting → P1/P2 — ball ensemble output, immediately upstream of the physics arc solver's Phase A/B bounce-arc safeguards
- **what:** Standard outlier-rejection pass (sample-subset trajectory fit, count inliers) applied to per-frame ball detections before physics-arc curve fitting, to strip false-positive detections (e.g. ensemble locking onto a hand or court line) before they contaminate the fit.
- **evidence:** established use in patented sports-tracking pipelines and open ball/curve-fitting repos; no new dependency needed (OpenCV/scikit-learn already provide it)
- **expected_gain:** reduces contamination of arc fits by single wildly-wrong detections; low-cost guardrail given our documented zero-shot-wall / honest-miss history
- **confidence:** high — decades-old, unambiguous technique, trivial to implement/verify

### [NOW] Levenberg-Marquardt with robust loss (Ceres Solver or scipy soft_l1/huber) → P2 (physics arc solver LM fit) and P3 (court auto-find camera pose/homography solve)
- **what:** Confirm/upgrade the nonlinear least-squares solver used for court-calibration bundle adjustment and physics arc-parameter fitting to LM with a robust (Huber/soft-L1) loss rather than plain least-squares.
- **evidence:** Ceres Solver github.com/ceres-solver/ceres-solver; LM is the de facto standard for PnP/reprojection-error minimization
- **expected_gain:** less sensitivity to occasional bad ball-detection outliers feeding the arc fit; likely already present under the hood via OpenCV/scipy, so the actionable step is a solver-choice audit
- **confidence:** high — near-zero engineering risk, standard infra (BSD-licensed Ceres, ~4.5k stars, used at Google for bundle adjustment/SfM)
- **url:** https://github.com/ceres-solver/ceres-solver

### [NOW] Monocular 3D ball localization via known ball diameter (SoccerNet-v3D method) → P1/P2 — ball ensemble box-tightening and an independent depth/scale cross-check for the arc solver
- **what:** Geometric scale-anchoring trick: back-solve 3D ball position/depth from a single calibrated camera using the ball's known real-world diameter (Eq. 5-7 in the paper) plus a bounding-box size optimization step that tightens detector boxes to match the known-diameter 3D ray.
- **evidence:** arXiv 2504.10106 / github.com/mguti97/SoccerNet-v3D; bbox IoU improved 0.57->0.66 (size error 19.01%->7.27%) after the optimization step on SoccerNet-v3
- **expected_gain:** an independent, per-frame scale/depth signal to sanity-check or supplement the physics-arc-derived depth, cheap to prototype
- **confidence:** medium-high — cheap, well-specified formula; pickleball ball diameter (2.87-2.97in) is fixed just like a soccer ball, but soccer-trained weights/annotations don't transfer, only the geometric formula does
- **url:** https://github.com/mguti97/SoccerNet-v3D

### [SOON] Temporal calibration propagation recipe (SoccerNet 2025 GSR challenge / BroadTrack) → P3 — court auto-find lane, as a lightweight wrapper around whatever single-frame court-keypoint model is used
- **what:** Turn any single-frame court calibrator into a temporally-smooth video calibrator cheaply: track detected keypoints via Lucas-Kanade optical flow across frames, then re-solve homography per-frame with RANSAC (GSR-13 reports up to 4% improvement over single-frame baseline); BroadTrack instead propagates full calibration params via an explicit camera-motion model, halving mean reprojection error vs single-frame baselines.
- **evidence:** SoccerNet 2025 GSR results arXiv 2508.19182v1; BroadTrack arXiv 2412.01721 (WACV 2025), halves reprojection error / >15% Jaccard-index gain vs single-frame
- **expected_gain:** stabilizes court calibration across a moving/handheld clip instead of independent per-frame re-registration, directly relevant since our capture is handheld not tripod-fixed
- **confidence:** medium — recipe is simple and code-available (BroadTrack: github.com/evs-broadcast/BroadTrack) but validated on soccer broadcast cameras (slow pan), not handheld iPhone motion
- **url:** https://github.com/evs-broadcast/BroadTrack

### [SOON] AnyCalib (model-agnostic single-view intrinsic calibration) → P0/P3 — pre-step before court homography fitting, to auto-recover intrinsics across different owner iPhone models/lenses instead of assuming a fixed lens model
- **what:** Regresses per-pixel incident rays and recovers camera intrinsics/distortion in closed form for multiple lens models (pinhole, Brown-Conrady, Kannala-Brandt) from a single image, without knowing the camera model in advance, and handles cropped/stretched frames.
- **evidence:** arXiv 2503.12701, reported to outperform alternatives including 3D foundation models despite less training data
- **expected_gain:** removes a manual/fixed-camera-model assumption that would otherwise need per-device recalibration as owner captures more clips on different phones
- **confidence:** medium — ICCV 2025 accepted, code public, but not validated on sports/court footage specifically
- **url:** https://arxiv.org/abs/2503.12701

### [SOON] JOSH3R (feed-forward distilled variant of JOSH) → P4 — candidate replacement/augmentation for SAM-3D-Body-per-frame + classical smoothing, specifically for temporally-consistent world-grounded player trajectories
- **what:** Distilled, optimization-free network approximating JOSH's joint human+scene reconstruction at 15.4 FPS (vs JOSH's 0.8 FPS full optimization), built on the TRAM/VIMO HMR backbone with human-scene contact constraints.
- **evidence:** arXiv 2501.02158 (verified, corrected from earlier placeholder ID); github.com/genforce/JOSH
- **expected_gain:** [VERIFIED numbers] beats TRAM on every EMDB metric (WA-MPJPE100 68.9mm vs 76.4mm, W-MPJPE100 174.7mm vs 222.4mm, RTE% 1.3); worth a fast JOSH3R prototype pass before committing to slower full optimization
- **confidence:** medium — public code (github.com/genforce/JOSH), A100-compatible (tested CUDA 12.8, 24GB VRAM), but validated on general in-the-wild human motion (EMDB/SLOPER4D/RICH), not 4-simultaneous-player sports scenes, and JOSH3R trades scene-reconstruction quality for speed per its own README
- **url:** https://arxiv.org/abs/2501.02158

### [SOON] RacketVision cross-attention ball+racket fusion architecture → P1/P5 — fusing the ball ensemble output with the fused wrist+box paddle estimator to improve arc/trajectory solving and hit-moment localization
- **what:** AAAI 2026 oral benchmark (1,672 clips/435k frames across table tennis/tennis/badminton) showing cross-attention fusion of ball-tracking and racket-pose features beats naive concatenation for trajectory prediction and hit-moment localization.
- **evidence:** arXiv 2511.17045 v3 (Jan 2026)
- **expected_gain:** best config: tennis mAP 81.9/MDE 1.96px, badminton mAP 83.1/MDE 1.54px, table tennis mAP 71.1/MDE 3.41px — architecture pattern directly maps onto our ball+paddle fusion problem
- **confidence:** medium-high — code public (github.com/OrcustD/RacketVision), peer-reviewed metrics, but racket-sport-general not pickleball-specific
- **url:** https://arxiv.org/abs/2511.17045

### [SOON] Uplifting Table Tennis (RoPE timestamp-aware 2D-to-3D lifting transformer) → P1/P2 — 2D-to-3D lift stage sitting between the WASB/blurball/TrackNetV4 ensemble and the physics arc solver, directly addressing our documented zero-shot wall
- **what:** Transformer that lifts noisy 2D ball detections to 3D position+spin using Rotary Positional Embeddings keyed on exact per-frame timestamps (not frame index), trained purely on physically-simulated synthetic trajectories, validated against real video via 2D reprojection error since no 3D ground truth exists.
- **evidence:** arXiv 2511.20250 (WACV 2026), github.com/KieDani/UpliftingTableTennis
- **expected_gain:** under missing-detection stress: 5.56px reprojection error (RoPE model) vs 24.15px (non-RoPE baseline); under combined frame-drop+missing-detection stress, spin accuracy 97.1% vs 67.7% baseline — directly the failure mode (missing/dropped detections) our BALL chain fights
- **confidence:** medium — WACV 2026 accepted, code public, large measured robustness gains, but table-tennis physics (spin/mass ratio, indoor lighting) differs from pickleball
- **url:** https://arxiv.org/abs/2511.20250

### [SOON] Multi-Focus Temporal Shifting (MFS) event spotting → P6 — hit/bounce/serve event timing, complementing or replacing bespoke logic in the physics arc solver's bounce-anchor recovery
- **what:** Lightweight temporal module (multi-scale ±1/±3/±6 frame shift + grouped spatial-focus attention) for frame-exact sports event spotting (hit/bounce/serve), only +1.3% params/+2.7 GFLOPs over baseline.
- **evidence:** arXiv 2507.07381 v3 (Dec 2025)
- **expected_gain:** tennis δ=0 tolerance: 55.51% mAP vs prior SOTA T-DEED-GSM 48.82%, at 33% fewer FLOPs than a heavier ASTRM baseline — minimal-overhead SOTA event-timing precision
- **confidence:** medium — code/data stated public, validated on tennis (racket sport) with new TTA benchmark, not pickleball
- **url:** https://arxiv.org/abs/2507.07381

### [SOON] CoachMe (reference-based coaching instruction generation) → P7 — grounded-LLM coaching stage, as the reference-comparison layer feeding an LLM
- **what:** Aligns a learner's motion against a reference/expert motion temporally+spatially via a skeletal-graph attention mechanism, computes per-joint deviations, and generates natural-language corrective instructions; domain-adapts from general movement pretraining to a new sport with limited sport-specific data.
- **evidence:** arXiv 2509.11698 (ACL 2025)
- **expected_gain:** beats GPT-4o by +31.6% G-Eval (figure skating) / +58.3% (boxing) on instruction quality — strong evidence that reference-motion comparison beats raw-LLM captioning for coaching text
- **confidence:** medium-high — ACL 2025 long paper, code released (MotionXperts/MotionExpert), explicitly targets our exact low-data-per-sport regime
- **url:** https://arxiv.org/abs/2509.11698

### [SPIKE] ExpertEdit (unpaired expert-motion editing/skill transfer) → P7 — coaching stage's compare-to-pro reference generation, feeding CoachMe-style deviation scoring without needing paired pickleball footage
- **what:** Trains an expert-motion prior on UNPAIRED expert clips (masked-motion-modeling), then edits a captured 3D motion toward higher skill to synthesize a personalized 'ideal technique' target skeleton — no matched novice/pro pairs required.
- **evidence:** arXiv 2604.10466 (UT Austin, Apr 2026)
- **expected_gain:** directly solves the 'compare-to-pro without paired data' problem given our scarce-in-domain-data constraint
- **confidence:** low-medium — very new (Apr 2026), validated on Ego-Exo4D + karate, not racket sports; would need a bank of expert pickleball clips (owner captures)
- **url:** https://arxiv.org/abs/2604.10466

### [SPIKE] BioCoach (biomechanics-grounded VLM coaching with DOF selector) → P7 — grounded-LLM coaching stage, specifically the feature-selection + grounding pattern for turning SAM-3D-Body + paddle 6-DOF into text
- **what:** 3-stage VLM coaching framework: exercise-specific degree-of-freedom selector (picks relevant joints/angles per skill) -> structured biomechanical-context module (numeric/verbal joint descriptors) -> vision+biomechanics-conditioned LLM feedback generator, explicitly designed to reduce hallucination by grounding text in real joint numbers.
- **evidence:** arXiv 2603.26938
- **expected_gain:** DOF-selector idea maps directly onto choosing paddle/wrist/hip-rotation features per pickleball shot type
- **confidence:** low — very recent (Mar 2026), fitness-domain not racket-sport, no code-availability confirmed
- **url:** https://arxiv.org/abs/2603.26938
