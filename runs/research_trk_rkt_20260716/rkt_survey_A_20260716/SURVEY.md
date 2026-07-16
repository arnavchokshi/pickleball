# RKT survey A — paddle detection and 6DoF orientation

Date: 2026-07-16  
Lane: `rkt_survey_A_20260716` (independent; no sibling research lane was read)  
Status: research only; no benchmark or GPU run was dispatched; **VERIFIED=0** remains binding.

## Executive decision

- [FETCHED-PRIMARY] No paper, repository, checkpoint, or benchmark found reports 6DoF accuracy for a **<80 px, motion-blurred, partially occluded, near-planar sports implement from one fixed consumer RGB camera**. BOP results are therefore not evidence for DinkVision's regime.
- [INFERENCE] The highest-leverage first path remains **RacketVision 5-keypoint detection -> exact/approximate paddle CAD -> both IPPE solutions -> a temporal two-hypothesis graph**. It directly attacks the visibility problem, is cheap to adapt, and matches the pipeline's repaired ambiguity representation. It is plausible for the interim <=30 degree milestone, but published evidence does not support <=5 degree p90.
- [INFERENCE] The best general-CAD methods are useful as **offline proposal/refinement oracles**, not first product candidates. GigaPose/RefPose/RACE-6D are the most interesting surrounding challengers; FoundationPose, SAM-6D, BundleTrack, and BundleSDF need depth or other incompatible inputs.
- [INFERENCE] The binding gap is observability plus ground truth. The face normal can be weakly or non-observable when the paddle is edge-on, blurred, or hidden. No model choice removes that. A blur-aware temporal posterior and gate-credible marker GT must be built.

## Ranking summary

| Rank | Candidate | Product input fit | Published evidence | Runtime reported by primary source | License posture | Adaptation / gate leverage |
|---|---|---|---|---|---|---|
| 1 | RacketVision -> both-IPPE -> temporal graph | RGB; sports-specific 2D keypoints | 1080p broadcast racket keypoints, but no 6DoF, pixel-size stratification, blur stratification, or pickleball | Not reported end-to-end for 6DoF | code MIT; data/weights **unknown-needs-review** | Low/medium cost; high leverage for <=30 degrees, unknown for <=5 degrees |
| 2 | RACE-6D trained on exact paddle | Monocular RGB, known object, joint detection+pose | 76.7% BOP AR on YCB-Video and 16.6 FPS for RGB; no tiny/blur study | 16.6 FPS | Apache-2.0 code; no released checkpoint | High training cost; medium leverage if synthetic+real pose GT exists |
| 3 | GigaPose coarse + MegaPose/RefPose refinement | RGB+CAD+crop/mask | BOP unseen-object results only | 0.9 s coarse; 7.3 s with MegaPose in FoundPose comparison | MIT code; released weights/data **unknown-needs-review** | Medium integration cost; useful offline oracle, weak real-time fit |
| 4 | RefPose / Pos3R / RayPose | RGB+CAD/templates+mask | Stronger BOP numbers, but no small/blur evidence | RefPose 3.1 s coarse / 3.9 s refined; Pos3R 1.4 s coarse / 8.0 s refined; RayPose not product-timed | no official code/weights found; **unknown-needs-review** | Research reference; not currently runnable off the shelf |
| 5 | MegaPose / FoundPose | RGB+CAD+ROI | BOP only; FoundPose public code omits refinement | MegaPose 15.5 s coarse; FoundPose 1.7 s coarse (all objects/image) | MegaPose code Apache-2.0, weights/data review; FoundPose CC BY-NC 4.0 | Offline oracle only; limited direct leverage |
| 6 | FoundationPose / SAM-6D / SinRef-6D | RGB-D in released/evaluated path | Strong robotics/BOP evidence, not product modality | FoundationPose 1.3 s registration, ~32 Hz tracking; SAM-6D not clearly reported | FoundationPose **R&D-only**; SAM-6D unknown; SinRef code MIT but weights/data review | Lab oracle only; no product fit |
| 7 | SpotPose / GCE-Pose category-level | RGB-D crop + point cloud | SpotPose 59.7 mAP at 5 degrees/2 cm on REAL275 | Not a real-time product result | no SpotPose official code found; **unknown-needs-review** | Low leverage: no paddle category, no thin/fast benchmark |

All numerical rows above are [FETCHED-PRIMARY] except the adaptation/leverage judgments, which are [INFERENCE]. Runtime values for GigaPose/MegaPose/FoundPose/Pos3R are for all objects in one BOP image and are not 60 fps claims.

## 1. Small/fast-object 6DoF SOTA

### Instance-level methods with CAD

#### 1. RACE-6D (CVPR Findings 2026) — most actionable new trained challenger

- [FETCHED-PRIMARY] RACE-6D extends RT-DETR to jointly output class, 2D box, continuous 6D rotation, depth/translation, keypoints, and visibility for **known objects**. The RGB model reports 76.7% BOP AR on YCB-Video at 16.6 FPS; the paper reports at least a 5.1x speedup over compared recent methods. [Paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf), [code](https://github.com/Yoonwoo-Ha/RACE-6D).
- [FETCHED-PRIMARY] The repository is live under Apache-2.0 and supplies training/evaluation/ONNX code, but has no GitHub release or published checkpoint. Training needs BOP-style pose annotations and CAD models.
- [INFERENCE] This is the strongest architecture-class alternative after a keypoint baseline because it can train on the exact owner CAD and run near real time. Its direct rotation regression may nevertheless collapse an ambiguous planar posterior, so a DinkVision adaptation should expose keypoint heatmaps or multiple rotation modes rather than keep only one rotation.
- [INFERENCE] Expected leverage: medium for <=30 degrees; unknown for <=5 degrees. Adaptation cost: high (synthetic renderer, real 6DoF labels, output-distribution changes).

#### 2. GigaPose (CVPR 2024) — fastest released coarse CAD proposal

- [FETCHED-PRIMARY] GigaPose uses one template correspondence plus learned in-plane/scale/translation estimation. In the FoundPose/Pos3R comparison it reaches 27.6 mean BOP AR at 0.9 s for all objects/image, and 57.9 AR at 7.3 s when coupled to MegaPose refinement. [Paper/code](https://github.com/nv-nguyen/gigapose), [checkpoint](https://huggingface.co/datasets/nv-nguyen/gigaPose/blob/main/gigaPose_v1.ckpt).
- [FETCHED-PRIMARY] Code is MIT. The 3.81 GB checkpoint is live. Training supports GSO, ShapeNet, or both; the released card labels the repository MIT but does not isolate upstream asset rights.
- [INFERENCE] Use as an offline crop-conditioned proposal generator and compare its hypotheses with IPPE. It has no published <80 px, blur, edge-on, or paddle result; it is too slow for 60 fps.

#### 3. RefPose, Pos3R, RayPose — newer accuracy ideas without a runnable product stack

- [FETCHED-PRIMARY] RefPose (CVPR 2025) predicts coarse geometric correspondences and then performs render-and-compare refinement. It reports 38.1 mean BOP AR at 3.1 s coarse and 61.4 AR at 3.9 s refined, outperforming its cited FoundPose/MegaPose combinations. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_RefPose_Leveraging_Reference_Geometric_Correspondences_for_Accurate_6D_Pose_Estimation_CVPR_2025_paper.pdf). No official code or weights were linked from the CVF paper page.
- [FETCHED-PRIMARY] Pos3R (CVPR 2025) uses MASt3R-derived 3D features with 40 CAD templates and PnP. It reports 39.5 mean BOP AR at 1.4 s coarse and 57.3 AR at 8.0 s after MegaPose refinement. Its paper explicitly identifies occlusion-driven matching failures. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Pos3R_6D_Pose_Estimation_for_Unseen_Objects_Made_Easy_CVPR_2025_paper.pdf). No official code/weights were linked.
- [FETCHED-PRIMARY] RayPose (ICCV 2025) learns diffusion over rays from multiple posed templates. The official CVF page links paper/supplement only, not code or weights. [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_RayPose_Ray_Bundling_Diffusion_for_Template_Views_in_Unseen_6D_ICCV_2025_paper.html).
- [INFERENCE] These methods contribute correspondence/refinement designs, but none is an off-the-shelf DinkVision candidate today. Their BOP gains do not answer whether a 40 px blurred planar crop contains enough evidence.

#### 4. MegaPose and FoundPose — useful established oracles, not small-object proof

- [FETCHED-PRIMARY] MegaPose accepts RGB (depth optional), intrinsics, CAD, and an object ROI. It trained on 2 million synthetic images of >20,000 GSO/ShapeNet objects. Pos3R reports its coarse runtime as 15.5 s/all objects and mean BOP AR 20.8. [Paper](https://arxiv.org/abs/2212.06870), [code](https://github.com/megapose6d/megapose6d), [models](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/).
- [FETCHED-PRIMARY] MegaPose code is Apache-2.0. The README warns users to ensure they have rights to original ShapeNet/GSO assets; therefore released weights/training data are not automatically commercial-clean.
- [FETCHED-PRIMARY] FoundPose uses DINOv2 template features and bag-of-visual-words retrieval. It reports 37.2 mean BOP AR at 1.7 s/all objects. The live repository explicitly releases **coarse pose only**, not the paper's featuremetric refinement, and uses CC BY-NC 4.0. [Paper](https://arxiv.org/abs/2311.18809), [code](https://github.com/facebookresearch/foundpose).
- [INFERENCE] These are good offline upper-bound tests on tightly cropped, relatively sharp frames. If they fail there, more integration is not justified. If they pass only on large crops, they still do not validate product capture.

#### 5. FoundationPose, SAM-6D, SinRef-6D — depth mismatch

- [FETCHED-PRIMARY] FoundationPose's released/evaluated path uses RGB-D plus a mask and either a mesh or reference views. It reports 1.3 s registration/object on RTX 3090 and ~32 Hz tracking; tracking uses the prior pose as the sole hypothesis. [Paper](https://arxiv.org/abs/2312.08344), [code](https://github.com/NVlabs/FoundationPose).
- [FETCHED-PRIMARY] Its NVIDIA Source Code License permits non-commercial research/evaluation only. The repository also states that diffusion-augmented training data/weights were not released for Stable Diffusion/LAION legal reasons and that the released model is slightly degraded.
- [FETCHED-PRIMARY] SAM-6D is RGB-D: SAM/FastSAM segmentation plus 3D-3D matching. Code and a Drive checkpoint are live, but the repo has no LICENSE file or SPDX license. [Paper](https://arxiv.org/abs/2311.15707), [code](https://github.com/JiehongLin/SAM-6D).
- [FETCHED-PRIMARY] SinRef-6D (T-RO 2026) uses a single pose-labeled **RGB-D** reference and RGB-D query, with RGB/point features and rigid alignment. MIT code and a weight link are live, but its input advantage is unavailable in product inference. [Code](https://github.com/CNJianLiu/SinRef-6D).
- [INFERENCE] They are useful only as lab/offline RGB-D comparators. FoundationPose's fast tracker is especially risky for a fast swing because local refinement assumes a good prior and outputs a single pose.

### Is there published evidence at <80 px with motion blur?

- [FETCHED-PRIMARY] None of the papers above reports accuracy binned at <80 px object extent, blur length/exposure, edge-on face angle, or sports contact. Their reported metrics aggregate BOP scenes and do not expose DinkVision's failure axes.
- [FETCHED-PRIMARY] BOP's own task definition assumes CAD models “typically with a color texture”; BOP-Classic test scenes are object-centric robotics captures. [BOP tasks](https://bop.felk.cvut.cz/tasks/). WACV 2026 BOP-Distrib further shows that ambiguity-aware re-annotation can change unseen-object rankings by as many as six places. [BOP-Distrib](https://openaccess.thecvf.com/content/WACV2026/papers/Meden_BOP-Distrib_Revisiting_6D_Pose_Estimation_Benchmarks_for_Better_Evaluation_under_WACV_2026_paper.pdf).
- [INFERENCE] BOP results should be treated as “can match/refine a visible object crop,” not “can recover a 40 px blurred paddle normal.”

### Category-level pose

- [FETCHED-PRIMARY] SpotPose (CVPR 2025) is a strong correspondence-based category model. On REAL275 it reports 59.7 mAP at 5 degrees/2 cm, but it uses Mask R-CNN masks, RGB-D point clouds (1,024 points), and 224x224 resized crops. It covers six NOCS household categories, not paddles. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Rethinking_Correspondence-based_Category-Level_Object_Pose_Estimation_CVPR_2025_paper.pdf).
- [FETCHED-PRIMARY] GCE-Pose (CVPR 2025) likewise starts from a partial RGB-D instance and reconstructs category geometry; HouseCat6D/REAL275 contain household categories and static object captures, not thin fast equipment. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_GCE-Pose_Global_Context_Enhancement_for_Category-level_Object_Pose_Estimation_CVPR_2025_paper.pdf), [HouseCat6D](https://arxiv.org/abs/2212.10428).
- [INFERENCE] No off-the-shelf category-level “paddle” model exists. Category learning is lower priority than exact-CAD instance pose until multiple-player/paddle generalization becomes the measured bottleneck.

## 2. Keypoint + PnP routes for planar objects

### RacketVision live recheck

- [FETCHED-PRIMARY] The AAAI-26 release contains 1,672 1080p clips, 435,179 frames, 24,621 sparse racket annotations, and five keypoints: top, bottom, handle, left, right. Racket pose is a **single-frame top-down RTMDet-M + RTMPose-M task**, not a temporal or 6DoF method. [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324), [repo](https://github.com/OrcustD/RacketVision), [dataset](https://huggingface.co/datasets/linfeng302/RacketVision).
- [FETCHED-PRIMARY] Multi-sport test results are: table tennis PCK@0.2 81.8 / MPJPE 9.71 px; tennis 89.6 / 5.34 px; badminton 88.5 / 5.00 px. Side-keypoint PCK is materially worse: 64.8/64.8 (table tennis), 79.7/80.1 (tennis), 74.6/75.5 (badminton). The paper attributes difficulty to hand occlusion, motion blur, and viewpoint sensitivity.
- [INFERENCE] For a line of projected length `L` with independent endpoint error `sigma`, in-plane angular noise is approximately `sqrt(2)*sigma/L`. At `sigma=5 px`, that is ~10 degrees for `L=40 px` and ~5 degrees for `L=80 px`, before out-of-plane planar conditioning. Published average MPJPE is therefore compatible with the <=30 degree milestone, but does not support <=5 degree p90.
- [FETCHED-PRIMARY] Code is MIT. The current HF dataset is 7.54 GB and labeled MIT. The HF model folder is 617 MB and lists pose weights of 411 MB and 107 MB, but has no model card/license and HF marks all six pickle files “unsafe” because they contain executable pickle imports.
- [INFERENCE] Commercial posture is **unknown-needs-review** for data/weights: the paper says source clips are professional YouTube broadcasts, while the repository/card provides no per-video commercial media grant. The MIT label does not itself prove rights to redistribute or commercially train on third-party broadcast footage.
- [FETCHED-PRIMARY] No independent primary-source follow-up using RacketVision for 6DoF, PnP, or pickleball was found through 2026-07-16. The only primary record found is the AAAI paper/release itself.

### IPPE ambiguity and resolution

- [FETCHED-PRIMARY] IPPE analytically returns two planar pose solutions; when corner noise is present, reprojection error alone can select the wrong one. The reference implementation is live under BSD-3-Clause. [Paper](https://encov.ip.uca.fr/publications/pubfiles/2014_Collins_etal_IJCV_plane.pdf), [code](https://github.com/tobycollins/IPPE).
- [FETCHED-PRIMARY] Multi-view marker work resolves ambiguity through global consistency/rotation averaging rather than a one-frame reprojection heuristic. [Robust rotation averaging](https://arxiv.org/abs/1909.11888). Object-SLAM work similarly represents pose predictions as a mixture and lets successive observations recover a globally consistent mode. [Multi-hypothesis object SLAM](https://arxiv.org/abs/2108.01225).
- [INFERENCE] The direct translation is a per-rally factor graph or Viterbi/beam search with two IPPE nodes plus a missing/abstain node per frame. Costs: keypoint heatmap likelihood, reprojection, wrist-grip transform, CAD silhouette/edge match, angular velocity/acceleration, and ball-contact compatibility. Hand, ball, and physics must remain soft factors; no paddle paper proves them as hard ambiguity resolvers.

### Motion blur: deblur first or blur-aware?

- [FETCHED-PRIMARY] BlurHandNet synthesizes blur by aggregating high-rate sharp frames and predicts a temporal hand-mesh sequence from one blurred image. It reports better blurry-image robustness than sharp-image pose models and explicitly preserves temporal information in the blur. [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Oh_Recovering_3D_Hand_Mesh_Sequence_From_a_Single_Blurry_Image_CVPR_2023_paper.html), [code](https://github.com/JaehaKim97/BlurHand_RELEASE).
- [FETCHED-PRIMARY] Human from Blur treats blur as forward image formation: a differentiable renderer temporally integrates sub-frame poses and optimizes the sequence explaining the blurred observation. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Human_from_Blur_Human_Pose_Tracking_from_Blurry_Images_ICCV_2023_paper.html).
- [INFERENCE] Prefer blur-aware training and, later, exposure-integrated render refinement over a standalone deblurring front end. Deblur can be an ablation, but hallucinated edges are dangerous for a two-solution planar problem. Synthetic training must match rolling shutter, exposure time, compression, and trajectory-dependent blur, not just apply a generic linear kernel.

### Other sports-equipment keypoint work in 2025-2026

- [FETCHED-PRIMARY] No other 2025-2026 primary release with code+weights was found for markerless monocular 2D keypoints plus 6DoF PnP on a tennis/badminton/table-tennis racket, baseball bat, or golf club. RacketVision remains the only live sports-equipment keypoint package located.
- [FETCHED-PRIMARY] The nearest 2026 racket work instead changes sensors: the badminton trajectory study uses an event camera and four OptiTrack markers, then evaluates the marker centroid as racket position; it does not publish phone-RGB paddle-face 6DoF. [Paper](https://www.nature.com/articles/s41598-026-46443-8).
- [INFERENCE] This negative result is consequential: equipment-specific geometry, blur-aware supervision, IPPE, and temporal face disambiguation must be assembled locally rather than imported as an existing bat/club/racket stack.

## 3. Video/temporal pose priors

- [FETCHED-PRIMARY] KV-Tracker (CVPR 2026) is the most relevant new monocular-RGB trajectory method: it caches key/value features from a multi-view network and reports up to ~30 FPS for online scene/object tracking and reconstruction without depth or an object prior. Evaluation is on TUM, 7-Scenes, Arctic, and OnePose—not tiny blurred sports equipment. [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Taher_KV-Tracker_Real-Time_Pose_Tracking_with_Transformers_CVPR_2026_paper.html), [code](https://github.com/Marwan99/kv_tracker). Its custom Imperial license is non-commercial/internal-or-academic only: **R&D-only**.
- [FETCHED-PRIMARY] BundleTrack is 10 Hz and BundleSDF is ~10 Hz, but both consume monocular RGB-D video; BundleSDF also needs a first-frame object mask. They demonstrate memory frames/pose-graph optimization under occlusion and texturelessness, not product compatibility. [BundleTrack](https://github.com/wenbowen123/BundleTrack), [BundleSDF](https://github.com/NVlabs/BundleSDF).
- [FETCHED-PRIMARY] The ICLR 2025 internet-video method retrieves a CAD-like mesh, aligns it per frame, tracks points, and optimizes a smooth trajectory. Its target trajectories are approximate and its pipeline is offline/heavy, but it validates trajectory-level reasoning over isolated pose selection. [Paper](https://openreview.net/forum?id=1CIUkpoata).
- [FETCHED-PRIMARY] A 2026 badminton study uses a DAVIS346 event camera plus OptiTrack GT and explicitly motivates event sensing by high-speed blur. This is the closest recent sports-equipment trajectory paper found, but its sensor is not a consumer RGB phone. [Paper](https://www.nature.com/articles/s41598-026-46443-8).
- [INFERENCE] Swing dynamics are useful only as weak priors: bounded jerk/angular acceleration, grip attachment, face/ball collision, and no teleportation. Player technique, grip changes, rolling shutter, and contact impulses make universal hard bounds unsafe.

## 4. Synthetic-data routes

### Tool ranking

1. **BlenderProc2 + Blender motion blur** — [FETCHED-PRIMARY] reads CAD, randomizes pose/material/light/camera/physics, and writes RGB/depth/normal/segmentation/BOP/COCO. Code is GPL-3.0. [Repo](https://github.com/DLR-RM/BlenderProc). [INFERENCE] Best first generator because MegaPose/GigaPose already speak BOP and the owner CAD is known. Treat GPL integration and every external asset license separately: **unknown-needs-review** until packaging is decided.
2. **Kubric** — [FETCHED-PRIMARY] Apache-2.0 framework using Blender+PyBullet for multi-object videos and rich annotations. [Repo](https://github.com/google-research/kubric). [INFERENCE] Commercial-clean code path if all scene/texture/hand assets are clean; more custom work than BlenderProc for BOP pose output.
3. **Isaac Sim Replicator** — [FETCHED-PRIMARY] its official object SDG tutorial randomizes camera, objects, light, velocity and produces path-traced motion blur from subframes. [Tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html), [repo](https://github.com/isaac-sim/IsaacSim). Repository code is Apache-2.0, but Omniverse Kit/dependencies/assets use additional NVIDIA terms: **unknown-needs-review**.

### What to render

- [INFERENCE] Render exact and approximate paddle CADs at native 1920x1080, not only enlarged crops. Stratify projected extent (20-40, 40-60, 60-80, 80-160 px), face angle, edge-on incidence, velocity, exposure, rolling-shutter readout, JPEG/HEVC artifacts, lighting, court backgrounds, grip hand/body occlusion, and left/right player location.
- [INFERENCE] Emit: exact 6DoF at exposure midpoint and subframes, four face corners, handle/grip keypoints, face normal, visibility fractions, segmentation, bbox, blur-path length, shutter metadata, and contact-point coordinates. Composite or render a hand/body so occlusion is geometrically correct.

### Sim2real evidence and limit

- [FETCHED-PRIMARY] MegaPose's 2M synthetic images and FoundationPose's large synthetic training set demonstrate broad zero-shot transfer on BOP. ROCK's synthetic-only object-centric keypoint model reports 59.4% overall average recall on its YCB-Video 6DoF evaluation, 10.8 points above weakly supervised Self6D(R) and 29.5 above Self6D. This is thresholded household-object recall, not angular p90. [ROCK paper/project](https://zhongcl-thu.github.io/rock/).
- [FETCHED-PRIMARY] None reports p90 face-normal error, <=5 degrees, a <80 px thin planar object, realistic swing blur, or hand occlusion. BOP aggregate AR and threshold mAP cannot be converted into DinkVision's p90 gates.
- [INFERENCE] Synthetic pretraining + a small real fine-tune is plausible and worth benchmarking; **synthetic-only <=5 degree p90 is unsupported**. Require real GT for calibration, domain-gap measurement, and final scoring.

## 5. Minimal owner GT capture recommendation

### Required GT accuracy

- [INFERENCE] To measure 5 degree / 3 cm gates credibly, reserve at most ~20% of each threshold for reference error: validate **p90 <=1 degree orientation**, **p90 <=5 mm paddle-frame/contact-location**, and a synchronization contribution <=5 mm. This is an engineering measurement target, not a published universal rule.
- [INFERENCE] A marker-position error `sigma` across rigid-body baseline `B` gives orientation noise roughly `sqrt(2)*sigma/B`. With 1 mm point error and 150 mm baseline that is ~0.54 degrees. A 1 degree rotation at a 200 mm paddle radius contributes ~3.5 mm positional error. Therefore use an asymmetric, non-coplanar marker cluster with >=150 mm effective baseline and validate it in the actual capture volume.
- [INFERENCE] Dynamic sync error is `v * delta_t`. To keep it under 5 mm, require `delta_t <= 5 mm / v` (0.5 ms at 10 m/s, 0.25 ms at 20 m/s). Unsynchronized 60/240 fps frame indices alone cannot establish that. Use a common hardware trigger where possible, plus a visible LED pulse and contact piezo/audio channel, then interpolate timestamped mocap poses to the product-camera shutter midpoint.
- [FETCHED-PRIMARY] HouseCat6D reports 1.35-1.74 mm GT error for static category-level objects. The 2026 badminton paper uses four retroreflective racket-head markers as an OptiTrack rigid body and acknowledges marker-occlusion noise at impact. These support a marker-based reference design, not a claim that any phone-only setup reaches 1 degree.

### Minimum gate-credible setup

- [INFERENCE] Product camera: the fixed 1080p60 phone in its real elevated geometry, with locked exposure/focus/white balance when possible and recorded rolling-shutter/exposure metadata.
- [INFERENCE] Reference: preferably rented 6-8 camera optical mocap at >=240 Hz; acceptable alternative is >=4 globally synchronized high-speed cameras with wide baselines. Two unsynchronized phones are useful training evidence but are **not gate-credible** unless a validation test proves the error budget.
- [INFERENCE] Paddle: exact CAD; >=4 small retroreflective markers in a non-coplanar asymmetric rigid cluster, with at least one handle/shaft marker to survive head occlusion. Measure the rigid transform from marker frame to CAD face/contact frame using a close-range calibrated jig/ChArUco capture.
- [INFERENCE] Calibration: large ChArUco boards for every camera's intrinsics/distortion and shared extrinsics; calibrated wand through the full swing volume; LED strobe/common trigger; rolling-shutter and exposure timing test.
- [INFERENCE] Contact: piezo/contact microphone for time, plus impact tape/transfer paint for a subset to provide an independent face-location label. Track a marked/contrasting ball in the high-speed views for continuous 3D ball-paddle intersection.

### Half-day contents

1. [INFERENCE] 30-45 min setup and validation: intrinsics/extrinsics, wand sweep, static rigid-body repeatability, LED/piezo sync test. Stop gate claims if validation misses 1 degree/5 mm/sync budget.
2. [INFERENCE] 60 static/slow poses spanning camera distance, face normals, both faces, edge-on, body-side occlusion, and frame corners. These calibrate marker-to-CAD and absolute bias.
3. [INFERENCE] 80 no-ball swings: dinks, drives, volleys, serves/overheads; forehand/backhand; slow/normal/max; intentional edge-on and body occlusion. Include exposure variants.
4. [INFERENCE] 80 contact trials across the same families, with known contact targets distributed over the face and impact tape on a validation subset.
5. [INFERENCE] 10-15 min natural rally footage with the same instrumentation for domain realism. If possible add two common commercial paddle shapes after the owner paddle.
6. [INFERENCE] Split by whole trial/session (and player/paddle when available), never adjacent frames. Publish raw timestamps, shutter/exposure, calibration, CAD transform, marker residuals, visibility, and contact labels. Score p90 overall and separately for 20-40/40-60/60-80/>80 px, blur, occlusion, edge-on, pre-contact/contact/post-contact.

## 6. Gap map — build-our-own specs

### Gap A — tiny blur-aware paddle evidence extractor

- **No off-the-shelf solution:** [FETCHED-PRIMARY] RacketVision is the only released sports-specific 5-keypoint baseline found, but side-keypoint PCK degrades and it has no 6DoF/blur-size bins.
- **Build:** [INFERENCE] input a 3-5 frame native-resolution window plus BODY wrist/palm ROI; use a high-resolution detector and heatmap keypoint head with visibility/blur heads and optional temporal deformable attention. Supervise 5 RacketVision points plus CAD corners, grip axis, segmentation, and uncertainty using synthetic pretraining + real marker GT. Never reduce heatmaps to one point before PnP.
- **Risks:** [INFERENCE] lost pixels, wrong player association, keypoint label swaps, synthetic hand/blur gap.

### Gap B — ambiguity-preserving trajectory inference

- **No off-the-shelf solution:** [FETCHED-PRIMARY] multi-hypothesis SLAM shows the principle, but no released paddle system combines two IPPE modes with hand/ball/swing evidence.
- **Build:** [INFERENCE] graph nodes `{IPPE_A, IPPE_B, learned-CAD proposal(s), missing}` per frame; factors for keypoint likelihood, silhouette, grip transform, continuous SE(3) velocity/acceleration, paddle-side identity, and probabilistic ball contact. Run beam/Viterbi or max-mixture smoothing; output both modes, posterior mass, covariance, and abstention.
- **GT:** [INFERENCE] full sequence 6DoF, keypoint visibility, exact contact time/location. **Risk:** a strong but wrong BODY/ball prior can lock the graph to the wrong face.

### Gap C — exposure-integrated planar render refinement

- **No off-the-shelf solution:** [FETCHED-PRIMARY] general render-and-compare assumes a single sharp observation; blur-aware human work integrates subframes but is not rigid-paddle code.
- **Build:** [INFERENCE] initialize from graph hypotheses; render a continuous paddle pose path across exposure and rolling-shutter rows; compare blurred edges/silhouette/appearance to the raw crop; optimize a short SE(3) spline and retain multimodality.
- **GT:** [INFERENCE] shutter timing and high-rate pose. **Risk:** textureless 20-40 px crops may remain unidentifiable, so refinement must be allowed to return broad uncertainty.

### Gap D — category/paddle-shape adapter

- **No off-the-shelf solution:** [FETCHED-PRIMARY] category SOTA is RGB-D household-object work; no paddle category or near-planar fast-object dataset exists.
- **Build:** [INFERENCE] owner exact CAD first; later a licensed CAD dictionary keyed by make/model, or low-dimensional outline/handle parameters selected from sharp frames. Share canonical semantic keypoints and learn shape-conditioned heatmaps/refinement.
- **GT:** [INFERENCE] paddle identity, dimensions/CAD, multi-shape pose sequences. **Risk:** wrong shape changes face center/contact geometry and hides systematic bias.

### Gap E — continuous-time contact estimator and evaluator

- **No off-the-shelf solution:** [FETCHED-PRIMARY] no surveyed product stack jointly estimates ball trajectory, paddle posterior, contact time, and face location at the required gate.
- **Build:** [INFERENCE] intersect probabilistic continuous-time ball and paddle trajectories with CAD face/thickness; infer contact time/location and propagate covariance. Train/calibrate against piezo timing + impact-tape location. Reject frames/trials whose posterior cannot support 3 cm.
- **Risk:** [INFERENCE] upstream BALL timing/3D error dominates; this is why “no BALL regression” and separate reference tracking are mandatory.

## Changed since 2026-07-09

- [FETCHED-PRIMARY] **RACE-6D code is live** (Apache-2.0, CVPR Findings 2026): 16.6 FPS RGB, but no released checkpoint/release. It is the clearest new exact-CAD trained challenger around the prior RacketVision ruling.
- [FETCHED-PRIMARY] **KV-Tracker code is live** and the CVPR 2026 paper reports ~30 FPS monocular RGB online object tracking/reconstruction; license is explicitly non-commercial. It is architectural evidence for trajectory processing, not small-paddle proof.
- [FETCHED-PRIMARY] **RacketVision release details are now concrete:** 7.54 GB dataset; 617 MB model folder; 411 MB and 107 MB pose checkpoints; no model card; HF unsafe-pickle flags. Dataset/card naming also has both `*_coco.json` and `*_pose_coco.json`; the live tree should be treated as authoritative for retrieval.
- [FETCHED-PRIMARY] **BOP-Distrib (WACV 2026)** demonstrates ranking changes under pose-distribution ambiguity, reinforcing the decision not to collapse planar hypotheses.
- [FETCHED-PRIMARY] **The closest 2026 high-speed racket papers use event/high-speed/mocap sensing**, not a single RGB phone. That is negative evidence for an off-the-shelf product solution.

## Source disagreements / caveats

- [FETCHED-PRIMARY] RacketVision calls its task “racket pose,” but its released labels and metrics are 2D keypoints. It is not published 6DoF.
- [FETCHED-PRIMARY] The RacketVision project says MIT, while the model hub has no model card/license and the source videos are third-party broadcasts. [INFERENCE] Code may be commercial-clean; weights/data still need rights review.
- [FETCHED-PRIMARY] FoundPose paper includes featuremetric refinement; public code explicitly omits it.
- [FETCHED-PRIMARY] FoundationPose paper results include a stronger diffusion-texture training recipe; the released repository says those data/weights were withheld for legal reasons and the public model is slightly worse.
- [FETCHED-PRIMARY] RACE-6D's paper says code and models are available; the repository currently has code but no release/checkpoint. Treat weights as unavailable.
- [INFERENCE] “5 degrees/2 cm mAP” is not p90 and cannot be compared directly with DinkVision's p90 5 degrees/3 cm gate.

## Recommended benchmark order (spec only; no dispatch)

1. **Freeze GT and slices first.** Validate reference p90 <=1 degree/5 mm and sync budget; define projected-size, blur, occlusion, edge-on, and contact slices. If GT fails, stop.
2. **RacketVision zero-shot.** Run official detector+pose weights unchanged; record detection recall, per-keypoint error/visibility, both IPPE modes, oracle-best-of-two face angle, and unresolved posterior. This isolates perception from disambiguation.
3. **Pickleball fine-tune.** Synthetic CAD/blur/hand-occlusion pretrain plus a small real fine-tune; compare blur-aware augmentation against deblur-first; keep the same frozen GT and model size.
4. **Two-IPPE temporal graph.** Compare single-frame lower-reprojection selection, simple continuity, and full soft-factor graph. Report oracle-best-of-two and chosen-mode p90 separately.
5. **RACE-6D exact-CAD challenger.** Train from clean synthetic + the same real GT; require multi-mode/uncertainty output or evaluate direct-regression collapse explicitly.
6. **Offline CAD oracle bake-off.** On identical supplied crops/masks, compare GigaPose, MegaPose, and any reproducible RefPose/Pos3R release. Run sharp/large first, then the frozen tiny/blur slices. Do not use runtime-incompatible wins as product promotion.
7. **Contact estimator last.** Only after face-angle evidence exists, fuse the pose posterior with separately verified BALL and BODY outputs; score contact point with independent piezo/impact-tape/multiview GT and enforce no BALL/BODY regression.

## Bottom line

[INFERENCE] The build decision is not “pick the best BOP model.” It is: make RacketVision-style evidence blur-aware at native resolution, preserve both planar modes through a temporal posterior, and acquire gate-credible owner GT. General CAD models should benchmark the available visual information and supply extra proposals; none currently closes the <80 px blurred planar-observability gap. The <=5 degree / <=3 cm promotion gates remain unsupported and **VERIFIED=0**.
