# RKT survey lane B — data-first paddle pose for DinkVision

Date: 2026-07-16  
Lane: `rkt_survey_B_20260716` (independent; no sibling research directory was read)  
Status: research and benchmark specification only; **VERIFIED=0**

## Executive answer

- **Synthetic-only at face-angle p90 ≤5°: unsupported.** [INFERENCE] Modern generators can render exact CAD, masks, keypoints, depth, motion blur, and—in BlenderProc—an explicit rolling-shutter model. The strongest relevant papers, however, report aggregate ADD/BOP or mean-angle results on much larger, mostly rigid tabletop objects, not p90 face angle for a <40–80 px blurred planar paddle. No fetched primary source demonstrates the target regime or the target statistic.
- **Synthetic + a small real owner set: plausible, not proven.** [FETCHED-PRIMARY] Self6D shows that real-domain adaptation can materially close a synthetic gap (LineMOD ADD(-S) recall 40.1% synthetic-only → 58.9% with unlabeled real; real-labeled upper bound 86.9%), but LM-O remains 15.1% → 32.1% versus a 70.2% real-labeled upper bound. These are not angular p90 results and cannot be transferred numerically to pickleball ([Self6D paper](https://arxiv.org/pdf/2004.06468)).
- **Validation is the first bottleneck.** [INFERENCE] A credible owner-paddle gate needs an independently validated reference near ≤1–1.5° orientation p90 and ≤1 cm point p90. The minimum credible consumer rig is three phones total: the untouched product view plus two close cross-angle 120/240 fps GT views, with a measured ChArUco volume, large flat paddle fiducials, audio + LED synchronization, and a held-out static fixture audit.
- **Biggest missing asset:** a commercially usable, time-synchronized real dataset containing tiny/blurred paddle pixels, exact per-frame 6DoF with ambiguity, 3D ball through impact, and face-coordinate impact labels. [INFERENCE]

## 1. Synthetic data as the primary route

### Ranked generator options

| Rank | Stack | What is primary-source supported | Missing for this task | License triage | Verdict |
|---:|---|---|---|---|---|
| 1 | **BlenderProc 2 / current BlenderProc** | [FETCHED-PRIMARY] PBR/physics, BOP-style pose outputs, transparent/random-background compositing, and a maintained example that calls `enable_motion_blur(..., rolling_shutter_type="TOP", rolling_shutter_length=...)`. The example defines blur as a fraction of inter-frame time and rolling-shutter length per scanline ([repo](https://github.com/DLR-RM/BlenderProc), [blur/RS example](https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/advanced/motion_blur_rolling_shutter/README.md)). | Phone-specific exposure/readout is not calibrated automatically; no turnkey human paddle grasp; synthetic skin/cloth/contact remains a domain gap. | Code: **GPL-3.0**, commercial-clean only with GPL obligations. Blender/third-party CAD/HDRI/body assets: separate licenses, review individually. | Best MVP for exact owner CAD and controlled ablations. |
| 2 | **NVIDIA Isaac Sim / Replicator** | [FETCHED-PRIMARY] Mature object SDG, path-traced rendering, physics, rich annotations, and motion-blur/subframe controls are documented in the Isaac/Omniverse stack ([Replicator docs endpoint](https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html)). | The checked tutorial URL moved to the docs index; no fetched primary source established a native phone rolling-shutter camera model. Heavy dependency footprint; human grasp still custom. | Platform: **NVIDIA Omniverse License Agreement/EULA**, `unknown-needs-review`; sample code/assets can differ. | Use only if already operational; not the lean first build. |
| 3 | **Kubric** | [FETCHED-PRIMARY] PyBullet + Blender pipeline, deterministic scene generation, RGB/depth/flow/segmentation and object poses; MOVi-F includes motion-blurred video generation ([repo](https://github.com/google-research/kubric)). | Archived/limited maintenance signal; no fetched native rolling-shutter support; less BOP/CAD-pose-oriented than BlenderProc. | Code: **Apache-2.0**, commercial-clean. Assets: separate/unknown-needs-review. | Viable research alternative, weaker fit. |
| 4 | **Unity Perception** | [FETCHED-PRIMARY] Synthetic-data labelers, human-pose labeling and HDRP rendering ([repo](https://github.com/Unity-Technologies/com.unity.perception)). | Repository explicitly says the package is discontinued and no longer supported; no fetched rolling-shutter proof. | Package code: **Apache-2.0**, commercial-clean; Unity Editor/assets have separate terms. | Do not start a new production path here. |
| 5 | **Unreal Engine** | [FETCHED-PRIMARY] Movie Render Queue/engine motion-blur controls exist ([official motion-blur docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-motion-blur)). | No fetched turnkey BOP annotations, phone rolling shutter, or hand-object pose pipeline. Engine motion blur can carry velocity-buffer artifacts. | **Unreal Engine EULA**, `unknown-needs-review`. | Only if an existing Unreal content pipeline offsets integration cost. |

### Proposed synthetic pipeline

1. **Geometry and labels.** [INFERENCE] Model the owner's face, edge, handle, grip wrap, thickness, center of mass, canonical face axes, and dense face coordinates. Export exact camera-space pose, visible/occluded 2D landmarks, masks, depth, optical flow, both face-normal signs, and a symmetry/ambiguity label.
2. **Motion source.** [INFERENCE] Begin with owner GT paddle/wrist trajectories rather than generic animation. Fit a rigged MANO/SMPL-X hand to those trajectories; add randomized grip placement, finger scale, sleeves, body occlusion, and left/right hands. This preserves sports kinematics while synthetic rendering supplies appearance breadth.
3. **Camera match.** [INFERENCE] Sample measured intrinsics/distortion, source-camera distance/elevation, exposure, readout direction/time, 60 fps cadence, compression, focus error and motion. Match blur length and rolling-shutter skew to calibration clips, rather than visually choosing them.
4. **Background integration.** [FETCHED-PRIMARY] BlenderProc supports transparent-object/random-image background composition. [INFERENCE] For DinkVision, use actual empty-court plates, camera-matched lighting/HDRIs, shadow catchers and body/hand occlusion mattes; an alpha paste without shadows/occlusion is an intentionally weak ablation, not the main training domain.
5. **Curriculum.** [INFERENCE] Oversample 20–80 px paddles, edge-on silhouettes, motion blur, hand/body occlusion, compression and textureless faces. Preserve realistic joint distributions: blur × edge-on × occlusion must co-occur, not be independently rare.
6. **Real adaptation.** [INFERENCE] Fine-tune/calibrate only on a small owner training split; keep games/sessions/camera days disjoint. Synthetic-only, real-only-small, synthetic→real and mixed training must all face the same frozen real test gate.

### Human-grasp source reality

| Source | Primary evidence | Fit | License triage |
|---|---|---|---|
| **GraspXL** | [FETCHED-PRIMARY] Policy trained on 58 objects reports 82.2% grasp success on >500k unseen objects; the hosted data card describes >10M motions and >500k objects ([paper](https://arxiv.org/abs/2403.19649), [dataset card](https://huggingface.co/datasets/ethHuiZhang/GraspXL)). | Useful diversity prior, but an optimized grasp is not a human paddle swing and says nothing about image realism. | Data: **CC BY-NC 4.0**, `R&D-only`; code/third-party meshes need separate review. |
| **GRAB** | [FETCHED-PRIMARY] Full-body/hand/object interactions, 10 subjects, 51 objects, 120 fps and marker-based capture ([repo](https://github.com/otaheri/GRAB)). | Best real human-interaction prior fetched, but no paddle/contact dynamics. | Custom **non-commercial scientific research** license, `R&D-only`. |
| **OakInk** | [FETCHED-PRIMARY] Project reports 230k frames, 12 subjects, 100 objects/32 categories, mocap object poses and MANO fits ([project](https://oakink.net/)). | Hand-object pose prior only; no sports blur or paddle. | No clear fetched data license: `unknown-needs-review`. |

**Finding:** no fetched source is a commercial-clean, turnkey human-paddle grasp/swing generator. [INFERENCE] The buildable route is owner-captured kinematics + a manually registered hand rig; GraspXL/GRAB/OakInk can inform research but must not silently contaminate deployable training assets.

### Published sim-to-real outcomes: strongest positive and negative

| Evidence | Exact published result | What it does / does not establish |
|---|---|---|
| **DOPE — positive synthetic-only evidence** | [FETCHED-PRIMARY] On real YCB-Video sugar-box imagery, mixed domain-randomized + photoreal synthetic training reached **77.00 ADD AUC**, versus **66.64** domain-randomized-only and **62.94** photoreal-only. The paper reports synthetic-only higher AUC than PoseCNN real+synthetic on 4/5 evaluated objects ([DOPE paper](https://proceedings.mlr.press/v87/tremblay18a/tremblay18a.pdf)). | Strong proof that synthetic composition can transfer to real rigid objects. It is not thin, <80 px, edge-on, sporting motion, angular p90, or contact localization. The paper's reflective potted-meat can failed under severe unmodeled occlusion: a relevant warning. |
| **MegaPose — broad positive, but metric/regime mismatch** | [FETCHED-PRIMARY] Synthetic-trained MegaPose reports mean BOP AR **54.5 RGB / 57.2 RGB-D** on seven real datasets. On ModelNet refinement, its 5°/5 cm accuracy is **88.6% RGB / 97.6% RGB-D** ([paper](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf)). | The 5° number is closest in threshold but is a refinement experiment from noisy GT initialization, not p90 end-to-end sim2real. It cannot certify DinkVision. |
| **Self6D — strongest negative/domain-gap result** | [FETCHED-PRIMARY] LineMOD ADD(-S) recall: synthetic-only **40.1%**, self-supervised real adaptation **58.9%**, real-label upper bound **86.9%**. Occluded LineMOD: **15.1%**, **32.1%**, **70.2%**, respectively ([paper](https://arxiv.org/pdf/2004.06468)). | Real images help, yet a large gap remains, especially under occlusion. No angular p90 or tiny planar sports case. |
| **RACE-6D — recent RGB reality check** | [FETCHED-PRIMARY] On YCB-V, mean rotation error is **10.3° RGB** and **8.0° RGB-D** ([CVPR 2026 Findings paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf)). | Even mean error is above 5° in a more favorable object-pixel regime; p90 would be stricter. This is evidence against assuming a generic direct model clears the gate. |

### Synthetic-to-real verdict with evidence

**Synthetic-only: unsupported at ≤5° face-angle p90. Synthetic + small real: plausible, not proven.** [INFERENCE] The renderers are mature enough; evidence, licensing-safe human occlusion, exact phone image formation, and target-domain GT are not. The first synthetic experiment should be a falsifiable owner-CAD keypoint/IPPE benchmark, not a platform build-out. Pass criteria remain the frozen real p90 gates; ADD/BOP/IoU and synthetic validation cannot substitute.

## 2. Ground-truth capture design

### What published GT systems demonstrate

| Method | Published accuracy/setup | Relevance and limitation |
|---|---|---|
| **Vicon + tiny IR markers** | [FETCHED-PRIMARY] Garon et al. use **8 Vicon MX-T40 cameras**, **3 mm** retroreflective markers, a 3×3×3 m volume, and report cited Vicon accuracy up to **0.15 mm static / 2 mm moving** ([paper](https://arxiv.org/pdf/1803.10075)). | Excellent dynamic reference and negligible visual footprint; expensive. Paper notes synchronization/calibration and marker-removal burdens. |
| **Robot-held reference + ICP** | [FETCHED-PRIMARY] PhoCaL's robot/hand-eye/ICP procedure reports **0.20 mm translation / 0.38° rotation RMSE** for pose refinement and about **0.89 mm** RGB-D hand-eye error ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf)). | Proves sub-degree static metrology is attainable; cannot label a free human swing. Use a robot/jig only to audit the rig. |
| **Two RGB-D cameras + HTC Vive** | [FETCHED-PRIMARY] Imitrob uses two 848×480@60 Hz RealSense cameras and 30 Hz Vive pose for 184k handheld-tool images; reported calibration residual is below 2 mm ([paper](https://arxiv.org/pdf/2209.07976)). | Closer hand occlusion, but a tracker attached to the tool changes geometry and 30 Hz is too sparse at contact. Data/code are **CC BY-NC-SA 4.0**, `R&D-only`. |
| **Multi-camera ChArUco/triangulation** | [FETCHED-PRIMARY] Anipose demonstrates precision ChArUco multi-camera calibration and bundle adjustment; in its six-camera microscope validation, >90% of board pose estimates were <20 μm length and <1° angle ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8498918/)). | Supports the method, not consumer-court accuracy. A new rig must prove its own held-out error. |

### Error budget for certifying 5° / 3 cm p90

[INFERENCE] Apply the requested 3–5× reference margin:

- **Reference orientation target:** p90 ≤1.0–1.5° for face normal; report full rotation and symmetry-aware normal error separately.
- **Reference point target:** p90 ≤1.0 cm in 3D and on face coordinates. Prefer ≤6–8 mm combined so GT does not consume a third of the 3 cm product budget.
- **Temporal target:** ≤1 ms residual at impact, or ≤0.25 of a 240 fps GT frame. The stated “≤0.5 frame” is acceptable only for the high-speed GT cameras (2.08 ms at 240 fps); **0.5 of the 60 fps product frame is 8.33 ms and is not enough**.
- **Reason:** at 10–20 m/s relative tangential speed, 1 ms corresponds to 1–2 cm and 8.33 ms to 8–17 cm. This is an error-budget calculation, not a measured DinkVision speed distribution.
- **Acceptance:** on held-out jig poses and repeat calibrations, the rig itself must meet both p90 targets. Reprojection error alone is not a 3D accuracy certificate.

### Minimal owner half-day capture spec

#### Rig

- **Cameras:** [INFERENCE] three phones total. `P0` is the unchanged fixed-ish production view at 1080p60. `G1/G2` are 1080p120 or 1080p240, fixed 2–3 m from the impact volume, with 60–100° convergence and useful vertical separation. Lock focus, exposure and white balance; target ~1/1000–1/2000 s if light permits. Preserve original timestamps/PTS and audio.
- **Why not two:** [INFERENCE] product + one GT camera leaves no triangulation redundancy when the hand/body hides a face or the paddle is edge-on. Two phones may prove feasibility; it cannot, by setup alone, certify the final p90 gate.
- **Volume calibration:** [FETCHED-PRIMARY] use ChArUco because interpolated chessboard corners remain usable under partial board visibility ([OpenCV docs](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)). [INFERENCE] Print/mount a rigid measured 0.8–1.2 m board with 60–100 mm squares; sweep it across depth, height and tilt throughout the actual hit volume before and after the session. Bundle-adjust all cameras and retain board measurements/calibration frames.
- **Paddle markers:** [INFERENCE] use thin matte, removable non-symmetric fiducials on **both faces**, four or more per visible face, registered to the exact CAD/face coordinates. Target 60–80 mm markers where possible. At `fx≈1400 px`, 70 mm width and 2.5 m distance project to ≈39 px; the design goal is ≥30–40 px in a GT view. Add a small non-coplanar handle-collar cluster only if it does not obstruct grip.
- **Mass/aero:** [INFERENCE] flat printed decals are the least intrusive option; weigh the paddle before/after and compare repeated swing-speed distributions. A protruding marker board or tracker changes mass/aero/grip and is suitable only for static jig validation. The 3 mm Vicon precedent shows why professional mocap can use much smaller markers, not that consumer phones can resolve them.
- **Synchronization:** [FETCHED-PRIMARY] audio correlation is a demonstrated consumer-camera synchronization method ([primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5051647/)). [INFERENCE] record a hard clap/impulse plus a bright LED visible to all cameras at start/end and every 1–2 minutes; correlate audio, fit a linear clock-drift warp, verify against LED transitions, and reject sequences with >1 ms residual or dropped/duplicated frames.

#### Half-day shot list and yield

| Block | Capture | Purpose |
|---|---|---|
| 0:00–0:30 | Intrinsics, ChArUco volume sweep, LED/audio sync, exposure/blur check | Calibrate and detect failure before swings |
| 0:30–1:00 | **90 held-out jig poses:** normals near 0, ±15, ±30, ±45, ±60° and near-edge-on; three translations; repeat after remount | Independent orientation/translation rig audit |
| 1:00–1:45 | **60 shadow/slow swings:** forehand/backhand dink, volley, drive, serve/overhead; left/right volume and deliberate hand/body occlusion | Pose coverage without ball/contact ambiguity |
| 1:45–3:15 | **150 controlled fed contacts:** ~25 per stroke family; varied open/closed face, speed, edge-on frames; center and safe off-center target regions | Owner pose + impact validation set |
| 3:15–3:45 | 10–15 minutes natural rally plus lighting/background variation | Real temporal/occlusion tails |
| 3:45–4:00 | Repeat ChArUco sweep, sync pulse and 15–20 jig poses | Detect drift; close metrology loop |

**Expected usable yield:** [INFERENCE] after rejecting events lacking two-view paddle/ball visibility or valid sync, roughly **100–180 contact events** and **2,000–5,000 high-confidence pose frames**. This is a planning estimate, not a guaranteed label count; log actual rejection reasons.

**Can certify:** the owner paddle, the production camera geometry, and the captured speed/lighting/occlusion envelope—only after the rig passes held-out p90 metrology.  
**Cannot certify:** unknown paddle models, unobserved lighting/speeds/camera placements, marker-free equivalence, or a universal product claim. Markers and extra cameras are validation-only; production input remains one camera.

## 3. Dataset landscape

### Sports datasets

| Dataset | Actual shipped supervision | Regime match | License triage |
|---|---|---|---|
| **RacketVision** | [FETCHED-PRIMARY] 942 professional broadcast videos sourced from YouTube, 1,672 clips, 435,179 frames / 12,755 s; 20% of frames sampled for manual labeling. It reports **24,621 racket annotations**, each a box plus five **2D** points (top, bottom, handle, left, right). It does **not** ship 3D paddle pose, camera intrinsics/extrinsics or face-coordinate contact GT ([AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324)). | Best fetched sports pretraining source for 2D structure and hard image formation. The multi-sport RTMPose baseline reports PCK@0.2 / MPJPE: table tennis **81.8% / 9.71 px**, tennis **89.6% / 5.34 px**, badminton **88.5% / 5.00 px**. Side points are materially weaker than structural points, supporting concern about face-plane cues under blur/view/occlusion. No pickleball and no metric-scale pose. | Code/annotation release: **MIT**, `commercial-clean` at license-text level ([repo/license](https://raw.githubusercontent.com/OrcustD/RacketVision/main/LICENSE)). Underlying YouTube broadcast video rights: `unknown-needs-review`; MIT metadata does not clear third-party footage. |
| **TT4D** | [FETCHED-PRIMARY] 45,946 broadcast table-tennis games, 211,534 reconstructed points and 146 h. Paddle trajectories are inferred by inverse control from reconstructed ball state rather than released measured per-frame paddle-pose GT. On 92 mocap strokes with IR racket markers, mean orientation error is **26.4 ± 4.4°** and velocity error **0.58 ± 0.40 m/s**; mean impact speed was 3.72 m/s ([paper](https://arxiv.org/pdf/2605.01234)). | Closest trajectory-level sports evidence. Its orientation result is near the 30° interim milestone, far from 5°, and broadcast table tennis differs in view/scale. | Paper says release upon acceptance; no fetched public data license/release artifact: `unknown-needs-review`. Broadcast provenance also needs review. |
| **Other tennis/badminton/table-tennis analytics sets** | [FETCHED-PRIMARY] The RacketVision paper's literature review identifies prior sport datasets around ball locations, player/action/stroke/event labels rather than racket 6DoF, motivating its own five-keypoint contribution. | May help ball/player/stroke priors, not face-angle/contact GT. | Per-dataset review required: `unknown-needs-review`; not ranked as pose training data. |

**RacketVision recheck conclusion:** [FETCHED-PRIMARY] calling its task “racket pose” means five 2D keypoints plus a box, not metric 6DoF. It remains useful auxiliary supervision for the already-settled first challenger, but it cannot evaluate either frozen gate.

### Industrial / hand-object pose datasets

| Dataset | Pose/GT content | Coverage versus DinkVision | License triage |
|---|---|---|---|
| **T-LESS (BOP)** | [FETCHED-PRIMARY] 30 textureless industrial objects, synchronized RGB/RGB-D sensors, tens of thousands of train/test images and accurate 6D annotations ([paper](https://arxiv.org/abs/1701.05498)). | Useful textureless/symmetry benchmark; mostly controlled/static viewpoint-sphere data with objects much larger/sharper than a 40–80 px moving paddle. Shapes are not thin sporting faces and there is no hand/contact. | Exact current data license was not resolved from the fetched paper: `unknown-needs-review`. BOP toolkit and every source dataset/mesh must be triaged separately. |
| **MVTec ITODD (BOP)** | [FETCHED-PRIMARY] Official page describes 28 industrial objects, 3,500 scenes and five sensing modalities ([official page](https://www.mvtec.com/research-teaching/datasets/mvtec-itodd)). | Textureless, cluttered industrial evaluation; static, close, no sporting motion/hand/rolling shutter. | Data: **CC BY-NC-SA 4.0**, `R&D-only`. |
| **Imitrob** | [FETCHED-PRIMARY] 184k images of hand-held tools with two RealSense cameras and Vive-derived 6D poses ([paper](https://arxiv.org/pdf/2209.07976)). | Valuable hand occlusion and moving rigid object; tools are large, tracker-assisted, 30 Hz pose reference and not planar/edge-on sports objects. | Code/data: **CC BY-NC-SA 4.0**, `R&D-only`. |
| **PhoCaL** | [FETCHED-PRIMARY] Multimodal category-level object pose with robot/hand-eye and ICP-validated labels ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf)). | High-quality static/category metrology and difficult materials; not dynamic small-object contact. | No exact deployable data/mesh license established in fetched materials: `unknown-needs-review`. |
| **GRAB / OakInk** | [FETCHED-PRIMARY] Human hand/body/object interaction with 3D object pose and fitted hands; see §1. | Useful priors for occlusion/compositing; neither supplies paddle faces, high-speed blur, rolling shutter, ball or contact. | GRAB `R&D-only`; OakInk `unknown-needs-review`. |

### Dataset gap conclusion

No fetched dataset jointly covers: **single elevated consumer view + 40–80 px planar sporting object + high angular velocity/rolling shutter + hand/body occlusion + metric face normal + face-coordinate ball impact**. [INFERENCE] This is not a minor domain adaptation gap; it removes the supervision needed to measure the product gate. The owner GT capture is therefore prerequisite evidence, not optional polish.

## 4. Method reality check

### Method elimination table

“Eliminated” here means **eliminated as the next primary challenger in the present regime**, not a universal claim that the family can never work.

| Family / representative | Required inputs and image quality | Primary evidence | Lane-B decision |
|---|---|---|---|
| **2D keypoints + PnP/IPPE, retaining both IPPE solutions** | Calibrated RGB; ≥4 stable non-collinear/coplanar correspondences; corners/handle endpoints must remain localizable. Edge-on collapse and weak-perspective noise amplify planar ambiguity. | [FETCHED-PRIMARY] IPPE analytically returns two candidate planar poses and documents ambiguity under weak perspective ([repo/paper links](https://github.com/tobycollins/IPPE)). RacketVision's side-point PCK lags structural points, providing sports-image evidence that the face-width cues are hardest. | **Keep, rank 1.** Exact owner CAD + synthetic heatmaps + both-solution temporal scoring is the cheapest falsifiable path. It is not validated until real p90 GT exists. |
| **MegaPose render-and-compare (RGB)** | RGB ROI, camera intrinsics, CAD mesh, initial box/detections; enough silhouette/appearance pixels for render comparison. Depth optional. | [FETCHED-PRIMARY] MegaPose assumes an ROI and CAD and is trained on synthetic renderings; real BOP mean AR is 54.5 RGB ([paper](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf), [repo](https://github.com/megapose6d/megapose6d)). | **Keep, rank 2 owner-CAD challenge.** Not eliminated, but no fetched evidence for <80 px blurred edge-on paddles. Code **Apache-2.0**; pretrained assets/upstream meshes/weights `unknown-needs-review`. |
| **GigaPose / template correspondence** | Segmented RGB crop + CAD render templates. The method resizes crops to 224×224 and produces a 16×16 feature map; small visible segments have limited appearance. | [FETCHED-PRIMARY] Paper reports LM-O robustness limits under occlusion and a failure from a small segment + low-fidelity CAD ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.pdf)). | **Defer / kill-test cheaply.** It is not outright impossible, but the known small-segment failure aligns with the regime. Code **MIT**; weights/upstream assets `unknown-needs-review`. |
| **FoundationPose as released** | Public model-based path requires RGB-D, CAD and a mask/ROI; model-free path needs a reference sequence. | [FETCHED-PRIMARY] Released repository documents RGB-D use ([repo](https://github.com/NVlabs/FoundationPose)); its license §3.3 restricts use to non-commercial research/evaluation ([license](https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE)). | **Eliminate as a product-path challenger:** wrong sensor assumption and `R&D-only` license. It may be an offline research ceiling only. |
| **FoundPose / other foundation-feature render matching** | RGB crop/segmentation, CAD/templates, enough visible texture/shape for feature matching. | [FETCHED-PRIMARY] FoundPose publishes a non-commercial research stack ([repo/license](https://github.com/facebookresearch/foundpose)). | **Defer.** No fetched small blurred planar sports evidence; **CC BY-NC 4.0**, `R&D-only`. |
| **Category-level pose** | Category training set with canonicalized shapes/poses, metric scale or shape model, masks/crops and enough features to resolve intra-class variation/symmetry. | [FETCHED-PRIMARY] Existing category datasets/methods target learned object categories; fetched paddle datasets provide only five 2D points, not category 6DoF/shape. | **Eliminate as current primary route.** Unknown paddles are the eventual motivation, but the required category 3D supervision does not exist. Revisit only after owner-instance proof and a licensed shape/GT set. |
| **Direct RGB regression (RACE-6D class)** | Object training data/CAD-derived labels, calibrated crop, sufficient pixels to regress rotation/translation; uncertainty handling needed for symmetries. | [FETCHED-PRIMARY] Recent RACE-6D mean rotation error is 10.3° RGB / 8.0° RGB-D on YCB-V ([paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf)). | **Eliminate as next primary route.** A >5° mean in easier imagery is not proof of impossibility, but is evidence against prioritizing it over geometry-aware instance methods. |
| **Trajectory/inverse-control pose (TT4D class)** | Long ball/player tracks, 3D court/camera, learned physics/control priors; can bridge invisible frames but inherits ball/camera errors. | [FETCHED-PRIMARY] TT4D's mocap comparison reports 26.4 ± 4.4° mean orientation error ([paper](https://arxiv.org/pdf/2605.01234)). | **Eliminate for the 5° gate; retain as interim/prior.** It could approach the ≤30° milestone or choose between IPPE hypotheses, but is not a precision pose source. |

### Pixel/blur implications

- [INFERENCE] At 40 px face width, a 1 px side-keypoint error is 2.5% of width; orientation sensitivity becomes highly pose- and calibration-dependent and diverges near edge-on. Therefore no universal “minimum pixels = N” is defensible from the fetched literature.
- [FETCHED-PRIMARY] The usable evidence is empirical: RacketVision side points are its weakest points; GigaPose explicitly flags small visible segments; RACE-6D remains above 5° mean on larger YCB objects; TT4D remains at 26.4° mean in sports. Together these eliminate confident off-the-shelf claims, not every possible custom model.
- [INFERENCE] Every retained method should be stratified by projected paddle width, visible face area, blur length, edge-on angle, occlusion fraction and repaired-confidence status. A single aggregate p90 can hide the precise contact regime.

## 5. Contact point specifically

### Published localization evidence

| Signal/method | Published result | Transfer judgment |
|---|---|---|
| **Single event-camera tennis** | [FETCHED-PRIMARY] 49/50 swing windows and 46/49 impact times were detected. When contours were successfully recovered, impact-position error was **<15 mm**. Contour success was **24/26 without direct sunlight but only 3/20 in direct sunlight** ([paper](https://arxiv.org/pdf/2506.08327)). | Strong proof that a high-temporal-resolution visual signal can beat 3 cm on successful cases; severe illumination brittleness and specialized event sensing prevent direct product transfer. |
| **Two event cameras, badminton** | [FETCHED-PRIMARY] 125 trials / 124 analyzable; impact location succeeded in **116/124 (93.5%)**. Bias: **1.84 ms** timing, **3.45 mm** medio-lateral, **−1.92 mm** longitudinal; 95% limits of agreement span **−3.35 to 10.24 mm** and **−10.63 to 6.78 mm** for the two face axes ([paper](https://arxiv.org/abs/2605.28011)). | Best fetched positive impact result, but two specialized event cameras, ~1.86 mm/px rear resolution, controlled smashes and high-speed reference are materially easier/different. |
| **Instrumented table-tennis paddle** | [FETCHED-PRIMARY] A 5×5 piezoelectric array with 1 cm spacing and 10 channels is proposed for impact sensing ([PLOS ONE paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0333735)). | Potential GT/debug signal, not product-compatible and the fetched study does not establish a 3 cm p90 benchmark under normal paddle construction. |
| **Ball-trajectory inverse dynamics** | [FETCHED-PRIMARY] TT4D derives racket state from 3D ball/control, but its 26.4 ± 4.4° mean orientation error shows the inverse is not a precision face estimate. | Useful temporal prior/oracle ablation, not independent contact truth. |

### Can ball track + face plane meet 3 cm?

[INFERENCE] In principle yes: estimate impact time `t*`, intersect a pre/post-impact 3D ball trajectory with the time-varying paddle plane, then express the intersection in face coordinates. A lower-bound-style budget is:

`e_contact ≳ sqrt(e_ball,face² + e_face_origin² + (r·e_angle)² + (v_rel,face·e_time)²)`

where `r` is distance from the paddle reference origin and `v_rel,face` is relative in-plane speed. Correlations and reconstruction bias can make the real error worse.

To leave margin under a 3 cm p90 gate, the contact-time **BALL 3D** point should be about **≤1 cm p90 in each face-relevant direction**, the face origin ≤1 cm p90, the normal ≤1–1.5° p90, and timing near ≤1 ms. [INFERENCE] A BALL estimate already at 3 cm p90 consumes the entire contact budget before pose/timing error. Nothing in this survey establishes that current DinkVision BALL meets these requirements.

### Cheapest reliable signal

**For GT:** audio/impulse timing + two-view high-speed geometry is cheapest and most defensible. [INFERENCE] Audio supplies a sub-frame event anchor but no `(u,v)` position on the paddle; geometry supplies location. LED/audio pulses synchronize cameras, while ball-trajectory discontinuity and visible paddle/ball determine the actual hit.

**For the one-camera product:** retain audio as a contact-time prior, then fuse ball trajectory with the distribution over both planar pose hypotheses. [INFERENCE] This is plausible but unsupported at 3 cm until oracle ablations separate timing, BALL, and pose errors. Audio alone cannot localize the strike on the face.

## 6. Gap map — build-our-own specs

| Gap | Buildable spec | Supervision | Main risks | Kill criterion |
|---|---|---|---|---|
| **Commercial-clean hand+paddle synthetic domain** | Inputs: exact owner CAD, measured camera/readout/exposure, owner wrist/paddle trajectories, licensed parametric hand/body, court plates. Output exact pose, keypoints, masks, visibility, ambiguity. Architecture: BlenderProc scene generator + MANO/SMPL-X-class rig or internally licensed equivalent. | Owner multiview GT trajectories; synthetic transforms are exact. | Human/cloth/occlusion realism; third-party body/asset licensing; implausible joint distributions. | Stop expanding synthetic scale if synthetic-only fails 30° interim on held-out real or synthetic pretraining worsens the same real fine-tune split. |
| **Tiny blurred planar pose model** | High-resolution temporal ROI; heatmap/distribution over top/bottom/handle/left/right plus optional dense face coordinates; both-IPPE solver; temporal hypothesis scoring and calibrated uncertainty. | Synthetic owner CAD + small real owner keypoint/6DoF split. | Side points vanish edge-on; blur bias; rolling-shutter model mismatch; confidence collapse. | Kill variant if it cannot beat the frozen current baseline at ≤30° p90 without worse tail/coverage, or if its apparent gain disappears by width/blur strata. |
| **Owner-CAD render comparator** | MegaPose RGB challenge using identical detections/masks/intrinsics; render exact face/edge/handle; output multi-hypothesis pose. | Synthetic pretraining plus owner real test only. | ROI/mask dominates, too few appearance pixels, symmetry flips, upstream weight licensing. | Kill if oracle-mask performance still misses 30° p90 or if runtime/coverage makes temporal use infeasible. |
| **Unknown-paddle canonical shape/pose** | Category morphable template from licensed CAD scans; box/mask + five points + canonical face axes; category keypoint/PnP or category pose distribution. | New licensed in-house multiview set across paddle models; RacketVision only as 2D auxiliary after legal review. | Shape/graphics variation, symmetric faces, no current category 6DoF labels, scale ambiguity. | Do not build until owner exact-CAD path proves value. Then kill if model-conditioned calibration cannot hold across unseen brands/shapes. |
| **Contact oracle and fusion** | Inputs: timestamped audio, 3D ball trajectory/covariance, time-varying paddle pose hypotheses/covariance. Factor graph or probabilistic plane-intersection model outputs face `(u,v)` and uncertainty. | Owner GT with face-coordinate impacts; event time from synchronized high-speed views/audio. | Timing dominates; ball disappears/deflects; correlated BALL/pose errors; paddle deformation. | First run oracles. Kill geometry-only product route if **GT time + GT face pose + predicted BALL** cannot meet 3 cm; that isolates BALL as limiting. Repeat with GT BALL/predicted pose. |
| **Ambiguity/uncertainty calibration** | Maintain both IPPE modes through a temporal factor graph with motion/hand/ball likelihoods; report mode probabilities, coverage and abstention. | Full owner sequences with GT pose; include edge-on/occluded tails. | Premature mode collapse; biased confidence from repaired inputs; temporal lag at impact. | Kill if expected calibration/coverage and p90 do not improve over retaining two unweighted modes, especially in edge-on strata. |
| **Commercially usable validation/training data** | Consent-based owner/multi-player capture with original synchronized media, CAD/marker registration, immutable raw observations and explicit rights. | The capture process itself; no broadcast dependency. | Small subject/paddle diversity; rig drift; privacy/legal paperwork; markers alter appearance. | Do not claim category/general-game performance until new players/paddles/days pass held-out gates; owner-only evidence stays owner-only. |

## Recommended benchmark order (specification only)

1. **GT rig gate before model work.** [INFERENCE] Freeze coordinate definitions and error metrics; pass held-out jig repeatability at face-normal p90 ≤1–1.5°, point p90 ≤1 cm, synchronization ≤1 ms. If not, repair capture—not the estimator.
2. **Score the current render-only baseline honestly.** [INFERENCE] Run on the owner real test split, retain both IPPE hypotheses/ambiguity and repaired-confidence flags, and report coverage plus p50/p90 by width/blur/occlusion/edge-on strata. Rectangle IoU is diagnostic only.
3. **Run the already-settled first challenger:** RacketVision five-keypoint zero-shot → pickleball/owner fine-tune, followed by both-IPPE. Use one frozen real split and no test-driven tuning.
4. **Synthetic owner-CAD keypoint benchmark.** [INFERENCE] Compare real-small, synthetic-only, synthetic→real and mixed. Ablate exact CAD, court compositing, hand/body occlusion, motion blur and rolling shutter one factor at a time, then test interactions.
5. **Owner-CAD MegaPose RGB challenge.** [INFERENCE] Use the same boxes/masks/intrinsics. Include oracle-box and oracle-mask runs so localization failure is not misattributed to pose. Do not substitute BOP AR for the frozen angular p90.
6. **Temporal ambiguity benchmark.** [INFERENCE] Add sequence inference only after frame methods are scored; compare single-best IPPE, two-mode persistence, hand/wrist prior and ball/physics prior. Measure lag at contact.
7. **Contact oracle ladder.** [INFERENCE] Evaluate `(GT time, GT BALL, GT face)`, then replace one input at a time: predicted time, predicted BALL, predicted face. This identifies the cheapest limiting sensor/model before building fusion.
8. **Unknown paddles last.** [INFERENCE] Only after owner exact-CAD evidence clears a meaningful gate should category data/model collection begin. No owner result promotes unknown-model capability.

### Frozen reporting contract for every candidate

- Face-angle error: symmetry policy stated; p50/p90 plus coverage/abstention; contact-window and whole-swing separately.
- Translation/contact: 3D and face-coordinate p50/p90; timing error separately; no 2D proxy as promotion evidence.
- Strata: projected width, visible face area, blur length, edge-on angle, occlusion, stroke, lighting, repaired confidence and ambiguity.
- Same immutable raw observations, same train/val/test session split, same GT version and same no-BALL/no-BODY-regression checks.
- Labels: `research-only`, `review-only`, `partial`, or `verified` exactly as gate evidence allows. This survey produces no promotion evidence; **VERIFIED=0 remains binding**.

## License ledger and disagreements

| Item | Code | Weights/models/assets | Data/video | Overall research-use label |
|---|---|---|---|---|
| BlenderProc | GPL-3.0 | Blender/CAD/HDRI/body assets separate | Generated data inherits input constraints | `commercial-clean` code with GPL obligations; assets review |
| Kubric | Apache-2.0 | source assets separate | generated data depends on assets | `commercial-clean` code; assets review |
| Isaac Sim/Replicator | NVIDIA Omniverse EULA | NVIDIA/third-party terms | generated data terms require review | `unknown-needs-review` |
| Unity Perception | Apache-2.0 package | Unity/editor/assets separate | generated data depends on assets | package `commercial-clean`; stack review |
| Unreal | Unreal Engine EULA | Marketplace/project assets separate | generated data terms require review | `unknown-needs-review` |
| GraspXL | not resolved | object assets separate | CC BY-NC 4.0 | `R&D-only` |
| GRAB | custom non-commercial | body/object assets covered/restricted by release | custom non-commercial | `R&D-only` |
| OakInk | not resolved | not resolved | no clear fetched license | `unknown-needs-review` |
| RacketVision | MIT | baseline weights/rependencies require review | annotation card MIT; YouTube footage rights unresolved | code/annotations `commercial-clean`; videos `unknown-needs-review` |
| MegaPose | Apache-2.0 | pretrained weights and GSO/ShapeNet dependencies require separate review | synthetic/BOP sources separate | code `commercial-clean`; deployed bundle `unknown-needs-review` |
| GigaPose | MIT | weights/templates/upstream meshes separate | BOP sources separate | code `commercial-clean`; deployed bundle `unknown-needs-review` |
| FoundationPose | NVIDIA custom non-commercial | same/referenced dependencies | benchmark data separate | `R&D-only` |
| FoundPose | CC BY-NC 4.0 | same/upstream assets separate | benchmark data separate | `R&D-only` |
| Imitrob | CC BY-NC-SA 4.0 | included release | CC BY-NC-SA 4.0 | `R&D-only` |
| ITODD | tools separate | object models under dataset terms | CC BY-NC-SA 4.0 | `R&D-only` |
| T-LESS / PhoCaL / TT4D | varies/not resolved | not resolved | exact deployable license not established here | `unknown-needs-review` |

Disagreements and cautions:

- [FETCHED-PRIMARY] “Racket pose” in RacketVision is five 2D points, while this product requires metric 6DoF and face-coordinate contact. Both descriptions can be linguistically true; the supervision is not equivalent.
- [FETCHED-PRIMARY] MegaPose's 88.6% at 5°/5 cm sounds closest to the gate, but it is ModelNet **refinement** accuracy, not end-to-end real small-object p90. The BOP real-image result uses average recall, not angular p90.
- [FETCHED-PRIMARY] BlenderProc has an explicit rolling-shutter control; no equivalent native phone rolling-shutter claim was verified for Kubric, Replicator, Unity or Unreal. Absence from fetched sources is not proof of absence.
- [INFERENCE] Published mean or recall numbers cannot be converted to p90 without the error distribution. This survey deliberately refuses that conversion.
- [INFERENCE] Flat decals likely perturb swing less than boards/trackers, but this was not quantified in a fetched paddle biomechanics study. The capture spec therefore requires weighing and an empirical marked-vs-unmarked swing check.
