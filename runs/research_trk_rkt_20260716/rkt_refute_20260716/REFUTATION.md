# RKT refutation lane: second-vote primary-source check

Fetch date: **2026-07-16** (America/Los_Angeles). This survey was performed independently and did not read either prohibited survey directory. A bundle is `CONFIRM` only when every material component survives. `REFUTE` means a material component is false; `PARTIAL` means the bundle mixes confirmed facts with a narrow falsehood or an absence claim that cannot be closed. Published benchmark results are not evidence of pickleball accuracy. **`VERIFIED=0` remains binding.**

## C1 — CONFIRM

**Primary URLs fetched**

- Paper: https://arxiv.org/abs/2511.17045 and https://arxiv.org/pdf/2511.17045
- Official repository, README, and license metadata: https://github.com/OrcustD/RacketVision, https://raw.githubusercontent.com/OrcustD/RacketVision/main/README.md, https://api.github.com/repos/OrcustD/RacketVision/license
- Dataset metadata: https://huggingface.co/api/datasets/linfeng302/RacketVision
- Model metadata/security and file HEADs: https://huggingface.co/api/models/linfeng302/RacketVision-Models?blobs=true&securityStatus=true, https://huggingface.co/api/models/linfeng302/RacketVision-Models/tree/main/checkpoints?recursive=false&expand=true, https://huggingface.co/linfeng302/RacketVision-Models/resolve/main/checkpoints/epoch_300.pth, https://huggingface.co/linfeng302/RacketVision-Models/resolve/main/checkpoints/best_PCK_epoch_90.pth?download=true

**Short evidence quotes**

- Paper/table: “942 ... 1,672 ... 435,179 ... 24,621”; baseline: “RTMDet-M” and “RTMPose-M.”
- Repository README annotation order: “top, bottom, handle, left, and right.”
- HF scan: “This file contains imports that may be unsafe.”

**Assessment**

- The paper reports 1,672 clips, 435,179 frames, 24,621 racket annotations, and 942 source broadcasts. It describes top-level professional matches collected from YouTube.
- The published racket annotation schema is a 2D box plus five 2D image keypoints. Neither the paper, repository schema, nor dataset card exposes a 3D/6DoF racket pose, camera-intrinsics field, or racket-contact ground truth. This is an absence statement about the published bundle/schema, not every latent fact inferable from video.
- The single-image baseline is RTMDet-M followed by RTMPose-M. PCK@0.2 / MPJPE is 81.8 / 9.71 px (table tennis), 89.6 / 5.34 px (tennis), and 88.5 / 5.00 px (badminton). Left/right PCK spans 64.8–80.1.
- GitHub reports MIT. The dataset card is MIT-tagged. The HF model repository has no README/model card and no license metadata.
- HF lists 616,512,333 bytes total (616.5 MB decimal). HEAD confirms `epoch_300.pth` = 411,293,859 bytes and `best_PCK_epoch_90.pth` = 106,524,759 bytes; no weight body was downloaded.
- HF currently gives all six `.pth` files repository-level `unsafe` status. More precisely, its pickle-import scan flags `__builtin__.getattr`; the same response reports JFrog `safe`, and the 411 MB file's VirusTotal result is 0/74. “Unsafe” is therefore an HF serialization/import warning, not proof of malware.

**Correction:** none to the stated claim; preserve the security-label nuance above.

## C2 — REFUTE

**Primary URLs fetched**

- CVPR Findings paper: https://openaccess.thecvf.com/content/CVPR2026F/html/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.html and https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf
- Official repository/API/tree/releases: https://github.com/Yoonwoo-Ha/RACE-6D, https://api.github.com/repos/Yoonwoo-Ha/RACE-6D/git/trees/main?recursive=1, https://api.github.com/repos/Yoonwoo-Ha/RACE-6D/releases

**Short evidence quotes**

- Paper Table 3: “RACE-6D ... 76.7 ... 84.0”; comparison row: “CRT-6D ... 16.6.”
- Abstract: “predicts detection and pose for all instances jointly.”

**Assessment**

- The RT-DETR-derived, single-pass joint detection/pose description, known BOP-object setting, RGB AR 76.7, and YCB-V mean rotation errors 10.3° RGB / 8.0° RGB-D are confirmed.
- **The speed attribution is false.** At batch 1, 640×480, RTX 3090, the paper assigns **84.0 FPS** to RACE-6D RGB and **16.6 FPS** to CRT-6D. RACE-6D RGB-D is 83.3 FPS.
- The live official repository is Apache-2.0 and contains code/configs but no `.pth`, `.pt`, or `.ckpt` in its recursive tree; releases and tags are empty on 2026-07-16. This conflicts with the paper's generic “code and models are available” sentence: no released checkpoint was found in the official repo.

**Corrected fact:** 76.7 BOP AR at 84.0 FPS, not 16.6 FPS. These are YCB-V/BOP results, not pickleball evidence.

## C3 — CONFIRM

**Primary URLs fetched**

- Paper: https://arxiv.org/abs/2605.01234 and https://arxiv.org/pdf/2605.01234
- Bounded official-host searches: https://api.github.com/search/repositories?q=%22TT4D%22%20table%20tennis, https://huggingface.co/api/models?search=TT4D, https://huggingface.co/api/datasets?search=TT4D

**Short evidence quote**

- Paper: “92 strokes”; orientation “26.4 ± 4.4°”; velocity “0.58 ± 0.40 m/s.”

**Assessment**

- TT4D infers racket stroke parameters through an inverse-control reconstruction driven by observed ball trajectories. Its mocap comparison uses IR racket markers and 92 strokes; both reported error values match the claim.
- The paper says the authors “will release” the dataset / intend publication upon acceptance. Exact-title GitHub, HF model, and HF dataset searches returned zero on 2026-07-16. No licensed public artifact was found.

**Correction:** the no-release statement is a dated, bounded search result, not a proof that no private or differently named artifact exists.

## C4 — CONFIRM

**Primary URLs fetched**

- Tennis event-camera paper: https://arxiv.org/abs/2506.08327 and https://arxiv.org/pdf/2506.08327
- Badminton dual-event-camera paper: https://arxiv.org/abs/2605.28011 and https://arxiv.org/pdf/2605.28011

**Short evidence quotes**

- Tennis: “24 out of 26” without direct sunlight and “3 out of 20” in direct sunlight.
- Badminton: “116 out of 124 trials (93.5%).”

**Assessment**

- For the tennis method, every successfully contour-recovered case had absolute impact-position difference below 15 mm. The sharp sunlight collapse, 24/26 to 3/20, is confirmed.
- The badminton study began with 125 trials, excluded one unusable trial, and localized 116/124. Biases are 1.84 ms, +3.45 mm medio-lateral, and −1.92 mm longitudinal.

**Correction:** “within ~±10 mm” is a fair shorthand but not exact. The 95% limits are asymmetric: −3.35 to +10.24 mm and −10.63 to +6.78 mm; timing is −0.84 to +4.52 ms.

## C5 — CONFIRM

**Primary URLs fetched**

- Self6D: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460103.pdf
- DOPE: https://proceedings.mlr.press/v87/tremblay18a/tremblay18a.pdf
- MegaPose: https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf
- ROCK project/paper: https://zhongcl-thu.github.io/rock/ and https://arxiv.org/abs/2202.00448

**Short evidence quotes**

- Self6D tables: “40.1 ... 58.9 ... 86.9” and “15.1 ... 32.1 ... 70.2.”
- MegaPose table heading: “Refiner from initial poses”; RGB result “88.6.”
- ROCK table: “Ours ... 59.4.”

**Assessment**

- Self6D's LineMOD and Occluded-LineMOD synthetic lower bound, unlabeled-real self-supervised result, and real-label upper bound match all six claimed numbers.
- DOPE's sugar-box ADD AUC is 66.64 DR-only, 62.94 photoreal-only, and 77.00 mixed synthetic.
- MegaPose's 88.6% ModelNet 5°/5 cm RGB result is explicitly refiner-from-initial-pose performance, not end-to-end detection. Its seven-dataset BOP RGB mean AR is 54.5.
- ROCK is trained synthetically and reports 59.4 average recall, above the listed Self6D variants.

**Correction:** ROCK's 59.4 is on the paper's **five-object YCB-Video subset**, not the full 21-object YCB-V set. None of these benchmarks establish pickleball performance.

## C6 — CONFIRM

**Primary URLs fetched**

- Garon et al.: https://arxiv.org/pdf/1803.10075
- PhoCaL: https://arxiv.org/abs/2205.08811 and https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf
- Imitrob: https://arxiv.org/pdf/2209.07976
- Anipose: https://pmc.ncbi.nlm.nih.gov/articles/PMC7423398/

**Short evidence quotes**

- Garon: “eight Vicon MX-T40 cameras” and “3 mm retro-reflective markers.”
- PhoCaL: ICP simulation reports “0.20 mm” and “0.38°.”
- Imitrob: “848×480 ... at 60 Hz” and “6D poses at 30 Hz.”
- Anipose: “over 90% ... less than 1°.”

**Assessment**

- All hardware, rate, image-count (~184,000), tracker, and license facts are confirmed. Imitrob's datasheet applies CC BY-NC-SA 4.0 to code and dataset.
- The Anipose six-camera ChArUco experiment reports over 90% of board-pose estimates below 1° angular error.

**Corrections/limits:** Garon's 0.15 mm static / 2 mm moving values are cited Vicon-system capability, not an empirical end-to-end error measured by that dataset paper. PhoCaL's 0.20 mm / 0.38° are ICP-refinement RMSE in a controlled simulation, not total robot/hand-eye/capture GT uncertainty. Anipose's result validates reconstructed board geometry/pose under its setup, not arbitrary high-speed tool tracking.

## C7 — REFUTE

**Primary URLs fetched**

- BlenderProc repository/license/example: https://api.github.com/repos/DLR-RM/BlenderProc, https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/LICENSE, https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/advanced/motion_blur_rolling_shutter/main.py
- Kubric repository/license/commits: https://api.github.com/repos/google-research/kubric, https://raw.githubusercontent.com/google-research/kubric/main/LICENSE, https://api.github.com/repos/google-research/kubric/commits?per_page=5
- Isaac Sim tutorial/licensing: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/replicator_tutorials/tutorial_replicator_object_based_sdg.html, https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses-isaac-sim.html

**Short evidence quotes**

- BlenderProc example: `enable_motion_blur(... rolling_shutter_type="TOP" ...)`.
- Isaac tutorial: “combining the number of pathtraced subframes samples.”
- Current license page: “Github repository is covered under the Apache 2.0 License.”

**Assessment**

- BlenderProc is GPL-3.0 and the exact live official example enables both motion blur and top-to-bottom rolling shutter.
- The Isaac object-based SDG tutorial exists and sets `/omni/replicator/pathTracedMotionBlurSubSamples`.
- **Kubric is Apache-2.0 but is not archived and does not currently show the claimed weak-maintenance signal:** GitHub reports `archived=false`; commits landed 2026-05-12 and 2026-05-21, including Blender fixes.
- **The blanket “under NVIDIA Omniverse EULA” characterization is stale/overbroad.** Current NVIDIA docs say Isaac Sim's GitHub source is Apache-2.0; required Kit/assets use separate NVIDIA Additional Software and Materials terms.

**Corrected fact:** BlenderProc and the Isaac tutorial are valid candidates. Kubric is live, and Isaac Sim has a split-license stack rather than a single blanket Omniverse-EULA label.

## C8 — PARTIAL

**Primary URLs fetched**

- FoundationPose: https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE.md and https://raw.githubusercontent.com/NVlabs/FoundationPose/main/readme.md
- FoundPose: https://raw.githubusercontent.com/facebookresearch/foundpose/main/LICENSE and https://github.com/facebookresearch/foundpose
- GigaPose: https://raw.githubusercontent.com/nv-nguyen/gigapose/main/LICENSE, https://huggingface.co/datasets/nv-nguyen/gigaPose/blob/main/gigaPose_v1.ckpt, https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.html
- KV-Tracker: https://api.github.com/repos/Marwan99/kv_tracker, https://raw.githubusercontent.com/Marwan99/kv_tracker/main/README.md, https://raw.githubusercontent.com/Marwan99/kv_tracker/main/LICENSE
- GRAB / GraspXL: https://raw.githubusercontent.com/otaheri/GRAB/master/LICENSE, https://huggingface.co/api/datasets/ethHuiZhang/GraspXL?blobs=true
- SAM-6D: https://api.github.com/repos/JiehongLin/SAM-6D, https://api.github.com/repos/JiehongLin/SAM-6D/git/trees/main?recursive=1, https://raw.githubusercontent.com/JiehongLin/SAM-6D/main/SAM-6D/Instance_Segmentation_Model/LICENSE
- IPPE: https://api.github.com/repos/tobycollins/IPPE and https://raw.githubusercontent.com/tobycollins/IPPE/master/README.md

**Short evidence quotes**

- FoundationPose license: “research or evaluation purposes only.”
- FoundPose README: “coarse pose estimation pipeline without the featuremetric refinement stage.”
- KV-Tracker license: “non-commercial internal use or academic research.”
- IPPE README: “always returns two candidate pose solutions.”

**Assessment**

- (a) Confirmed: FoundationPose's NVIDIA license is non-commercial; its official demo/evaluation path consumes RGB, depth, and intrinsics and is RGB-D.
- (b) Confirmed: FoundPose is CC BY-NC 4.0 and its release expressly omits featuremetric refinement.
- (c) Confirmed: GigaPose code is MIT; the live HF pointer is 3.81 GB; the paper identifies failure when a visible segment is too small (with low-fidelity CAD also implicated).
- (d) Confirmed: KV-Tracker is a live CVPR 2026 monocular-RGB online tracker. The paper says “up to 30 FPS”; its repository uses a custom Imperial College non-commercial license.
- (e) Confirmed: GRAB uses a custom non-commercial research license. GraspXL's card is CC BY-NC 4.0 (the live bundle totals ~376.7 GB, so it was not downloaded).
- (f) **Literal claim refuted:** the SAM-6D tree does contain `SAM-6D/Instance_Segmentation_Model/LICENSE` (MIT). However, GitHub reports `license: null` and there is no root/project-wide LICENSE, so SAM-6D itself still lacks a clear top-level license grant.
- (g) Confirmed: IPPE is BSD-3-Clause and returns two planar-pose candidates (then ranks them by reprojection error).

**Corrected fact:** say “SAM-6D has no root/project-wide license; one nested component has an MIT LICENSE,” not “no LICENSE file.”

## C9 — PARTIAL

**Primary URLs fetched**

- BlurHandNet paper/release: https://openaccess.thecvf.com/content/CVPR2023/html/Oh_Recovering_3D_Hand_Mesh_Sequence_From_a_Single_Blurry_Image_CVPR_2023_paper.html, https://api.github.com/repos/JaehaKim97/BlurHand_RELEASE, https://raw.githubusercontent.com/JaehaKim97/BlurHand_RELEASE/master/README.md
- Human from Blur: https://arxiv.org/abs/2303.17209 and https://arxiv.org/pdf/2303.17209
- Bounded release searches and closest generic counterexample: https://api.github.com/search/repositories?q=%22racket%22+%22motion+blur%22+%226D+pose%22, https://huggingface.co/api/models?search=racket%20motion%20blur%20pose, https://api.github.com/repos/rozumden/ShapeFromBlur

**Short evidence quotes**

- BlurHandNet: “unfolds a blurry input image to a 3D hand mesh sequence.”
- Human from Blur: “Using a differentiable renderer” and “sub-frame accuracy.”
- ShapeFromBlur release: “Recovering Textured 3D Shape and Motion of Fast Moving Objects.”

**Assessment**

- Both named methods exist as described. BlurHandNet maps one blurry image to three sequential hand meshes and materially outperforms sharp-trained hand baselines on blurry evaluation input. Human from Blur optimizes a pose sequence through differentiable rendering and temporal image aggregation.
- The sports-equipment absence claim cannot receive a global confirmation. Exact GitHub/HF searches found no racket-specific released 6DoF-from-blur package, but a released MIT **generic rigid-object** package, ShapeFromBlur, does recover textured 3D shape and sub-frame motion from a blurred image/background. Other released blur packages also address fast balls or object appearance, not racket 6DoF.

**Corrected fact:** “No sports-racket/equipment-specific released equivalent was found in the bounded search” is supportable. “No equivalent rigid-object package exists” is false; universal nonexistence remains unresolved.

## C10 — CONFIRM

**Primary URL:** not applicable; this claim is checked by dimensional analysis and a first-order uncertainty derivation.

**Calculation/evidence**

- Half a 60 FPS frame is `0.5 / 60 = 0.008333 s = 8.333 ms`.
- `10–20 m/s × 0.008333 s = 0.0833–0.1667 m = 8.33–16.67 cm`.
- A quarter of a 240 FPS frame is `0.25 / 240 = 1.0417 ms`; at 10–20 m/s that is 1.04–2.08 cm.
- For endpoints `p1,p2` with independent, zero-mean, isotropic coordinate errors of standard deviation `σ`, the direction-vector error is `e2−e1`, whose perpendicular variance is `2σ²`. Linearizing `atan2` gives `sd(θ) ≈ √2 σ/L` radians.
- `σ=5 px`: `L=40 px → 0.1768 rad = 10.13°`; `L=80 px → 0.0884 rad = 5.06°`.

**Correction/limits:** ≤1 ms is a sensible conservative target, not the unique mathematical threshold. A motion-only 3 cm budget permits 3.0 ms at 10 m/s or 1.5 ms at 20 m/s; calibration, rolling shutter, interpolation, and labeling consume part of that budget. The angle formula assumes `σ ≪ L`, per-coordinate σ, independent endpoints, and no correlated/systematic error. If σ denotes radial RMS or endpoint errors are correlated, the coefficient changes.
