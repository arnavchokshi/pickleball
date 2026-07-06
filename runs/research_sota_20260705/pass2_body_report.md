# PASS 2 — Temporally-stable world-grounded 3D human mesh from monocular sports video

2026-07-06. 43 agents.


# Pass-2 synthesis: temporally-stable world-grounded 3D human mesh (new-vs-pass-1 only)

Pass 1 already covers GVHMR, WHAM, TRAM, PromptHMR/-Vid, CoMotion, RAM, OnlineHMR, SMART, SAM-3D-Body, Fast-SAM-3D-Body, SAM-Body4D, MHR, SmoothNet, DPoser/-X, PhysPT, PhysDiff, IPMAN, MultiPly, 4DHumans, SLAHMR, HuMoR, SKEL. Everything below is new. Organized by pipeline sub-problem, all 2024-2026 unless noted.

## 1. Post-hoc stabilization built directly on our current backbone

The single most on-target hit of this pass is **World-Coordinate Human Motion Retargeting via SAM 3D Body** (arXiv 2512.21573, submitted 25 Dec 2025, cs.RO, Tu/Zhu/Su/Zheng). It wraps our *exact* production backbone — frozen SAM-3D-Body + MHR — with three lightweight mechanisms: (1) per-subject identity/skeleton-scale locking for consistent bone length across frames, (2) sliding-window optimization in the low-dimensional MHR latent space (not raw joint/vertex space) for temporal smoothing, and (3) a differentiable soft foot-ground contact model plus contact-aware global trajectory optimization for physically-plausible root motion, then retargets to a Unitree G1 humanoid via two-stage IK ([arXiv 2512.21573](https://arxiv.org/abs/2512.21573)). [VERIFIED partially]: adversarial review confirms all three mechanisms are real and match the abstract, but flags two corrections — this is a robotics-retargeting paper with only qualitative (single-figure) evidence, no jitter/foot-slide numbers, and zero independent citations at ~2 weeks old. It's a recipe to prototype against our own floor-sink/root-jitter defects, not a proven fix.

**SOMA-X** (NVlabs, v0.2.1, 2026-06-05, tech report [arXiv 2603.16858](https://arxiv.org/abs/2603.16858)) is the infrastructure piece that makes such recipes portable: an Apache-2.0, GPU-accelerated (NVIDIA Warp) canonical body-topology "pivot" mapping SMPL/SMPL-X/MHR onto one shared representation, with MHR as its default identity model and a PyPI package (`py-soma-x`). Since SAM-3D-Body emits MHR while most academic smoothing/physics priors (SmoothNet, DPoser-X — both already known) expect SMPL-X, this closes an interop gap we would otherwise patch by hand.

**DuoMo** ([arXiv 2603.03265](https://arxiv.org/pdf/2603.03265), CVPR 2026) factorizes world-grounded recovery into a camera-space diffusion model plus a world-space diffusion model that lifts and refines it, generating mesh vertex motion directly (bypassing parametric regression). It reports **16% world-space error reduction on EMDB and 30% on RICH** versus prior world-grounded HMR baselines, with low foot-skating — the largest margin found this pass — but no public code/checkpoints exist yet (project page: yufu-wang.github.io/duomo). Monitor for release.

## 2. Unified feedforward camera + scene + human (potential stage-collapse)

**Human3R** ("Everyone Everywhere All at Once," [arXiv 2510.06219](https://arxiv.org/abs/2510.06219), ICLR 2026) is built on CUT3R via parameter-efficient visual prompt tuning, trained on BEDLAM for **one day on one GPU**, and jointly recovers global multi-person SMPL-X bodies, dense 3D scene geometry, and camera trajectory in a **single forward pass** at **15 FPS with 8GB memory** [VERIFIED — supported by two independent fetches of the abstract]. This is architecturally the biggest potential simplification surfaced across both passes: it could collapse separate mesh-estimation + camera/court-solve + smoothing stages into one model, and its real-time throughput opens a path toward live coaching feedback. It needs a hands-on bake-off against SAM-3D-Body + classical smoothing before any commitment — no sports-domain or elevated/far-camera validation exists yet.

Two more feedforward VGGT-style siblings appeared but estimate camera pose rather than consume it (a mismatch for our ARKit-fed pipeline): **UniCon3R** ([arXiv 2604.19923](https://arxiv.org/abs/2604.19923)) adds a contact-as-corrective-signal against floating/penetration; **SHOW** ([arXiv 2606.27720](https://arxiv.org/html/2606.27720v1), June 2026, UPenn) adapts VGGT with a mask encoder, DensePose head, and geometry-aware SMPL-X decoder for joint metric-scale human+scene reconstruction. Both are unconfirmed for code release — watch-items, not near-term candidates.

## 3. Occlusion-robust temporal consistency

**MoPO** ([arXiv 2605.09856](https://arxiv.org/abs/2605.09856)) is a two-stage plug-in explicitly framed around "severe motion jitter due to insufficient spatial features for occluded body parts": a spatial-temporal visibility detector + lightweight motion predictor completes occluded joints from pose history, then a motion-aware fusion/IK-refinement stage combines the completion with image features. This targets occlusion-driven jitter specifically (net crossings, paddle swings, player-player overlap) rather than generic smoothing — a gap our classical smoothing doesn't close. Authors claim code/demo in supplementary material (unconfirmed).

**HMRMamba** ([arXiv 2601.21376](https://arxiv.org/abs/2601.21376), Jan 2026) is the first application of structured state-space models (Mamba) to video HMR: a dual-scan "Geometry-Aware Lifting Module" plus a "Motion-Guided Reconstruction Network," reporting MPJPE/PA-MPJPE/MPVPE/Accel of **64.8/45.5/79.8mm/6.5** on 3DPW (vs. ARTS prior-SOTA 67.7mm MPJPE), 68.3/50.2mm on MPI-INF-3DHP, and 49.3/35.7mm on Human3.6M, at only 7.88-9.32 GFLOPs — but **no code released** ("code will be released soon" as of fetch).

**Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning** ([arXiv 2508.21363](https://arxiv.org/abs/2508.21363), Aug 2025) shows diffusion-style temporal denoising for pose sequences can now be made cheap: 38.5% less training MACs, 56.8% less inference MACs, 81.1% faster inference vs. prior diffusion pose methods, SOTA on H36M/MPI-INF-3DHP — relevant if we ever want a genuine learned diffusion motion prior instead of classical smoothing.

## 4. Perspective / close-range geometry correction (single iPhone specific)

Our capture geometry — a single consumer iPhone often close to a small court — creates per-frame scale/shape jitter from perspective distortion that generic HMR training data under-represents. **BLADE** ([github.com/NVlabs/blade](https://github.com/NVlabs/blade), CVPR 2025) predicts pelvis depth first, conditions pose/shape on that depth, then recovers a full perspective camera via differentiable rasterization — designed for exactly this close-range regime, with full code released. **PersPose** ([arXiv 2508.17239](https://arxiv.org/abs/2508.17239), ICCV 2025) encodes camera intrinsics directly (Perspective Encoding) and applies a Perspective Rotation to normalize crop-to-scene relationships, reporting **60.1mm MPJPE on 3DPW (~7.5% better than prior SOTA)**, code+weights released. **KASportsFormer** ([arXiv 2507.20763](https://arxiv.org/abs/2507.20763), MMSports 2025) adds a Bone Extractor + Limb Fuser encoding anatomical bone-length constraints for short sports-scene video, reporting **58.0mm MPJPE on SportsPose and 34.3mm on WorldPose** (both claimed SOTA), code released.

Note a naming collision: **SMART: SMPLest-X Mesh Adaptation and RAFT Tracking for Soccer** ([arXiv 2605.31551](https://arxiv.org/html/2605.31551)) is a *different, newer* paper from the already-known "SMART" — it fine-tunes SMPLest-X (ViT-H, 687M params) with a multi-task 3D MPJPE + 2D reprojection + pelvis-depth loss, RAFT-small optical flow for camera/background motion (vs. Lucas-Kanade/ECC baselines), MAD outlier rejection, foot-plane anchoring, and two-pass (global/local) temporal smoothing. Exact benchmark numbers were not extractable from the fetched source; internal docs should disambiguate the two "SMART"s before citing either.

## 5. Scarce in-domain data (our binding constraint)

Two Stanford papers (Weng et al.) offer a concrete but non-unified recipe. **DAPA** ([arXiv 2206.10457](https://arxiv.org/pdf/2206.10457)) fine-tunes HMR using only 2D keypoints from a target dataset plus synthetic pose-augmented mesh pairs, reporting competitive results vs. 3D-annotation fine-tuning (MPJPE 168.3 DAPA vs. 153.4 3D-FT vs. 175.1 pretrained on AGORA/3DPW) — but [VERIFIED partially]: this was **never validated on sports poses**, only a qualitative gymnastics demo. **Diffusion-HPC** ([arXiv 2303.09541](https://arxiv.org/pdf/2303.09541)) generates photorealistic synthetic images with full synthetic 3D supervision via depth-conditioned diffusion and *is* validated on sports domains: **Ski-Pose MPJPE 225.1mm (pretrained) → 111.3mm (Diffusion-HPC)**. Independently, Diffusion-HPC's own benchmarking found DAPA underperforms plain 2D-only fine-tuning on real sports datasets — so the "2D keypoints only" recipe and the "sports-validated" recipe are two separate papers, not one combined result.

On data sources: **RacketVision** ([arXiv 2511.17045](https://arxiv.org/pdf/2511.17045), AAAI 2026 oral) is the first large multi-sport racket dataset (table tennis/tennis/badminton; 435,179 frames, 24,621 racket annotations, ~5.6x TrackNetV2's 78k frames) but [VERIFIED partially]: racket "pose" is **2D-only** (bbox + 5 pixel keypoints, no depth/calibration) and has **zero pickleball data** — useful at best as a 2D pretraining/eval signal, not a 6-DOF ground-truth source. **PadelTracker100** ([Zenodo](https://zenodo.org/records/14653706), companion paper in *Data in Brief*, Feb 2026 — [VERIFIED partially]: **not** an arXiv paper as originally logged) is the closest racket-sport analogue to our capture setup: ~100,000 frames from 2 padel matches, ball trajectory, ViTPose-L pose, 6-class shot events. **CalTennis** ([arXiv 2606.20542](https://arxiv.org/abs/2606.20542)) offers a reusable label-free validation *methodology*: 11M+ frames, 2-6 synced cameras, cross-checking monocular 3D output against multi-view triangulation without manual MoCap — a cheap pattern if we ever add a second phone for spot-checking.

## 6. Multi-player occlusion-ID benchmarks and rendering/avatar layer

**TrackID3x3** ([arXiv 2503.18282](https://arxiv.org/abs/2503.18282)) introduces TI-HOTA, a HOTA variant fusing spatial + team/jersey-attribute matching for basketball (Indoor TI-HOTA 80.75±13.16, Outdoor 46.11±20.55). Not directly reusable (different sport/data) but its identity-swap-aware metric design is worth borrowing if pickleball player-ID bugs surface. On the rendering side — lower priority, viewer-facing only — **STG-Avatar** ([arXiv 2510.22140](https://arxiv.org/abs/2510.22140), IROS 2025) couples LBS with Spacetime Gaussians whose density increases in fast-moving regions (paddle swings, footwork), and **FastHMR** ([arXiv 2510.10868](https://arxiv.org/pdf/2510.10868)) reports up to 2.3x HMR speedup via token/layer merging plus a temporal-context diffusion decoder — both spike-tier, unproven on our stack.

## Constraint frictions

None of the world-grounding candidates above (Human3R, UniCon3R, SHOW, DuoMo, GVHMR-class methods from pass 1) have been benchmarked on **quasi-static handheld phone motion** — EMDB/RICH/3DPW all assume walking/running camera egomotion, not our near-static tripod-like framing. Several strong leads (DuoMo, HMRMamba, WATCH, VIMCAN) have **no public code**, only papers — real integration risk. SAM License gating (already known) and MPI's SMPL-X non-commercial license are moot per project's private-use ruling but complicate any code reuse that bundles those checkpoints.


## Missed in pass 1

- **World-Coordinate Human Motion Retargeting via SAM 3D Body** — A Dec 2025 robotics paper (arXiv 2512.21573) that fine-tunes nothing but wraps our exact production backbone (frozen SAM-3D-Body + MHR) with identity/skeleton-scale locking, MHR-latent-space sliding-window smoothing, and a differentiable soft foot-ground contact model + contact-aware global trajectory optimization. _Why:_ This is the single most on-target hit in either research pass for our specific 'floor-sink / root-jitter on top of SAM-3D-Body' bugs (logged 2026-07-05 visual-polish and joint-visual-placement lanes) — it is not a new backbone to evaluate, it is a lightweight recipe to bolt onto the backbone we already ship. A pass-1 search for '[our exact model name] + temporal smoothing' should have surfaced this; instead it took a second, deeper pass. Caveat: paper's own evidence is qualitative-only (no jitter/foot-slide numbers) and its validated task is humanoid retargeting, not mesh cleanup, so it needs a hands-on port, not blind trust. https://arxiv.org/abs/2512.21573
- **SOMA-X** — NVlabs' canonical body-topology pivot (Apache-2.0, GPU-accelerated via NVIDIA Warp, MHR as default identity model, PyPI package) that provides an officially maintained conversion layer between SMPL, SMPL-X, and MHR. _Why:_ Our stack sits exactly at the seam this tool patches: SAM-3D-Body emits MHR, but SmoothNet/DPoser-X-class priors and most academic tooling (already known from pass 1) expect SMPL-X. Without this we'd hand-roll shape-space conversion; pass 1 should have flagged the MHR<->SMPL-X interop gap as an open risk even without knowing this specific tool existed. https://github.com/NVlabs/SOMA-X
- **Human3R** — ICLR 2026 unified feed-forward model (built on CUT3R) that jointly outputs multi-person world-frame SMPL-X, dense scene geometry, and camera trajectory from monocular video in one pass at 15 FPS / 8GB, trained in a single GPU-day on BEDLAM. _Why:_ This is architecturally the biggest potential simplification found across both passes — it could collapse our separate mesh-estimation + camera/court-solve + smoothing stages into one model — yet it did not surface in pass 1 despite matching the DOMAIN description almost verbatim ('temporally-stable world-grounded 3D human mesh from monocular video'). It needs a hands-on eval against SAM-3D-Body before any commitment, but the omission itself is the miss. https://arxiv.org/abs/2510.06219

## New adoptions

### [NOW] World-Coordinate Human Motion Retargeting via SAM 3D Body (2512.21573 recipe) → Player-mesh temporal-stabilization stage (post SAM-3D-Body per-frame, replacing/augmenting current classical smoothing) — directly targets VP-C nose-root floor-sink and root-jitter defects
- **what:** Identity/skeleton-scale locking + MHR-latent-space sliding-window smoothing + differentiable soft foot-ground contact model, applied directly on top of our existing frozen SAM-3D-Body output.
- **evidence:** [VERIFIED partially] arXiv 2512.21573, 25 Dec 2025 — abstract confirms all three mechanisms; robotics-application framing and lack of quantitative smoothing metrics confirmed by adversarial check
- **expected_gain:** Unquantified by the source paper — needs our own before/after jitter + foot-slide measurement, but mechanism directly matches our logged failure class
- **confidence:** medium (idea directly applicable; paper's own evidence is qualitative-only, no jitter/foot-slide numbers reported)
- **url:** https://arxiv.org/abs/2512.21573

### [NOW] SOMA-X (MHR<->SMPL-X conversion pivot) → Mesh-format interop layer between SAM-3D-Body's native MHR output and any SMPL-X-based tooling (smoothing priors, retargeting, coaching-viz asset pipeline)
- **what:** Apache-2.0 GPU-accelerated canonical body-topology converter (NVIDIA Warp + PyTorch, PyPI py-soma-x) with MHR as the default identity model.
- **evidence:** v0.2.1 released 2026-06-05, tech report arXiv 2603.16858
- **expected_gain:** Removes need for a bespoke MHR->SMPL-X converter; risk-reduction rather than accuracy gain
- **confidence:** medium-high (maintained, versioned release; not independently load-tested by us yet)
- **url:** https://github.com/NVlabs/SOMA-X

### [SPIKE] Human3R → Candidate full or partial replacement of the multi-stage player-mesh + camera/court-grounding pipeline (would need side-by-side eval against SAM-3D-Body + classical smoothing on owner clips)
- **what:** Single feed-forward pass jointly producing multi-person world-frame SMPL-X + scene geometry + camera trajectory, 15 FPS/8GB, 1-GPU-day BEDLAM fine-tune.
- **evidence:** [VERIFIED supported,supported] arXiv 2510.06219, ICLR 2026 accepted
- **expected_gain:** Potentially collapses 2-3 pipeline stages into one pass; real-time-capable (15 FPS) opens a path to near-live coaching feedback
- **confidence:** medium (numbers verified from abstract; no sports-domain or pickleball-scale validation exists yet)
- **url:** https://arxiv.org/abs/2510.06219

### [SOON] DuoMo → Second-pass world-space stabilizer to run after SAM-3D-Body per-frame mesh, as an alternative/upgrade to classical smoothing
- **what:** Camera-space diffusion + world-space diffusion refinement operating on mesh vertices directly, bypassing parametric regression for the world-frame correction step.
- **evidence:** 16% world-space error reduction on EMDB, 30% on RICH vs prior world-grounded HMR baselines
- **expected_gain:** Largest reported world-space error reduction of any candidate found this pass, if code lands and reproduces
- **confidence:** low (no public code/weights found; numbers are the paper's own, not independently reproduced)
- **url:** https://arxiv.org/pdf/2603.03265

### [SOON] MoPO (occlusion de-occlusion motion prior) → Occlusion-specific pre-processing before/inside our classical smoothing stage, targeting jitter from paddle/net/player self-occlusion specifically (vs generic temporal smoothing)
- **what:** Spatial-temporal visibility detector + motion predictor that completes occluded joints from pose history, then a motion-aware fusion + IK refinement stage.
- **evidence:** Framed explicitly around reducing 'severe motion jitter due to insufficient spatial features for occluded body parts'
- **expected_gain:** Unquantified in fetched abstract; targets exactly our known occlusion-jitter failure mode
- **confidence:** low-medium (authors claim code/demo in supplementary material; not independently confirmed runnable)
- **url:** https://arxiv.org/abs/2605.09856

### [SOON] BLADE → Per-frame mesh scale/placement stabilization stage, upstream of smoothing, specifically for close-range iPhone framing where perspective distortion varies player-to-player
- **what:** Single-image SMPL(-X) recovery that predicts pelvis depth first, conditions pose/shape on that depth, then recovers full perspective camera via differentiable rasterization — built for close-range perspective distortion.
- **evidence:** NVLabs repo + CVPR 2025 paper; targets exactly our 'iPhone close to court' geometry regime
- **expected_gain:** Not quantified for our domain; addresses a distortion source our current pipeline doesn't explicitly model
- **confidence:** medium (CVPR 2025, code released, but not tested on sports/pickleball footage)
- **url:** https://github.com/NVlabs/blade

### [SPIKE] DAPA + Diffusion-HPC domain-adaptation recipe → Fine-tuning strategy for whatever mesh backbone we adopt, to address the project's core 'scarce in-domain owner-captured data' constraint
- **what:** Two complementary fine-tuning recipes for closing the domain gap with scarce real target-domain data: DAPA uses only 2D keypoints + synthetic pose-augmented mesh pairs; Diffusion-HPC uses a depth-conditioned diffusion model to generate full synthetic 3D-supervised images, validated on sports (Ski-Pose MPJPE 225.1mm->111.3mm).
- **evidence:** [VERIFIED partially both] DAPA arXiv 2206.10457; Diffusion-HPC arXiv 2303.09541, Ski-Pose 225.1mm->111.3mm MPJPE
- **expected_gain:** Concrete lever for scarce-data constraint but requires building a synthetic-generation pipeline, not a drop-in tool
- **confidence:** medium (both numbers verified from primary PDFs; DAPA specifically NOT validated on sports poses, so the combined recipe is analyst-assembled, not one paper's result)
- **url:** https://arxiv.org/pdf/2303.09541
