# TRK research survey B — finding and tracking four pickleball players

**Survey date:** [FETCHED-PRIMARY] 2026-07-16  
**Lane:** [FETCHED-PRIMARY] independent B; no sibling research directory was read  
**Scope:** [FETCHED-PRIMARY] literature/artifact/license survey and benchmark specification only; no GPU dispatch  
**Promotion state:** [FETCHED-PRIMARY] `VERIFIED=0`

## Decision summary

- [FETCHED-PRIMARY] The repository gate and current scored artifacts make coverage—not identity association—the blocking metric: the worst post-margin/OSNet clip has `cov4=0.7117`, zero switches, and the fresh-clip gate requires coverage at least 0.95 with zero switches and zero off-court-person false positives. [NORTH_STAR_ROADMAP.md](../../../NORTH_STAR_ROADMAP.md)
- [INFERENCE] Therefore the benchmark order should remain detector zero-shot -> detector domain fine-tune -> ReID enrollment -> mask cue. The 2025–2026 sports-MOT leaders are useful decompositions, but most of their measured edge is association, global tracklet linking, or mask-memory—not recall of people the detector never emits.
- [FETCHED-PRIMARY] The strongest current SportsMOT paper claim found is SAM 3-Deep-EIoU + GTA at 87.2 HOTA, but its validation ablation assigns only +1.5 HOTA to selective masks and +7.8 HOTA to GTA over Deep-EIoU; the detector is held fixed. [Selective Mask Propagation v3](https://arxiv.org/html/2606.13033)
- [INFERENCE] RF-DETR-L detection (`rf-detr-large-2026.pth`) is the first zero-shot challenger; RF-DETR-L/XL segmentation is the first mask-producing challenger. D-FINE-L and DEIMv2-S/M are controls, not earlier bets, because no primary source demonstrates superior partial-player recall under net/player occlusion.
- [FETCHED-PRIMARY] RF-DETR's current release surface is substantially more mature than a preview: v1.7.1 fixes official-checkpoint loading and BF16 segmentation training, and the current package exposes N/S/M/L detectors plus N/S/M/L/XL/2XL segmenters. [release 1.7.1](https://github.com/roboflow/rf-detr/releases/tag/1.7.1), [current README](https://github.com/roboflow/rf-detr)
- [FETCHED-PRIMARY] Its licensing is now explicit: open-source package and Apache-designated weights are Apache-2.0; detection XL/2XL and `rfdetr_plus` are PML-1.0. All listed segmentation sizes are Apache-designated. [RF-DETR license section](https://github.com/roboflow/rf-detr#license)
- [INFERENCE] No surveyed public tracker is purpose-built for exactly four enrolled court players while treating all other people as distractors. SportsMOT explicitly removes spectators, referees, and coaches from the target set, so it tests a related but easier open-set-rejection boundary.

## Changed since 2026-07-09

- [FETCHED-PRIMARY] **Selective Mask Propagation v3 (2026-07-09)** now reports 87.2 SportsMOT HOTA, a selective assignment-margin dispatch policy, a gap trigger, and 0.105–0.114 SAM seconds per processed frame on RTX 5090. No code or weight release is linked from the paper. [paper](https://arxiv.org/html/2606.13033)
- [FETCHED-PRIMARY] **SAMIDARE (CVPRW 2026)** released code for density-aware mask regeneration, selective memory update, and state-aware object addition; its repo has no LICENSE file. [paper](https://arxiv.org/html/2604.22162), [repo](https://github.com/ZabuZabuZabu/SAMIDARE)
- [FETCHED-PRIMARY] **SAM-MT (2026-07-09)** released inference code and a checkpoint and reports 36.8 FPS for three targets and 36.5 FPS for five targets at 1024p on an RTX A6000. The repo has no LICENSE and the model card is CC-BY-NC-SA-4.0. [paper](https://arxiv.org/html/2607.08688), [model card](https://huggingface.co/FudanCVL/SAM-MT/blob/main/README.md)
- [FETCHED-PRIMARY] **YOLO26 is a real public Ultralytics family**, not an internal shorthand: `yolo26m.pt` exists, default inference is one-to-one/NMS-free, and the family supports detection and segmentation. Code, models, and docs are AGPL-3.0 or Enterprise-licensed. [official docs](https://docs.ultralytics.com/models/yolo26/), [paper](https://arxiv.org/abs/2606.03748)
- [FETCHED-PRIMARY] **DEIMv2** released in September 2025 with DINOv3-based S/M/L/X checkpoints and a documented custom-data fine-tune path; TensorRT FP16 requires at least 10.6 for correct results. [repo](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [FETCHED-PRIMARY] **SapiensID** now has public code and a single Google Drive pretrained checkpoint, but the whole repository is CC-BY-NC-4.0. [repo](https://github.com/mk-minchul/sapiensid)
- [FETCHED-PRIMARY] **RF-DETR v1.7.1** is current and its core-versus-Plus weight license boundary is explicit; the current default large class points to `rf-detr-large-2026.pth`. [weights registry](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/assets/model_weights.py), [config](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py)

## 1. What wins on sports MOT in 2025–2026

### Ranked systems

| Rank | System | Published evidence and decomposition | Artifact/license posture | DinkVision leverage and risk |
|---|---|---|---|---|
| 1 | Selective Mask Propagation: SAM 3-Deep-EIoU + GTA | [FETCHED-PRIMARY] SportsMOT test: HOTA 87.2, AssA 84.2, IDF1 93.6, MOTA 98.1. Validation: Deep-EIoU 78.9 HOTA; +GTA 86.7; selective masks alone 80.4; masks+GTA 88.5. The edge is predominantly global association, not detector recall. [tables 3–4](https://arxiv.org/html/2606.13033) | [FETCHED-PRIMARY] Paper/pseudocode only; no official code or weights linked. Paper CC-BY-4.0; implementation and SAM 3 weight terms unresolved. **`unknown-needs-review`**. | [INFERENCE] Best new design reference for hard-window dispatch, but low first-order leverage on 0-switch/low-coverage clips. Implement only after detector and ReID gates; do not transplant GTA as an association sweep. |
| 2 | SAM2MOT | [FETCHED-PRIMARY] DanceTrack test: Co-DINO-L detector 75.5 HOTA/83.4 IDF1/80.3 DetA; Grounding-DINO-L 75.8/83.9/79.7. The two-detector comparison shows most of this system's strength is mask propagation/association, while detector choice still changes FN/FP. [repo table](https://github.com/TripleJoy/SAM2MOT) | [FETCHED-PRIMARY] Live code, checkpoints/instructions; Apache-2.0 repo. Detector and upstream weight/data terms remain separate. **`unknown-needs-review`** overall. | [INFERENCE] Open-code reference for persistent masks and missed-box windows, but heavy adoption and more state machinery than the four-target problem needs. |
| 3 | SAMIDARE | [FETCHED-PRIMARY] SportsMOT validation improves SAM2MOT by +2.5 HOTA and +4.2 IDF1 via density-aware regeneration, selective memory update, and state-aware reactivation. [paper](https://arxiv.org/html/2604.22162) | [FETCHED-PRIMARY] Code live, uses SAM2.1-L and YOLOX weights, but no repository LICENSE. **`unknown-needs-review`**. | [INFERENCE] Relevant to corrupted masks under crossings and frame edge exits; lower priority because results are validation-only, code is unlicensed, and much of Stage 3 is re-association. |
| 4 | GTA / GTATrack | [FETCHED-PRIMARY] On SportsMOT, Deep-EIoU 77.21 -> 81.04 HOTA and 79.81 -> 86.51 IDF1 while DetA is unchanged (88.22 -> 88.21). On SoccerNet, the paper reports +5.85% HOTA. [paper](https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/papers/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.pdf) | [FETCHED-PRIMARY] Live MIT code; repo pushed 2025-12-12. Published recipes depend on sports ReID/team/jersey cues and research datasets. **`R&D-only`** as released. | [INFERENCE] Clean proof that a leaderboard jump can be almost entirely association. It is banned as a next experiment here because current worst clips already have zero switches. |
| 5 | SportMamba | [FETCHED-PRIMARY] SportsMOT test: 77.3 HOTA, 77.7 IDF1, 66.8 AssA, 96.9 MOTA, 89.5 DetA versus Deep-EIoU 77.2/79.8/67.7/96.3/88.2. Its 0.1 HOTA edge coincides with higher DetA but worse IDF1/AssA. [paper table 1](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Khanna_SportMamba_Adaptive_Non-Linear_Multi-Object_Tracking_with_State_Space_Models_for_CVPRW_2025_paper.pdf) | [FETCHED-PRIMARY] No code URL in the paper and no official repository found in the live search. License/weights unavailable. **`unknown-needs-review`**. | [INFERENCE] Purpose-built nonlinear sports motion is interesting, but the result does not justify an association experiment for DinkVision. |
| 6 | McByte | [FETCHED-PRIMARY] Fixed detections, training-free mask cue: SportsMOT validation 69.0 -> 83.9 HOTA, test 76.9; DanceTrack validation 47.1 -> 62.3, test 67.1; SoccerNet 2022 test 72.1 -> 85.0. [paper](https://arxiv.org/html/2506.01373) | [FETCHED-PRIMARY] Live MIT code, original SAM ViT-B + Cutie base-mega; published detector weights are SportsMOT-trained. **`R&D-only`** as released because SportsMOT is CC-BY-NC-4.0. | [INFERENCE] Still the quickest mask-cue benchmark after detector/ReID. It cannot prove detection coverage because it holds detections fixed. |
| 7 | MATR / OA-SORT / Gated Temporal Fusion | [FETCHED-PRIMARY] MATR reports 71.3 HOTA on DanceTrack with extra data and 72.2 on SportsMOT; OA-SORT reports 63.1/64.2 HOTA/IDF1 on DanceTrack; Gated Temporal Fusion reports 76.4 SportsMOT HOTA. [MATR](https://arxiv.org/abs/2509.21715), [OA-SORT](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Occlusion-Aware_SORT_Observing_Occlusion_for_Robust_Multi-Object_Tracking_CVPR_2026_paper.html), [GTF](https://openaccess.thecvf.com/content/WACV2026/html/Kim_Gated_Temporal_Fusion_Transformers_for_Robust_Multi-Object_Tracking_WACV_2026_paper.html) | [FETCHED-PRIMARY] No usable official checkpoint/code release was found for these ranked paper results. **`unknown-needs-review`**. | [INFERENCE] Lower numbers and association-centric mechanisms do not beat the detector-first opportunity. |

### Leaderboard cautions

- [FETCHED-PRIMARY] SportsMOT's objective is to track only players on the playground and exclude spectators, referees, and coaches. [official README](https://github.com/MCG-NJU/SportsMOT#sportsmot)
- [INFERENCE] A method can lead SportsMOT without solving DinkVision's explicit spectator/passers-by rejection because excluded people are not enrolled identities and may not be exhaustively represented as scored hard negatives.
- [FETCHED-PRIMARY] SoccerNet's official tracking repo still publishes a 2023 detection+association leaderboard led by Kalisteo at 75.61 HOTA; its 2022 association-only leaderboard uses ground-truth detections and is not comparable to an end-to-end detector gate. [official repo](https://github.com/SoccerNet/sn-tracking)
- [FETCHED-PRIMARY] DanceTrack's official dataset is intentionally uniform-appearance/diverse-motion; recent open-code test results include SAM2MOT at 75.8 HOTA and ColTrack at 75.3 when validation data is added. [SAM2MOT](https://github.com/TripleJoy/SAM2MOT), [ColTrack](https://github.com/bytedance/ColTrack)
- [INFERENCE] These are not directly ordered against DinkVision because detector inputs, training splits, offline/global processing, and target definitions differ.

### Few-target / many-distractor and court-sport fit

- [FETCHED-PRIMARY] Global ID Fusion introduces MuPNIT and supports seen/unseen player IDs in broadcast sports, but the paper provides no public code or checkpoint link. [WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/html/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.html)
- [FETCHED-PRIMARY] FieldMOT registers detections to field coordinates before tracking, but evaluates multi-camera broadcast behavior on synthesized football data; its `FieldTrack` repository has no LICENSE. [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.html), [repo](https://github.com/maomao726/FieldTrack)
- [INFERENCE] No live artifact found combines exactly four enrolled identities, explicit non-player-person rejection, fixed elevated phone geometry, edge truncation, and court-aware track management. This remains an application-specific layer rather than an off-the-shelf tracker choice.

## 2. Occlusion-robust detection ranked for coverage

### [INFERENCE] 1 — RF-DETR-L detection, then RF-DETR-L/XL segmentation

- [FETCHED-PRIMARY] Current detection table: M 54.7 COCO AP at 576 px/4.4 ms T4; L 56.5 at 704 px/6.8 ms. The current L class uses `rf-detr-large-2026.pth`. [README](https://github.com/roboflow/rf-detr), [config](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py)
- [FETCHED-PRIMARY] Current segmentation table: L 47.1 mask AP at 504 px/8.8 ms; XL 48.8 at 624 px/13.5 ms; 2XL 49.9 at 768 px/21.8 ms. All reported latency is TensorRT FP16 batch 1 on T4. [README](https://github.com/roboflow/rf-detr#benchmark-results)
- [FETCHED-PRIMARY] Fine-tuning supports custom COCO-format data, detection and segmentation, checkpoint resume, GPU augmentation for segmentation, LoRA/frozen encoder controls, and explicit negative/background images. v1.7.1 fixed official starter-checkpoint loading. [training docs](https://rfdetr.roboflow.com/develop/learn/train/), [release](https://github.com/roboflow/rf-detr/releases/tag/1.7.1)
- [INFERENCE] Expected leverage: **coverage high**, switches neutral, spectator FP medium risk until domain negatives are trained. L's 704 input is preferable to M for small/truncated players; segmentation should be scored as a detector separately because lower box resolution can lose boxes even when its masks help association.
- [FETCHED-PRIMARY] License: package and Apache-designated weights Apache-2.0; detector XL/2XL are PML-1.0; COCO training imagery has source-specific rights. **Posture: `unknown-needs-review` overall**, despite commercially permissive code/weight terms for L and all segmenters.

### [INFERENCE] 2 — D-FINE-L (Objects365 -> COCO checkpoint)

- [FETCHED-PRIMARY] Official table reports D-FINE-L at 54.0 COCO AP/8.07 ms for COCO-only and 57.3 AP for Objects365+COCO; exact released checkpoint is `dfine_l_obj2coco_e25.pth`. [model zoo](https://github.com/Peterande/D-FINE#model-zoo)
- [FETCHED-PRIMARY] Code is Apache-2.0 and the repo documents custom COCO-format fine-tuning. The repository does not separately license checkpoints or the COCO/Objects365 data. **Posture: `unknown-needs-review`**.
- [INFERENCE] Expected leverage: coverage medium-high, switches neutral, spectator FP medium. Use as a same-family control only after RF-DETR-L because no primary occlusion study establishes better partial-person recall.

### [INFERENCE] 3 — DEIMv2-S/M

- [FETCHED-PRIMARY] DEIMv2 reports S 50.9 AP, 9.7M parameters, 5.78 ms and M 53.0 AP, 18.1M, 8.80 ms; released Hugging Face and Google Drive artifacts exist. [official model zoo](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo)
- [FETCHED-PRIMARY] The repo has Apache-2.0 code and a 20-epoch fine-tune example; TensorRT FP16 correctness requires TensorRT >=10.6. Weight/data licenses are not separately stated. **Posture: `unknown-needs-review`**.
- [INFERENCE] Expected leverage: coverage medium, excellent latency control, spectator FP medium. Benchmark only if RF-DETR latency or training stability fails.

### [INFERENCE] 4 — RT-DETRv3-R50

- [FETCHED-PRIMARY] Official WACV 2025 artifact reports 53.4 COCO AP at 640 and 108 FPS T4 TensorRT FP16, with a released Google Drive checkpoint. [repo](https://github.com/clxia12/RT-DETRv3)
- [FETCHED-PRIMARY] Code is Apache-2.0; checkpoint and COCO data are not separately licensed. **Posture: `unknown-needs-review`**.
- [INFERENCE] Expected leverage: coverage medium, spectator FP medium. Its mature NMS-free design is a useful fallback but does not outrank newer RF-DETR/D-FINE recipes for this gate.

### Baseline-name pin: public YOLO26m

- [FETCHED-PRIMARY] Ultralytics YOLO26 has n/s/m/l/x detection checkpoints and a real `yolo26m.pt`; default one-to-one inference emits final detections without NMS, while the optional one-to-many head requires NMS. [official model page](https://docs.ultralytics.com/models/yolo26/)
- [FETCHED-PRIMARY] The training recipe adds Progressive Loss, STAL for positive-label coverage, and MuSGD. The public family spans 40.9–57.5 COCO mAP at 1.7–11.8 ms T4 TensorRT. [official docs](https://docs.ultralytics.com/models/yolo26/)
- [INFERENCE] The frozen baseline record should pin: `ultralytics` package version/commit, `yolo26m.pt` SHA-256, one-to-one versus one-to-many head, image size, confidence threshold, augmentation/export engine, and NMS setting. “YOLO26m” alone is not a reproducible detector identity.
- [FETCHED-PRIMARY] Code/models/docs are AGPL-3.0 or Enterprise. **Posture: `R&D-only` under the public AGPL path unless DinkVision holds compatible Enterprise terms.**

### Crowd, visibility, and amodal directions

- [FETCHED-PRIMARY] CrowdHuman supplies head, visible-region, and full-body boxes for 470K people across 15K train/4,370 validation/5K test images. [paper](https://arxiv.org/abs/1805.00123)
- [INFERENCE] Visible/full paired supervision is useful for truncation-aware loss and evaluation, but a CrowdHuman-trained model is not automatically product-clean; dataset access/rights need review and its street/crowd domain lacks court negatives.
- [FETCHED-PRIMARY] DETR for Crowd Pedestrian Detection demonstrated that set prediction can remove hand-designed NMS in crowded scenes. [paper](https://arxiv.org/abs/2012.06785)
- [INFERENCE] NMS-free design reduces one known overlap-suppression failure, but no surveyed paper proves that RF-DETR, D-FINE, or DEIMv2 beats YOLO26m on partially visible pickleball players. That remains the purpose of the frozen benchmark.
- [FETCHED-PRIMARY] SAMEO performs amodal segmentation with a SAM decoder and synthetic Amodal-LVIS training; Video Amodal Segmentation with diffusion improves occluded-region completion on its benchmarks. [SAMEO](https://openaccess.thecvf.com/content/CVPR2025/html/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.html), [video amodal](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Using_Diffusion_Priors_for_Video_Amodal_Segmentation_CVPR_2025_paper.html)
- [INFERENCE] Amodal outputs hallucinate invisible extent and must not count as measured-player coverage. They are diagnostic/mask-memory inputs only, not detection truth.

## 3. ReID for same clothing and a four-person gallery

### Ranked usable or diagnostic candidates

| Rank | Candidate | Evidence and exact artifact | License posture | Expected effect/cost |
|---|---|---|---|---|
| 1 | Owned per-game centroid gallery on frozen embeddings | [FETCHED-PRIMARY] A rugby player-tracking study reports full-game identity classification with six short game-specific tracklets per player; the SoccerNet sports-ReID study uses hierarchical sampling and centroid loss for few samples per identity. [few-game paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.html), [sports ReID](https://arxiv.org/abs/2206.02373) | [INFERENCE] Owned game clips/labels can be **`commercial-clean`**; the current OSNet feature extractor remains R&D-only. | [INFERENCE] Best fit to four known people: enroll multiple clean crops per identity, use robust centroids, require an absolute reject threshold and a margin over runner-up for spectators. Low implementation/training cost; must validate leave-one-occlusion-window-out. |
| 2 | SapiensID checkpoint | [FETCHED-PRIMARY] CVPR 2025 model combines scale-adaptive retina patches, semantic body-part attention, and masked recognition; official repo exposes one Google Drive checkpoint. [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.html), [checkpoint instructions](https://github.com/mk-minchul/sapiensid/tree/main/tasks/sapiensID) | [FETCHED-PRIMARY] Repo/checkpoint governed by CC-BY-NC-4.0; WebBody training imagery is web-derived. **`R&D-only`**. | [INFERENCE] Highest-interest 2025 scale/pose checkpoint for an offline diagnostic; likely too heavy and legally blocked for the default. Coverage neutral, switches possibly lower after long occlusion, spectator rejection unknown. |
| 3 | Pose2ID NFC on an owned gallery | [FETCHED-PRIMARY] CVPR 2025 code applies neighbor feature centralization to existing ReID or even ImageNet features; Hugging Face artifact includes a Market-1501 TransReID checkpoint. [repo](https://github.com/yuanc3/Pose2ID), [model](https://huggingface.co/yuanc3/Pose2ID) | [FETCHED-PRIMARY] Code/model card MIT, but supplied TransReID weights and demo rely on Market-1501; Market-1501 is academic-use-only. **`R&D-only`** for released recipe; an owned-data/ImageNet-only variant is `unknown-needs-review`. | [INFERENCE] Low-cost post-embedding diagnostic. Risk: NFC can pull a spectator into an identity cluster; enforce open-set rejection before neighborhood aggregation. |
| 4 | KPR | [FETCHED-PRIMARY] Keypoint Promptable ReID compares only mutually visible parts and supports positive/negative keypoint prompts; repo's last advertised joint-dataset weight remains planned and training configs warn they may not reproduce the paper. [repo](https://github.com/VlSomers/keypoint_promptable_reidentification) | [FETCHED-PRIMARY] **Hippocratic License 3.0 `HL3-LAW-MEDIA-MIL-SOC-SV`**, with research datasets/weights. **`R&D-only` / license-blocked diagnostic**. | [INFERENCE] Best conceptual visibility-aware match for net occlusion, but not adoptable and not the next experiment. |
| 5 | Baseline OSNet x1.0 Market-1501 | [FETCHED-PRIMARY] Torchreid code and Hugging Face model card are MIT; the default checkpoint is explicitly Market-1501-trained. [repo](https://github.com/KaiyangZhou/deep-person-reid), [weights](https://huggingface.co/kaiyangzhou/osnet) | [FETCHED-PRIMARY] Market-1501 states academic use only and prohibits redistribution. **`R&D-only`** despite MIT code/model-card metadata. [official dataset page](https://zheng-lab-anu.github.io/Datasets.html) | [INFERENCE] Current accuracy evidence is useful, but this should not become a commercial default without replacing/retraining the weights on cleared data. |
| 6 | GIF / MuPNIT | [FETCHED-PRIMARY] WACV 2026 reports zero-shot global identity fusion and large gains over OC-SORT on its new multi-perspective sports benchmark. [paper](https://openaccess.thecvf.com/content/WACV2026/html/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.html) | [FETCHED-PRIMARY] No code, weights, or dataset license link in the paper landing page. **`unknown-needs-review`**. | [INFERENCE] Research reference for seen/unseen separation, not a benchmarkable checkpoint. |

### Enrollment conclusion

- [FETCHED-PRIMARY] Published sports support exists for a small game-specific gallery: six short tracklets per player were sufficient in the rugby study, and SoccerNet ReID explicitly contains few samples per identity. [few-game paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.html), [SoccerNet ReID repo](https://github.com/SoccerNet/sn-reid)
- [INFERENCE] No primary source found validates exactly four pickleball identities plus open-set spectator rejection. The benchmark must measure both genuine re-lock and impostor rejection, not only rank-1 among the four enrolled players.
- [INFERENCE] Recommended protocol: 8–20 high-quality crops per player across near/far court, serve/ready posture, and both sides of the net; robust median/trimmed centroid; cosine threshold calibrated only on train clips; reject unless best score clears an absolute threshold and the best-vs-second margin; never update a gallery from low-confidence or occluded crops.

## 4. Mask cues and VOS tracking

### McByte status

- [FETCHED-PRIMARY] Repo is live, MIT, last pushed 2025-07-22, and uses original SAM ViT-B to initialize masks plus Cutie base-mega to propagate them. [repo](https://github.com/tstanczyk95/McByte)
- [FETCHED-PRIMARY] The mask changes association only under ambiguity/isolation and after confidence, fill-ratio, and box-coverage checks; detections remain those supplied by YOLOX. [paper method](https://arxiv.org/html/2506.01373)
- [FETCHED-PRIMARY] End-to-end throughput is only 3–5 FPS on one A100. For 60 FPS input, that is about 12–20x video duration before I/O variability. [paper appendix D](https://arxiv.org/html/2506.01373)
- [INFERENCE] This runtime is the largest practical surprise: McByte remains a good correctness probe but is not a near-real-time default. It should report wall-clock RTF and measured-box versus propagated-mask coverage separately.

### Selective propagation is the better new architecture

- [FETCHED-PRIMARY] Selective Mask Propagation opens VOS windows only when assignment margin, a disappearance/reappearance gap, or a nearby witness track signals risk; weak/degraded/edge outcomes leave the base output unchanged. [paper](https://arxiv.org/html/2606.13033)
- [FETCHED-PRIMARY] At the chosen DanceTrack threshold it reports 0.107 SAM seconds per frame on RTX 5090, about 9.3 propagated FPS inside opened windows; 94.7% of SportsMOT windows leave base output unchanged. [paper tables/analysis](https://arxiv.org/html/2606.13033)
- [INFERENCE] For DinkVision, dispatch should include a **detection-gap trigger** in addition to association ambiguity because coverage is the failure. The mask may preserve identity through a gap, but mask-only frames must be labeled as propagated hypotheses, not detector measurements.

### SAM2.1, Cutie, SAM-MT, MASA, DEVA

- [FETCHED-PRIMARY] SAM2.1 official A100 single-model table: tiny 91.2 FPS, small 84.8, base+ 64.1, large 39.5; code and checkpoints are Apache-2.0. [official repo](https://github.com/facebookresearch/sam2#sam-21-checkpoints)
- [FETCHED-PRIMARY] A later multi-target benchmark on A6000 measures SAM2.1-B+ at 37.2 FPS for one target, 22.9 for two, 17.8 for three, and 12.4 for five. [SAM-MT table 3](https://arxiv.org/html/2607.08688)
- [INFERENCE] Four targets should be measured directly; interpolation between three and five suggests nowhere near 60 FPS. Re-seed from high-confidence detector masks/boxes before crossings, after low mask confidence, after edge re-entry, and periodically only on clean isolated frames.
- [FETCHED-PRIMARY] Cutie code is MIT; SAM-MT's comparison measures Cutie-base at 30.4 FPS for three and 27.4 for five targets in one setting, with a second configuration at 14.7/13.2. [Cutie repo](https://github.com/hkchengrex/Cutie), [SAM-MT table 3](https://arxiv.org/html/2607.08688)
- [FETCHED-PRIMARY] SAM-MT holds 36.8 FPS at three and 36.5 at five targets on A6000/1024p, but its code repo has no LICENSE and checkpoint is CC-BY-NC-SA-4.0. **`R&D-only` weights; `unknown-needs-review` code.** [repo](https://github.com/FudanCVL/SAM-MT), [model card](https://huggingface.co/FudanCVL/SAM-MT/blob/main/README.md)
- [FETCHED-PRIMARY] MASA code/weights are Apache-2.0, but McByte's same-detector SportsMOT validation comparison reports 73.6 HOTA for MASA versus 83.9 for McByte; DEVA and Grounded SAM2 are lower. [MASA repo](https://github.com/siyuanliii/masa), [McByte comparison](https://arxiv.org/html/2506.01373)
- [INFERENCE] MASA/DEVA are not preferred for a four-player benchmark; they add integration cost without stronger sports evidence than McByte or selective SAM2.1 windows.

## 5. Domain adaptation and spectator rejection

### Evidence for small-data adaptation

- [FETCHED-PRIMARY] A soccer detection study reports an average +16 mAP after annotating about 100 domain samples per video and using transductive reliable instances. [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.html)
- [FETCHED-PRIMARY] A teacher-student soccer detector uses labeled images plus unlabeled broadcast video and reports a 52.3% mAP benchmark, with BSD-3-Clause code. [paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Vandeghen_Semi-Supervised_Training_To_Improve_Player_and_Ball_Detection_in_Soccer_CVPRW_2022_paper.html), [code](https://github.com/rvandeghen/SST)
- [FETCHED-PRIMARY] Six few-seconds tracklets per player support game-specific identity learning in rugby. [CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.html)
- [INFERENCE] These motivate a small owned-data fine-tune but do not predict DinkVision coverage or IDF1. Start with approximately 100 deliberately stratified labeled frames per worst clip, then add only error-driven frames; freeze a held-out clip-level gate before training.

### Explicit negatives and court geometry

- [INFERENCE] Use a **single positive class** `on_court_player` and include empty/background frames plus hard-negative frames containing spectators, passers-by, far-court people, reflections/screens, and edge-only people. Do not leave near-court distractors unlabeled in positive frames; either label them as an explicit `off_court_person` class or mask/ignore them consistently so the detector is not trained on contradictory background.
- [INFERENCE] Stratify loss/evaluation by visible fraction, net overlap, player-player overlap, truncation, distance band, lighting, and court side. Report per-stratum recall and spectator FP, not just COCO AP.
- [FETCHED-PRIMARY] FieldMOT provides primary evidence for “register then track” in field coordinates, but its setting is camera-switching broadcast video and synthesized football evaluation. [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.html)
- [INFERENCE] Replace the current hard center/margin heuristic only in a later controlled experiment with a calibrated **soft footpoint-on-court prior**: projected ankle/box-bottom location, uncertainty that grows under truncation, hysteresis at boundaries, and an explicit edge-entry exception. Geometry may veto spectator detections but must not suppress partially visible legitimate players.

### Public dataset license register

| Dataset | Primary-source terms | Use here |
|---|---|---|
| SportsMOT | [FETCHED-PRIMARY] 240 clips, >150K frames and >1.6M boxes; official README says CC-BY-NC-4.0 and no redistribution without permission. **`R&D-only`**. [repo](https://github.com/MCG-NJU/SportsMOT) | [INFERENCE] Good sports recall/association diagnostic; not product training data. |
| DanceTrack | [FETCHED-PRIMARY] Annotation license CC-BY-4.0, but videos/images are non-commercial research only; code MIT. **`R&D-only`** data. [agreement](https://github.com/DanceTrack/DanceTrack#agreement) | [INFERENCE] Uniform-appearance motion diagnostic, weak spectator-rejection fit. |
| SoccerNet tracking/ReID | [FETCHED-PRIMARY] FAQ says dataset is research-only/noncommercial and videos require NDA due league copyright. ReID has 340,993 crops from 400 games; code MIT. **`R&D-only`** data. [FAQ](https://www.soccer-net.org/faq), [ReID repo](https://github.com/SoccerNet/sn-reid) | [INFERENCE] Strong same-uniform/few-gallery R&D benchmark, not deployable training data. |
| CrowdHuman | [FETCHED-PRIMARY] Visible/full/head annotations and dense crowds; the paper/site do not provide a product-clean training-data grant in the surveyed artifact. **`unknown-needs-review`**. [paper](https://arxiv.org/abs/1805.00123) | [INFERENCE] Occlusion pretraining/evaluation diagnostic only until rights review. |
| Owned DinkVision clips | [INFERENCE] Rights depend on capture consent/product policy; with cleared capture and annotations, this is the only surveyed route capable of **`commercial-clean`** detector/ReID domain supervision. | [INFERENCE] Primary fine-tune and frozen fresh-clip gate source. |

## Source disagreements and caveats

- [FETCHED-PRIMARY] The official SportsMOT README says **CC-BY-NC-4.0**. [official](https://github.com/MCG-NJU/SportsMOT#terms-of-use)
- [SECONDARY] Roboflow Trackers' dataset table instead labels SportsMOT **CC-BY-4.0**. [third-party table](https://github.com/roboflow/trackers#datasets)
- [INFERENCE] Use the dataset owner's restrictive CC-BY-NC-4.0 statement; the third-party table is not authority.
- [FETCHED-PRIMARY] Selective Mask Propagation v3 (2026-07-09) reports 87.2 HOTA; earlier search indexing exposed 86.8 from an older version. [current v3](https://arxiv.org/abs/2606.13033)
- [INFERENCE] Cite the versioned v3 number and record the paper version in any reproduction.
- [FETCHED-PRIMARY] RF-DETR's benchmark table calls all segmentation sizes Apache-2.0 while only detection XL/2XL are Plus/PML-1.0. [README](https://github.com/roboflow/rf-detr)
- [INFERENCE] Do not generalize the detection Plus restriction to segmentation XL/2XL; do not generalize Apache code to unmentioned third-party data.
- [FETCHED-PRIMARY] GitHub/Hugging Face model-card metadata can say MIT/Apache while the supplied checkpoint was trained on Market-1501, SportsMOT, COCO, or another separately governed dataset.
- [INFERENCE] This survey's posture uses the most restrictive unresolved component and does not treat repository metadata as training-data clearance.

## Open items answered

1. [FETCHED-PRIMARY] **Current sports leader:** SAM 3-Deep-EIoU+GTA claims 87.2 SportsMOT HOTA; the bulk of its validation gain is GTA/global association, not detector recall.
2. [INFERENCE] **Coverage-first detector:** RF-DETR-L exact 2026 checkpoint first; RF-DETR Seg-L/XL next for a mask-producing detector; D-FINE-L and DEIMv2-S/M are controls.
3. [FETCHED-PRIMARY] **YOLO26 identity:** real Ultralytics n/s/m/l/x family, NMS-free one-to-one default, AGPL-3.0/Enterprise; pin weight SHA and head mode.
4. [FETCHED-PRIMARY] **ReID legality:** current OSNet code is MIT but Market-1501 checkpoint posture is R&D-only; KPR remains Hippocratic-license blocked; SapiensID is CC-BY-NC-4.0.
5. [FETCHED-PRIMARY] **Few-shot support:** published sports evidence supports a few game-specific tracklets/identity and centroid-style learning, but not four-player open-set pickleball.
6. [FETCHED-PRIMARY] **McByte:** code is live/MIT and sports gains are real under fixed detections; end-to-end speed is only 3–5 FPS on A100.
7. [FETCHED-PRIMARY] **Best new mask idea:** selective uncertainty/gap-window dispatch; no released code, and its published SportsMOT gain is association-heavy.
8. [FETCHED-PRIMARY] **SAM2.1 cost:** official single-model numbers overstate four-target throughput; independent multi-target measurement drops B+ to 17.8 FPS at three and 12.4 at five on A6000.
9. [INFERENCE] **Spectator rejection:** no off-the-shelf tracker directly solves it; use owned hard-negative supervision plus a soft court-footpoint prior, with edge/truncation exceptions.
10. [FETCHED-PRIMARY] **Public sports data:** SportsMOT, DanceTrack video, and SoccerNet are noncommercial/research-only; they are R&D diagnostics, not clean product-training sources.

## Recommended benchmark order on the frozen worst-clip set

### Common protocol

- [FETCHED-PRIMARY] Keep the current worst-clip set, frozen identity baseline, court margin, association/ReID settings, and full per-clip gate unchanged. `VERIFIED=0` remains binding. [NORTH_STAR_ROADMAP.md](../../../NORTH_STAR_ROADMAP.md)
- [INFERENCE] For every detector candidate, freeze one confidence threshold from development clips, then score the same raw detections and the same downstream association. Record: player recall/coverage, `cov4`, IDF1, switches, spectator FP, far-off-court FP, visible-fraction recall, net-overlap recall, edge-truncation recall, latency, peak VRAM, and wall-clock real-time factor.
- [INFERENCE] Store three separate frame states: detector-measured, tracker-predicted, and mask-propagated. Do not let propagated hypotheses silently satisfy measured coverage.
- [INFERENCE] Reject any candidate that creates a switch, spectator FP, or far-off-court FP even if coverage rises. Published COCO/SportsMOT numbers are motivation only.

### Ordered experiments

1. **[INFERENCE] RF-DETR-L zero-shot detection**
   - [FETCHED-PRIMARY] Exact starter: `RFDETRLarge`, `rf-detr-large-2026.pth`, default 704 resolution, Apache-designated weight. [registry/config](https://github.com/roboflow/rf-detr/tree/develop/src/rfdetr)
   - [INFERENCE] Replace only detector boxes; keep current BoT-SORT/OSNet/raw-pool/margin. Compare against a rerun of frozen YOLO26m on identical decoded frames.
   - [INFERENCE] Expected: highest immediate coverage leverage; main failure risk is extra off-court people.

2. **[INFERENCE] RF-DETR-L owned-domain fine-tune**
   - [INFERENCE] Start with ~100 stratified frames per worst clip: clean full players, net occlusion, player overlap, similar clothing, edge truncation, near spectators/passers-by, far-court people, and negative-only frames. Split by clip/game, never adjacent frame.
   - [INFERENCE] Train `on_court_player`; either explicitly label off-court people as a negative class or use consistent ignore regions. Compare full fine-tune with frozen-encoder/LoRA only if the full recipe overfits.
   - [INFERENCE] Expected: highest coverage plus spectator-rejection leverage.

3. **[INFERENCE] RF-DETR Seg-L, then Seg-XL, same owned split**
   - [FETCHED-PRIMARY] Exact starters: `rf-detr-seg-l-ft.pth` and `rf-detr-seg-xl-ft.pth`, Apache-designated. [weights registry](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/assets/model_weights.py)
   - [INFERENCE] Score their boxes first; archive masks for the later cue experiment. Stop if box recall is below detector-L or runtime is unacceptable.

4. **[INFERENCE] D-FINE-L Objects365+COCO control; DEIMv2-S control if latency is binding**
   - [FETCHED-PRIMARY] Exact D-FINE weight: `dfine_l_obj2coco_e25.pth`; exact DEIM card: `Intellindust/DEIMv2_DINOv3_S_COCO`. [D-FINE](https://github.com/Peterande/D-FINE), [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
   - [INFERENCE] Zero-shot first; permit a matched owned-data recipe only if RF-DETR misses the gate or runtime target.

5. **[INFERENCE] Four-person enrollment gallery after the best detector is frozen**
   - [INFERENCE] A/B current OSNet raw pool versus robust per-player centroids and explicit open-set rejection. Use only clean owned enrollment crops; prohibit online gallery writes from uncertain frames.
   - [INFERENCE] Score re-lock after each occlusion plus spectator false accepts. Do not promote Market-1501/KPR/SapiensID weights.

6. **[INFERENCE] McByte cue on the ruled worst clips**
   - [FETCHED-PRIMARY] Start from paper constants: SAM ViT-B, Cutie base-mega, ambiguity/isolation gating, mask confidence 0.6, box coverage 0.9, fill ratio 0.05. [paper](https://arxiv.org/html/2506.01373)
   - [INFERENCE] Feed detections from the best frozen detector, not the paper's SportsMOT YOLOX weight. Score measured boxes separately and expect 12–20x source duration at the published 3–5 FPS before optimization.

7. **[INFERENCE] Selective SAM2.1-B+ gap/ambiguity windows**
   - [INFERENCE] Spec-only adaptation of the 2026 selective paper: seed on the last isolated high-confidence mask, open on low assignment margin or detection gap, add nearby witness target, and accept only confident corrections. Begin with SAM2.1-B+ because code/weights are Apache-2.0.
   - [INFERENCE] This is preferable to always-on SAM2MOT/SAMIDARE if McByte proves useful but too slow. Tune dispatch once on development clips; no per-clip thresholds.

8. **[INFERENCE] Research-only diagnostics, never promotion candidates**
   - [INFERENCE] SapiensID, Pose2ID/Market, KPR, SAM-MT, SAMIDARE, SportsMOT-trained detector/ReID, and SAM 3-Deep-EIoU until their relevant license/code/data gaps are resolved.

### Stop and promotion rules

- [INFERENCE] Stop a detector branch after two frozen-threshold attempts if coverage does not materially exceed YOLO26m or any prohibited FP/switch appears.
- [INFERENCE] Stop a ReID/mask branch if the best detector still fails coverage: it is treating a downstream symptom before the missing-box cause.
- [FETCHED-PRIMARY] No candidate is promoted until every fresh clip independently passes IDF1 >=0.85, zero switches, zero spectator FP, zero far-off-court FP, and coverage >=0.95. [NORTH_STAR_ROADMAP.md](../../../NORTH_STAR_ROADMAP.md)
- [INFERENCE] This survey authorizes no default flip and supplies no pickleball accuracy claim.

## Compact license register

| Candidate | Code | Released weights | Training data | Posture |
|---|---|---|---|---|
| RF-DETR L / Seg L–2XL | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] Apache-2.0 designated | [FETCHED-PRIMARY] COCO; source-image rights vary | **`unknown-needs-review`** |
| RF-DETR detection XL/2XL | [FETCHED-PRIMARY] PML-1.0 Plus | [FETCHED-PRIMARY] PML-1.0 | [FETCHED-PRIMARY] COCO | **`unknown-needs-review`** |
| D-FINE | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] no separate declaration found | [FETCHED-PRIMARY] COCO/Objects365 | **`unknown-needs-review`** |
| DEIMv2 | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] no separate declaration found | [FETCHED-PRIMARY] COCO; DINOv3-derived backbone | **`unknown-needs-review`** |
| RT-DETRv3 | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] no separate declaration found | [FETCHED-PRIMARY] COCO | **`unknown-needs-review`** |
| YOLO26 | [FETCHED-PRIMARY] AGPL-3.0 / Enterprise | [FETCHED-PRIMARY] AGPL-3.0 / Enterprise | [FETCHED-PRIMARY] COCO | **`R&D-only` unless Enterprise-covered** |
| GTA | [FETCHED-PRIMARY] MIT | [FETCHED-PRIMARY] released assets/dependencies vary | [FETCHED-PRIMARY] SportsMOT/SoccerNet recipes | **`R&D-only`** |
| SAM2MOT | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] upstream detector/SAM terms vary | [FETCHED-PRIMARY] DanceTrack/CrowdHuman/COCO recipes | **`unknown-needs-review`** |
| SAMIDARE | [FETCHED-PRIMARY] no LICENSE | [FETCHED-PRIMARY] SAM2 Apache; detector asset separate | [FETCHED-PRIMARY] SportsMOT | **`unknown-needs-review`** |
| McByte released recipe | [FETCHED-PRIMARY] MIT | [FETCHED-PRIMARY] SAM/Cutie permissive; SportsMOT YOLOX asset | [FETCHED-PRIMARY] SportsMOT and upstream VOS data | **`R&D-only`** |
| SAM2.1 | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] SA-V and other sources per paper | **`commercial-clean` artifact surface; data review advisable** |
| Cutie | [FETCHED-PRIMARY] MIT | [FETCHED-PRIMARY] no separate declaration found | [FETCHED-PRIMARY] VOS datasets | **`unknown-needs-review`** |
| MASA | [FETCHED-PRIMARY] Apache-2.0 | [FETCHED-PRIMARY] Hugging Face card Apache-2.0 | [FETCHED-PRIMARY] SAM/image corpora | **`unknown-needs-review`** |
| SAM-MT | [FETCHED-PRIMARY] no repo LICENSE | [FETCHED-PRIMARY] CC-BY-NC-SA-4.0 | [FETCHED-PRIMARY] VOS datasets | **`R&D-only`** |
| OSNet Market default | [FETCHED-PRIMARY] MIT | [FETCHED-PRIMARY] card MIT | [FETCHED-PRIMARY] Market-1501 academic-only | **`R&D-only`** |
| SapiensID | [FETCHED-PRIMARY] CC-BY-NC-4.0 | [FETCHED-PRIMARY] same repo license | [FETCHED-PRIMARY] WebBody4M web data | **`R&D-only`** |
| Pose2ID released demo | [FETCHED-PRIMARY] MIT | [FETCHED-PRIMARY] card MIT | [FETCHED-PRIMARY] Market-1501 demo/checkpoint | **`R&D-only`** |
| KPR | [FETCHED-PRIMARY] Hippocratic HL3-LAW-MEDIA-MIL-SOC-SV | [FETCHED-PRIMARY] research checkpoints | [FETCHED-PRIMARY] Market/Occluded-Duke/etc. | **`R&D-only`, license-blocked** |

## Survey limitations

- [INFERENCE] This is a primary-source survey, not a systematic meta-analysis; inaccessible/unreleased code cannot be audited beyond the paper and live URL checks.
- [INFERENCE] HTTP reachability proves only that an artifact endpoint responded on 2026-07-16, not that a weight is correct, safe, reproducible, or licensed for DinkVision.
- [INFERENCE] License posture is engineering triage, not legal advice. “Commercial-clean” requires project counsel/policy to accept the complete dependency and data chain.
- [INFERENCE] Published benchmark metrics are motivation to run the frozen DinkVision gate and are not evidence of pickleball accuracy.
