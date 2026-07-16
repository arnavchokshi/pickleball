# Live checks — RKT survey B

Checked 2026-07-16 (America/Los_Angeles). `bytes` is the response body actually transferred by the check, not an advertised artifact size. PDF checks marked `range` intentionally fetched only the first 1,048,576 bytes. No dataset archive or multi-GB asset was downloaded. A `000` row records a timed-out endpoint, not a dead-link conclusion.

## Synthetic generation and grasp sources

| Artifact | URL | HTTP | bytes | Note |
|---|---|---:|---:|---|
| BlenderProc repository | https://github.com/DLR-RM/BlenderProc | 200 | 381012 | Live repository page |
| BlenderProc GPL-3.0 license | https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/LICENSE | 200 | 35147 | Raw primary license |
| BlenderProc motion blur / rolling shutter example | https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/advanced/motion_blur_rolling_shutter/README.md | 200 | 2653 | Raw example documentation |
| Kubric repository | https://github.com/google-research/kubric | 200 | 327743 | Live repository page |
| Kubric Apache-2.0 license | https://raw.githubusercontent.com/google-research/kubric/main/LICENSE | 200 | 11358 | Raw primary license |
| Isaac Sim Replicator object SDG documentation | https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html | 200 | 144219 | Redirected to `docs.isaacsim.omniverse.nvidia.com/latest/index.html`; exact old page moved |
| Unity Perception repository | https://github.com/Unity-Technologies/com.unity.perception | 200 | 328195 | Repository states discontinued |
| Unity Perception Apache-2.0 license | https://raw.githubusercontent.com/Unity-Technologies/com.unity.perception/main/LICENSE.md | 200 | 11410 | Raw primary license |
| Unreal Engine motion-blur documentation | https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-motion-blur | 200 | 55729 | Live engine documentation |
| GraspXL paper | https://arxiv.org/abs/2403.19649 | 200 | 43470 | Abstract/metadata page |
| GraspXL dataset card | https://huggingface.co/datasets/ethHuiZhang/GraspXL | 000 | 0 | Timed out twice at 20–35 s; card content was fetched separately during source review |
| GRAB repository | https://github.com/otaheri/GRAB | 200 | 328503 | Live repository page |
| GRAB custom non-commercial license | https://raw.githubusercontent.com/otaheri/GRAB/master/LICENSE | 200 | 7189 | Raw primary license |
| OakInk project | https://oakink.net/ | 200 | 16658 | Live project page; no clear data license surfaced |

## Sim-to-real and pose methods

| Artifact | URL | HTTP | bytes | Note |
|---|---|---:|---:|---|
| DOPE paper | https://proceedings.mlr.press/v87/tremblay18a/tremblay18a.pdf | 200 | 7594559 | Full PDF |
| Self6D paper | https://arxiv.org/pdf/2004.06468 | 200 | 8089151 | Full PDF |
| MegaPose paper | https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf | 200 | 13291973 | Full PDF |
| MegaPose repository | https://github.com/megapose6d/megapose6d | 200 | 413163 | Live repository page |
| MegaPose Apache-2.0 license | https://raw.githubusercontent.com/megapose6d/megapose6d/master/LICENSE | 200 | 605 | Raw primary license; repository default branch is `master` |
| GigaPose paper | https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.pdf | 206 | 1048576 | Range check |
| GigaPose repository | https://github.com/nv-nguyen/gigapose | 200 | 354184 | Live repository page |
| GigaPose MIT license | https://raw.githubusercontent.com/nv-nguyen/gigapose/main/LICENSE | 200 | 1074 | Raw primary license |
| FoundationPose repository | https://github.com/NVlabs/FoundationPose | 200 | 373545 | Live repository page |
| FoundationPose NVIDIA non-commercial license | https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE | 200 | 4352 | Section 3.3 limits use to non-commercial research/evaluation |
| FoundPose repository | https://github.com/facebookresearch/foundpose | 200 | 335877 | Live repository page |
| FoundPose CC BY-NC 4.0 license | https://raw.githubusercontent.com/facebookresearch/foundpose/main/LICENSE | 200 | 19334 | Raw primary license |
| RACE-6D paper | https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf | 206 | 1048576 | Range check |
| IPPE repository | https://github.com/tobycollins/IPPE | 200 | 285010 | Live repository page |

## Datasets and ground-truth methodology

| Artifact | URL | HTTP | bytes | Note |
|---|---|---:|---:|---|
| RacketVision AAAI paper | https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324 | 200 | 7405673 | Full PDF |
| RacketVision repository | https://github.com/OrcustD/RacketVision | 200 | 349591 | Live repository page |
| RacketVision MIT license | https://raw.githubusercontent.com/OrcustD/RacketVision/main/LICENSE | 200 | 1069 | Raw primary license |
| RacketVision Hugging Face card | https://huggingface.co/datasets/OrcustD/RacketVision | 000 | 0 | Timed out twice at 35–60 s; card content was fetched separately during source review |
| TT4D paper | https://arxiv.org/pdf/2605.01234 | 200 | 15882932 | Full PDF |
| Imitrob paper | https://arxiv.org/pdf/2209.07976 | 200 | 10894861 | Full PDF |
| Garon et al. 6DoF tracking paper | https://arxiv.org/pdf/1803.10075 | 200 | 10295265 | Full PDF |
| PhoCaL paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf | 206 | 1048576 | Range check |
| Anipose paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC8498918/ | 200 | 341316 | Full HTML |
| T-LESS paper | https://arxiv.org/abs/1701.05498 | 200 | 44578 | Abstract/metadata page |
| MVTec ITODD dataset page | https://www.mvtec.com/research-teaching/datasets/mvtec-itodd | 200 | 70200 | Official dataset page |
| OpenCV ChArUco calibration documentation | https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html | 200 | 40478 | Redirected to OpenCV 4.13.0 docs |
| Consumer-camera audio synchronization paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC5051647/ | 200 | 170677 | Full HTML |

## Contact localization

| Artifact | URL | HTTP | bytes | Note |
|---|---|---:|---:|---|
| Event-camera tennis impact paper | https://arxiv.org/pdf/2506.08327 | 200 | 4675924 | Full PDF |
| Dual-event-camera badminton impact paper | https://arxiv.org/abs/2605.28011 | 200 | 43330 | Abstract/metadata page |
| Piezoelectric table-tennis impact paper | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0333735 | 200 | 183605 | Full HTML |

## Disagreements / endpoint cautions

- [FETCHED-PRIMARY] The former Isaac Sim tutorial URL currently resolves to the documentation index, so the survey treats Replicator's general SDG/path-tracing maturity as supported but does **not** claim verified native rolling-shutter support from that link.
- [FETCHED-PRIMARY] FoundationPose has no `LICENSE.md` (404, 14 bytes); the live canonical file is `LICENSE` and contains the non-commercial restriction.
- [FETCHED-PRIMARY] MegaPose's license exists on `master`, not `main`; the failed `main/LICENSE` check returned 404/14 bytes before the canonical 200/605-byte check.
- [INFERENCE] Hugging Face timeouts are availability observations from this machine only. They do not invalidate the dataset cards, but the underlying video rights still require legal review.
