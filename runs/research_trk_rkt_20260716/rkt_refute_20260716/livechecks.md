# Live URL checks

All checks were performed **2026-07-16**. `GET bytes` is the response body actually received. For weight `HEAD` requests, `bytes` is the server-declared `Content-Length`/`X-Linked-Size`; the body received was zero. `n/e` means the browser fetch returned HTTP 200 but did not expose a byte count. Failed and deliberately invalid probes are retained for auditability. No multi-GB object body was fetched.

| Claim | Method | HTTP | Bytes | Date | URL / note |
|---|---:|---:|---:|---|---|
| C1 | GET | 200 | 42,688 | 2026-07-16 | https://arxiv.org/abs/2511.17045 |
| C1 | GET | 200 | 7,503,719 | 2026-07-16 | https://arxiv.org/pdf/2511.17045 |
| C1 | GET | 200 | 5,972 | 2026-07-16 | https://api.github.com/repos/OrcustD/RacketVision |
| C1 | browser | 200 | n/e | 2026-07-16 | https://github.com/OrcustD/RacketVision |
| C1 | GET | 200 | 4,044 | 2026-07-16 | https://api.github.com/repos/OrcustD/RacketVision/contents |
| C1 | GET | 200 | 10,577 | 2026-07-16 | https://raw.githubusercontent.com/OrcustD/RacketVision/main/README.md |
| C1 | GET | 200 | 2,486 | 2026-07-16 | https://api.github.com/repos/OrcustD/RacketVision/license |
| C1 | GET | 200 | 1,929,685 | 2026-07-16 | https://huggingface.co/api/datasets/linfeng302/RacketVision |
| C1 | GET | 504 | 132 | 2026-07-16 | https://huggingface.co/datasets/linfeng302/RacketVision/raw/main/README.md — transient gateway failure; API metadata succeeded |
| C1 | GET | 200 | 761 | 2026-07-16 | https://huggingface.co/api/models/linfeng302/RacketVision-Models |
| C1 | GET | 200 | 1,945 | 2026-07-16 | https://huggingface.co/api/models/linfeng302/RacketVision-Models?blobs=true |
| C1 | GET | 200 | 2,387 | 2026-07-16 | https://huggingface.co/api/models/linfeng302/RacketVision-Models?blobs=true&securityStatus=true |
| C1 | GET | 200 | 12,040 | 2026-07-16 | https://huggingface.co/api/models/linfeng302/RacketVision-Models/tree/main/checkpoints?recursive=false&expand=true |
| C1 | GET | 400 | 422 | 2026-07-16 | https://huggingface.co/api/models/linfeng302/RacketVision-Models?blobs=true&expand[]=securityStatus — unsupported expansion probe |
| C1 | GET | 200 | 88,902 | 2026-07-16 | https://huggingface.co/linfeng302/RacketVision-Models/blob/main/checkpoints/epoch_300.pth |
| C1 | HEAD | 200 | 411,293,859 | 2026-07-16 | https://huggingface.co/linfeng302/RacketVision-Models/resolve/main/checkpoints/epoch_300.pth |
| C1 | HEAD | 200 | 106,524,759 | 2026-07-16 | https://huggingface.co/linfeng302/RacketVision-Models/resolve/main/checkpoints/best_PCK_epoch_90.pth?download=true |
| C2 | browser | 200 | n/e | 2026-07-16 | https://openaccess.thecvf.com/content/CVPR2026F/html/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.html |
| C2 | browser | 200 | n/e | 2026-07-16 | https://openaccess.thecvf.com/content/CVPR2026F/papers/Ha_RACE-6D_Real-time_Accurate_Coarse-to-finE_Object_6D_Pose_Transformer_CVPRF_2026_paper.pdf |
| C2 | GET | 200 | 5,773 | 2026-07-16 | https://api.github.com/repos/Yoonwoo-Ha/RACE-6D |
| C2 | browser | 200 | n/e | 2026-07-16 | https://github.com/Yoonwoo-Ha/RACE-6D |
| C2 | GET | 200 | 10,365 | 2026-07-16 | https://raw.githubusercontent.com/Yoonwoo-Ha/RACE-6D/main/README.md |
| C2 | GET | 200 | 41,805 | 2026-07-16 | https://api.github.com/repos/Yoonwoo-Ha/RACE-6D/git/trees/main?recursive=1 |
| C2 | GET | 200 | 5 | 2026-07-16 | https://api.github.com/repos/Yoonwoo-Ha/RACE-6D/releases — `[]` |
| C2 | GET | 200 | 5 | 2026-07-16 | https://api.github.com/repos/Yoonwoo-Ha/RACE-6D/tags — `[]` |
| C3 | GET | 200 | 43,303 | 2026-07-16 | https://arxiv.org/abs/2605.01234 |
| C3 | GET | 200 | 15,882,932 | 2026-07-16 | https://arxiv.org/pdf/2605.01234 |
| C3 | GET | 200 | 73 | 2026-07-16 | https://api.github.com/search/repositories?q=%22TT4D%22%20table%20tennis — zero results |
| C3 | GET | 200 | 2 | 2026-07-16 | https://huggingface.co/api/models?search=TT4D — `[]` |
| C3 | GET | 200 | 2 | 2026-07-16 | https://huggingface.co/api/datasets?search=TT4D — `[]` |
| C4 | GET | 200 | 42,975 | 2026-07-16 | https://arxiv.org/abs/2506.08327 |
| C4 | GET | 200 | 4,675,924 | 2026-07-16 | https://arxiv.org/pdf/2506.08327 |
| C4 | GET | 200 | 43,330 | 2026-07-16 | https://arxiv.org/abs/2605.28011 |
| C4 | GET | 200 | 964,569 | 2026-07-16 | https://arxiv.org/pdf/2605.28011 |
| C5 | GET | 200 | 9,641,895 | 2026-07-16 | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460103.pdf |
| C5 | GET | 200 | 7,594,559 | 2026-07-16 | https://proceedings.mlr.press/v87/tremblay18a/tremblay18a.pdf |
| C5 | GET | 200 | 13,291,973 | 2026-07-16 | https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf |
| C5 | GET | 200 | 9,916 | 2026-07-16 | https://zhongcl-thu.github.io/rock/ |
| C5 | browser | 200 | n/e | 2026-07-16 | https://arxiv.org/abs/2202.00448 |
| C6 | GET | 200 | 10,295,265 | 2026-07-16 | https://arxiv.org/pdf/1803.10075 |
| C6 | browser | 200 | n/e | 2026-07-16 | https://arxiv.org/abs/2205.08811 |
| C6 | GET | 200 | 32,339,129 | 2026-07-16 | https://arxiv.org/pdf/2205.08811 |
| C6 | browser | 200 | n/e | 2026-07-16 | https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf |
| C6 | GET | 200 | 10,894,861 | 2026-07-16 | https://arxiv.org/pdf/2209.07976 |
| C6 | GET | 000 | 0 | 2026-07-16 | https://imitrob.ciirc.cvut.cz/imitrobdataset.php — connection failed; paper datasheet supplied license evidence |
| C6 | browser | 200 | n/e | 2026-07-16 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7423398/ |
| C7 | GET | 200 | 7,237 | 2026-07-16 | https://api.github.com/repos/DLR-RM/BlenderProc |
| C7 | GET | 200 | 35,147 | 2026-07-16 | https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/LICENSE |
| C7 | GET | 200 | 1,660 | 2026-07-16 | https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/advanced/motion_blur_rolling_shutter/main.py |
| C7 | GET | 200 | 7,517 | 2026-07-16 | https://api.github.com/repos/google-research/kubric |
| C7 | GET | 200 | 11,358 | 2026-07-16 | https://raw.githubusercontent.com/google-research/kubric/main/LICENSE |
| C7 | GET | 200 | 26,155 | 2026-07-16 | https://api.github.com/repos/google-research/kubric/commits?per_page=5 |
| C7 | GET | 200 | 489,371 | 2026-07-16 | https://docs.isaacsim.omniverse.nvidia.com/6.0.1/replicator_tutorials/tutorial_replicator_object_based_sdg.html |
| C7 | browser | 200 | n/e | 2026-07-16 | https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses-isaac-sim.html |
| C7 | browser | 200 | n/e | 2026-07-16 | https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/license-faq.html |
| C8 | GET | 200 | 4,352 | 2026-07-16 | https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE.md |
| C8 | GET | 200 | 10,937 | 2026-07-16 | https://raw.githubusercontent.com/NVlabs/FoundationPose/main/readme.md |
| C8 | GET | 404 | 14 | 2026-07-16 | https://raw.githubusercontent.com/NVlabs/FoundationPose/main/README.md — filename case probe |
| C8 | GET | 200 | 19,334 | 2026-07-16 | https://raw.githubusercontent.com/facebookresearch/foundpose/main/LICENSE |
| C8 | browser | 200 | n/e | 2026-07-16 | https://github.com/facebookresearch/foundpose |
| C8 | GET | 200 | 1,074 | 2026-07-16 | https://raw.githubusercontent.com/nv-nguyen/gigapose/main/LICENSE |
| C8 | browser/HEAD | 200 | 3,810,000,000 approx. | 2026-07-16 | https://huggingface.co/datasets/nv-nguyen/gigaPose/blob/main/gigaPose_v1.ckpt — pointer/metadata only |
| C8 | browser | 200 | n/e | 2026-07-16 | https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.html |
| C8 | GET | 200 | 5,643 | 2026-07-16 | https://api.github.com/repos/Marwan99/kv_tracker |
| C8 | GET | 200 | 9,809 | 2026-07-16 | https://api.github.com/repos/Marwan99/kv_tracker/contents |
| C8 | GET | 200 | 2,785 | 2026-07-16 | https://raw.githubusercontent.com/Marwan99/kv_tracker/main/README.md |
| C8 | GET | 200 | 11,820 | 2026-07-16 | https://raw.githubusercontent.com/Marwan99/kv_tracker/main/LICENSE |
| C8 | GET | 200 | 5,641 | 2026-07-16 | https://api.github.com/repos/otaheri/GRAB |
| C8 | GET | 200 | 7,189 | 2026-07-16 | https://raw.githubusercontent.com/otaheri/GRAB/master/LICENSE |
| C8 | GET | 404 | 14 | 2026-07-16 | https://raw.githubusercontent.com/otaheri/GRAB/main/LICENSE — branch probe |
| C8 | GET | 200 | 4,780 | 2026-07-16 | https://huggingface.co/api/datasets/ethHuiZhang/GraspXL?blobs=true |
| C8 | GET | 200 | 5,558 | 2026-07-16 | https://api.github.com/repos/JiehongLin/SAM-6D |
| C8 | GET | 200 | 2,291 | 2026-07-16 | https://api.github.com/repos/JiehongLin/SAM-6D/contents |
| C8 | GET | 200 | 58,347 | 2026-07-16 | https://api.github.com/repos/JiehongLin/SAM-6D/git/trees/main?recursive=1 |
| C8 | GET | 200 | 1,074 | 2026-07-16 | https://raw.githubusercontent.com/JiehongLin/SAM-6D/main/SAM-6D/Instance_Segmentation_Model/LICENSE |
| C8 | GET | 200 | 5,866 | 2026-07-16 | https://api.github.com/repos/tobycollins/IPPE |
| C8 | GET | 200 | 8,326 | 2026-07-16 | https://raw.githubusercontent.com/tobycollins/IPPE/master/README.md |
| C9 | browser | 200 | n/e | 2026-07-16 | https://openaccess.thecvf.com/content/CVPR2023/html/Oh_Recovering_3D_Hand_Mesh_Sequence_From_a_Single_Blurry_Image_CVPR_2023_paper.html |
| C9 | GET | 200 | 6,045 | 2026-07-16 | https://api.github.com/repos/JaehaKim97/BlurHand_RELEASE |
| C9 | GET | 404 | 14 | 2026-07-16 | https://raw.githubusercontent.com/JaehaKim97/BlurHand_RELEASE/main/README.md — branch probe |
| C9 | GET | 200 | 1,797 | 2026-07-16 | https://raw.githubusercontent.com/JaehaKim97/BlurHand_RELEASE/master/README.md |
| C9 | browser | 200 | n/e | 2026-07-16 | https://arxiv.org/abs/2303.17209 |
| C9 | GET | 200 | 46,720,487 | 2026-07-16 | https://arxiv.org/pdf/2303.17209 |
| C9 | GET | 200 | 73 | 2026-07-16 | https://api.github.com/search/repositories?q=%22racket%22+%22motion+blur%22+%226D+pose%22 — zero results |
| C9 | GET | 200 | 2 | 2026-07-16 | https://huggingface.co/api/models?search=racket%20motion%20blur%20pose — `[]` |
| C9 | GET | 200 | 6,165 | 2026-07-16 | https://api.github.com/repos/rozumden/ShapeFromBlur |
| C10 | derivation | n/a | n/a | 2026-07-16 | No URL; arithmetic and first-order error propagation only |

## Scope notes

- The browser-backed rows record the primary page's returned HTTP status; that fetch interface does not expose raw transfer bytes, so `n/e` is explicit rather than invented.
- Search discovery pages and search-engine transport URLs are not evidentiary sources and are not listed. Every primary URL quoted or relied upon in `REFUTATION.md`, plus failed primary/API probes that affected the audit, is logged above.
- Weight/checkpoint and GraspXL size checks used metadata or HEAD only. No GPU work and no large artifact download occurred.
