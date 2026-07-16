# Live URL checks

Fetch date: **2026-07-16** (America/Los_Angeles). `GET bytes` is the response body size observed after redirects. Binary assets were checked with redirect-following `HEAD`; their `bytes` values are `Content-Length`, and no checkpoint body was downloaded. CVF OpenAccess was successfully fetched by the browser extractor, but direct curl retries timed out; those rows state both observations rather than inventing byte counts.

## C1-C5 sources

| URL | Method | HTTP | Bytes | Observation |
|---|---|---:|---:|---|
| <https://export.arxiv.org/api/query?id_list=2506.01373> | GET | 200 | 2,478 | McByte arXiv metadata |
| <https://arxiv.org/pdf/2506.01373> | GET | 200 | 8,245,260 | McByte paper PDF |
| <https://api.github.com/repos/tstanczyk95/McByte> | GET | 200 | 5,741 | Repo metadata |
| <https://raw.githubusercontent.com/tstanczyk95/McByte/main/README.md> | GET | 200 | 7,994 | External detection CLI/readme |
| <https://raw.githubusercontent.com/tstanczyk95/McByte/main/INSTALLATION.md> | GET | 200 | 5,030 | SAM/Cutie checkpoints |
| <https://raw.githubusercontent.com/tstanczyk95/McByte/main/yolox/tracker/mcbyte_tracker.py> | GET | 200 | 28,067 | Gating constants |
| <https://raw.githubusercontent.com/tstanczyk95/McByte/main/LICENSE> | GET | 200 | 1,073 | MIT text |
| <https://export.arxiv.org/api/query?id_list=2607.08688> | GET | 200 | 2,698 | SAM-MT metadata |
| <https://arxiv.org/pdf/2607.08688> | GET | 200 | 9,729,130 | SAM-MT paper PDF |
| <https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md> | GET | 200 | 16,249 | SAM 2.1 speeds/license |
| <https://export.arxiv.org/api/query?id_list=2606.13033> | GET | 200 | 2,451 | SMP metadata/v3 date |
| <https://arxiv.org/pdf/2606.13033> | GET | 200 | 3,246,924 | SMP paper PDF |
| <https://api.github.com/search/repositories?q=%22Selective+Mask+Propagation%22+in%3Aname%2Cdescription%2Creadme> | GET | 200 | 13,714 | Search returned live implementation |
| <https://api.github.com/repos/holma91/selective-mask-propagation> | GET | 200 | 6,322 | SMP repo metadata |
| <https://github.com/holma91/selective-mask-propagation> | GET | 200 | 348,175 | Public code repository page |
| <https://raw.githubusercontent.com/holma91/selective-mask-propagation/main/README.md> | GET | 200 | 11,795 | Quickstart/reproduction |
| <https://api.github.com/repos/holma91/selective-mask-propagation/git/trees/main?recursive=1> | GET | 200 | 240,962 | Code/test/license tree |
| <https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md> | GET | 200 | 24,172 | Model/license tables |
| <https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/config.py> | GET | 200 | 44,316 | Large config |
| <https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/assets/model_weights.py> | GET | 200 | 16,110 | Weight URL map |
| <https://raw.githubusercontent.com/roboflow/rf-detr/develop/rfdetr/config.py> | GET | 404 | 14 | Initial stale-path probe; corrected to `src/` |
| <https://raw.githubusercontent.com/roboflow/rf-detr/develop/rfdetr/assets/model_weights.py> | GET | 404 | 14 | Initial stale-path probe; corrected to `src/` |
| <https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth> | HEAD | 200 | 135,954,129 | Live checkpoint; body not downloaded |
| <https://api.github.com/repos/roboflow/rf-detr/commits/69b12dbf8d40> | GET | 200 | 37,545 | Commit and patch |
| <https://api.github.com/repos/roboflow/rf-detr/releases/latest> | GET | 200 | 14,305 | Latest release 1.8.3 |
| <https://api.github.com/repos/roboflow/rf-detr/git/trees/develop?recursive=1> | GET | 200 | 117,689 | Official tree bounded search |
| <https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md> | GET | 200 | 27,369 | Current Objects365+COCO table |
| <https://api.github.com/repos/Peterande/storage/releases/tags/dfinev1.0> | GET | 200 | 42,360 | Both asset records |
| <https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth> | HEAD | 200 | 126,069,154 | Live legacy asset; body not downloaded |
| <https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth> | HEAD | 200 | 126,083,766 | Live current asset; body not downloaded |

## C6 sources and artifacts

| URL | Method | HTTP | Bytes | Observation |
|---|---|---:|---:|---|
| <https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md> | GET | 200 | 22,708 | Eight-size model zoo and links |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_HGNetv2_ATTO_COCO> | GET | 200 | 769 | HF model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO> | GET | 200 | 769 | HF model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_HGNetv2_PICO_COCO> | GET | 200 | 771 | HF model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_HGNetv2_N_COCO> | GET | 200 | 775 | HF model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_DINOv3_S_COCO> | GET | 200 | 772 | HF S model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_DINOv3_M_COCO> | GET | 200 | 774 | HF M model exists |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_DINOv3_L_COCO> | GET | 200 | 774 | HF L model exists |
| <https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO> | GET | 200 | 98,088 | Benchmark-spec L model page |
| <https://huggingface.co/api/models/Intellindust/DEIMv2_DINOv3_X_COCO> | GET | 200 | 775 | HF X model exists |
| <https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view?usp=sharing> | GET | 200 | 75,373 | Atto checkpoint view |
| <https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view?usp=sharing> | GET | 200 | 75,305 | Femto checkpoint view |
| <https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view?usp=sharing> | GET | 200 | 75,656 | Pico checkpoint view |
| <https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view?usp=sharing> | GET | 200 | 75,489 | N checkpoint view |
| <https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing> | GET | 200 | 75,362 | S checkpoint view |
| <https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=sharing> | GET | 200 | 75,372 | M checkpoint view |
| <https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view?usp=sharing> | GET | 200 | 75,355 | L checkpoint view |
| <https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view?usp=sharing> | GET | 200 | 75,085 | X checkpoint view |

## C7-C10 sources and artifacts

| URL | Method | HTTP | Bytes | Observation |
|---|---|---:|---:|---|
| <https://api.github.com/repos/Intellindust-AI-Lab/EdgeCrafter> | GET | 200 | 7,932 | Canonical repo metadata |
| <https://api.github.com/repos/capsule2077/edgecrafter> | GET | 200 | 5,915 | Asset-host repo metadata |
| <https://raw.githubusercontent.com/Intellindust-AI-Lab/EdgeCrafter/main/README.md> | GET | 200 | 8,351 | Claims, authors, artifact links |
| <https://raw.githubusercontent.com/Intellindust-AI-Lab/EdgeCrafter/main/LICENSE> | GET | 200 | 11,357 | Apache 2.0 text |
| <https://api.github.com/repos/capsule2077/edgecrafter/releases> | GET | 200 | 47,870 | 24 release assets |
| <https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth> | HEAD | 200 | 132,351,848 | Live ECDet-L; body not downloaded |
| <https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth> | HEAD | 200 | 136,096,162 | Live ECSeg-L; body not downloaded |
| <https://export.arxiv.org/api/query?id_list=2603.18739> | GET | 200 | 3,434 | EdgeCrafter metadata |
| <https://arxiv.org/pdf/2603.18739> | GET | 200 | 7,733,756 | EdgeCrafter paper PDF |
| <https://openaccess.thecvf.com/content/WACV2022/papers/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.pdf> | Browser GET / curl retry | 200 / 000 | extractor did not expose / 0 | Browser extractor returned 10-page PDF; curl timed out |
| <https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Vandeghen_Semi-Supervised_Training_To_Improve_Player_and_Ball_Detection_in_Soccer_CVPRW_2022_paper.pdf> | Browser GET / curl retry | 200 / 000 | extractor did not expose / 0 | Browser extractor returned 10-page PDF; curl timed out |
| <https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.pdf> | Browser GET / curl retry | 200 / 000 | extractor did not expose / 0 | Browser extractor returned 11-page PDF; curl timed out |
| <https://openaccess.thecvf.com/content/WACV2022/html/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.html> | Browser GET / curl retry | 200 / 000 | extractor did not expose / 0 | HTML evidence fetched; curl timed out |
| <https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Vandeghen_Semi-Supervised_Training_To_Improve_Player_and_Ball_Detection_in_Soccer_CVPRW_2022_paper.html> | GET | 200 | 6,271 | Paper landing page |
| <https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.html> | Browser GET / curl retry | 200 / 000 | extractor did not expose / 0 | HTML evidence fetched; curl timed out |
| <https://api.github.com/repos/rvandeghen/SST> | GET | 200 | 5,534 | SST repo metadata |
| <https://raw.githubusercontent.com/rvandeghen/SST/main/LICENSE> | GET | 200 | 1,523 | BSD-3-Clause text |
| <https://raw.githubusercontent.com/MCG-NJU/SportsMOT/main/README.md> | GET | 200 | 9,117 | Terms and CC designation |
| <https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/README.md> | GET | 200 | 9,700 | Dataset/annotation/code agreement |
| <https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/LICENSE> | GET | 200 | 1,066 | MIT code text |
| <https://zheng-lab-anu.github.io/Project/project_reid.html> | GET | 200 | 13,336 | Current Market-1501 page |
| <https://raw.githubusercontent.com/VlSomers/keypoint_promptable_reidentification/main/LICENSE> | GET | 200 | 18,440 | HL3 variant text |
| <https://huggingface.co/FudanCVL/SAM-MT/raw/main/README.md> | GET | 200 | 33 | HF card YAML license header |
| <https://huggingface.co/api/models/FudanCVL/SAM-MT> | GET | 200 | 569 | HF cardData license |
| <https://api.github.com/repos/FudanCVL/SAM-MT> | GET | 200 | 6,558 | Repo license metadata null |
| <https://api.github.com/repos/FudanCVL/SAM-MT/git/trees/main?recursive=1> | GET | 200 | 2,003,888 | No license-named file in tree |
| <https://www.ultralytics.com/license> | GET | 200 | 332,620 | AGPL/Enterprise terms page |
| <https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE> | GET | 200 | 11,357 | Apache 2.0 text |
| <https://raw.githubusercontent.com/ultralytics/ultralytics/main/README.md> | GET | 200 | 34,658 | YOLO26 tables and asset links |
| <https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo26.md> | GET | 200 | 24,252 | Variants and default-head semantics |
| <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt> | HEAD | 200 | 44,255,705 | Live medium detector; body not downloaded |
| <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt> | HEAD | 200 | 54,750,385 | Live medium segmenter; body not downloaded |
