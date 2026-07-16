# Live checks — TRK survey lane A

Fetch date for every entry: **2026-07-16**. No full model was downloaded. `HEAD 200` means `curl -I/-IL`; HF entries use the concrete file listing when the binary redirect endpoint timed out in this environment. GitHub/HF rendered listings are fetched primary pages and include the displayed remote size where available.

## Detection and segmentation

| URL | Result |
|---|---|
| https://github.com/roboflow/rf-detr | HEAD 200; official code/README/license fetched. |
| https://github.com/roboflow/rf-detr#benchmarks | Rendered/fetched with the repository page; official aggregate detection/segmentation tables. |
| https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/assets/model_weights.py | GET fetched; exact official artifact registry. |
| https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth | HEAD 200; binary not downloaded. |
| https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth | HEAD 200; RF-DETR-N binary not downloaded. |
| https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth | HEAD 200; RF-DETR-S binary not downloaded. |
| https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth | HEAD 200; RF-DETR-M binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth | HEAD 200; RF-DETR-Seg-N binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth | HEAD 200; RF-DETR-Seg-S binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth | HEAD 200; RF-DETR-Seg-M binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth | HEAD 200; binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth | HEAD 200; RF-DETR-Seg-XL binary not downloaded. |
| https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth | HEAD 200; RF-DETR-Seg-2XL binary not downloaded. |
| https://github.com/roboflow/rf-detr/commit/69b12dbf8d40 | Rendered/fetched; July 15 `scale_jitter` commit, 10 files changed. |
| https://github.com/roboflow/rf-detr/issues/674 | Rendered/fetched; CrowdHuman fine-tune issue, not benchmark evidence. |
| https://github.com/Intellindust-AI-Lab/DEIMv2 | HEAD 200; official code/README/license fetched. |
| https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO/tree/main | Rendered/fetched; `model.safetensors` listed at 39.4 MB. |
| https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO/resolve/main/model.safetensors | `curl -IL` timed out with HTTP 000/0 bytes; concrete HF listing above independently fetched and shows the file. |
| https://github.com/Peterande/D-FINE | HEAD 200; official code/README/license fetched. |
| https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth | HEAD-follow 200 at GitHub release-assets; binary not downloaded. |
| https://github.com/capsule2077/edgecrafter | HEAD 200; official code/README fetched. |
| https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth | HEAD-follow 200 at GitHub release-assets; binary not downloaded. |
| https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth | HEAD-follow 200 at GitHub release-assets; binary not downloaded. |
| https://github.com/ultralytics/ultralytics | HEAD 200; official code/license surface fetched. |
| https://docs.ultralytics.com/models/yolo26/ | Rendered/fetched; official YOLO26 model page. |
| https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt | HEAD-follow 200 at GitHub release-assets; binary not downloaded. |
| https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt | HEAD-follow 200 at GitHub release-assets; binary not downloaded. |
| https://docs.ultralytics.com/models/yoloe/ | Rendered/fetched; official YOLOE/YOLOE-26 model page and artifact names. |

## Video mask sources

| URL | Result |
|---|---|
| https://github.com/facebookresearch/sam2 | HEAD 200; official code/README/license/checkpoint script fetched. |
| https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt | HEAD 200; binary not downloaded. |
| https://github.com/yformer/EfficientTAM | HEAD 200; official code/README/license/checkpoint script fetched. |
| https://huggingface.co/yunyangx/efficient-track-anything/tree/main | Rendered/fetched; `efficienttam_s.pt` listed at 136 MB. |
| https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s.pt | Direct binary endpoint not downloaded; concrete listing above fetched with size and pickle scan. |
| https://github.com/FudanCVL/SAM-MT | HEAD 200; official code/README fetched; commits page shows initial July 9 and arXiv-link July 10 commits. |
| https://huggingface.co/FudanCVL/SAM-MT/blob/main/README.md | Rendered/fetched; model-card license is CC-BY-NC-SA-4.0. |
| https://huggingface.co/FudanCVL/SAM-MT/resolve/main/checkpoints/sam-mt.pt | Direct binary endpoint not downloaded; official repo links the concrete `checkpoints` listing and HF commit history shows the initial checkpoint. |

## ReID

| URL | Result |
|---|---|
| https://github.com/kaiyangzhou/deep-person-reid | HEAD 200; official code/README/license fetched. |
| https://huggingface.co/kaiyangzhou/osnet/tree/main | Rendered/fetched; 149 MB repository and exact OSNet files listed. |
| https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth | Direct binary endpoint not downloaded; concrete listing fetched and displays 17.3 MB. |
| https://www.diaochapai.com/survey/a61751ca-4210-4df1-a5bb-1e7a71b5262b | Rendered/fetched; official Market-1501 download terms say academic use only/no redistribution. |
| https://github.com/tinyvision/SOLIDER | HEAD 200; official code/README/license fetched. |
| https://drive.google.com/file/d/12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8/view?usp=share_link | HEAD-follow 200; official SOLIDER checkpoint page, binary not downloaded. |
| https://github.com/Syliz517/CLIP-ReID | HEAD 200; official code/README/license fetched. |
| https://drive.google.com/file/d/1s-nZMp-LHG0h4dFwvyP_YNBLTijLcrb0/view?usp=share_link | HEAD-follow 200; official CLIP-ReID checkpoint page, binary not downloaded. |
| https://github.com/damo-cv/TransReID | HEAD 200; official code/README/license fetched. |
| https://drive.google.com/file/d/1iF5JNPw9xi-rLY3Ri9EY-PFAkK6Vg_Pf/view?usp=sharing | HEAD-follow 200; official TransReID model page, binary not downloaded. |
| https://github.com/VlSomers/keypoint_promptable_reidentification | HEAD 200; official code/README/license fetched. |
| https://drive.google.com/file/d/1B1v11Yw56AIxxzDHnnymi4NPkNRDYkvJ/view?usp=sharing | HEAD-follow 200; official KPR artifact page, binary not downloaded. |
| https://raw.githubusercontent.com/VlSomers/keypoint_promptable_reidentification/main/LICENSE | GET fetched; Hippocratic License version 3.0 text. |
| https://github.com/SoccerNet/sn-reid | HEAD 200; official toolkit/dataset README and MIT code license fetched. |

## Domain datasets

| URL | Result |
|---|---|
| https://github.com/MCG-NJU/SportsMOT | HEAD 200; official README, dataset statistics, target definition, and CC BY-NC 4.0 terms fetched. |
| https://github.com/DanceTrack/DanceTrack | HEAD 200; official README and agreement fetched (MIT code, CC BY 4.0 annotations, noncommercial dataset). |
| https://github.com/SoccerNet/sn-tracking | HEAD 200; official tracking dataset/toolkit README fetched; no dataset license file found at the checked repo root. |
| https://github.com/SoccerNet/sn-reid | HEAD 200; official 340,993-crop dataset/toolkit README fetched. |

## Mask cue and learned association

| URL | Result |
|---|---|
| https://github.com/tstanczyk95/McByte | HEAD 200; official README/MIT license fetched; commit history shows last activity 2025-07-22 and repo shows no releases. |
| https://github.com/hkchengrex/Cutie | HEAD 200; official README/MIT license/checkpoint instructions fetched. |
| https://github.com/siyuanliii/masa | HEAD 200; official README/license fetched. |
| https://huggingface.co/dereksiyuanli/masa/tree/main | Rendered/fetched; Apache-2.0 card, `masa_r50.pth` listed at 528 MB. |
| https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth | Direct binary endpoint not downloaded; concrete listing above fetched with size and security scan. |
| https://github.com/TrackingLaboratory/CAMELTrack | HEAD 200; official README/Apache-2.0 license fetched. |
| https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_global.ckpt | Rendered/fetched; 526 MB, SHA256 `978d7d20cfb7492c928af5e18f49cd59fa271c924b23fe1083b38e8cdbda3bc7`, Apache-2.0 card, four training datasets listed. |
| https://huggingface.co/trackinglaboratory/CAMELTrack/resolve/main/camel_bbox_app_kps_global.ckpt | Direct binary endpoint not downloaded; concrete blob page above fetched with byte size and SHA256. |

## Primary papers fetched

| URL | Result |
|---|---|
| https://arxiv.org/abs/2511.09554 | Rendered/fetched; RF-DETR paper abstract. |
| https://arxiv.org/abs/2509.20787 | Rendered/fetched; DEIMv2 paper abstract. |
| https://arxiv.org/abs/2410.13842 | Rendered/fetched; D-FINE paper abstract. |
| https://arxiv.org/abs/2606.03748 | Rendered/fetched; YOLO26 paper abstract. |
| https://arxiv.org/abs/2607.08688 | Rendered/fetched; SAM-MT paper abstract. |
| https://arxiv.org/abs/2304.05170 | Rendered/fetched; SportsMOT paper abstract. |
| https://arxiv.org/abs/2407.18112 | Rendered/fetched; KPR paper abstract. |
| https://arxiv.org/abs/2303.11855 | Rendered/fetched; CLIP-ReIdent paper abstract. |
| https://arxiv.org/abs/2401.09942 | Rendered/fetched; PRTReID paper abstract. |
| https://arxiv.org/abs/2506.01373 | Rendered/fetched; McByte paper abstract. |
| https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.pdf | PDF fetched/parsed; accepted McByte workshop paper. |
| https://arxiv.org/abs/2406.04221 | Rendered/fetched; MASA paper abstract. |
| https://arxiv.org/abs/2505.01257 | Rendered/fetched; CAMELTrack paper abstract. |
