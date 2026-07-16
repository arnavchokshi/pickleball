# Live URL checks — TRK survey B

Checked **2026-07-16 (America/Los_Angeles)**. One URL per line. `bytes` is the final HTTP `Content-Length` when the server supplied one; `unknown/chunked` means no final length header. Checkpoint endpoints used HEAD or a one-byte range and were not downloaded. `TIMEOUT` records a failed curl check rather than pretending reachability.

## Detection artifacts

- `HTTP 200; bytes unknown/chunked` — https://github.com/roboflow/rf-detr
- `HTTP 200; bytes 11,345` — https://raw.githubusercontent.com/roboflow/rf-detr/develop/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://github.com/roboflow/rf-detr/releases/tag/1.7.1
- `HTTP 206; 1 byte fetched; total bytes 404,992,918` — https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth
- `HTTP 200; bytes 135,954,129` — https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth
- `HTTP 200; bytes 145,055,866` — https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth
- `HTTP 200; bytes 152,713,150` — https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth
- `HTTP 200; bytes unknown/chunked` — https://github.com/Peterande/D-FINE
- `HTTP 200; bytes 11,357` — https://raw.githubusercontent.com/Peterande/D-FINE/master/LICENSE
- `HTTP 200; bytes 126,083,766` — https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth
- `HTTP 200; bytes unknown/chunked` — https://github.com/Intellindust-AI-Lab/DEIMv2
- `HTTP 200; bytes 11,564` — https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/LICENSE
- `HTTP 200; bytes 75,224 (Drive metadata page, weight not downloaded)` — https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing
- `TIMEOUT after 20 s; HTTP 000; bytes 0` — https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO
- `HTTP 200; bytes unknown/chunked` — https://github.com/clxia12/RT-DETRv3
- `HTTP 200; bytes 11,357` — https://raw.githubusercontent.com/clxia12/RT-DETRv3/main/LICENSE
- `HTTP 200; bytes 74,622 (Drive metadata page, weight not downloaded)` — https://drive.google.com/file/d/1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ/view?usp=drive_link
- `HTTP 200; bytes unknown/chunked` — https://docs.ultralytics.com/models/yolo26/
- `HTTP 200; bytes unknown/chunked` — https://github.com/ultralytics/ultralytics
- `HTTP 200; bytes 34,523` — https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE
- `HTTP 200; bytes 44,255,705` — https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/abs/2606.03748
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/abs/2012.06785
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/abs/1805.00123
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025/html/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.html
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Using_Diffusion_Priors_for_Video_Amodal_Segmentation_CVPR_2025_paper.html

## MOT and sports-tracking artifacts

- `HTTP 200; bytes 41,284` — https://arxiv.org/abs/2606.13033
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/html/2606.13033
- `HTTP 200; bytes 43,221` — https://arxiv.org/abs/2506.03335
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Khanna_SportMamba_Adaptive_Non-Linear_Multi-Object_Tracking_with_State_Space_Models_for_CVPRW_2025_paper.pdf
- `HTTP 200; bytes unknown/chunked` — https://github.com/sjc042/gta-link
- `HTTP 200; bytes 1,069` — https://raw.githubusercontent.com/sjc042/gta-link/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/papers/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.pdf
- `HTTP 200; bytes unknown/chunked` — https://github.com/TripleJoy/SAM2MOT
- `HTTP 200; bytes 11,357` — https://raw.githubusercontent.com/TripleJoy/SAM2MOT/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://github.com/ZabuZabuZabu/SAMIDARE
- `HTTP 404; bytes 14` — https://raw.githubusercontent.com/ZabuZabuZabu/SAMIDARE/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/html/2604.22162
- `HTTP 200; bytes unknown/chunked` — https://github.com/tstanczyk95/McByte
- `HTTP 200; bytes 1,073` — https://raw.githubusercontent.com/tstanczyk95/McByte/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/html/2506.01373
- `HTTP 200; bytes unknown/chunked` — https://github.com/bytedance/ColTrack
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/abs/2509.21715
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2026/html/Li_Occlusion-Aware_SORT_Observing_Occlusion_for_Robust_Multi-Object_Tracking_CVPR_2026_paper.html
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/WACV2026/html/Kim_Gated_Temporal_Fusion_Transformers_for_Robust_Multi-Object_Tracking_WACV_2026_paper.html
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/WACV2026/html/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.html
- `HTTP 200; bytes unknown/chunked` — https://github.com/maomao726/FieldTrack
- `HTTP 404; bytes 14` — https://raw.githubusercontent.com/maomao726/FieldTrack/master/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.html
- `HTTP 200; bytes unknown/chunked` — https://github.com/SoccerNet/sn-tracking

## ReID artifacts

- `HTTP 200; bytes unknown/chunked` — https://github.com/mk-minchul/sapiensid
- `HTTP 200; bytes 19,342` — https://raw.githubusercontent.com/mk-minchul/sapiensid/main/LICENSE
- `HTTP 200; bytes 75,257 (Drive metadata page, weight not downloaded)` — https://drive.google.com/file/d/18d8Ogw60zxnaIIjSb99Y8oLvzCEn_L-Z/view?usp=sharing
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025/html/Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.html
- `HTTP 200; bytes unknown/chunked` — https://github.com/yuanc3/Pose2ID
- `HTTP 200; bytes 1,063` — https://raw.githubusercontent.com/yuanc3/Pose2ID/main/LICENSE
- `TIMEOUT after 12 s; HTTP 000; bytes 0; search index independently exposed live model card` — https://huggingface.co/yuanc3/Pose2ID
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2025/html/Yuan_From_Poses_to_Identity_Training-Free_Person_Re-Identification_via_Feature_Centralization_CVPR_2025_paper.html
- `HTTP 200; bytes unknown/chunked` — https://github.com/VlSomers/keypoint_promptable_reidentification
- `HTTP 200; bytes 18,440` — https://raw.githubusercontent.com/VlSomers/keypoint_promptable_reidentification/main/LICENSE
- `HTTP 200; bytes 75,347 (Drive metadata page, weight not downloaded)` — https://drive.google.com/file/d/1B1v11Yw56AIxxzDHnnymi4NPkNRDYkvJ/view?usp=sharing
- `HTTP 200; bytes unknown/chunked` — https://github.com/KaiyangZhou/deep-person-reid
- `HTTP 200; bytes 1,069` — https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/LICENSE
- `TIMEOUT after 12 s; HTTP 000; bytes 0; search index independently exposed live model card` — https://huggingface.co/kaiyangzhou/osnet
- `HTTP 200; bytes unknown/chunked` — https://zheng-lab-anu.github.io/Datasets.html
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/abs/2206.02373
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.html

## Mask/VOS artifacts

- `HTTP 200; bytes unknown/chunked` — https://github.com/facebookresearch/sam2
- `HTTP 200; bytes 11,357` — https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE
- `HTTP 200; bytes 323,606,802` — https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
- `HTTP 200; bytes unknown/chunked` — https://github.com/hkchengrex/Cutie
- `HTTP 200; bytes 1,069` — https://raw.githubusercontent.com/hkchengrex/Cutie/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://github.com/siyuanliii/masa
- `HTTP 200; bytes 11,342` — https://raw.githubusercontent.com/siyuanliii/masa/main/LICENSE
- `TIMEOUT after 20 s; HTTP 000; bytes 0` — https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth
- `HTTP 200; bytes unknown/chunked` — https://github.com/FudanCVL/SAM-MT
- `HTTP 404; bytes 14` — https://raw.githubusercontent.com/FudanCVL/SAM-MT/main/LICENSE
- `TIMEOUT after 12 s; HTTP 000; bytes 0; primary model-card search fetch returned HTTP 200` — https://huggingface.co/FudanCVL/SAM-MT
- `HTTP 200; bytes 33` — https://huggingface.co/FudanCVL/SAM-MT/raw/main/README.md
- `HTTP 200; bytes unknown/chunked` — https://arxiv.org/html/2607.08688

## Datasets and domain-adaptation artifacts

- `HTTP 200; bytes unknown/chunked` — https://github.com/MCG-NJU/SportsMOT
- `HTTP 200; bytes unknown/chunked` — https://github.com/DanceTrack/DanceTrack
- `HTTP 200; bytes 1,066` — https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/LICENSE
- `HTTP 200; bytes 0/chunked` — https://www.soccer-net.org/faq
- `HTTP 200; bytes unknown/chunked` — https://github.com/SoccerNet/sn-reid
- `HTTP 200; bytes 1,069` — https://raw.githubusercontent.com/SoccerNet/sn-reid/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://github.com/rvandeghen/SST
- `HTTP 200; bytes 1,523` — https://raw.githubusercontent.com/rvandeghen/SST/main/LICENSE
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/WACV2022/html/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.html
- `HTTP 200; bytes unknown/chunked` — https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Vandeghen_Semi-Supervised_Training_To_Improve_Player_and_Ball_Detection_in_Soccer_CVPRW_2022_paper.html
- `HTTP 200; bytes 6,355` — https://raw.githubusercontent.com/roboflow/trackers/main/README.md

## Interpretation

- [INFERENCE] A `200`/`206` result is reachability only; it is not a checksum, reproducibility, malware, license, or model-quality verification.
- [INFERENCE] The three `404` LICENSE endpoints are affirmative evidence that no root LICENSE exists at that exact default-branch path on the check date; they do not prove the author supplied no terms elsewhere.
- [INFERENCE] Hugging Face timeouts are retained as failed live checks; they are not rewritten as successes based solely on search indexing.
