# Live checks — RKT survey A

Checked: **2026-07-16** (America/Los_Angeles). No checkpoint was downloaded. `bytes` is the HTTP response body fetched for repository/license/landing pages, not repository or model size. For Hugging Face pages, the browser fetcher returned HTTP 200 but did not expose transfer bytes; the remote asset size shown by the primary page is recorded. A direct file-page timeout is recorded rather than hidden.

## Ranked code, checkpoint, and dataset artifacts

- `https://github.com/OrcustD/RacketVision` — HTTP 200; 349,596 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/OrcustD/RacketVision/main/LICENSE` — HTTP 200; 1,069 bytes; MIT; 2026-07-16.
- `https://huggingface.co/datasets/linfeng302/RacketVision` — browser HTTP 200; transfer bytes n/a; primary page reports 7.54 GB; MIT label; 2026-07-16. Direct curl also timed out with HTTP 000 / 0 bytes.
- `https://huggingface.co/datasets/linfeng302/RacketVision/tree/main/info` — browser HTTP 200; transfer bytes n/a; remote folder 73.7 MB; lists current COCO annotation files; 2026-07-16.
- `https://huggingface.co/linfeng302/RacketVision-Models/tree/main/checkpoints` — browser HTTP 200; transfer bytes n/a; remote folder 617 MB; lists `epoch_300.pth` 411 MB and `best_PCK_epoch_90.pth` 107 MB; no model card; six files marked unsafe pickle; 2026-07-16.
- `https://huggingface.co/linfeng302/RacketVision-Models/blob/main/checkpoints/epoch_300.pth` — direct browser/curl file page timed out; HTTP 000 / 0 bytes; parent tree above independently lists 411 MB artifact; 2026-07-16.
- `https://huggingface.co/linfeng302/RacketVision-Models/blob/main/checkpoints/best_PCK_epoch_90.pth` — direct browser/curl file page timed out; HTTP 000 / 0 bytes; parent tree above independently lists 107 MB artifact; 2026-07-16.

- `https://github.com/megapose6d/megapose6d` — HTTP 200; 413,163 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/megapose6d/megapose6d/master/LICENSE` — HTTP 200; 605 bytes; Apache-2.0; 2026-07-16.
- `https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/` — HTTP 200; 1,437 bytes; official model directory listing; 2026-07-16.

- `https://github.com/NVlabs/FoundationPose` — HTTP 200; 373,545 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE` — HTTP 200; 4,352 bytes; NVIDIA Source Code License, non-commercial research/evaluation; 2026-07-16.
- `https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing` — HTTP 200; 300,699 bytes; official weights folder linked by README; 2026-07-16.

- `https://github.com/facebookresearch/foundpose` — HTTP 200; 335,877 bytes; official code; README says coarse stage only; 2026-07-16.
- `https://raw.githubusercontent.com/facebookresearch/foundpose/main/LICENSE` — HTTP 200; 19,334 bytes; CC BY-NC 4.0; 2026-07-16.
- FoundPose checkpoint — **none published in the official repository**; method is training-free and repo exposes no learned FoundPose weight artifact; 2026-07-16.

- `https://github.com/nv-nguyen/gigapose` — HTTP 200; 354,184 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/nv-nguyen/gigapose/main/LICENSE` — HTTP 200; 1,074 bytes; MIT; 2026-07-16.
- `https://huggingface.co/datasets/nv-nguyen/gigaPose/blob/main/gigaPose_v1.ckpt` — browser HTTP 200; transfer bytes n/a; remote checkpoint 3.81 GB, SHA256 `0f60a23b03ddc41d2135c916ed1e66fb16f814f612dbde0305ae5a2c0f45c932`; 2026-07-16. Direct Python/curl fetch timed out before data; no checkpoint content downloaded.

- `https://github.com/JiehongLin/SAM-6D` — HTTP 200; 276,202 bytes; official code; 2026-07-16.
- `https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing` — HTTP 200; 75,208 bytes; official `sam-6d-pem-base.pth` landing page referenced by downloader; 2026-07-16.
- `https://raw.githubusercontent.com/JiehongLin/SAM-6D/main/LICENSE` — HTTP 404; 14 bytes; no root license file; 2026-07-16.
- `https://api.github.com/repos/JiehongLin/SAM-6D` — HTTP 200; 5,558 bytes; GitHub reports no detected license/SPDX; 2026-07-16.

- `https://github.com/Yoonwoo-Ha/RACE-6D` — HTTP 200; 343,737 bytes; official code; no GitHub releases/checkpoints; 2026-07-16.
- `https://raw.githubusercontent.com/Yoonwoo-Ha/RACE-6D/main/LICENSE` — HTTP 200; 11,357 bytes; Apache-2.0; 2026-07-16.
- RACE-6D checkpoint — **not found** in official README/releases; README accepts a local `path/to/checkpoint.pth`; 2026-07-16.

- `https://github.com/Marwan99/kv_tracker` — HTTP 200; 297,470 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/Marwan99/kv_tracker/main/LICENSE` — HTTP 200; 11,820 bytes; custom Imperial College non-commercial internal/academic research license; 2026-07-16.
- KV-Tracker own checkpoint — **not published as a standalone artifact** in the official README; setup pulls third-party SAM2.1/model dependencies; 2026-07-16.

- `https://github.com/CNJianLiu/SinRef-6D` — HTTP 200; 358,247 bytes; official code; README advertises pretrained weights; 2026-07-16.
- `https://raw.githubusercontent.com/CNJianLiu/SinRef-6D/main/LICENSE` — HTTP 200; 1,065 bytes; MIT; 2026-07-16.

- `https://github.com/tobycollins/IPPE` — HTTP 200; 285,015 bytes; reference implementation; 2026-07-16.
- `https://api.github.com/repos/tobycollins/IPPE` — HTTP 200; 5,866 bytes; GitHub SPDX `BSD-3-Clause`; 2026-07-16.

- `https://github.com/wenbowen123/BundleTrack` — HTTP 200; 329,571 bytes; official code; 2026-07-16.
- `https://github.com/NVlabs/BundleSDF` — HTTP 200; 332,910 bytes; official code; 2026-07-16.

## Synthetic-data and blur artifacts

- `https://github.com/DLR-RM/BlenderProc` — HTTP 200; 381,016 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/LICENSE` — HTTP 200; 35,147 bytes; GPL-3.0; 2026-07-16.
- `https://github.com/google-research/kubric` — HTTP 200; 327,744 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/google-research/kubric/main/LICENSE` — HTTP 200; 11,358 bytes; Apache-2.0; 2026-07-16.
- `https://github.com/isaac-sim/IsaacSim` — HTTP 200; 468,184 bytes; official code; 2026-07-16.
- `https://raw.githubusercontent.com/isaac-sim/IsaacSim/main/LICENSE` — HTTP 200; 10,797 bytes; Apache-2.0 code plus explicit additional terms for Omniverse Kit/dependencies/assets; 2026-07-16.
- `https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html` — HTTP 200; 489,371 bytes; official motion-blur/object-SDG tutorial; 2026-07-16.
- `https://github.com/JaehaKim97/BlurHand_RELEASE` — HTTP 200; 266,351 bytes; official blur-aware hand-pose code; root `LICENSE` URL returned 404 when inspected; 2026-07-16.
- `https://zhongcl-thu.github.io/rock/` — HTTP 200; 9,916 bytes; official ROCK project/code landing page; 2026-07-16.

## Paper-only candidates / negative release checks

- `https://openaccess.thecvf.com/content/CVPR2025/html/Deng_Pos3R_6D_Pose_Estimation_for_Unseen_Objects_Made_Easy_CVPR_2025_paper.html` — browser HTTP 200; transfer bytes n/a; related material lists paper/supplement, no official code/weights; 2026-07-16.
- `https://openaccess.thecvf.com/content/CVPR2025/html/Kim_RefPose_Leveraging_Reference_Geometric_Correspondences_for_Accurate_6D_Pose_Estimation_CVPR_2025_paper.html` — browser HTTP 200; transfer bytes n/a; no official code/weights linked; 2026-07-16.
- `https://openaccess.thecvf.com/content/ICCV2025/html/Huang_RayPose_Ray_Bundling_Diffusion_for_Template_Views_in_Unseen_6D_ICCV_2025_paper.html` — browser HTTP 200; transfer bytes n/a; related material lists paper/supplement, no official code/weights; 2026-07-16.
- `https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Rethinking_Correspondence-based_Category-Level_Object_Pose_Estimation_CVPR_2025_paper.html` — browser HTTP 200; transfer bytes n/a; no official SpotPose code/weights linked; 2026-07-16.

## Primary-paper reachability spot checks

- `https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324` — HTTP 200; 7,405,673 bytes; RacketVision AAAI-26 paper; 2026-07-16.
- CVF PDF endpoints for Pos3R, RefPose, RayPose, RACE-6D, and SpotPose — browser HTTP 200 and content parsed; direct curl attempts hit SSL/connect timeouts and returned HTTP 000 / 0 bytes on 2026-07-16. The HTML primary pages above remained reachable.
