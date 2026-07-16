# TRK refutation lane — second-vote primary-source verification

Fetch date for all checks: **2026-07-16** (America/Los_Angeles). This was an independent fetch pass; the two survey directories named in the prompt were not read. Negative findings are stated as bounded searches, not proofs of universal absence. These published results are candidate-screening evidence only; they are not pickleball accuracy evidence. **VERIFIED=0 remains binding.**

## Verdict summary

| Claim | Verdict | Material result |
|---|---|---|
| C1 | **CONFIRM** | McByte's paper and repository support the full bundle. |
| C2 | **CONFIRM** | 17.8/12.4 FPS are in SAM-MT Table 3; Meta's 64.1 FPS uses a different A100 protocol. |
| C3 | **REFUTE** | The numbers hold, but code is now publicly released under MIT. |
| C4 | **PARTIAL** | (a)-(e) hold; (f) was not found but is not provable as an absolute. |
| C5 | **PARTIAL** | Both filenames are live; the current advertised 57.3-AP row uses `_e25`. |
| C6 | **PARTIAL** | Eight sizes exist; the claimed L metrics hold, but the `_S_COCO` HF artifact is S, not L. |
| C7 | **PARTIAL** | Real paper/code/assets and metrics; not organizationally independent of DEIMv2. |
| C8 | **PARTIAL** | All papers/code exist; C8(a)'s 100 labels are domain labels on predicted boxes. |
| C9 | **PARTIAL** | Six subclaims hold; the current Market-1501 page does not state the alleged restrictions. |
| C10 | **CONFIRM** | All n/s/m/l/x detect+segment variants and the medium release asset are live. |

## C1 — McByte runtime, implementation, constants, and license

**Verdict: CONFIRM. Fetched: 2026-07-16.**

- The [paper PDF](https://arxiv.org/pdf/2506.01373) Appendix D says the speed “oscillates around 3-5 FPS” on a single A100. At 60 source FPS, `60/5=12` and `60/3=20`, so **12-20x source duration is arithmetic derived from the paper**, not a separately measured paper result.
- The paper implementation section gives `mc=0.6`, `mb=0.9`, and `mf=0.05`, and says it uses SAM with original ViT-B weights. The [tracker source](https://raw.githubusercontent.com/tstanczyk95/McByte/main/yolox/tracker/mcbyte_tracker.py) independently contains `MIN_MASK_AVG_CONF = 0.6`, `MIN_MM1 = 0.9`, and `MIN_MM2 = 0.05`.
- The [repository README](https://raw.githubusercontent.com/tstanczyk95/McByte/main/README.md) exposes `--det_path` and describes oracle/custom detections. [INSTALLATION.md](https://raw.githubusercontent.com/tstanczyk95/McByte/main/INSTALLATION.md) pins Cutie `cutie-base-mega.pth` and SAM `sam_vit_b_01ec64.pth`.
- The [GitHub repository API](https://api.github.com/repos/tstanczyk95/McByte) reports `license.spdx_id=MIT` and `pushed_at=2025-07-22T08:03:27Z`; the [license text](https://raw.githubusercontent.com/tstanczyk95/McByte/main/LICENSE) is MIT.
- The [arXiv record](https://export.arxiv.org/api/query?id_list=2506.01373) identifies the cited McByte paper as arXiv:2506.01373.

**Corrected fact:** No correction. Preserve the 12-20x number as a calculation from 3-5 FPS, not an independently benchmarked range.

## C2 — SAM 2.1 Base+ multi-target throughput

**Verdict: CONFIRM, with a comparability caveat. Fetched: 2026-07-16.**

- [SAM-MT arXiv record](https://export.arxiv.org/api/query?id_list=2607.08688) is arXiv:2607.08688, submitted 2026-07-09.
- [SAM-MT PDF](https://arxiv.org/pdf/2607.08688) Table 3 reports official SAM2.1-B+ at **17.8 FPS for 3 objects** and **12.4 FPS for 5 objects** on one NVIDIA RTX A6000 48GB; the table uses 20 synthetic 1024p sequences.
- Meta's [SAM 2 README](https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md) reports SAM 2.1 base+ at **64.1 FPS**, and says speed was measured on A100 with Torch 2.5.1/CUDA 12.4 and compilation.

**Corrected fact:** The values are real, but 64.1 vs 17.8/12.4 is not a controlled scaling curve: hardware, target count, and benchmark protocol differ.

## C3 — Selective Mask Propagation

**Verdict: REFUTE. Fetched: 2026-07-16.** The result and ablation claims confirm; the no-code/no-weights claim is false as of the fetch date.

- The [arXiv record](https://export.arxiv.org/api/query?id_list=2606.13033) shows v3 updated **2026-07-09**. The [paper PDF](https://arxiv.org/pdf/2606.13033) Table 3 reports SportsMOT test **87.2 HOTA** for SAM3-Deep-EIoU; the abstract/prose makes clear the sports result includes GTA.
- Table 4 on SportsMOT val reports Deep-EIoU 78.9, `+GTA` 86.7 (**+7.8**), selective Deep-EIoU 80.4 (**+1.5**), and selective+GTA 88.5. The paper also summarizes GTA gains of +7.8/+8.1 versus selective-mask gains of +1.5/+1.8.
- A GitHub [repository search](https://api.github.com/search/repositories?q=%22Selective+Mask+Propagation%22+in%3Aname%2Cdescription%2Creadme) finds [holma91/selective-mask-propagation](https://api.github.com/repos/holma91/selective-mask-propagation). Its README gives a working clone/quickstart and reproduction commands; its recursive tree contains implementation modules, scripts, tests, and an MIT license. The repo was pushed **2026-07-14**, five days after arXiv v3.

**Corrected fact:** Public MIT code exists at <https://github.com/holma91/selective-mask-propagation>. It is training-free and downloads upstream SAM/tracker assets, so there may be no novel learned weight to release; “NO released code or weights” must not appear in the benchmark spec.

## C4 — RF-DETR bundle

**Verdict: PARTIAL. Fetched: 2026-07-16.** Subclaims (a)-(e) confirm. Subclaim (f) remains a bounded negative finding, not a universal fact.

- **Licenses:** The official [develop README](https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md) tables designate detection N/S/M/L as Apache 2.0 and detection XL/2XL as PML 1.0. It says the latter require the `rfdetr_plus` package. All segmentation N/S/M/L/XL/2XL rows are designated Apache 2.0.
- **Large weight:** [`RFDETRLargeConfig`](https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/config.py) sets `pretrain_weights="rf-detr-large-2026.pth"`; the [weight map](https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/assets/model_weights.py) resolves it to <https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth>. HEAD returned **200**, `Content-Length: 135954129`.
- **Scale-jitter commit:** [69b12dbf8d40](https://api.github.com/repos/roboflow/rf-detr/commits/69b12dbf8d40) exists as full SHA `69b12dbf8d40a739ff22a8463f682fa4a066c2ba`, authored 2026-07-15. Its patch adds `TrainConfig.scale_jitter=True`; `False` returns the direct-resize branch and tests assert `RandomSizedCrop` is absent. The docs say border annotations are then never clipped.
- **Latest release:** The [latest-release API](https://api.github.com/repos/roboflow/rf-detr/releases/latest) returns tag **`1.8.3`**, name “RF-DETR v1.8.3,” published **2026-06-29**. `1.7.1` is not latest.
- **Person/crowd/MOT detector benchmarks:** The official README and [develop tree](https://api.github.com/repos/roboflow/rf-detr/git/trees/develop?recursive=1) publish whole-COCO/RF100-VL/model-size results. Targeted official-domain/repository searches found no Roboflow-published COCO-person-only, CrowdHuman, or MOT20-det table. A CrowdHuman GitHub issue is user fine-tuning, not an official benchmark.

**Corrected fact:** (e) is 1.8.3. For (f), write **“no official result located in the RF-DETR repo/docs/Roboflow search on 2026-07-16”**, not “none exists anywhere.” Also distinguish a model's designated checkpoint license from blanket claims about every package component or downstream use.

## C5 — D-FINE Objects365→COCO L checkpoint

**Verdict: PARTIAL. Fetched: 2026-07-16.** Both artifacts exist, but only one is the current advertised row.

- The current [D-FINE README](https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md) Objects365+COCO table links **`dfine_l_obj2coco_e25.pth`** and claims **57.3 AP**, 31M parameters, 8.07 ms.
- The official [storage release API](https://api.github.com/repos/Peterande/storage/releases/tags/dfinev1.0) lists both `dfine_l_obj2coco.pth` (126,069,154 bytes) and `dfine_l_obj2coco_e25.pth` (126,083,766 bytes).
- HEAD returned **200** for both:
  - Legacy live asset: <https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth>
  - Current benchmark asset: <https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth>

**Corrected fact / spec pin:** Use `dfine_l_obj2coco_e25.pth`, the exact current URL above, **HTTP 200**, claimed **57.3 AP**. Record the non-e25 asset as a live legacy/ambiguous release asset, not the current table target.

## C6 — DEIMv2 released variants and artifact mapping

**Verdict: PARTIAL. Fetched: 2026-07-16.** The L, S, and M numbers quoted in the claim are correct, but the proposed L→`DEIMv2_DINOv3_S_COCO` mapping is wrong.

The official [DEIMv2 model zoo](https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md) reports the following COCO results and links. Each linked HF model API and Google Drive view returned 200.

| Size | AP | Params | Latency | Actual backbone | Exact HF artifact | Drive checkpoint ID |
|---|---:|---:|---:|---|---|---|
| Atto | 23.8 | 0.5M | 1.10 ms | HGNetv2 | `DEIMv2_HGNetv2_ATTO_COCO` | `18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc` |
| Femto | 31.0 | 1.0M | 1.45 ms | HGNetv2 | `DEIMv2_HGNetv2_FEMTO_COCO` | `16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds` |
| Pico | 38.5 | 1.5M | 2.13 ms | HGNetv2 | `DEIMv2_HGNetv2_PICO_COCO` | `1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT` |
| N | 43.0 | 3.6M | 2.32 ms | HGNetv2 | `DEIMv2_HGNetv2_N_COCO` | `1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K` |
| S | 50.9 | 9.7M | 5.78 ms | ViT-Tiny distilled from DINOv3-S | `DEIMv2_DINOv3_S_COCO` | `1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL` |
| M | 53.0 | 18.1M | 8.80 ms | ViT-Tiny+ distilled from DINOv3-S | `DEIMv2_DINOv3_M_COCO` | `1nPKDHrotusQ748O1cQXJfi5wdShq6bKp` |
| L | 56.0 | 32.2M | 10.47 ms | DINOv3-S (`dinov3_vits16`) | `DEIMv2_DINOv3_L_COCO` | `1dRJfVHr9HtpdvaHlnQP460yPVHynMray` |
| X | 57.8 | 50.3M | 13.75 ms | DINOv3-S+ (`dinov3_vits16plus`) | `DEIMv2_DINOv3_X_COCO` | `1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7` |

**Corrected fact / spec pins:** S and M use distilled ViT-Tiny variants despite `DINOv3` in their artifact names. L uses direct DINOv3-S and must pin <https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO>, never the `_S_COCO` artifact. The model zoo also exposes live Drive checkpoints for all eight sizes.

## C7 — EdgeCrafter existence, metrics, license, and independence

**Verdict: PARTIAL. Fetched: 2026-07-16.** It is a real released candidate, but it is not organizationally independent of DEIMv2.

- The canonical [Intellindust-AI-Lab/EdgeCrafter repository API](https://api.github.com/repos/Intellindust-AI-Lab/EdgeCrafter) reports a live, non-fork, Apache-2.0 repository. The `capsule2077/edgecrafter` [repository](https://api.github.com/repos/capsule2077/edgecrafter) also exists and hosts the release assets, but the lab repo is the canonical code URL.
- The official [README](https://raw.githubusercontent.com/Intellindust-AI-Lab/EdgeCrafter/main/README.md) claims, under T4 FP16 TensorRT 10.6 batch-1 at 640: **ECDet-L 57.0 box AP / 10.49 ms** and **ECSeg-L 47.1 mask AP / 12.56 ms**. The [Apache-2.0 text](https://raw.githubusercontent.com/Intellindust-AI-Lab/EdgeCrafter/main/LICENSE) is present.
- The [release API](https://api.github.com/repos/capsule2077/edgecrafter/releases) reports tag `edgecrafterv1` and 24 assets. HEAD returned 200 for [ECDet-L](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth) (132,351,848 bytes) and [ECSeg-L](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth) (136,096,162 bytes).
- The [paper record](https://export.arxiv.org/api/query?id_list=2603.18739) and [PDF](https://arxiv.org/pdf/2603.18739) describe a compact ViT backbone and edge encoder-decoder, so this is more than a renamed checkpoint. However, EdgeCrafter and DEIMv2 are both Intellindust-AI-Lab projects, share Longfei Liu and other authors, and DEIMv2's own README announces EdgeCrafter as its latest work. EdgeCrafter also acknowledges D-FINE/DEIM lineage.

**Corrected fact:** Treat EdgeCrafter as a distinct architecture/release candidate, **not an independent research-family vote** against DEIMv2. Its numbers are self-reported motivation until scored by the frozen pickleball gate.

## C8 — small-data sports adaptation evidence

**Verdict: PARTIAL. Fetched: 2026-07-16.** All three works exist and the numeric statements are recognizable, but C8(a) needs a crucial annotation-type correction.

- **(a)** The [Gadde et al. WACV 2022 paper](https://openaccess.thecvf.com/content/WACV2022/papers/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.pdf) says “16 point improvement in mAP” from “around a 100 samples per video.” Those samples receive **instance-level domain labels on bounding-box predictions from an inductive model**; they are not 100 newly drawn full bounding boxes or 100 annotated frames.
- **(b)** The [Vandeghen et al. CVPRW 2022 paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Vandeghen_Semi-Supervised_Training_To_Improve_Player_and_Ball_Detection_in_Soccer_CVPRW_2022_paper.pdf) exists. Its linked [SST repository API](https://api.github.com/repos/rvandeghen/SST) reports BSD-3-Clause, and the [license text](https://raw.githubusercontent.com/rvandeghen/SST/main/LICENSE) has the three BSD conditions.
- **(c)** The [Maglo et al. CVPRW 2022 paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Maglo_Efficient_Tracking_of_Team_Sport_Players_With_Few_Game-Specific_Annotations_CVPRW_2022_paper.pdf) reports tracking a full rugby sevens game from 70 short tracklet annotations over 12 observable players—approximately **six few-second tracklets per player**.

**Corrected fact:** C8(a) supports low-volume weak/transductive **domain labeling**, not a general “100 box annotations gives +16 mAP” adaptation law. C8(c) is game-specific identity supervision with observability assumptions, not generic detector-label efficiency.

## C9 — license bundle

**Verdict: PARTIAL. Fetched: 2026-07-16.** (a), (b), (d), (e), (f), and (g) confirm. (c) is not supported by the current official page.

- **(a) CONFIRM:** The official [SportsMOT README](https://raw.githubusercontent.com/MCG-NJU/SportsMOT/main/README.md) says users “agree not to distribute” without prior written permission and designates CC BY-NC 4.0.
- **(b) CONFIRM:** The official [DanceTrack README](https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/README.md) says annotations are CC BY 4.0, the dataset is noncommercial-research-only, internet videos/images are not owned by the maintainers, and code is MIT. The [code license](https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/LICENSE) is MIT.
- **(c) REFUTE AS WRITTEN:** The current [Market-1501 official page](https://zheng-lab-anu.github.io/Project/project_reid.html) provides download links and a citation request, but searches of the fetched page find no `academic`, `redistribut`, `commercial`, or `license` terms. Absence of those terms is not permission; it means the claimed official-page restriction is unsupported and dataset-use rights remain unresolved without another authoritative agreement.
- **(d) CONFIRM:** KPR's [actual LICENSE](https://raw.githubusercontent.com/VlSomers/keypoint_promptable_reidentification/main/LICENSE) is Hippocratic License 3.0 and points to variant `law-media-mil-soc-sv` (the repo badge renders HL3-LAW-MEDIA-MIL-SOC-SV).
- **(e) CONFIRM:** The [SAM-MT HF model API/card](https://huggingface.co/api/models/FudanCVL/SAM-MT) declares `cc-by-nc-sa-4.0`. The [GitHub repo API](https://api.github.com/repos/FudanCVL/SAM-MT) reports `license: null`; its [recursive tree](https://api.github.com/repos/FudanCVL/SAM-MT/git/trees/main?recursive=1) has no case-insensitive `LICENSE`, `LICENCE`, or `COPYING` path.
- **(f) CONFIRM:** Ultralytics' [official licensing page](https://www.ultralytics.com/license) says trained YOLO models are AGPL-3.0 by default, with Enterprise licensing for proprietary/commercial embedding that does not satisfy AGPL obligations. YOLO26 docs also link AGPL-3.0 and Enterprise.
- **(g) CONFIRM:** Meta's [SAM 2 README](https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md) says SAM 2 model checkpoints, demo code, and training code are Apache 2.0; the [repository LICENSE](https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE) is Apache License 2.0.

**Corrected fact:** Remove the Market-1501 “official page = academic only/no redistribution” assertion. Record it as **license/use terms unresolved from the current official page**, pending a separately fetched authoritative agreement.

## C10 — Ultralytics YOLO26 family and medium asset

**Verdict: CONFIRM. Fetched: 2026-07-16.**

- The official [YOLO26 docs](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo26.md) enumerate detection and instance-segmentation filenames for all five scales: n/s/m/l/x.
- The docs call the default a one-to-one head that emits end-to-end predictions without NMS; the alternative `end2end=False` one-to-many head requires NMS.
- The official [Ultralytics README](https://raw.githubusercontent.com/ultralytics/ultralytics/main/README.md) reports YOLO26m detection at **52.5 COCO AP50:95 on the default e2e head**, or 53.1 on the non-e2e head. It links the exact asset <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt>.
- HEAD on that exact asset returned **200**, `Content-Length: 44255705`. The medium segmentation asset <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt> also returned **200**, `Content-Length: 54750385`.

**Corrected fact / spec pin:** Pin `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt`, 44,255,705 bytes, and label its default-head metric **52.5 AP e2e**. Do not silently substitute the 53.1 non-e2e number.
