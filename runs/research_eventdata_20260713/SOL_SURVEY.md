# Public event/contact data for racket-sport bootstrap

Date: 2026-07-13  
Lane: `eventdata_sol_20260713`  
Status: **VERIFIED=0** — research survey only. No archive was fully downloaded, no labels were audited against decoded frames, no checkpoint was run, and no pickleball event gate passed.

## Bottom line

The owner hypothesis is only partly right. Public data is sufficient to bootstrap **pieces** of a pickleball event head:

- exact visual BOUNCE frames from OpenTTGames;
- exact or near-exact HIT anchors from ShuttleSet and the CC BY badminton hit-frame release;
- useful impact-event audio from table tennis, squash, and padel;
- point-event spotting architectures and downloadable weights from E2E-Spot, T-DEED, SwingNet, and PANNs;
- physics distributions from synthetic tennis/table-tennis releases.

It is not sufficient to replace product-domain labeling. I found no fetched public dataset that combines fixed-camera pickleball video, synchronized audio, exact HIT and BOUNCE point labels, hard negatives, and commercially clear media rights. Several frequently cited “hit” datasets annotate a whole stroke interval or a rally phase, not the physical contact instant. Those must not be treated as frame-accurate ground truth.

The best defensible route is public, source-aware pretraining followed by high-recall audio proposals and human correction on our own synchronized captures. Public noncommercial or broadcast-derived assets should live in an isolated R&D lineage; they must not silently flow into a commercial model.

## Evidence and terminology

Evidence state is attached to every row:

- **FETCHED-primary**: I opened the official project/repository/paper/model card and derived the claim from it, not from a search snippet.
- **FETCHED-live**: in addition, the repository, metadata page, or named download endpoint resolved during this survey. This does **not** mean a multi-GB archive was downloaded or checksum-verified.
- **PUBLISHED-ACCESS-BLOCKED**: a fetched primary source publishes a download path, but the current endpoint required login/registration, showed reCAPTCHA, or otherwise did not yield the asset anonymously.
- **NOT FOUND**: the fetched primary materials did not state an asset license, size, checkpoint, or download as indicated. Absence was not converted into permission.
- **SEARCH-RESULT-ONLY**: a lead appeared in search but could not be confirmed from a primary source. Such leads are not used in the recipe.

Granularity is normalized as:

- **POINT**: one frame or timestamp per event; suitable in principle for a contact target after auditing synchronization.
- **INTERVAL**: onset/end or a span of frames; useful as weak supervision, not exact contact truth.
- **CLIP**: only clip/video action class.
- **OBJECT/STATE**: boxes, tracks, or simulated kinematics without an observed event point.

“License” below reproduces the license name or operative restriction displayed by the fetched source. A code-repository license does not automatically clear third-party broadcast/YouTube footage or derived weights.

## Ranked dataset adoption table

Ranks reflect transfer value for a single fixed-camera pickleball HIT/BOUNCE head, label precision, present access, and rights risk. They are not claims of product readiness.

| Rank | Source | Exact label granularity | Size and camera/source | Current path and evidence | License / rights posture | Pickleball transfer fit |
|---:|---|---|---|---|---|---|
| 1 | [OpenTTGames / TTNet](https://lab.osai.ai/) plus [single-frame extension](https://github.com/moamal01/table_tennis_data) | **POINT.** Original JSON has one `event_frame` for ball bounce, net hit, or empty, with ball coordinates/masks for 4 frames before and 12 after. Extension adds stroke/rally-ending events as one frame. Extension explicitly says true contact can fall between 120-fps frames and selects the following frame. | 4,271 manually annotated events; 5 full training games plus 7 test clips; full-HD, 120 fps, static side camera. | **FETCHED-primary, FETCHED-live.** Per-video and markup links are present on the official page; train videos are roughly 3.9–10.8 GB each. | Exact displayed text: **“Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).”** Research-only bootstrap; commercial launch lineage blocked without separate rights. | **Very high for BOUNCE**, useful for NET/empty negatives; stroke extension is useful HIT supervision. Sport/camera and ball-speed differences remain. |
| 2 | [A New Perspective for Shuttlecock Hitting Event Detection](https://github.com/wish44165/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection) / [Zenodo release](https://zenodo.org/records/14677727) | **POINT.** Clip label/output is `HitFrame`, i.e. the designated hitting frame. | 800 train + 169 validation + 230 test clips = 1,199 clips; release archive about 14.5 GB; 1280×720 RGB match clips. The fetched card does not document camera composition sufficiently to call it fixed-camera. | **FETCHED-primary, FETCHED-live.** Zenodo provides data and trained SwingNet, ViT, and YOLOv5m models. | Exact Zenodo license: **“Creative Commons Attribution 4.0 International.”** Media provenance should still be recorded before commercial use. | **Very high for HIT.** It is the most turnkey fetched point-label package with data and weights, but has no bounce labels and shuttle/racket kinematics differ. |
| 3 | [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) | **POINT by timestamp.** Each stroke row has `time (hr:min:sec)` and the documentation derives `frame_num = seconds × fps`. This is an impact anchor, although source A/V/frame synchronization still needs sample audit. | 44 broadcast matches, 104 sets, 3,685 rallies, 36,492 strokes, 27 players. | **FETCHED-primary, FETCHED-live metadata.** CSVs are in GitHub; `match.csv` provides source video URLs, so media must be reacquired and aligned. | Repository LICENSE: **“MIT License.”** That covers repository material; it does not establish reuse rights for linked broadcast/YouTube video. | **Very high for HIT temporal pretraining.** Large anchor set; no bounces; alignment and broadcast-rights audit required. |
| 4 | [Multi-Modal Hit Detection and Positional Analysis in Padel Competitions](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf) | **INTERVAL audio:** manually labeled hit onset/offset. A subset of 319 hits also has the closest visible video frame and player identity, making that subset near-POINT at 25 fps, not 40-ms-proof. | 5 h 28 min; 99 rallies from 11 tournaments; 2,377 hit sounds; fixed-camera padel broadcasts, 25 fps. | **FETCHED-primary.** Paper publishes [this data path](https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3), but current fetch was **PUBLISHED-ACCESS-BLOCKED** by the host/reCAPTCHA. No code or checkpoint was found in the paper/project trail. | Dataset license: **NOT FOUND** in fetched primary materials. Paper landing page reports **“No license (in copyright)”** for the manuscript record. Do not ingest until dataset terms are obtained. | **Potentially highest audio transfer** because padel paddle impacts and fixed cameras resemble pickleball. Rights/access uncertainty prevents immediate adoption. |
| 5 | [TT Sounds](https://github.com/cogsys-tuebingen/tt_sounds) | **POINT-like audio snippets.** Extracted samples are 15 ms impact snippets separated into racket-ball stroke types, table, floor, and other impacts. Original recordings are also linked. | 3,396 racket-ball events across 10 rackets; 777 table, 290 floor, 1,239 other = 5,702 labeled snippets. Controlled recordings rather than broadcast match audio. | **FETCHED-primary, FETCHED-live.** Repository and named Nextcloud data link resolved at the source level; full archive not downloaded. | Exact repository text: **“Creative Commons Attribution-NonCommercial 4.0 International License.”** Research-only. | **High for audio HIT-vs-BOUNCE-vs-other pretraining.** Excellent ontology match, but table-tennis timbre and controlled acoustics differ sharply from outdoor/indoor pickleball captures. |
| 6 | [Audio-based performance evaluation of squash players](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194394) / [Figshare project](https://figshare.com/projects/Audio-based_performance_evaluation_of_squash_players/30115) | **POINT/onset audio.** Human labels identify racket, front-wall, floor, glass-wall, and false detections at audio-event times; video was used to resolve doubtful events. | 5,791 labeled channel-events across six synchronized 96-kHz microphones, including a controlled exercise and a 7-minute match. | **FETCHED-primary, FETCHED-live project page.** Full asset was not downloaded. | Article exact text: **“Creative Commons Attribution License.”** Asset-level license was not independently exposed in the fetched project metadata, so confirm it before ingestion. | **High for acoustic factorization** of racket HIT vs floor BOUNCE vs environmental impact; multi-mic glass-court acoustics are a material mismatch. |
| 7 | [GolfDB / SwingNet](https://github.com/wmcnally/golfdb) | **POINT.** Eight exact swing event frames include `Impact`. | 1,400 YouTube swing clips with diverse views and speeds; not a rally camera. | **FETCHED-primary, FETCHED-live.** Preprocessed data and `swingnet_1800.pth.tar` are linked. | Exact repository license: **“Creative Commons Attribution-NonCommercial 4.0 International.”** Research-only; source-video rights remain separate. | **High for point-event model initialization**, especially SwingNet timing heads; low direct visual/domain transfer and no bounce class. |
| 8 | [FineBadminton-20K](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K) | **INTERVAL, not contact.** Released JSON gives each hitting action `start_frame` and `end_frame` plus player and hit type. Raw examples tile stroke/rally phases; they do not publish a separate contact frame. | Released subset: 70 full matches, 2,066 rallies, 20,757 hits, about 43.4 GB, generally 1280×720 30-fps broadcasts. Paper describes a larger 120-match/33,325-stroke collection. | **FETCHED-primary, FETCHED-live.** Hugging Face metadata and raw JSON resolved. | Model card exact identifier: **“apache-2.0.”** This does not by itself clear the underlying broadcast footage; preserve provenance and review media rights. | **Medium-high weak supervision** for who/when a stroke occurs. Do not collapse the interval boundary to physical contact without a decoded-frame audit. |
| 9 | [TenniSet](https://github.com/HaydenFaulkner/Tennis) | **INTERVAL/per-frame action labels.** Eleven serve/hit actions are labeled at frame precision, but events last roughly tens of frames; they are not single impact points. | 5 full broadcast matches; over 4,000 labeled events (1,017 serves, 2,551 hits reported in the paper); 746 point clips in the repository. | **FETCHED-primary, FETCHED-live.** Google Drive links advertise videos (11.1 GB), annotations (9.5 MB), frames/flow (217 GB), splits, and models. | Repository LICENSE: **“MIT License.”** It does not establish rights to the broadcast match footage. | **Medium-high for tennis action spotting and E2E-Spot pretraining**, but not exact HIT labels and no bounce. |
| 10 | [PadelTracker100](https://zenodo.org/records/17020011) | **INTERVAL/window.** Shot flags cover a neighborhood selected from racket motion around the actual shot, not a single contact frame. Also provides ball boxes and player pose/position labels. | About 99,887 frames from two World Padel Tour Finals videos, 1920×1080 at 30 fps, standard single broadcast camera; shot annotations cover about 40,135 frames. Label archive is about 69.7 MB; source videos are referenced rather than packaged. | **FETCHED-primary, FETCHED-live.** Zenodo labels resolve; two source-video URLs are published. | Exact Zenodo license: **“Creative Commons Attribution 4.0 International.”** Source-video rights remain separate. | **Medium-high** for fixed-camera ball/player context, candidate mining, and weak shot windows; not point-contact ground truth. |
| 11 | [VNL-STES](https://hoangqnguyen.github.io/stes/) | **POINT.** Human annotations identify an event frame and image position for serve, receive, set, spike, block, and score. Some are contact-like, but label semantics are volleyball actions rather than an explicit physical-touch ontology. | 8 full volleyball matches, 1,028 rallies, 251,110 frames, 6,137 events; HD broadcast, 25 fps. | **FETCHED-primary.** The advertised 13-GB download redirected to a SharePoint login: **PUBLISHED-ACCESS-BLOCKED**. | Dataset license: **NOT FOUND.** Associated code repository uses **“BSD 3-Clause License.”** Code license is not media permission. | **Medium** for exact multi-event spotting under broadcast clutter; ball scale, team layout, and event semantics differ. |
| 12 | [MediaEval 2023 SportsVideo table-tennis task](https://multimediaeval.github.io/editions/2023/tasks/sportsvideo/) | **POINT/moment according to the task page.** Task 2 detects when a player performs a stroke; Task 5 detects when the ball hits table or racket. Ground truth says experts annotated “moments in the video.” The fetched public page does not expose the timestamp file schema or tolerance, so exact frame precision is unconfirmed. | GoPro through Blackmagic 4K recordings from multiple angles; the task working-notes material reports 56 audio files, but total duration/event count is not stated on the public task page. | **FETCHED-primary. PUBLISHED-ACCESS-BLOCKED.** Registration and signed agreements are required; no anonymous archive path is exposed. | Exact agreement language includes **“MediaEval 2023 Research Collections”** and requires a separate University of Bordeaux SportsVideo data-usage agreement. The public agreement is research-use access, not an open commercial license. | **Potentially high** because it explicitly separates racket/table impacts. Access, schema, count, and rights must be resolved with organizers before ranking higher. |
| 13 | [P2ANet table-tennis dataset/code](https://github.com/Fred1991/P2ANET) / [paper](https://arxiv.org/abs/2207.12730) | **INTERVAL/segment.** Dense temporal action localization for 14 fine-grained action classes, not physical contact points. | 2,721 broadcast clips from 200 source videos at 25 fps; paper reports large dense action annotation volume. | **FETCHED-primary.** GitHub is archived; the documented Baidu dataset path is deprecated and access is by email, so **PUBLISHED-ACCESS-BLOCKED** for practical use. | Code repository: **“MIT License.”** Dataset/broadcast license: **NOT FOUND** in fetched materials. | **Medium-low** for action-phase representation only. Poor fit to a 40-ms HIT/BOUNCE decision. |
| 14 | [BadmintonDB](https://github.com/kwban/badminton-db) | **INTERVAL.** EAF/JSON fields include `StrokeBegin` and `StrokeEnd`; no separate impact point. | 9 YouTube match videos, 811 rallies, 9,671 strokes. | **FETCHED-primary, FETCHED-live annotations.** Source-video URLs are listed rather than redistributed. | **No LICENSE file found.** Treat as all-rights-reserved/permission-required; YouTube URLs do not grant training rights. | **Medium-low** weak stroke supervision; useful only after rights and synchronization review. |
| 15 | [Automated Hit-frame Detection for Badminton Match Analysis](https://github.com/arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis) | **POINT output.** Pipeline is designed to emit hit frames. | Dataset size and camera composition are **NOT FOUND** in the fetched repository. | **FETCHED-primary.** A Google Drive dataset link is published. The repository references several weight paths, but no downloadable trained checkpoint was confirmed. | Exact code license: **“MIT License.”** Dataset/media license: **NOT FOUND.** | **Medium-low until audited.** Relevant code path, but insufficiently documented data and checkpoint provenance. |
| 16 | [THETIS 3D tennis actions](https://github.com/THETIS-dataset) / [paper](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W08/papers/Gourgari_THETIS_Three_Dimensional_2013_CVPR_paper.pdf) | **CLIP.** Twelve action classes; no hit-contact or bounce timestamp. | 8,374 sequences, 55 subjects, 7 h 15 min; controlled Kinect RGB, silhouette, depth, 2D/3D skeleton; 31 beginners and 24 experts. | **FETCHED-primary.** The historical host is published, but no currently verified anonymous full-data download was established. | Dataset license: **NOT FOUND.** | **Low** for contact timing. Could pretrain pose/action features, but camera, actors, and clip labels are far from the product problem. |
| 17 | [AudioSet](https://research.google.com/audioset/) and [strong labels](https://research.google.com/audioset/download_strong.html) | Original set: **CLIP**, weak presence over 10 s. Strong subset: **INTERVAL** onset/offset in decimal seconds; an evaluation derivative uses 960-ms frames. No fetched tennis/table-tennis/pickleball contact class; “Basketball bounce” exists. | 2,084,320 ten-second YouTube clips, 527 classes. Strong set: 103,463 train clips / 934,821 events and 16,996 eval clips / 139,538 segments. | **FETCHED-primary, FETCHED-live metadata.** Media remains referenced through YouTube availability. | Exact label license: **“Creative Commons Attribution 4.0 International (CC BY 4.0).”** Underlying audio/video is governed by its source rights. | **Low direct / medium encoder pretraining.** Strong labels are still much coarser than a 40-ms contact target and lack racket-specific ontology. |
| 18 | [AVE audio-visual event dataset](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/ave.html) | **INTERVAL at 1-second resolution**, one audio-visual event per 10-second clip. | 4,143 ten-second clips, 28 event classes, in-the-wild YouTube media. | **FETCHED-primary metadata.** Original-code/data trail exists, but no current archive was downloaded. | Dataset/media license: **NOT FOUND** in fetched catalog metadata. | **Low.** Useful for generic A/V correspondence, not impulse contact timing. |
| 19 | [Roboflow Pickleball Detector](https://universe.roboflow.com/pickleball-ball-detection/pickleball-detector) | **OBJECT only.** Image bounding boxes for one `pickleball` class; no temporal HIT/BOUNCE labels. | Version 3 reports 1,267 images; mixed submitted pickleball frames, camera consistency not documented. | **FETCHED-primary, FETCHED-live model page/API metadata.** | Exact displayed license: **“CC BY 4.0.”** Confirm contributor/media provenance for commercial use. | **Low for events**, potentially useful ball-detector augmentation only. It does not validate the owner hypothesis about public contact labels. |
| 20 | [PKLMARTS competitive pickleball extracts](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts) | **TABULAR/aggregate, not video time.** Shot/rally outcome records have no decoded-frame contact ground truth. | More than 300,000 shot/rally records from competitive doubles; camera not applicable. | **FETCHED-primary metadata.** Kaggle account/API may be required for download. | Exact displayed license: **“CC BY-NC-SA 4.0.”** Research-only. | **Low for event timing**; potentially useful class/sequence priors, never a frame-label substitute. |
| 21 | Hugging Face pickleball search | No verified pickleball HIT/BOUNCE video dataset was found. A prominent “Picklebot-130K” result is [explicitly baseball](https://huggingface.co/datasets/hbfreed/Picklebot-130K), not pickleball. | Not applicable. | **FETCHED-primary** for the false lead; negative search result is date-bounded to 2026-07-13, not proof of permanent absence. | Not applicable. | **None.** Do not contaminate the inventory through keyword ambiguity. |

## Frame-precise model and checkpoint inventory

These are model/tooling candidates, not additional claims that their training media is commercially cleared.

| Model | Temporal contract and checkpoint status | Code/license | Recommended use |
|---|---|---|---|
| [E2E-Spot](https://github.com/jhong93/spot) + [model repository](https://github.com/jhong93/e2e-spot-models) | Designed for single- or few-frame event spotting. Input annotations use one `frame` plus class. Downloadable Git-LFS models include Tennis RGB, flow, GRU, and larger backbones. **FETCHED-primary, FETCHED-live repository; weights not executed.** | Exact license: **“BSD 3-Clause License.”** Tennis weights inherit unresolved TenniSet/broadcast provenance. | Best first visual backbone comparison: initialize from Tennis RGB, replace the classifier with HIT/BOUNCE/OTHER, and retain a scratch baseline. |
| [T-DEED](https://github.com/arturxe2/t-deed) | Precise event spotting across Tennis, FineDiving/FineGym, SoccerNet and SoccerNet Ball; README publishes pretrained checkpoint links and reports first place in SoccerNet Ball Action Spotting 2024. **FETCHED-primary; checkpoint link published, not run.** | Exact license: **“GNU General Public License v3.0.”** This is a product-integration and distribution review item; training-data rights are separate. | Strong research benchmark and longer temporal-context teacher. Do not select it as product code without GPL review. |
| [SwingNet/GolfDB](https://github.com/wmcnally/golfdb) | Per-frame sequence head with a live pretrained `swingnet_1800.pth.tar`, trained on eight point events including Impact. **FETCHED-live, not run.** | **“Creative Commons Attribution-NonCommercial 4.0 International.”** | Good timing-head initialization/ablation, R&D lineage only. |
| [Badminton HitFrame models](https://zenodo.org/records/14677727) | Trained SwingNet, ViT, and YOLOv5m variants accompany the 1,199-clip release. **FETCHED-live metadata, not run.** | **“Creative Commons Attribution 4.0 International.”** | Highest-priority sport-specific checkpoint smoke test after archive/provenance audit. |
| [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) / [Zenodo checkpoints](https://zenodo.org/records/3576403) | AudioSet-trained CNNs include frame-wise decision-level variants; Zenodo package is about 19 GB. **FETCHED-primary, checkpoint endpoint live, not run.** | Exact code license: **“MIT License.”** AudioSet/source-media rights remain separate. | Initialize an audio encoder, then train a small high-resolution contact head on racket-specific and product labels. Never use AudioSet clip logits as contact labels. |
| [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) / [TF Hub](https://tfhub.dev/google/yamnet/1) | 521 AudioSet classes; scores/embeddings over 0.96-s frames with 0.48-s hop. **FETCHED-primary, model endpoint published, not run.** | TensorFlow Models repository: **“Apache License 2.0.”** | Cheap generic-audio baseline/embedding source. Native resolution is decisively too coarse for the ≤40-ms product gate. |
| [SoccerNet action spotting](https://github.com/SoccerNet/sn-spotting) | Point timestamps for broad game events; SoccerNet-v2 has 500 games, 17 classes, roughly 300k annotations. Ball Action Spotting 2024 uses 7 games and 12 classes. Evaluation is broadcast event spotting, not physical contact. Videos require password/NDA. | SoccerNet FAQ exact text: **“The SoccerNet dataset is meant for research purposes, it is not intended for commercial purposes.”** | Architecture/evaluation reference only. Domain, 25-fps timing, seconds-scale tolerances, and license make it poor direct pretraining relative to tennis/badminton checkpoints. |

## Synthetic and simulation routes

| Source | What is actually available | License/access | Useful route and limitation |
|---|---|---|---|
| [AD-Rallies](https://huggingface.co/datasets/XSpaceCoderX/AD-Rallies) | About 3.2 million synthetic tennis rallies sampled at 500 Hz (2 ms), with ball position plus linear/angular velocity in MuJoCo arrays; roughly 92.9 GB total / 87-GB main archive. It does **not** provide product-like rendered RGB/audio or a separate observed contact-label file. **FETCHED-primary, live model card.** | Exact card license: **“gpl-3.0.”** | Derive simulator contact events and train trajectory priors; render our fixed camera, blur, occlusion, compression, and synthetic impulses. It cannot prove real contact accuracy. |
| [DeepMind competitive robot table tennis](https://github.com/google-deepmind/competitive_robot_table_tennis) | 15,792 post-hit ball initial states (13,088 rallies + 2,704 serves) with position, velocity, and spin plus MuJoCo visualization. No match video/audio or event sequence labels. **FETCHED-primary, live repository.** | Exact text: software **“Apache License 2.0”**; other materials **“Creative Commons Attribution 4.0 International.”** | Sample realistic post-contact state distributions and physics hard cases; not a direct event dataset. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics engine with exact simulated contacts and rendering; no packaged pickleball scene or labels. **FETCHED-primary.** | Exact license: **“Apache License 2.0.”** | Build paddle/ball/court assets, emit exact HIT/BOUNCE state and time, and domain-randomize. Synthetic audio must be explicitly modeled and will remain a large reality gap. |
| [Sony 3D Ball Tracking / Ace data](https://sonyresearch.github.io/ace_public/data/) | Published CSV-style event/state records include shot, bounce, and net timestamps with 3D ball state; fetched page does not state a complete count, product-like RGB/audio pairing, or a reusable trained contact checkpoint. | **FETCHED-primary. License NOT FOUND** on the fetched data page. | Potential trajectory/state supervision after permission and schema audit; not safe to ingest now. |
| [SPIN table-tennis paper](https://arxiv.org/abs/1912.06640) | Paper reports roughly 53 h training + 1 h test with stereo 150-fps capture. A live public archive/license was **NOT FOUND**. | **FETCHED-primary paper; no verified data path.** | Paper reference only; unavailable evidence cannot be part of the bootstrap. |
| [1000 Rallies](https://arxiv.org/abs/2606.25620) | June 2026 paper reports more than 1,000 table-tennis rallies, event cameras plus 14 synchronized 200-fps cameras, and 1-kHz pseudo-ground-truth ball state. | **FETCHED-primary paper. No public download or license found as of 2026-07-13.** | Watch-list item with strong future value; currently unavailable and therefore excluded from the recipe. |

## Important nonpublic evidence

Two papers show that the desired supervision has existed inside research organizations, but neither supplied a usable public release in the fetched primary materials:

- [Detection of Tennis Events from Acoustic Data](https://research.ibm.com/publications/detection-of-tennis-events-from-acoustic-data) reports 6,568 individually labeled one-second tennis sounds and hit/announcer/applause classification. No public dataset, code, or checkpoint path was found. It supports audio feasibility, not public availability.
- [Learning to Localize Sound Source in Visual Scenes: Analysis of Audio-Visual Synchronization in Tennis](https://arxiv.org/abs/2104.10116) reports about 504,300 frames / 6 hours labeled frame-by-frame as hit, bounce, or neither. No public dataset, code, or checkpoint was found. This is the closest described ontology to our need and the clearest evidence that the public gap is a release/rights gap rather than a scientific impossibility.

## Recommended bootstrap recipe

### 1. Freeze one contact ontology and preserve source precision

Use `HIT`, `BOUNCE`, `NET_OR_OTHER_IMPACT`, and `NONE`, with an optional actor/player field. Store:

- source timestamp in audio sample time;
- decoded video frame and presentation timestamp;
- label precision (`point`, `interval`, `clip`, `derived`);
- uncertainty and occlusion;
- source dataset/license/provenance;
- measured A/V offset and correction.

Never turn a stroke interval into an exact contact label by taking its midpoint. Interval sources should train only a weak/auxiliary objective unless a human corrects the contact frame.

### 2. Public pretraining, separated by modality and license lineage

For the visual branch:

1. Compare E2E-Spot Tennis RGB initialization, the CC BY badminton HitFrame checkpoint, and a scratch model under the same frozen product validation split.
2. Train point losses with OpenTTGames BOUNCE/NET and badminton/ShuttleSet HIT anchors. Keep sport/source embeddings or adapters so the model can learn source-specific appearance and timing.
3. Use TenniSet, FineBadminton, PadelTracker100, and P2ANet only for interval/phase auxiliary losses. They are not exact-contact rows.
4. Use public ball boxes/trajectories only as context inputs, never as replacement event labels. Geometry has already produced too many candidates in our product setting.

For the audio branch:

1. Initialize a compact PANNs-style encoder or a small CRNN; retain a scratch comparator.
2. Learn transient and impact-type separation from TT Sounds and the squash labels. Add padel only after its data license and access are resolved.
3. Operate on 10–20-ms log-mel hops and retain sample/time offsets. YAMNet’s 0.96-s/0.48-s cadence is only an embedding baseline, not the final head.
4. Do not assume public acoustics transfer. Paddle material, room impulse response, wind, multiple courts, footwear, voices, and camera automatic gain are product-domain variables.

All noncommercial, research-only, GPL-sensitive, or broadcast-ambiguous sources should be recorded in a segregated **R&D lineage**. A commercial candidate must either obtain rights or be retrained without restricted data/weights; “fine-tuned on our data” does not erase upstream licensing.

### 3. Generate high-recall proposals on our product captures

Run an audio transient proposer using spectral flux/energy change plus audio-encoder embeddings. Fuse it with, but do not gate it on:

- WASB ball-track position, confidence, velocity, and direction change;
- wrist/racket proximity where visible;
- court-plane proximity and vertical trajectory around bounce candidates;
- visibility/occlusion state;
- local E2E-Spot visual logits.

First calibrate per-capture A/V offset. Present reviewers a short window centered on every audio proposal plus geometry-only candidates with no audio proposal. This avoids baking audio dropouts into the label distribution.

### 4. Human-correct product-domain labels and mine hard negatives

A practical first tranche is **recommended**, not claimed sufficient: at least 2,000 reviewed HITs, 2,000 reviewed BOUNCEs, and 8,000 temporally matched hard negatives, with learning curves used to decide the next tranche. Prioritize negatives that defeat geometry/audio shortcuts:

- shoe squeaks, speech, claps, paddle taps, and ball handling;
- adjacent-court contacts;
- net/court-fence/wall impacts;
- off-screen or occluded contacts;
- ball-detector jumps and missing detections;
- replay cuts, dropped audio, variable frame rate, and A/V drift.

Split by capture session/venue/player before proposal review. Otherwise near-duplicate rally acoustics and camera appearance will leak across train and evaluation.

### 5. Train a calibrated multimodal event head

Align audio tokens at 10–20 ms, visual tokens at decoded presentation timestamps, and geometry tokens in the same clock. Use modality dropout and explicit missingness so the head works when audio or ball/wrist features fail. Predict event class, time offset within the video frame, and confidence; use non-maximum suppression with class-specific refractory periods.

H100 availability makes checkpoint comparison and source-balanced ablations cheap. The bottlenecks are accurate synchronization, label quality, hard-negative coverage, and licensing—not accelerator throughput.

### 6. Gate only on a fresh product holdout

The public-data survey does not move `VERIFIED=0`. Evaluate on untouched product sessions with:

- HIT and BOUNCE precision/recall/F1 separately;
- HIT↔BOUNCE confusion and NET/other-impact confusion;
- false positives per minute outside rallies;
- timing error after measured A/V correction, including the roadmap contact target of p90 ≤ 40 ms;
- slices for venue, camera distance, audio quality, occlusion, multiple courts, and detector failure;
- audio-only, visual-only, geometry-only, and fused ablations.

Public benchmark gains, a browser load, a visually plausible replay, or a copied checkpoint are not promotion proof.

## What does not exist publicly, based on this fetched survey

As of 2026-07-13, I did **not** find a fetched public source with all of the following:

1. pickleball rather than a neighboring sport;
2. synchronized fixed-camera RGB and product-like audio;
3. exact HIT **and** BOUNCE point labels on a shared clock;
4. hard negatives and complete events through occlusion/off-screen cases;
5. a live anonymous download plus an executable checkpoint;
6. media, annotations, code, and weights all clearly licensed for commercial model training/deployment;
7. a held-out product-style gate demonstrating p90 timing ≤ 40 ms.

Specific negative findings:

- Public pickleball pages found on Roboflow label balls in images, not contacts in time.
- The Kaggle pickleball source found is tabular rally/shot analytics, not frame labels.
- No verified Hugging Face pickleball HIT/BOUNCE video release was found; a tempting `Picklebot` result was baseball.
- Many tennis/badminton/padel “hit” datasets label action windows. Frame-indexed boundaries are not necessarily contact frames.
- AudioSet and AVE are too weak/coarse and lack a racket-contact ontology.
- OpenTTGames and TT Sounds are excellent research bootstrap data but explicitly noncommercial.
- ShuttleSet, TenniSet, FineBadminton, PadelTracker100, and several others depend on broadcast/YouTube media whose rights are not made commercial-clear merely by an open annotation/code license.
- The padel A/V source has unusually good transfer fit, but current access and dataset-license terms were not verifiable.
- The Amazon tennis A/V paper describes almost exactly the desired hit/bounce/neither frame labels, but does not release them.
- Emerging high-speed/event-camera table-tennis data is described in papers, but no live public release/license was found.

Therefore the public corpus can reduce cold-start cost and teach event priors, but **our synchronized, reviewed product captures remain the promotion dataset**.

## Ranked action list

1. Request/inspect the padel A/V data terms and MediaEval SportsVideo agreement/schema; do not ingest before rights are explicit.
2. Download a small, provenance-recorded R&D sample from OpenTTGames, the Zenodo badminton HitFrame release, TT Sounds, and squash; audit 100 labels per source against decoded media/audio before training.
3. Smoke-test E2E-Spot Tennis RGB, the badminton HitFrame checkpoint, PANNs initialization, and scratch baselines under one environment. Record that this is checkpoint execution, not accuracy verification.
4. Build product-capture audio proposals, measure A/V offset, and begin source-precision-aware human review.
5. Freeze a session-held-out product gate before the first fused experiment. Only that gate can change capability status.
