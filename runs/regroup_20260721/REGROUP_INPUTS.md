# REGROUP_INPUTS
Compiled evidence pack for strategy synthesis. Source: rawData (disk inventory), modelMap (training-consumption inventory), pbvInv (pb.vision schema/data inventory), labelLedger (owner-label provenance). No strategy decisions are made below — this is the evidence only.

Note on scope: the task asked for the "GAP MATRIX (verbatim)" as an input. No file or JSON block containing a pre-built gap matrix was present in the supplied inputs, and a repo-wide search (`grep -rli "gap matrix"`, `find -iname "*gap*"/"*regroup*"`) found no such artifact on disk. Section 3 is therefore reconstructed directly from the organizational-miss statements embedded in modelMap and labelLedger, flagged as such rather than presented as a verbatim quote.

---

## 1. DATA WE OWN (from inventory, with numbers)

### 1.1 Top-level footprint
`data/` has 13 subdirs beyond `credentials/`. Disk footprint (`du -sh`):

| Asset | Size |
|---|---|
| event_public_20260713 | 6.0G |
| roboflow_universe_20260706 | 6.9G |
| online_harvest_20260706 | 2.3G |
| pbvision_gallery_20260719 | 136M |
| pbvision_11min_20260713 | 127M |
| pbv_replay_20260720 | 59M |
| event_bootstrap_20260713 | 15M |
| event_labels_owner_20260719 | 16K |
| online_harvest_20260712 | 20K |
| testclips | 16K |
| credentials | 8K |

Plus `eval_clips/=21M total`, `cvat_upload/=755M total`.

### 1.2 pb.vision competitor data (gallery + 11-min demo + replay)
- **data/pbvision_gallery_20260719**: 13 total pb.vision video IDs have full `cv_export.json`+`insights.json` supervision on disk (12 freshly fetched 2026-07-19/20 + 1 pre-existing, 83gyqyc10y8f, not currently in the live gallery listing — status `PRE_EXISTING_NOT_RE_HARVESTED`). Combined duration across the 13: **9,865.034s = 164.42 min = 2.74 hr**. Export JSON on Mac: **155,143,435 bytes (~148MB)**. The actual video pixels for 11 of the 13 exist **only on VM `pickleball-gpu-evhead`** (2,390,645,811 bytes / 2.39GB, not on Mac disk); only 2 of the 13 have MP4 physically on the Mac (83gyqyc10y8f at `data/pbvision_11min_20260713/`, 119,911,668 bytes; xkadsq9bli3h at `data/pbv_replay_20260720/`, 61,628,656 bytes / 186.15s).
- **Gallery totals (12 videos)**: duration 9,167.6s (152.79 min); rallies 495 (cv_export)/488 (insights); shots 2,831 (2,627 crossed_net=true, 92.8%); dense per-frame ball 2D action-output present in 100% of 114,421 frames (conf≥0.9 in 12.3%, conf≥0.5 in 56.5% — no owner-set detection threshold exists in the raw data); discrete 3D ball positions (`balls.ball`) 84,620 frames (2.98% interpolated); bounce events 2,184 frames (1.37% interpolated); net-crossing dense-frame events 188 (29.79% interpolated); shot events 3,347 (0.24% interpolated); court calibration keypoints 144 slots (12 venues × 12 pts), 143 non-null, confidence range 0.696–0.981 (mean 0.945); player-position slots 457,684 total, 429,783 non-null; avatars (tracked players) 44; highlights 209; coach_advice 48. **No player pose/skeleton data anywhere** (`has_player_pose_or_skeleton=false` on all 12 + the demo).
- **11-min demo (83gyqyc10y8f)**: 697.4s (11.62 min), 4 players; 42 rallies (cv_export)/41 (insights); 256 shots (230 crossed_net); 7,318/10,322 frames with 3D ball; 170 bounce, 28 net, 292 shot-events; 12/12 court points non-null; unique among the 13 in also having a `stats.json` (51,915 bytes) not fetched for the 12 newer gallery captures.
- **Combined 13-video total**: 9,865.0s / 164.42 min / 2.74 hr of source video with full JSON supervision on disk (video pixels themselves largely off-Mac).
- Discrete 3D ball/bounce/net/shot events carry **no numeric confidence**, only a boolean `interpolated` flag — a materially weaker teacher-quality signal than a detector confidence score.
- `usage_posture` (verbatim field): **"RD_ONLY competitor-processed; never protected-eval; never redistributed; no ball-detector training without a manager ruling; NS-07.3 review before any commercial use."**

### 1.3 Derived pb.vision teacher-event corpus
**runs/lanes/pbv_pickleball_corpus_20260720**: 4,637 teacher-derived HIT/BOUNCE events across 10 of the 12 gallery videos (3 compare-only holdouts excluded: iottnc0h3ekn, o4dee9dn0ccr, 83gyqyc10y8f). Split 7 train/2 val/1 test (source-disjoint); `manifest_sha256=cf8f251827688c7923e35ce93b06b66c014ba9192b9d18f4ecbd2a256195451b`. Projected windows: 4,852 train / 1,634 val / 828 test = **7,314 total, 0 currently decodable** (media_present=false for every row — source MP4s are VM-only). License posture: `pbvision_signed_full_usage` (owner-stated signed rights, 2026-07-20). Status explicitly VERIFIED=0 / teacher-derived, not human GT; the corpus report states the protected 50-row owner seed was not read or copied into it.

### 1.4 Owner-answered event labels
**data/event_labels_owner_20260719**: `PROVENANCE.json` + `results_batch1_102rows.json`. n_answered=102 (60 typed events, 42 hard negatives, 46 with coords, 57 with dt). `intended_use`: "TRAINING-candidate for the pickleball event-head fine-tune. NOT eval." Mandatory pre-train check: verify zero clip/time overlap with the protected 50-row eval seed. A superseded 71-row partial export exists in the owner's Downloads, not in-repo.

### 1.5 Protected 50-row owner event eval seed
`runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json` (+`.md`) + `owner_spot_check_results_20260715.json`. 29 typed contacts + 21 hard negatives = 50 rows. Owner spot-check (2026-07-15) of Tier-A audio-cross-track auto-labels: **29/50 true contacts vs a ≥47/50 bar — FAILED** (15/29 true windows mistimed |dt|≥0.2s). Status: permanently eval-only, never training, per repo doctrine.

### 1.6 Event bootstrap (audio-onset) data
**data/event_bootstrap_20260713**: `manifest_v0.json` (41,723 bytes) with sha256 per file — `contact_windows_v0.jsonl` (2,966,224 bytes), `diagnostic_contact_windows_v0.jsonl` (467,666 bytes), `negative_windows_v0.jsonl` (2,648,386 bytes), `inventory_v0.json` (57,245 bytes), plus `audio_onsets_v0/` (40 per-rally JSON files, tens-of-KB to ~547KB each). Underlies the rejected Tier-A audio-cross-track auto-labeling.

### 1.7 Public multi-sport event datasets
**data/event_public_20260713** (6.0GB): Tennis jhong93/spot 33,791 events (far/near bounce 8,150/8,127; swing 7,123/7,044; serve 1,657/1,690); F3Set tennis singles 42,829 + doubles 465; Table tennis OpenTTGames 4,271 events (bounce 1,777/net 1,350/empty 1,144) + 52,987 dense ball-track points; Extended OpenTTGames 4,945; F3Set table-tennis 361; Badminton ShuttleSet 36,484, ShuttleSet22 52,356; Padel PadelTracker100 13,250 shot-flagged frames (906 windows); Golf GolfDB 11,200; TT Sounds (audio) 5,702 (4,561 train/1,141 test); Shuttlecock Zenodo mirror + squash figshare = 0 confirmed structured events. Media on disk is partial (jhong93/spot 6/28 videos 963MB; OpenTTGames 2 videos 4.2GB; Shuttlecock mirror 8 clips 19MB; TT Sounds 6,037 wavs 17.6MB; squash figshare 1/11 parts 138MB; PadelTracker100/GolfDB/ShuttleSet/F3Set = labels only). License tiers recorded (`licenses_ledger.json`): RD_ONLY, RD_ONLY_STRICT, COMMERCIAL_CLEAN_LABELS_ONLY, UNCLEAR — e.g. jhong93/spot code is BSD-3-Clause but posture is COMMERCIAL_CLEAN_LABELS_ONLY because the underlying YouTube pixels aren't separately cleared.

### 1.8 YouTube harvests
- **data/online_harvest_20260706**: 25 candidates enumerated, 8 downloaded (HyUqT7zFiwk 971.0s, 73VurrTKCZ8 334.0s, wBu8bC4OfUY 543.2s, Ezz6HDNHlnk 829.5s, zwCtH_i1_S4 609.9s, _L0HVmAlCQI 576.8s, pwxNwFfYQlQ 862.3s, vQhtz8l6VqU 583.3s = 5,310.0s/88.5min), 1 failed, 16 skipped. All 8 raw MP4s present on Mac (~1.37GB). Subdirs: court_calibrations (7), prelabels (40), rallies (6 subfolders for 5 of the 8 sources), screening (32). This is the source pool behind the w5/w6/w7 CVAT ball/court-keypoint packs. **License field is null for every entry** — unrecorded at harvest time.
- **data/online_harvest_20260712**: 39 attempted, 28 selected, 8 failed, 1 excluded. Raw video was **never persisted beyond 60–90s extraction segments, deleted immediately after** — metadata/frame-images only; re-acquisition requires re-running yt-dlp per entry. Feeds `cvat_upload/court_diversity_20260712` (100 frames, 28 sources).

### 1.9 Public image dataset aggregation
**data/roboflow_universe_20260706** (6.9G): 75 pickleball-related datasets enumerated, 65 downloaded (9 no-published-version, 1 dead-link). `total_bytes_downloaded=6,907,211,259`. 67 top-level project dirs + `aggregated/`. `corpus_card.json`: **core_pickleball=80,967 image samples/59 sources**, adjacent_sport_aux=29,036/3 sources (one source spot-checked 12/12 actually tennis, not pickleball), excluded_duplicate=359/3, excluded_dead=0. Image-level, not video; separate from the ball training corpus below.

### 1.10 Owner/harvest-reviewed ball label corpus (growth chain, all 6 sources: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4)
1. w6_labelingest_20260708 → **1,121 rows** (682 pos/439 neg)
2. w7_ballingest2_20260709 → **1,750 rows** (1,240 pos/510 neg)
3. w7_ballingest3_20260709 → **2,388 rows** (1,828 pos/560 neg; manifest md5 `0ae65f014ce26b2ddf8573427c60853d`) — the "~2388" figure
4. w7_ballingest4_20260709 → **3,026 rows** (2,410 pos/616 neg) — **current on-disk state**; per-source: 73VurrTKCZ8=397, Ezz6HDNHlnk=400, HyUqT7zFiwk=560, _L0HVmAlCQI=554, wBu8bC4OfUY=555, zwCtH_i1_S4=560.

No ingest lane newer than 2026-07-09 was found for this corpus. **NORTH_STAR_ROADMAP.md's DATA row still cites the stale 1,750/1,121 figures** — it does not reflect the 2,388/3,026-row states on disk.

### 1.11 Protected eval fixtures
- **eval_clips/ball/**: 21M, 4 clips (burlington_gold, wolverine_mixed, outdoor_webcam_iynbd, indoor_doubles_fwuks), combined duration 69.22s; 120 ball_points rows, 4 events, 16 foot_contact rows, 4 court_corners sets. Explicitly `gate_status='local_eval_fixture_only_not_promotion_ready'`. Plus 2 court-keypoint-only additions layered on pbvision_11min and owner_IMG_1605 (no separate source video).
- **The only human-reviewed PERSON boxes in the repo: 11,459 total**, all on these same 4 protected videos (2 = frozen scoring card, 2 = strict holdout) — i.e. usable person-box *training* data elsewhere is zero, and CVAT is closed at the API level to further person-box tasks beyond these four.

### 1.12 CVAT staging area
**cvat_upload/** (755M): 4 raw MP4s mirroring the eval_clips 4 clips + `CVAT_LABELING_INSTRUCTIONS.md` + 2 owner-session logs. Subdirs: `court_diversity_20260712` (100 images/28 sources), `court_keypoints_20260707` (6 images/6 sources), `w5_labelpack_20260708` (640 frames/16 clips/8 sources, disagreement-type breakdown 320 large-offset/160 student-only/160 teacher-only, est. 2.67 labeling hours), `w6_labelpack_20260708` (22 clips/4 sources), `w7_audit_stratum_20260709` (32 clips/350 frames, uniform-random scratch, 566,890,613-byte image zip). `exports/` holds the actual CVAT annotation zips per pack.

### 1.13 Assets with no usable payload
- **data/testclips**: 4 clip_metadata.json entries (gear360, ppa_austin_md_qf, ppa_singles, side_view_game5), each declares duration_s=90.0, **no video files present on disk for any of the 4**, no location field to say where the source lives.
- **data/online_harvest_20260712**: raw video deliberately deleted post-extraction (see 1.8).
- **web/replay/public/** (new untracked dir): only a `critique/` subfolder seen at the depth inspected; not identified as training/eval data, not fully enumerated.

---

## 2. WHAT EACH MODEL TRAINS ON TODAY (the consumption map)

| Component | Default/production checkpoint | Trains on pickleball data today? | Detail |
|---|---|---|---|
| **Ball detector (WASB)** | `models/checkpoints/wasb/wasb_tennis_best.pth.tar`, `best_stack.json` entry `ball.wasb_checkpoint`, status WIRED_DEFAULT | **No — 0 pickleball rows** | Official WASB-SBDT tennis pretrain (gdown, repo commit 923462cacdeb), `fine_tuned_on_pickleball: false`. Labeled "raw WASB tennis zero-shot checkpoint... not a promotion of official retrains." |
| Ball detector — candidate fine-tunes | `ball.seed_official_checkpoint` (PENDING, not default) | Consumed the **1,121-row** corpus only | w7_ballretrain_20260709 (H100): 2 arms, pooled micro F1@20 0.615/0.612 vs zero-shot control 0.361 (+15.5% relative over prior 486-row checkpoint 0.533). Not promoted. |
| Ball detector — later retrain | no promoted output | Consumed the **3,026-row** corpus | w7_ballretrain2_20260709: found **26/40 clips (1,787/2,388 rows, 74.8% of the then-current corpus) CONTAMINATED** — "zero-shot" control predictions matched CVAT GT centers to ~0.003–0.006px (pooled F1 0.710 contaminated vs 0.236 clean-subset vs 0.361 historical). No report.md/final ruling in the lane dir; **not referenced anywhere in NORTH_STAR_ROADMAP.md or RUNBOOK.md** — status UNDETERMINED per doc-of-record. |
| **Event head** | none — **no entry at all in `best_stack.json`** (not wired into the default stack) | Trains on 3 of the fetched public datasets only | Builder (`datasets.py`, EXPECTED_UNIVERSE) consumes jhong93_spot (tennis, 33,791 events, 20/4/4 split), openttgames (table tennis, 4,271 events, 8/2/2 split), coachai_shuttleset (badminton, 36,484 events, 31/7/6 split) = **74,546 events total**. f3set, golfdb, squash_audio_figshare, shuttlecock_hitting_zenodo, padeltracker100, extended_openttgames, tt_sounds — all fetched 2026-07-13 but **not read by build_public_manifest**. |
| Event head — last GPU run | step-16918 checkpoint (not promoted) | Public corpus + resumed pretrain | runs/lanes/event_head_retrain_20260720 (A100, 2026-07-20T15:22–19:30Z): resumed from step 9000/118,770 (7.6% of target) with class-weighted loss [1,5,5]. Verdict: escapes all-negative collapse (TP 44/70/107 vs 0 baseline across tolerances) but precision still very low (466 FP vs 44 TP @ tol 1). "Does not yet produce a usable detector." VERIFIED=0. |
| Event head — pb.vision teacher data | not wired | 4,637 staged HIT/BOUNCE events, **0 training-eligible rows** | Not read by `datasets.py`/`finetune_event_head.py` — a `LOADER_CHANGE_REQUIRED.diff` sits unapplied. |
| Event head — owner GT | blocked (T11) | 102 owner-labeled events banked, not yet consumed | Blocked because the pretrain checkpoint was 0-detection at time of ruling (T8_RULING.md). |
| **Court keypoint net (learned U-Net)** | `court.court_unet_v2` / `court.e4_fusion_default`, both status **PENDING** (production default is `court.profile_driven_cluster`, a classical solver, status FENCED) | **100% synthetic — 0 real training rows** | 230,400 procedurally-generated CAL-SYNTH samples (1,800 steps × batch 128), 2.87 GPU-hrs, ResNet34 encoder from **random init** (no ImageNet pretrain — launch script dropped `--encoder-weights-path` because the pretrained file doesn't exist locally). Gate result PCK@5px 0.6414 → retrain 0.7512, vs 0.95 bar — not met. A separate 32-row real-label set is reserved strictly for the eval gate, never training. Real-venue CVAT labeling (100 diversity frames, 6 keypoint frames) has not fed any retrain of this network. |
| **Person detector** | YOLO26m, `tracking.person_detector`, WIRED_DEFAULT | **No — stock COCO checkpoint** | `ultralytics YOLO('yolo26m.pt')`, no pickleball fine-tune, no `fine_tuned_on_pickleball` field asserted. |
| Person detector — candidate | RF-DETR-L, `tracking.person_detector_rfdetr_large_2026`, PENDING, `do_not_promote=true` | No — COCO-pretrained zero-shot alt | Improves burlington (IDF1 0.8831→0.9220), regresses wolverine (0.8516→0.8036). |
| **ReID** | OSNet x1.0, `tracking.reid_model`, WIRED_DEFAULT | No | Trained on Market-1501 (94.2% rank-1/82.6% mAP upstream), `fine_tuned_on_pickleball: false`. |
| Paddle/racket detector | none promoted | 3 fine-tune attempts, all **FAILED probes** | yolo11n_paddle_cpu320_e2, yolo11n_paddle_a100_img960_e50, yolo26s_paddle_a100_img1280_e80 — all explicitly "must not be promoted," trained on the CVAT paddle dataset, unused anywhere in `best_stack.json`. |
| **BODY (Fast-SAM-3D-Body / SAM-3D-Body-DINOv3)** | external checkpoint, `available_on_h100` | N/A — not trainable by this project | Used as-is; a downstream fusion layer (Track I placement_trajectory_refine, adopted 2026-07-16) sits after it but does not retrain the backbone. |

---

## 3. THE GAP MATRIX + THE NEVER-QUEUED LIST

**No verbatim gap-matrix artifact was supplied with the task inputs or found on disk** (repo-wide search for "gap matrix" / `*gap*` / `*regroup*` returned no matching document). The table below is reconstructed strictly from the organizational-miss statements already present in modelMap/labelLedger; it is not a quote of a pre-existing file.

### Reconstructed gap matrix (asset × consuming model)

| Data asset | Size/count | Model that could use it | Consumed today? | Why not |
|---|---|---|---|---|
| roboflow_universe images | 80,967 core_pickleball samples/59 sources | Ball/court/paddle/person pretraining | No | Not referenced by any `best_stack.json` entry or training lane found; sits as raw aggregation only |
| pb.vision teacher event corpus | 4,637 events / 7,314 projected windows | Event head | No | `LOADER_CHANGE_REQUIRED.diff` unapplied; also 0/7,314 windows decodable (media VM-only) |
| pb.vision gallery raw video (11 of 13) | 2.39GB | any video-consuming trainer | No | Physically absent from Mac; VM-only |
| event_public: f3set, golfdb, squash_audio_figshare, shuttlecock_hitting_zenodo, padeltracker100, extended_openttgames | fetched 2026-07-13, not counted in the 74,546-event builder total | Event head | No | Fetched/surveyed but not read by `build_public_manifest` |
| Ball corpus 2,388→3,026-row states | 2,388 / 3,026 rows | Ball detector fine-tune | Partially — 3,026-row training ran (w7_ballretrain2) but its contamination finding was never ruled | The only landed/promoted arm used the older 1,121-row state; the newer states' training run has no report.md and is absent from NORTH_STAR_ROADMAP.md/RUNBOOK.md — **UNDETERMINED per doc-of-record** |
| 102 owner event labels | 102 rows | Event head fine-tune (T11) | No | Blocked on pretrain checkpoint being 0-detection at ruling time; no post-2026-07-19 fine-tune result found |
| CVAT court_diversity_20260712 | 100 frames/28 sources | Court keypoint net | No | No export dir found at `cvat_upload/exports/court_diversity_20260712/`; no mention of tasks 88–91 in NORTH_STAR_ROADMAP.md; last recorded status (memory, 2026-07-12/13) was "staged/READY, not yet done," and no later completion evidence exists |
| court_keypoints_20260707 CVAT pack | 6 images/6 sources | Court keypoint net | No | Real-venue labeling continues independently but no real-data retrain of the network is recorded anywhere |
| ImageNet pretrain weights for court U-Net encoder | 1 file (`resnet34-b627a593.pth`) | Court keypoint net | No | File does not exist on disk; launch script silently dropped `--encoder-weights-path`, so the encoder trained from random init |
| data/testclips | 4 entries, metadata only | any | No | No video files ever landed; no location recorded for re-acquisition |
| data/online_harvest_20260712 raw video | deleted post-extraction | any | No | By design — segments deleted immediately after frame extraction; would require re-running yt-dlp |
| Paddle CVAT dataset | used in 3 fine-tune attempts | Paddle detector | Attempted, all failed | All 3 probes explicitly "must not be promoted"; no retry recorded |
| 11,459 person boxes | 4 protected clips | Person detector fine-tune | No | Only exists on protected/holdout clips; CVAT closed at API level to further person-box tasks |

### Never-queued list (organizational misses, stated plainly)
- **The 2,388-row and 3,026-row ball-corpus states exist and have training results, but NORTH_STAR_ROADMAP.md still narrates the older 1,750-row / 1,121-row figures** — the roadmap text was not updated to reflect on-disk lane output.
- **w7_ballretrain2_20260709 (the 3,026-row training run and its contamination finding) is not referenced in any governing doc** (NORTH_STAR_ROADMAP.md or RUNBOOK.md) — it has no ruling, no promotion decision, and no rejection decision on record. It is simply absent from doc-of-record, not resolved.
- **CVAT tasks 88–91 (court_diversity_20260712, 100 images, imported 2026-07-12)** have no completion evidence and no export directory, and are not mentioned in the North Star at all — an owner-time item that appears to have been staged and then dropped from tracking rather than explicitly deferred.
- **6 of the 7 fetched public event datasets under `data/event_public_20260713`** (f3set, golfdb, squash_audio_figshare, shuttlecock_hitting_zenodo, padeltracker100, extended_openttgames) were fetched and inventoried on 2026-07-13 but were never wired into the event-head training builder — fetched, then never queued into a pipeline.
- **The pb.vision teacher corpus's loader-change diff sits unapplied** — the integration work to make the 4,637-event corpus trainable was drafted (as a diff) but never applied/merged.
- **The court U-Net's missing ImageNet encoder weights** (`models/checkpoints/court_external/torchvision/resnet34-b627a593.pth`) were never fetched, silently degrading the launch script to random-init rather than surfacing as a blocking issue.
- **`roboflow_universe_20260706`'s 80,967-image aggregation (6.9GB, the single largest data asset by sample count) has no consuming training lane identified anywhere in modelMap** — it was harvested and aggregated but never queued into any pretraining step.
- **11 of 13 pb.vision videos' pixels were never transferred off the VM to a location the training pipeline can read from** — the corpus manifest was built assuming media presence and only later revealed 0/7,314 windows are decodable.

---

## 4. OWNER LABELS + THEIR USES

| Owner label asset | Count | Stated use | Actual status today |
|---|---|---|---|
| Event labels (`event_labels_owner_20260719`, session `event_labels_20260715`) | 102 rows (60 typed events, 42 hard negatives, 46 w/coords, 57 w/dt) | "TRAINING-candidate for the pickleball event-head fine-tune. NOT eval." | Banked; blocked (T11) — pretrain checkpoint was 0-detection at ruling time; must be checked for zero overlap with the protected 50-row seed before use; no post-2026-07-19 fine-tune result found |
| Protected 50-row event eval seed (`spot_check_tier_a_50.json` / owner_spot_check_results_20260715.json) | 29 typed contacts + 21 hard negatives = 50 | Eval-only, permanently protected, never training | Code-enforced: `finetune_event_head.py` hardcodes this path and calls `_reject_protected_rows()` to refuse training on any overlapping row. Owner spot-check (2026-07-15) of Tier-A audio auto-labels against it **FAILED 29/50 vs ≥47/50 bar** — this failure is why Tier-A audio bootstrap labels were rejected for training |
| Ball corpus (CVAT-reviewed) | 1,121 → 1,750 → 2,388 → 3,026 rows | LOSO/candidate eval and fine-tune input | Only the 1,121-row state was used in a promoted-to-candidate fine-tune arm (`ball.seed_official_checkpoint`, PENDING, not default). The 3,026-row state's training run surfaced a 74.8%-of-corpus contamination finding with no doc-of-record ruling |
| 15-pt court calibration review (`metric_15pt_reviewed`) | 1 reviewed seed | Consumed directly by the static-camera CAL solve path | Completed 2026-07-17 (commit 1075cee57). Has a **named, unfixed defect**: measures 19.16px median error vs a 2.61px line-evidence solve on the same video, because it was fit with a zero-distortion config against a k1=-0.28 camera (homography fit on distorted pixels) — ruled a config fix, not a fundamental limit, not yet fixed as of the roadmap text read |
| CVAT tasks 88–91 (court_diversity_20260712) | 100 images/28 sources, 4 shard tasks of 25 each | Court-keypoint labeling for real-venue diversity | Imported 2026-07-12; no export found; last known status (2026-07-12/13) "staged/READY, ~45–60 min owner time," not confirmed done; not mentioned in North Star |
| 32-row reviewed real court label set | 32 rows | Reserved strictly for the CAL promotion **eval** gate | Never used for training, per design |
| Person boxes | 11,459 | Human-reviewed detection ground truth | Exist only on the 4 protected eval_clips/ball videos (2 = frozen scoring card, 2 = strict holdout); **zero usable for training**; CVAT closed at API level to any further person-box tasks |
| CVAT w5/w6/w7 ball labelpacks (disagreement-stratified) | w5=640 frames/16 clips, w6=640 frames/22 clips, w7_audit_stratum=350 frames/32 clips (uniform-random scratch) | Feed the ball corpus growth chain above | Rolled into the ingest lanes (w6_labelingest → w7_ballingest2/3/4) described in §1.10 |

---

## 5. EFFORT AUDIT of the last 3 days (2026-07-19 to 2026-07-21, per dated artifacts found)

*Compiled strictly from dated lane/asset artifacts present in the supplied inventories — this is not a time-tracked log, so gaps in coverage are possible.*

**2026-07-19**
- `data/event_labels_owner_20260719` banked (102-row owner event export) — **committed** (data landed), consumption **blocked**.
- `runs/lanes/event_head_corpus_20260719` (`LOCAL_EVIDENCE.json`, `T8_RULING.md`) — ruling that the event-head pretrain checkpoint was unusable (0-detection) at that time, which is why the 102-row fine-tune (T11) is blocked. **Overhead/diagnostic**, not a landing.
- `data/pbvision_gallery_20260719` scrape initiated (obtained_utc recorded as 2026-07-20T07:00:35Z, spanning the boundary) — 12 fresh pb.vision videos' JSON supervision fetched (~148MB). **Committed** as a data asset; usage still gated by `usage_posture` (no ball-detector training without a manager ruling).

**2026-07-20**
- `runs/lanes/pbv_corpus_rebuild_20260720` + `runs/lanes/pbv_pickleball_corpus_20260720` — built the 4,637-event teacher-pseudo-label corpus from pb.vision's own outputs. **Overhead-heavy**: explicitly VERIFIED=0, training_ready=false, **0 of 7,314 projected windows decodable** (media not staged), and the loader integration diff was left unapplied. No promotion, no rejection — parked mid-integration.
- `runs/lanes/event_head_retrain_20260720` — GPU (A100) training run, resumed to step 16918 with class-weighted loss. **Partially committed**: the class-weighting fix itself is a validated finding (escapes all-negative collapse, TP 44/70/107 vs 0 baseline) but the checkpoint is explicitly **not promoted** ("does not yet produce a usable detector," VERIFIED=0) — net effort output is a negative/inconclusive result plus one reusable technique.
- `data/pbv_replay_20260720` — one additional gallery video (xkadsq9bli3h, 186.15s) staged with its actual MP4 onto the Mac (59M). **Committed** (infra progress: 1 more of 13 videos now locally decodable), but 10 of 13 remain VM-only.
- Harness-bug correction in `eval_event_head.py` (hardcoded 15-frame windows vs a 64-frame-context model) — re-eval of an existing checkpoint at the corrected window size produced 9 TP/0 FP (max prob 0.937), revealing the earlier "0 TP" public number was a harness artifact, not a true model verdict. **Committed** (a correction to measurement, not a model gain).

**2026-07-21 (today)**
- This REGROUP_INPUTS compilation — an evidence-synthesis task, not a training/lane landing. **Overhead** (planning/audit work) by nature of the task itself.

### Rollup: committed vs rejected vs overhead (last 3 days)
- **Committed / landed**: pb.vision gallery JSON supervision fetch (12 videos, ~148MB); class-weighted-loss technique for event head (escapes collapse); eval-harness bug fix (corrects a false "0 TP" reading); one additional gallery video's MP4 staged to Mac; 102-row owner event export banked.
- **Rejected / explicit non-promotion**: event_head_retrain_20260720 checkpoint (VERIFIED=0, "not yet a usable detector"); pb.vision teacher corpus not training-ready (0/7,314 windows decodable).
- **Overhead / blocked / parked**: T11 owner fine-tune blocked on pretrain-checkpoint status; pb.vision teacher-corpus loader-integration diff unapplied; 11 of 13 gallery videos still VM-only, blocking corpus decodability entirely.

*Not independently re-verified in this pass: whether any lane activity occurred on 2026-07-19–21 outside the specific dated artifact paths enumerated in the supplied inventories (e.g., CVAT task 88–91 owner labeling time, if any occurred in this window, was not evidenced by a dated artifact and is not counted above).*

---

## 6. IDEA POOL per track (all proposals, deduped, with effort/risk — no ranking, no strategy decision)

*Proposals below are those that surface in the evidence as unresolved next-steps, unapplied diffs, unfixed named defects, or explicitly-parked decisions. Effort/risk figures are given only where the source material states them; otherwise marked "not stated."*

### BALL
- **Rule on the w7_ballretrain2_20260709 contamination finding** (26/40 clips / 74.8% of the 3,026-row corpus flagged contaminated) and decide whether the corpus needs re-cleaning before any further fine-tune. Effort/risk: not stated; currently UNDETERMINED per doc-of-record.
- **Reconcile NORTH_STAR_ROADMAP.md's stale 1,750/1,121-row DATA figures** with the on-disk 2,388/3,026-row states. Effort: doc-only, low. Risk: low (documentation correctness only).
- **Decide whether to promote `ball.seed_official_checkpoint`** (currently PENDING, trained on the 1,121-row state, +15.5% relative over prior checkpoint) to default. Effort/risk not stated in source.
- **Re-run a clean fine-tune on the decontaminated subset** (14 clean clips identified in the contamination finding, F1 0.236) rather than the contaminated 26 clips. Effort/risk not stated.

### EVENT HEAD
- **Apply the `LOADER_CHANGE_REQUIRED.diff`** to wire the 4,637-event pb.vision teacher corpus into `datasets.py`/`finetune_event_head.py`. Effort: a diff already exists (drafted), so mechanically low; risk: corpus is VERIFIED=0/teacher-derived (noisy weak supervision), so training on it risks reinforcing pb.vision's own model errors rather than ground truth.
- **Stage the 11 VM-only pb.vision gallery videos to a decodable location** so the 7,314 projected teacher-event windows (currently 0/7,314 decodable) become usable. Effort: data-transfer only (2.39GB), not stated in GPU-hours; risk: low, mechanical.
- **Run the 102-row owner fine-tune (T11)** now that a class-weighted checkpoint exists that escapes all-negative collapse (unblock condition per T8_RULING.md may now be reconsidered given 2026-07-20 progress). Effort/risk not stated; note the fine-tune-authorization gate requires median G_val ≥ +0.10 but < 0.80 macro-F1 before further owner labeling is authorized.
- **Wire the 6 unused public event datasets** (f3set, golfdb, squash_audio_figshare, shuttlecock_hitting_zenodo, padeltracker100, extended_openttgames — already fetched, 2026-07-13) into the event-head builder alongside the 3 currently consumed. Effort: builder-code change (scope not stated); risk: cross-sport transfer already showed a zero-shot tennis→pickleball FAILURE (degenerate near-constant-HIT predictor, audio cross-check at chance), so added sports may not transfer either.
- **Continue precision reduction** on the class-weighted checkpoint (currently 466 FP vs 44 TP at tolerance 1) — no specific technique proposed in evidence beyond noting the FP rate is the blocker.

### COURT
- **Fetch the missing ImageNet ResNet34 encoder weights** (`resnet34-b627a593.pth`) and re-run the U-Net training with proper pretraining instead of random init. Effort: a single file download; risk: low.
- **Run a real-data retrain** of the court keypoint U-Net (currently 100% synthetic, 0 real rows) using the 100 court_diversity + 6 court_keypoints CVAT frames already labeled. Effort/risk not stated; note current synthetic-only gate result (PCK@5px 0.7512 vs 0.95 bar) already misses the promotion bar, so it's unclear if 106 real frames alone would close the gap.
- **Confirm/complete CVAT tasks 88–91** (100 court_diversity images, ~45–60 min owner time per prior estimate) — status currently unconfirmed as done.
- **Fix the 15-pt calibration solve's distortion-config bug** (zero-distortion homography fit against a k1=-0.28 camera, producing 19.16px median error vs an achievable 2.61px on the same video) — already ruled a config fix, not a fundamental limit, but not yet applied per the roadmap text read. Effort: config change, low; risk: low.

### PERSON / TRACKING
- **Decide RF-DETR-L promotion** — improves burlington (IDF1 0.8831→0.9220) but regresses wolverine (0.8516→0.8036); currently `do_not_promote=true`. Effort/risk: evaluation already done; decision only.
- **Acquire more person-box training data** — currently blocked structurally (CVAT closed at API level to further person-box tasks beyond the 4 protected clips; 11,459 boxes are eval-only). Any proposal here requires first reopening CVAT capacity for this class — not currently queued.
- **Retry a paddle/racket detector fine-tune** with a different recipe or more data, given all 3 prior probes (yolo11n cpu320, yolo11n a100 img960, yolo26s a100 img1280) failed and were explicitly marked must-not-promote. Effort/risk not stated.

### DATA / INFRA (cross-cutting)
- **Consider whether/how to use the 80,967-image roboflow_universe aggregation** for any pretraining step — currently has zero identified consumer in modelMap. Effort/risk not stated.
- **Re-harvest `online_harvest_20260712` raw video** if full-frame (not just extracted-segment) access is ever needed — requires re-running yt-dlp per source; license/availability not re-verified.
- **Resolve the pb.vision `iottnc0h3ekn` fps discrepancy** (camera.fps field says 30.0 vs api_get_metadata reporting 59.94fps) before using that video's frame-rate-dependent fields. Effort: investigation only, low.
- **Determine per-video pb.vision detection-confidence threshold** — no repo document defines what confidence pb.vision itself uses to call a frame a "real" detection; needed before treating dense per-frame actions.ball outputs as discrete labels.
