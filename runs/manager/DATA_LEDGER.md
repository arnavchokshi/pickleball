<!-- GENERATED, do not hand-edit. Source: runs/manager/data_ledger.json via scripts/racketsport/audit_data_utilization.py. -->
# Data Ledger (generated view)

This is a coordination view for data lineage and utilization only. `NORTH_STAR_ROADMAP.md` remains product truth.

- Ledger schema: `2`
- Snapshot UTC: `2026-07-22T05:25:39Z`
- Assets: `27`
- States: `BLOCKED=9`, `CONSUMED=6`, `QUARANTINED=6`, `REJECTED=6`

| Asset ID | State | Bytes | Raw | Kept | Decoded | Labels | Authority | Owner | Next check |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `ball_reviewed_corpus_chain_1121_3026` | `QUARANTINED` | 16450892 | 3026 reviewed_rows | 3026 current_corpus_rows | 3026 image_label_rows | 3026 BALL_labels | corrected_prelabel, confirmed_prelabel, human_gt | BALL data owner | Reconcile all 350 scratch labels, hold out HyUqT7zFiwk/Ezz6HDNHlnk by parent source, and issue an explicit contamination ruling. |
| `court_diversity_100_20260712` | `BLOCKED` | 170624051 | 100 frame_images | 97 potentially_allowed_after_source_denial | 100 frame_images | 0 reviewed_labels | none | COURT data owner | Export tasks 88-91, deny exactly three IYnbdRs1Jdk frames before label read, then report reviewed/usable/train/holdout/rejected counts. |
| `court_keypoints_6_20260707` | `QUARANTINED` | 28412080 | 6 reviewed_images | 3 fully_usable_rows | 6 frame_images | 6 human_review_outcomes | human_gt | COURT independent eval owner | Score the frozen control and eligible challenger once on the three usable rows after the eight-source holdout, without tuning. |
| `eval_clips_ball_protected_4` | `QUARANTINED` | 24804487 | 4 protected_video_clips | 4 protected_video_clips | 4 video_clips | 11459 human_person_boxes | human_gt | independent cross-component eval owner | Require every new training exporter to prove exhaustive zero collision against all four clips and their derivatives. |
| `event_abc_inputs_20260720` | `CONSUMED` | 119496 | 102 owner_rows | 102 owner_rows | 102 64_frame_windows | 102 human_labels | human_gt | EVENT training owner | Reuse unchanged for corrected B/C only; keep validation rows evaluator-only. |
| `event_abc_vm_pull_20260721` | `REJECTED` | 48755968 | 2192 agreement_decisions | 1481 accepted_arm_b_rows_under_old_rule | 1189 rows_with_non_audio_agreement | 1481 teacher_rows_in_invalid_arm | teacher | EVENT method-audit owner | Build new B/C manifests that require ball_velocity_kink; do not mutate or reuse these rejected artifacts. |
| `event_bootstrap_audio_20260713` | `REJECTED` | 15570255 | 3173 jsonl_windows | 3173 jsonl_windows | 3173 materialized_windows | 3173 teacher_windows | teacher | EVENT data owner | Retain only as immutable negative evidence or as a weighted auxiliary cue after a non-audio physical agreement. |
| `event_public_extended_opentt_20260713` | `BLOCKED` | 4174806880 | 4945 event_rows | 4945 event_rows | 0 verified_64_frame_windows | 4945 event_rows | human_gt | EVENT public-data owner | Run inventory-only mapping against game_4/test_2; queue CPU only if source-disjoint compatible windows are nonzero. |
| `event_public_f3set_20260713` | `BLOCKED` | 29126284 | 43655 event_rows | 43655 event_rows | 0 64_frame_windows | 43655 event_rows | human_gt | EVENT public-data owner | Run CPU inventory-only adapter and record BLOCKED_NO_PIXELS unless context resolves. |
| `event_public_golfdb_20260713` | `BLOCKED` | 1974078 | 11200 event_rows | 11200 event_rows | 0 64_frame_windows | 11200 event_rows | human_gt | EVENT public-data owner | Run inventory-only adapter and record BLOCKED_NO_PIXELS if reproduced. |
| `event_public_padeltracker100_20260713` | `BLOCKED` | 670150891 | 13250 shot_flagged_frames | 906 contiguous_shot_windows | 0 64_frame_windows | 13250 shot_flagged_frames | human_gt | EVENT public-data owner | Run inventory-only adapter; no GPU from label rows alone. |
| `event_public_shuttlecock_zenodo_20260713` | `REJECTED` | 20343257 | 8 sample_video_clips | 8 sample_video_clips | 0 structured_event_rows | 0 structured_event_rows | none | EVENT public-data owner | No training queue; part1 may be inventoried only if the expected label file is named first. |
| `event_public_shuttleset_20260713` | `BLOCKED` | 129883486 | 88840 event_rows | 88840 event_rows | 0 64_frame_windows | 88840 event_rows | human_gt | EVENT public-data owner | Run inventory-only adapter and preserve BLOCKED_NO_PIXELS unless a lawful context source resolves. |
| `event_public_squash_figshare_20260713` | `REJECTED` | 138216934 | 1 audio_parts | 1 audio_parts | 0 structured_event_rows | 0 structured_event_rows | none | EVENT public-data owner | No training queue; reopen only if an authoritative structured-label release appears. |
| `event_public_tt_sounds_20260713` | `REJECTED` | 17006390 | 5702 labeled_audio_rows | 5702 labeled_audio_rows | 5702 audio_snippets | 5702 audio_event_rows | human_gt | EVENT public-data owner | Retain as negative evidence; do not queue to typed event GPU training. |
| `online_harvest_20260706` | `CONSUMED` | 2487780499 | 25 candidate_sources | 8 downloaded_sources | 8 source_videos | 0 raw_human_labels | none | BALL/CAL data owner | Record source license rulings before any commercial-bound training. |
| `online_harvest_20260712` | `CONSUMED` | 170641232 | 39 attempted_sources | 28 selected_sources | 100 derived_frame_images | 0 reviewed_labels | none | COURT data owner | Preserve source-family IDs during CVAT export and deny all three IYnbdRs1Jdk frames before label read. |
| `owner_event_labels_102_20260719` | `CONSUMED` | 12872 | 102 owner_answers | 102 owner_answers | 102 resolved_64_frame_windows | 102 human_labels | human_gt | EVENT data owner | Keep the 61/41 split frozen for corrected B/C; do not relabel validation as training. |
| `owner_img_1605_court_review_20260721` | `QUARANTINED` | 520522 | 3 review_frames | 3 review_frames | 3 review_frame_images | 15 reviewed_court_keypoints | human_gt | COURT independent eval owner | Keep separate from training exporters and preserve the exact owner source identity. |
| `pbv_pickleball_teacher_events_20260720` | `BLOCKED` | 8881500 | 7314 projected_windows | 7314 source_disjoint_windows | 0 locally_decodable_windows | 4637 teacher_events | teacher | EVENT data owner | Materialize corrected non-audio-agreement B/C rows after all media and PTS hashes are local. |
| `pbvision_gallery_20260719` | `CONSUMED` | 325537988 | 13 video_ids | 13 video_ids | 2 local_videos | 124743 dense_teacher_frame_rows | teacher | DATA steward | Stage and hash the eleven VM-only MP4s; keep the three compare IDs unread by trainers. |
| `protected_event_seed_50_20260713` | `QUARANTINED` | 212768 | 50 protected_rows | 50 protected_rows | 50 reviewed_windows | 50 human_labels | human_gt | independent EVENT eval owner | No read until the frozen owner-41 A/B/C gate passes and the atomic one-touch claim is acquired. |
| `roboflow_ball_core_pretrain_20260706` | `CONSUMED` | 2174383220 | 34658 core_pickleball_ball_rows | 34658 indexed_ball_images | 34658 indexed_ball_images | 44458 ball_boxes | human_gt | BALL public-data owner | Retain as negative evidence only; no repeat because transfer regressed and NC rows were consumed. |
| `roboflow_person_adjacent_20260706` | `REJECTED` | 780082673 | 29036 pre_dedup_adjacent_samples_all_classes | 15469 adjacent_person_images | 15469 indexed_images | 15469 person_images_with_boxes | human_gt | PERSON public-data owner | Do not queue in the regroup product-domain arm. |
| `roboflow_person_core_20260706` | `BLOCKED` | 872492382 | 15334 core_pickleball_person_rows_before_rights_exclusion | 15312 commercial_clean_core_person_rows | 15312 selector_bound_indexed_images | 47044 person_boxes | human_gt | PERSON public-data owner | Audit precision/recall and prove zero protected collisions before any diagnostic GPU. |
| `roboflow_person_nc_20260706` | `QUARANTINED` | 548850 | 34 person_index_rows | 22 unique_images | 22 indexed_images | 22 person_images_with_boxes | human_gt | DATA rights owner | Keep source_slug on every exporter denylist. |
| `w7_audit_stratum_scratch_350` | `BLOCKED` | 566895615 | 350 frame_images | 350 uniform_random_scratch_frames | 350 frame_images | 0 reviewed_labels | none | BALL independent-review owner | Finish/export 350/350 scratch labels, reconcile lineage, and prove zero protected collisions before any BALL GPU. |

## Per-asset rulings and utilization

### `ball_reviewed_corpus_chain_1121_3026`

- State reason: UNRULED contamination finding: the 3026-row run was consumed, but the blended metric cannot decide a model and the corpus is quarantined pending source-held reconciliation.
- Paths: `runs/lanes/w7_ballingest4_20260709`, `runs/lanes/w7_ballretrain2_20260709`
- Source families: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=[]
- Overlap coverage: FAIL — all rows have source IDs; no immutable uncontaminated selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 1

### `court_diversity_100_20260712`

- State reason: Tasks 88-91 were imported, but no authoritative reviewed export exists; only 97 are potentially allowed before quality rejection.
- Paths: `cvat_upload/court_diversity_20260712`
- Source families: 1or-bXVM80M, 3sC53GlvW_s, 4qSoA-jwpVM, 6HqhDYnYcFU, 90UfH4QRSIg, A9H6EWfXht0, AXv70OhwStI, C5YUQlqZqBY, FGbGQgKPCfQ, IYnbdRs1Jdk, O29MdlviAqI, Se7M6ZKaC4Y, X42ktPBs140, Y6vE9AtIH4M, _4g4JnkG91Y, a_HzWrwK6vM, iMKDCfjfBNU, jr_60WVlG4c, kS_LzZGkdkg, ltIxlS0QJhg, n5D0179RNoA, q3575jnmjJQ, t9aeCo9I7Aw, uf0XWI7zZoY, wv3aPJrDwK4, y3c71Q0E1nE, z7PctOec41w, zXp76RZ0aVU
- Partition: train=['3sC53GlvW_s', '6HqhDYnYcFU', '90UfH4QRSIg', 'AXv70OhwStI', 'FGbGQgKPCfQ', 'O29MdlviAqI', 'X42ktPBs140', 'Y6vE9AtIH4M', '_4g4JnkG91Y', 'iMKDCfjfBNU', 'jr_60WVlG4c', 'kS_LzZGkdkg', 'ltIxlS0QJhg', 'n5D0179RNoA', 't9aeCo9I7Aw', 'uf0XWI7zZoY', 'y3c71Q0E1nE', 'z7PctOec41w', 'zXp76RZ0aVU']; val=['1or-bXVM80M', '4qSoA-jwpVM', 'C5YUQlqZqBY', 'A9H6EWfXht0', 'Se7M6ZKaC4Y', 'a_HzWrwK6vM', 'q3575jnmjJQ', 'wv3aPJrDwK4']; test=[]
- Overlap coverage: PARTIAL — three known protected-family derivatives denied; no immutable 97-frame clean subset selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 0

### `court_keypoints_6_20260707`

- State reason: Three usable rows are frozen external audit; three are rejected. None may train the preview challenger.
- Paths: `cvat_upload/court_keypoints_20260707`
- Source families: HyUqT7zFiwk, 73VurrTKCZ8, wBu8bC4OfUY, Ezz6HDNHlnk, zwCtH_i1_S4, _L0HVmAlCQI
- Partition: train=[]; val=[]; test=['court_keypoints_20260707:3_usable_external_audit']
- Overlap coverage: PARTIAL — source identity only
- Immutable clean-subset selectors: 0
- Consumers: 0

### `eval_clips_ball_protected_4`

- State reason: Four clips, 120 BALL points, and 11459 PERSON boxes are protected evaluation only and unreachable by trainers.
- Paths: `eval_clips/ball/burlington_gold_0300_low_steep_corner`, `eval_clips/ball/wolverine_mixed_0200_mid_steep_corner`, `eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline`, `eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline`, `runs/cvat_imports/2026_06_30/burlington_gold_0300_low_steep_corner/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/indoor_doubles_fwuks_0500_long_mid_baseline/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/outdoor_webcam_iynbd_1500_long_high_baseline/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/wolverine_mixed_0200_mid_steep_corner/person_ground_truth.json`
- Source families: burlington_gold_0300_low_steep_corner, wolverine_mixed_0200_mid_steep_corner, outdoor_webcam_iynbd_1500_long_high_baseline, indoor_doubles_fwuks_0500_long_mid_baseline
- Partition: train=[]; val=[]; test=['burlington_gold', 'wolverine_mixed', 'outdoor_webcam_iynbd', 'indoor_doubles_fwuks']
- Overlap coverage: PARTIAL — all four source videos registered; new trainer inputs must prove exhaustive derivative non-collision
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_abc_inputs_20260720`

- State reason: The frozen A input ran to 1000 steps and was recovered with exact hash evidence; the 41 validation rows were not loaded for gradient updates.
- Paths: `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json`
- Source families: video_sha256 and clip IDs embedded per row
- Partition: train=['owner_102_manifest:61_rows']; val=['owner_102_manifest:41_rows']; test=[]
- Overlap coverage: PASS — all training windows checked against protected 50; overlap_rows=0
- Immutable clean-subset selectors: 0
- Consumers: 1

### `event_abc_vm_pull_20260721`

- State reason: METHOD_INVALID_AUDIO_ONLY=292; B/C partial outputs were killed and preserved for forensics, never scoreable causal evidence.
- Paths: `runs/lanes/abc_experiment_20260721/vm_pull`
- Source families: ten non-compare pb.vision video families plus frozen owner manifest
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — pulled files verified; method validity failed independently
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_bootstrap_audio_20260713`

- State reason: Owner review measured 29/50 true contacts against a >=47/50 gate; Tier-A audio auto-labels are rejected for training.
- Paths: `data/event_bootstrap_20260713`
- Source families: pbvision_11min_20260713, 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=['pbvision_11min_20260713:diagnostic_only']
- Overlap coverage: PARTIAL — Tier-A quality check only; exhaustive protected collision scan not recorded for all bootstrap rows
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_extended_opentt_20260713`

- State reason: 4945 labels exist, but verified decodable 64-frame windows and semantic mapping are still zero in this ledger snapshot.
- Paths: `data/event_public_20260713/extended_openttgames`, `data/event_public_20260713/openttgames/videos`
- Source families: Extended OpenTTGames, OpenTTGames game_4, OpenTTGames test_2
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — event-to-local-video mapping and source overlap not yet inventoried
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_f3set_20260713`

- State reason: Labels-only asset with 0 decodable 64-frame windows.
- Paths: `data/event_public_20260713/f3set`
- Source families: F3Set tennis singles/doubles and table-tennis labels
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no pixels available for source-family collision audit
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_golfdb_20260713`

- State reason: Labels-only asset with 0 decodable 64-frame windows.
- Paths: `data/event_public_20260713/golfdb`
- Source families: GolfDB
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no local pixels
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_padeltracker100_20260713`

- State reason: Labels exist but no locally resolvable 64-frame source context; semantics are intervals rather than contact points.
- Paths: `data/event_public_20260713/padeltracker100`
- Source families: 2022_BCN_FinalM_1, 2022_BCN_FinalF_1
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no source pixels
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_shuttlecock_zenodo_20260713`

- State reason: Fetched sample contains video only; no independent event GT was found.
- Paths: `data/event_public_20260713/shuttlecock_hitting_zenodo`
- Source families: Zenodo 14677727 part2 sample clips 00170-00177
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — eight sample clips with no structured labels
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_shuttleset_20260713`

- State reason: Label-only badminton asset with 0 decodable windows and uncleared broadcast pixels.
- Paths: `data/event_public_20260713/coachai_shuttleset`
- Source families: ShuttleSet, ShuttleSet22
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no local pixels
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_squash_figshare_20260713`

- State reason: One audio file is present but zero distributable structured event rows were found.
- Paths: `data/event_public_20260713/squash_audio_figshare`
- Source families: Figshare article 5962015
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — one raw audio part with zero structured labels
- Immutable clean-subset selectors: 0
- Consumers: 0

### `event_public_tt_sounds_20260713`

- State reason: Audio-only snippets cannot supply 64-frame visual context or typed contact truth.
- Paths: `data/event_public_20260713/tt_sounds_data`
- Source families: TT Sounds
- Partition: train=['tt_sounds:4561_rows']; val=[]; test=['tt_sounds:1141_rows']
- Overlap coverage: PASS — all registered label rows rejected from typed-event training
- Immutable clean-subset selectors: 0
- Consumers: 0

### `online_harvest_20260706`

- State reason: Six of eight downloaded source families supplied the reviewed BALL corpus and other staged review packs.
- Paths: `data/online_harvest_20260706`
- Source families: HyUqT7zFiwk, 73VurrTKCZ8, wBu8bC4OfUY, Ezz6HDNHlnk, zwCtH_i1_S4, _L0HVmAlCQI, pwxNwFfYQlQ, vQhtz8l6VqU
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=[]
- Overlap coverage: PARTIAL — original source identities; not every extracted frame against every protected frame
- Immutable clean-subset selectors: 0
- Consumers: 1

### `online_harvest_20260712`

- State reason: The selected source segments were consumed into 100 still frames, then intentionally deleted; reacquisition is required for video context.
- Paths: `data/online_harvest_20260712`, `cvat_upload/court_diversity_20260712`
- Source families: 28 selected YouTube source IDs recorded in court package
- Partition: train=['19 allowed court source groups in court_diversity row']; val=['8 frozen court holdout groups']; test=[]
- Overlap coverage: PARTIAL — 100 staged frames; no immutable 97-frame clean subset selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 1

### `owner_event_labels_102_20260719`

- State reason: Arm A loaded the frozen 61 training rows; the 41 validation rows remained evaluation-only and scored 0.0.
- Paths: `data/event_labels_owner_20260719`
- Source families: owner event query windows
- Partition: train=['owner_102_manifest:61_rows']; val=['owner_102_manifest:41_rows']; test=[]
- Overlap coverage: PASS — all owner training windows; two windows shifted; overlap_rows=0
- Immutable clean-subset selectors: 0
- Consumers: 1

### `owner_img_1605_court_review_20260721`

- State reason: The owner court-review derivatives are bound separately from eval_clips_ball_protected_4 and remain evaluation-only.
- Paths: `eval_clips/ball/owner_IMG_1605_8a193402780b`
- Source families: owner_IMG_1605_8a193402780b
- Partition: train=[]; val=[]; test=['owner_IMG_1605_8a193402780b']
- Overlap coverage: PASS — all six derivative files
- Immutable clean-subset selectors: 0
- Consumers: 0

### `pbv_pickleball_teacher_events_20260720`

- State reason: 0/7314 windows are locally decodable and the recovered B/C materialization accepted 292 audio-only rows, so no valid training consumption is recorded.
- Paths: `runs/lanes/pbv_pickleball_corpus_20260720`
- Source families: 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, xkadsq9bli3h, pldtjpw3h0jw, utasf5hnozwz, 0tmdeghtfvjx
- Partition: train=['143sf3gdwxsa', '98z43hspqz13', 'bewqc0glhgpq', 'st0epgnab7dr', 'td2szayjwtrj', 'tqjlrcntpjvt', 'xkadsq9bli3h']; val=['pldtjpw3h0jw', 'utasf5hnozwz']; test=['0tmdeghtfvjx']
- Overlap coverage: PARTIAL — ten corpus video families; declared corpus path is not bound to an immutable clean subset selector
- Immutable clean-subset selectors: 0
- Consumers: 0

### `pbvision_gallery_20260719`

- State reason: All 13 JSON exports were inventoried and ten entered the teacher materializer; local pixel availability remains only 2/13.
- Paths: `data/pbvision_gallery_20260719`, `data/pbvision_11min_20260713/source_video.mp4`, `data/pbv_replay_20260720/xkadsq9bli3h/max.mp4`, `eval_clips/ball/pbvision_11min_20260713`
- Source families: 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, xkadsq9bli3h, pldtjpw3h0jw, utasf5hnozwz, 0tmdeghtfvjx, 83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr
- Partition: train=['143sf3gdwxsa', '98z43hspqz13', 'bewqc0glhgpq', 'st0epgnab7dr', 'td2szayjwtrj', 'tqjlrcntpjvt', 'xkadsq9bli3h']; val=['pldtjpw3h0jw', 'utasf5hnozwz']; test=['0tmdeghtfvjx', '83gyqyc10y8f', 'iottnc0h3ekn', 'o4dee9dn0ccr']
- Overlap coverage: PARTIAL — all registered video IDs; no immutable clean subset selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 1

### `protected_event_seed_50_20260713`

- State reason: Protected evaluation identity; the prior owner spot check failed 29/50 versus the >=47/50 Tier-A bar, but the set is not reusable for threshold shopping.
- Paths: `runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json`, `runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json`
- Source families: protected event window source identities in spot_check_tier_a_50.json
- Partition: train=[]; val=[]; test=['protected_event_seed_50:50_rows']
- Overlap coverage: PASS — all protected rows compared with owner training windows
- Immutable clean-subset selectors: 0
- Consumers: 0

### `roboflow_ball_core_pretrain_20260706`

- State reason: The 34658-row core BALL subset, not the 15312-row PERSON subset, supplied the historical pretrain and internal validation.
- Paths: `data/roboflow_universe_20260706/aggregated/subset_indexes/ball_index.json`
- Source families: 36 core_pickleball BALL source slugs, including the historically consumed testing-esifc NC source
- Partition: train=['roboflow_ball_core_train:32018_rows']; val=['roboflow_ball_core_internal_val:2640_rows']; test=[]
- Overlap coverage: PASS — all core BALL rows are frozen negative evidence and refused from future training
- Immutable clean-subset selectors: 0
- Consumers: 1

### `roboflow_person_adjacent_20260706`

- State reason: The intervention is in-domain pickleball; the adjacent bucket is dominated by tennis and is categorically excluded.
- Paths: `data/roboflow_universe_20260706/aggregated/subset_indexes/person_index.json`
- Source families: three adjacent-sport Roboflow sources
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — all adjacent-sport person images denied from product-domain arm
- Immutable clean-subset selectors: 0
- Consumers: 0

### `roboflow_person_core_20260706`

- State reason: The exact 15312-row PERSON selector is now immutable, but this PERSON subset has never been consumed and remains blocked on annotation and protected-collision audits.
- Paths: `runs/lanes/data_steward_ledger_20260721/person_core_commercial_15312_selector.json`
- Source families: 14 CC BY 4.0 core person sources after NC exclusion
- Partition: train=[]; val=['pickleball-od8al/pickleball-version2']; test=['hemel/pickleball-cedmo']
- Overlap coverage: NOT_RUN — new PERSON subset has not completed exhaustive protected collision proof
- Immutable clean-subset selectors: 0
- Consumers: 0

### `roboflow_person_nc_20260706`

- State reason: The source is noncommercial share-alike and permanently excluded from the commercial-bound product-domain arm.
- Paths: `data/roboflow_universe_20260706/aggregated/per_dataset/testing-esifc__pickle-ball-labeling-mff1d.index.json`
- Source families: testing-esifc/pickle-ball-labeling-mff1d
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — all raw index rows
- Immutable clean-subset selectors: 0
- Consumers: 0

### `w7_audit_stratum_scratch_350`

- State reason: All 350 images are staged and decodable, but no authoritative reviewed export exists; label_count is zero.
- Paths: `cvat_upload/w7_audit_stratum_20260709`
- Source families: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8:27', '_L0HVmAlCQI:34', 'wBu8bC4OfUY:60', 'zwCtH_i1_S4:62']; val=['HyUqT7zFiwk:100', 'Ezz6HDNHlnk:67']; test=[]
- Overlap coverage: NOT_RUN — not run; old 35-frame sample is insufficient
- Immutable clean-subset selectors: 0
- Consumers: 0
