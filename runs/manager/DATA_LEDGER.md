<!-- GENERATED, do not hand-edit. Source: runs/manager/data_ledger.json via scripts/racketsport/audit_data_utilization.py. -->
# Data Ledger (generated view)

This is a coordination view for data lineage and utilization only. `NORTH_STAR_ROADMAP.md` remains product truth.

- Ledger schema: `3`
- Snapshot UTC: `2026-07-23T16:44:55Z`
- Assets: `32`
- States: `BLOCKED=10`, `CONSUMED=7`, `DEFERRED_WITH_REASON=2`, `QUARANTINED=5`, `READY=1`, `REJECTED=7`
- License state gate: `False` (license is FYI only)
- Directive: Owner directive 2026-07-22: internal use has no licensing constraints; license metadata is FYI only and never determines ledger state. Protected-eval and compare-only restrictions remain protocol quarantines.

| Asset ID | State | Disposition | Bytes | Raw | Kept | Decoded | Labels | Authority | Owner | Next check |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| `ball_reviewed_corpus_chain_1121_3026` | `QUARANTINED` | Track B: Reconcile the 350-row scratch audit, freeze source-held partitions, and issue a contamination ruling before any Track B reuse. | 16450892 | 3026 reviewed_rows | 3026 current_corpus_rows | 3026 image_label_rows | 3026 BALL_labels | corrected_prelabel, confirmed_prelabel, human_gt | BALL data owner | Reconcile all 350 scratch labels, hold out HyUqT7zFiwk/Ezz6HDNHlnk by parent source, and issue an explicit contamination ruling. |
| `court_diversity_100_20260712` | `BLOCKED` | Track A: After the owner exports tasks 88-91, add only the protocol-eligible reviewed rows to the Track A court retrain pool with the frozen source-family split. | 170624051 | 100 frame_images | 97 potentially_allowed_after_source_denial | 100 frame_images | 0 reviewed_labels | none | COURT data owner | Owner exports tasks 88-91; adapter then denies the three IYnbdRs1Jdk frames and reports reviewed/usable/train/holdout/rejected counts. |
| `court_keypoints_6_20260707` | `QUARANTINED` | Track A: Use the three fully usable rows once as the frozen Track A external audit; never train or tune on any of the six rows. | 28412080 | 6 reviewed_images | 3 fully_usable_rows | 6 frame_images | 6 human_review_outcomes | human_gt | COURT independent eval owner | Score the frozen control and eligible challenger once on the three usable rows after the eight-source holdout, without tuning. |
| `data_testclips_metadata_4` | `DEFERRED_WITH_REASON` | RULED OUT: NO_MEDIA_ON_DISK: all four data/testclips directories contain metadata only, no source location, and unknown retrievability; evidence runs/regroup_20260721/REGROUP_INPUTS.md. | 856 | 4 metadata_entries | 4 metadata_entries | 0 video_clips | 0 labels | none | DATA steward | Reopen only if an exact retrievable source location and content identity are supplied. |
| `eval_clips_ball_protected_4` | `QUARANTINED` | Track C: Use the 11,459 reviewed person boxes only for Track C evaluation and protected-collision auditing; never train, tune, or expand labels from these four protected clips. | 24804487 | 4 protected_video_clips | 4 protected_video_clips | 4 video_clips | 11459 human_person_boxes | human_gt | independent cross-component eval owner | Require every new training exporter to prove exhaustive zero collision against all four clips and their derivatives. |
| `event_abc_inputs_20260720` | `CONSUMED` | Track D: Reuse the exact frozen 61-train/41-validation manifest for corrected exposure-matched Track D arms; validation rows stay gradient-excluded. | 119496 | 102 owner_rows | 102 owner_rows | 102 64_frame_windows | 102 human_labels | human_gt | EVENT training owner | Reuse unchanged for corrected B/C only; keep validation rows evaluator-only. |
| `event_abc_vm_pull_20260721` | `READY` | Track D: Pretrain Stage-P/E-v2 only from the SHA-bound corrected 1189-row arm_b_manifest.json and initialize model-only from the SHA-bound frozen_t20_event_head.pt; keep owner validation gradient-excluded. | 15160442 | 2192 corrected_agreement_decisions | 1189 ball_velocity_kink_qualified_rows | 1189 materialized_training_rows | 1189 teacher_rows_with_non_audio_physical_agreement | teacher | EVENT training data owner | Require a fresh passing training-input gate for the exact corrected manifest and checkpoint on every dispatch; never substitute the old vm_pull/abc_out manifest or a compare-only/protected identity. |
| `event_bootstrap_audio_20260713` | `REJECTED` | RULED OUT: REJECTED_AUDIO_ONLY_TEACHER: owner review measured 29/50 true contacts versus the >=47/50 gate at runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json; untyped audio alone is not training truth. | 15570255 | 3173 jsonl_windows | 3173 jsonl_windows | 3173 materialized_windows | 3173 teacher_windows | teacher | EVENT data owner | Retain only as immutable negative evidence; any future audio use requires independent non-audio physical agreement and a new ruled asset. |
| `event_public_extended_opentt_20260713` | `BLOCKED` | Track B: Run one controlled OpenTTGames dense-ball pretrain experiment against the official control, using only source-mapped local pixels and a frozen source-disjoint manifest; the Roboflow negative-transfer result makes the control mandatory. | 4297458956 | 52987 dense_ball_track_points | 52987 dense_ball_track_points | 7852 ball_points_with_local_game_4_or_test_2_pixels | 52987 dense_ball_points | human_gt | BALL public-data owner | Freeze the game_4/test_2 mapping and controlled Track B pretrain manifest with the official control. |
| `event_public_f3set_20260713` | `BLOCKED` | Track D: Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS if no local 64-frame context resolves. | 29126284 | 43655 event_rows | 43655 event_rows | 0 64_frame_windows | 43655 event_rows | human_gt | EVENT public-data owner | Run CPU inventory-only adapter and record BLOCKED_NO_PIXELS unless context resolves. |
| `event_public_golfdb_20260713` | `BLOCKED` | Track D: Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_LABEL_MAPPED_SOURCE_RESOLVED_PIXELS unless the local test video is authoritatively mapped to label rows and source identity. | 1974078 | 11200 event_rows | 11200 event_rows | 0 64_frame_windows | 11200 event_rows | human_gt | EVENT public-data owner | Run the inventory-only adapter and resolve label-to-video/source identity before any training use. |
| `event_public_padeltracker100_20260713` | `BLOCKED` | Track C: Audit the 906 shot-window player intervals as a PERSON/ReID auxiliary candidate; accept only if an inventory adapter proves usable player-box/pose and identity-continuity semantics, otherwise record a technical not-usable ruling. | 670150891 | 13250 shot_flagged_frames | 906 contiguous_shot_windows | 0 64_frame_windows | 13250 shot_flagged_frames | human_gt | EVENT/PERSON public-data owner | Run Track C and D inventory-only adapters; no training from interval rows alone. |
| `event_public_shuttlecock_zenodo_20260713` | `REJECTED` | Track D: Run the Track D inventory-only adapter and preserve BLOCKED_NO_STRUCTURED_EVENTS; do not infer GT from submission CSVs. | 20343257 | 8 sample_video_clips | 8 sample_video_clips | 0 structured_event_rows | 0 structured_event_rows | none | EVENT public-data owner | Inventory only; reopen training eligibility only if the expected authoritative label file is named and present. |
| `event_public_shuttleset_20260713` | `BLOCKED` | Track D: Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS unless local source context resolves. | 129883486 | 88840 event_rows | 88840 event_rows | 0 64_frame_windows | 88840 event_rows | human_gt | EVENT public-data owner | Run inventory-only adapter and preserve BLOCKED_NO_PIXELS unless local context resolves. |
| `event_public_squash_figshare_20260713` | `REJECTED` | Track D: Run the Track D inventory-only adapter and record BLOCKED_NO_STRUCTURED_EVENTS unless an authoritative structured-label file is present. | 138216934 | 1 audio_parts | 1 audio_parts | 0 structured_event_rows | 0 structured_event_rows | none | EVENT public-data owner | Inventory only; reopen training eligibility only if an authoritative structured-label release appears. |
| `event_public_tt_sounds_20260713` | `REJECTED` | Track D: Include in the Track D inventory-only corpus-expansion audit as audio-only negative evidence; do not queue it to typed visual event training. | 17006390 | 5702 labeled_audio_rows | 5702 labeled_audio_rows | 5702 audio_snippets | 5702 audio_event_rows | human_gt | EVENT public-data owner | Inventory in Track D as audio-only negative evidence; no typed event GPU training. |
| `online_harvest_20260706` | `CONSUMED` | Track A: Keep the unregistered court_calibrations directory items audit-only; use them only to reproduce and verify the frozen Track A external audit packaging, and exclude every derivative from training and tuning. | 1087546365 | 40 rally_video_clips | 40 source_grouped_rally_video_clips | 40 training_or_scoring_media_clips | 0 raw_human_labels | none | BALL/EVENT data owner | Use the training-input gate for every Stage-F dispatch and keep prelabels, court-calibration derivatives, protected media, and compare-only media outside the input manifest. |
| `online_harvest_20260712` | `CONSUMED` | Track A: Feed only protocol-eligible reviewed rows from the derived 100-frame court package into the Track A court pool through the court_diversity_100_20260712 row after owner export. | 170641232 | 39 attempted_sources | 28 selected_sources | 100 derived_frame_images | 0 reviewed_labels | none | COURT data owner | Preserve source-family IDs during owner export and deny all three IYnbdRs1Jdk frames before label read. |
| `online_harvest_person_gap_20260706` | `DEFERRED_WITH_REASON` | RULED OUT: CVAT_CLOSED_TO_PERSON_TASKS: no person-box task was created on the eight raw videos, and the route is superseded by Track C's stratified few-shot verify-not-draw pack; evidence runs/research_sota_20260722/INTERNAL_AUDIT.json and runs/research_sota_20260722/PROGRAM.md#track-c-person-identity-pose. | 50349 | 8 source_videos_without_person_task | 8 source_videos_without_person_task | 0 person_task_rows | 0 person_boxes | none | PERSON data owner | Do not reopen this route; Track C owns the few-shot pack. |
| `owner_event_labels_102_20260719` | `CONSUMED` | Track D: Reuse the frozen 61 training rows for corrected Track D fine-tuning and keep the 41 validation rows gradient-excluded. | 12872 | 102 owner_answers | 102 owner_answers | 102 resolved_64_frame_windows | 102 human_labels | human_gt | EVENT data owner | Keep the 61/41 split frozen for corrected B/C; do not relabel validation as training. |
| `owner_img_1605_court_review_20260721` | `QUARANTINED` | Track A: Retain as the Track A owner-reviewed metric audit only; never expose these derivatives to training exporters. | 520522 | 3 review_frames | 3 review_frames | 3 review_frame_images | 15 reviewed_court_keypoints | human_gt | COURT independent eval owner | Keep separate from training exporters and preserve the exact owner source identity. |
| `pbv_pickleball_teacher_events_20260720` | `BLOCKED` | Track D: Materialize corrected non-audio-agreement pretrain rows after all media and PTS hashes are local; never treat teacher rows as human GT. | 8881500 | 7314 projected_windows | 7314 source_disjoint_windows | 0 locally_decodable_windows | 4637 teacher_events | teacher | EVENT data owner | Materialize corrected non-audio-agreement B/C rows after all media and PTS hashes are local. |
| `pbv_replay_xkadsq9bli3h_20260720` | `CONSUMED` | Track B: Resume B1 SST materialization for remaining sources tqjlrcntpjvt and xkadsq9bli3h with skip-if-exists/resume support; do not arm B2 without a fresh explicit dispatch decision. | 61628656 | 1 source_video | 1 source_video | 1 ffprobe_verified_video | 0 human_labels | none | BALL/E2E data owner | Resume only the two remaining B1 sources with skip-if-exists/resume support; obtain a fresh explicit dispatch before B2. |
| `pbvision_gallery_20260719` | `CONSUMED` | Track A: After the approximately 10-frame owner-tap spot-check, add only corpus-eligible pseudo-label IDs 0tmdeghtfvjx, 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, pldtjpw3h0jw, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, utasf5hnozwz, and xkadsq9bli3h to the Track A court-keypoint retrain pool. | 263909332 | 13 video_ids | 13 video_ids | 1 local_video_owned_by_gallery_row | 124743 dense_teacher_frame_rows | teacher | DATA steward | Stage and hash the eleven VM-only MP4s, spot-check about 10 court-keypoint frames against owner taps, and keep the three compare IDs unread by trainers. |
| `person_mixed_pool_no_lift_20260722` | `REJECTED` | RULED OUT: PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED: od8al precision -0.1924 (F1 -0.0842) LOSS, hemel_test F1 +0.046 WIN; the both-families-nonnegative bar failed, and 13.5x gradient-update plus 6.75x anchor-exposure asymmetry makes the miss unattributable among three live hypotheses; evidence runs/lanes/person_mixed_20260722/gpu_phase_report.json and ORCHESTRATOR_STATE.md section 5. | 195972447 | 2 heldout_families | 2 heldout_families | 2 scored_heldout_families | 0 reusable_training_labels | none | PERSON method-audit owner | Any future rerun must be exposure-matched with equal steps and matched pseudo-loss caps; do not threshold-tune this result. |
| `protected_event_seed_50_20260713` | `QUARANTINED` | Track D: Keep sealed for Track D's final one-touch evaluation; never train, tune, or repeatedly inspect it. | 212768 | 50 protected_rows | 50 protected_rows | 50 reviewed_windows | 50 human_labels | human_gt | independent EVENT eval owner | No read until the frozen owner-41 A/B/C gate passes and the atomic one-touch claim is acquired. |
| `roboflow_ball_core_pretrain_20260706` | `CONSUMED` | RULED OUT: PROVEN_NEGATIVE_TRANSFER: Roboflow-only BALL pretraining scored reviewed-real F1@20 0.2971 versus official control 0.3611 in runs/lanes/w7_ballretrain_20260709/REPORT.md; do not repeat this corpus/recipe. | 2174383220 | 34658 core_pickleball_ball_rows | 34658 indexed_ball_images | 34658 indexed_ball_images | 44458 ball_boxes | human_gt | BALL public-data owner | Retain as immutable negative evidence only; no repeat of this corpus/recipe. |
| `roboflow_court_taxonomy_20260706` | `BLOCKED` | Track A: Audit whether the normalized court boxes/masks can provide auxiliary Track A supervision; build a source-grouped adapter if valid, otherwise issue a technical not_usable_because ruling because they are not 15-keypoint GT. | 1994304 | 2694 court_labeled_images | 2694 indexed_images | 2694 indexed_images | 2734 court_boxes_or_masks | human_gt | COURT public-data owner | Run the inventory-only semantics/source-family audit before any trainer dispatch. |
| `roboflow_person_adjacent_20260706` | `REJECTED` | RULED OUT: DOMAIN_MISMATCH_ADJACENT_TENNIS: the 15,469-image adjacent PERSON bucket is dominated by tennis and is explicitly excluded from the pickleball product-domain arm; evidence data/roboflow_universe_20260706/aggregated/corpus_card.json. | 780082673 | 29036 pre_dedup_adjacent_samples_all_classes | 15469 adjacent_person_images | 15469 indexed_images | 15469 person_images_with_boxes | human_gt | PERSON public-data owner | Do not queue in the regroup product-domain arm. |
| `roboflow_person_core_20260706` | `REJECTED` | RULED OUT: PERSON_RF_POOL_TOO_THIN: REJECTED_FOR_TRAINING; P2: NO_ATTEMPT_PREREQ, permanently closed for this export. The protected-collision audit ALREADY PASSED with 0 collisions across 45,844,128 frame pairs and 366,753,024 descriptor comparisons; Human quality card: NOT_COMPLETED_PROTOCOL. Any Track C aux/eval use requires a NEW ruling. Binding refs: runs/lanes/person_p1_roboflow_20260721/RULING.md and runs/lanes/person_p1_roboflow_20260721/report_fix2.json. | 872492382 | 15334 core_pickleball_person_rows_before_rights_exclusion | 15312 commercial_clean_core_person_rows | 15312 selector_bound_indexed_images | 47044 person_boxes | human_gt | PERSON public-data owner | Do not train from this export. Reopen any Track C aux/eval use only under a NEW ruling. |
| `roboflow_person_nc_20260706` | `BLOCKED` | Track C: Audit the 22 unique testing-esifc PERSON images for exhaustive protected collisions, then admit them only as a Track C judge/aux candidate with source_slug preserved. | 548850 | 34 person_index_rows | 22 unique_images | 22 indexed_images | 22 person_images_with_boxes | human_gt | PERSON public-data owner | Prove exhaustive zero protected collisions, then release to the named Track C judge/aux action. |
| `w7_audit_stratum_scratch_350` | `BLOCKED` | Track B: Finish and export all 350 scratch labels, reconcile lineage, and prove zero protected collisions before Track B consumption. | 566895615 | 350 frame_images | 350 uniform_random_scratch_frames | 350 frame_images | 0 reviewed_labels | none | BALL independent-review owner | Finish/export 350/350 scratch labels, reconcile lineage, and prove zero protected collisions before any BALL GPU. |

## Per-asset rulings and utilization

### `ball_reviewed_corpus_chain_1121_3026`

- State reason: UNRULED contamination finding: the 3026-row run was consumed, but the blended metric cannot decide a model and the corpus is quarantined pending source-held reconciliation.
- Paths: `runs/lanes/w7_ballingest4_20260709`, `runs/lanes/w7_ballretrain2_20260709`
- Source families: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=[]
- Overlap coverage: FAIL — all rows have source IDs; no immutable uncontaminated selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Source license fields are recorded for information only; contamination and source-held evaluation determine state.
- Disposition: Track B — Reconcile the 350-row scratch audit, freeze source-held partitions, and issue a contamination ruling before any Track B reuse. (evidence: `runs/lanes/w7_ballretrain2_20260709/provenance_split_corrected_3026.json`)

### `court_diversity_100_20260712`

- State reason: Blocked only on owner export of CVAT tasks 88-91; the three IYnbdRs1Jdk derivatives are a standing protocol denial, not a separate owner action.
- Paths: `cvat_upload/court_diversity_20260712`
- Source families: 1or-bXVM80M, 3sC53GlvW_s, 4qSoA-jwpVM, 6HqhDYnYcFU, 90UfH4QRSIg, A9H6EWfXht0, AXv70OhwStI, C5YUQlqZqBY, FGbGQgKPCfQ, IYnbdRs1Jdk, O29MdlviAqI, Se7M6ZKaC4Y, X42ktPBs140, Y6vE9AtIH4M, _4g4JnkG91Y, a_HzWrwK6vM, iMKDCfjfBNU, jr_60WVlG4c, kS_LzZGkdkg, ltIxlS0QJhg, n5D0179RNoA, q3575jnmjJQ, t9aeCo9I7Aw, uf0XWI7zZoY, wv3aPJrDwK4, y3c71Q0E1nE, z7PctOec41w, zXp76RZ0aVU
- Partition: train=['3sC53GlvW_s', '6HqhDYnYcFU', '90UfH4QRSIg', 'AXv70OhwStI', 'FGbGQgKPCfQ', 'O29MdlviAqI', 'X42ktPBs140', 'Y6vE9AtIH4M', '_4g4JnkG91Y', 'iMKDCfjfBNU', 'jr_60WVlG4c', 'kS_LzZGkdkg', 'ltIxlS0QJhg', 'n5D0179RNoA', 't9aeCo9I7Aw', 'uf0XWI7zZoY', 'y3c71Q0E1nE', 'z7PctOec41w', 'zXp76RZ0aVU']; val=['1or-bXVM80M', '4qSoA-jwpVM', 'C5YUQlqZqBY', 'A9H6EWfXht0', 'Se7M6ZKaC4Y', 'a_HzWrwK6vM', 'q3575jnmjJQ', 'wv3aPJrDwK4']; test=[]
- Overlap coverage: PARTIAL — three known protected-family derivatives denied; no immutable 97-frame clean subset selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Source license metadata is FYI only; this row is blocked only on owner export of tasks 88-91.
- Disposition: Track A — After the owner exports tasks 88-91, add only the protocol-eligible reviewed rows to the Track A court retrain pool with the frozen source-family split. (evidence: `cvat_upload/court_diversity_20260712/package_manifest.json`)

### `court_keypoints_6_20260707`

- State reason: Three usable rows are frozen external audit; three are rejected. None may train the preview challenger.
- Paths: `cvat_upload/court_keypoints_20260707`
- Source families: HyUqT7zFiwk, 73VurrTKCZ8, wBu8bC4OfUY, Ezz6HDNHlnk, zwCtH_i1_S4, _L0HVmAlCQI
- Partition: train=[]; val=[]; test=['court_keypoints_20260707:3_usable_external_audit']
- Overlap coverage: PARTIAL — source identity only
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: License is FYI only; eval-only quarantine is a frozen-audit protocol restriction.
- Disposition: Track A — Use the three fully usable rows once as the frozen Track A external audit; never train or tune on any of the six rows. (evidence: `cvat_upload/court_keypoints_20260707/validation_report.json`)

### `data_testclips_metadata_4`

- State reason: NO_MEDIA_ON_DISK and retrievability is unknown.
- Paths: `data/testclips/gear360_0200_high_near_overhead/clip_metadata.json`, `data/testclips/ppa_austin_md_qf_1200_high_baseline/clip_metadata.json`, `data/testclips/ppa_singles_0500_high_baseline/clip_metadata.json`, `data/testclips/side_view_game5_0100_high_side_fence/clip_metadata.json`
- Source families: gear360_0200_high_near_overhead, ppa_austin_md_qf_1200_high_baseline, ppa_singles_0500_high_baseline, side_view_game5_0100_high_side_fence
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — four metadata entries, zero video files
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: No license metadata is recorded; FYI only. The blocking fact is absent media with unknown retrievability.
- Disposition: not usable because NO_MEDIA_ON_DISK: all four data/testclips directories contain metadata only, no source location, and unknown retrievability; evidence runs/regroup_20260721/REGROUP_INPUTS.md.

### `eval_clips_ball_protected_4`

- State reason: Four clips, 120 BALL points, and 11459 PERSON boxes are protected evaluation only and unreachable by trainers.
- Paths: `eval_clips/ball/burlington_gold_0300_low_steep_corner`, `eval_clips/ball/wolverine_mixed_0200_mid_steep_corner`, `eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline`, `eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline`, `runs/cvat_imports/2026_06_30/burlington_gold_0300_low_steep_corner/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/indoor_doubles_fwuks_0500_long_mid_baseline/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/outdoor_webcam_iynbd_1500_long_high_baseline/person_ground_truth.json`, `runs/cvat_imports/2026_06_30/wolverine_mixed_0200_mid_steep_corner/person_ground_truth.json`
- Source families: burlington_gold_0300_low_steep_corner, wolverine_mixed_0200_mid_steep_corner, outdoor_webcam_iynbd_1500_long_high_baseline, indoor_doubles_fwuks_0500_long_mid_baseline
- Partition: train=[]; val=[]; test=['burlington_gold', 'wolverine_mixed', 'outdoor_webcam_iynbd', 'indoor_doubles_fwuks']
- Overlap coverage: PARTIAL — all four source videos registered; new trainer inputs must prove exhaustive derivative non-collision
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: License is FYI only; all four clips and 11,459 PERSON boxes remain protected by protocol and trainer-forbidden.
- Disposition: Track C — Use the 11,459 reviewed person boxes only for Track C evaluation and protected-collision auditing; never train, tune, or expand labels from these four protected clips. (evidence: `eval_clips/ball/manifest.json`)

### `event_abc_inputs_20260720`

- State reason: The frozen A input ran to 1000 steps and was recovered with exact hash evidence; the 41 validation rows were not loaded for gradient updates.
- Paths: `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json`
- Source families: video_sha256 and clip IDs embedded per row
- Partition: train=['owner_102_manifest:61_rows']; val=['owner_102_manifest:41_rows']; test=[]
- Overlap coverage: PASS — all training windows checked against protected 50; overlap_rows=0
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Owner-produced labels; license is FYI only and the frozen 61/41 split is a technical protocol gate.
- Disposition: Track D — Reuse the exact frozen 61-train/41-validation manifest for corrected exposure-matched Track D arms; validation rows stay gradient-excluded. (evidence: `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json`)

### `event_abc_vm_pull_20260721`

- State reason: ready_for_named_consumer
- Paths: `runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2`, `runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt`
- Source families: 143sf3gdwxsa, 98z43hspqz13, st0epgnab7dr, td2szayjwtrj, utasf5hnozwz, xkadsq9bli3h, frozen_t20_event_head:f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082
- Partition: train=['143sf3gdwxsa', '98z43hspqz13', 'st0epgnab7dr', 'td2szayjwtrj', 'utasf5hnozwz', 'xkadsq9bli3h']; val=[]; test=[]
- Overlap coverage: PASS — all corrected rows require ball_velocity_kink; 83gyqyc10y8f, iottnc0h3ekn, and o4dee9dn0ccr remain excluded; old vm_pull/abc_out is outside registered paths
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: PBV-FULL-USAGE-20260720 and the owner internal-use directive allow the registered training use; license remains FYI and teacher authority remains explicit.
- Disposition: Track D — Pretrain Stage-P/E-v2 only from the SHA-bound corrected 1189-row arm_b_manifest.json and initialize model-only from the SHA-bound frozen_t20_event_head.pt; keep owner validation gradient-excluded. (evidence: `runs/lanes/ev2_realrun_20260723/training_inputs_PREFLIGHT_DRYRUN.json`)

### `event_bootstrap_audio_20260713`

- State reason: Owner review measured 29/50 true contacts against a >=47/50 gate; Tier-A audio auto-labels are rejected for training.
- Paths: `data/event_bootstrap_20260713`
- Source families: pbvision_11min_20260713, 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=['pbvision_11min_20260713:diagnostic_only']
- Overlap coverage: PARTIAL — Tier-A quality check only; exhaustive protected collision scan not recorded for all bootstrap rows
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Owner internal-use directive 2026-07-22 applies; license did not determine this rejection.
- Disposition: not usable because REJECTED_AUDIO_ONLY_TEACHER: owner review measured 29/50 true contacts versus the >=47/50 gate at runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json; untyped audio alone is not training truth.

### `event_public_extended_opentt_20260713`

- State reason: 52987 dense ball points exist and 7852 map to the two local videos, but a frozen source-disjoint controlled-pretrain manifest has not been built.
- Paths: `data/event_public_20260713/extended_openttgames`, `data/event_public_20260713/openttgames/videos`
- Source families: Extended OpenTTGames, OpenTTGames game_4, OpenTTGames test_2
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — event-to-local-video mapping and source overlap not yet inventoried
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY-NC-SA 4.0 is recorded for information only; license is not a state gate under the owner directive 2026-07-22.
- Disposition: Track B — Run one controlled OpenTTGames dense-ball pretrain experiment against the official control, using only source-mapped local pixels and a frozen source-disjoint manifest; the Roboflow negative-transfer result makes the control mandatory. (evidence: `runs/lanes/w7_ballretrain_20260709/REPORT.md`)

### `event_public_f3set_20260713`

- State reason: BLOCKED_NO_PIXELS: labels-only asset with 0 decodable 64-frame windows.
- Paths: `data/event_public_20260713/f3set`
- Source families: F3Set tennis singles/doubles and table-tennis labels
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no pixels available for source-family collision audit
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Repository license is unasserted/academic-research in the manifest; FYI only and not a state gate under the owner directive 2026-07-22.
- Disposition: Track D — Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS if no local 64-frame context resolves. (evidence: `data/event_public_20260713/f3set/manifest.json`)

### `event_public_golfdb_20260713`

- State reason: BLOCKED_NO_LABEL_MAPPED_SOURCE_RESOLVED_PIXELS: test_video.mp4 exists, but 0 decodable 64-frame windows are authoritatively mapped to GolfDB label rows.
- Paths: `data/event_public_20260713/golfdb`
- Source families: GolfDB
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — local test_video.mp4 is not label-mapped or source-resolved
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Dataset license metadata is FYI only under the owner internal-use directive 2026-07-22.
- Disposition: Track D — Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_LABEL_MAPPED_SOURCE_RESOLVED_PIXELS unless the local test video is authoritatively mapped to label rows and source identity. (evidence: `data/event_public_20260713/golfdb/manifest.json`)

### `event_public_padeltracker100_20260713`

- State reason: BLOCKED_NO_PIXELS: 906 motion intervals exist but no locally resolvable source context or proven PERSON/ReID supervision semantics.
- Paths: `data/event_public_20260713/padeltracker100`
- Source families: 2022_BCN_FinalM_1, 2022_BCN_FinalF_1
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no source pixels
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY 4.0 label metadata and separate broadcast-pixel terms are FYI only; neither determines state.
- Disposition: Track C — Audit the 906 shot-window player intervals as a PERSON/ReID auxiliary candidate; accept only if an inventory adapter proves usable player-box/pose and identity-continuity semantics, otherwise record a technical not-usable ruling. (evidence: `data/event_public_20260713/padeltracker100/manifest.json`)

### `event_public_shuttlecock_zenodo_20260713`

- State reason: Fetched sample contains video only; no independent event GT was found.
- Paths: `data/event_public_20260713/shuttlecock_hitting_zenodo`
- Source families: Zenodo 14677727 part2 sample clips 00170-00177
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — eight sample clips with no structured labels
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY 4.0 is recorded for information only; the absence of independent structured GT determines state.
- Disposition: Track D — Run the Track D inventory-only adapter and preserve BLOCKED_NO_STRUCTURED_EVENTS; do not infer GT from submission CSVs. (evidence: `data/event_public_20260713/shuttlecock_hitting_zenodo/manifest.json`)

### `event_public_shuttleset_20260713`

- State reason: BLOCKED_NO_PIXELS: label-only badminton asset with 0 decodable windows.
- Paths: `data/event_public_20260713/coachai_shuttleset`
- Source families: ShuttleSet, ShuttleSet22
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — no local pixels
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: MIT annotation and linked broadcast metadata are FYI only; license is not a state gate.
- Disposition: Track D — Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS unless local source context resolves. (evidence: `data/event_public_20260713/coachai_shuttleset/manifest.json`)

### `event_public_squash_figshare_20260713`

- State reason: One audio file is present but zero distributable structured event rows were found.
- Paths: `data/event_public_20260713/squash_audio_figshare`
- Source families: Figshare article 5962015
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — one raw audio part with zero structured labels
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY 4.0 is recorded for information only; the absence of structured labels determines state.
- Disposition: Track D — Run the Track D inventory-only adapter and record BLOCKED_NO_STRUCTURED_EVENTS unless an authoritative structured-label file is present. (evidence: `data/event_public_20260713/squash_audio_figshare/manifest.json`)

### `event_public_tt_sounds_20260713`

- State reason: Audio-only snippets cannot supply 64-frame visual context or typed contact truth.
- Paths: `data/event_public_20260713/tt_sounds_data`
- Source families: TT Sounds
- Partition: train=['tt_sounds:4561_rows']; val=[]; test=['tt_sounds:1141_rows']
- Overlap coverage: PASS — all registered label rows rejected from typed-event training
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY-NC 4.0 is recorded for information only; technical audio-only semantics determine the current rejection.
- Disposition: Track D — Include in the Track D inventory-only corpus-expansion audit as audio-only negative evidence; do not queue it to typed visual event training. (evidence: `data/event_public_20260713/tt_sounds_data/manifest.json`)

### `online_harvest_20260706`

- State reason: Six source families supplied the historical reviewed BALL corpus; the exact rally-media path is additionally queue-authorized for Stage-F E-v2 training and owner-41 scoring while audit derivatives remain path-excluded.
- Paths: `data/online_harvest_20260706/rallies`, `data/online_harvest_20260706/manifest.json`
- Source families: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8', 'Ezz6HDNHlnk', 'HyUqT7zFiwk', '_L0HVmAlCQI', 'wBu8bC4OfUY', 'zwCtH_i1_S4']; val=[]; test=[]
- Overlap coverage: PASS — 40 rally clips across six YouTube source families; protected eval, pb.vision compare-only media, prelabels, and court-calibration derivatives are outside the registered paths
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Source license fields are FYI only and not a state gate under the owner directive 2026-07-22.
- Disposition: Track A — Keep the unregistered court_calibrations directory items audit-only; use them only to reproduce and verify the frozen Track A external audit packaging, and exclude every derivative from training and tuning. (evidence: `cvat_upload/court_keypoints_20260707/validation_report.json`)

### `online_harvest_20260712`

- State reason: The selected source segments were consumed into 100 still frames, then intentionally deleted; the derived package is queued through its own Track A row.
- Paths: `data/online_harvest_20260712`, `cvat_upload/court_diversity_20260712`
- Source families: 28 selected YouTube source IDs recorded in court package
- Partition: train=['19 allowed court source groups in court_diversity row']; val=['8 frozen court holdout groups']; test=[]
- Overlap coverage: PARTIAL — 100 staged frames; no immutable 97-frame clean subset selector is registered
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Source metadata is FYI only; technical/protocol gates determine eligibility under the owner directive 2026-07-22.
- Disposition: Track A — Feed only protocol-eligible reviewed rows from the derived 100-frame court package into the Track A court pool through the court_diversity_100_20260712 row after owner export. (evidence: `cvat_upload/court_diversity_20260712/package_manifest.json`)

### `online_harvest_person_gap_20260706`

- State reason: The raw person-box CVAT route is closed and superseded by the Track C few-shot pack.
- Paths: `runs/research_sota_20260722/INTERNAL_AUDIT.json`
- Source families: eight online_harvest_20260706 source videos
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — all eight raw videos are covered by the explicit route ruling
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: The source videos have null license fields, recorded for information only; CVAT closure and route supersession determine this ruling.
- Disposition: not usable because CVAT_CLOSED_TO_PERSON_TASKS: no person-box task was created on the eight raw videos, and the route is superseded by Track C's stratified few-shot verify-not-draw pack; evidence runs/research_sota_20260722/INTERNAL_AUDIT.json and runs/research_sota_20260722/PROGRAM.md#track-c-person-identity-pose.

### `owner_event_labels_102_20260719`

- State reason: Arm A loaded the frozen 61 training rows; the 41 validation rows remained evaluation-only and scored 0.0.
- Paths: `data/event_labels_owner_20260719`
- Source families: owner event query windows
- Partition: train=['owner_102_manifest:61_rows']; val=['owner_102_manifest:41_rows']; test=[]
- Overlap coverage: PASS — all owner training windows; two windows shifted; overlap_rows=0
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Owner-produced labels; license is FYI only under the owner internal-use directive 2026-07-22.
- Disposition: Track D — Reuse the frozen 61 training rows for corrected Track D fine-tuning and keep the 41 validation rows gradient-excluded. (evidence: `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json`)

### `owner_img_1605_court_review_20260721`

- State reason: The owner court-review derivatives are bound separately from eval_clips_ball_protected_4 and remain evaluation-only.
- Paths: `eval_clips/ball/owner_IMG_1605_8a193402780b`
- Source families: owner_IMG_1605_8a193402780b
- Partition: train=[]; val=[]; test=['owner_IMG_1605_8a193402780b']
- Overlap coverage: PASS — all six derivative files
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Owner-provided evidence; license is FYI only and evaluation-only posture is protocol.
- Disposition: Track A — Retain as the Track A owner-reviewed metric audit only; never expose these derivatives to training exporters. (evidence: `eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt.json`)

### `pbv_pickleball_teacher_events_20260720`

- State reason: 0/7314 windows are locally decodable and the recovered B/C materialization accepted 292 audio-only rows, so no valid training consumption is recorded.
- Paths: `runs/lanes/pbv_pickleball_corpus_20260720`
- Source families: 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, xkadsq9bli3h, pldtjpw3h0jw, utasf5hnozwz, 0tmdeghtfvjx
- Partition: train=['143sf3gdwxsa', '98z43hspqz13', 'bewqc0glhgpq', 'st0epgnab7dr', 'td2szayjwtrj', 'tqjlrcntpjvt', 'xkadsq9bli3h']; val=['pldtjpw3h0jw', 'utasf5hnozwz']; test=['0tmdeghtfvjx']
- Overlap coverage: PARTIAL — ten corpus video families; declared corpus path is not bound to an immutable clean subset selector
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Owner internal-use directive 2026-07-22 applies; license is FYI only and the blockers are technical/protocol gates.
- Disposition: Track D — Materialize corrected non-audio-agreement pretrain rows after all media and PTS hashes are local; never treat teacher rows as human GT. (evidence: `runs/lanes/pbv_pickleball_corpus_20260720`)

### `pbv_replay_xkadsq9bli3h_20260720`

- State reason: The fresh-clip E2E proof already ran and was browser-verified as an honest partial; B1 SST materialization remains unfinished after 5/7 sources and B2 remains unauthorized.
- Paths: `data/pbv_replay_20260720/xkadsq9bli3h/max.mp4`
- Source families: xkadsq9bli3h
- Partition: train=['xkadsq9bli3h']; val=[]; test=[]
- Overlap coverage: PASS — xkadsq9bli3h is one of the seven B1 training sources and is not a compare-only ID
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Training and commercial use authorized by PBV-FULL-USAGE-20260720 and owner directive 2026-07-22; license is not a state gate.
- Disposition: Track B — Resume B1 SST materialization for remaining sources tqjlrcntpjvt and xkadsq9bli3h with skip-if-exists/resume support; do not arm B2 without a fresh explicit dispatch decision. (evidence: `runs/lanes/ball_b2_seed1_20260722/RESULTS.md`)

### `pbvision_gallery_20260719`

- State reason: All 13 JSON exports were inventoried and ten entered the teacher materializer; local pixel availability remains only 2/13.
- Paths: `data/pbvision_gallery_20260719`, `data/pbvision_11min_20260713/source_video.mp4`, `eval_clips/ball/pbvision_11min_20260713`
- Source families: 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, xkadsq9bli3h, pldtjpw3h0jw, utasf5hnozwz, 0tmdeghtfvjx, 83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr
- Partition: train=['143sf3gdwxsa', '98z43hspqz13', 'bewqc0glhgpq', 'st0epgnab7dr', 'td2szayjwtrj', 'tqjlrcntpjvt', 'xkadsq9bli3h']; val=['pldtjpw3h0jw', 'utasf5hnozwz']; test=['0tmdeghtfvjx', '83gyqyc10y8f', 'iottnc0h3ekn', 'o4dee9dn0ccr']
- Overlap coverage: PASS — ten corpus-eligible IDs are machine-pinned; 83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr and every derivative remain compare-only
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Training and commercial use authorized by PBV-FULL-USAGE-20260720 and the owner no-licensing-constraints directive 2026-07-22; license is not a state gate.
- Disposition: Track A — After the approximately 10-frame owner-tap spot-check, add only corpus-eligible pseudo-label IDs 0tmdeghtfvjx, 143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, pldtjpw3h0jw, st0epgnab7dr, td2szayjwtrj, tqjlrcntpjvt, utasf5hnozwz, and xkadsq9bli3h to the Track A court-keypoint retrain pool. (evidence: `runs/lanes/court_pbv_extract_20260722/spec.md`)

### `person_mixed_pool_no_lift_20260722`

- State reason: The preregistered both-families-nonnegative bar failed under an under-controlled exposure design; verified=false and do_not_promote=true.
- Paths: `runs/lanes/person_mixed_20260722/gpu_phase_report.json`, `runs/lanes/person_mixed_20260722/vm_pull`, `runs/handoff_20260722/ORCHESTRATOR_STATE.md`
- Source families: roboflow anchor pool plus online-harvest/pb.vision pseudo pool
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — all pulled artifacts verified; design control remained insufficient
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: License is FYI only; the under-controlled design and failed preregistered bar determine rejection.
- Disposition: not usable because PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED: od8al precision -0.1924 (F1 -0.0842) LOSS, hemel_test F1 +0.046 WIN; the both-families-nonnegative bar failed, and 13.5x gradient-update plus 6.75x anchor-exposure asymmetry makes the miss unattributable among three live hypotheses; evidence runs/lanes/person_mixed_20260722/gpu_phase_report.json and ORCHESTRATOR_STATE.md section 5.

### `protected_event_seed_50_20260713`

- State reason: Protected evaluation identity; the prior owner spot check failed 29/50 versus the >=47/50 Tier-A bar, but the set is not reusable for threshold shopping.
- Paths: `runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json`, `runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json`
- Source families: protected event window source identities in spot_check_tier_a_50.json
- Partition: train=[]; val=[]; test=['protected_event_seed_50:50_rows']
- Overlap coverage: PASS — all protected rows compared with owner training windows
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: License is FYI only; this quarantine is a permanent protected-evaluation protocol restriction.
- Disposition: Track D — Keep sealed for Track D's final one-touch evaluation; never train, tune, or repeatedly inspect it. (evidence: `runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json`)

### `roboflow_ball_core_pretrain_20260706`

- State reason: The 34658-row core BALL subset supplied the historical pretrain and internal validation, then regressed against the official control.
- Paths: `data/roboflow_universe_20260706/aggregated/subset_indexes/ball_index.json`
- Source families: 36 core_pickleball BALL source slugs, including the historically consumed testing-esifc NC source
- Partition: train=['roboflow_ball_core_train:32018_rows']; val=['roboflow_ball_core_internal_val:2640_rows']; test=[]
- Overlap coverage: PASS — all core BALL rows are frozen negative evidence and refused from repeat training
- Immutable clean-subset selectors: 0
- Consumers: 1
- License FYI: Mixed public licenses, including testing-esifc BY-NC-SA, are recorded for information only; the measured negative transfer determines this ruling.
- Disposition: not usable because PROVEN_NEGATIVE_TRANSFER: Roboflow-only BALL pretraining scored reviewed-real F1@20 0.2971 versus official control 0.3611 in runs/lanes/w7_ballretrain_20260709/REPORT.md; do not repeat this corpus/recipe.

### `roboflow_court_taxonomy_20260706`

- State reason: Court taxonomy exists, but conversion from heterogeneous boxes/masks to useful Track A auxiliary supervision and protected-collision safety are unproved.
- Paths: `data/roboflow_universe_20260706/aggregated/subset_indexes/court_index.json`, `data/roboflow_universe_20260706/aggregated/corpus_card.json`
- Source families: 15 Roboflow source slugs normalized to court
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — 2,694 indexed images require source-family and protected-collision audit
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Mixed source licenses are recorded for information only; technical label semantics and protected-collision gates determine eligibility.
- Disposition: Track A — Audit whether the normalized court boxes/masks can provide auxiliary Track A supervision; build a source-grouped adapter if valid, otherwise issue a technical not_usable_because ruling because they are not 15-keypoint GT. (evidence: `data/roboflow_universe_20260706/aggregated/subset_indexes/court_index.json`)

### `roboflow_person_adjacent_20260706`

- State reason: The intervention is in-domain pickleball; the adjacent bucket is dominated by tennis and is categorically excluded.
- Paths: `data/roboflow_universe_20260706/aggregated/subset_indexes/person_index.json`
- Source families: three adjacent-sport Roboflow sources
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: PASS — all adjacent-sport person images denied from product-domain arm
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Source licenses are recorded for information only; domain mismatch determines rejection.
- Disposition: not usable because DOMAIN_MISMATCH_ADJACENT_TENNIS: the 15,469-image adjacent PERSON bucket is dominated by tennis and is explicitly excluded from the pickleball product-domain arm; evidence data/roboflow_universe_20260706/aggregated/corpus_card.json.

### `roboflow_person_core_20260706`

- State reason: PERSON_RF_POOL_TOO_THIN: 8,887 train images across 7 source families fails the binding >=8-family bar; P2 is permanently closed for this export, while the protected-collision audit already passed.
- Paths: `runs/lanes/data_steward_ledger_20260721/person_core_commercial_15312_selector.json`
- Source families: 14 CC BY 4.0 core person sources after NC exclusion
- Partition: train=[]; val=['pickleball-od8al/pickleball-version2']; test=['hemel/pickleball-cedmo']
- Overlap coverage: PASS — 0 protected collisions across 45,844,128 frame pairs and 366,753,024 descriptor comparisons
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: CC BY 4.0 is recorded for information only; owner directive 2026-07-22 removes license as a state gate.
- Disposition: not usable because PERSON_RF_POOL_TOO_THIN: REJECTED_FOR_TRAINING; P2: NO_ATTEMPT_PREREQ, permanently closed for this export. The protected-collision audit ALREADY PASSED with 0 collisions across 45,844,128 frame pairs and 366,753,024 descriptor comparisons; Human quality card: NOT_COMPLETED_PROTOCOL. Any Track C aux/eval use requires a NEW ruling. Binding refs: runs/lanes/person_p1_roboflow_20260721/RULING.md and runs/lanes/person_p1_roboflow_20260721/report_fix2.json.

### `roboflow_person_nc_20260706`

- State reason: License-only quarantine is dropped; the only remaining blocker is an exhaustive protected-collision audit.
- Paths: `data/roboflow_universe_20260706/aggregated/per_dataset/testing-esifc__pickle-ball-labeling-mff1d.index.json`
- Source families: testing-esifc/pickle-ball-labeling-mff1d
- Partition: train=[]; val=[]; test=[]
- Overlap coverage: NOT_RUN — 22 unique images require exhaustive collision proof against protected eval derivatives
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: BY-NC-SA 4.0 is recorded for information only; it no longer determines state or dispatch eligibility.
- Disposition: Track C — Audit the 22 unique testing-esifc PERSON images for exhaustive protected collisions, then admit them only as a Track C judge/aux candidate with source_slug preserved. (evidence: `data/roboflow_universe_20260706/aggregated/per_dataset/testing-esifc__pickle-ball-labeling-mff1d.index.json`)

### `w7_audit_stratum_scratch_350`

- State reason: All 350 images are staged and decodable, but no authoritative reviewed export exists; label_count is zero.
- Paths: `cvat_upload/w7_audit_stratum_20260709`
- Source families: 73VurrTKCZ8, Ezz6HDNHlnk, HyUqT7zFiwk, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4
- Partition: train=['73VurrTKCZ8:27', '_L0HVmAlCQI:34', 'wBu8bC4OfUY:60', 'zwCtH_i1_S4:62']; val=['HyUqT7zFiwk:100', 'Ezz6HDNHlnk:67']; test=[]
- Overlap coverage: NOT_RUN — not run; old 35-frame sample is insufficient
- Immutable clean-subset selectors: 0
- Consumers: 0
- License FYI: Source license metadata is FYI only; missing reviewed labels and collision proof determine state.
- Disposition: Track B — Finish and export all 350 scratch labels, reconcile lineage, and prove zero protected collisions before Track B consumption. (evidence: `cvat_upload/w7_audit_stratum_20260709`)
