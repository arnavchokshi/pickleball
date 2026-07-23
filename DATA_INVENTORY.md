# DinkVision Data Inventory

> **GENERATED FILE — do not hand-edit.** Authority is `runs/manager/data_ledger.json`; this is its scannable human view.
> Regenerate after any ledger change: `.venv/bin/python scripts/racketsport/build_data_inventory.py`
> Ledger generated: `2026-07-23T16:44:55Z` · 33 registered datasets · `VERIFIED=0` binding.

The single place to see **all data we have on every lane and whether it is used**. Every dataset the project has touched is a registered asset here with a state and a reason. If a dataset is not in this table, it is not registered — add a ledger row before any training touches it (the data-safety gate enforces this).

**Legend** — ✅ used (training) · 📊 used (audit/eval only) · 🟡 authorized — not yet trained · 🔴 not used — blocked · 🔒 held out (eval/protected) · ⏸ parked · ❌ ruled out

## Summary

| Lane | ✅ used | 🟡 authorized | 🔴 blocked | 🔒 held-out | ❌ rejected |
|---|---|---|---|---|---|
| **COURT** | 3 | 1 | 2 | 2 | 0 |
| **BALL** | 1 | 0 | 2 | 1 | 0 |
| **PERSON** | 0 | 0 | 2 | 1 | 0 |
| **EVENT** | 2 | 1 | 4 | 1 | 3 |
| **SHARED / OTHER** | 1 | 0 | 2 | 0 | 4 |
| **TOTAL** | 7 | 2 | 12 | 5 | 7 |

## COURT

| Dataset | What / source | Size | Status | Why not used / next step |
|---|---|---|---|---|
| `online_harvest_20260712` | source channel IDs in harvest metadata | 28 rows | ✅ used (training) | Feed only protocol-eligible reviewed rows from the derived 100-frame court package into the Track A court pool through the court_diversity_100_202607… |
| `pbvision_gallery_20260719` | pb.vision gallery and owner-provided demo | 124,743 labels | ✅ used (training) | After the approximately 10-frame owner-tap spot-check, add only corpus-eligible pseudo-label IDs 0tmdeghtfvjx, 143sf3gdwxsa, 98z43hspqz13, bewqc0glhg… |
| `online_harvest_20260706` | source YouTube channels recorded in manifest | 40 rows | 📊 used (audit/eval only) | Keep the unregistered court_calibrations directory items audit-only; use them only to reproduce and verify the frozen Track A external audit packagin… |
| `roboflow_court_keypoints_adapted_20260723` | seven owner-approved Roboflow court-keypoint workspaces | 3,168 labels | 🟡 authorized — not yet trained | Use this 264-row pack3-disjoint curated subset in the next court fine-tune instead of adding it to the duplicate-heavy 2833-row parent pool; keep net… |
| `court_diversity_100_20260712` | 28 source YouTube channel families | 97 rows | 🔴 not used — blocked | After the owner exports tasks 88-91, add only the protocol-eligible reviewed rows to the Track A court retrain pool with the frozen source-family spl… |
| `roboflow_court_taxonomy_20260706` | Roboflow workspaces/projects | 2,734 labels | 🔴 not used — blocked | Audit whether the normalized court boxes/masks can provide auxiliary Track A supervision; build a source-grouped adapter if valid, otherwise issue a… |
| `court_keypoints_6_20260707` | six source YouTube channels | 6 labels | 🔒 held out (eval/protected) | Use the three fully usable rows once as the frozen Track A external audit; never train or tune on any of the six rows. |
| `owner_img_1605_court_review_20260721` | owner-provided media | 15 labels | 🔒 held out (eval/protected) | Retain as the Track A owner-reviewed metric audit only; never expose these derivatives to training exporters. |

## BALL

| Dataset | What / source | Size | Status | Why not used / next step |
|---|---|---|---|---|
| `pbv_replay_xkadsq9bli3h_20260720` | pb.vision gallery | 1 rows | ✅ used (training) | Resume B1 SST materialization for remaining sources tqjlrcntpjvt and xkadsq9bli3h with skip-if-exists/resume support; do not arm B2 without a fresh e… |
| `event_public_extended_opentt_20260713` | lab.osai.ai | 52,987 labels | 🔴 not used — blocked | Run one controlled OpenTTGames dense-ball pretrain experiment against the official control, using only source-mapped local pixels and a frozen source… |
| `w7_audit_stratum_scratch_350` | six source YouTube channels | 350 rows | 🔴 not used — blocked | Finish and export all 350 scratch labels, reconcile lineage, and prove zero protected collisions before Track B consumption. |
| `ball_reviewed_corpus_chain_1121_3026` | six source YouTube channels | 3,026 labels | 🔒 held out (eval/protected) | Reconcile the 350-row scratch audit, freeze source-held partitions, and issue a contamination ruling before any Track B reuse. |

## PERSON

| Dataset | What / source | Size | Status | Why not used / next step |
|---|---|---|---|---|
| `event_public_padeltracker100_20260713` | broadcast pixels not locally licensed/fetched | 13,250 labels | 🔴 not used — blocked | Audit the 906 shot-window player intervals as a PERSON/ReID auxiliary candidate; accept only if an inventory adapter proves usable player-box/pose an… |
| `roboflow_person_nc_20260706` | Roboflow testing-esifc | 22 labels | 🔴 not used — blocked | Audit the 22 unique testing-esifc PERSON images for exhaustive protected collisions, then admit them only as a Track C judge/aux candidate with sourc… |
| `eval_clips_ball_protected_4` | historical eval sources | 11,459 labels | 🔒 held out (eval/protected) | Use the 11,459 reviewed person boxes only for Track C evaluation and protected-collision auditing; never train, tune, or expand labels from these fou… |

## EVENT

| Dataset | What / source | Size | Status | Why not used / next step |
|---|---|---|---|---|
| `event_abc_inputs_20260720` | owner event labeling | 102 labels | ✅ used (training) | Reuse the exact frozen 61-train/41-validation manifest for corrected exposure-matched Track D arms; validation rows stay gradient-excluded. |
| `owner_event_labels_102_20260719` | owner labeling session | 102 labels | ✅ used (training) | Reuse the frozen 61 training rows for corrected Track D fine-tuning and keep the 41 validation rows gradient-excluded. |
| `event_abc_vm_pull_20260721` | pb.vision gallery and frozen T20 initialization lineage | 1,189 labels | 🟡 authorized — not yet trained | Pretrain Stage-P/E-v2 only from the SHA-bound corrected 1189-row arm_b_manifest.json and initialize model-only from the SHA-bound frozen_t20_event_he… |
| `event_public_f3set_20260713` | source YouTube channels not fetched | 43,655 labels | 🔴 not used — blocked | Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS if no local 64-frame context resolves. |
| `event_public_golfdb_20260713` | test_video.mp4 exists locally but is not source-resolved or… | 11,200 labels | 🔴 not used — blocked | Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_LABEL_MAPPED_SOURCE_RESOLVED_PIXELS unless the local test video is aut… |
| `event_public_shuttleset_20260713` | broadcast videos not downloaded | 88,840 labels | 🔴 not used — blocked | Run the Track D inventory-only pretrain-corpus adapter and preserve BLOCKED_NO_PIXELS unless local source context resolves. |
| `pbv_pickleball_teacher_events_20260720` | pb.vision gallery | 4,637 labels | 🔴 not used — blocked | Materialize corrected non-audio-agreement pretrain rows after all media and PTS hashes are local; never treat teacher rows as human GT. |
| `protected_event_seed_50_20260713` | owner protected review | 50 labels | 🔒 held out (eval/protected) | Keep sealed for Track D's final one-touch evaluation; never train, tune, or repeatedly inspect it. |
| `event_public_shuttlecock_zenodo_20260713` | private competition lineage unknown | 8 rows | ❌ ruled out | Run the Track D inventory-only adapter and preserve BLOCKED_NO_STRUCTURED_EVENTS; do not infer GT from submission CSVs. |
| `event_public_squash_figshare_20260713` | audio only | 1 rows | ❌ ruled out | Run the Track D inventory-only adapter and record BLOCKED_NO_STRUCTURED_EVENTS unless an authoritative structured-label file is present. |
| `event_public_tt_sounds_20260713` | audio only | 5,702 labels | ❌ ruled out | Include in the Track D inventory-only corpus-expansion audit as audio-only negative evidence; do not queue it to typed visual event training. |

## SHARED / OTHER

| Dataset | What / source | Size | Status | Why not used / next step |
|---|---|---|---|---|
| `roboflow_ball_core_pretrain_20260706` | Roboflow workspaces/projects | 44,458 labels | 📊 used (audit/eval only) | PROVEN_NEGATIVE_TRANSFER: Roboflow-only BALL pretraining scored reviewed-real F1@20 0.2971 versus official control 0.3611 in runs/lanes/w7_ballretrai… |
| `data_testclips_metadata_4` | unknown; no retrievability location recorded | 4 rows | ⏸ parked | NO_MEDIA_ON_DISK: all four data/testclips directories contain metadata only, no source location, and unknown retrievability; evidence runs/regroup_20… |
| `online_harvest_person_gap_20260706` | source channels recorded in online_harvest manifest | 8 rows | ⏸ parked | CVAT_CLOSED_TO_PERSON_TASKS: no person-box task was created on the eight raw videos, and the route is superseded by Track C's stratified few-shot ver… |
| `event_bootstrap_audio_20260713` | source video audio tracks | 3,173 labels | ❌ ruled out | REJECTED_AUDIO_ONLY_TEACHER: owner review measured 29/50 true contacts versus the >=47/50 gate at runs/lanes/event_bootstrap_20260713/owner_spot_chec… |
| `person_mixed_pool_no_lift_20260722` | Roboflow, online harvest, pb.vision | 2 rows | ❌ ruled out | PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED: od8al precision -0.1924 (F1 -0.0842) LOSS, hemel_test F1 +0.046 WIN; the both-families-nonnegative bar fai… |
| `roboflow_person_adjacent_20260706` | Roboflow workspaces/projects | 15,469 labels | ❌ ruled out | DOMAIN_MISMATCH_ADJACENT_TENNIS: the 15,469-image adjacent PERSON bucket is dominated by tennis and is explicitly excluded from the pickleball produc… |
| `roboflow_person_core_20260706` | Roboflow workspaces/projects | 47,044 labels | ❌ ruled out | PERSON_RF_POOL_TOO_THIN: REJECTED_FOR_TRAINING; P2: NO_ATTEMPT_PREREQ, permanently closed for this export. The protected-collision audit ALREADY PASS… |

---
_Not sure why a dataset is blocked or held out? The full provenance, hashes, partitions, and rulings live in `runs/manager/data_ledger.json` (per-asset) and `runs/manager/DATA_LEDGER.md` (the audit view)._
