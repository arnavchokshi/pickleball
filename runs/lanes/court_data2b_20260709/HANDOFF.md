# COURT-DATA-2b training handoff

## Corpus

- Root: `runs/lanes/court_data2b_20260709/real_court_corpus_partial/`
- Total: **3,921 rows from 9 datasets**.
- Every item contains exactly the 15 canonical keypoint names. Unmapped channels are JSON `null`.
- Every item, including Chetan full-15 rows, uses `status: reviewed_external_dataset`.
- Loader accounting is clean: 3,921 external rows, 0 owner-independent rows, 0 static-camera-copy rows, and 0 synthetic rows.

| Dataset | Rows | Labeled points | Source group | Proposed split |
|---|---:|---:|---|---|
| `chetan-rajagiri-9abfm__pickleball-court-v2__v1` | 35 | 15 | `chetan_court_v2` | train |
| `n-do-tran__pickleball-court-p3chl__v4` | 120 | 14 | `p3chl_14kp_family` | train |
| `necromancer__pickleball-court-vbmkq__v2` | 1,772 | 12 | `vbmkq_vhpgp_12kp_family` | train |
| `nigh-workspace__pickleball-court-vhpgp__v11` | 292 | 12 | `vbmkq_vhpgp_12kp_family` | train |
| `pickleball-ball-detection__pickleball-court-keypoints-syncz__v6` | 41 | 12 | `syncz_12kp` | train |
| `ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3` | 288 | 14 | `p3chl_14kp_family` | train |
| `stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1` | 285 | 12 | `stump_front_view_12kp` | train |
| `testworkspace-i8nb1__pickle-court-keypoints__v2` | 54 | 12 | `xuann_testworkspace_12kp_family` | validation |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | 1,034 | 12 | `xuann_testworkspace_12kp_family` | validation |

Labeled-keypoint histogram: **12: 3,478; 14: 408; 15: 35**. Fourteen exact duplicates from later datasets were removed, and 22 annotations missing at least one mapped source point were excluded rather than partially relabeled below their audited mapping bucket.

## Dataset split

`split_proposal.json` holds out Testworkspace and Xuann together: **2 validation datasets / 1,088 rows**, leaving **7 train datasets / 2,833 rows**. This keeps their shared `xuann_testworkspace_12kp_family` entirely out of training and gives validation broadcast, elevated, low, and steep viewpoints. No dataset is frame-random-split.

## Loader and emission proof

- `loader_contract_proof.json`: 3,921 loaded rows, 0 schema errors, histogram exactly matching the emitted JSON, and a 14-point row round-tripped through `court_keypoint_label_rows` with `net_center` remaining null in the raw schema and absent from the loader's labeled-coordinate map.
- `emission_overlays/`: 5 overlays from emitted rows for each included dataset, 45 total. Each banner prints its emitted labeled/null count; points are read from emitted JSON rather than from the audit mapping.
- Five deterministic random 12-point spot samples all have `net_left_sideline`, `net_center`, and `net_right_sideline` equal to JSON null:

| Dataset | Emitted frame |
|---|---|
| `nigh-workspace__pickleball-court-vhpgp__v11` | `frame_000197.jpg` |
| `necromancer__pickleball-court-vbmkq__v2` | `frame_001366.jpg` |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | `frame_000683.jpg` |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | `frame_000365.jpg` |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | `frame_000049.jpg` |

## Guards and licensing

- Reused parent evidence: `runs/lanes/court_data2_20260709/corpus_stats.json` (SHA-256 `bf26e461240151fdee55e348fbfbf7c2f5cb24e9228482a06805f53df5fdc011`).
- The guard was reconstructed to the parent's exact 82-file / 70-unique identity: 33 protected eval label-image blob IDs came from Git's index without opening any `eval_clips/**/labels/*` file; 49 non-protected harvest/GT images produced 37 unique SHA-256 values. Counts match the parent artifact exactly, and 0 emitted rows match either identity set.
- `pwxNwFfYQlQ` and `vQhtz8l6VqU` remain denylisted. Exact byte hashes are unique across all emitted rows.
- All included datasets are recorded as CC BY 4.0. Unknown/noncommercial licenses still fail closed; the focused CLI suite directly covers quarantine.

## Training invocation

The exact corpus argument is:

```bash
--real-root runs/lanes/court_data2b_20260709/real_court_corpus_partial
```

To consume the proposed dataset holdout with the current trainer:

```bash
.venv/bin/python scripts/racketsport/train_court_keypoint_heatmap.py \
  --real-root runs/lanes/court_data2b_20260709/real_court_corpus_partial \
  --holdout-clip testworkspace-i8nb1__pickle-court-keypoints__v2__train \
  --holdout-clip testworkspace-i8nb1__pickle-court-keypoints__v2__valid \
  --holdout-clip testworkspace-i8nb1__pickle-court-keypoints__v2__test \
  --holdout-clip xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10__train \
  --holdout-clip xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10__valid \
  --holdout-clip xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10__test \
  --out runs/lanes/<training_lane>/court_partial_training
```

Best-stack delta: **none**. This is data preparation only; `VERIFIED=0` is unchanged.
