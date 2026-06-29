# Pickleball Pretraining Runs 2026-06-28

Date: 2026-06-28

Scope: TrackNetV3 ball tracking fine-tune and court-keypoint heatmap pretraining only. These artifacts are pretraining experiments, not BALL or CAL verification.

## Outcome Ranking

| Rank | Technology | Pipeline part | Result | Use now? |
| ---: | --- | --- | --- | --- |
| 1 | Court-keypoint heatmap model | No-tap court calibration / court corner prior | Synthetic error improved from 67.31 px to 27.53 px. Tiny real holdout improved from 77.40 px to 72.69 px. | Keep as a pretraining artifact; not CAL-3 verified. |
| 2 | TrackNetV3 fine-tune from sparse/interpolated click labels | BALL stage ball trajectory tracking | Training completed, but both fine-tuned checkpoints predicted zero visible ball frames on held-out Wolverine. Held-out visible recall dropped from 0.75/1.00 to 0.00/0.00. | Do not deploy this fine-tuned checkpoint. Keep original pretrained TrackNet for now. |

Interpretation: before running, TrackNetV3 was the higher-impact target because ball tracking drives BALL/contact evidence. After benchmarking, the court-keypoint lane is the only one that produced a useful improvement signal. The TrackNetV3 sparse-click fine-tune is important, but this dataset/training recipe made it worse.

## H100 Runtime

- GCP VM: `body4d-gcp-prod`
- Zone: `us-west1-b`
- GPU observed inside container: NVIDIA H100 80 GB HBM3
- Container: `sam4dbody-pod-agent`
- Repo inside container: `/workspace/pickleball`
- Training output root: `/workspace/runs/pickleball_pretraining`

TrackNetV3 needed a DataLoader fallback. The first run with one PyTorch worker failed with a shared-memory bus error in Docker. The isolated TrackNetV3 copy was patched to `num_workers = 0`; that completed but was data-loader bound, with low GPU utilization.

## TrackNetV3 Dataset

Dataset builder:

```bash
PYTHONPATH=/workspace/pickleball /opt/conda/envs/fast_sam_3d_body/bin/python \
  scripts/racketsport/prepare_tracknetv3_finetune_dataset.py \
  --run-root runs/eval0/prototype_gate_h100_v2 \
  --review-root runs/eval0/prototype_gate_h100_v2/ball_click_review_30 \
  --out /workspace/runs/pickleball_pretraining/tracknetv3_clicks_dataset_20260628 \
  --overwrite
```

Local manifest:

- `runs/pickleball_pretraining/tracknetv3_clicks_dataset_20260628/pickleball_tracknetv3_dataset_manifest.json`

Generated dataset size on H100: 2.9 GB. The full frame dataset was not copied to the laptop; the manifest and converter are local, and the generated frames remain on the H100.

Split summary:

| Split | Clip | Frames | Human clicks | Interpolated visible frames | Total visible labels |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `burlington_gold_0300_low_steep_corner` | 600 | 24 | 336 | 360 |
| train | `outdoor_webcam_iynbd_1500_long_high_baseline` | 600 | 20 | 295 | 315 |
| train | `indoor_doubles_fwuks_0500_long_mid_baseline` | 300 | 21 | 129 | 150 |
| val/test | `wolverine_mixed_0200_mid_steep_corner` | 300 | 24 | 186 | 210 |

## TrackNetV3 Training

Training command:

```bash
cd /workspace/runs/pickleball_pretraining/TrackNetV3_finetune_repo
/opt/conda/envs/fast_sam_3d_body/bin/python train.py \
  --model_name TrackNet \
  --seq_len 8 \
  --epochs 34 \
  --batch_size 10 \
  --optim Adam \
  --learning_rate 0.001 \
  --bg_mode concat \
  --alpha 0.5 \
  --resume_training \
  --save_dir /workspace/runs/pickleball_pretraining/tracknetv3_finetune_20260628/train \
  --verbose
```

The run resumed from the pretrained `TrackNet_cur.pt` seeded at epoch 29 and completed printed epochs 31 through 34. Final checkpoint metadata:

| Checkpoint | Epoch | Validation max acc | SHA-256 |
| --- | ---: | ---: | --- |
| `TrackNet_best.pt` | 33 | 1.0 | `b09c8b7bc2d8bceb2da22757e1df213500a38363073faa76a10c6a5f6b75a55d` |
| `TrackNet_cur.pt` | 33 | 1.0 | `1a28c3a4e11b7bdc3f3fb896d73f3802f7efdf83a15fb0e9bb57e3d98af8e0c1` |

The validation accuracy is not reliable for model selection here. After epoch 32, validation counts showed the model predicting no visible balls and still receiving high accuracy from true negatives.

Local artifacts:

- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/train/TrackNet_best.pt`
- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/train/TrackNet_cur.pt`
- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/train/train.log`
- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/train/train.failed_workers1.log`

## TrackNetV3 Before/After Benchmark

Held-out clip: `wolverine_mixed_0200_mid_steep_corner`

Baseline artifact:

- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/before/ball_benchmark_before.json`

After artifacts:

- `runs/pickleball_pretraining/tracknetv3_finetune_20260628/after/ball_benchmark_after.json`
- `runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/tracknetv3_finetune_20260628_best/ball_track_0000_0010.json`
- `runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/tracknetv3_finetune_20260628_cur/ball_track_0000_0010.json`

| Candidate | Visible hit recall | Visible presence recall | Median error px | Hidden false positive rate | Teleports | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained TrackNetV3 raw | 0.75 | 1.00 | 6.06 | 0.80 | 27 | -0.23 |
| Fine-tuned `TrackNet_best.pt` | 0.00 | 0.00 | n/a | 0.00 | 0 | -0.35 |
| Fine-tuned `TrackNet_cur.pt` | 0.00 | 0.00 | n/a | 0.00 | 0 | -0.35 |

Conclusion: the sparse/interpolated click-label fine-tune completed technically, but it regressed the model. Do not promote the fine-tuned TrackNetV3 checkpoints to the BALL runtime.

## Court-Keypoint Pretraining

Training command:

```bash
cd /workspace/pickleball
PYTHONPATH=/workspace/pickleball /opt/conda/envs/fast_sam_3d_body/bin/python \
  scripts/racketsport/train_court_keypoint_heatmap.py \
  --real-root runs/eval0/prototype_gate_h100_v2 \
  --out /workspace/runs/pickleball_pretraining/court_keypoint_20260628 \
  --epochs 120 \
  --batch-size 32 \
  --real-finetune-start-epoch 70 \
  --eval-every 20 \
  --device cuda
```

Local artifacts:

- `runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt`
- `runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_metrics.json`
- `runs/pickleball_pretraining/court_keypoint_20260628/train.log`

Hashes:

| Artifact | SHA-256 |
| --- | --- |
| `court_keypoint_heatmap.pt` | `f750568336cb2b6e65bc10452b112210165bc220ded1de91d5d95ab99f6740f8` |
| `court_keypoint_metrics.json` | `70cdad78404498c50ce7602f5d38e6449345762d3eefe3f86adc8b40d93d06d3` |
| `train.log` | `6d84ee332dee8f1137ca8e1dec4caa17a71dc7d8d27f4c33cc68b62747c713d1` |

Metrics:

| Metric | Before | After |
| --- | ---: | ---: |
| Synthetic mean keypoint error | 67.31 px | 27.53 px |
| Real holdout mean corner error | 77.40 px | 72.69 px |
| Real train corners | 3 | 3 |
| Real holdout clips | 1 | 1 |

Conclusion: useful as a court-keypoint pretraining artifact, but the real holdout set is far too small to claim CAL verification. Treat status as `trained_not_phase_verified`.

## Local Verification

Focused local tests passed:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_pretraining_dataset_prep.py
python -m py_compile scripts/racketsport/prepare_tracknetv3_finetune_dataset.py scripts/racketsport/train_court_keypoint_heatmap.py
```

Result: `4 passed`.
