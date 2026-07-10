# COURT-TRAIN-2 handoff

## Outcome and safety contract

- The `court_unet_v2` trainer now accepts LOADER-1 partial rows whether the internal row is
  sparse or explicitly carries `None`. Unlabeled channels have zero heatmap and visibility
  supervision and are excluded from trainer metric aggregation.
- `--init-from-checkpoint` is model-weights-only initialization. It deliberately does **not**
  restore optimizer, scheduler, scaler, epoch, or step state. `--resume` remains the separate
  full-state recovery path; the two flags fail loudly when combined.
- `--encoder-weights-path` accepts the pinned official torchvision ResNet34 file. That legacy
  file omits only `BatchNorm.num_batches_tracked`; PyTorch reconstructs those non-learned counters
  as zero. Missing learned parameters/running statistics, unexpected non-FC keys, and shape
  mismatches still fail loudly.
- `--real-split-proposal` restricts the real root to `train_datasets` from the proposal. Use it
  for both arms so the xuann/testworkspace validation family never trains.
- `--real-photometric-aug` is real-row-only, bounded, label-preserving color/blur/noise jitter.
  It defaults off.
- Protected evaluation clips remain EVAL-ONLY. Do not point `--real-root` at `eval_clips`, and do
  not run the owner gate without the required preregistration.

## ARM-A: synthetic-pretrained court_unet_v2 initialization

Run from repository root on a GPU worker with the repository checkpoints and corpus present:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/racketsport/train_court_model_v2.py \
  --out runs/lanes/court_train2_20260710/arm_a_<RUN_ID> \
  --epochs <EPOCHS> \
  --steps-per-epoch <STEPS_PER_EPOCH> \
  --batch-size <SYNTHETIC_BATCH_SIZE> \
  --real-batch-size <REAL_BATCH_SIZE> \
  --image-width 640 \
  --image-height 360 \
  --real-root runs/lanes/court_data2b_20260709/real_court_corpus_partial \
  --real-split-proposal runs/lanes/court_data2b_20260709/split_proposal.json \
  --real-weight <REAL_WEIGHT> \
  --synthetic-weight <SYNTHETIC_WEIGHT> \
  --init-from-checkpoint models/checkpoints/court_unet_v2/court_model_v2.pt \
  --real-photometric-aug \
  --checkpoint-every-eval \
  --keep-last-checkpoints 3 \
  --eval-every <EVAL_EVERY_EPOCHS> \
  --amp \
  --device cuda
```

Pinned initialization SHA-256:
`cdf0555d49335a946e518b177d85e2ab5be02100ba46eb3e634785c84f337c22`.
The output summary must report
`training.initialization.mode=model_checkpoint_fresh_optimizer` and `start_epoch=0`.

## ARM-B: ImageNet ResNet34 initialization

```bash
MPLBACKEND=Agg .venv/bin/python scripts/racketsport/train_court_model_v2.py \
  --out runs/lanes/court_train2_20260710/arm_b_<RUN_ID> \
  --epochs <EPOCHS> \
  --steps-per-epoch <STEPS_PER_EPOCH> \
  --batch-size <SYNTHETIC_BATCH_SIZE> \
  --real-batch-size <REAL_BATCH_SIZE> \
  --image-width 640 \
  --image-height 360 \
  --real-root runs/lanes/court_data2b_20260709/real_court_corpus_partial \
  --real-split-proposal runs/lanes/court_data2b_20260709/split_proposal.json \
  --real-weight <REAL_WEIGHT> \
  --synthetic-weight <SYNTHETIC_WEIGHT> \
  --encoder-weights-path models/checkpoints/court_external/torchvision/resnet34-b627a593.pth \
  --real-photometric-aug \
  --checkpoint-every-eval \
  --keep-last-checkpoints 3 \
  --eval-every <EVAL_EVERY_EPOCHS> \
  --amp \
  --device cuda
```

Pinned encoder SHA-256:
`b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`.
ARM-B must omit both `--init-from-checkpoint` and `--resume` for a fresh decoder and fresh
training state.

`<EPOCHS> * <STEPS_PER_EPOCH>` is the optimizer-step budget. The trainer always computes the
synthetic batch; `REAL_WEIGHT / (REAL_WEIGHT + SYNTHETIC_WEIGHT)` is the probability that a real
mini-batch is added on a step. For real supervision on every step alongside synthetic fallback,
use `--real-weight 1 --synthetic-weight 0`; for a 35% real-step probability, use `0.35 / 0.65`.

## CPU smoke evidence

The actual train-dataset corpus smoke used 2,833 rows before the two-row internal holdout:
34,917 labeled channels and 7,578 null channels. It ran 20 optimizer steps at 640x360 with batch
size 1, one real batch on every step, real photometric augmentation, and ARM-A initialization.

- wall: 17.8 seconds (epoch training time 12.36 seconds), below 15 minutes;
- step loss: `16.4797688 -> 10.5442257`;
- fixed real-train probe: `7.7103510 -> 7.0740108`;
- sampled real channels: 246 labeled, 54 null;
- null heatmap supervision violations: 0;
- null visibility supervision violations: 0.

Evidence: `cpu_smoke_actual/court_keypoint_metrics.json`. This is trainer integrity evidence,
not accuracy promotion evidence.

## Best-stack delta

**NO stack delta.** This lane changes trainer infrastructure only. No runtime default,
checkpoint selection, promotion status, or best-stack entry changed. `VERIFIED=0` remains binding.
