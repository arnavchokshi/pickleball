#!/usr/bin/env bash
# CAL-MODEL v2 A100 launch script (2026-07-05). NOT executed by this lane -- CPU-only sandbox,
# no network, no GPU. A GPU executor runs this later. Wraps the trainer in the shared-GPU
# exclusive lock (scripts/gpu-train-lock.sh) since it is a full training job, not a short eval.
#
# Usage: scripts/gpu-train-lock.sh bash runs/lanes/cal_model_20260705/train_a100.sh [extra args...]
#
# Prerequisites on the GPU host:
#   - CAL-SYNTH's threed/racketsport/court_synth_stream.py landed (falls back to
#     --synthetic-fallback automatically if not, but that fallback is a much weaker renderer --
#     do not promote off it).
#   - models/checkpoints/court_external/torchvision/resnet34-b627a593.pth present (or pass
#     --encoder-weights-path pointing at wherever it was synced to on this host).
#   - Optional: an on-disk external pickleball court corpus tier
#     (<root>/<clip>/labels/court_keypoints.json) for --real-root; the 32 owner-reviewed rows
#     under eval_clips/ball/ must NEVER be passed as --real-root (they are the CAL-3 promotion
#     eval set -- see scripts/racketsport/evaluate_court_model_v2.py, which is the ONLY script
#     allowed to touch them, read-only).
#
# Batch size / epoch time estimate (A100 40GB, AMP on, resnet34+U-Net decoder ~23.9M params,
# 640x360 input): this CPU sandbox measured ~1.9-2.3 s/step at batch=64 on a laptop CPU with no
# GPU and no AMP (see runs/lanes/cal_model_20260705/report.md "CPU smoke numbers"). An A100 with
# AMP should comfortably run batch=128-256 at well under 100 ms/step for this model size (a
# resnet34-scale backbone at 640x360 is a small fraction of an A100's throughput), so 500
# steps/epoch at batch=128 is a conservative ~1-2 minutes/epoch; scale --epochs and
# --steps-per-epoch to fit the GPU time budget actually granted by gpu-train-lock.sh. Re-measure
# actual step time in the first few logged steps and adjust rather than trusting this estimate
# blindly -- it is an extrapolation from a CPU run, not a GPU measurement.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${CAL_MODEL_V2_OUT_DIR:-runs/cal_model_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
ENCODER_WEIGHTS="${CAL_MODEL_V2_ENCODER_WEIGHTS:-models/checkpoints/court_external/torchvision/resnet34-b627a593.pth}"

python scripts/racketsport/train_court_model_v2.py \
  --out "$OUT_DIR" \
  --device cuda \
  --amp \
  --epochs 300 \
  --steps-per-epoch 500 \
  --batch-size 128 \
  --image-width 640 \
  --image-height 360 \
  --encoder-weights-path "$ENCODER_WEIGHTS" \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seg-loss-weight 1.0 \
  --vis-loss-weight 0.2 \
  --geometric-loss-weight 0.1 \
  --val-samples 256 \
  --eval-every 5 \
  --seed 13 \
  "$@"

echo "CAL-MODEL v2 training run complete. Checkpoint + court_keypoint_metrics.json under: $OUT_DIR"
echo "Next: score the resulting checkpoint against the 32 reviewed real rows (never trained on):"
echo "  python scripts/racketsport/evaluate_court_model_v2.py \\"
echo "    --checkpoint $OUT_DIR/court_model_v2.pt \\"
echo "    --out $OUT_DIR/owner_gate_report_v2.json --device cuda"
