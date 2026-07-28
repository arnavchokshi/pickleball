#!/usr/bin/env bash
# ev2_train_20260728: fine-tune on owner's 102 labels + eval discipline. Run on the VM
# from the repo root, AFTER the resumed pretrain (train_resume/{best,last}_event_head.pt)
# has completed. Lane-owned; not a new pipeline entrypoint (reuses finetune_event_head.py
# and eval_event_head.py exactly as shipped).
set -euo pipefail

LANE_DIR="runs/lanes/ev2_train_20260728"
INIT_CKPT="${1:-$LANE_DIR/train_resume/best_event_head.pt}"

echo "== using init checkpoint: $INIT_CKPT =="
test -f "$INIT_CKPT"

# 1. Fresh Step-0 gate proof for the fine-tune inputs (proofs expire in 900s).
.venv/bin/python scripts/racketsport/verify_training_inputs.py \
  --inputs "$LANE_DIR/training_inputs_finetune.json" \
  --ledger runs/manager/data_ledger.json \
  --repo-root . \
  --gate-proof "$LANE_DIR/gate_proof_finetune_FRESH.json"

# 2. Fine-tune on the owner's 61-train/41-val split (owner-val mode: val is never
#    trained on; protected 50-row seed is hard-fail-checked before any decode).
.venv/bin/python scripts/racketsport/finetune_event_head.py \
  --gate-proof "$LANE_DIR/gate_proof_finetune_FRESH.json" \
  --owner-manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
  --init-checkpoint-model-only "$INIT_CKPT" \
  --out "$LANE_DIR/finetune" \
  --device cuda \
  --steps 400 \
  --image-size 224 \
  --window-frames 64 \
  --batch-size 8 \
  --lr 0.0005 \
  --val-every 50 \
  --seed 20260716 \
  --stride-frames 32 \
  --num-workers 4 \
  --checkpoint-selection owner-val \
  --class-weights 1 5 5 \
  --max-wall-minutes 15 \
  | tee "$LANE_DIR/finetune_result.json"

FT_CKPT="$LANE_DIR/finetune/best_event_head_finetuned.pt"
test -f "$FT_CKPT"

# 3. Matched-window public eval, threshold sweep (>=50 clips, default --max-clips 50).
for THRESH in 0.5 0.3 0.2 0.1 0.05; do
  .venv/bin/python scripts/racketsport/eval_event_head.py \
    --checkpoint "$FT_CKPT" \
    --mode public \
    --out "$LANE_DIR/eval_public_thresh_${THRESH}.json" \
    --threshold "$THRESH" \
    --max-clips 50 \
    --device cuda
done

# 4. Owner-val eval on the fine-tuned checkpoint: macro-F1@2, timing p90, and the
#    automatic full-video firing rate over the distinct owner-val source videos.
.venv/bin/python scripts/racketsport/eval_event_head.py \
  --checkpoint "$FT_CKPT" \
  --mode owner-val \
  --out "$LANE_DIR/eval_owner_val.json" \
  --threshold 0.5 \
  --arm A \
  --seed 20260716 \
  --device cuda

# 5. Direct firing-rate measurement on the 697s pb.vision demo clip, for numeric
#    comparability with the North Star's 7.16 HIT/s zero-shot-transfer figure.
.venv/bin/python "$LANE_DIR/measure_firing_rate.py" \
  --checkpoint "$FT_CKPT" \
  --video "$LANE_DIR/eval_media/pbvision_11min_demo.mp4" \
  --threshold 0.5 \
  --device cuda \
  --out "$LANE_DIR/firing_rate_pbvision_demo.json"

# 6. ONE-TOUCH protected 50-row owner seed score (eval only, never repeated).
.venv/bin/python scripts/racketsport/eval_event_head.py \
  --checkpoint "$FT_CKPT" \
  --mode protected-seed \
  --out "$LANE_DIR/eval_protected_seed_ONE_TOUCH.json" \
  --threshold 0.5 \
  --device cuda

echo "== all evidence written under $LANE_DIR =="
