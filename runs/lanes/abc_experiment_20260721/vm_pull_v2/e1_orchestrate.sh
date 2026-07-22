#!/bin/bash
set -u
cd /home/arnavchokshi/pickleball
echo "=== E1 orchestration start $(date -u) ==="
run_arm () {
  ARM_NAME="$1"; MANIFEST="$2"; OUTDIR="runs/lanes/ball_event_abc_20260720/seed_20260720/$3"
  rm -rf "$OUTDIR"
  .venv/bin/python scripts/racketsport/finetune_event_head.py \
    --owner-manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
    --pseudo-manifest "runs/lanes/abc_experiment_20260721/abc_out_v2/$MANIFEST" \
    --init-checkpoint-model-only runs/lanes/ball_event_abc_20260720/inputs/frozen_t20_event_head.pt \
    --out "$OUTDIR" \
    --device cuda --steps 1000 --val-every 100 --batch-size 8 \
    --lr 0.001 --image-size 224 --window-frames 64 --stride-frames 32 \
    --num-workers 4 --class-weights 1.0 5.0 5.0 \
    --pseudo-weight-cap 1.0 --seed 20260720 --max-wall-minutes 120
  echo "$ARM_NAME train exit $? $(date -u)"
  .venv/bin/python - "$OUTDIR" "$ARM_NAME" <<PY
import json, sys
m = json.load(open(sys.argv[1] + "/finetune_manifest.json"))
assert m["completed_steps"] == m["target_steps"] == 1000, (sys.argv[2], m["completed_steps"])
print(sys.argv[2], "STEPS_OK 1000/1000 best_val_macro_f1_at_2:", m.get("best_val_macro_f1_at_2"))
PY
  if [ $? -ne 0 ]; then echo "${ARM_NAME}_VERIFY_FAILED"; exit 1; fi
}
run_arm B arm_b_manifest.json B_pbvision_teacher
run_arm C arm_c_manifest.json C_placebo
echo "=== evals $(date -u) ==="
for SPEC in "A:A_owner_only" "B:B_pbvision_teacher" "C:C_placebo"; do
  ARM="${SPEC%%:*}"; DIR="${SPEC##*:}"
  LOWER=$(echo "$ARM" | tr "A-Z" "a-z")
  .venv/bin/python scripts/racketsport/eval_event_head.py \
    --mode owner-val --device cuda \
    --manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
    --arm "$ARM" --seed 20260720 \
    --checkpoint "runs/lanes/ball_event_abc_20260720/seed_20260720/$DIR/best_event_head_finetuned.pt" \
    --threshold 0.5 \
    --out "runs/lanes/abc_experiment_20260721/abc_out_v2/${LOWER}_owner41.json"
  echo "$ARM eval exit $? $(date -u)"
done
echo "=== E1 orchestration DONE $(date -u) ==="
