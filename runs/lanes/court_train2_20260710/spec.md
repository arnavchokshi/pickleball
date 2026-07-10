# COURT-TRAIN-2 — bridge the v2 trainer to the real masked corpus (repo gap from TRAIN-1)

## HARD RULES
- No branches/commits/pushes. Read NORTH_STAR_ROADMAP.md CAL rows,
  runs/lanes/court_wave_20260709/DESIGN_RULING.md, runs/lanes/court_train1_20260709/REPORT.md
  (the architecture-gap finding), runs/lanes/court_loader1_20260709/HANDOFF.md (null schema).
- Protected eval clips EVAL-ONLY; tests use fixtures. No training data from gate sources.
- Honest reporting; wide suite (MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport) at end.
- Artifacts under runs/lanes/court_train2_20260710/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/train_court_model_v2.py + tests/racketsport/test_train_court_model_v2.py
- runs/lanes/court_train2_20260710/**
Do NOT touch: train_court_keypoint_heatmap.py (only import from it), the corpus builder,
evaluate_court_keypoint_owner_gate.py, threed/racketsport/court_keypoint_net.py.

## CONTEXT (verified facts)
- train_court_model_v2.py already: trains the real court_unet_v2 (640x360, resnet34), supports
  --real-root via load_real_court_keypoint_labels (which NOW returns rows that may carry
  per-keypoint None/null for unlabeled points, per COURT-LOADER-1), --real-weight/
  --synthetic-weight sampling, deterministic real holdout split, resumable checkpoints
  (CALV1 trainerfix), loud-fail encoder loading.
- GAP: real_row_to_sample_arrays + the loss path were written when real rows always had 15
  labeled points. With null keypoints they will either crash or (worse) silently supervise
  garbage. TRAIN-1 proved the ladder needs THIS trainer for court_unet_v2 arms.
- The evaluate_court_keypoint_owner_gate.py CLI already scores court_unet_v2 checkpoints.

## MISSION
1. MASKED-NULL SAFETY: make real_row_to_sample_arrays + the training loss consume LOADER-1's
   null-marker rows correctly: per-row per-keypoint supervision mask; null channels contribute
   ZERO to heatmap loss (peak AND background) and are excluded from any metric aggregation;
   fail-loud on malformed rows. Mirror the semantics LOADER-1 landed in the heatmap trainer
   (read its implementation; import shared helpers where clean rather than duplicating).
2. CHECKPOINT-INIT ARM SUPPORT: verify/complete an --init-from-checkpoint path that loads
   models/checkpoints/court_unet_v2/court_model_v2.pt weights as INITIALIZATION (distinct from
   resume: fresh optimizer/step count). If the existing resume machinery already supports this
   cleanly, document the exact invocation instead of adding a flag.
3. IMAGENET ARM SUPPORT: confirm the existing encoder path can init from
   models/checkpoints/court_external/torchvision/resnet34-b627a593.pth (loud-fail loader
   already exists); document exact invocation.
4. REAL-ROW PHOTOMETRIC AUG (small, bounded): add a --real-photometric-aug flag (default OFF)
   applying label-preserving color/blur/noise jitter to real rows only. Keep it <80 lines.
5. TESTS: (a) null-masked real row -> zero loss contribution on null channels (garbage-pred
   invariance test); (b) full-15 real row -> loss identical pre/post change (regression);
   (c) init-from-checkpoint loads court_model_v2.pt and produces a checkpoint whose eval runs
   through evaluate_court_keypoint_owner_gate.py on a 2-frame fixture root (CPU, tiny); 
   (d) direct-CLI smoke: 5-step CPU train on a fixture mixing real partial rows + synthetic
   fallback completes, writes checkpoint, checkpoint loads back.
6. CPU SMOKE E2E: run a real 20-step train on the ACTUAL corpus
   (runs/lanes/court_data2b_20260709/real_court_corpus_partial, train datasets only per
   split_proposal.json) at reduced batch, assert loss decreases and no null-channel supervision
   occurs (instrument + assert). Wall-bounded (<15 min).

## ACCEPTANCE
- A1: all tests green incl. garbage-pred invariance + full-15 regression.
- A2: CPU smoke E2E documented with loss curve numbers.
- A3: exact GPU invocations for ARM-A (init court_model_v2.pt) and ARM-B (imagenet) written
  into HANDOFF.md, incl. --real-root/--real-weight/--synthetic-weight/steps placeholders.
- A4: wide suite failures==0 or proven pre-existing; write scope clean.
## BEST-STACK DELTA
Expected (c) NO stack delta (trainer infra). State explicitly.
## REPORT
Structured report per schema + HANDOFF.md as above.
