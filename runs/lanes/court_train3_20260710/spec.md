# COURT-TRAIN-3 — court_unet_v2 real-transfer arms on H100 (the rung TRAIN-1 could not run)

Same operating rules as runs/lanes/court_train1_20260709/spec.md (anti-passive-wait, nohup,
ssh -i direct, sequential arms, protected-data rules, Protocol S, fleet caps, teardown+cost).
Differences and mission below. Wall cap 3.0h from VM RUNNING; budget ~$2-13 (ceiling $20).

## READ FIRST
- runs/lanes/court_train2_20260710/HANDOFF.md — the EXACT ARM invocations (consume verbatim).
- runs/lanes/court_train1_20260709/REPORT.md — control rows already banked; the architecture-gap
  context; the 27-attempt stockout note (budget provisioning patience; ase1-b then -c ladder).
- runs/lanes/court_wave_20260709/DESIGN_RULING.md (R1/R2/R7/R8).

## DELTAS vs TRAIN-1
- Code pin: the commit named in the DISPATCH NOTE below (includes the v2-trainer bridge).
- TRAINER: scripts/racketsport/train_court_model_v2.py (court_unet_v2 640x360) — NOT the
  legacy heatmap trainer.
- TRANSFER models/checkpoints/court_external/torchvision/resnet34-b627a593.pth from Mac
  (md5-verify; absent from snapshot — confirmed by TRAIN-1).
- CONTROL: re-run the frozen court_model_v2.pt control on CARD-A/CARD-B once (cheap; verify
  it reproduces TRAIN-1's banked rows within noise: CARD-A 0.0 PCK/median 942.34px, CARD-B
  0.0/675.15px; divergence >1px median = STOP and diagnose before training).
- ARMS (sequential, per TRAIN-2 HANDOFF invocations, both on the real partial corpus train
  datasets + synthetic minority ~35%; real photometric aug ON if TRAIN-2 shipped it):
  ARM-A: init from court_model_v2.pt (synthetic-pretrained v2).
  ARM-B: imagenet resnet34 encoder init (commercial-clean lineage).
  Probe ~100 steps first on ARM-A config; step budget = min(6000, floor(50min*steps/s)).
- EVAL per arm: CARD-A (corrected_r2 harvest GT, raw + aggregated), CARD-B (Burl+Wolv temp
  root), external val datasets (split_proposal.json), each with and without
  --enable-homography-refinement. Render 10 prediction overlays per card per arm.
- KILL BAR (R2, unchanged): CARD-A pooled median <25px AND PCK@5 >= +0.30 over control.
  Secondary report: PCK@10, per-source, floor-12-only PCK (null-net honesty), val-external.
- If ARM-A fires the bar: run ONE extra eval variant — aggregated_static_camera_median over
  16 sampled frames per harvest source video nearest each GT frame timestamp (clip-level
  inference simulation; frames from the SAME rally segment only, camera cuts noted by
  DATA-1 make cross-rally aggregation invalid).
## ACCEPTANCE
A1 control reproduction; A2 probe-set budget; A3 both arms trained+evaled (or honest drop of
ARM-B at wall cap with banked checkpoint); A4 kill-bar verdicts per arm stated plainly;
A5 teardown list-confirmed + cost span; A6 md5 manifest of pulled artifacts.
## BEST-STACK DELTA
(b) at most: PENDING candidate note if a bar fires; manager wires manifest at ruling.
## REPORT
REPORT.md + final JSON: {objective_result, control_reproduction, probe, arms[], overlays,
honest_issues, artifacts, cost, fleet_accounting}.
