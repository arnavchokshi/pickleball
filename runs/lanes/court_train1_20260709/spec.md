# COURT-TRAIN-1 — real-transfer probe ladder on H100 (design ruling R2/R8)

You are a self-contained GPU lane. ANTI-PASSIVE-WAIT: ending your turn to wait = lane death;
you will NOT be re-woken; background monitors will not fire for you. Poll every remote job
with bounded foreground until-loops (sleep 60-240s checking progress AND failure signatures).
End only with the final structured report or a hard blocker.

## HARD RULES
- Read NORTH_STAR_ROADMAP.md CAL rows + runs/lanes/court_wave_20260709/DESIGN_RULING.md first.
- NO PROMOTION CLAIMS. VERIFIED=0. This is a dev-gate probe ladder.
- Protected eval clips: Burlington + Wolverine labels = internal-val ALLOWED. Outdoor/Indoor
  labels FORBIDDEN (never load/score them; when scoring eval_clips, build a temp root
  symlinking ONLY burlington_gold_0300_low_steep_corner and wolverine_mixed_0200_mid_steep_corner).
- Protocol S: NO training row may come from 73VurrTKCZ8/HyUqT7zFiwk/zwCtH_i1_S4 or any
  eval clip. Training data = Roboflow corpora + synthetic ONLY.
- Never co-schedule two training arms on one GPU (standing gotcha). Sequential arms.
- nohup every VM step. All Mac-side artifacts under runs/lanes/court_train1_20260709/.
- Fleet rules: SPOT H100 a3-highgpu-1g from snapshot pickleball-fleet-snap-20260709-w7close,
  zone asia-southeast1-b then -c ladder, pd-balanced 200GB, labels fable-lane=court-train1,
  fable-fleet=pickleball; in-VM 60-min no-heartbeat self-stop watchdog; DELETE + list-confirm
  + cost accounting at end (wall cap 3.5h, budget ~$2-15, ceiling $22 w/ contingency).

## PHASES
P0 PROVISION + SYNC: create VM; wait sshd (~90s); refresh known_hosts. VM repo: git fetch
   origin && git reset --hard <the commit this spec names in DISPATCH NOTE below>; md5-compare
   scripts/racketsport/{train_court_keypoint_heatmap,evaluate_court_keypoint_owner_gate}.py +
   threed/racketsport/court_keypoint_net.py against Mac; version-stamp in lane dir. Verify
   baked corpora: data/roboflow_universe_20260706 exists on VM (snapshot-baked) — verify 3
   random file md5s vs Mac; models/checkpoints/court_unet_v2/court_model_v2.pt exists (baked,
   sha256 must match best_stack pin cdf0555d...). CORPUS PLAN (PRIMARY): the Mac corpus symlinks are ABSOLUTE Mac paths and WILL dangle
   on the VM — REBUILD both corpora ON the VM via the committed build_real_court_corpus.py
   CLI (roboflow corpus is snapshot-baked; verify row counts + a sample of row sha256s match
   the Mac lane manifests exactly; mismatch = STOP and report). rsync from Mac only the small
   JSON manifests for cross-checking + split_proposal.json + runs/lanes/w7_courtkpingest_20260709/gt_roots/corrected_r2 (harvest dev GT, ~15MB) +
   models/checkpoints/court_external/torchvision/resnet34-b627a593.pth (md5-verify).
   Synthetic corpus: check runs/training_corpora_20260701/court_synthetic on VM; if absent,
   generate a fresh bounded corpus on VM via generate_synthetic_court_keypoints.py (document
   count + families; target the trainer's expected layout).
P1 CONTROL ROW (before any training): score frozen court_model_v2.pt with
   evaluate_court_keypoint_owner_gate.py on:
   - CARD-A: --real-root <corrected_r2 root> (5 harvest frames, 3 sources; raw + aggregated)
   - CARD-B: temp root w/ Burlington+Wolverine only
   Record PCK@5/PCK@10/median/p95 per clip + pooled. Expected ~0.0 PCK / ~10^2-10^3 px.
   Also score CARD-A/B with --enable-homography-refinement for the control (snap baseline).
P2 PROBE: ~100-step fine-tune run (ARM-A config) to measure steps/s; set step budget =
   min(4000, floor(45min * steps_per_sec)); report the formula inputs.
P3 ARM-A: init from court_model_v2.pt; train on TRAIN datasets from split_proposal.json
   (partial corpus w/ masked nulls + chetan full15 rows) + synthetic minority (~35% of batches
   if the trainer supports mixing; else document the closest supported mix); dataset-balanced
   sampling to the extent the trainer supports; standard aug flags. Mandatory training-lane
   check: push ONE sample through the training dataloader AND the production inference
   preprocessor; assert identical tensors or document the exact mapping in the report.
P4 EVAL ARM-A: CARD-A + CARD-B, raw + aggregated + homography-refinement variants. Also score
   the VAL datasets from split_proposal.json (source-disjoint external val).
   RULING BARS (R2): CARD-A pooled median <25px AND PCK@5 >= +0.30 absolute over control.
P5 ARM-B (if wall budget allows; drop first): identical recipe, init ImageNet resnet34
   (commercial-clean lineage; checkpoint_policy local_only w/ the court_external path).
   Same eval. Purpose: does synthetic pretraining help or hurt real transfer?
P6 PULL + TEARDOWN: pull checkpoints (final + best-val), all eval reports, training logs,
   sample prediction overlays (render 10 CARD-A/B frames w/ predicted vs GT points) to the
   lane dir with md5 manifest; DELETE VM; gcloud list-confirm absent; report uptime + $ range.

## ACCEPTANCE
- A1: control rows recorded for both cards BEFORE training (exact numbers).
- A2: probe-measured step budget documented.
- A3: ARM-A trained + evaluated on all cards; per-source numbers + kill-bar verdicts stated.
- A4: dataloader/production-preprocessor parity check documented.
- A5: teardown list-confirmed + honest cost span.
- HONEST-OUTCOME RULE: if the kill bar fails, that is a finding — report it plainly; do NOT
  tune thresholds or re-slice to manufacture a pass.

## BEST-STACK DELTA
Expected (b) at most: if an arm clears the R2 bar, add/update a PENDING candidate row note in
the report (manager wires the manifest at ruling). NO promoted rows from this lane.

## REPORT
Return final structured JSON as your last message AND write runs/lanes/court_train1_20260709/
REPORT.md via Bash: {objective_result, control_rows, probe, arms:[{name, init, data_mix,
steps, card_a, card_b, val_external, homography_variant, kill_bar_verdict}], honest_issues,
artifacts, cost, fleet_accounting}.
