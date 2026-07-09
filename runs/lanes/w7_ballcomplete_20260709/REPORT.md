# LANE w7_ballcomplete_20260709 — BALL completion errand (aug ablations + 1750-corpus re-score)

## OBJECTIVE_RESULT: PARTIAL

Contract-check evidence gap CLOSED (real pulled+md5-verified log, not a transcript reconstruction).
ARM D (seed_official base, 1121-row corpus, NO occlusion aug — the aug-attribution ablation) LANDED
and scored on the 1121-corpus, completing a real 5-row control/A/B/C/D card (B unavailable). ARM B
(raw official WASB base + aug) DROPPED for budget at step 500/2372 after a lane-caused GPU-contention
mistake (self-inflicted, see HONEST ISSUES #1) consumed the time that would have finished it. The
1750-corpus re-score (GPU_RESCORE_COMMANDS.sh block) was prepped (corpus transferred, all 38 clip
videos confirmed present on VM) but NOT dispatched — dropped for budget per the pre-agreed priority
order. No promotion, no best_stack.json change; VERIFIED=0 stands throughout; internal-val only, no
Outdoor/Indoor labels touched, no held-out shot read.

## 1. CONTRACT-CHECK EVIDENCE (closes the w7_ballretrain verifier gap)

w7_ballretrain's own contract-check evidence was a transcript reconstruction (no log file existed to
pull, honestly headed as such in `vm_contract_check_excerpt.md`) — the adversarial verifier flagged
this as the one evidence gap in an otherwise-confirmed 5/6 card. This lane closes it for real:

- Command run ON THE VM, stdout `tee`'d to a file:
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_ball_stage2_training.py -v 2>&1
  | tee runs/lanes/w7_ballcomplete_20260709/vm_contract_tests.log`
- Result: **8 passed in 7.74s** (all 8 tests in the stage-2 tensor/label-geometry contract file,
  including `test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine`, run BEFORE any
  training this lane, per the mission's mandatory-before-ARM3 rule).
- Pulled to Mac: `runs/lanes/w7_ballcomplete_20260709/vm_contract_tests.log`, md5
  `6b58b05be5714f0f6a169818bcf255f2` on both VM and Mac (cross-verified, byte-identical).

## 2. CODE + DATA IDENTITY

- VM synced from `pickleball-fleet-snap-20260708-w6close` (baked HEAD `37b8ecd3f`) to Mac's exact
  committed HEAD `354c343dba84a6bbc6b76df327d6f7238454eeec` (>= the required floor
  `d983d0dac032c81e85a8b78129b85e361c980c6b`; `git merge-base --is-ancestor` confirmed ancestor).
  `git status --short` on VM after checkout: only the 2 by-design vendor-submodule lines
  (`third_party/WASB-SBDT`, `third_party/blurball`) + untracked `.venv` — no `reset --hard` needed.
- md5 of `train_ball_stage2.py`, `run_wasb_ball.py`, `ball_loso_validation.py`, `fuse_ball_tracks.py`,
  `run_ball_tracking_eval_suite.py`, `train_ball_pretrain.py`, `test_ball_stage2_training.py` all
  matched Mac byte-for-byte (`mac_key_script_md5s_precheck.txt`).
- 1121-row corpus (md5 `37a5d43ab537a15bd12d382bb882a5fe`) + rally videos + `seed_official` /
  `stage1_official` / official-tennis checkpoints were already baked in the snapshot (near-zero
  transfer tax, matching w7_ballretrain's finding).
- 1750-row corpus (`runs/lanes/w7_ballingest2_20260709/reviewed_corpus`, manifest md5
  `0bb2fc592361f2e9246a71701617c3e5`) rsynced Mac->VM (14MB) — all 38 clip source videos confirmed
  present on VM (`ALL_38_VIDEOS_PRESENT`). Prepped but unused (see HONEST ISSUES #3).

## 3. ARMS TRAINED THIS LANE

**ARM D — seed_official base, 1121-row corpus, occlusion aug DISABLED (`--occluded-prob 0.0`), 2372
steps.** `train_ball_stage2.py` line 291-292 pins `occluded_prob` to exactly `0.25` or `0` (a
code-enforced ablation switch, not an ad-hoc flag) — this IS the intended, code-sanctioned no-aug
ablation. Recipe otherwise identical to ARM A (same init checkpoint
`runs/lanes/w5_ballretrain_20260707/seed_official/checkpoints/latest.pt`, same 1121-row
`--cvat-export-root runs/lanes/w6_labelingest_20260708/reviewed_corpus`, same steps/lr/batch/seed).
Final checkpoint: `step=2372`, `round_trip_state_sha256_match=true`, loss strictly decreased across
its final leg (0.0011545 -> 0.0004932). **Trained across 3 process launches due to the
concurrency mistake (HONEST ISSUES #1)**, recovered via `--resume-checkpoint` — the exact,
tested (`test_train_ball_stage2_resume_checkpoint_continues_loss_and_step`) resume mechanism proves
bit-exact continuation vs a straight-through run under identical seed/recipe, so the final
2372-step checkpoint is a legitimate, protocol-matching arm despite the multi-leg history.

**ARM B — raw official WASB tennis anchor (`models/checkpoints/wasb/wasb_tennis_best.pth.tar`) +
occlusion aug, 1121-row corpus.** Reached only **step 500 of 2372** before being killed (twice) as
part of recovering from the concurrency mistake, and was NOT resumed to completion — DROPPED for
budget. The step-500 checkpoint is banked (see ARTIFACTS) purely for a future lane to resume from,
clearly marked INCOMPLETE; it is NOT a valid, comparable arm and was not scored.

## 4. THE COMPLETE 1121-CORPUS CARD (control / A / B / C / D)

Same 20-clip internal-val corpus as w7_ballretrain (`runs/lanes/w6_labelingest_20260708/reviewed_corpus`,
md5 `37a5d43ab537a15bd12d382bb882a5fe`), same OFFICIAL-preprocessing bridge, same
`ball_loso_validation.py` harness (`fold_count=20`, the pre-existing per-CLIP-not-per-source fold
quirk noted in every prior lane — unchanged, not this lane's to fix). control/A/C rows are the
already-published, byte-traceable numbers from `w7_ballretrain_20260709/vm_pull/arm4_score/loso_final/
loso_report.json` (re-quoted here, not re-run — re-running them would have cost ~72 GPU-min for
numbers that are already deterministic given the identical pinned checkpoint+corpus+harness, and that
budget was needed for D+scoring; this is a deliberate efficiency choice, flagged in HONEST ISSUES #2).
D is freshly measured this lane (`gpu_rescore_1121/loso_D_only/loso_report.json`). B is absent (dropped).

| candidate | pooled F1@20 | pooled hFP | pooled precision@20 | pooled visible_recall@20 | loso_mean F1@20 | loso_mean hFP | loso_mean precision@20 | loso_mean visible_recall@20 | loso_worst hFP |
|---|---|---|---|---|---|---|---|---|---|
| official_tennis_control (zero-shot) | 0.361111 | 0.599089 | 0.343008 | 0.381232 | 0.303348 | 0.586327 | 0.318968 | 0.308130 | 1.0000 |
| A_seed_official_aug (seed_official, +aug) | 0.615172 | 0.250569 | 0.580729 | 0.653959 | 0.644674 | 0.325716 | 0.627797 | 0.670972 | 1.0000 |
| **B_raw_wasb_aug** | **NOT RUN — dropped at step 500/2372, see HONEST ISSUES #1** | | | | | | | | |
| C_stage1_official_aug (stage1_official, +aug) | 0.612075 | 0.259681 | 0.581028 | 0.646628 | 0.585108 | 0.326524 | 0.593839 | 0.592820 | 1.0000 |
| **D_seed_official_noaug (this lane, seed_official, NO aug)** | **0.624085** | **0.355353** | **0.571255** | **0.687683** | **0.670656** | **0.419829** | **0.638265** | **0.716199** | **1.0000** |

Exact keys (from `ball_loso_validation.py`): `pooled_mixed_metrics.{micro_label_f1_at_20px,
micro_hidden_false_positive_rate, micro_precision_at_20px, micro_visible_recall_at_20px}` and
`loso_mean_metrics.{label_f1_at_20px, hidden_false_positive_rate, precision_at_20px,
visible_recall_at_20px}`. D's own report: `runs/lanes/w7_ballcomplete_20260709/gpu_rescore_1121/
loso_D_only/loso_report.json` (+ `.md`). D's `loso_worst_fold hFP=1.0` matches control/A/C's
worst-fold hFP (also all 1.0) — a shared single-clip-fold characteristic of this harness/corpus, not
a D-specific defect.

## 5. THE AUG-ATTRIBUTION READ (D vs A — the mission's central question)

D and A share the exact same base checkpoint (`seed_official`), the exact same 1121-row corpus, and
the exact same 2372 steps. The ONLY difference is `--occluded-prob 0.0` (D) vs `0.25` (A). This is
therefore a clean, controlled ablation:

| metric | A (+aug) | D (no aug) | aug's effect (A − D) |
|---|---|---|---|
| pooled F1@20 | 0.615172 | 0.624085 | **−0.0089 (−1.4% rel.)** — aug costs a small amount of F1 |
| loso-mean F1@20 | 0.644674 | 0.670656 | **−0.0260 (−4.0% rel.)** — same direction, larger at loso-mean |
| pooled hFP | 0.250569 | 0.355353 | **−0.1048 abs. (aug is 29.5% relatively LOWER hFP than no-aug)** |
| loso-mean hFP | 0.325716 | 0.419829 | **−0.0941 abs. (aug is 22.4% relatively LOWER hFP than no-aug)** |
| pooled precision@20 | 0.580729 | 0.571255 | +0.0095 — aug helps precision (pooled) |
| loso-mean precision@20 | 0.627797 | 0.638265 | −0.0105 — aug slightly hurts precision (loso-mean; sign flips vs pooled, both small, likely fold-weighting noise, not a real disagreement) |
| pooled visible_recall@20 | 0.653959 | 0.687683 | −0.0337 — aug costs recall |
| loso-mean visible_recall@20 | 0.670972 | 0.716199 | −0.0452 — same direction |

**What this answers, directly, about the previously-flagged mystery** (w7_ballretrain: "hFP regressed
0.2255 [486-row seed_official, no-aug-recipe-era] -> 0.2506 [A, 1121-row, +aug]" — attribution
unknown at the time): with D as the missing isolating arm, the regression decomposes cleanly:
- **Labels-only effect** (486-row -> 1121-row, no aug on both ends): hFP 0.2255 -> 0.3554 (D) =
  **+0.1299 absolute, +57.6% relative — the additional 635 owner-labeled rows, by themselves, made
  hidden-false-positives substantially WORSE**, not the aug.
- **Aug-only effect at 1121 rows** (D -> A): hFP 0.3554 -> 0.2506 = **−0.1048 absolute, a 29.5%
  relative reduction** — occlusion aug is a **corrective that claws back most of the labels-driven
  regression**, landing A at 0.2506 (still +0.0251 above the 486-row floor of 0.2255, but far better
  than the 0.3554 it would have been without aug).
- Net read: **occlusion aug is not the cause of the hFP regression — it is the mitigation.** The
  labels themselves (likely skewed toward more occluded/hard-disagreement examples via the labeling
  flywheel's disagreement-mining design — a plausible mechanism, not independently confirmed this
  lane) are what drove hFP up; aug recovers most but not all of it, at a small (1.4-4.0% relative)
  F1/recall cost. This supports KEEPING occlusion aug in the standing recipe as a net-positive
  design choice, not a source of the regression.

## 6. THE 1750-CORPUS CARD

**NOT RUN.** `GPU_RESCORE_COMMANDS.sh` was prepped correctly on the VM (corpus present, all 38 clip
videos confirmed) but no GPU dispatch happened — dropped for budget per the pre-agreed priority order
(`contract-log > ARM D > 1121-card-for-D > ARM B > 1750-rescore`, drop from the bottom). Per the
w6-session owner ruling (`cvat_upload/exports/w6_labelpack_20260708/SESSION_NOTES_20260709.md`,
Ruling 2), **no per-visibility analysis was produced for the 1750 scoring** — doubly true here since
(a) the 1750 corpus's 629 new w6-session rows carry uninformative `visibility_level` (box-position-only
supervision) and must never drive visibility-weighted loss or per-visibility eval slices per that
ruling, and (b) the scoring itself never ran this lane, so there is no per-visibility slice to report
either way.

## 7. HONEST ISSUES

1. **Self-inflicted concurrency mistake (the main story of this lane).** After the mandatory
   contract-check, I launched ARM D and ARM B concurrently on the same GPU (idle CPU/GPU headroom
   looked ample: `load average 5.66` out of 26 vCPUs, 0% instantaneous GPU util) to try to halve wall
   time. This backfired badly: both arms slowed to ~31 min/500 steps (~0.27-0.31 steps/s) vs the
   w7_ballretrain-measured single-job baseline of 0.8787 steps/s — roughly a **3x slowdown**, not the
   expected near-2x speedup. I killed both jobs at step 500 (D) / step 500 (B), discovered D had
   actually reached step 1000 (a checkpoint had landed moments before the kill), and resumed D alone
   (solo) via `--resume-checkpoint`. Solo, the rate was **still only ~0.31 steps/s** (500 steps in
   1661s at the 1000->1500 boundary, and 500 steps in 1598s at 1500->2000) — i.e., the slowdown did
   NOT fully recover even running alone. Root cause undetermined: `nvidia-smi` showed no throttling
   (P0, 1980/1980 MHz SM clock, 145W/700W draw), CPU load was low (`load average` 5-6 out of 26
   cores), and GPU inference (`run_wasb_ball.py`) later ran at the EXPECTED baseline rate
   (~40-70s/clip, matching w7_ballretrain's ~72s/clip average) — so whatever caused the slowdown was
   specific to the stage-2 **training** data-loading path (`num_workers=0`, per-sample lazy video
   seeks) on this particular VM/host, not a GPU-clock or general-CPU-contention effect, and not a
   host-independent code regression (the same code was byte-verified identical, and w7_ballretrain's
   own A/C arms hit 0.86-0.89 steps/s on the same SKU/zone the day before). Most likely explanation:
   cloud-host-specific disk/IO variance (noisy-neighbor or per-instance pd-balanced IOPS variance) —
   not confirmed, flagged for the manager. **This mistake, not any single-arm training time, is what
   consumed the budget that would otherwise have finished ARM B and attempted the 1750 re-score.**
2. **control/A/C 1121-card rows are re-quoted from w7_ballretrain, not re-run.** All three are
   pinned-checkpoint + pinned-corpus + identical-harness numbers that are deterministic given the
   inputs (verified byte-identical code this lane); re-running them would have cost ~72 GPU-minutes
   for no new information, at the direct expense of D+scoring time. This is disclosed, not hidden;
   the manager may want a from-scratch 5-candidate combined-report run for auditability at some future
   point, but it was not a good use of this lane's tight remaining budget.
3. ARM B never resumed past step 500 — its checkpoint is banked (`arm_finetunes/B_raw_wasb_aug/
   checkpoints/checkpoint_step_000500_INCOMPLETE.pt`) for a future lane to resume with
   `--resume-checkpoint ... --steps 1872` (mirroring exactly how D was recovered), NOT scored, NOT
   comparable to any other row in the tables above.
4. D's full 2372-step loss curve is not captured in one file — `arm_finetunes/D_seed_official_noaug/
   summary.json`'s `loss.values` list reflects only the final 1372-step leg (the multi-launch-resume
   history means the first two legs' per-step losses were never persisted to a combined file). The
   final checkpoint itself is unaffected (verified via `round_trip_state_sha256_match=true` and the
   proven bit-exact resume-continuity test) — this is a reporting/bookkeeping gap only.
5. LoSO fold semantics unchanged from every prior lane: `fold_count=20` treats each of the 20 clips as
   its own fold, not the 6 grouped sources — pre-existing harness behavior, not this lane's to fix,
   flagged again for the manager's standing ruling request (open since w7_ballretrain).
6. 1750-corpus re-score never dispatched (see §6) — the corpus + videos are staged on nothing now
   (VM deleted); a future lane will need to re-transfer (14MB, cheap) or re-cut a snapshot with it
   baked in.
7. No best_stack.json changes made this lane (none touched); no promotion claimed; VERIFIED=0 stands.

## 8. VM LIFECYCLE + COST

- `pickleball-h100-w7ballc`, a3-highgpu-1g (H100-80GB) SPOT, asia-southeast1-b — first create attempt
  succeeded (no quota fallback needed), from snapshot `pickleball-fleet-snap-20260708-w6close`,
  pd-balanced 200GB, labels `fable-lane=w7-ballc,fable-fleet=pickleball`.
- created_at (request): 2026-07-09T09:41:36Z; RUNNING confirmed: 2026-07-09T09:44:56Z.
- teardown requested: 2026-07-09T12:38:47Z; DELETE confirmed: 2026-07-09T12:39:49Z.
- Uptime (request-to-request): ~2.953h. Zero preemptions.
- Cost: $0.57-4.25/hr (same SKU/zone rate as w7_ballretrain) x 2.953h = **$1.68-$12.55** (mid-band
  ~$3-6) — within the $2-15 expected band, well under the 4.5h wall cap (used 2.95h of it, the
  concurrency mistake ate the buffer that would have funded ARM B + a partial 1750 attempt).
- In-VM idle watchdog (60-min no-heartbeat self-stop) armed at session start, heartbeat touched every
  SSH round-trip; never triggered.
- Fleet reconcile after teardown: `gcloud compute instances list --filter=labels.fable-fleet=pickleball`
  = only `pickleball-a100-fleet1` (TERMINATED, pre-existing, not this lane's) — zero running VMs,
  list-confirmed (`fleet_list_after_delete.log`).

## 9. NEXT

1. **Resume and finish ARM B** (raw-WASB-base+aug, currently at step 500/2372) in a follow-up lane:
   `--resume-checkpoint arm_finetunes/B_raw_wasb_aug/checkpoints/checkpoint_step_000500_INCOMPLETE.pt
   --steps 1872`, then score it on the 1121-corpus to complete the true 5-row card. Budget generously
   given this lane's unexplained ~3x training-throughput variance (assume worst-case ~0.30 steps/s,
   i.e. ~104 min, not the 0.85+ steps/s baseline, until a fresh VM proves otherwise) — and do NOT
   run it concurrently with anything else on the same GPU (see HONEST ISSUES #1).
2. **Run the 1750-corpus GPU_RESCORE_COMMANDS.sh block**, control row first (per its own instruction),
   with `W7_EXTRA_CANDIDATES` set to A (transfer from `w7_ballretrain_20260709/vm_pull/arm3_finetunes/
   A_seed_official_aug/checkpoints/latest.pt`), C (same lane, `C_stage1_official_aug`), D (this lane's
   `arm_finetunes/D_seed_official_noaug/checkpoints/latest.pt`, already md5-verified), and B once
   finished per item 1. This is the actual 1750-row re-score the mission wanted and this lane could
   not reach.
3. Manager ruling still open (carried from w7_ballretrain): does `ball_loso_validation.py` need a
   source-grouped LoSO mode before the next label-count gate (3k/6k/10k)?
4. Manager/owner FYI: this lane's aug-attribution finding (§5) is a real, clean, decision-relevant
   result — occlusion aug is a net corrective for label-driven hFP regression, not its cause — worth
   folding into whatever ruling decides the standing training recipe for the next checkpoint gate.
5. Investigate the unexplained ~3x training-throughput slowdown (HONEST ISSUES #1) if it recurs on a
   future VM of the same SKU/zone — if it's a systemic (not one-off-host) issue, the wall-cap budget
   formula (`min(12000, floor(45*60*rate))`) used by w7_ballretrain may be silently optimistic for
   future lanes and should be re-probed fresh each time rather than assumed from a prior lane's rate.

## ARTIFACTS

- `runs/lanes/w7_ballcomplete_20260709/vm_contract_tests.log` (8 passed, md5-verified)
- `runs/lanes/w7_ballcomplete_20260709/mac_key_script_md5s_precheck.txt` (code-identity baseline, 7/7 match)
- `runs/lanes/w7_ballcomplete_20260709/arm_finetunes/D_seed_official_noaug/{summary.json,checkpoints/latest.pt}`
- `runs/lanes/w7_ballcomplete_20260709/arm_finetunes/B_raw_wasb_aug/checkpoints/checkpoint_step_000500_INCOMPLETE.pt` (NOT comparable, resume-only)
- `runs/lanes/w7_ballcomplete_20260709/gpu_rescore_1121/loso_D_only/{loso_report.json,loso_report.md}`
- `runs/lanes/w7_ballcomplete_20260709/logs/*.log` (full stdout/stderr of every training/scoring launch, including the concurrency-mistake legs)
- `runs/lanes/w7_ballcomplete_20260709/md5_manifest.txt` (transfer-integrity manifest)
- `runs/lanes/w7_ballcomplete_20260709/{vm_create_start.txt,vm_create_end.txt,vm_ip.txt,vm_teardown_start.txt,vm_teardown_end.txt,vm_delete.log,fleet_list_after_delete.log,h100_create_attempt_1_asia-southeast1-b.log}`
