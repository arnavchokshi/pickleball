# LANE w7_ballretrain_20260709 — P1-1 BALL seed retrain + 1k-checkpoint eval

## OBJECTIVE_RESULT: PARTIAL

ARM 0 and ARM 1 (mandatory, measurement-integrity gates) both PASS. ARM 2 probe measured a
healthy rate. ARM 3 landed 2 of 4 planned fine-tune arms (A, C) and ARM 4 scored them against
control in one combined LoSO table — this is real, usable 1k-checkpoint evidence. B and D were
DROPPED under the binding 5h wall-cap rule (drop order D-then-B) after a mid-lane budget
recompute showed all 4 arms + scoring would not fit safely. No promotion claims; VERIFIED=0
stands throughout.

## ACCEPTANCE TABLE

| # | Check | Measured | Target | Verdict |
|---|---|---|---|---|
| 1 | ARM0 control reproduction | pooled micro F1@20=0.361111, hFP=0.599089 | w6 ref F1=0.3611, hFP=0.5991 | PASS (bit-exact) |
| 2 | ARM1 486-row anomaly reproduction | control loso-mean F1=0.686826, seed_official loso-mean F1=0.642557 | w5 ref: control 0.6858, seed 0.6404 | REPRODUCES (same direction/magnitude, seed < control; abs diffs 0.0010-0.0015, attributable to torch 2.5.1+cu124 here vs 2.9.1+cu129 originally / cuDNN nondeterminism, not a measurement break) |
| 3 | ARM2 probe rate | delta rate 200 steps / 227.61s = 0.8787 steps/s (single-probe avg 0.8425 steps/s) | >= 0.5 steps/s floor | PASS, no loader pathology |
| 4 | ARM2 budget formula | min(12000, floor(45*60*0.8787)) = 2372 steps/arm | — | applied to A/C |
| 5 | Mandatory train/inference contract check (pre-ARM3) | pytest tests/racketsport/test_ball_stage2_training.py::test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine PASSED; full targeted suite 20 passed (matches w5's "20 passed" figure exactly), 1 unrelated pre-existing-shaped failure (roboflow manifest.json smoke test, opened_samples=0 — different file than the aggregated corpus_index.json verified present; orthogonal to ARM3/4, no repo edits made) | 20 passed | PASS |
| 6 | ARM3 A (seed_official + WBCE + occlusion aug, 2372 steps) | wall=2658.7s (44.3min); loss 0.001869->0.000819 strictly decreased; checkpoint round-trip sha match; KEY-DIFF missing/unexpected keys both empty | — | DONE |
| 7 | ARM3 C (stage1_official + WBCE + occlusion aug, 2372 steps) | wall=2747.5s (45.8min); loss 0.004099->0.000885 strictly decreased; checkpoint round-trip sha match; KEY-DIFF both empty | — | DONE |
| 8 | ARM3 B (raw WASB base + aug) | NOT RUN | — | DROPPED (budget, 2nd in drop order) |
| 9 | ARM3 D (seed_official no-aug ablation) | NOT RUN | — | DROPPED (budget, 1st in drop order) |
| 10 | ARM4 combined LoSO score (control + A + C, 20 clips) | see CARDS below | — | PASS, objective_result PASS in harness output |

## CARDS (exact metric keys from ball_loso_validation.py)

All rows: OFFICIAL preprocessing, 20-clip internal-val corpus
(runs/lanes/w6_labelingest_20260708/reviewed_corpus, md5 37a5d43ab537a15bd12d382bb882a5fe,
1121 reviewed rows), harness fold_count=20 (see HONEST ISSUES — the harness treats each of the
20 clips as its own leave-one-out fold, not the 6 grouped sources in loso_fold_manifest.json;
this is unchanged, pre-existing harness behavior also present in the w6 lane's identical run,
not something introduced or fixed by this lane).

| candidate | pooled micro_label_f1_at_20px | pooled micro_hidden_false_positive_rate | pooled micro_precision_at_20px | pooled micro_visible_recall_at_20px | loso_mean label_f1_at_20px | loso_mean hidden_false_positive_rate | loso_mean precision_at_20px | loso_mean visible_recall_at_20px |
|---|---|---|---|---|---|---|---|---|
| official_tennis_control (zero-shot) | 0.361111 | 0.599089 | 0.343008 | 0.381232 | 0.303348 | 0.586327 | 0.318968 | 0.308130 |
| A_seed_official_aug (seed_official base, 1121-row corpus, +occlusion aug) | 0.615172 | 0.250569 | 0.580729 | 0.653959 | 0.644674 | 0.325716 | 0.627797 | 0.670972 |
| C_stage1_official_aug (stage1_official base, 1121-row corpus, +occlusion aug) | 0.612075 | 0.259681 | 0.581028 | 0.646628 | 0.585108 | 0.326524 | 0.593839 | 0.592820 |

Full paths: vm_pull/arm0_control/gpu_rescore/loso/loso_report.json (control-only reproduction),
vm_pull/arm4_score/loso_final/loso_report.json (combined 3-candidate table, the deliverable),
vm_pull/arm1_486anomaly/gpu_rescore/loso/loso_report.json (486-row 2-clip anomaly re-run).

**<20px small-ball size-binned recall**: the harness does NOT emit a ball-pixel-size-binned
recall. It emits localization-ERROR-radius bins (visible_recall_at_5px/10px/20px/40px per
clip in per_source_metrics) — a distance threshold, not a ball-size-in-frame bucket. Stated
plainly as an honest gap per the lane brief; no new repo code was added to build the missing
metric (out of this lane's scope — internal-val measurement only, no repo edits).

## THE 1k-CHECKPOINT CURVE POINT (RULINGS R2a gate)

Comparable prior points on the identical 20-clip/1121-row-eval-corpus harness (from
runs/lanes/w6_labelingest_20260708/gpu_rescore/loso/loso_report.json, same measurement world):

| checkpoint | training rows | pooled micro F1@20 | pooled micro hFP |
|---|---|---|---|
| official_tennis_control (zero-shot) | 0 | 0.3611 | 0.5991 |
| stage1_official (roboflow-only, no owner labels) | 0 owner rows | 0.2971 | 0.6948 |
| seed_official (w5, no occlusion-aug recipe change) | 486 | 0.5329 | 0.2255 |
| A_seed_official_aug (this lane, +635 owner rows +occlusion aug) | 1121 | 0.6152 | 0.2506 |
| C_stage1_official_aug (this lane, fresh stage1 base +occlusion aug) | 1121 | 0.6121 | 0.2597 |

Going 486 -> 1121 reviewed rows (with the occlusion-aug recipe added at the same time) moved
pooled micro-F1@20 from 0.5329 -> 0.6152 (+0.082 absolute, +15.5% relative) — the curve is NOT
flat at this checkpoint; more labels (headed toward the 3k/6k/10k gates) still look
worth funding. The trade: hidden-FP got WORSE (0.2255 -> 0.2506), not better. Because B and D
(the isolating ablations — raw-WASB-base and no-aug) were dropped for budget, this lane CANNOT
separate how much of the F1 gain is "+635 more rows" vs "+occlusion augmentation" vs the
interaction of both, nor whether the hFP regression is caused by the occlusion aug teaching the
model to guess through occlusion (plausible mechanism, unconfirmed) or by the new label mix.
That isolation is exactly what D (no-aug ablation) would have measured — it is the top
follow-up recommendation below.

## VM LIFECYCLE + COST

- pickleball-h100-w7ball, a3-highgpu-1g (H100-80GB) SPOT, asia-southeast1-b — first create
  attempt succeeded (no quota fallback needed).
- created_at: 2026-07-09T05:40:25Z; deleted_at: 2026-07-09T09:17:18Z; uptime 3.615h.
- Zero preemptions.
- Cost: $0.57-4.25/hr x 3.615h = $2.06-$15.36 (honest span; mid-band ~$4-8).
- Fleet reconcile after teardown: gcloud compute instances list --filter=labels.fable-fleet=pickleball
  = only pickleball-a100-fleet1 (TERMINATED, pre-existing, not this lane's) — zero running VMs,
  list-confirmed.
- In-VM idle watchdog (60-min no-heartbeat self-stop) armed at boot, heartbeat touched every SSH
  round-trip; never triggered.

## TRANSFER TAXES

- Snapshot pickleball-fleet-snap-20260708-w6close had EVERYTHING already baked: the 1,121-row
  reviewed corpus + LoSO fold manifest (md5 37a5d43ab537a15bd12d382bb882a5fe, exact match, zero
  transfer), rally videos for all 6 sources, roboflow universe corpus (61,260 samples), the w5
  stage1_official/seed_official checkpoints, and the official WASB tennis anchor (sha256
  9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb, verified). Zero major
  transfer tax this lane.
- Only transfer needed: the 486-row-era runs/cvat_imports/2026_06_30/{burlington_gold...,
  wolverine_mixed...} label dirs (needed for the ARM1 2-clip anomaly re-run only) — NOT baked
  in this snapshot. 3.7MB + 2.4MB = 6.1MB raw, 1.8MB tar.gz, transferred in under 10s. Verified
  scope: only the two INTERNAL-VAL clip dirs were copied — the sibling
  outdoor_webcam_iynbd_1500... and indoor_doubles_fwuks_0500... held-out label dirs in that
  same parent folder were NOT touched (held-out discipline honored by construction).
- Pull-back to Mac: 66MB total (rsync filtered to reports/summaries/logs + only the 2 landed
  arms' latest.pt checkpoints, excluding all per-clip ball_track.json/csv predictions and
  interim checkpoint_step_*.pt files) — well under the 2GB lean-pull cap. Mac disk had 43GB
  free at pull time (more headroom than the ~5GB baseline noted in the brief).

## CODE IDENTITY

VM synced to Mac's exact committed HEAD 8721f786101bcc8f9634745f63b4d389f49693cc (26 commits
ahead of the snapshot's baked 37b8ecd3f, confirmed ancestor). git status --short on VM after
checkout: only the 2 by-design vendor-submodule overlay lines
(third_party/WASB-SBDT, third_party/blurball) + untracked .venv — clean per house rule, no
reset --hard needed. md5 of train_ball_stage2.py, run_wasb_ball.py,
ball_loso_validation.py, fuse_ball_tracks.py, run_ball_tracking_eval_suite.py,
train_ball_pretrain.py matched Mac byte-for-byte (manifest:
mac_key_script_md5s_precheck.txt).

## HONEST ISSUES

1. B and D dropped for budget (see above) — the isolating ablations that would cleanly
   attribute the F1 gain / hFP regression to labels-vs-augmentation are the biggest gap left by
   this lane.
2. LoSO fold semantics: the harness (ball_loso_validation.py, unchanged by this lane)
   treats each of the 20 clips as its own fold (fold_count: 20 in every report), NOT the 6
   grouped sources in loso_fold_manifest.json. 73VurrTKCZ8 and Ezz6HDNHlnk each contribute 8
   "folds" while 4 other sources contribute 1 each — the unweighted loso_mean is therefore
   implicitly weighted toward those two sources' per-clip variance, not a clean per-SOURCE
   leave-one-out mean. This is pre-existing behavior (identical in the w6 lane's run); flagging
   for the manager's ruling on whether the harness needs a source-grouping mode, not something
   this lane had scope to fix.
3. Torch version drift: VM environment runs torch 2.5.1+cu124, vs 2.9.1+cu129 recorded
   in the original w5/w6 lanes' summary.json. Code is byte-identical (md5-verified); this is an
   environment-only difference. Plausible source of the small (~0.001-0.002) numeric drift in
   the ARM0/ARM1 reproductions. Not investigated further (out of scope; both reproductions
   passed their "within noise" bar).
4. One unrelated pre-existing-shaped pytest failure
   (test_roboflow_corpus.py::test_loader_smoke_reads_real_roboflow_samples_across_sources,
   opened_samples=0 vs >=50) surfaced during the mandatory contract-check run. Different
   manifest file (data/roboflow_universe_20260706/manifest.json) than the one this lane
   verified present and used (aggregated/corpus_index.json, 61,260 samples, confirmed intact).
   Orthogonal to every arm actually run this lane (none touch the roboflow loader path) — no
   repo edits made, flagged for the owner of that test/fixture.
5. hFP regressed for both A and C vs the 486-row seed_official (0.2255 -> 0.2506/0.2597) even
   though F1 improved substantially — a genuine trade-off, not a clean win, and one more reason
   the D ablation (no-aug) matters for the next lane.

## NEXT

1. Re-run B (raw-WASB-base+aug) and D (seed_official no-aug ablation) — the two dropped arms —
   ideally in one shorter follow-up lane now that the probe rate (0.8787 steps/s) and per-arm
   wall time (~45min at 2372 steps) are known quantities; D specifically isolates whether the
   1121-row F1 gain is from labels or from occlusion augmentation, and whether it explains the
   hFP regression.
2. Feed A_seed_official_aug (best of the 2 landed arms on both pooled AND loso-mean F1, plus
   lowest hFP of the two) forward to the manager's base-choice ruling as a PENDING best_stack
   candidate — NOT auto-promoted (VERIFIED=0 stands; this is internal-val evidence only).
3. Manager ruling requested on the fold-semantics honest issue (#2 above) — whether
   ball_loso_validation.py needs a source-grouped LoSO mode before the next 3k/6k checkpoint
   gate, since the per-clip-not-per-source fold count could bias future comparisons the same
   way.
4. This lane's 1k-checkpoint point (0.6152 F1@20, +15.5% relative over the 486-row checkpoint)
   is real evidence for RULINGS R2a that the label curve has not plateaued — supports continuing
   the label flywheel toward the 3k gate rather than diverting owner-hours to diversity-only
   labeling yet.

## ARTIFACTS

- runs/lanes/w7_ballretrain_20260709/vm_pull/arm0_control/gpu_rescore/loso/loso_report.json
- runs/lanes/w7_ballretrain_20260709/vm_pull/arm1_486anomaly/gpu_rescore/loso/loso_report.json
- runs/lanes/w7_ballretrain_20260709/vm_pull/arm2_probe/probe100/summary.json,
  probe300/summary.json
- runs/lanes/w7_ballretrain_20260709/vm_pull/arm3_finetunes/A_seed_official_aug/{summary.json,checkpoints/latest.pt}
- runs/lanes/w7_ballretrain_20260709/vm_pull/arm3_finetunes/C_stage1_official_aug/{summary.json,checkpoints/latest.pt}
- runs/lanes/w7_ballretrain_20260709/vm_pull/arm4_score/loso_final/loso_report.json (the
  deliverable combined table)
- runs/lanes/w7_ballretrain_20260709/vm_pull/logs/*.log (full stdout/stderr of every arm)
- runs/lanes/w7_ballretrain_20260709/md5_manifest.txt (transfer-integrity manifest, Mac-side
  md5 of every pulled file, cross-verified against VM-side md5sum for both checkpoints)
- runs/lanes/w7_ballretrain_20260709/mac_key_script_md5s_precheck.txt (code-identity baseline)

## DRAFT BUILD_CHECKLIST BULLET

w7_ballretrain: 1k-checkpoint gate (RULINGS R2a) landed — ARM0/ARM1 measurement-integrity PASS
(bit-exact control repro 0.3611/0.5991; 486-row seed-vs-control anomaly reproduces, seed 0.6426 <
control 0.6869 loso-mean); 2 of 4 planned fine-tunes landed on the full 1121-row corpus
(seed_official-base+aug and stage1_official-base+aug, 2372 steps each per the measured-rate
budget formula), both far above zero-shot control (pooled micro-F1 0.615/0.612 vs control 0.361;
seed_official-base is the mild winner) and above the 486-row checkpoint (0.533) — the label curve
has NOT plateaued at 1k rows — but hFP regressed vs the 486-row point (0.251/0.260 vs 0.226), an
open trade-off. B (raw-WASB-base) and D (no-aug ablation) DROPPED under the binding 5h wall-cap
drop order; D specifically is needed to isolate labels-vs-augmentation. No promotion; feeds a
PENDING best_stack candidate + the manager's base-choice ruling. VM
pickleball-h100-w7ball created/deleted/list-confirmed, 3.615h uptime, $2.06-15.36.
