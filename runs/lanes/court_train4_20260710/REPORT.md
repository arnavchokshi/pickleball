# COURT-TRAIN-4 — court_unet_v2 real-transfer arms (the literal R2/R8 recipe, finally run)

## Objective result: BOTH ARMS FAIL the R2 kill bar (honest negative, decisive, per HONEST-OUTCOME RULE)

The literal ARM-A ("init from court_model_v2.pt") and ARM-B ("ImageNet resnet34 encoder init")
recipes from `court_wave_20260709/DESIGN_RULING.md` R2/R8 — which TRAIN-1 proved impossible on the
legacy trainer and TRAIN-3 could never reach a GPU to attempt — were **both fully trained and fully
evaluated** this lane, on the exact pinned `train_court_model_v2.py` bridge, on an H100. Both fail
the kill bar (CARD-A pooled median <25px AND PCK@5 >= +0.30 absolute over control) **by three
orders of magnitude on the median leg and the full +0.30 on the PCK leg**:

| arm | CARD-A median | CARD-A PCK@5 | delta vs control | median bar | PCK bar | verdict |
|---|---:|---:|---:|---|---|---|
| control (frozen court_model_v2.pt) | 942.34px | 0.0 | — | — | — | (baseline) |
| ARM-A (court_model_v2.pt init) | 936.63px | 0.0 | +0.0 | FAIL (>>25px) | FAIL | **FAIL** |
| ARM-B (ImageNet resnet34 init) | 833.37px | 0.0 | +0.0 | FAIL (>>25px) | FAIL | **FAIL** |

**The decisive scientific finding — this kills the recipe far more conclusively than TRAIN-1 could:**
both arms *demonstrably learned the real corpus* (trainer-internal 8-row real holdout, drawn from
the train-family datasets: ARM-A median **2.80px** / PCK@5 **0.758**, ARM-B median **2.73px** /
PCK@5 **0.758**; real-probe loss ARM-A 7.46->3.98, ARM-B 9.71->4.08; 1151 real batches each; zero
null-channel supervision violations) — and yet transfer to **nothing** outside the training source
families:

- NOT to the source-disjoint Roboflow val family (xuann/testworkspace, 13,056 keypoints):
  ARM-A 293.47px / PCK@5 0.0; ARM-B 290.53px / 0.0 (control: 345.39px / 0.0). A ~55px median
  improvement and nothing else.
- NOT to the owner harvest GT (CARD-A): 936.63 / 833.37px, per-clip PCK@5 uniformly 0.0.
- NOT to Burlington/Wolverine (CARD-B): ARM-A 714.56px, ARM-B 797.30px, PCK@5 0.0 everywhere.

Contrast with TRAIN-1's legacy tiny-CNN arms, which reached **13.5px median / PCK@5 0.16-0.21** on
the same disjoint val family with the same corpus: the 21M-parameter resnet34 U-Net at this step
budget memorizes its training sources and generalizes cross-source dramatically *worse* than the
tiny CNN, while mastering in-family data. The bottleneck is now proven to be **data diversity, not
architecture, not initialization lineage** — neither synthetic-pretrained nor ImageNet-clean init
rescues cross-source transfer on a 7-train-source corpus. R2's question is answered: the recipe
(not the program) is dead as specified; the next lever is more diverse real sources (owner O1 ask /
R4 architecture wave), not more training rungs on this corpus.

VERIFIED=0 unchanged. No promotion claim. No best-stack delta.

## Provisioning (the lane's raison d'etre after TRAIN-3's 52-attempt stockout)

Widened multi-SKU/multi-region ladder per the MANAGER RULING POST-STOCKOUT (spec §), 120s
inter-attempt backoff, 2h cap: **succeeded on attempt 7 of the first H100 pass** —
`pickleball-gpu-court3`, a3-highgpu-1g SPOT, **europe-west4-b** (the 7th and final H100 rung),
created 2026-07-10T17:03:27Z from `pickleball-fleet-snap-20260709-w7close` (cross-region global
snapshot clone worked; ~7min create). Attempts 1-6 (ase1-b/-c, us-central1-a/-b, us-east4-a/-c)
were all genuine `ZONE_RESOURCE_POOL_EXHAUSTED`/STOCKOUT. **Zero snapshot-clone rate-throttle
errors** (TRAIN-3's novel failure mode, ~20/52 attempts at 15-55s spacing) — the 120s backoff
ruling is validated. Logs: `create_attempt_{1..7}_*.log`, `create_loop.log`.

## Sync + integrity (all verified before any training)

- VM HEAD = `3cfede00b91f382fdcf36b62e9792f3940c805d4` (exact pin; git fetch+reset).
- 3 pinned files md5 Mac == VM, 3/3 exact: `train_court_model_v2.py` `96453dff...`,
  `court_keypoint_net.py` `5b467806...`, `evaluate_court_keypoint_owner_gate.py` `06be3f76...`.
- `court_model_v2.pt` sha256 `cdf0555d...` on-snapshot, matches best-stack pin.
- `resnet34-b627a593.pth` rsynced from Mac (still absent from w7close snapshot, as TRAIN-1 found);
  VM sha256 `b627a593...` matches the HANDOFF pin exactly.
- CARD-A GT root `w7_courtkpingest_20260709/gt_roots/corrected_r2` also absent from snapshot;
  rsynced from Mac (13MB, 5 frames / 3 harvest clips).
- **Corpus rebuilt ON the VM** via committed `build_real_court_corpus.py` (git-index guard,
  `--partial-rows`): **3,921 rows / 9 datasets, histogram {12: 3478, 14: 408, 15: 35}**,
  rows_by_dataset and 7/2 split == `court_data2b_20260709/STATS_REPORT.md` exactly.

## Control reproduction (gate before training)

Frozen `court_model_v2.pt` scored via the same `evaluate_court_keypoint_owner_gate.py`:

| card | banked (TRAIN-1) | this lane | delta |
|---|---:|---:|---:|
| CARD-A raw_independent median | 942.34px | 942.3439px | **0.004px** |
| CARD-B raw_independent median | 675.15px | 675.1467px | **0.003px** |

Both under the 1px STOP threshold — the eval bridge is deterministic across VMs/regions. PCK@5 0.0
both cards, matching banked rows.

## Probe + a live trainer/env bug (booked)

**BUG (repo-vs-fleet-venv):** `train_court_model_v2.py:560` passes `DataLoader(in_order=True)`
whenever `--synthetic-workers > 0` (the default resolves to 8). `in_order` does not exist in the
fleet venv's pinned `torch==2.5.1+cu124` — training crashes immediately
(`DataLoader.__init__() got an unexpected keyword argument 'in_order'`). Same class of defect as
the coordinates.py StrEnum/py3.10 incident. Worked around with the documented `--synthetic-workers 0`
CLI flag (no code change; single-process synthetic generation). Consequence: synthetic sample
generation is CPU-bound and dominates step time — at batch 256 the H100 sat ~0% GPU-util at
8.87s/step. Re-probed at batch 32 (VRAM was never the constraint): **100 steps in 165.66s =
0.6036 steps/s**. `step_budget = min(6000, floor(50*60*0.6036)) = 1810` -> trained both arms at
**1800 steps (18 epochs x 100 steps/epoch), batch 32 synthetic / 32 real, eval-every-2**.
Fix-forward for whoever owns the trainer: gate the kwarg on torch version (or bump fleet torch);
with workers restored the same budget formula would give the full 6,000 steps on this SKU.

## Arms (both to full 1800-step budget, sequential, never co-scheduled)

Both: `--real-root` = VM-rebuilt corpus, `--real-split-proposal` (7 train datasets only; xuann/
testworkspace never trained), `--real-weight 0.65 --synthetic-weight 0.35` (real-batch probability
0.65, trainer-confirmed 1151/1800 real batches), `--real-photometric-aug`, AMP, identical seeds.

- **ARM-A**: `--init-from-checkpoint court_model_v2.pt`. Summary reports
  `initialization.mode=model_checkpoint_fresh_optimizer`, `start_epoch=0` — the exact HANDOFF
  contract. Wall ~52min. train 17:36->18:28Z.
- **ARM-B**: `--encoder-weights-path resnet34-b627a593.pth` (no init/resume). Summary reports
  `initialization.mode=encoder_weights`, encoder path echoed. Wall ~52min. train 18:34->19:24Z.

## Full eval battery (every row also run with --enable-homography-refinement)

raw_independent (primary; pooled over keypoints):

| arm | card | median | p95 | PCK@5 | PCK@10 | homography variant |
|---|---|---:|---:|---:|---:|---|
| control | CARD-A | 942.34px | 1484.75px | 0.0 | 0.0 | identical (exact no-op) |
| control | CARD-B | 675.15px | 1432.20px | 0.0 | 0.0 | cardb raw_all median 705.13->718.54 (worse) |
| ARM-A | CARD-A | 936.63px | 1570.76px | 0.0 | 0.0 | identical |
| ARM-A | CARD-B | 714.56px | 1481.29px | 0.0 | 0.0 | identical |
| ARM-B | CARD-A | 833.37px | 1551.68px | 0.0 | 0.0 | median 900.57px (worse) |
| ARM-B | CARD-B | 797.30px | 1698.92px | 0.0 | 0.0 | identical |
| control | VAL-ext | 345.39px | 583.04px | 0.0 | n/m | 349.76px |
| ARM-A | VAL-ext | 293.47px | 590.96px | 0.0 | n/m | 295.94px |
| ARM-B | VAL-ext | 290.53px | 590.71px | 0.0 | n/m | 294.66px |

(CARD PCK@10 values measured 0.0 via the floor-12/net-3 split scorer, both subsets 0.0 at 10px;
VAL-ext PCK@10 not separately measured — see honest issues. Aggregated_independent modes also all
0.0 PCK@5: ARM-A CARD-A 891.75px, ARM-B CARD-A 860.54px, ARM-A CARD-B 680.21px, ARM-B CARD-B
815.98px.)

Floor-12-only honesty split (net channels excluded; the 12 channels 89% of corpus rows supervise):

| arm | card | floor-12 median | floor-12 PCK@5/@10 | net-3 median |
|---|---|---:|---|---:|
| control | CARD-A | 944.38px | 0.0 / 0.0 | 921.35px |
| ARM-A | CARD-A | 939.49px | 0.0 / 0.0 | 905.31px |
| ARM-A | CARD-B | 731.80px | 0.0 / 0.0 | 686.98px |
| ARM-B | CARD-A | 883.81px | 0.0 / 0.0 | 661.82px |
| ARM-B | CARD-B | 899.56px | 0.0 / 0.0 | 814.65px |

The failure is NOT a masked-net artifact — floor-12-only is exactly as bad.

Per-source (raw_all medians, px): CARD-A 73V/HyU/zwC — control 999.7/981.2/768.0; ARM-A
880.6/1061.9/901.6; ARM-B 823.5/829.5/887.1. CARD-B burl/wolv — control 812.8/675.2; ARM-A
812.8/698.2; ARM-B 1098.6/734.8. VAL-ext per-dataset: ARM-A xuann 289-293px, testworkspace
313-374px (control 332-371px). PCK@5 is 0.0 for every source, every arm, every mode.

Trainer-internal real holdout (8 rows, train-family — NOT source-disjoint, NOT promotion evidence):
ARM-A median 2.80px / PCK@5 0.758 / PCK@10 0.908; ARM-B 2.73px / 0.758 / 0.867. This is the
in-family vs cross-source contrast that makes the negative decisive.

## Overlays

31 prediction-vs-GT overlays pulled to `vm_pull/overlays/` (red hollow = GT, green filled = pred):
ARM-A CARD-A 5 (all 5 existing GT rows — the card only has 5 frames), ARM-A CARD-B 10,
ARM-B CARD-A 5 (same 5 rows) + ARM-B CARD-B 10, plus per-dir `overlay_manifest.json` (x4, = 34
files). Visual story matches the numbers: predictions cluster in plausible court-shaped
configurations that are globally misplaced/mis-scaled vs the actual camera view.

## Teardown + cost

- VM created 17:03:27Z, RUNNING confirmed 17:03:47Z, deleted 19:29:35Z, delete rc=0,
  list-confirmed absent (`fleet_list_after_delete.log`: only pre-existing unrelated
  `pickleball-a100-fleet1` TERMINATED remains); disk-orphan check `name~court3` = 0 items.
- Uptime **2.430h**, inside the 3h wall cap. H100 SPOT rate band $0.57-4.25/hr ->
  **$1.39-$10.33** (mid ~$3-5), inside the ~$2-13 budget, under the $20 ceiling. Zero preemptions.
- Provisioning span 16:35:10Z -> 17:03:27Z (28min, 7 attempts, $0 while failing).

## Honest issues

1. **Kill bar FAIL is the finding.** Both literal R2/R8 arms fail at 833-937px CARD-A median vs
   the 25px bar, PCK@5 delta +0.0 vs the +0.30 bar. Uniform per-clip, per-source, both eval modes,
   raw and homography-refined, floor-12-only included. This is a clean negative for the
   "fine-tune court_unet_v2 on the 3,921-row Roboflow partial corpus" recipe at both inits.
2. **in_order/torch-2.5.1 trainer bug** (above) — repo trainer's default configuration cannot run
   on the current fleet venv at all; booked for the trainer owner. The `--synthetic-workers 0`
   workaround cut the realized step budget to 1800 (vs 6000 cap): a re-run with a fixed loader
   would get 3.3x the steps. Given both arms sit >800px from a 25px bar with in-family learning
   already saturated (2.8px), more steps on the same corpus are extremely unlikely to close a
   30x cross-source gap — but it is honest to note the budget was throughput-limited, not
   convergence-verified.
3. VAL-ext PCK@10 was not separately measured (gate evaluator reports one threshold; the VM was
   torn down inside the wall cap before a second @10 pass on 13k keypoints). Cards' PCK@10 = 0.0
   measured. At 290px median the val-ext PCK@10 cannot be materially above zero.
4. ARM-A burlington raw_all per-clip median is bit-identical to control's (812.8309374145589)
   while means/maxes differ — coincidence of argmax-degenerate OOD predictions snapping to the
   same peak on one row, not an eval-plumbing bug (row-level medians differ on other rows).
5. Homography refinement remains a numerically exact no-op in most runs (insufficient confident
   peaks at garbage-prediction level), and where it did engage (ARM-B CARD-A, control/ARM-A
   valext, control cardb raw_all) it made medians slightly WORSE. Consistent with TRAIN-1's flag.
6. Overlay count is 5/card for CARD-A by GT-row supply (only 5 corrected_r2 rows exist), 10
   elsewhere — TRAIN-1 hit the same limit.
7. `court_external/` and `w7_courtkpingest` GT roots are still missing from the w7close snapshot
   (rsynced again this lane); fold into the next snapshot cut with TRAIN-1's flag.
8. ARM-A/ARM-B evals ran on GPU sequentially after their trainings; ARM-A CPU-side overlays/floor12
   ran during ARM-B GPU training (CPU-only, no GPU co-scheduling of arms).

## Best-stack delta

**None.** Neither arm cleared the bar; per spec "(b) at most" and the HONEST-OUTCOME RULE no
PENDING row is added. `configs/racketsport/best_stack.json` untouched. VERIFIED=0 unchanged.

## Artifacts (lane dir; all VM pulls under `vm_pull/`, 71-file `md5_manifest.txt`)

- Provisioning: `create_loop.sh`, `create_loop.log`, `create_attempt_{1..7}_*.log`,
  `vm_create_start.txt`, `vm_create_success.txt`, `vm_running_confirmed.txt`
- Integrity: `version_stamp.json`, `control_check.json`, `vm_pull/corpus_rebuild/{corpus_stats.json,STATS_REPORT.md}`
- Control: `vm_pull/control_eval/` (6 gate JSONs + carda floor12)
- Probe: `probe_summary_final.json`, `vm_pull/probe/probe_stdout.log`
- Arms: `vm_pull/arm_{a,b}/{court_model_v2.pt,court_keypoint_metrics.json}` (md5s in manifest:
  arm_a `ecf1855aac688b86b41cec61f5da8401`, arm_b `13c0c0acc24e102bfd62c2d5638c3164`),
  `vm_pull/logs/arm_{a,b}_train_stdout.log`
- Evals: `vm_pull/arm_{a,b}_eval/` (6 gate JSONs + 2 floor12 each), `all_rows_summary.json`,
  `arm_a_per_source.txt`, `arm_b_summary.txt`, `kill_bar_arm_a.json`
- Overlays: `vm_pull/overlays/arm_{a,b}/{carda,cardb}/` (30 JPEGs + 4 manifests)
- Teardown: `vm_delete.log`, `vm_delete_{start,end}.txt`, `fleet_list_after_delete.log`
- Lane infra: `scripts/` (VM-side pipeline scripts incl. `floor12_score.py`, `render_overlays.py`)
