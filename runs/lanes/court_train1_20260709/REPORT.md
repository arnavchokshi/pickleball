# COURT-TRAIN-1 — real-transfer probe ladder on H100 (design ruling R2/R8)

## Objective result: FAIL (honest negative, per HONEST-OUTCOME RULE)

The R2 kill bar (CARD-A pooled median <25px AND PCK@5 >= +0.30 absolute over the frozen
control) FAILS decisively for both trained arms. A load-bearing architecture/CLI gap was
discovered live: **the literal "ARM-A init from court_model_v2.pt" / "ARM-B init ImageNet
resnet34" recipes named in `court_wave_20260709/DESIGN_RULING.md` R2/R8 are NOT ACHIEVABLE**
with `scripts/racketsport/train_court_keypoint_heatmap.py` as it exists at the pinned commit.
This is a genuine, code-verified repo gap (below), not a workaround-able CLI oversight. Given
that, the lane ran the closest honest, achievable substitutes on the real recipe (masked-null
Roboflow corpus + synthetic-curriculum mixing + dataset-disjoint holdout) and reports their
real-transfer numbers plainly -- both fail the kill bar hard on the primary card. VERIFIED=0
unchanged; no promotion claim.

## ARCHITECTURE MISMATCH (the decisive finding)

`models/checkpoints/court_unet_v2/court_model_v2.pt` (sha256 `cdf0555d...`, matches the
best-stack pin exactly) is a `court_unet_v2` checkpoint: a torchvision-resnet34-backbone U-Net
with a **dict** output (`keypoint_heatmaps`, `line_family_logits`, `keypoint_vis_logits`),
input size 640x360, trained by a *different* script (`args.out` embedded in the checkpoint
reads `runs/lanes/calv1_unet_train_20260708/full_run` -- the CALV1 wave's own trainer, with
flags like `--steps-per-epoch`, `--seg-loss-weight`, `--resume` that don't exist in
`train_court_keypoint_heatmap.py`'s CLI at all).

`scripts/racketsport/train_court_keypoint_heatmap.py`'s own `run_training()` **hardcodes**
`net_architecture = "local_conv_v1" if use_line_segmentation else "encoder_decoder_v1"`
(line ~1259) -- two small custom single-tensor-output CNNs, never `court_unet_v2`. Its
`--model-architecture` CLI flag only selects between those two *logical* labels
(`keypoint_heatmap_v1`, `line_segmentation_intersection_v1`); neither reaches the
resnet34-backbone architecture. There is also **no `--init-checkpoint`/`--resume` flag of any
kind** -- the trainer always starts from a fresh random init. Consequently:

- ARM-A ("init from court_model_v2.pt") is impossible: even if a checkpoint-loading flag
  existed, `load_state_dict` would hard-fail immediately (`stem.*`/`layer1.*` resnet keys vs.
  `encoder.*`/`decoder.*` custom-CNN keys -- genuinely incompatible state dicts).
- ARM-B ("init ImageNet resnet34") is impossible via this CLI for the same reason: the only
  resnet34/ImageNet-capable path (`make_court_unet_v2_model` + `encoder_weights_path`, in
  `threed/racketsport/court_keypoint_net.py`) is never called by `run_training()`.
- `court_unet_v2`'s own module docstring is explicit that it is "a new, purpose-built
  architecture (not routed through the legacy single-tensor trainer path)" -- wiring it into
  `train_court_keypoint_heatmap.py` would mean writing a new loss/target/eval path for the
  3-head dict output, a real feature, not a small additive patch, and out of this lane's scope
  and file ownership.

**Substitute recipe run instead (both fully achievable, no code changes, documented as
deviations, not silent substitutions):**
- **ARM-A** = from-scratch (`encoder_decoder_v1`, random init) fine-tune with
  `--real-finetune-start-epoch 0` (real-supervision from step 0 -- "real-only floor transfer"
  in R2's own language, since a literal court_model_v2.pt warm start is unreachable).
- **ARM-B'** = identical recipe, `--real-finetune-start-epoch 120` (this trainer's own CLI
  default -- 120 epochs of its built-in *procedural* synthetic-only pretraining before real
  data enters, the closest in-trainer analog to "does synthetic pretraining help or hurt real
  transfer" that doesn't require ImageNet weights this trainer cannot load).

Both arms share the identical corpus, masking, synthetic-curriculum mix, and step budget, so
the ARM-A vs. ARM-B' delta is still a clean, honest ablation of "more vs. less built-in
synthetic pretraining" -- it is just not the specific "ImageNet resnet34" ablation R8 asked
for.

## P0 -- provision + sync

- GPU stockout: H100 SPOT was unavailable in **both** `asia-southeast1-b` and `-c` for
  **27 create attempts across ~65 minutes** (`h100_create_*.log`, `h100_create_batch*_*.log`).
  Attempt 27 (zone `-c`) succeeded. This is a genuine, sustained capacity shortage (error
  `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / `stockout` / `resource_availability`), not a
  config error, and cost nothing (failed creates aren't billed). The 3.5h wall cap is counted
  from actual VM RUNNING confirmation (`vm_running_confirmed.txt`), not from the first attempt.
- VM: `pickleball-h100-court1`, zone `asia-southeast1-c`, a3-highgpu-1g SPOT, from snapshot
  `pickleball-fleet-snap-20260709-w7close`, created `2026-07-10T11:27:42Z`.
- Code sync: `git fetch origin && git reset --hard 497b64dbd` -- VM HEAD =
  `497b64dbddda5d25dbbfef615a6ff2ee72810128` (exact pin). Known by-design dirt only
  (`third_party/WASB-SBDT`, `third_party/blurball` submodule overlays; `.venv` untracked).
- Version stamp (md5, Mac == VM, all 3/3 exact):
  `train_court_keypoint_heatmap.py` `ab537fe4c62e807e393815552949bed2`;
  `evaluate_court_keypoint_owner_gate.py` `06be3f76ca4a6250c9258414be6c96f6`;
  `court_keypoint_net.py` `5b4678069fa43706c12d4ba0111c330f`.
- `court_model_v2.pt` sha256 `cdf0555d49335a946e518b177d85e2ab5be02100ba46eb3e634785c84f337c22`
  -- matches the best-stack pin exactly (baked in the snapshot, verified live).
- `data/roboflow_universe_20260706` baked (67 dataset dirs); 3/3 random-sample sha256 match
  Mac exactly.
- **SNAPSHOT GAP (new)**: `models/checkpoints/court_external/` (the entire tree -- resnet34
  ImageNet checkpoint, PnLCalib, TennisCourtDetector) was **completely absent** from the
  `w7close` snapshot, not just the 2 files a prior lane (`w7_p22gate`) flagged as missing from
  the older `w6close` snapshot. `court_model_v2.pt` itself *is* present and correct on this
  snapshot (that earlier gap is fixed). rsynced `resnet34-b627a593.pth` from Mac (md5
  `78fe1097b28dbda1373a700020afeed9`, matches) and created the missing directory tree.
  Fold `court_external/` into the next snapshot cut.
- `runs/training_corpora_20260701/court_synthetic` absent on the VM (and on Mac) -- generated a
  fresh bounded corpus via `generate_synthetic_court_keypoints.py --count 800 --seed 20260710`
  (17s, 7 scenario families, `runs/lanes/court_train1_20260709/court_synthetic_v1`).
- **Corpus rebuild -- bit-identical to the Mac reference.** Ran
  `build_real_court_corpus.py --dataset-root data/roboflow_universe_20260706 --mapping-table
  runs/lanes/court_data2_20260709/keypoint_mappings.json --partial-rows
  --guard-git-index-prefix eval_clips/` (no `--reuse-guard-report`/`--guard-image-root`: could
  not reconstruct the exact 49-file harvest guard root the original court_data2b build used --
  no invocation was recorded anywhere in the repo; see honest_issues). Result: **3,921 rows / 9
  datasets**, histogram `{12: 3478, 14: 408, 15: 35}`, `rows_by_dataset` and `split_proposal`
  (7 train / 2 val datasets) **exactly** match `court_data2b_20260709/HANDOFF.md` and
  `split_proposal.json`. 6/6 spot-checked row sha256 (3 chosen on Mac, re-verified against the
  VM-rebuilt `corpus_stats.json` provenance) are exact matches. Guard: 0 matches (clean).

## P1 -- control rows (frozen `court_model_v2.pt`, BEFORE any training)

| card | frames | pooled PCK@5 | pooled PCK@10 | median (px) | p95 (px) | per-clip PCK@5 |
|---|---:|---:|---:|---:|---:|---|
| CARD-A raw | 5 | 0.0 | 0.0 | 942.34 | 1484.75 | 73VurrTKCZ8=0.0, HyUqT7zFiwk=0.0, zwCtH_i1_S4=0.0 |
| CARD-A homography | 5 | 0.0 | 0.0 | 942.34 | 1484.75 | (identical to raw) |
| CARD-B raw | 2 | 0.0 | 0.0 | 675.15 | 1432.20 | burlington=0.0, wolverine=0.0 |
| CARD-B homography | 2 | 0.0 | 0.0 | 675.15 | 1432.20 | (identical to raw) |

Matches the spec's expectation almost exactly ("~0.0 PCK / ~10^2-10^3 px"). CARD-A = 5 frames
across 3 harvest sources (`73VurrTKCZ8`, `HyUqT7zFiwk`, `zwCtH_i1_S4`); CARD-B = 2 independent
frames (Burlington + Wolverine). **Homography refinement produced numerically identical output
to raw in every single run this lane executed** (control and both arms) -- flagged as a probable
silent no-op (insufficient confident heatmap peaks to trigger refinement) rather than
investigated further; see honest_issues.

## P2 -- probe

This trainer's `run_training()` performs exactly **one `optimizer.step()` per `--epochs` loop
iteration**, so `--epochs` *is* the "step" unit the spec's probe formula refers to (confirmed
by reading the training loop, not assumed).

- Probe: 100 epochs of the real recipe (masked corpus + `court_synthetic_v1`,
  `--real-finetune-start-epoch 0`, `--real-batch-size 64`, synthetic-curriculum 0.35/0.35) --
  **77.003s wall**, steps_per_sec = 100 / 77.003 = **1.29865**.
- `step_budget = min(4000, floor(45*60 * 1.29865)) = min(4000, 3506) = 3506`.

## P3 -- training (both arms, same corpus/masking/mix/steps)

- Data mix: `--real-root real_court_corpus_partial --real-root court_synthetic_v1`, holdout =
  the 6 `testworkspace`/`xuann` clip-name variants (train/valid/test) per
  `split_proposal.json`'s 2-dataset val holdout, `--synthetic-curriculum-start/end-fraction
  0.35` (constant ~35% synthetic mix per batch, not a ramp -- closest literal reading of "R0
  proportions ~35% of batches"), `--real-batch-size 64`. **No augmentation CLI flags exist in
  this trainer at all** (full argparse read; none applied -- "standard aug flags" from the spec
  could not be honored because none exist).
- Mandatory dataloader/production-preprocessor parity check (`parity_check.py`,
  `parity_check/parity_result.txt`): `predict_source_keypoints` (the exact primitive
  `evaluate_court_keypoint_owner_gate.py` uses at inference time, and the closest thing to a
  "production preprocessor" this trainer's checkpoint family has) and the training dataloader's
  `real_batch()` per-row preprocessing are **by construction the same code path** (shared
  `load_label_image -> resize -> /255 -> permute` primitive). Verified empirically:
  `max_abs_diff = 0.0`, tensors identical. Caveat: this is *not* parity with `court_model_v2.pt`'s
  own production wiring -- see the architecture-mismatch section; that checkpoint family has no
  reachable trainer at all in this repo.
- ARM-A: `--epochs 3506 --real-finetune-start-epoch 0`. Ran clean, 0 stderr lines. Loss fell
  monotonically (5.45 -> 4.67); VAL-external (source-disjoint testworkspace+xuann) improved
  median 423.21px -> 13.49px, PCK@5 0.0 -> 0.1556.
- ARM-B': `--epochs 3506 --real-finetune-start-epoch 120`. Ran clean, 0 stderr lines.
  VAL-external median 423.21px -> 13.77px, PCK@5 0.0 -> 0.2056.

## P4 -- eval (kill-bar verdicts)

**R2 bar: CARD-A pooled median <25px AND PCK@5 delta >= +0.30 absolute over control.**

| arm | card | pooled PCK@5 | delta vs control | median (px) | p95 (px) | median bar | PCK@5 bar | verdict |
|---|---|---:|---:|---:|---:|---|---|---|
| ARM-A | CARD-A | 0.0 | +0.0 | 197.46 | 1138.87 | FAIL (>>25px) | FAIL (<+0.30) | **FAIL** |
| ARM-A | CARD-B | 0.0 | +0.0 | 431.68 | 1219.17 | FAIL | FAIL | FAIL |
| ARM-B' | CARD-A | 0.0 | +0.0 | 225.76 | 1187.72 | FAIL (>>25px) | FAIL (<+0.30) | **FAIL** |
| ARM-B' | CARD-B | 0.0 | +0.0 | 357.11 | 1071.87 | FAIL | FAIL | FAIL |

Homography-refinement variants are numerically identical to raw for both arms (same caveat as
control). Per-clip PCK@5 is 0.0 for every single clip in both cards, both arms -- the failure is
uniform, not one bad source dragging a pooled average down.

**VAL-external (source-disjoint, in-Roboflow-family) numbers, for context -- NOT the kill bar:**

| arm | median before | median after | PCK@5 before | PCK@5 after |
|---|---:|---:|---:|---:|
| ARM-A | 423.21px | 13.49px | 0.0 | 0.1556 |
| ARM-B' | 423.21px | 13.77px | 0.0 | 0.2056 |

**Reading**: both arms clearly *learned something real* -- VAL-external error collapsed 30x and
PCK@5 went from a random-init 0.0 to 0.16-0.21 on held-out Roboflow-family data. The failure is
specifically a **domain-gap / real-transfer failure**: whatever the model learned from Roboflow
photos does not transfer to the owner's harvest/eval-clip video-frame domain (steep broadcast
angles, YouTube compression, different court paint/lighting). This is exactly the question R2
was designed to ask, and the honest answer is **no, this recipe does not survive real
transfer** -- at the achievable (non-court_unet_v2, non-ImageNet, tiny-custom-CNN) architecture
this lane was actually able to run. ARM-B' (more synthetic pretraining) was slightly *worse* on
CARD-A, slightly *better* on CARD-B, and slightly better on VAL-external -- a mixed, inconclusive
signal, not a rescue: neither ordering gets remotely close to the kill bar.

## P5 -- ARM-B (literal ImageNet resnet34 init): NOT ATTEMPTED

Per the architecture-mismatch finding above: the only resnet34/ImageNet-capable path in this
repo for a court model (`make_court_unet_v2_model` + `encoder_weights_path`) is not reachable
from `train_court_keypoint_heatmap.py`'s CLI, and wiring it in is a real feature (new
loss/eval path for a 3-head dict output), not something safe to patch live inside this lane's
scope/file ownership. Ran ARM-B' (see above) as the closest achievable, honestly-labeled
substitute instead of silently skipping the "does pretraining help transfer" question.

## P6 -- pull + teardown

- Checkpoints (ARM-A `court_keypoint_heatmap.pt` + `court_keypoint_metrics.json`, ARM-B' same),
  all eval JSON reports (control + ARM-A + ARM-B', raw/PCK@5/PCK@10/homography variants),
  training stdout/stderr logs, probe logs, parity-check output, corpus_stats.json/STATS_REPORT
  from the VM rebuild -- all pulled to `vm_pull/`, md5-manifested (`md5_manifest.txt`, 51 files).
- Prediction overlays: rendered **21** frames (CARD-A's 5 independently-reviewed frames +
  CARD-B's 16 `raw_all` rows across Burlington/Wolverine, since `raw_independent` alone only
  totals 7 frames across both cards, short of "10") using ARM-A's checkpoint, red hollow circle
  = GT, green filled circle = prediction (`vm_pull/overlays/`, `overlays/` on VM).
- VM: created `2026-07-10T11:27:42Z`, delete requested `2026-07-10T13:14:28Z`, delete confirmed
  `2026-07-10T13:15:53Z`. Uptime **1.803h**. `gcloud compute instances list
  --filter=labels.fable-fleet=pickleball` after delete shows only the pre-existing, unrelated
  `pickleball-a100-fleet1` (TERMINATED) -- `pickleball-h100-court1` is list-confirmed absent
  (`fleet_list_after_delete.log`).
- Cost: H100 SPOT ase1 rate band (per this project's own prior same-SKU/zone billing,
  `$0.57-4.25/hr`) x 1.803h uptime = **$1.03-$7.66** (mid ~$2-4), well under the $22 ceiling.
  Zero preemptions.

## Honest issues

1. **Architecture mismatch (decisive)**: literal ARM-A/ARM-B as specified in
   `DESIGN_RULING.md` R2/R8 are not achievable with the pinned trainer -- see above. This is the
   single most important finding of the lane and should inform how R2/R8 are re-scoped: either
   (a) accept a `court_unet_v2`-compatible trainer needs to be built/adopted before this probe
   can be run as literally specified, or (b) formally re-scope the ruling to the
   achievable-architecture ablation this lane actually ran.
2. Could not reconstruct the exact 49-file harvest `--guard-image-root` the original
   `court_data2b` corpus build used (no invocation was recorded in the repo). Used
   git-index-only guarding instead; verified functionally equivalent via exact row-count/
   histogram/split/6-of-6-sha256 parity with the Mac-committed reference (zero risk of leakage
   since `data/roboflow_universe_20260706` is 100% external, physically disjoint from any local
   harvest/eval frames).
3. Homography refinement is a numerically exact no-op in every run this lane executed (control
   and both arms, CARD-A and CARD-B) -- worth a dedicated look, not investigated here (out of
   scope, and the kill bar already fails without it).
4. No augmentation CLI flags exist in `train_court_keypoint_heatmap.py`; the spec's "standard
   aug flags" instruction could not be honored because there is nothing to apply.
5. `models/checkpoints/court_external/` (resnet34 + PnLCalib + TennisCourtDetector) is
   completely absent from the `pickleball-fleet-snap-20260709-w7close` snapshot -- a new,
   larger snapshot gap than the one two prior lanes (`w7_p22gate`, `w7_masklet`) already
   flagged against the older `w6close` snapshot (which was just 2 files). Fold into the next
   snapshot cut.
6. GPU stockout cost ~65 minutes of retries (27 attempts, zero dollars) before the first VM
   creation succeeded -- sustained, not transient; both named fallback zones (`-b`, `-c`) were
   affected simultaneously.
7. `--enable-homography-refinement` aside, PCK@5 was uniformly 0.0 across every single clip in
   both cards for both arms -- this is a clean, unambiguous FAIL, not a borderline call the
   HONEST-OUTCOME RULE required any judgment on.

## Best-stack delta

**None.** Neither arm cleared the R2 bar, so per spec's "Expected (b) at most" guidance and the
HONEST-OUTCOME RULE, no PENDING candidate row is added. `configs/racketsport/best_stack.json`
is untouched by this lane. `VERIFIED=0` unchanged.

## Artifacts (all under this lane dir)

- `version_stamp.json`, `vm_running_confirmed.txt`, `vm_create_success.txt`
- `h100_create_*.log` (27 attempts documenting the stockout)
- `control_rows/` (VM), `vm_pull/control_rows/` (Mac) -- 8 control-row JSON reports
- `probe/probe_summary.json`, `vm_pull/logs/probe_std{out,err}.log`
- `parity_check/parity_result.txt`, `parity_check/parity_check.py`
- `vm_pull/arm_a/`, `vm_pull/arm_bprime/` -- checkpoints + training metrics
- `vm_pull/arm_a_eval/`, `vm_pull/arm_bprime_eval/` -- CARD-A/B eval JSON per arm
- `vm_pull/corpus_rebuild/corpus_stats.json`, `STATS_REPORT.md` -- VM-rebuilt corpus provenance
- `vm_pull/overlays/` -- 21 prediction-vs-GT overlay JPEGs
- `vm_pull/logs/` -- full training stdout/stderr for both arms
- `md5_manifest.txt` -- 51-file manifest of everything pulled
- `vm_delete.log`, `fleet_list_after_delete.log`, `vm_delete_start.txt`, `vm_delete_end.txt`
