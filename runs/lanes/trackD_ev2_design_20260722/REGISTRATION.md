# E-v2 frozen experiment registration

Date: 2026-07-22  
Lane: `trackD_ev2_design_20260722`  
Track: D / EVENTS  
Status before execution: `REGISTERED_NOT_RUN`, `VERIFIED=0`

This document freezes one bundled recipe-repair experiment before GPU training. The executable
operator procedure is [VM_RUN_PLAN.md](VM_RUN_PLAN.md). A GPU operator may substitute no data,
seed, threshold policy, loss setting, step count, guard, or judge call without a new registration.

## Quarantine and judge boundary

- The four protected clips are evaluation-only and are not used here.
- For the registered GPU experiment, the protected-50 answer inventory is sealed. Neither Stage P
  nor Stage F may open it. The SHA-pinned owner manifest already carries its prior
  protected-overlap attestation; Stage F uses that attestation and inspects only `split` on the 41
  validation rows. This CPU design lane did not preserve that seal: legacy/repository tests opened
  protected answer JSONs as disclosed in `INPUT_LOCK.json` and `REPORT.md`. No protected scoring,
  training, tuning, or manual answer inspection occurred, but the design-lane result is therefore
  `PARTIAL`, not a clean protected-token PASS.
- Owner-41 is frozen. Stage P and Stage F do not construct its windows, inspect its event fields,
  score it, select checkpoints on it, or tune a threshold on it. Stage V invokes the frozen judge
  exactly once after all train-side guards pass.
- `scripts/racketsport/eval_event_head.py` and
  `threed/racketsport/event_head/matcher.py` remain unchanged from the E1 recorded code commit
  `9bbd8011828631b4cc7df4afdf3b1932e758914a`. Their current SHA-256 values are respectively
  `a0c172f73231113af3c14bcfb8b91dd83415e5406ab89d0439b697d27848e22f` and
  `2272a01d94a02d6663764b3fc7018f43b70bec428a8ad7c2c3fc125373149b62`.
- The frozen matcher uses `peak_pick(..., nms_radius=2)`. E-v2 therefore registers NMS suppression
  at `+/-2 frames`; the judge path and metric math are not modified.
- All outputs remain `verified: false`. An owner-41 pass makes a later promotion lane eligible; it
  is not itself a protected-set promotion claim.

## Historical baseline and experimental question

The immutable historical comparator is E1-B, not a newly retrained control. E1 established the
directional cause but failed both guards:

| Historical arm | macro-F1 @ +/-2f | negative FP | timing p90 | full-video rate | selected step |
|---|---:|---:|---:|---:|---:|
| E1-A owner only | 0.0 | 2/22 | 64f | 0.0152/s | 900 |
| **E1-B teacher** | **0.13043478260869568** | **4/22** | **2f** | **0.1065/s** | 800 |
| E1-C shuffled placebo | 0.0 | 0/22 | 64f | 0.0216/s | 1000 |

The E1 verdict was `EVENT_PBV_SEED1_NO_LIFT`. E-v2 asks one question: does the registered
pretrain-then-fine-tune curriculum and repair bundle retain at least E1-B's macro-F1 while fixing
its over-fire and under-fire guards?

E1's `best` checkpoint was not train-side selected. At code commit `9bbd8011`,
`finetune_event_head.py` constructed all 41 owner-validation windows, scored them at step 0 and
every 100 steps, and selected by owner-validation macro-F1 (with positive confidence as the
zero-F1 tie-break). E-v2 deliberately uses a stricter policy: Stage P selects on a deterministic,
source-disjoint teacher-corpus holdout; Stage F selects the fixed terminal step. Owner-41 has no
selection role.

## Registered arm and exposure

There is exactly one new arm and one seed. The frozen evaluator only accepts schema labels A/B/C,
so the sole Stage-V call uses judge label `B`; its semantic arm name is `EV2_RECIPE`.

| Arm | Seed | Stage P optimizer steps | Stage F optimizer steps | Owner-41 scores |
|---|---:|---:|---:|---:|
| `EV2_RECIPE` (`--arm B` at the judge) | 20260722 | 1000 | 1000 | exactly 1 |

No ablation arm is purchased. E1 already established B>A and B>C; this run repairs the recipe as a
bundle and does not claim per-element causality. The Stage-F 1000-step budget matches E1-B. Every
implemented element has a CLI toggle for a future separately registered, exposure-matched
ablation, but no toggle may change in this run. There is one seed maximum and no automatic retry.

## Immutable inputs

`INPUT_LOCK.json` is the machine-readable companion. The binding files are:

| Role | Path | SHA-256 |
|---|---|---|
| Stage-P agreement manifest | `runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json` | `f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7` |
| Stage-F owner manifest | `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json` | `84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b` |
| Invalid E0 manifest, hard-negative superset only | `runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json` | `9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5` |
| T20 model-only initialization | `runs/lanes/abc_experiment_20260721/vm_pull/inputs/frozen_t20_event_head.pt` | `f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082` |
| Agreement decisions audit | `runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/agreement_decisions.jsonl` | `81df518a85ce891b4b2da1b494f8b123979367d9b70432b4c9af850f4a88792c` |
| Complete rate-media inventory | `runs/lanes/trackD_ev2_design_20260722/RATE_MEDIA_LOCK.json` | `79ecae3a6bb57af0b1d3a2548c05b0be70ac42600a50c22c2586752c111de5ee` |

`INPUT_LOCK.json` additionally freezes all six Stage-P public-video IDs, their manifest-recorded
SHA-256 values, and the unauthenticated provenance URL template used to reconstruct their exact
absolute VM paths. `RATE_MEDIA_LOCK.json` freezes each of the 40 rally-media relative paths and
byte SHA, split membership, exact per-source counts, and train-side frame/duration facts. The GPU
plan validates that lock before VM creation and again after transfer; no owner-manifest field can
condition this inventory.

The repaired Stage-P manifest has 1,189 rows: 636 HIT and 553 BOUNCE. Its agreement-family counts
are 773 `ball_velocity_kink`-only and 416 `audio_onset+ball_velocity_kink`. Contrary to the
dispatch's shorthand, the pinned file's **effective** weights are 803 rows at 0.25 and 386 at
0.5. Thirty of the 416 two-family records keep weight 0.25 because their audio match did not pass
the preregistered per-video time-shift-null eligibility check. E-v2 consumes the pinned
`sample_weight` values; it does not upgrade those 30 rows and thereby undo the E0 ruling.

The T20 checkpoint is loaded model-only into Stage P: all legacy classifier tensors are retained,
the new offset layer alone is seeded and initialized, Adam is fresh, and the Stage-P counter starts
at zero. Stage F then loads the selected Stage-P checkpoint model-only with another fresh Adam and
another zeroed step counter.

## Train-side hard negatives

The four E1-B owner-validation over-fires establish the failure signature only. Their row identities
are not opened, copied, or trained on. The replacement pool is entirely train-side and exactly
reproducible:

1. Key the invalid 1,481-row E0 Arm-B manifest and repaired 1,189-row manifest by
   `focal_event_id`.
2. Require the repaired rows to be an identity-preserving subset.
3. Take the exact 292-row difference and require every difference row's only independent family to
   be `audio_onset`. This is the E0-excluded audio-only false-positive family.
4. Remove all 30 rows whose `source_video`/video ID is the Stage-P held-out source
   `st0epgnab7dr`; hard-assert `292 - 30 = 262` candidates and assert no remaining candidate's
   source ID equals the held-out source. The removed rows may not affect mining, training,
   class-weight computation, or the pre-score guard.
5. Relabel each of the 262 source-clean candidates to `events=[]` with
   `[background,HIT,BOUNCE]` all loss-valid.
6. Before Stage-F updates, score the excluded teacher frame with the selected Stage-P model, rank by
   descending maximum positive probability, break ties by ascending `focal_event_id`, and select
   the top 96.
7. Mix exactly 4 selected hard-negative windows with exactly 8 owner windows on every one of the
   1,000 steps. The 61 owner windows use a seed-20260722 permutation; when a permutation cannot
   fill an eight-window batch, a newly seeded deterministic reshuffle tops it up, and the unused
   suffix carries into the next DataLoader iteration. Both 8 and 4 are asserted before every
   merge; `drop_last` is not used and the step count remains 1,000. Cap hard-negative
   post-class-weight aggregate loss at 0.5 times the human-owner aggregate loss in that batch.

The 21 negative rows already in owner-train remain human-weight training rows and later form the
owner-train negative proxy. No owner-validation row can enter either pool.

## Frozen recipe

| Element | Stage P | Stage F | Toggle / exact rule |
|---|---|---|---|
| Class weighting | sqrt-frequency | sqrt-frequency after top-96 selection | Counts are the actual loss-eligible dense target mass in the constructed training windows. `w_c=sqrt(n_background/n_c)`, normalized to background=1. Fixed `(1,5,5)` and inverse-frequency are not used. |
| Label dilation | on | on | `+/-1` frame; center remains one-hot, eligible background neighbor becomes 0.5 background + 0.5 positive. UNKNOWN frames are never dilated into. |
| Label assignment | Hungarian | fixed | Stage P uses detached min-cost one-to-one assignment within `+/-2f`: `1.0 * -log p(class) + 0.25 * abs(timestamp-candidate)/2`. UNKNOWN frames are ineligible. Stage F owner truth stays fixed; the same CLI can register Hungarian later. |
| Offset regression | on | on | Additive two-channel HIT/BOUNCE sub-frame head, Smooth-L1 beta 1.0, auxiliary loss weight 0.2. `forward()` and judge logits remain classification-only; old checkpoints still load. |
| Decode | source-held threshold lock | same numeric lock | Threshold grid `0.20,0.25,...,0.70`; NMS radius 2. Stage F and Stage V may not retune it. |

`checkpoint-selection=final-step` is the registered E-v2 mode. Its first executable act, before
mkdir/unlink or any manifest, media, checkpoint, or lock read, validates the complete argument set.
Both the 100-step probe and 1,000-step full run require image 224, window/stride 64/32, owner batch
8 plus hard-negative batch 4, LR 0.001, validation-cadence argument 100 (selection remains terminal
only), seed 20260722, workers 4, fixed class-weight argument `(1,5,5)` superseded by actual-count
sqrt weighting, and pseudo cap 1.0. The probe's sole exposure exception is `steps=100`; the full run
requires `steps=1000`. Both require fixed (not Hungarian) assignment, shift 0, class cost 1.0,
temporal cost 0.25, dilation 1 with neighbor weight 0.5, offset loss 0.2 with beta 1.0,
hard-negative loss cap 0.5, candidate/selected counts 262/96, the single held-out-source exclusion,
owner/val row counts 61/41, guard bounds 21, 2, 26, `[0.3,1.0]`, inventory counts 38/4, and the
SHA-pinned rate-media lock. The probe wall is exactly 180 minutes; a measured full-stage wall must
be positive and at most 180. Legacy defaults remain available only in owner-val mode; any absent or
different final-step value is a typed error with zero input opens.

`GAUSSIAN_SOFT_LABEL_FALLBACK_NOT_USED` is a named non-adoption: clean Hungarian assignment was
implemented and tested, so the registered fallback is unnecessary. The `+/-1f` dilation is not
misreported as a Gaussian target.

## Stage-P selection and threshold lock

Seeded SHA-256 source ordering with seed 20260722 holds out exactly one of the six teacher source
videos: `st0epgnab7dr` (226 rows). The other five sources supply 963 training rows. This split is
byte-deterministic and source-disjoint.

For the registered `64`-frame windows and `+/-1f` dilation, the 963 Stage-P training rows have
loss-eligible dense class mass `[57563.0, 1035.5, 888.5]` in
`[background,HIT,BOUNCE]` order. The resulting frozen formula resolves to
`[1.0, 7.45584135131073, 8.049019765763125]`. Their row-weight tiers are 642 at `0.25` and 321 at
`0.5`. The GPU plan asserts these values after construction. Stage F recomputes the same formula
only after its model-dependent top-96 mining step and records both counts and weights; every class
count must remain positive.

At steps 100, 200, ..., 1000, each validation batch is inferred once and its logits are decoded at
the frozen threshold grid. Select the checkpoint/threshold pair by this lexicographic order:

1. macro-F1 at `+/-2f`, descending;
2. false positives, ascending;
3. false negatives, ascending;
4. threshold, ascending;
5. on an exact remaining tie, retain the earlier checkpoint.

The resulting numeric threshold, NMS radius, checkpoint step, checkpoint SHA, split, grid, and
tie-break are written to `stage_p_decode_threshold_lock.json` before Stage F. Stage F may start
only after the **Stage-F CLI itself** requires both the lock and Stage-P `train_manifest.json` and
hard-verifies: lock SHA from the train manifest; threshold against lock, train manifest, and
checkpoint; the exact 11-value grid; NMS 2; exact tie-break; held-out source; selected step against
checkpoint `completed_steps`; checkpoint SHA; and the Stage-P data-manifest cross-SHA. Shell
preflight repeats these checks but is not the enforcement boundary. The lower
threshold is only a final prediction-plateau tie-break; false positives remain the first safety
tie-break. Neither the threshold nor checkpoint is selected from owner data.

## Exact hyperparameters

| Setting | Stage P | Stage F |
|---|---:|---:|
| seed | 20260722 | 20260722 |
| image size | 224 | 224 |
| temporal window / stride | 64 / 32 frames | 64 / 32 frames |
| optimizer | fresh Adam | fresh Adam |
| learning rate | 0.001 | 0.001 |
| target optimizer steps | 1000 | 1000 |
| validation cadence | every 100 steps, teacher internal only | none; fixed terminal step |
| owner / primary batch | 8 teacher windows | 8 owner windows |
| auxiliary batch | none | 4 mined hard negatives |
| dataloader workers | 4 | 4 |
| hard-negative raw / held-out removed / candidate / selected | n/a | 292 / 30 / 262 / 96 |
| hard-negative loss cap | n/a | 0.5 x human aggregate |
| offset loss / beta | 0.2 / 1.0 | 0.2 / 1.0 |

The architecture is the existing RGB MobileNetV3-small-frame encoder plus bidirectional GRU event
head. This experiment adds no track, wrist, pose, ball-state, or audio conditioning channels.
Track/wrist conditioning is a named follow-up. Audio late fusion remains gated on Track B's own-clip
measurement and is not smuggled into E-v2.

## Train-side pre-score guards

After the terminal Stage-F update and before the sole owner-41 call, decode with the Stage-P numeric
threshold and frozen NMS radius 2:

- Full-video firing rate on the SHA-pinned, complete inventory of exactly 38 MP4s spanning the four
  registered train-source directories (`73VurrTKCZ8`: 8, `Ezz6HDNHlnk`: 8, `_L0HVmAlCQI`: 19,
  `wBu8bC4OfUY`: 3) must be in `[0.3, 1.0] events/s`. The frozen inventory totals 57,025 frames and
  2,063.1827083333333 seconds. `RATE_MEDIA_LOCK.json` is the sole membership rule: it binds all
  paths and bytes plus the two validation media objects, and the six on-disk source directories
  must contain exactly that 38+2 set. Membership never comes from owner label rows. Counts 38/4,
  validation count 2, per-source counts, aggregate decode facts, and empty overlap are hard
  assertions.
- Total predictions across the 21 owner-train zero-event windows must be `<=2` (stricter than a
  row-level false-positive count).
- Rows with any prediction across all 262 source-clean audio-only train-side candidates must be
  `<=26/262` (10%, floored; both fired-row and total-event counts are recorded). The 30 held-out
  `st0epgnab7dr` rows do not enter this guard.

Sqrt weighting, one-frame dilation, and the locked threshold are the registered under-firing
levers. Sqrt weighting, the relabeled hard negatives, and both train-side negative proxies target
over-fire. A guard failure emits `EVENT_EV2_INTERNAL_GUARD_FAIL_NO_SCORE`, marks the checkpoint
owner-score-ineligible, and spends zero owner-41 touches.

## Frozen Stage-V gate

Only an exact 1000-step Stage-P completion, exact 1000-step Stage-F completion, finite losses,
matching input/output hashes, a present threshold lock, and all internal guards permit the one
judge invocation. PASS requires every row below:

| Metric / invariant | PASS bound |
|---|---:|
| owner-41 macro-F1 at +/-2f | `>= 0.13043478260869568` |
| owner-41 negative false positives | `<= 2/22` |
| owner-41 full-video firing rate | `0.3-1.0/s` inclusive |
| owner-41 scoring calls | exactly 1 |
| protected-50 scoring calls | 0 |
| seed count | exactly 1 |

There is deliberately no standalone timing-p90 PASS row. The frozen judge constructs timing errors
only from pairs accepted by `greedy_match(..., tolerance_frames=2)`, so macro-F1@+/-2 greater than
zero mathematically implies every matched timing error, and therefore its p90, is `<=2f`. When
there are no matches, the frozen judge reports `window_frames=64` as its sentinel. Thus E1-B's
reported p90 `2f` was already implied by its positive macro-F1, while E1-A/C's `64f` values were the
zero-match sentinel. The descriptive p90 remains in evidence but is not an independent gate; the
judge and matcher are unchanged. Passing verdict:
`EVENT_EV2_RECIPE_REPAIR_PASS`. Any scored metric failure is
`EVENT_EV2_RECIPE_REPAIR_NO_LIFT`. An incomplete exposure is `EVENT_EV2_RUN_INCOMPLETE`. None may
be retried under this registration.

## Probe, wall, spend, and stop rules

- Use one A100-40 GB. E1 ran a larger 8-owner + 8-pseudo batch on that device; E-v2's maximum
  Stage-F batch is 8+4, so the registered device is sufficient.
- Provision one fresh `pickleball-gpu-ev2` in project `gifted-electron-498923-h1`, zone
  `us-central1-f`, as exact `a2-highgpu-1g` Spot with one
  `NVIDIA A100-SXM4-40GB`, exact image `pickleball-cache-image-20260722` (family
  `pickleball-cache`), one 200 GB `pd-balanced` auto-delete boot disk, and shared zonal disk
  `pickleball-cache-data-usc1f` attached `mode=ro,device-name=cache` and mounted read-only at
  `/cache`, with provider max-run action retained as `DELETE`. Labels are exactly lowercase:
  `fable-fleet=pickleball,fable-lane=trackd_ev2_20260722,owner=arnavchokshi`. Before create, assert
  the registered machine/accelerator exists in usc1f, regional Spot CPU/A100 quota is sufficient,
  and the image/disk are `READY`. A capacity failure is a registered
  `ABORT` requiring a new registration to move zones; there is no automatic zone fallback.
  The cache image supplies repo `e1e2184df` and the registered environment
  `torch 2.13.0+cu130`, `torchvision 0.28.0+cu130`, CUDA 13.0. Setup performs
  `git fetch origin` and checks out exact `RUN_COMMIT` in that baked repo; clone, apt, venv creation,
  and pip mutation are removed. The deleted E1 VM is not an input.
- Cache substitution is conservative and flag-gated. Swap only intended rows whose
  `CACHE_MANIFEST.json` flags include exact `sha256_matches`; any row carrying
  `SHA256_MISMATCH`, `QUARANTINED*`, or `COMPARE_ONLY_NEVER_TRAIN` is ineligible regardless of a
  stale snapshot claim. The six Stage-P teacher clips `143sf3gdwxsa`, `98z43hspqz13`,
  `st0epgnab7dr`, `td2szayjwtrj`, `utasf5hnozwz`, and `xkadsq9bli3h` are the only swapped
  inputs, and each is rehashed against the unchanged `INPUT_LOCK.json` pin before use.
  `83gyqyc10y8f` is not an E-v2 teacher clip and is independently excluded by its
  mismatch/compare-only/quarantine posture. The 40-MP4 owner rally-media universe, owner and
  Stage-P manifests, hard-negative manifest/decisions, and T20 checkpoint retain their existing
  registered SCP/tar transport paths verbatim because the cache does not prove those exact objects
  at those pins. This swapped-versus-registered-path split is binding.
- Before any training input read, run the fail-closed
  `scripts/racketsport/verify_training_inputs.py` gate at `RUN_COMMIT` and preserve successful
  `gate_proof.json` with the run artifacts. `RUN_COMMIT` must contain the separately reviewed
  Track-E safety revision; its exact SHA is filled at dispatch by the serialized integration owner,
  not in this registration. Missing, stale, malformed, or failed proof is terminal, and the
  Stage-F trainer receives a freshly regenerated, at-most-900-second proof path explicitly before
  each probe/full invocation; the direct guard probe validates its own fresh proof before reads.
  Dispatch is forbidden until the ledger at `RUN_COMMIT` queue-authorizes every declared
  pre-existing input and the in-run generated Stage-P/Stage-F asset contracts. The controller otherwise transport-hashes
  all 40 MP4s in the rally-media directory without reading the owner manifest (these are not the
  protected-4 or protected-50 assets). Hashing and staging read bytes but do not decode or infer.
  Stage F decodes the complete label-independent 38-file inventory from the four frozen train
  source directories. After the one-touch marker is written, Stage V verifies the two unique val
  media paths against their manifest SHA pins and then runs the sole registered judge.
- Run a fresh 100-step probe for each stage, discard its weights, and then restart that stage from
  its registered initialization.
- For Stage P compute
  `ceil(1.5 * probe_elapsed_seconds / 100 * 1000 / 60)` minutes. For Stage F preserve the measured
  one-time mining cost and scale only the training component:
  `ceil(1.5 * (probe_one_time_seconds + probe_training_seconds/100*1000) / 60)`.
  The 50% factor is the only contingency. The Stage-P probe includes the same decode cadence; the
  Stage-F probe includes hard-negative mining. Its intentionally immature checkpoint skips
  terminal guards and is score-ineligible. Run that train-side guard workload once on the
  score-ineligible probe model to measure it, then give the full Stage-F outer timeout 150% of the
  measured guard time in addition to its optimizer/mining cap.
- The Stage-F `--max-wall-minutes` clock starts before hard-negative derivation/mining and remains
  binding through the final optimizer step. Code checks it before each batch and immediately after
  every `optimizer.step()`; a post-update expiry raises typed exit 31 before any terminal guard.
  Mining cannot borrow the separately measured terminal-guard allowance. The outer process timeout
  is `F_CAP + guard allowance` only so a within-cap optimizer may run terminal train-side guards.
- Expected setup is 20 minutes total: approximately four minutes for cache-image boot and
  read-only disk attach/mount (the fleet smoke measured about 3.6 minutes), plus 16 minutes for
  exact checkout, retained ~1.01 GiB owner-media transport, small registered inputs, hashes, and
  the step-0 gate proof. Abort if either optimizer/mining full-stage cap exceeds 180 minutes, if the Stage-F all-in outer
  cap exceeds 210 minutes, or if the sum of both 100-step probe times, the guard-probe time, both
  full-stage caps, the Stage-F guard allowance, and a 30-minute Stage-V val-media-integrity plus
  judge allowance exceeds 300
  compute minutes. E1-B's observed 5,790 seconds for 1000 steps implies a historical 145-minute
  Stage-F optimizer contingency cap. The all-in registered arithmetic is therefore
  `20 setup + <=300 compute + 30 pull/hash/stop = <=350 minutes`.
- Before VM creation, fetch the complete current Compute Engine SKU catalog from Google's Cloud
  Billing Catalog API v1 and select the five frozen official SKU IDs: Spot A2 core
  `3178-715E-CFB6`, Spot A2 RAM `65A3-16DB-D57A`, Spot A100-40 `39D4-516A-0317`, balanced PD
  `6AE1-525F-8B80`, and Spot external IPv4 `4AF8-7C1F-39C4`. Mechanically price the exact
  usc1f 12-vCPU/85-GiB/one-GPU shape, disposable 200 GiB boot disk at 730 hours/month, and one
  IPv4. The pre-existing shared cache disk and image are fleet resources, not charged to this run.
  Persist exact SKU
  descriptions, units, component prices, API URL, retrieval time, and effective times; reject a
  missing/duplicate component, a retrieval older than 15 minutes, a future effective time, an
  incomplete/nonfinite sum, or a total above `$3.30/hour`. An older effective date is valid for an
  unchanged current SKU price and is not treated as quote staleness. Operator-entered rates and
  source strings are forbidden. Starting at
  GCE's recorded `creationTimestamp` (conservative because it precedes SSH readiness), derive an all-in hard
  deadline as `min(350 minutes, floor($19.50 / quoted_rate * 60))`. The clock therefore includes
  provisioning, cache-image checkout and disk attach, retained SCP/tar staging, the step-0 gate,
  probes, training, the judge, pull, hashing, and
  confirmed shutdown—not only CUDA
  work. Install a guest-shutdown watchdog for that deadline and never cancel or extend it.
- Reserve 30 all-in minutes for pull/hash/stop and assert sufficient remaining time before every
  probe, full stage, and judge call. The controller records the confirmed-stop timestamp, computes
  a conservative start-to-stop spend upper bound from the frozen quote, writes it to the handoff,
  and requires it to be `<= $20`. The `$19.50` planning ceiling and 350-minute maximum leave both
  dollar and time headroom beneath the hard cap. At the admitted rate ceiling, 350 minutes costs at
  most `$19.25` in hourly resources. Final accounting adds `$0.50` as a conservative non-hourly
  network/rounding reserve, leaving `$0.25` for stop latency before checking the `$20` cap.
- Stages and probes run sequentially, never concurrently. Abort on a SHA mismatch, missing media,
  wrong GPU, non-finite loss, OOM, wall exit, unequal steps, missing lock, guard failure, or any
  owner-validation access before the registered judge call.
- Every workload timeout sends TERM at the bound and KILL 30 seconds later. Outer timeouts bind
  even pre-cap work: 15 minutes for the Stage-P probe, 30 minutes for the
  Stage-F mining/training probe, 20 minutes for the train-side guard probe, the computed cap for
  each full stage, and 30 minutes for the val-media SHA check plus sole judge call. Any Stage-V
  timeout still consumes the one-touch owner-41 token.
- Before VM creation, the controller proves the frozen commit is reachable from `origin/main`,
  requires every formal lane artifact (including both repair briefs/reports, cross-track
  assumptions, and the rate-media lock), and reads both `CODE_SHA256SUMS` and every reviewed target
  with `git show RUN_COMMIT:path`. The working tree cannot satisfy this proof. The VM checks out
  that exact commit on its local branch named `main`; it never pulls a moving branch head.
- Stale controller state is rejected before create. The create-attempt flag is armed before the
  provider call, and the first provider ID capture uses temp-plus-atomic-move. Setup/bootstrap
  failures before ID capture still delete the one freshly named resource and confirm both VM and
  disk absence. Every later terminal route uses the same identity-bound tolerant finalizer, pulls
  whatever evidence exists, falls back to the content-blind spend bootstrap, and writes durable
  teardown confirmation. Every failure, watchdog, max-run, and success teardown retains `DELETE`
  for the disposable VM and boot disk while detaching and never deleting
  `pickleball-cache-data-usc1f` or `pickleball-cache-image-20260722`. The fleet snippet's
  `instance-termination-action=STOP` example is not adopted. A recycled same-name instance is
  never targeted.
- Maximum seeds is one. Infrastructure failure, wall failure, and gate failure all require a new
  registration; there is no silent resume, alternate threshold, extra seed, or second score.

## Result routing and best-stack policy

On PASS, the E-v2 GPU execution lane fsyncs a temporary handoff and publishes it with one atomic
`os.replace` as immutable `BEST_STACK_PENDING.json` evidence for a disabled `PENDING`
`events.ev2_checkpoint` candidate. A `finally` removes any temporary on every failure. It does not mutate
`configs/racketsport/best_stack.json` or its revision-pinned test. The serialized Track-D
integration owner may later validate and apply both production-file changes in a separate atomic
integration transaction. Wiring `sequence_dp.py` becomes eligible only in a separate reviewed
lane. On FAIL, sequence-DP stays dormant and no pending handoff or best-stack entry is added.

There is **no best-stack delta in this design/code lane**: nothing was trained or promoted.

## Cross-signal row

**CONSUMES:** `ball_velocity_kink` + `audio_onset` agreement families (corpus tiers), pb.vision
teacher timestamps, owner event labels.  
**FEEDS:** ball-3D arc anchors (event candidates), rally segmentation, `sequence_dp` decode stage,
audio late-fusion gate (Track B artifact pending).

## Cross-track assumptions (manager addendum, 2026-07-22 — non-operative; changes no gate, bound, command, or data value)

- st0epgnab7dr (the Stage-P source-disjoint holdout on which checkpoint + threshold selection run)
  is EXCLUDED from Track A court-training promotion for the lifetime of this holdout, by Track A's
  own record (runs/lanes/court_owner_pack_20260722/results/PROMOTION_RECORD.md), to prevent
  cascade-fusion selection bias (court-derived inputs unrepresentatively strong exactly where E-v2
  selects). Track D notifies the manager when this holdout retires so the video rejoins court
  training. Track D side: runs/lanes/trackD_ev2_design_20260722/CROSS_TRACK_ASSUMPTIONS.md.
