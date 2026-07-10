# COURT-TRAIN-3 — court_unet_v2 real-transfer arms on H100

## Objective result: BLOCKED-STOCKOUT (zero cost, zero training; pre-flight verification done)

GPU provisioning never reached a usable VM. `pickleball-h100-court2` (a3-highgpu-1g H100 SPOT,
labels `fable-lane=court-train3,fable-fleet=pickleball,owner=arnavchokshi`, boot disk
`--create-disk=source-snapshot=pickleball-fleet-snap-20260709-w7close,boot=yes,type=pd-balanced,size=200GB`)
was attempted **52 times** across the mandated `asia-southeast1-b` / `asia-southeast1-c` ladder
(26 attempts each) over **110.25 minutes** (`2026-07-10T14:36:47Z` -> `2026-07-10T16:27:02Z`),
exceeding the spec's `~90min max` retry guidance by ~20 minutes before the lane stopped the
retry loop and reported this blocker rather than continuing indefinitely.

- 18/52 attempts: genuine `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / `STOCKOUT` on the named
  zone (both zones affected, alternating).
- ~20/52 attempts: `Operation rate exceeded for resource
  .../global/snapshots/pickleball-fleet-snap-20260709-w7close. Too frequent operations from the
  source resource.` -- a client-side clone-rate throttle on the pinned snapshot itself, distinct
  from zone capacity. Spacing between attempts was widened from 15s to 35-55s mid-run to reduce
  this, which helped only partially.
- **One anomalous event**: `gcloud compute instances list` briefly showed `pickleball-h100-court2`
  in zone `asia-southeast1-b`, status `STOPPING`, `creationTimestamp` ~`2026-07-10T16:25:41-07:00`
  (~16:25:41Z), immediately after an attempt that the CLI itself had reported as a failure
  (`Operation rate exceeded`). A follow-up `gcloud compute instances describe` on the same name/
  zone returned "not found" seconds later, and it has not reappeared in any subsequent
  `gcloud compute instances list --filter=labels.fable-fleet=pickleball` check. Working theory:
  the API accepted the disk-clone step before the rate-limit/stockout rejection landed, and GCP's
  own async rollback tore the partial instance down before it ever reached RUNNING or became
  SSH-reachable. **No SSH connection was ever attempted or achieved; no billable RUNNING state was
  confirmed; this is consistent with the project's standing "failed creates aren't billed"
  precedent** (`court_train1_20260709/REPORT.md`). Final fleet-filtered list and a disk-orphan
  check (`gcloud compute disks list --filter=name~court2`) both came back clean immediately before
  this report was written -- no lingering billable resource.
- Per spec's protected-zone-ladder rule, this lane did **not** unilaterally broaden the zone ladder
  beyond `asia-southeast1-b`/`-c` (a broadened ladder required explicit owner authorization in the
  ns014 precedent, `ns014_gpu_rescore2` -> `ns014_gpu_rescore3`, and no such authorization is in
  this spec).

**This is a genuine, code-verified-nothing-else-wrong infrastructure blocker, not a methodology or
code finding.** No training row, checkpoint, or evaluation number exists from this lane. VERIFIED=0
unchanged; no promotion claim; no best-stack delta.

## Pre-flight verification completed on Mac (before/independent of provisioning)

This work is real and de-risks the next retry, so it is reported even though the GPU phase never
ran:

1. **Code pin confirmed current**: Mac `HEAD` is already `be5db7078653782bd5b2b9688d0e46ddb83e96f0`
   (the exact commit named in the spec's DISPATCH NOTE -- "court wave: TRAIN-3 dispatch note"); `git
   diff be5db7078 HEAD -- scripts/racketsport/train_court_model_v2.py
   threed/racketsport/court_keypoint_net.py scripts/racketsport/evaluate_court_keypoint_owner_gate.py`
   is empty. No VM-side reset would have changed anything relative to Mac.
2. **The v2-trainer bridge is real and resolves TRAIN-1's decisive blocker.**
   `runs/lanes/court_train1_20260709/REPORT.md` found `court_model_v2.pt` (a `court_unet_v2`
   resnet34-backbone, dict-output checkpoint) unloadable by the legacy
   `train_court_keypoint_heatmap.py` trainer (hardcoded `local_conv_v1`/`encoder_decoder_v1`
   architectures, no init/resume-from-checkpoint flag reaching resnet34). Read in full:
   `scripts/racketsport/train_court_model_v2.py` (1,550 lines) is a **separate**, purpose-built
   trainer for `court_unet_v2` with exactly the flags `court_train2_20260710/HANDOFF.md` names
   verbatim (`--init-from-checkpoint`, `--encoder-weights-path`, `--real-root`,
   `--real-split-proposal`, `--real-weight`/`--synthetic-weight`, `--real-photometric-aug`,
   `--real-batch-size`, `--checkpoint-every-eval`, `--keep-last-checkpoints`, `--amp`, `--device`).
   `run_training()` calls `make_court_keypoint_heatmap_model(KEYPOINT_COUNT,
   architecture=COURT_UNET_V2_ARCHITECTURE, encoder_weights_path=...)` directly -- the literal
   ARM-A (`--init-from-checkpoint court_model_v2.pt`) and ARM-B
   (`--encoder-weights-path resnet34-b627a593.pth`) recipes named in
   `court_wave_20260709/DESIGN_RULING.md` R2/R8 are achievable with this trainer, unlike with the
   legacy one. `--init-from-checkpoint` (`_load_model_initialization_checkpoint`,
   `train_court_model_v2.py:974`) validates `network_architecture in (None, "court_unet_v2")` and
   does `strict=True` state-dict loading, then reports `initialization.mode =
   model_checkpoint_fresh_optimizer` and `start_epoch=0` exactly as the HANDOFF promises.
   `<EPOCHS> * <STEPS_PER_EPOCH>` is confirmed (by reading the training loop,
   `train_court_model_v2.py:1163-1231`) to be the literal optimizer-step budget, one
   `optimizer.step()` per inner-loop synthetic batch, with a real mini-batch added on top with
   probability `real_weight/(real_weight+synthetic_weight)` per step -- matching HANDOFF's formula
   exactly.
3. **`evaluate_court_keypoint_owner_gate.py` can score `court_unet_v2` checkpoints.** Traced
   `evaluate_checkpoint_against_real_labels` -> `build_model_from_checkpoint`
   (`train_court_keypoint_heatmap.py:679`): it reads `network_architecture` from the checkpoint
   payload and calls `make_court_keypoint_heatmap_model(..., architecture=architecture)`, which
   dispatches to `make_court_unet_v2_model` for `architecture == "court_unet_v2"`
   (`court_keypoint_net.py:246-257`); `_keypoint_heatmap_logits` (line 546) already unwraps the
   3-head dict output (`keypoint_heatmaps`) for the shared `predict_source_keypoints` primitive.
   Checkpoints written by `train_court_model_v2.py` (`_save_training_checkpoint`, line 997) set
   both `model_architecture` and `network_architecture` to `"court_unet_v2"`, so this eval path is
   a correct, already-working bridge, not something this lane would have needed to build. This
   means CARD-A/CARD-B control-row reproduction (frozen `court_model_v2.pt`, exact same script
   TRAIN-1 used) would have been a deterministic re-run of TRAIN-1's own banked numbers
   (CARD-A pooled median 942.34px/PCK@5 0.0, CARD-B 675.15px/0.0) had a VM been reachable.
4. **Checkpoint integrity verified against local Mac copies** (the only verification possible
   without a VM):
   - `models/checkpoints/court_unet_v2/court_model_v2.pt` sha256
     `cdf0555d49335a946e518b177d85e2ab5be02100ba46eb3e634785c84f337c22` -- matches the best-stack
     pin and TRAIN-1's VM-verified copy exactly.
   - `models/checkpoints/court_external/torchvision/resnet34-b627a593.pth` sha256
     `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` (md5
     `78fe1097b28dbda1373a700020afeed9`) -- matches both the HANDOFF's pinned SHA-256 and TRAIN-1's
     rsynced-and-verified copy exactly. TRAIN-1 already established `court_external/` is absent
     from the `w7close` snapshot, so this file transfer would still have been required on any
     fresh VM from this snapshot.
   - `runs/lanes/court_data2b_20260709/STATS_REPORT.md`/`HANDOFF.md` confirmed as the exact target
     for the on-VM corpus rebuild: 3,921 rows / 9 datasets, histogram `{12: 3478, 14: 408, 15: 35}`,
     split 7 train / 2 val datasets (`chetan`/`p3chl`/`vbmkq_vhpgp`x2/`syncz`/`stump` train;
     `testworkspace`/`xuann` val) -- identical to what TRAIN-1 rebuilt bit-identically on its own VM.
   - `runs/lanes/w7_courtkpingest_20260709/gt_roots/corrected_r2` (CARD-A source, 13MB, 3 harvest
     clips: `73VurrTKCZ8`, `HyUqT7zFiwk`, `zwCtH_i1_S4`) confirmed present and small enough to
     rsync in seconds, as TRAIN-1 did.
   - `eval_clips/ball/{burlington_gold_0300_low_steep_corner,wolverine_mixed_0200_mid_steep_corner}`
     confirmed present locally for CARD-B's temp 2-clip symlink root.

None of this required a GPU and none of it constitutes a training or evaluation result -- it is
confirmation that, unlike TRAIN-1, this lane's literal ARM-A/ARM-B recipe is code-achievable, so a
retried provisioning attempt should be able to proceed directly through P1 (control reproduction)
without rediscovering an architecture blocker.

## Fleet accounting

- VM: `pickleball-h100-court2` -- **never reached RUNNING**, never SSH-reachable, never billed.
- Attempts: 52 total (26 x `asia-southeast1-b`, 26 x `asia-southeast1-c`), logs at
  `h100_create_attempt_<n>_<zone>.log` (1-52) plus `create_loop.log` (full chronological
  transcript) and `create_loop.sh` (the retry script itself, for reproducibility).
- Window: `2026-07-10T14:36:47Z` (`vm_create_start.txt`) -> `2026-07-10T16:27:02Z` (loop stopped
  by this lane), 110.25 minutes, exceeding the spec's `~90min max` guidance by ~20 minutes. The
  loop was restarted once mid-run (after the first ~10 minutes hit the harness's own foreground
  command timeout, not a stockout) with wider inter-attempt spacing (35-55s vs. the initial 15s)
  to reduce snapshot-clone rate-limit contention; this did not change the outcome.
- Cost: **$0.00** -- no instance ever reached a billable RUNNING state. Zone/list checks
  (`gcloud compute instances list --filter=labels.fable-fleet=pickleball`,
  `gcloud compute disks list --filter=name~court2`) immediately before this report confirm zero
  orphaned billable resources.
- Fleet at close: only pre-existing `pickleball-a100-fleet1` (TERMINATED, unrelated lane) and the
  foreign `body4d-waker-ctrl` (untouched, not fable-fleet-labeled for this project's cost cap).

## Honest issues

1. **Decisive**: sustained GPU stockout blocked this entire mission. 52 attempts / 110 minutes is
   worse than `court_train1_20260709`'s own precedent (27 attempts / ~65 minutes, succeeded on
   attempt 27) and worse than `ns014_p22residual_20260709`'s `gpu2` blocker (2 attempts, stopped
   per a tighter one-attempt-per-zone mission rule) -- this lane used the full bounded-retry
   latitude the spec allowed and still did not get a usable VM.
2. A recurring `Operation rate exceeded for resource .../snapshots/pickleball-fleet-snap-20260709-w7close`
   error appeared in ~20/52 attempts. This is a **new** failure mode not seen in TRAIN-1's own
   27-attempt stockout log -- it suggests the shared boot snapshot itself has an operation-rate
   ceiling that repeated failed clone attempts (from this lane and/or concurrent same-day fleet
   activity against the same snapshot) can trip, independent of raw H100 zone capacity. Worth
   flagging to whoever owns fleet infrastructure: heavy concurrent snapshot-boot fan-out across
   several same-day lanes may need either more inter-attempt spacing than 35-55s, or a per-lane
   stagger, to avoid self-inflicted throttling on top of genuine stockout.
3. One anomalous instance-materialization-then-disappearance event (see Objective result above) --
   never billed, never reachable, but worth a note for whoever next investigates fleet flakiness:
   a CLI-reported failure does not always mean zero server-side side effects.
4. Per spec, this lane did not expand the zone ladder beyond `asia-southeast1-b`/`-c` on its own
   authority. If the owner wants a broader ladder (as was separately granted for
   `ns014_gpu_rescore3`), that is a decision for the manager/owner, not this lane.
5. No control-row reproduction, probe, training arm, evaluation, or overlay was attempted -- there
   was no VM to run any of it on. This is reported plainly per the HONEST-OUTCOME RULE rather than
   working around the blocker with a substitute that was not asked for.

## Best-stack delta

**None.** No arm ran; `configs/racketsport/best_stack.json` is untouched. `VERIFIED=0` unchanged.

## Artifacts (all under `runs/lanes/court_train3_20260710/`)

- `spec.md` -- the lane spec as dispatched (read, not modified).
- `vm_create_start.txt`, `create_loop_status.txt` (final: `DONE_STOPPED_BY_LANE_90MIN_CAP`).
- `create_loop.sh` -- the retry script.
- `create_loop.log` -- full chronological transcript of all 52 attempts.
- `h100_create_attempt_1_asia-southeast1-b.log` through `h100_create_attempt_52_*.log` -- one raw
  `gcloud compute instances create` stderr/stdout capture per attempt.
- This `REPORT.md`.
