# E-v2 event-head scale-up + pickleball fine-tune — ev2_train_20260728

Lane: `ev2_train_20260728`. Date: 2026-07-28. `VERIFIED=0` remains binding throughout —
this lane produces a candidate checkpoint and honest evidence, never a promotion. North
Star queue row 6 (`NORTH_STAR_ROADMAP.md` §5).

**Verdict: PARTIAL.** Real, non-zero, above-chance HIT-class discriminative signal was
recovered for the first time in this program's history (unlike the prior 0-TP and
7.16/s-noise failures), but the plausible firing-rate gate is not cleared at any tested
threshold, BOUNCE class shows zero learned signal at every threshold, and macro-F1
is 0.0 at the standard 0.5 threshold on both the owner-41 validation split and the
one-touch protected-50 score. **No anchor from this checkpoint may be ingested.**

VM: `pickleball-gpu-night2`, us-central1-f, A100-SXM4-40GB spot, external IP
`35.188.46.15`. Provisioned by the parent orchestrator before this lane started; this
lane did not create, stop, or delete it and will not — that remains the orchestrator's
responsibility per `runs/manager/gpu_fleet.md` fleet policy. Boot time (`uptime -s`)
2026-07-28 08:20:55 UTC; auto-poweroff rail confirmed armed for 2026-07-28T20:21:47Z
(`/run/systemd/shutdown/scheduled` USEC=1785270107181875, cross-checked against
`date -u`). Machine type `a2-highgpu-1g` (1x A100-40GB).

## 1. Step-0 verdict: PASS

Ran the fail-closed gate fresh at current `main` HEAD before any GPU work:

```
.venv/bin/python scripts/racketsport/verify_training_inputs.py \
  --inputs runs/ball_lane_20260723/ev2_unblock/training_inputs_ev2.json \
  --ledger runs/manager/data_ledger.json \
  --repo-root . \
  --gate-proof runs/lanes/ev2_train_20260728/gate_proof_STEP0_20260728.json
```

Result: exit 0, `status: PASS`, 4/4 inputs PASS, zero reasons, against
`data_ledger.json` sha256 `f09e62e1951fc81585445ab6b1d7efaedd19464a20b24749e7b54d161e831313`,
repo HEAD `f3e31606e72ac3ab601a53827361933218152700`. Proof saved at
`runs/lanes/ev2_train_20260728/gate_proof_STEP0_20260728.json`. Both historical
blockers (ledger queue-authorization `c28951b`, gate-proof assert `f29145a`)
reconfirmed landed on `main`.

## 2. Code-path findings before spending GPU time

Investigated current `main` against `SCALE_UP_SPEC.md`
(`runs/lanes/event_head_pretrain_20260716/SCALE_UP_SPEC.md`) before dispatching:

- **Lever 2 (multi-window-per-row extraction) is already landed** —
  `threed/racketsport/event_head/datasets.py:594` `manifest_windows()` is
  sliding-window (`DEFAULT_WINDOW_STRIDE=32`), not single-window-per-row. The
  single-window function (`manifest_event_centered_windows`) is now explicitly
  scoped to eval only.
- **Lever 3 (dataloader workers) is already landed** — `train_event_head.py` and
  `finetune_event_head.py` both expose `--num-workers`/`--prefetch-factor`, wired
  into a real `DataLoader`.
- **The `eval_event_head.py` window-mismatch bug is already fixed** — it derives
  `window_frames` from the checkpoint's own `config.window_frames` and raises
  `ValueError` on a disagreeing explicit request (`_resolve_window_frames`,
  `eval_event_head.py:83`). No matched-window discipline work remained to do.
- Only **Lever 1 (stage more video)** remained open, and a prior real GPU attempt
  (`event_head_corpus_20260719`, "T8") had already trained a checkpoint using
  these levers to step 9,000/118,770 before spot preemption, with
  `best_val_f1=0.0` (0 TP/0 FP at every threshold). A CPU-only follow-up review
  (`event_head_extraction_review_20260720`) concluded the extraction is
  **correct**, not buggy — the 0-detection is genuine under-training on an
  imbalanced set — and recommended resuming from that exact checkpoint
  (`last_event_head.pt`, step 9000) with weighted CE `[1,5,5]` and a model-only
  init (Adam reset), watching for first non-zero detection within +2,000 to
  +7,918 additional steps.

This lane adopted that recommendation rather than re-running pretrain from zero,
given the session's time budget: **resume the T8 checkpoint** (confirmed
byte-identical to the ledger-authorized "frozen T20" init checkpoint, sha256
`f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082`, step 9000,
`window_frames=64`, `image_size=224`).

## 3. Media staged this lane

Lane time budget did not support live-fetching all 22+10 unstaged jhong93/OpenTTGames
videos via `yt-dlp` (SCALE_UP_SPEC's own Lever-1 estimate: ~1-1.5h VM wall for that
alone). Instead:

- Synced `data/event_public_20260713/{jhong93_spot/{manifest.json,data,videos_pilot},
  openttgames/{manifest.json,markup,videos},coachai_shuttleset}` from Mac to the VM —
  the same 6 jhong93 + 2 OpenTTGames videos already locally present from the
  original 2026-07-13 acquisition (the historical "18.1% media coverage" baseline
  set), ~5.2 GB.
- **Hit the known AV1 decode wall** (`SCALE_UP_SPEC.md` §4 ops lesson: "VM
  cv2/bundled ffmpeg cannot decode AV1"): 5 of 6 jhong93 videos were AV1-encoded;
  the first training attempt hard-failed with `DatasetFormatError: decode failed
  at frame 104882`. Fixed by transcoding all 5 to h264 on the VM with the `ffmpeg`
  CLI (`libx264 -preset veryfast -crf 20`, ~19x realtime — the CLI decodes AV1
  fine; only `cv2`'s bundled backend cannot), then re-verified cv2 decode at
  start/middle/end of every file before relaunching training.
- **Data-governance correction made mid-lane.** The corpus manifest builder
  (`build_public_manifest`) unconditionally also reads `extended_openttgames/`
  (the HIT label overlay) if present. That asset's ledger entry
  (`event_public_extended_opentt_20260713`) is `state: BLOCKED` — a member of
  `TRAIN_REFUSAL_STATES` — even though its EVENT-component ruling is
  `CONDITIONAL` and arguably satisfied by this lane's use. Per this dispatch's
  explicit "exclude and note it" instruction, this lane **quarantined
  `extended_openttgames` off the VM's staging path**, rebuilt the manifest
  without it (openttgames HIT count correctly drops to 0, BOUNCE unaffected),
  and **killed and restarted the already-running training process** rather than
  let it finish on ledger-tainted data (no checkpoint from the tainted run was
  used anywhere downstream). `coachai_shuttleset` (labels only, ledger state
  also `BLOCKED`, 0 media ever present or staged) contributes
  inventory-reconciliation label counts only — the code's own
  `EXPECTED_UNIVERSE` assertion structurally requires calling its loader — and
  contributes **zero actual training windows**, since `manifest_windows()`
  always skips rows with `media_present=False`. `jhong93_spot` and base
  `openttgames` (excluding the extended HIT overlay) carry no explicit ledger
  row at all; both were already used unchallenged across two prior real GPU
  dispatches (T4 2026-07-16, T8 2026-07-19/20) for this exact pretrain purpose,
  and `.claude/skills/run-lane/SKILL.md` standing rule 4 states license is
  FYI-only under the owner's 2026-07-22 directive (`policy_directives.
  license_is_state_gate: false` in `data_ledger.json` itself) — only
  PROTOCOL-based ledger states gate, and none exists for these two.
  `online_harvest_20260706` (the owner fine-tune media) has an explicit
  `EVENT: ALLOW` ruling, `trainer_forbidden: false`.

Post-correction corpus manifest (`runs/lanes/ev2_train_20260728/dataset_manifest.json`,
built on the VM): jhong93_spot 641/3445 media-present rows, openttgames 2/12
(BOUNCE-only, HIT=0 after exclusion), shuttleset 0/104 (labels-only). Inventory
totals reconcile exactly to `EXPECTED_UNIVERSE` (33,791 / 4,271 / 36,484).
Sliding-window extraction (stride 32, window 64) over this corpus:
**train_windows=5,646, val_windows=1,390, test_windows=1,972** — a real, honest
25x multiplier over the original 226-window starved baseline, using 8 of the
~40 available public-corpus videos (this lane's time-boxed subset, not the full
~1,301-row Lever-1 target).

## 4. Pretrain resume — partial, inconclusive

```
.venv/bin/python scripts/racketsport/train_event_head.py --full \
  --manifest runs/lanes/ev2_train_20260728/dataset_manifest.json \
  --device cuda --out runs/lanes/ev2_train_20260728/train_resume \
  --weights imagenet --steps 12000 --image-size 224 --window-frames 64 \
  --batch-size 8 --lr 0.001 --val-every 750 --seed 20260716 \
  --max-wall-minutes 45 \
  --init-checkpoint-model-only runs/lanes/ev2_train_20260728/init/frozen_t20_event_head.pt \
  --class-weights 1 5 5 --stride-frames 32 --num-workers 8 --prefetch-factor 4
```

Result: `status: partial_wall_stop`, `completed_steps: 2977` of `target_steps: 12000`
(self-imposed 45-minute wall cap reached, not the step target), `steps_per_s: 1.10`.
Validation trajectory:

| step | macro_f1_at_2 | max_positive_class_probability | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 750 | 0.0 | 0.0960 | 0 | 0 | 1840 |
| 1500 | 0.0 | 0.0931 | 0 | 0 | 1840 |
| 2250 | 0.0 | 0.4354 | 0 | 0 | 1840 |

F1 stayed at 0 through every validation reached, but `max_positive_class_probability`
rose sharply between steps 1500 and 2250 (0.093 → 0.435) — the model is moving away
from pure background collapse. This is squarely within the extraction review's own
predicted pre-escape window ("+2,000 to +7,918 steps" before first non-zero
detection); this run only reached +2,977 of that range before the time-box cut it
off. **Honest characterization: encouraging trend, not a resolved result** — neither
"still collapsed" nor "escaped" can be claimed from 3 validation points.

Checkpoints: `train_resume/best_event_head.pt` (step 2250, highest probability among
tied-zero-F1 checkpoints) and `train_resume/last_event_head.pt` (step 2977, most
gradient updates) — pulled to `runs/lanes/ev2_train_20260728/train_resume/`.
`last_event_head.pt` was used as the fine-tune init (more training, metric tied at
zero either way).

**Scope cut, stated per dispatch instruction:** the full SCALE_UP_SPEC target was
12,000+ steps over the full ~1,301-row/~15k-window corpus (~2-3h A100). This lane
time-boxed pretrain to 45 minutes over an 8-video/5,646-window subset — a real,
smaller, honest slice of the spec, not the full scale-up. The rail was never
disarmed; the cut is in step/data budget, exactly as directed.

## 5. Fine-tune on owner's 102 labels — partial (wall-capped, not graceful)

First attempt hard-failed cleanly before touching training data: the fine-tune
script's own protected-frame safety check
(`_reject_protected_rows`/`SEED = runs/lanes/event_bootstrap_20260713/
spot_check_tier_a_50.json`) refused to run because that file — deliberately
`.gitignore`d (`runs/lanes/*` pattern) and never present on a fresh VM checkout —
was absent. Copied it to the VM at its canonical path (sha256
`b7f2386d91ba4564e52de1e87eb7907990f539e79e503587aeff380806ee4e60`, matches Mac's
copy exactly) — needed only for the overlap-safety check, never read as training
signal — and relaunched:

```
.venv/bin/python scripts/racketsport/finetune_event_head.py \
  --gate-proof runs/lanes/ev2_train_20260728/gate_proof_finetune_FRESH.json \
  --owner-manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
  --init-checkpoint-model-only runs/lanes/ev2_train_20260728/train_resume/last_event_head.pt \
  --out runs/lanes/ev2_train_20260728/finetune --device cuda \
  --steps 400 --image-size 224 --window-frames 64 --batch-size 8 --lr 0.0005 \
  --val-every 50 --seed 20260716 --stride-frames 32 --num-workers 4 \
  --checkpoint-selection owner-val --class-weights 1 5 5 --max-wall-minutes 15
```

This is the **legacy owner-val mode** (61 train rows, one sliding window each since
each row is already a pre-cut 64-frame clip; the 41 val rows are never trained on,
and the protected 50-row inventory is checked before any decode). Gate proof PASS
(fresh, generated immediately before use; `training_inputs_finetune.json` — a
minimal 2-input manifest covering exactly `owner_102_manifest.json` +
`data/online_harvest_20260706/rallies`, both ledger `EVENT: ALLOW`/`CONSUMED`).

The 15-minute `--max-wall-minutes` cap hit at **step 100 of the 400 target** and
raised `STAGE_F_OPTIMIZER_WALL_EXPIRED: cap reached immediately after
optimizer.step(); terminal guards are forbidden` — an exception, not a graceful
partial-save exit like the pretrain script. The periodic checkpoint save (every 50
steps) had already run, so `finetune/best_event_head_finetuned.pt` at step 100
survived on disk; only the final unconditional save and the script's continuation
into eval were lost. This lane used the step-100 checkpoint directly rather than
re-running fine-tune with a larger wall budget, given the session's own time
constraints — a second, honestly-noted scope cut.

Owner-val validation metrics recorded in the checkpoint at step 100:
`best_val_macro_f1_at_2 = 0.0`, `best_val_max_positive_class_probability = 0.0367`.

## 6. Eval discipline

All eval below used `scripts/racketsport/eval_event_head.py` unmodified, which
already asserts the eval window against the checkpoint's own `config.window_frames`
at load (`window_frames=64` throughout — confirmed, not assumed).

### 6a. Matched-window public eval, threshold sweep, 50 clips

`--mode public --max-clips 50` on the fine-tuned checkpoint (val+test split,
event-centered windows, disjoint from every training source):

| threshold | HIT TP/FP/FN | HIT precision/recall/F1 | BOUNCE TP/FP/FN | BOUNCE F1 |
|---:|---|---|---|---:|
| 0.5 | 0/0/85 | 0/0/0.0 | 0/0/49 | 0.0 |
| 0.3 | 0/1/85 | 0/0/0.0 | 0/0/49 | 0.0 |
| 0.2 | 0/1/85 | 0/0/0.0 | 0/0/49 | 0.0 |
| 0.1 | 4/4/81 | 0.5/0.047/0.086 | 0/0/49 | 0.0 |
| 0.05 | 32/81/53 | 0.283/0.376/0.323 | 0/12/49 | 0.0 |

**Read honestly.** At the standard 0.5 threshold there is no detectable signal.
Below threshold 0.2 real, non-chance HIT-class structure appears — precision 0.5 at
threshold 0.1 (4 true HIT matches, mean timing error 41.7ms, well inside a
sub-frame-scale useful window) is a first for this program's public-eval history;
every prior checkpoint scored either exactly 0 TP or (the zero-shot tennis
transfer) fired on 98% of seconds with no discrimination at all. **BOUNCE class
scored zero TP at every threshold tested, at any point in this lane** — the class
weighting `[1,5,5]` and the short fine-tune did not move it off collapse. This is
the single clearest concrete finding: whatever explains HIT's partial escape has
not yet reached BOUNCE.

### 6b. Owner-41 validation eval

`--mode owner-val --arm A --seed 20260716 --threshold 0.5`: `macro_f1_at_2 = 0.0`,
`negative_rows = 22`, `negative_false_positives = 0` (no false alarms on true
negatives — the model is conservative, not trigger-happy, at this threshold),
`timing_error_p90_frames = 64.0` (the "no matches" fallback value — there were no
TP pairs to time). Full-video firing rate on the two distinct owner-val source
videos (971.0s + 606.0s = 1,577.0s total): **0 events, 0.0/s** at threshold 0.5.

### 6c. Protected 50-row owner seed — ONE-TOUCH, eval only

Run exactly once, on the final chosen checkpoint, `--threshold 0.5`. **Had to run
locally on Mac, not the VM** — the protected label file's `source.video_path`
fields are absolute Mac paths (`/Users/arnavchokshi/Desktop/pickleball/...`) baked
in at authoring time; the VM's `/home/arnavchokshi/coldstart_20260706/repo/...`
tree does not contain those paths, and this lane did not modify the protected file
to work around that. Result: `seed_rows=50`, `typed_rows=28` (contact-typed),
`other_rows_reported_separately=1`, `negative_rows=21`,
`negative_false_positives=0` (0.0 FP rate — consistent, conservative behavior),
**0 TP across all 28 typed rows at both tolerance 1 and 2**. This one-touch score
is now consumed; it will not be re-run against this or any resumed checkpoint.

## 7. Firing-rate plausibility gate — not cleared at any tested threshold

Measured directly on the 697.4s pb.vision demo clip (`data/pbvision_11min_20260713/
source_video.mp4`, sha256 `272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383`
— the same clip the failed 2026-07-16 zero-shot transfer fired 7.16 HIT/s on),
using non-overlapping 64-frame windows covering every frame once (reusing the
harness's own `_predict`/peak-pick, `runs/lanes/ev2_train_20260728/
measure_firing_rate.py`, a lane-owned evidence generator, not a new inference
path):

| threshold | HIT | BOUNCE | total events | events/s |
|---:|---:|---:|---:|---:|
| 0.5 | 1 | 0 | 1 | 0.0014 |
| 0.1 | 131 | 0 | 131 | 0.188 |
| 0.05 | 1,372 | 169 | 1,541 | 2.210 |

**None of the three tested thresholds lands inside the required 0.3-1.0 events/s
plausible band.** Threshold 0.5 is silent; threshold 0.1 undershoots (0.188/s);
threshold 0.05 overshoots (2.21/s) — well below the failed zero-shot's 7.16/s
(not "a HIT in 98% of seconds" the way that checkpoint was), but still over the
gate. The true crossing point is very likely somewhere between 0.05 and 0.1 given
the monotonic trend, but this lane did not search for it: per §2.3's explicit
no-repeat ruling, threshold-shopping to force a fit inside the plausible band
without independent evidence that the underlying detections are correct is exactly
the failure mode already ruled out for audio anchors, and is not attempted here
for the same reason. **Gate result: NOT PASSED. Per this dispatch's explicit kill
rule, no anchor from this checkpoint may be ingested at any coverage.**

## 8. Checkpoints, cost, verdict

### Checkpoints (two-sided sha256, both pulled to `runs/lanes/ev2_train_20260728/`)

| file | role | sha256 | VM path |
|---|---|---|---|
| `train_resume/last_event_head.pt` | pretrain-resume, step 2977/12000, partial_wall_stop | `30b25a8ec00d4ebad93ff416993baafff971b567b2d22b8be05b95ed48729d5b` | `/home/arnavchokshi/coldstart_20260706/repo/runs/lanes/ev2_train_20260728/train_resume/last_event_head.pt` |
| `finetune/best_event_head_finetuned.pt` | owner fine-tune, step 100/400, wall-capped | `a3fdf12d92d3bac4b6dfa586a41d96a223428925fbcf4005f32aede3b226a69c` | `/home/arnavchokshi/coldstart_20260706/repo/runs/lanes/ev2_train_20260728/finetune/best_event_head_finetuned.pt` |

`finetune/best_event_head_finetuned.pt` is also the file uploaded to
`s3://sway-videos/pickleball-models/20260728/event_head_ev2_best.pt` (S3
`ContentLength: 358699`, matching local file size exactly). Per this dispatch's
instruction, **neither checkpoint is committed to git** (small metrics/report
files only); this REPORT.md records both shas and VM paths as the durable
reference, alongside the S3 copy of the fine-tuned checkpoint.

**Proposed `models/MANIFEST.json` PENDING entry** (not applied — this lane does
not own that file; text provided for the integration owner per T8-lane
precedent):

```json
{
  "id": "event_head_ev2_20260728",
  "stage": "event_detection_head",
  "use": "HIT/BOUNCE event-head candidate, owner-fine-tuned; PENDING, not promoted",
  "source": "runs/lanes/ev2_train_20260728/REPORT.md",
  "license": "RD_ONLY (public pretrain pixels) + OWNER_REVIEWED_INTERNAL (fine-tune)",
  "commercial_posture": "research_only_do_not_promote",
  "status": "PENDING",
  "local_path": "runs/lanes/ev2_train_20260728/finetune/best_event_head_finetuned.pt",
  "s3_path": "s3://sway-videos/pickleball-models/20260728/event_head_ev2_best.pt",
  "sha256": "a3fdf12d92d3bac4b6dfa586a41d96a223428925fbcf4005f32aede3b226a69c",
  "init_checkpoint_sha256": "f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082",
  "notes": [
    "Resumed from T8's step-9000 pretrain checkpoint (= ledger frozen_t20_event_head.pt), 2,977 additional pretrain steps, then 100 owner fine-tune steps, both time-boxed by this lane's session budget, not the full SCALE_UP_SPEC target.",
    "Owner-41 macro-F1@2f = 0.0 at threshold 0.5; protected-50 one-touch 0 TP at threshold 0.5; public 50-clip sweep shows real HIT-class signal only at threshold <=0.1 (BOUNCE: 0 TP at every threshold).",
    "Firing-rate plausibility gate (0.3-1.0/s) NOT cleared at thresholds 0.5/0.1/0.05 on the 697s pb.vision demo (0.0014 / 0.188 / 2.21 events/s respectively).",
    "DO NOT INGEST any anchor from this checkpoint. Candidate for a longer, non-wall-capped resume, not a promotion."
  ]
}
```

### Cost estimate

Wall-clock from first active use of the VM (staging + transcode + train + fine-tune
+ eval, 2026-07-28 08:40 UTC) through final eval completion (10:48 UTC) ≈ **2.13
hours** of this lane's attributable usage at the fleet-ledger rate of ~$1.93/hr
(`runs/manager/gpu_fleet.md`, "2026-07-28T01:2x PDT" entry) ≈ **~$4.1**. This is
*not* the VM's full billing lifetime — `pickleball-gpu-night2` was provisioned by
the parent orchestrator before this lane started and remains running/billing under
its own rail regardless of this lane's usage window; this lane neither created nor
will stop it, per explicit dispatch instruction. Breakdown: staging/transcode
~25min, pretrain resume 45min (wall-capped), fine-tune attempts ~17min
(wall-capped), eval sweeps (public + owner-val + firing-rate) ~26min.

### Verdict: PARTIAL

- **Step-0 gate: PASS.** No data-safety blocker.
- **Both historical blockers (staging AV1 wall, gate-proof assert bug) reconfirmed
  fixed on `main`**, plus one **newly found and fixed mid-lane**: the
  `extended_openttgames` ledger-BLOCKED asset would have silently entered the
  corpus if not caught (the manifest builder reads it unconditionally when
  present) — corrected before any checkpoint trained on the tainted manifest was
  used downstream.
- **Real, above-chance discriminative signal recovered for the HIT class** — a
  genuine first for this program (every prior checkpoint scored either exactly
  0 TP everywhere, or fired near-uniformly with no discrimination). This is a
  meaningful, honest advance over the 2026-07-16/2026-07-19 failure states.
- **BOUNCE class remains fully unlearned** at every threshold tested, on every
  eval surface (public, owner-val, protected-seed).
- **Firing-rate plausibility gate (0.3-1.0/s) is not cleared** at any of the three
  tested thresholds on the 697s pb.vision demo.
- **Owner-41 and protected-50 macro-F1 are both 0.0** at the standard threshold.
- Both the pretrain resume and the fine-tune were **time-boxed by this session's
  own budget, not by the VM's rail** (2,977/12,000 pretrain steps; 100/400
  fine-tune steps) — this is an honestly-reduced-scale result, not a completed
  scale-up.

**What remains before any anchor from this head may be ingested:**
1. A longer, properly-budgeted resume (ideally on a dedicated multi-hour dispatch,
   not wall-capped by a single interactive session) to reach or exceed the
   extraction review's own +7,918-step pre-escape ceiling, watching specifically
   for BOUNCE to leave collapse (it has not yet moved at all).
2. A fine-tune run to its full step target (400, or longer) without a 15-minute
   cap, so `checkpoint-selection owner-val` can actually select a best-by-F1
   checkpoint rather than whatever the wall interrupted mid-epoch.
3. A repeat of this exact eval battery (matched-window public sweep, owner-41,
   firing-rate sweep) on that better-trained checkpoint, searching for a single
   threshold that clears the 0.3-1.0/s band on independent evidence, not by
   threshold-shopping against the plausibility number alone.
4. Only after (1)-(3): typed-anchor ingestion, agreement filtering against the
   pb.vision teacher signal, and re-evaluation against the still-sealed protected
   50-row seed is **not** available again — its one touch is spent for this
   checkpoint lineage; a materially different (not just longer-trained) checkpoint
   would need a fresh registration before touching it again.

`VERIFIED=0` remains binding. `G_val` improvement relative to the last honest
public-eval baseline (0.3631 F1@±2f from the 2026-07-16 T4 checkpoint, itself
measured on a different, tiny corpus under a different protocol) is **not
comparable** — this lane's numbers are the first same-protocol matched-window
figures on the fixed harness and should be treated as the new baseline going
forward, not compared numerically to pre-fix figures per §2.2's own numbers-across-
protocols caveat.
