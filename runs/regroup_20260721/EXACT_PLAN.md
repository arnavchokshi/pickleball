# EXACT PLAN — five-day data-adjudication sprint

Date: 2026-07-21  
Status: `VERIFIED=0`  
Authority: `NORTH_STAR_ROADMAP.md` remains the only product/current-truth/future-plan authority. This is a dated execution proposal, not a second roadmap and not a promotion record.

## Executive decision

Do **not** launch five model-training lanes tomorrow morning. The evidence does not support that.

The correct change in kind is a data-adjudication sprint:

1. recover and close the already-paid EVENT A/B/C experiment;
2. create source-held human judges before fitting BALL, PERSON, or ReID;
3. permit GPU work only after a named data gate passes;
4. run one matched data intervention per component, with architecture, updates, thresholds, and scorer frozen;
5. end every day with a number or a named negative; and
6. record every acquired asset as consumed, blocked, quarantined, rejected, or deliberately deferred.

Only EVENT is currently close to training/evaluation ready. BALL has a defensible next experiment after the staged 350-frame scratch audit is reviewed. COURT has a defensible *preview-only* diversity reopen after the staged frames are actually exported. PERSON and ReID first need a new independent human judge; forcing either onto a GPU before that would reproduce the exact evaluation failure this regroup is meant to stop.

The five trainable components in this plan are: **BALL detector, event head, court keypoint net, person detector, and ReID**. “Calibration/eval harnesses” in `gap_matrix.json` are not a trainable component. PADDLE is outside this five-component regroup and remains governed by its existing negative evidence.

Throughout this document, `P(fail)` means probability of failing the stated acceptance check, not probability that the code can be made to run. A failed prerequisite produces `NO_ATTEMPT`; a completed candidate that misses its check produces `NO_LIFT` or `HARM`. Neither is a promotion.

## 1. First correct the evidence pack

The gap matrix is useful as a lead list, but several of its largest claims are stale or over-counted. Planning from those claims literally would repeat completed experiments.

| Pack premise | Fresh tree evidence | Planning consequence |
|---|---|---|
| “80,967 Roboflow pickleball images used by nothing.” | `80,967` is the raw pre-dedup core count. The combined retained index has 61,260 core+adjacent samples; retained core is 45,575. BALL has 34,658 retained core images. A 12,000-step Roboflow BALL pretrain already ran: public internal-val F1@20 rose `0.0615 -> 0.6104`, but the public-only checkpoint then **regressed** on reviewed real rows versus the official tennis control: `0.2971 vs 0.3611`. See `runs/lanes/w3_p11_train_20260707/report.md` and `runs/lanes/w7_ballretrain_20260709/REPORT.md`. | Do not rerun Roboflow BALL pretraining or call it a never-queued cell. The next BALL intervention is in-domain pb.vision teacher pixels, judged on newly scratch-labeled, source-held human rows. |
| “Court is still synthetic-only; real Roboflow was never trained.” | `runs/lanes/court_train4_20260710/REPORT.md` records two real-transfer arms on 3,921 rows/9 datasets. Both learned their train families (`~2.8px`, PCK@5 `0.758`) and failed every source-disjoint card (`833–937px`, PCK@5 `0.0`). | Do not rerun the same real corpus/init recipe. A scoped reopen is justified only by *new source diversity*: the 27 non-protected source videos in the staged court pack, not more steps or another initialization. |
| “106 reviewed court frames are ready.” | The 100-frame diversity package was imported as tasks 88–91, but no reviewed export exists. The older six-frame package has only three full usable rows; three are explicitly rejected. Three of the new 100 frames are from `IYnbdRs1Jdk`, the same original source family as strict protected `outdoor_webcam_iynbd`; those three are permanently denied. | Court starts with `USABLE_REVIEWED_ROWS=n`, never with 106. Maximum potential from the new pack is 97 rows/27 sources before annotation-quality failures, plus three older usable audit rows. |
| “pb.vision loader diff is unapplied; 0 windows usable.” | Commit `e3f47d651` landed the A/B/C loader, UNKNOWN mask, SHA chain, owner-val evaluator, causal gate, and one-touch protected guard. Media/output recovery is now the operational blocker. | Finish the registered experiment; do not build a second loader or redesign the event architecture. |
| “pb.vision is RD_ONLY for training.” | `NORTH_STAR_ROADMAP.md` §2.3 now records owner-signed full training and commercial usage rights, superseding the July-19 legal restriction. Teacher-quality and compare-holdout restrictions remain. | Update the data ruling, not the discipline: pixels may train; predictions remain teacher-only, and three videos remain permanent compare-only. |
| Owner event fields are `46 coords / 57 dt`. | The live owner manifest records all 60 typed rows with x/y/dt and marks 46/57 as stale bookkeeping. The frozen split is 61 train / 41 validation with zero protected overlap. | Use the live manifest, never the narrative count. |
| A/B/C currently has a verdict. | The user reports Arm A owner-val F1 `0.0` and B/C training. Locally, no `seed_20260720/` artifacts or B/C verdict are present; the last local fleet entry says A completed on the VM, B/C had not started when auth died. Live cloud refresh was unavailable. | Treat A=`0.0` as **user/live-reported, not locally verified**. Recover and hash artifacts before treating any arm as evidence. One seed is directional only; the registered verdict requires 3 arms × 3 seeds. |

This changes the shortlist. The highest-value idle assets are not “all 80,967 Roboflow images.” They are:

- the already-materialized EVENT A/B/C inputs and any VM-only outputs;
- the 350 no-prelabel BALL scratch frames, currently staged but not exported;
- the ten allowed pb.vision videos and their dense BALL/event/player/court teacher signals;
- the 97 potentially usable, source-diverse court frames after protected-source denial;
- 15,312 retained core-pickleball person images / 47,044 boxes from 14 CC BY 4.0 Roboflow sources after excluding adjacent-sport rows and the NC source; and
- the three permanent pb.vision compare videos as a shared human-labeled PERSON/ReID judge.

## 2. Binding rulings and quarantines

These are prerequisites, not suggestions.

### 2.1 pb.vision ruling update

Adopt this exact ruling:

> `PBV-FULL-USAGE-20260720`: pixels from the ten non-holdout pb.vision source videos may be used for training and commercial-bound experiments under the owner-signed grant. `cv_export.json` and `insights.json` remain model-teacher outputs, never human ground truth. `83gyqyc10y8f`, `iottnc0h3ekn`, and `o4dee9dn0ccr`, plus every derivative, remain permanent compare-only holdouts. A model trained on a pb.vision source may not make a head-to-head claim on that source.

For EVENT, the current materializer needs one quality correction before its B/C result can be trusted: an audio-only match may not make a row eligible. Require a non-audio physical cue (`ball_velocity_kink` now; a typed wrist cue when available). Audio may add weight only after its observed per-video match rate beats the preregistered time-shift null. This reconciles the code with the standing “untyped audio alone never decides” kill.

For BALL, a pb.vision label is eligible only when the high-confidence 2D teacher location independently agrees with frozen WASB or a preregistered temporal/geometry check. It remains `teacher_derived=true`, receives low weight, and cannot serve as evaluation truth.

### 2.2 Permanent evaluation quarantines

- `indoor_doubles_fwuks`: strict protected; no training, tuning, threshold selection, or repeated use.
- `outdoor_webcam_iynbd`: strict protected/historical; no new threshold shopping. Deny all `IYnbdRs1Jdk` derivatives, including the three court-package frames.
- Burlington/Wolverine person and BALL labels: never training. They remain historical/internal cards, not fresh promotion proof.
- EVENT protected 50: never training; exactly one frozen touch only after the executable owner-41 A/B/C gate passes.
- pb.vision compare-only IDs: `83gyqyc10y8f`, `iottnc0h3ekn`, `o4dee9dn0ccr`.
- Roboflow `testing-esifc/pickle-ball-labeling-mff1d`: exclude because it is BY-NC-SA 4.0. Exclude all 15,469 adjacent-sport person images from the product-domain arm; that bucket is dominated by tennis and is not the intervention being tested.
- Any model prediction or accepted prelabel is teacher/pseudo supervision, not independent evaluation truth.
- Split by original video/game/session/channel family. Never split adjacent frames or Roboflow train/val folders randomly.

### 2.3 Scoped standing-rule reopens

- **BALL:** no standing kill is reopened for Roboflow; that experiment already ran and transferred negatively. A new pb.vision-teacher A/B is allowed because it uses signed, in-domain pixels and a new source-held scratch judge.
- **COURT:** learned court authority remains killed. One preview-only, source-diversity challenger is allowed if the reviewed 97-row pack yields at least 60 train rows across at least 15 train source groups and all eight frozen holdout groups survive. This is materially different from the failed seven-family/3,921-row volume experiment. It cannot change v1 authority or the static-lock requirement.
- **PERSON:** the July-16 fine-tune park is reopened for a diagnostic only because the exact retained in-domain box pool is now quantified and this plan creates a new source-disjoint human judge. No default change can follow from the old four protected clips.
- **ReID:** association-only sweeps remain killed. A data-leverage ReID arm is allowed only if pb.vision avatar-to-box assignments pass a manual label-precision gate before training.

## 3. Exact per-component sequence

### 3.1 EVENT head — finish the causal experiment already in flight

#### E0 — recover, hash, and method-audit the current seed

**Asset and command-level action**

Recover the durable VM disk before any new training:

```bash
gcloud auth login
gcloud compute instances list \
  --filter='name=pickleball-gpu-abc' \
  --format='table(name,zone.basename(),status,lastStartTimestamp)'
# If TERMINATED, start the same durable disk; do not create a replacement first.
gcloud compute instances start pickleball-gpu-abc --zone=us-central1-a
gcloud compute ssh pickleball-gpu-abc --zone=us-central1-a --command='\
  find /home/arnavchokshi/pickleball/runs/lanes/ball_event_abc_20260720 \
  -maxdepth 4 -type f -print | sort'
gcloud compute scp --recurse --zone=us-central1-a \
  pickleball-gpu-abc:/home/arnavchokshi/pickleball/runs/lanes/ball_event_abc_20260720/seed_20260720 \
  runs/lanes/abc_experiment_20260721/vm_pull/
```

Recompute every input/checkpoint/manifest SHA and require `completed_steps == target_steps == 1000`. Audit `agreement_decisions.jsonl` by agreement family. Exact code change if any accepted row is audio-only: change `scripts/racketsport/build_abc_arm_manifests.py` so `accepted` requires `ball_velocity_kink` (or a future typed wrist cue); add an audio time-shift-null field and tests; audio alone stays weight zero.

**Expected measurable effect:** no model lift; recovery converts a VM-only claim into durable evidence and quantifies `accepted_rows`, `audio_only_rows`, `non_audio_rows`, and B/C status.

**CHECK:** end with exactly one of `RECOVERED_HASHED`, `A_ARTIFACT_UNRECOVERABLE`, or `METHOD_INVALID_AUDIO_ONLY=n`. Do not score a method-invalid B/C as causal evidence.

**Quarantine:** protected 50 absent; three pb.vision compare IDs absent; owner 41 read only by the evaluator, never the trainer.

**Effort / GPU:** 1–2 operator hours; $0 new GPU if disk is recoverable.

**P(fail): 30%.** Failure teaches whether the dominant problem was artifact durability/ops rather than learning. If A is unrecoverable, rerun A only from the frozen hashes; do not reconstruct a favorable checkpoint.

#### E1 — one-seed directional A/B/C, then early stop or replicate

**Asset and command-level action**

Use the exact frozen commands in `runs/lanes/finetune_contract_repair_20260720/ABC_READY.md`. For B, for example:

```bash
.venv/bin/python scripts/racketsport/finetune_event_head.py \
  --owner-manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
  --pseudo-manifest runs/lanes/ball_event_abc_20260720/inputs/pbvision_filtered_teacher_manifest.json \
  --init-checkpoint-model-only runs/lanes/ball_event_abc_20260720/inputs/frozen_t20_event_head.pt \
  --out runs/lanes/ball_event_abc_20260720/seed_20260720/B_pbvision_teacher \
  --device cuda --steps 1000 --val-every 100 --batch-size 8 \
  --lr 0.001 --image-size 224 --window-frames 64 --stride-frames 32 \
  --num-workers 4 --class-weights 1.0 5.0 5.0 \
  --pseudo-weight-cap 1.0 --seed 20260720 --max-wall-minutes 90
```

A omits `--pseudo-manifest`; C uses `pbvision_placebo_manifest.json`. Score all three on owner-41 with `eval_event_head.py --mode owner-val` at the frozen threshold.

**Expected measurable effect:** B should exceed A by at least `+0.10` macro-F1 at +/-2 frames and exceed C if teacher timing contributes causal signal rather than mere pixel exposure.

**CHECK:** seed 20260720 is an early-stop screen: continue only if `B-A >= +0.10`, `B>C`, B negative FP `<=2/22` and `<=A+1`, B timing p90 is non-worse, and event rate is `0.3–1.0/s`. Otherwise record `EVENT_PBV_SEED1_NO_LIFT` and do not buy eight more arms. This early stop is a negative, not the registered three-seed verdict.

**Effort / GPU:** 2–4 operator hours; at most 3 A100 arm-runs for the first screen, `<=4.5 GPU-h`, approximately `$5–7` at the recorded A100 spot band if all three must be rerun.

**P(fail): 80%.** A user-reported F1 of 0.0 signals a weak initialization/data regime, and prior cross-sport transfer failed. Failure means agreement-filtered pb.vision timing did not produce usable causal signal at this scale; stop adding teacher datasets and do not request more owner EVENT labels yet.

#### E2 — registered three-seed causal gate, protected only on PASS

**Asset and command-level action**

Only after E1 passes, run seeds `20260721` and `20260722` for A/B/C with the identical 1,000-step contract, then:

```bash
.venv/bin/python scripts/racketsport/abc_decision_gate.py \
  --arm-a "20260720=$A20" "20260721=$A21" "20260722=$A22" \
  --arm-b "20260720=$B20" "20260721=$B21" "20260722=$B22" \
  --arm-c "20260720=$C20" "20260721=$C21" "20260722=$C22" \
  --out "$ABC_OUT/owner41_abc_gate.json"
```

**Expected measurable effect:** replicated median B-A macro-F1 `>=+0.10` with a paired-bootstrap 95% lower bound above zero.

**CHECK:** the executable gate additionally requires every seed nonnegative, B>C, per-class regression `<=0.03`, negative-FP/timing/rate guards, and exact step parity. Only `PASS` allows the existing atomic one-touch wrapper to open the protected 50. Any other verdict permanently closes this frozen experiment.

**Quarantine:** the protected scorer cannot choose arm, seed, checkpoint, threshold, or code. Its one-touch token remains permanent after success or failure.

**Effort / GPU:** incremental maximum 6 arm-runs × 90 minutes = `<=9 A100 GPU-h`, approximately `$10–14`.

**P(fail conditional on E1 pass): 55%.** Failure shows the seed-one lift was unstable or failed a safety dimension. It does not justify threshold tuning or a fourth seed.

#### E3 — close the seven “unused public event datasets” honestly

**Asset and command-level action**

Add `--inventory-only --require-window-frames 64` to `build_public_manifest.py` and emit per-dataset `label_rows / media_files / decodable_windows / semantic_mapping / rights` without training.

Expected disposition to verify, not assume:

- F3Set, GolfDB, PadelTracker100, and label-only ShuttleSet: `BLOCKED_NO_PIXELS` unless a local 64-frame context resolves;
- TT Sounds: `REJECTED_AUDIO_ONLY` for typed event training;
- squash Figshare and shuttlecock Zenodo: `NO_STRUCTURED_EVENTS` if the current zero-row inventory reproduces; and
- Extended OpenTTGames: queue only if its events map to the two local videos with source-disjoint 64-frame windows and compatible HIT/BOUNCE semantics.

**Expected measurable effect:** convert seven misleading “never queued” cells into exact usable/blocked/rejected counts; no accuracy claim.

**CHECK:** every dataset ends with a nonzero decodable-window count and semantic map, or a named negative above. No GPU dispatch from labels alone.

**Effort / GPU:** 3–5 engineer hours; CPU only, $0 GPU.

**P(fail as a trainable-data gate): 85%.** Failure teaches that downloaded labels are not a training corpus. Do not spend another day writing adapters for absent pixels or incompatible sports.

### 3.2 BALL detector — new judge first, then human-only vs pb.vision data

#### B0 — close the Roboflow repeat and build the source-held scratch judge

**Asset and command-level action**

Do not rerun `configs/racketsport/ball_pretrain_roboflow_wasb.json`. Finish/export the existing no-prelabel task `w7_audit_stratum_uniform350`, then add one deterministic splitter:

```bash
.venv/bin/python scripts/racketsport/build_ball_regroup_split.py \
  --reviewed-root runs/lanes/w7_ballingest4_20260709/reviewed_corpus \
  --scratch-package cvat_upload/w7_audit_stratum_20260709/package_manifest.json \
  --scratch-export cvat_upload/exports/w7_audit_stratum_20260709/w7_audit_stratum_uniform350_annotations.zip \
  --holdout-source HyUqT7zFiwk --holdout-source Ezz6HDNHlnk \
  --out runs/lanes/ball_data_regroup_<date>
```

The new command must compare every final label to its original prelabel/package lineage and emit `scratch`, `corrected_prelabel`, or `confirmed_prelabel`. It must also correct the existing “LoSO” semantics from per-clip to parent-source grouping.

Expected exact split before annotation failures:

- validation: 167 scratch rows, `HyUqT7zFiwk=100` indoor court-level and `Ezz6HDNHlnk=67` outdoor night/fenced;
- training: all 3,026 reviewed rows except every row from those two source videos (`2,066` old rows), plus the remaining 183 scratch rows from four sources; and
- no row from the four protected eval clips.

Confirmed-prelabel rows may train only as explicitly low-weight teacher rows; they may never evaluate. Scratch/corrected rows carry human authority.

**Expected measurable effect:** replace the contaminated blended score with official-control F1@20, recall, precision, and hidden FP on 167 source-held scratch frames spanning the failed Indoor-like and outdoor-night domains.

**CHECK:** `350/350` images reconciled; train/val source intersection is empty; protected collision count is zero against **every** protected frame rather than the old 35-frame sample; every evaluation row is `scratch`; and metrics are reported separately for HyU and Ezz. Otherwise `BALL_NO_CLEAN_JUDGE` and no GPU.

**Effort / GPU:** 4–6 engineer hours plus 45–75 minutes owner/independent-reviewer labeling; CPU only.

**P(fail): 25%.** Failure teaches that the staged scratch package was not completed or cannot produce a source-held judge. That blocks model comparison; it is not permission to reuse the contaminated 3,026-row score.

#### B1 — materialize low-weight pb.vision BALL supervision

**Asset and command-level action**

Add `build_pbvision_ball_sst.py` targeting the existing `train_ball_stage2.py --sst-manifest` schema. Use only the seven frozen train IDs:

`143sf3gdwxsa`, `98z43hspqz13`, `bewqc0glhgpq`, `st0epgnab7dr`, `td2szayjwtrj`, `tqjlrcntpjvt`, `xkadsq9bli3h`.

Keep `pldtjpw3h0jw` and `utasf5hnozwz` internal teacher-validation only, `0tmdeghtfvjx` internal teacher-test only, and the three permanent compare IDs unread.

```bash
.venv/bin/python scripts/racketsport/build_pbvision_ball_sst.py \
  --gallery-root data/pbvision_gallery_20260719 \
  --media-root "$PBV_MEDIA_ROOT" \
  --split-manifest runs/lanes/pbv_pickleball_corpus_20260720/manifest.json \
  --wasb-checkpoint models/checkpoints/wasb/wasb_tennis_best.pth.tar \
  --teacher-confidence-min 0.90 --agreement-radius-px 20 \
  --pseudo-weight 0.25 \
  --out runs/lanes/ball_data_regroup_<date>/pbv_ball_sst.json
```

The builder must SHA-bind media and PTS, emit positives only, never treat teacher absence as negative, and require either frozen-WASB spatial agreement or the preregistered temporal/geometry agreement. It must preserve `teacher_derived=true` and `ground_truth=false`.

**Expected measurable effect:** at least 1,000 accepted positive windows across at least five of seven train videos, adding in-domain venue/camera diversity absent from the six-source human corpus.

**CHECK:** accepted count `>=1,000`; accepted source count `>=5`; zero permanent-holdout rows; every window decodes; each row contains the agreement reason and exact dependency hashes. Otherwise record `PBV_BALL_INSUFFICIENT_AGREEMENT` and do not train B.

**Effort / GPU:** 5–8 engineer hours; 0.5–1 GPU-hour for bounded WASB inference, approximately `$0.30–4.25` on the recorded H100 band.

**P(fail): 45%.** Failure means high-confidence pb.vision labels do not independently agree often enough to be safe pseudo supervision; do not lower the confidence/radius after seeing the count.

#### B2 — matched A/B with equal human exposure

**Asset and command-level action**

Add `--sst-batch-size` and `--sst-loss-cap` to `train_ball_stage2.py` so B receives the exact same eight human rows/step as A and pseudo loss cannot exceed 25% of human loss. Keep architecture, official tennis initialization, steps, preprocessing, threshold, and seed identical. Disable the prior occlusion augmentation because its hFP regression was never isolated.

```bash
# A: source-held human control
.venv/bin/python scripts/racketsport/train_ball_stage2.py \
  --cvat-export-root "$BALL_TRAIN_ROOT" \
  --init-checkpoint models/checkpoints/wasb/wasb_tennis_best.pth.tar \
  --out-dir "$OUT/A_human_only" --model-family wasb_hrnet \
  --wasb-repo third_party/WASB-SBDT --device cuda \
  --steps 2372 --batch-size 8 --frames-in 3 --output-channels 3 \
  --occluded-prob 0 --seed 20260721

# B: identical human batches + bounded pb.vision pseudo loss
.venv/bin/python scripts/racketsport/train_ball_stage2.py \
  --cvat-export-root "$BALL_TRAIN_ROOT" \
  --sst-manifest "$PBV_BALL_SST" --sst-batch-size 8 --sst-loss-cap 0.25 \
  --init-checkpoint models/checkpoints/wasb/wasb_tennis_best.pth.tar \
  --out-dir "$OUT/B_human_plus_pbv" --model-family wasb_hrnet \
  --wasb-repo third_party/WASB-SBDT --device cuda \
  --steps 2372 --batch-size 8 --frames-in 3 --output-channels 3 \
  --occluded-prob 0 --seed 20260721
```

Run `run_wasb_ball.py` at the frozen visible threshold on every HyU/Ezz scratch frame and score with a new parent-source mode in `ball_loso_validation.py`.

**Expected measurable effect:** A should beat the untouched official control by `>=+0.05` pooled F1@20; B should beat A by `>=+0.03`, reflecting data-domain lift rather than more human exposure.

**CHECK:** for B over A, paired-bootstrap 95% lower bound `>0`, both source deltas nonnegative, pooled F1@20 `>=+0.03`, and hidden FP no more than `A+0.02` absolute. Seed one is the early-stop screen; only a pass gets seeds 20260720 and 20260722. Protected Indoor/Outdoor remain unopened.

**Effort / GPU:** seed-one A/B about 1.5 H100 GPU-hours from the measured prior throughput; three seeds plus scoring capped at 5 H100 GPU-hours, approximately `$2.85–21.25`.

**P(fail A vs official): 65%.** Failure means the four-source reviewed corpus still does not generalize to held source videos. **P(fail B vs A): 70%.** Failure means pb.vision teacher diversity does not transfer safely to the human judge. Either failure stops same-recipe BALL fitting; it does not reopen Roboflow pretraining or detector voting.

### 3.3 COURT keypoint net — source diversity or no attempt

#### C0 — make the staged 100-frame package truthful

**Asset and command-level action**

Finish and export CVAT tasks 88–91. Add an image-task adapter that emits the trainer’s existing `<source>/labels/court_keypoints.json` format:

```bash
.venv/bin/python scripts/racketsport/ingest_cvat_court_images.py \
  --package-manifest cvat_upload/court_diversity_20260712/package_manifest.json \
  --cvat-export cvat_upload/exports/court_diversity_20260712/annotations.zip \
  --deny-source IYnbdRs1Jdk \
  --protected-root eval_clips/ball \
  --out runs/lanes/court_data_regroup_<date>/real_court_diversity
```

Freeze this pre-label-inspection holdout set (27 expected frames):

- outdoor: `1or-bXVM80M`, `4qSoA-jwpVM`, `C5YUQlqZqBY`, `q3575jnmjJQ`;
- indoor: `A9H6EWfXht0`, `Se7M6ZKaC4Y`, `a_HzWrwK6vM`, `wv3aPJrDwK4`.

The other 19 source IDs are train candidates (70 expected frames). The three older fully usable `court_keypoints_20260707` rows stay external audit only. Group by original source and channel/venue family; no frame-random split.

**Expected measurable effect:** turn “100 staged” into exact `reviewed / usable / protected-denied / train / holdout / rejected` counts.

**CHECK:** deny exactly the three `IYnbdRs1Jdk` frames before label read; dense pHash every remaining image against every protected frame; require `>=60` usable train rows across `>=15` train source groups and at least two valid rows in all eight holdout groups. Otherwise record `COURT_DIVERSITY_ROWS_INSUFFICIENT` and spend $0 GPU.

**Effort / GPU:** owner/independent reviewer 45–60 minutes; adapter and audit 6–10 engineer hours; CPU only.

**P(fail): 35%.** Failure teaches that the pack was staged rather than completed or is too thin after quarantine. It does not justify training on the three older rows or protected derivatives.

#### C1 — one human-diversity preview challenger

**Asset and command-level action**

Explicitly supersede the July-19 “all tasks 88–91 are eval-only” posture with the frozen 19-train/8-holdout source partition above. Add `--real-source-balanced` to the trainer so every source, not every row, has equal sampling probability.

```bash
.venv/bin/python scripts/racketsport/train_court_model_v2.py \
  --out runs/lanes/court_train_regroup_<date>/H_human_diversity \
  --epochs 18 --steps-per-epoch 100 --batch-size 32 \
  --real-batch-size 32 --real-root "$COURT_REAL_ROOT" \
  --real-split-proposal "$COURT_SOURCE_SPLIT" --real-source-balanced \
  --init-from-checkpoint models/checkpoints/court_unet_v2/court_model_v2.pt \
  --real-weight 0.65 --synthetic-weight 0.35 --real-photometric-aug \
  --synthetic-workers 0 --amp --device cuda --seed 20260721
```

Score frozen control and H once on the eight heldout sources plus the three older audit rows:

```bash
.venv/bin/python scripts/racketsport/evaluate_court_model_v2.py \
  --checkpoint "$CHECKPOINT" --real-root "$COURT_HOLDOUT_ROOT" \
  --out "$OUT/court_keypoint_metrics.json" --device cuda
```

Also run `evaluate_court_finding_technologies.py` on the same root so the learned preview is compared with the existing classical line/front-end family, not only another neural checkpoint.

**Expected measurable effect:** source diversity should move heldout median below 25px and PCK@5 at least `+0.30` absolute over the frozen synthetic control—the previously frozen R2 kill bar.

**CHECK:** same kill bar, no moving target: heldout median `<25px`, PCK@5 delta `>=+0.30`, and every heldout source median improves by at least 25% without a source regression. One seed screens; only a pass gets two replication seeds. Product gate remains PCK@5 `>=0.95` per viewpoint. Even a pass remains `preview`, not authority.

**Effort / GPU:** first seed about 1 H100 GPU-hour based on the prior 1,800-step run; full replication capped at 3 H100 GPU-hours, approximately `$1.71–12.75`.

**P(fail): 80%.** Failure means sparse still-frame source diversity is insufficient for shadows/lookalikes/camera transfer. Re-close learned training and put COURT effort into static lock, line evidence, and capture discipline—not another architecture/init sweep.

#### C2 — conditional pb.vision venue-teacher ablation

**Asset and command-level action**

Run only if C1 passes. Add `ingest_pbvision_court_teacher.py` to emit one static-aggregated calibration row per allowed pb.vision venue. Accept a venue only when its teacher keypoints agree with the frozen classical line evidence at median `<=5px`. Add a separate `--teacher-real-root` and `--teacher-loss-cap 0.10`; never mix teacher rows into validation.

```bash
.venv/bin/python scripts/racketsport/ingest_pbvision_court_teacher.py \
  --gallery-root data/pbvision_gallery_20260719 \
  --exclude-video 83gyqyc10y8f --exclude-video iottnc0h3ekn --exclude-video o4dee9dn0ccr \
  --require-line-agreement-px 5 --out "$PBV_COURT_TEACHER_ROOT"
```

Train H+T with C1’s exact initialization, human batches, steps, and seed; add only `--teacher-real-root "$PBV_COURT_TEACHER_ROOT" --teacher-loss-cap 0.10`.

**Expected measurable effect:** H+T improves heldout human PCK@5 by `>=+0.05` over H through venue/camera diversity.

**CHECK:** every heldout-source delta nonnegative and pooled PCK@5 `>=H+0.05`; otherwise `COURT_PBV_TEACHER_NO_LIFT`. pb.vision comparisons remain teacher-agreement diagnostics, never accuracy proof.

**Effort / GPU:** 4–6 engineer hours; one seed about 1 H100 GPU-hour, replication only on pass.

**P(fail conditional on C1 pass): 85%.** Failure means the 12-venue court output is too sparse/noisy to improve a human-diverse model. Preserve it as a comparison artifact, not training truth.

### 3.4 PERSON detector — public-box diagnostic with a new human judge

#### P0 — build one shared PERSON/ReID human compare card

**Asset and command-level action**

Use only the three permanent pb.vision compare videos. Stage their exact MP4s, then add a scratch-only pack builder that samples 40 frames/video: 20 uniform, 10 far-player, 5 spectator/off-court, and 5 empty/sparse candidates. The page shows no model proposals. Reviewer labels every visible person as `on_court_player` or `off_court_person`, confirms zero-person frames, and assigns a stable within-video player ID to on-court players.

```bash
.venv/bin/python scripts/racketsport/build_pbvision_person_holdout_pack.py \
  --video 83gyqyc10y8f="$PBV_MEDIA/83gyqyc10y8f.mp4" \
  --video iottnc0h3ekn="$PBV_MEDIA/iottnc0h3ekn.mp4" \
  --video o4dee9dn0ccr="$PBV_MEDIA/o4dee9dn0ccr.mp4" \
  --frames-per-video 40 --uniform 20 --far 10 --off-court 5 --empty 5 \
  --scratch-only --seed 20260721 --out runs/lanes/person_reid_holdout_<date>
```

**Expected measurable effect:** one 120-frame, three-video, source-disjoint human card supplies person AP/recall/FP and ReID identity retrieval without touching the old four protected clips.

**CHECK:** `120/120` frames reviewed; every frame explicitly complete or rejected; all visible persons typed; player IDs are temporally consistent; no training lane can read this directory; content/hashes match the three named videos. Otherwise `PERSON_REID_NO_FRESH_JUDGE` and both product-domain GPU arms stop.

**Effort / GPU:** 4–6 engineer hours plus 2–3 reviewer hours; CPU only.

**P(fail): 30%.** Failure is a media/review completeness negative. It means PERSON/ReID can still run corpus diagnostics, but cannot make a product-domain lift claim this week.

#### P1 — audit and export the actually eligible Roboflow person subset

**Asset and command-level action**

Add a Roboflow-index exporter; do not weaken the existing CVAT exporter’s protected-data refusal.

```bash
.venv/bin/python scripts/racketsport/export_roboflow_person_yolo_dataset.py \
  --index data/roboflow_universe_20260706/aggregated/subset_indexes/person_index.json \
  --bucket core_pickleball \
  --exclude-source testing-esifc/pickle-ball-labeling-mff1d \
  --val-source pickleball-od8al/pickleball-version2 \
  --test-source hemel/pickleball-cedmo \
  --group-forks --source-balanced \
  --audit-samples-per-source 15 \
  --protected-root eval_clips/ball \
  --out runs/lanes/person_data_regroup_<date>/roboflow_person
```

The starting eligible inventory is 15,312 images / 47,044 boxes / 14 CC BY 4.0 sources. The exporter must keep fork families together and build whole-source splits; it must never pull the 15,469 adjacent-sport rows or the 22-image NC source into this arm.

**Expected measurable effect:** produce exact train/val/test counts and a measured annotation-quality card, not an assumed “person class exists” claim.

**CHECK:** sampled box precision `>=98%`; visible on-court-person annotation recall `>=95%`; any source below 90% is quarantined; after quarantine retain at least 5,000 images across at least eight train source groups; exhaustive protected-frame pHash/embedding collision count zero. Because original-game overlap still cannot be proven from hashes alone, result remains diagnostic until P0 supplies the human judge.

**Effort / GPU:** 8–12 engineer/reviewer hours; CPU only.

**P(fail): 50%.** Failure means public annotations omit far players/spectators or collapse onto a few projects. That is the answer; do not convert unlabeled people into background negatives.

#### P2 — one YOLO26m data-domain arm

**Asset and command-level action**

Freeze stock YOLO26m, image size, augmentations, threshold, and source split. Train one arm:

```bash
.venv/bin/yolo detect train \
  model=models/checkpoints/yolo26m.pt \
  data=runs/lanes/person_data_regroup_<date>/roboflow_person/data.yaml \
  imgsz=960 epochs=20 batch=-1 device=0 seed=20260721 \
  project=runs/lanes/person_train_regroup_<date> name=rf_person
```

Score stock and candidate first on the frozen Roboflow source-held test, then once on the P0 human compare card at the unchanged production confidence. Do not use Burlington/Wolverine to select the checkpoint.

**Expected measurable effect:** on the new pb.vision human card, AP50 `>=+0.05` and far/small-player recall `>=+0.10` over stock YOLO26m.

**CHECK:** above lifts; off-court/empty false positives no more than `baseline+0.02/frame`; no compare video AP regression greater than `0.02`; source-held Roboflow AP50 also improves `>=0.05`. One seed screens; only a pass gets a second seed and frozen downstream-tracking replay. No default change.

**Effort / GPU:** 2–4 A100/H100 GPU-hours, approximately `$2–17`, plus 2–3 evaluation hours.

**P(fail): 70%.** Failure means public boxes do not represent the product’s far-court/spectator/camera regime. It also confirms that another association or selection layer is not the next lever.

### 3.5 ReID — turn pb.vision player tracks into audited identity supervision

#### R0 — build and manually audit avatar-to-box teacher crops

**Asset and command-level action**

Extend `build_person_reid_crop_dataset.py` with `--pbvision-manifest`, or add `build_pbvision_reid_teacher_crops.py`. For each frame, map `sessions[].player_avatars[index]` to `court.player_points[index]`, convert normalized `(u,v)` to an image footpoint, and match it to frozen YOLO26m box bottom-centers. Accept only:

- teacher point confidence `>=0.90`;
- distance `<=max(20px, 0.25*bbox_height)`;
- second-best distance at least `2x` the best; and
- the same assignment surviving at least 30 consecutive frames.

Identity key is `<video_id>:<avatar_id>`. Use the same seven train videos as B1; use `pldtjpw3h0jw` and `utasf5hnozwz` for teacher-val and `0tmdeghtfvjx` for teacher-test. Never read the three permanent compare IDs during construction.

```bash
.venv/bin/python scripts/racketsport/build_pbvision_reid_teacher_crops.py \
  --gallery-root data/pbvision_gallery_20260719 --media-root "$PBV_MEDIA_ROOT" \
  --split-manifest runs/lanes/pbv_pickleball_corpus_20260720/manifest.json \
  --detector models/checkpoints/yolo26m.pt --point-confidence-min 0.90 \
  --second-best-ratio 2.0 --min-continuity-frames 30 \
  --out runs/lanes/reid_data_regroup_<date>/pbv_teacher_crops
```

**Expected measurable effect:** at least 20 train identities across at least six videos, each with at least 30 accepted crops.

**CHECK:** independent manual audit of 200 crops must show `>=98%` correct avatar assignment and zero identity swaps; train/val/test source intersection zero; every row marked `teacher_derived`; protected/compare IDs absent. Otherwise `REID_TEACHER_ASSIGNMENT_TOO_NOISY`, no train.

**Effort / GPU:** 10–14 engineer/audit hours; up to 1 GPU-hour for frozen detection/embedding prep.

**P(fail): 45%.** Failure means pb.vision points cannot be reliably matched to image people under occlusion/far scale. Do not relax thresholds after looking at the audit.

#### R1 — one explicit teacher-derived OSNet arm

**Asset and command-level action**

Add `--allow-teacher-derived` to `train_person_osnet_reid.py`. It must require the R0 audit artifact, write `do_not_promote=true`, and preserve the protected-clip guard; it is not a general bypass of the current `uses_cvat_labels=true` refusal.

```bash
.venv/bin/python scripts/racketsport/train_person_osnet_reid.py \
  --dataset-dir runs/lanes/reid_data_regroup_<date>/pbv_teacher_crops \
  --save-dir runs/lanes/reid_train_regroup_<date>/osnet_pbv \
  --weights models/checkpoints/osnet_x1_0_market1501.pt \
  --loss triplet --max-epoch 10 --batch-size 32 --num-instances 4 \
  --eval-freq 1 --allow-teacher-derived
```

**Expected measurable effect:** on the frozen non-compare teacher test, rank-1 `>=+0.10` and mAP `>=+0.05` over Market1501 OSNet.

**CHECK:** both metrics pass and neither heldout video regresses. This is only a screen because the labels are teacher-derived. Otherwise `REID_PBV_NO_RETRIEVAL_LIFT` and do not open P0.

**Effort / GPU:** 1–2 GPU-hours, approximately `$0.6–8.5`; 2 engineer hours.

**P(fail): 70%.** Failure means video-local pseudo identities are too same-camera/noisy to improve appearance features. Stop ReID tuning; the next evidence would need human identity labels, not more association sweeps.

#### R2 — one human compare-card retrieval/downstream check

**Asset and command-level action**

Add `score_person_reid_retrieval.py` to score frozen Market1501 and R1 checkpoints against P0’s human IDs without fitting or threshold search. Then export candidate embeddings through the existing CLI:

```bash
.venv/bin/python scripts/racketsport/score_person_reid_retrieval.py \
  --dataset runs/lanes/person_reid_holdout_<date>/human_reid_manifest.json \
  --baseline models/checkpoints/osnet_x1_0_market1501.pt \
  --candidate runs/lanes/reid_train_regroup_<date>/osnet_pbv/<frozen_checkpoint> \
  --out runs/lanes/reid_eval_regroup_<date>/retrieval.json

.venv/bin/python scripts/racketsport/export_person_reid_embeddings.py \
  --video "$VIDEO" --detections "$FROZEN_DETECTIONS" \
  --model "$CANDIDATE" --backend osnet --device cuda \
  --output "$OUT/embeddings.json"
```

Replay global association with frozen detections, court margin, and all non-ReID settings, then score the human IDs.

**Expected measurable effect:** human-card rank-1 `>=+0.10`, mAP `>=+0.05`, and either worst-video IDF1 `>=+0.02` or four-player coverage `>=+0.05` over stock OSNet.

**CHECK:** above lift, no video IDF1 regression `>0.01`, no new identity switches, and no spectator/far-off-court FP regression. One frozen comparison only. Even a pass is a scoped candidate; existing TRK `VERIFIED=0` remains.

**Effort / GPU:** 3–5 engineer hours; `<=1` GPU-hour.

**P(fail conditional on R1 pass): 65%.** Failure says the teacher-test gain was not human/product-domain gain. Preserve the negative and do not change association thresholds.

## 4. Five-day dependency schedule

### Standing roles

- **Data steward:** owns source groups, rights, hashes, quarantines, and the utilization ledger. Cannot select checkpoints.
- **Independent eval owner:** owns human holdouts, scorer hashes, thresholds, and one-touch execution. Trainers cannot read protected labels.
- **File-fenced component lanes:** EVENT, BALL, COURT, PERSON, ReID may work in parallel after their data gate.
- **Integration owner:** the only lane allowed to serialize `scripts/racketsport/process_video.py` or default-selection changes. This sprint plans no default flip.

Use as many CPU agents as useful for the five data audits. GPU concurrency is limited by passed gates, not by available quota. Output-sync happens after every completed arm; no result stays VM-only until a lane ends.

| Day | Parallel work and dependencies | Mandatory end-of-day measurement or named negative |
|---|---|---|
| **Day 1 — recover truth and freeze data contracts** | **EVENT:** owner reauths GCP; recover/hash A and any B/C; count audio-only rows. **BALL:** reconcile the 3,026-row lineage, parent-source split, and status of the 350 scratch task. **COURT:** export tasks 88–91; deny `IYnbdRs1Jdk`; freeze 19/8 source split. **PERSON:** compute exact 14-source/15,312-image audit sample and prepare P0. **ReID:** stage allowed media and run a bounded crop-match feasibility sample. Data steward creates first ledger snapshot. | EVENT: A/B/C macro-F1 if present, or `A_ARTIFACT_UNRECOVERABLE`, `B_C_NOT_RUN`, or `METHOD_INVALID_AUDIO_ONLY=n`. BALL: exact `scratch_reviewed` and lineage counts. COURT: exact `usable/protected/rejected` count. PERSON: measured label precision/recall on audit sample. ReID: accepted/misassigned crops in the first 50. No “tooling complete” is the headline. |
| **Day 2 — establish independent baselines** | **EVENT:** repair/rebuild B/C only if method-invalid; otherwise score seed one. **BALL:** finish scratch review; freeze 167-row HyU/Ezz judge; score untouched official control. **COURT:** ingest reviewed rows; score frozen learned and classical baselines on eight source groups. **PERSON/ReID:** reviewer completes the shared 120-frame compare card; finish the 200-crop R0 audit. | One frozen baseline row per component: BALL F1@20/recall/hFP by source; EVENT macro-F1/negative FP/timing/rate; COURT median/PCK by held source and technology; PERSON AP/far recall/off-court FP; ReID Market1501 rank-1/mAP. If a valid judge is absent, end with `NO_CLEAN_JUDGE`, not an internal metric substitute. |
| **Day 3 — seed-one data interventions** | Only lanes whose Day-2 data gate passed train. EVENT runs/finishes seed-one B/C. BALL runs human-only A vs human+pbv B. COURT runs H human-diversity. PERSON runs one Roboflow-domain YOLO arm. ReID runs one teacher-derived OSNet arm. Architectures, update budgets, and thresholds stay frozen. | Candidate-minus-control delta on the unchanged source-held validation for every dispatched component, plus exact GPU-hours and dollars. Each result is `DIRECTIONAL_PASS`, `NO_LIFT`, `HARM`, or `NO_ATTEMPT_PREREQ`. |
| **Day 4 — replicate survivors; no new ideas** | Replicate only seed-one survivors. EVENT completes the formal three-seed owner-41 gate. BALL gets two more matched A/B seeds. COURT gets two more H seeds; C2 runs only if H passed. PERSON gets one replication seed and its one frozen human card. ReID opens the human card only if teacher retrieval passed. No selection layer, architecture, threshold, or data-source addition. | Median delta, per-source worst delta, bootstrap interval, safety regressions, and cost. Label each component `DATA_LIFT`, `NO_LIFT`, `HARM`, or `NOT_RUN_PREREQ_FAILED`. |
| **Day 5 — frozen decisions and utilization closeout** | Independent eval owner performs only already-earned frozen checks, including EVENT protected-50 only if its executable gate passed. Integration owner may run a non-authoritative whole-video replay for surviving candidates but does not change defaults. Data steward records every asset/run/result and closes never-queued cells. | One final table for all five components: baseline, candidate, source-held delta, interval, worst-source result, human/protected touch count, GPU-hours/cost, and `REJECTED`, `PARKED`, `scoped candidate`, or `NO_ATTEMPT`. `VERIFIED=0` remains unless a pre-existing named independent promotion gate genuinely passes; no such pass is forecast here. |

Dependencies are strict:

```text
durable artifact + data ruling + source split + clean judge
                        ↓
                 seed-one A/B check
                        ↓ pass only
                 replication / human card
                        ↓ pass only
                 existing promotion gate
```

## 5. What we stop doing now

1. **Stop building selectors, physics patches, association layers, or threshold rules before source diversity exists.** The last honest failures were all domain/venue failures; more downstream logic cannot manufacture missing perception evidence.
2. **Stop presenting old experiments as untried data.** Roboflow BALL and real-transfer COURT already ran and failed cross-domain checks.
3. **Stop random frame splits.** A clip, game, venue, session, channel/fork family, and its derivatives stay in one partition.
4. **Stop scoring on labels descended from the evaluated model.** Accepted prelabels and teacher predictions may train at reduced weight; they never judge the parent/candidate.
5. **Stop GPU dispatch before payload proof.** Every source must have pixels, decode success, exact hashes/PTS where temporal, annotation authority, rights, and a frozen scorer.
6. **Stop wiring downloaded public labels whose media or 64-frame context is absent.** First emit a decodable-window count and semantic map.
7. **Stop requesting more owner labels before the existing staged packs are exported and their learning curve is measured.** The only owner work this week is finishing the already-staged BALL/COURT tasks and one shared PERSON/ReID judge.
8. **Stop learned court work as authority.** The one scoped reopen is preview-only; capture discipline, static lock, and classical line evidence remain the product path.
9. **Stop audio-only event anchors.** Audio can support a typed/non-audio cue only after beating its shifted null.
10. **Stop protected-set threshold shopping.** A protected result cannot select threshold, seed, checkpoint, arm, or code.
11. **Stop leaving successful work on a VM.** Pull/hash after each arm; a watchdog must match the broad training process class; the hard spend rail stays armed.
12. **Stop machinery-only days.** “Loader landed,” “pack staged,” “VM created,” “loss fell,” and “overlay looks plausible” are not day outcomes. The day ends with a metric or a named negative.

## 6. Standing organization so never-queued cells cannot recur

Do **not** add a root `DATA_LEDGER.md`; that would drift into a second planning authority and violate the repository documentation policy. Add one machine-readable coordination registry under `runs/manager/` and generate the human view from it:

- canonical: `runs/manager/data_ledger.json`;
- generated view: `runs/manager/DATA_LEDGER.md`;
- audit command: `scripts/racketsport/audit_data_utilization.py`.

Every asset row must contain:

- stable asset ID, paths, byte count, raw count, dedup-kept count, decoded count;
- original source/game/session/channel/fork family and immutable hashes;
- rights posture and the ruling that allows/forbids each component;
- label authority: human GT, corrected prelabel, confirmed prelabel, teacher, synthetic, or none;
- protected/compare/quarantine identities and overlap-check coverage;
- exact train/val/test partition;
- consumers: lane, command/config hash, rows actually loaded, result path, metric/verdict;
- current state: `READY`, `BLOCKED`, `QUARANTINED`, `CONSUMED`, `REJECTED`, or `DEFERRED_WITH_REASON`; and
- named owner and next check.

Every lane `spec.md` must have a mandatory `DATA CONTRACT` block:

1. ledger asset IDs and exact hashes;
2. utilization delta (`unused -> consumed`, `blocked`, or `rejected`);
3. train/val/holdout source groups and zero-overlap proof;
4. baseline, expected delta, kill check, effort hours, and GPU-hour/$ cap;
5. data owner, training owner, and independent eval owner;
6. output-sync-after-each-arm path; and
7. the named end-of-day number or negative.

`audit_data_utilization.py` should fail pre-dispatch when:

- an input is absent from the ledger or its current hash differs;
- any train/holdout source family overlaps;
- a teacher row is represented as GT;
- a protected asset is reachable by the trainer;
- the baseline/check/kill threshold is missing; or
- a GPU command references an asset with zero decoded rows.

It should also emit a sorted “never queued” report for any acquired asset older than 24 hours that has bytes/labels but no consumer, blocker, quarantine, rejection, or explicit defer ruling. Lane closeout is incomplete until it appends actual loaded-row counts and the result. The North Star remains product truth; this ledger owns only data lineage/utilization truth.

## 7. Honest confidence summary

| Decision | Estimated chance the proposed model/data arm fails its check | What a failure buys us |
|---|---:|---|
| EVENT pb.vision B beats A and placebo across three seeds | 80% at seed one; 55% replication failure conditional on seed-one pass | Closes the current teacher-timing hypothesis without touching protected 50 or collecting more owner events. |
| BALL human+pb.vision beats equal-human A on source-held scratch rows | 70% after the data gate | Separates “more in-domain teacher diversity” from human exposure and retires Roboflow/same-source reruns. |
| COURT 19-source human-diversity preview clears the old kill bar | 80% | Determines whether source diversity, rather than volume/init, can reopen learned preview at all; failure re-centers capture/static/classical work. |
| PERSON Roboflow box arm improves new pb.vision human card | 70% | Determines detector-domain leverage before another association layer; failure exposes public-label incompleteness/domain mismatch. |
| ReID pb.vision avatar arm improves teacher retrieval | 70% | Tests whether dense player points can become usable identity supervision. |
| ReID candidate then improves the human compare card | 65% conditional on teacher-screen pass | Detects a teacher-only illusion and prevents an association/default change. |

These are deliberately pessimistic. The plan promises five days of decisions, not five improved models.

## 8. The single highest-yield first action tomorrow morning

**Owner runs interactive `gcloud auth login`; the EVENT/eval owner immediately reconciles `pickleball-gpu-abc`, starts the same durable disk if needed, pulls and hashes Arm A plus any B/C artifacts, audits audio-only eligibility, and produces the seed-20260720 owner-41 A/B/C row before any new model work.**

Why this is first: it protects the only completed already-paid artifact, is the closest available controlled test of the highest-value in-domain pb.vision data, and can produce a same-morning `DIRECTIONAL_PASS`, `NO_LIFT`, `METHOD_INVALID`, or `ARTIFACT_UNRECOVERABLE`. Every one of those outcomes changes what should happen next. Building another layer does not.

No promotion is claimed anywhere in this plan. `VERIFIED=0` remains binding.
