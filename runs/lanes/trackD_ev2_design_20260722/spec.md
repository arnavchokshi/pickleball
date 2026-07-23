# LANE trackD_ev2_design_20260722 — E-v2 registered design + recipe implementation (NO training run)

Codex gpt-5.6-sol, effort ULTRA (novel design + gate-adjacent correctness; the guards and
curriculum are the whole game). CPU-only lane. Dispatched by Track-D Fable agent under
runs/research_sota_20260722/PROGRAM.md Track D.

## HARD RULES
- No branches, no commits (joint-commit rule; Track-D manager commits after ultra review).
- Read first: runs/research_sota_20260722/PROGRAM.md (Track D) + EVENTS.md (the evidence base —
  every recipe element below is cited there), runs/lanes/abc_experiment_20260721/{E0_VERDICT.md,
  E1_RELAUNCH.md,e1_screen.json}, runs/regroup_20260721/EXACT_PLAN.md (E-series discipline),
  NORTH_STAR_ROADMAP.md, AGENTS.md.
- 4 protected clips EVAL-ONLY. The protected-50 one-touch token is SEALED — never touch
  runs/lanes/event_bootstrap_20260713/ answer files. Owner-41 val is FROZEN: NO scoring, NO
  peeking, NO threshold/hyperparameter tuning against it. This lane runs ZERO owner-41 evals.
- VERIFIED=0 stays. Honest reporting. Artifacts under runs/lanes/trackD_ev2_design_20260722/.
- Preserve unrelated dirty worktree changes (there are uncommitted files from other tracks).
- Wide blast-radius test suite before final report: MPLBACKEND=Agg .venv/bin/python -m pytest
  tests/racketsport/ (failures>0 while claiming PASS = rejected unless proven pre-existing).
- Every new CLI ships a direct-CLI reference test same-lane and registers in
  scripts/racketsport/list_scaffold_tools.py (ADDITIVE, edit last, fresh-read before edit —
  other tracks append concurrently).

## FILE OWNERSHIP (file-disjoint with in-flight lanes; ball_b1_resume_20260722 owns ball files)
- OWNS: threed/racketsport/event_head/{model.py,datasets.py}, NEW threed/racketsport/event_head/
  assignment.py (training-side Hungarian/dynamic assignment lives here, NOT in matcher.py),
  scripts/racketsport/{finetune_event_head.py,train_event_head.py}, their tests under
  tests/racketsport/, lane dir.
- READ-ONLY / FROZEN JUDGE: scripts/racketsport/eval_event_head.py and threed/racketsport/
  event_head/matcher.py metric/matching math. If an ADDITIVE eval flag is unavoidable, default
  behavior must be byte-identical, the change is listed under HONEST ISSUES, and the E1
  reproduction control (below) is mandatory proof.
- READ-ONLY: threed/racketsport/event_head/sequence_dp.py (dormant until E-v2 passes its gate),
  scripts/racketsport/build_abc_arm_manifests.py, everything else.

## OBJECTIVE
Produce (A) a frozen, complete E-v2 experiment registration and (B) the implemented + tested
training-recipe changes it requires, so a Sonnet GPU lane can execute the run verbatim. E-v2 =
the SoccerNet-BAS winning curriculum applied to our assets (EVENTS.md Q1 item 3 / ranked list #1):

  STAGE P (pretrain): train the event head on the 1,189-row agreement corpus
    (runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json — 773 kink-only
    @0.25 + 416 audio+kink @0.5; audio-only EXCLUDED by E0 ruling; SHA-pin it in the registration).
  STAGE F (fine-tune): re-init from Stage-P weights, fine-tune on owner-61 train rows
    (runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json) WITH hard-negative mining
    from E1-B's known over-fire cases: extract the exact false-positive negative rows from
    runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/b_owner41.json — BUT NOTE these 4
    over-fires live in the owner-41 negative subset (22 rows) which is FROZEN VAL. You may NOT
    train on them. Design decision required: mine hard negatives from TRAIN-SIDE material that
    matches their failure signature (e.g. near-event background windows on owner-61 clips +
    teacher-clip windows where audio fired but kink did not — the exact E0-excluded audio-only
    family is a natural hard-negative pool). Justify the chosen pool in the registration; using
    frozen-val rows for training is an automatic reject.
  STAGE V (validate): ONE registered owner-41 scoring per arm via the frozen judge command
    (eval_event_head.py --mode owner-val, threshold registered up front). Executed later by the
    GPU lane, never by this lane.

Recipe elements to implement (each cited in EVENTS.md; each unit-tested; each toggleable by CLI
flag so arms/ablations stay exposure-matched):
1. sqrt-frequency class weighting computed from the actual training manifest class counts
   (replaces fixed DEFAULT_CLASS_WEIGHTS (1,5,5); inverse-frequency is FORBIDDEN — it measurably
   causes rare-event over-firing, our exact negFP failure).
2. +-1-frame label dilation (soft positives around each event timestamp) in the dense targets.
3. Training-side Hungarian/dynamic label assignment (DETR-style min-cost matching over class
   confidence + temporal offset) so the model may deviate from noisy pb.vision teacher
   timestamps. If a clean implementation for the per-frame CE + GRU head is not achievable in
   this lane, the REGISTERED FALLBACK is Gaussian-soft labels only — record it as a named
   exclusion, do not fake it.
4. T-DEED-style offset-regression head (auxiliary sub-frame offset output). CONSTRAINT: the
   frozen judge loads checkpoints via event_head/model.py load_checkpoint and scores via
   peak_pick — the offset head must be additive (old checkpoints still load; judge path
   unchanged). If judge compatibility cannot be preserved without touching metric math: DROP the
   offset head from E-v2, record as named exclusion. Never modify the judge to fit the model.
5. NMS decode registered at +-2f (matcher default nms_radius=2 — verify and register, don't
   assume) and a REGISTERED decode threshold chosen from Stage-P internal validation only.
6. Under-firing countermeasures: the E1-B failure was rate 0.1065/s vs required 0.3-1.0/s.
   The registration must name which levers target rate (weighting, dilation, threshold) and
   define an INTERNAL pre-scoring rate check on train-side/teacher clips (full-video firing
   rate needs no labels) so the single owner-41 scoring is not burned on an obviously
   under-firing checkpoint. Same for an internal negFP proxy using TRAIN-side negatives only.

## REGISTRATION.md must freeze (before any training)
- Arms + exposure matching: propose the minimal arm set. Causality is already proven (E1:
  B 0.1304 vs A 0.0 vs C 0.0) — E-v2's job is recipe repair, compared against the FROZEN E1-B
  numbers as historical baseline. If you add any new arm, equal steps + matched loss caps,
  and each arm gets exactly ONE owner-41 scoring, all registered here.
- Checkpoint-selection policy WITHOUT owner-41 shopping. State exactly what E1 used (read
  finetune_event_head.py validation path and the E1 artifacts to determine what best_val was
  computed on) and either match it apples-to-apples or justify a stricter internal-val policy.
- Pass/fail gate, guards AT LEAST as strict as E1's: macro-F1@+-2 >= 0.1304 (E1-B), negFP
  <= 2/22, full-video rate in 0.3-1.0/s, timing p90 not worse than a registered bound you
  justify (E1-B was 2f; A was 64f). Kill/stop rules, max seeds, spend cap.
- Exact GPU commands, hyperparameters, seeds, SHA pins of all input manifests + init
  checkpoint, steps/s probe plan (~100-step probe first, then wall cap = measured + 50%
  contingency), single-A100-40 sizing, sequential arms (never concurrent — E1 lesson).
- What happens on PASS: sequence_dp.py wiring becomes eligible (separate lane); best_stack
  PENDING entry. On FAIL: named negative verdict string, no retries without new registration.
- Scope honesty: RGB-only architecture this experiment; track/wrist conditioning channels are
  a named follow-up, not smuggled in.

## MANDATORY CONTROLS (implement + run in this lane, CPU)
- E1 reproduction control: with the final lane code, load the three banked E1 checkpoints
  (vm_pull_v2/{B_pbvision_teacher,C_placebo}/... and inputs A checkpoint per e1_screen evidence
  paths) and re-run the owner-41-SHAPED scoring path ON THE BANKED EVAL JSONs' recorded
  numbers?? NO — you cannot re-score owner-41. Instead: verify the judge code path is
  byte-identical to E1 (git diff of eval_event_head.py + matcher.py vs commit used by E1 per
  vm_pull_v2 provenance) AND that old checkpoints still load + produce identical logits on a
  fixture window (regression test with a tiny synthetic fixture, no owner data). If you did
  touch eval additively, prove default-path identity by diff + fixture, and flag it.
- Dataloader==inference tensor parity: push one sample through the training dataloader AND the
  production inference preprocessor; assert identical tensors (standing wave-4 rule).
- Determinism: same-seed manifest/window construction byte-identical across two runs.

## CROSS-SIGNAL ROW (mandatory in report)
CONSUMES: ball_velocity_kink + audio_onset agreement families (corpus tiers), pb.vision teacher
timestamps, owner event labels. FEEDS: ball-3D arc anchors (event candidates), rally
segmentation, sequence_dp decode stage, audio late-fusion gate (Track B artifact pending).

## BEST-STACK DELTA
(c) No stack delta this lane (design+code only, nothing trained/promoted); the E-v2 GPU run
adds the PENDING entry on PASS. State this in the report.

## REPORT (schema-enforced)
objective_result vs the acceptance items below, full_suite counts, HONEST ISSUES (esp. any
judge-adjacent touch, any dropped recipe element + why), artifacts list, and the exact
VM_RUN_PLAN.md path for the GPU leg.

ACCEPTANCE (all must be true):
1. REGISTRATION.md complete per the freeze list above, internally consistent, executable
   verbatim by a GPU lane.
2. Every adopted recipe element implemented behind a flag + unit-tested; every dropped element
   a named exclusion with reason.
3. E1 judge-identity control PASS; parity check PASS; determinism check PASS.
4. Wide suite green or failures proven pre-existing.
5. VM_RUN_PLAN.md with probe-then-cap math and sequential-arm schedule.

Anti-passive-wait: this is a CPU lane; run everything foreground; end only with the final
report or a hard blocker.
