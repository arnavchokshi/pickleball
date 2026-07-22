# LANE SPEC — ball_b2_trainer_repair_20260722 (parity-baseline + SST-uniqueness repair)

Dispatched 2026-07-22 by the Track-B agent. Codex `gpt-5.6-sol`, effort `ultra` (gate-adjacent
correctness with real design latitude: a wrong sampler fix silently corrupts the flagship B2
experiment). CPU-only.

## 1. HARD RULES

- Stay on `main`. No branches. No commits. The manager commits after ultra adversarial review.
- Read `NORTH_STAR_ROADMAP.md`, `AGENTS.md`, `runs/regroup_20260721/EXACT_PLAN.md` sec 3.2, and —
  MANDATORY, this is your contract —
  `runs/lanes/ball_b1b2_prep_20260722_review_r4/review_r4.json` (blocks: DISPATCH_ANALYSIS,
  EXPOSURE_MATCH_CHECK.blocking_caveat, GPU_DISPATCH_DECISION.minimum_reopen_conditions).
- The four protected clips EVAL-ONLY; compare-only pb.vision IDs unreachable; `VERIFIED=0`
  binding; honest reporting; NO JUDGE PEEKING (never execute any scorer against the 167-row judge).
- Run the WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q`; classify
  failures vs `runs/lanes/ball_b2_seed1_20260722/scripts_used/full_test_failures.txt`.
- All artifacts under `runs/lanes/ball_b2_trainer_repair_20260722/`.

## 2. EXPLICIT FILE OWNERSHIP

This lane owns exactly:
- `scripts/racketsport/train_ball_stage2.py` (current sha256 MUST equal
  `9d2d326103da97afa1adc5f13bffc9560cc8f2d2af9942199a198568c9d44f3a` before you start — verify;
  if not, STOP and report IDENTITY_MISMATCH)
- the existing test module(s) covering the trainer + parity harness (locate; no duplicate modules)
- `runs/lanes/ball_b2_trainer_repair_20260722/**`

NOT owned (concurrent lanes / pinned): `scripts/racketsport/build_pbvision_ball_sst.py` (in-flight
resume lane), `scripts/racketsport/measure_audio_event_alignment.py` + its test (in-flight audio
lane), `ball_loso_validation.py`, `threed/racketsport/wasb_adapter.py`,
`configs/racketsport/best_stack.json`.

## 3. DEFECT 1 — parity baseline hardcodes HEAD (review_r4 blocking fact 1)

`verify-head-parity` (locate it in the trainer CLI) hardcodes HEAD as its baseline. Current main
HEAD already contains the candidate trainer bytes (9d2d3261...), so the harness can only compare
candidate-versus-itself and can never satisfy review_r3's requirement: baseline
`8118ff0e8fbf1d573f61e1ce09de140cb2c9e9e62bf1d57b030560a55a157f47` vs candidate, DISTINCT and
both pinned.

REQUIRED FIX: make the baseline revision-explicit and fail-closed:
- A `--baseline-rev <git-rev>` (and/or `--baseline-sha256`) selector; the harness loads the
  baseline trainer source from the named git blob (e.g. `git show <rev>:scripts/racketsport/train_ball_stage2.py`),
  computes its sha256, and REQUIRES it to equal the expected pinned baseline hash passed
  explicitly (no default that silently resolves to HEAD).
- Hard fail-closed assert: baseline sha256 != candidate (running-file) sha256. A
  candidate-versus-itself invocation must ERROR, never report PASS.
- Record both hashes + the git rev in the parity output artifact.
- Historical note for your report: the old premise (baseline=HEAD, candidate=working tree) died
  when commit 4c27023f landed the candidate on main — this also explains the known-failing test
  `test_pinned_head_harness_loads_actual_git_source_and_records_exact_compute`; repair or replace
  that test's premise as part of this fix rather than deleting coverage.

## 4. DEFECT 2 — SST within-batch duplicate sample-IDs accepted (review_r4 blocking fact 2)

Reproduced by the reviewer on current bytes: DeterministicFullBatchSampler splices the tail of one
permutation to the head of the next; with sample_count=1001, batch_size=8, seed=20260722, 2372
batches, zero-based step 1751 contained [256, 432, 546, 743, 401, 497, 432, 947] — duplicate 432.
The SST batch assertion checks size and parents but NOT sample-ID uniqueness, violating review_r3
precondition 6 ("if eight distinct pseudo rows per batch are intended, explicitly reject
within-batch SST duplicates").

REQUIRED FIX — design latitude, choose with written justification, ultra bar:
- Option A: fail-closed uniqueness assertion only, PLUS a CPU preflight subcommand that
  deterministically enumerates ALL planned SST batches for a given manifest size/seed/steps and
  fails fast (before any GPU) if any batch would contain a duplicate.
- Option B: make the sampler duplicate-free by construction at permutation splice boundaries
  (deterministic, seeded, documented) PLUS the same fail-closed runtime assertion as belt-and-
  braces.
Constraints either way:
1. The HUMAN-ONLY code path's compute must be BIT-IDENTICAL to current bytes — the revision-
   explicit parity run against baseline 8118ff0e is exactly what will prove your change didn't
   touch it. Do not alter human loaders, human sampler, loss path, or step count in any way.
2. Deterministic given (manifest, seed); no RNG state leakage into the human path.
3. Unit tests MUST include the exact reviewer reproduction (1001/8/20260722/step-1751) asserting
   the old behavior is now impossible (assertion fires or batch is duplicate-free, per your
   option), plus a boundary-splice case that passes.
4. If you find that NO fix can avoid touching the human path, STOP and report
   `REPAIR_TOUCHES_HUMAN_PATH` with the exact coupling — do not trade one defect for another.

## 5. ACCEPTANCE — numbers, not adjectives

1. Parity harness: candidate-vs-itself invocation errors (test); distinct-baseline invocation on a
   TINY fixture config runs end-to-end on CPU and records both hashes + rev (test or scripted
   proof; no GPU, no judge).
2. SST: reviewer reproduction test passes; runtime assertion present and fail-closed.
3. Human-path invariance argument: named list of every changed function/branch and why none is on
   the human-only path (the parity run remains the executable proof).
4. Wide suite counts + classifications (the parity-premise test is expected to move from the
   known-failing list to passing/repaired — say so explicitly).
5. `new_trainer_sha256` reported (the arming review re-pins the candidate to this).
KILL: `REPAIR_TOUCHES_HUMAN_PATH` per above; or `PARITY_HARNESS_NOT_LOCALIZABLE` if
verify-head-parity turns out to live outside your owned files — report, do not expand ownership.

## 6. CROSS-SIGNAL ROW

- CONSUMES: review_r4 defect evidence; review_r3 pinned hashes; frozen B0 split design.
- FEEDS: CUDA parity run -> fresh B2 arming review (review_r5-equivalent) -> B2 arms A/B.

## 7. BEST-STACK DELTA

(c) No stack delta — trainer harness correctness only; nothing promoted; best_stack untouched
(assert in report).

## 8. MANDATORY STRUCTURED REPORT

report.json: objective_result per acceptance item; `new_trainer_sha256`; option chosen (A/B) with
justification; human-path invariance list; full_suite counts + classification; HONEST ISSUES;
artifacts.
