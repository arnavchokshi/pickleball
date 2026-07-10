# COURT-TRAIN-2 structured report

## objective_result

`PASS`

All requested trainer-infrastructure targets pass. The required wide suite has seven unrelated,
pre-existing failures documented under `full_suite`; all 31 owned trainer tests pass.

## acceptance

| metric | baseline | after | target | verdict |
|---|---|---|---|---|
| A1 masked-null, full-15, init, and direct-CLI tests | 22 owned tests; null visibility was incorrectly supervised | 31/31 owned tests pass; 5-step partial-row CLI sampled 15 null channels with zero heatmap/visibility supervision | All requested tests green | PASS |
| A2 actual-corpus CPU smoke | No v2 partial-corpus smoke | 20 steps at 640x360 in 17.8s; step loss 16.4798 to 10.5442; fixed real probe 7.7104 to 7.0740; 54 sampled null channels and zero supervision violations | Loss decreases, no null supervision, under 15 minutes | PASS |
| A3 GPU arm commands | No fresh model-only ARM-A path; pinned ARM-B file rejected by an over-strict legacy-counter check | Exact ARM-A and ARM-B invocations are in `HANDOFF.md`; both initialization paths have direct tests | Commands include real root, weights, and step placeholders | PASS |
| A4 wide suite and write scope | Unknown | 3,422 passed, 7 pre-existing/unrelated failed, 24 skipped; source writes limited to the two owned files | Zero failures or all failures proven pre-existing; exclusive scope | PASS |

## changes

- `scripts/racketsport/train_court_model_v2.py`: added per-keypoint coordinate and visibility
  masks; sparse/explicit-null validation; fresh model-only checkpoint initialization; bounded
  real-only photometric augmentation; train-dataset split filtering; fixed real-loss probes and
  null-supervision counters; pinned official ImageNet ResNet34 compatibility without weakening
  learned-parameter or shape validation.
- `tests/racketsport/test_train_court_model_v2.py`: added null garbage-prediction/gradient/metric
  invariance, full-15 total-loss regression, malformed rows, deterministic photometric
  augmentation, 5-step partial-real direct CLI, actual checkpoint fresh-init plus two-frame
  owner-gate fixture, and pinned ImageNet encoder coverage.
- `runs/lanes/court_train2_20260710/cpu_smoke_actual/`: actual-corpus smoke metrics and checkpoint.
- `runs/lanes/court_train2_20260710/HANDOFF.md`: exact arm commands and safety semantics.

## full_suite

- command: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport`
- result: **3,422 passed, 7 failed, 24 skipped** in 2,629.73 seconds.
- failures_all_preexisting: `true`
- six failures are HTTP-server tests in
  `test_court_keypoint_review_server.py` and `test_review_input_server.py`. The managed sandbox
  rejects even a standalone `socket.bind(("127.0.0.1", 0))` with
  `PermissionError: [Errno 1] Operation not permitted`. This lane changed none of those tests or
  server files.
- one failure is the stale evidence assertion in
  `test_overlapping_court_calibration_eval.py`: it expects `27.538259`, but the scanner selects
  `runs/lanes/court_train1_20260709/vm_pull/arm_a/court_keypoint_metrics.json` at `13.487954`.
  That TRAIN-1 artifact was born at 05:17 local, before this lane began at 06:22, and is explicitly
  reported as 13.49px in TRAIN-1's required input report. The selected path is not TRAIN-2's smoke
  output. This lane changed neither the scanner nor its snapshot test.

## honest_issues

- The wide suite is not literally green: six sandbox socket-bind failures and one stale
  pre-existing evidence-snapshot assertion remain.
- Neither GPU arm was launched in this infrastructure lane. Initialization and execution paths
  are directly tested; transfer accuracy remains unmeasured until the arm runs occur.
- The 20-step CPU smoke and two-frame evaluator fixture are execution/integrity evidence only.
  They are not protected-gate evaluations and do not promote CAL.
- Photometric augmentation is bounded and label-preserving by construction, but this lane did not
  claim or measure an accuracy gain from it.
- Repository structure checks: scaffold index passed with zero missing direct CLI references;
  dead-code audit passed with zero unknown Python sources. The storage-policy audit returned
  `status: fail` only because many allowlisted untracked source packages are absent from this
  checkout; it reported zero unknown large tracked files and zero unknown large untracked files.
  TRAIN-2 created no unknown storage-policy violation.

## next

Run ARM-A and ARM-B on one GPU with identical corpus, split proposal, weights, augmentation,
steps, seed, and frozen scoring protocol. Do not touch protected eval clips until preregistered.

## session_id

`null`

## best-stack delta

**NO stack delta.** Trainer infrastructure only; `configs/racketsport/best_stack.json` is
untouched and `VERIFIED=0` remains binding.
