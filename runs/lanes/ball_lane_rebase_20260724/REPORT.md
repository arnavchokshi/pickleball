# Ball-lane rebase merge-readiness report — 2026-07-24

Lane: `ball-lane-20260723` (measurement instrumentation: metric-3D contract, frozen eval
judge, ball-3D characterization harness, observation-log tooling, lifter prep).

Worktree: `/Users/arnavchokshi/Desktop/pickleball/.claude/worktrees/ball-lane-20260723`
**No merge and no push were performed. No cloud/GPU was used.**

## VERIFIED=0 — scope of this evidence

Everything below is **integration evidence only**. It shows this branch replays onto `main`
without conflict and does not regress the suite. It is **not capability evidence**: no
metric-3D accuracy claim, no solver-quality claim, and no gate promotion follows from any
of it. `VERIFIED=0` remains binding for the ball lane.

## 1. Provenance

| Item | Value |
|---|---|
| Fork point (merge-base) | `ce32f1f30cdeec5ac556f2fa2479b7d6cbb7f08e` |
| Branch tip before rebase | `ca3d5f427dc1c5ad11ebd7d1b7203e1498671aba` |
| Safety tag (recovery point) | `ball-lane-prerebase-20260724` -> `ca3d5f4` |
| Rebase target (`main` at rebase time) | `9d318ad0c86ee424a5d01aae5eac0adaac578dd7` |
| Branch tip after rebase | `35b44f39c91f44dab9162c3871c68765b83b52be` |
| Rebased tree hash | `a2d62bfa656177c6332e870b9ec675ffebbd72f3` |
| `main` at time of writing | `2e073c8d5bd5210dc792a2ca159bbdd1fb813554` (advanced 4 commits during verification) |

## 2. Collision surface — EMPTY

Computed before rebasing.

- Branch changed **137** files (`ce32f1f..ca3d5f4`).
- `main` changed **33** files since the fork point (`ce32f1f..9d318ad`), all BODY/court/placement work.
- **Intersection: 0 files.**

High-risk shared-ownership files:

| File | Branch | main |
|---|---|---|
| `scripts/racketsport/process_video.py` | not touched | touched |
| `configs/racketsport/best_stack.json` | not touched | not touched |
| `models/MANIFEST.json` | not touched | not touched |
| `scripts/racketsport/list_scaffold_tools.py` | not touched | not touched |
| `threed/racketsport/orchestrator.py` | not touched | not touched |

No high-risk file is co-edited, so no additive-registry merge was required.

Branch non-`runs/` surface is 16 files: 3 new CLIs, 4 new `threed/racketsport/` modules
(incl. `eval/ball_metric3d_eval.py`), 9 test files (6 added / 3 modified), `DATA_INVENTORY.md`.
The other 121 files are `runs/` evidence and `runs/manager` ledger rows.

### 2b. Semantic collision check (textual disjointness is not sufficient)

Disjoint filenames do not rule out a branch module importing something `main` changed
underneath it. One real dependency exists:

- `threed/racketsport/virtual_world.py` was modified by `main` (+166 lines) and is imported
  by branch modules `ball_solver_characterization.py` and `build_ball_observation_log.py`.
- Branch imports 4 symbols: `BALL_ARC_FAIL_CLOSED_MAX_REPROJECTION_PX`,
  `BALL_ARC_FAIL_CLOSED_MIN_INLIERS`, `BALL_ARC_FAIL_CLOSED_POLICY`,
  `ball_arc_segment_fail_closed_verdicts`.
- `main`'s edits are **purely additive**: 0 `def`/`class` removed; all 4 symbols present at
  `9d318ad` **and** at the newer `60631f1`.
- `threed/racketsport/eval/` is otherwise identical between merge-base and `main`; the branch
  only adds `ball_metric3d_eval.py`.

## 3. Rebase result — SUCCEEDED, zero conflicts

`git rebase main` -> **exit code 0**, 15/15 commits replayed, **no conflicts, no manual
resolution required**. Nothing was dropped from either side, because nothing overlapped.

Independent corroboration, three ways:

1. **Pre-flight dry run.** Replayed first in a throwaway clone under the scratchpad (never in
   the main checkout). Produced tree `a2d62bf` — identical to the real rebase.
2. **Byte-identical delta.** The branch's diff against its base is byte-for-byte identical
   before and after the rebase (two 358,577-line patches, `diff` empty):
   `diff <(git diff ce32f1f ca3d5f4) <(git diff 9d318ad 35b44f3)` -> no output.
3. **Evidence preserved.** `runs/ball_lane_20260723` subtree hash is
   `71d976c139f1abc00eb33129b7634e7c8cc154b2` both before and after — all 118 committed
   evidence files unchanged. Nothing regenerated or overwritten.

`main`'s BODY/court work is fully intact: the branch modifies 0 of `main`'s files.

## 4. Test evidence

Environment: `.venv` (Python 3.14.6, pytest 9.1.1), CPU only. Marker filter
`-m "not h100 and not integration"` on wide runs. Exit codes are **unpiped** (`$?` captured
directly, not through a pipeline).

> **Machine contention caveat.** Verification ran while concurrent agents held the box at
> load average 5–13. This repo has a documented history of fixed wall-clock in-test budgets
> flaking under load; that is exactly what was observed (section 4.4).

### 4.1 Branch-owned tests — GREEN

All 9 test files the branch adds or modifies:

```
132 passed in 16.28s
BRANCH_OWNED_EXIT=0
```

### 4.2 main-owned tests that main just rewrote — GREEN

The 11 test files `main` changed (court/placement/body/worldhmr/process_video):

```
264 passed, 1 error in 37.48s
MAIN_TOUCHED_POSTREBASE_EXIT=1
```

The single error is a **collection** error in `test_process_video.py`, pre-existing on `main`
(section 4.3). Exit 1 is caused by that pre-existing collection error, not a test failure.

### 4.3 The 28 errors — COLLECTION errors, 100% pre-existing on main

**These are collection (import-time) errors, not test failures.** The module never imports,
so its tests never run.

Attribution is exact, not inferred. The error-module list was captured at detached `9d318ad`
(main) and at the rebased tip, then diffed:

```
main (9d318ad):     28 collection errors, 4169 tests collected
rebased (35b44f3):  28 collection errors, 4286 tests collected
diff of the two error-module lists: EMPTY
```

**Zero rebase-induced collection errors.** The branch adds 117 collectible tests and breaks
no imports.

Root causes:

- **27 of 28** — `BestStackManifestError`: the `best_stack` manifest points at assets that are
  **gitignored and therefore absent from this worktree** (`models/checkpoints/`, `runs/**`).
  Representative text:

```
___________ ERROR collecting tests/racketsport/test_process_video.py ___________
tests/racketsport/test_process_video.py:15: in <module>
    from scripts.racketsport import process_video
scripts/racketsport/process_video.py:122: in <module>
    from threed.racketsport import orchestrator  # noqa: E402
threed/racketsport/orchestrator.py:22: in <module>
    from .body_compute import (
threed/racketsport/body_compute.py:21: in <module>
    DEFAULT_BODY_SKELETON_STRIDE = int(load_best_stack_manifest().value("body.skeleton_stride"))
threed/racketsport/best_stack.py:164: in load_best_stack_manifest
    manifest = _build_manifest(manifest_path, raw)
E   threed.racketsport.best_stack.BestStackManifestError: best_stack entry
    'ball.wasb_checkpoint' points at missing path
    .../models/checkpoints/wasb/wasb_tennis_best.pth.tar
```

  This is a **worktree provisioning gap, not a code defect**. Symlinking the checkpoints in
  advanced the error to the next missing asset
  (`runs/waveb_confidence_gate_20260702T183158Z/calibration_curves.json`), confirming an asset
  chain. Those symlinks were then **removed**: pointing a test tree at the protected main
  checkout risks write-through, which was not acceptable under the safety fence.

- **1 of 28** — `tests/racketsport/test_audit_data_utilization.py`:
  `ImportError: cannot import name 'REQUIRED_CONTRACT_ASSET_IDS' from 'scripts.racketsport.audit_data_utilization'`.
  This **confirms the other lane's datapoint**: present on both `main` and this branch, and a
  genuine pre-existing repo defect (not asset-related).

### 4.4 Test failures — CLEAN FULL RUN: 72 failed, 71 of them pre-existing on main

**Clean, complete run** at rebased tip `35b44f3` (see 4.5 for invocation):

```
72 failed, 4179 passed, 35 skipped, 6 warnings, 28 errors in 2849.71s (0:47:29)
```

This run is **complete, not truncated**: 72 + 4179 + 35 = **4286**, exactly the collected
count. (Two `KeyboardInterrupt` strings in the log are source lines inside a traceback, not
an interruption.)

**Attribution of all 72 — every one re-run at detached `9d318ad`:**

| Bucket | Count | Evidence |
|---|---|---|
| Pre-existing on `main` | **71** | 69 from the first attribution batch + 2 `test_train_court_model_v2` |
| In-suite ordering fragility (main-owned) | **1** | `test_court_model_infer` |
| **Rebase-induced** | **0** | — |
| In branch-owned test files | **0** | verified per file |

- **69 of 72** were re-run at `9d318ad`: `69 failed, 2 passed in 72.20s`, `ATTRIB_MAIN_EXIT=1`.
- **2 more** (`test_train_court_model_v2`, absent from the earlier truncated run because it
  never reached them) re-run at `9d318ad`: `2 failed, 1 passed in 4.16s`, `ATTRIB3_MAIN_EXIT=1`
  — both fail with `FileNotFoundError` at `scripts/racketsport/train_court_model_v2.py:963`,
  a missing gitignored asset.
- **1 remaining**, `test_court_model_infer.py::test_infer_court_model_returns_stable_contract_keys_and_shapes`:
  **passes standalone on `main`** and **passes standalone on the rebased branch**
  (`1 passed in 6.73s`, exit 0, at load 2.65). It fails only inside the full suite. That is
  **ordering / shared-state fragility in a main-owned test**, not a functional regression:
  the branch modifies zero files this test depends on. Honest caveat: the branch adds 117
  tests, which can perturb execution order, so it may shift *which* ordering-sensitive tests
  trip — but the underlying fragility is main's and the test is green in isolation on both.

**Failure clustering (all main-owned):** `test_build_person_fewshot_pack` (10),
`test_build_pbvision_ball_sst` (9), `test_cli_help` (8), `test_coords_parity_real_fixture` (7),
`test_ball_stage2_training` (6), `test_ball_wasb_dataset` (4), and 19 other main-owned files.
The 8 `test_cli_help` failures are all for **main-owned** scripts (`process_video.py`,
`remote_body_dispatch.py`, `benchmark_person_trackers.py`, body scripts) — **not** the 3 CLIs
this branch adds.

The count reflects **scope, not breakage**: this is the entire `tests/racketsport` tree in a
worktree missing every gitignored asset. A blast-radius-scoped run sees only a handful. The
cross-lane datapoints are visible here too and are repo-wide conditions, not lane damage:
`test_scaffold_tool_index::test_real_scaffold_tool_index_matches_checked_in_schema` (1) and
the `test_cli_help` unknown-category CLI failures (8).

**Earlier truncated runs, recorded for honesty and superseded by the above:**
- Wide run on the branch: `71 failed, 3937 passed, 30 skipped, 28 errors in 2585.57s` —
  **SIGINT-killed at 43:05** by a concurrent agent whose `pytest tests/` process match was too
  broad; ~248 tests never ran.
- `main` baseline: killed at ~37% (exit 144, no summary): `1518 passed, 56 failed, 6 skipped`
  of 1580 — **partial**.

**Net: zero rebase-induced test failures and zero rebase-induced collection errors.**

### 4.5 Invocation of the clean run — and an exit-code gap I must flag

Relaunched via `python -c "...pytest.main([...])"` so the command line contains **no
`pytest tests/` substring** and could not be caught by a broad process kill. Marker
`BALLLANE_REBASE_SUITE_20260724`, **pid 56042**, at rebased tip `35b44f3`. It ran 47:29
undisturbed to completion.

> **EXIT CODE NOT OBSERVED — stated as a gap, not inferred.** I launched this run with
> `nohup ... &` and **failed to write `$?` to a file**. Each tool call gets a fresh shell, so
> once the process exited its status was unrecoverable. I therefore **cannot report an
> observed exit code for this specific run**, and I am not going to substitute pytest's
> documented convention for a measurement I did not take. What *is* directly observed is the
> summary line above (72 failed / 4179 passed / 35 skipped / 28 errors) and the complete
> 4286-test accounting.
>
> Exit codes reported elsewhere in this document **were** captured directly and unpiped:
> `BRANCH_OWNED_EXIT=0`, `MAIN_TOUCHED_POSTREBASE_EXIT=1`, `ATTRIB_MAIN_EXIT=1`,
> `ATTRIB3_MAIN_EXIT=1`, `ISOLATED_court_model_infer_ON_BRANCH_EXIT=0`, and the three
> structure checks in section 5.

## 5. Repo structure checks (AGENTS.md)

Run at the rebased tip; `main` baselines captured at `9d318ad` for comparison.

| Check | main `9d318ad` | rebased `35b44f3` | Attribution |
|---|---|---|---|
| `list_scaffold_tools.py --root .` | exit 0 | **exit 0** | pass |
| `audit_dead_code.py --root .` | exit 0, `status: pass` | **exit 0, `status: pass`** | pass |
| `audit_storage_policy.py --root . --json` | exit 1, `status: fail` | exit 1, `status: fail` | **pre-existing on main** |

- **Scaffold registry — clean additive merge.** 323 -> **326** tools (+3). All three new CLIs
  register with matching `related_test` and `direct_cli_reference_test`:
  `build_ball_observation_log` (BALL), `characterize_ball_solver` (BALL),
  `build_multimodal_event_dataset` (DATA). **Nothing removed.**
- **Dead code — clean.** 631 -> **638** Python sources (+7), `unknown_python_sources: 0`.
- **Storage policy — fails identically on `main`.** Field-by-field diff of the two JSON outputs
  shows **delta 0 on every field**: `unknown_large_tracked_files` 6 -> 6,
  `missing_allowed_large_untracked_source_files` 120 -> 120, `observed_large_tracked_files`
  17 -> 17. **New violations introduced by this branch: NONE.** The 6 unknown large tracked
  files belong to another lane (`runs/lanes/holdout_eval_20260721/vm_pull/...`); the 120
  "missing allowed" entries are gitignored assets absent from this worktree.

## 6. Remaining risks and known gaps

1. **Rebase base is stale — BASE DRIFT, the one thing that must be re-checked.**
   This branch is rebased onto `9d318ad`. During verification `main` advanced **four commits**
   to `2e073c8`: `a83ca9b` (isolate remote BODY parent from CUDA), `244c9c1` (propagate BODY
   foot phase agreement), `60631f1` (adapt refined skeletons for world assembly), `2e073c8`
   (fix replay camera help assertion). Re-rebase was deliberately **NOT** performed, to avoid
   invalidating the 47-minute test evidence against a target that was still moving.
   **The integration owner must re-rebase and re-check before merging.**
   Forward risk was measured and is low: intersection of the branch's 137 files with
   `ce32f1f..2e073c8` is still **0**. Note `60631f1` **touches `threed/racketsport/virtual_world.py`**,
   which is a genuine semantic dependency of this branch — but all 4 branch-imported symbols
   are still present at current `main` and no defs were removed.

2. **This worktree cannot produce a green suite.** `models/checkpoints/` and `runs/*` are
   gitignored, so 28 modules cannot import and many tests fail on absent fixtures. Equally true
   of `main` here. A clean-tree verdict requires a fully provisioned checkout.
3. **`test_audit_data_utilization.py` is a genuine pre-existing defect** (missing
   `REQUIRED_CONTRACT_ASSET_IDS` export), unrelated to this lane but needs an owner.
4. **Fixed wall-clock budgets are load-fragile.** `test_flight_simulator`'s `assert elapsed < 5.0`
   fails on an otherwise-healthy tree whenever the box is busy.
5. **VERIFIED=0.** Nothing here promotes any ball-lane capability.

## 7. Verdict

### READY_TO_MERGE — conditional on a re-check against current `main`

What this verdict rests on:

- The collision surface is **empty** (0 of 137 files co-edited), so there was no conflict to
  resolve and no opportunity to drop `main`'s work. Verified, not assumed.
- The rebase applied **cleanly, exit 0**, and is **provably content-preserving** (byte-identical
  delta, tree matches an independent dry run, evidence subtree hash unchanged).
- The only semantic coupling (`virtual_world.py`) was checked symbol-by-symbol and holds at both
  `9d318ad` and current `2e073c8` — including across `60631f1`, which modifies that very file.
- Branch-owned tests are **fully green (132 passed, exit 0)**, and **0 of the 72 suite failures
  land in a branch-owned test file**.
- Every failure and collection error was **attributed by re-running it on `main`**: **71 of 72**
  failures and **28 of 28** collection errors are pre-existing; the 1 remainder passes standalone
  on both branch and `main`. **Zero rebase-induced regressions.**
- Structure checks introduce **no new violations**; the scaffold registry merged additively.

This verdict explicitly does **not** rest on a clean suite — the suite is not clean, on this
branch or on `main`, because this worktree lacks gitignored assets. It rests on *differential*
evidence: the branch changes nothing `main` changes, and every failure reproduces without it.

Conditions before merge:

1. Re-rebase onto current `main` (`2e073c8` or later) and confirm still conflict-free (measured: intersection is still 0).
2. Re-run at minimum the 9 branch-owned test files after that re-rebase.
3. Merge from a checkout that has the gitignored model/run assets, so the 28 collection errors
   cannot mask a real import break.
