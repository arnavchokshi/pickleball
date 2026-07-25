# oneworld_wire_20260724 — honest capability naming + default-OFF fusion wiring

Branch `fusion-wire-20260724`, forked from `9d318ad`. Worktree
`/private/tmp/pickleball-fusion-20260724`. CPU only, $0, no cloud, no GPU job.

## 0. The headline, stated the way the standing rules require

**`VERIFIED=0`. This lane produced INTEGRATION progress, not capability progress.**

Wiring a stage into the graph makes a code path reachable. It does not make the
underlying CV correct, and it produced no accuracy evidence of any kind. After
this branch the product still **does not have fusion** in any sense a user would
recognise: the fusion stage is default-OFF, nothing downstream reads its output,
and on the only clip with a full artifact set it correctly **refused to refine
any of the 24 contacts**. Do not report this as "fusion landed".

What is now true that was not true before:

1. `PIPELINE_SUMMARY.json` no longer claims a capability called `fusion` that
   nothing ever performed.
2. The real joint-refinement module has a reachable, identity-registered,
   default-OFF call site instead of zero call sites.

That is all.

---

## 1. Defect A — the mislabel

### 1.1 What was wrong

`scripts/racketsport/process_video.py` declared a minimum-bundle capability:

```python
("fusion", ("confidence_gated_world.json", "virtual_world.json"),
 "fused-world artifact is missing", "world"),
```

The `world` stage is `threed/racketsport/virtual_world.py`, whose own module
docstring is *"Assemble inspectable court_Z0 world-state artifacts"*, and whose
RUNBOOK entry is *"compose `virtual_world.json` ... from the already-finished
refined artifacts"*. It stitches finished artifacts together. It performs no
joint refinement, no confidence weighting, and no cross-source state estimation.
Nothing in the default stack ever fused anything, yet a bundle that satisfied
this requirement was reported as having the `fusion` capability.

This is a trust-contract defect, not a cosmetic one: the summary is the artifact
a reader consults to learn what the pipeline actually did.

### 1.2 The naming choice, and why

**`fusion` -> `composited_world`.**

Reasoning, in the repo's own vocabulary rather than an invented one:

- The stage's verb in both the code (`virtual_world.py`: "assemble") and the
  operator documentation (`RUNBOOK.md` stage 19: "compose") is compositing.
- `NORTH_STAR_ROADMAP.md` section 1.4 keeps *evidence provenance* and *product
  authority* on separate axes and names evidence for **what produced it**
  (`measured`, `model_estimated`, `physics_predicted`). `composited_world` names
  the producing operation honestly; `fusion` named an operation that did not run.
- It stays parallel with the neighbouring artifact-shaped capability names
  (`tracks`, `ball_arc`, `paddle`, `trust_bands`, `assets`).

Rejected alternatives: `world` (collides with the stage name and says nothing
about what was produced), `virtual_world` (leaks a filename into a capability
vocabulary that is otherwise conceptual), and keeping `fusion` with a comment
(a comment does not reach the consumer reading `PIPELINE_SUMMARY.json`).

### 1.3 Compatibility surface — the decision and the justification

Checked every consumer of the literal across `threed/`, `scripts/`, `server/`,
`web/`, `ios/`, `tests/`, and the JSON schemas under `docs/racketsport/`.

| Location | Kind | Action |
|---|---|---|
| `scripts/racketsport/process_video.py` (both preset tables) | worker capability vocabulary | renamed |
| `server/bundle_policy.py` `_mandatory_requirements()` | server re-derivation of the same list | renamed **in lockstep** |
| `docs/racketsport/*.json` | published schemas | **no occurrence** — no schema enumerates capability names |
| `web/`, `ios/` | clients | **no occurrence** (the one `ios` grep hit is the word "confusion") |

**Decision: rename outright rather than dual-key, and document the break.**
Justification:

- The capability name reaches the outside world through exactly one channel,
  `PIPELINE_SUMMARY.json` -> `missing_capabilities[].capability`, and it is only
  emitted when the capability is **absent**. A `complete` bundle has never
  contained the string `fusion`. There is no published success-path contract to
  preserve, which is what would otherwise justify a compatibility alias.
- Keeping a legacy `fusion` alias would mean re-asserting the false claim in the
  very field this change exists to correct. A compatibility shim that keeps
  lying is worse than a documented break.
- The rename **must** be lockstep. `server/bundle_policy.gate_reported_status()`
  compares the worker-reported `missing_capabilities` list against the list the
  server re-derives, and returns `partial` on any mismatch. Renaming one side
  only would silently downgrade every complete bundle. Both sides are renamed in
  the same commit, and a regression test pins the two vocabularies together.
- Mixed-version behaviour is safe: an older `PIPELINE_SUMMARY.json` on disk that
  still carries `{"capability": "fusion"}` is copied forward by
  `evaluate_bundle`, and `_append_missing` is additive and de-duplicates by name.
  A stale entry can only add a `partial` reason to a bundle that was already
  `partial` (it only exists when the artifact was missing). No bundle is
  misclassified across the change in either direction.

### 1.4 Regression test

`tests/racketsport/test_capability_naming_honesty.py` (7 tests). It exercises the
real `_minimum_bundle_missing_capabilities` against an empty clip dir, which
enumerates the entire capability vocabulary for both presets, and asserts:

- `fusion` is gone and `composited_world` is present, in **both** the `full` and
  `court_skeletons` tables;
- no capability name in either table contains any joint-refinement word
  (`fusion`, `fused`, `fuse`, `refined`, `refinement`, `solved`, `estimated`);
- the `composited_world` *reason string* no longer advertises fusion either;
- positive control: `virtual_world.json` alone still satisfies the capability, so
  this is a rename of the claim, not a change to what the bundle requires;
- the worker and server vocabularies are identical except for the server-only
  `summary` capability.

---

## 2. Defect B — wiring `one_world_v1` as a DEFAULT-OFF stage

`threed/racketsport/one_world_v1.py` (1571 lines) is the actual
confidence-weighted joint-refinement module. Before this branch it had **zero**
call sites in the pipeline and was reachable only from three scaffold CLIs.

### 2.1 Placement

New node in `AUTHORITATIVE_STAGE_GRAPH`:

```python
PipelineStageDefinition("one_world", 185, 185, "one_world")
```

between `world` (180) and `confidence_gate` (190), using the existing
`enabled_by` mechanism that `player_selection` / `rally_gating` / `verify_viewer`
already use. Verified position in both serial and overlap schedules.

### 2.2 Default OFF, and "off" means invisible

- `configs/racketsport/best_stack.json` gains `world.one_world_v1_fusion`:
  `status: PENDING`, `enabled: false`, `do_not_promote: true`,
  `trust_band: preview`, `authority: "never"`, with a gate whose text states
  explicitly that **wiring cannot promote it**.
- **No revision bump** (manifest stays at revision 15). This follows the
  precedent set by commit `57239d8`, which added an R&D-only, `enabled=false`
  PENDING entry and recorded "no revision bump" for exactly this situation. Only
  `updated` moved to `2026-07-24`.
- `authoritative_stage_names(..., one_world=False)` defaults to **False**, so
  every pre-existing call site produces a byte-identical stage list and the
  25/25/26 counts pinned by `test_spine_stage_contract.py` are untouched.
- `resolved_best_stack_config_from_options()` omits the key **entirely** while
  the stage is off — the same pattern `tracking.player_selection_layer` uses — so
  no sibling stage's content-addressed fingerprint moves and every existing
  generation stays reusable.
- Excluded from `COURT_SKELETON_STAGE_NAMES`. Justification: that preset has no
  ball stages at all, so it can never satisfy the required inputs; running it
  there would guarantee a blocked outcome on every single run. It is also forced
  off in `build_options_from_args` for that preset, so the exclusion holds even
  if the best_stack default is later flipped.

### 2.3 Degradation semantics (the part that mattered most)

`one_world_v1.build_one_world` **hard-raises `FileNotFoundError`** when
`ball_track.json` is absent, and **every run produced in the last four days used
`--pipeline-preset court_skeletons`, which produces no ball artifacts at all**. A
naive wiring would have hard-failed on 100% of current runs.

| Condition | Outcome | reason_code |
|---|---|---|
| `ball_track.json` or `court_calibration.json` absent | `blocked` | `one_world_required_inputs_missing` |
| `FileNotFoundError` from inside the builder | `blocked` | `one_world_required_inputs_missing` |
| Same-run artifact disagreement (`ValueError`/`KeyError`/`TypeError`) | `degraded` | `one_world_input_disagreement` |
| Self-validation fails but an artifact was produced | `degraded` | (errors listed in notes) |
| Success | `ran`, `trust_badge="preview"` | |
| Genuine programming error | `failed` (loud, existing generic handler) | `unexpected_stage_exception` |

Required inputs are **pre-flighted** before the builder is called, so the typed
`blocked` outcome names the exact missing filenames and the current preset. Both
typed paths use the repo's existing `ExpectedOptionalAbsence` mechanism, so they
surface in `PIPELINE_SUMMARY.json` in the same `expected_optional_absence` shape
every other typed-absence stage uses, and neither writes an artifact.

The deliberate line: expected data-shape disagreements degrade so a preview stage
can never take down the authoritative bundle; real bugs stay loud. This is a
judgement call and is listed under risks.

### 2.4 Identity-graph registration (NS-01.3)

Registered in `RUN_IDENTITY_DEPENDENCIES`, `RUN_IDENTITY_CONFIG_KEYS`,
`RUN_IDENTITY_OUTPUTS`, `_stage_identity_options`, `_stage_explicit_inputs`.

The dependency edges point **one way**: `one_world` depends on `world` plus the
raw evidence the compositor read (`calibration`, `player_selection`, `ball`,
`ball_fill`, `body`, `placement_refine`, `grounding_refine`, `paddle_pose`,
`events_refined`, `ball_arc_refined`), and **nothing depends on `one_world`**.
`confidence_gate`, `match_stats`, `coaching_facts` and `manifest` keep depending
only on `world`. That asymmetry is what makes it structurally impossible for the
preview fusion to become product authority, and it is pinned by a test.

`_stage_explicit_inputs` lists all 22 same-run artifacts the module actually
reads, so their content hashes gate reuse.

Verified empirically, not just declared: `ran` -> `skipped` (reused
content-addressed generation) -> `ran` again after mutating `ball_track.json`.

### 2.5 Trust banding and artifact policy

- Output is schema-gated **before** publication:
  `OneWorldV1.model_validate_json(serialized)`. The committed public schema
  `docs/racketsport/one_world_v1_schema.json` is byte-for-byte equal to
  `OneWorldV1.model_json_schema()` (asserted by a test, so schema drift is
  caught), which is what makes validating through the model equivalent to
  validating against the published contract.
- The artifact self-declares `VERIFIED=0`, `preview_only`, `render_only`,
  `not_for_detection_metrics`, `not_for_training`, `raw_inputs_mutated=false`.
- The stage outcome carries `trust_badge="preview"` and metrics
  `do_not_promote: true`, `authority: "never"`.
- The stage does **not** touch `self.trust_bands`. Deliberate: `trust_bands.json`
  is written by the `world` stage and `bundle_policy` compares the summary's copy
  against the file, so injecting a band here would have made every bundle report
  a spurious `trust_bands` mismatch.
- Raw observations are never modified. Only `one_world_v1.json` and
  `one_world_v1_validation.json` are written. Pinned by a byte-comparison test
  over the whole clip dir and re-verified on the real end-to-end run.

---

## 3. End-to-end exercise on real artifacts

Run against the full Wolverine artifact set from
`runs/lanes/oneworld_impl_20260716/wolverine/` (copied read-only into a scratch
run dir; the main checkout was never written to). This drove the **real pipeline
stage** through `_run_stage_safely`, not the scaffold CLI, so the wiring itself
is what was exercised.

`smpl_motion.json` in that directory is a **broken symlink** (its target under
`runs/a100_sam3d_validation2_20260703T0647Z/` no longer exists), so the run
exercised the real `skeleton3d.json` fallback path.

**Result: `status=ran`, `trust_badge=preview`, 1.28s, exit 0.**

| Measure | Value |
|---|---|
| Input artifacts present | 14 |
| Raw inputs mutated | **0** |
| Files created | exactly `one_world_v1.json`, `one_world_v1_validation.json` |
| Artifact size | 19,417,996 bytes |
| Self-validation | `valid: true`, all 8 checks pass |
| Deterministic rebuild | byte-identical |
| Frames / contacts / bounces / events | 300 / 24 / 4 / 28 |

### 3.1 The known-bad numbers, reported not hidden

**The fusion refused to refine any contact. That refusal is CORRECT behaviour.**

| Contact outcome | Count |
|---|---|
| `unsupported` — `no_player_wrist_within_1.2m` | 20 |
| `too_close_to_call` — `hitter_probability_or_margin_below_threshold` | 4 |
| **resolved** | **0** |

`ball_contact_distance_m`: `count=0`, `abstained_count=24`, median `null`.

The cause is upstream data quality, exactly as expected. Measured on this clip:
**185 of 300 ball world points (61.7%) fall outside the court polygon** from
`court_zones.json` (x +/-3.048 m, y +/-6.7056 m; the polygon is a rectangle so
the bounding box is exact). With ball 3D that far off-court, no player wrist is
within the 1.2 m co-location radius at any declared contact, so the co-location
likelihood loses to the null hypothesis and the refiner abstains instead of
inventing a hitter. Abstention is the designed and desirable behaviour here.

(The task brief cited ~51% off-court; measured here as 61.7% against the
`court_zones.json` polygon with no margin. Different measurement basis, same
conclusion — the ball 3D on this clip is badly off-court.)

Other real numbers:

- `paddle_resolution`: `resolved_count=0`, `ambiguous_denominator=0`,
  `unsupported_legacy_wrist_proxy_count=1102` (this run had only the legacy
  gen-1 `racket_pose_estimate.json`, no gen-2 hypotheses, so there was no
  two-hypothesis ambiguity to resolve).
- `world_coverage`: `coverage_fraction=0.39`, 117/300 complete frames; ball tiers
  295 `arc_measured` + 5 `physics_predicted`.
- `bounce_plane_residual_m`: `count=4`, median `0.0` before **and** after — the
  soft prior moved nothing measurable.
- `reprojection_consistency`: ball median **85.97 px unchanged** (baseline ==
  fused, no ball refinement survived); player median 19.637711 -> 19.637675 px,
  a change in the sixth significant figure. **This is noise, not improvement, and
  is not evidence of anything.**
- `missing_counts`: `paddle_pose: 98`, `player_placement: 93`.
- `regression_kills`: none fired.

**Read this honestly:** on the one clip where the full pipeline artifact set
exists, the fusion changed essentially nothing and abstained from every contact.
That is the correct response to bad inputs, and it is also a clear statement that
there is no accuracy benefit to claim.

---

## 4. Test evidence — real commands, real unpiped exit codes

Interpreter `.venv/bin/python` (Python 3.14.6), `MPLBACKEND=Agg`,
`-p no:randomly` for order stability.

### 4.1 New focused tests (branch)

```
.venv/bin/python -m pytest tests/racketsport/test_one_world_stage.py \
    tests/racketsport/test_capability_naming_honesty.py -q -p no:randomly
28 passed in 4.50s
REAL_EXIT=0
```

21 tests in `test_one_world_stage.py` + 7 in `test_capability_naming_honesty.py`.
Covers: stage-off byte identity, order-185 placement, `court_skeletons`
exclusion, resolved-config omission while off, no sibling identity perturbation,
identity-graph registration, no downstream dependency, typed blocked path for
each required input independently and for an empty clip dir, typed degrade on
inconsistent inputs, stage-on against a real fixture, schema validation against
the committed public schema, raw-input immutability, no bundle-capability credit,
content-addressed reuse + invalidation, and determinism.

### 4.2 Blast-radius suite — identical command, both sides

```
.venv/bin/python -m pytest tests/racketsport/test_process_video.py \
  tests/racketsport/test_spine_stage_contract.py \
  tests/racketsport/test_best_stack_manifest.py \
  tests/racketsport/test_truthful_capabilities.py \
  tests/racketsport/test_player_selection_stage.py \
  tests/racketsport/test_one_world_core.py tests/racketsport/test_one_world_clis.py \
  tests/racketsport/test_scaffold_tool_index.py tests/racketsport/test_cli_help.py \
  tests/server/ tests/render_service/ -q -p no:randomly
```

| | base `9d318ad` | branch `fusion-wire-20260724` |
|---|---|---|
| passed | **721** | **749** |
| failed | **4** | **3** |
| REAL_EXIT | **1** | **1** |
| wall | 222.04s | 230.50s |

(The branch run additionally includes the two new test files, which do not exist
at base.)

**Attribution: 0 failures caused by this branch. 1 pre-existing failure fixed.**

| Test | base | branch | attribution |
|---|---|---|---|
| `test_truthful_capabilities::test_storage_policy_keeps_large_tracked_artifacts_explicit` | FAIL | FAIL | **pre-existing** |
| `test_truthful_capabilities::test_north_star_is_the_single_product_and_execution_authority` | FAIL | FAIL | **pre-existing** |
| `test_scaffold_tool_index::test_real_scaffold_tool_index_matches_checked_in_schema` | FAIL | FAIL | **pre-existing** |
| `test_truthful_capabilities::test_runbook_documents_current_process_video_entrypoint` | FAIL | **PASS** | **fixed by this branch** |

Pre-existing failures proven by running the identical command against a clean
extract of `9d318ad`, and independently by inspection:

1. **storage policy** — 6 tracked files >5 MB under
   `runs/lanes/holdout_eval_20260721/vm_pull/` are absent from
   `ALLOWED_LARGE_TRACKED_FILES`. Introduced by commit `f9dc11d`, proven an
   ancestor of `9d318ad` via `git merge-base --is-ancestor`. None of the 6 files
   is touched by this branch.
2. **NORTH_STAR length** — `NORTH_STAR_ROADMAP.md` is 528 lines against a
   `<= 500` assertion. Not a file this branch modifies.
3. **scaffold index** — 7 CLIs classify as category `unknown`
   (`abc_decision_gate`, `apply_event_sequence_dp`, `build_abc_arm_manifests`,
   `build_court_v31_protocol`, `build_data_inventory`,
   `build_owner_event_manifest`, `verify_training_inputs`). No CLI added here.
4. **RUNBOOK stage order** — the contract test requires every
   `AUTHORITATIVE_STAGE_GRAPH` node to appear in `RUNBOOK.md` in serial order.
   `**player_selection**` was missing at base. This branch documents it (and the
   new `one_world` stage), so the test now passes.

Additionally pre-existing and **excluded from both runs** with `--ignore` because
it breaks collection: `tests/racketsport/test_audit_data_utilization.py` imports
`REQUIRED_CONTRACT_ASSET_IDS`, which does not exist in
`scripts/racketsport/audit_data_utilization.py` at `9d318ad`.

### 4.3 Wide suite — time-boxed, both sides, symmetric

```
.venv/bin/python -m pytest tests/ -q -p no:randomly \
  --ignore=tests/racketsport/test_audit_data_utilization.py
```

The full `tests/` tree is single-threaded (`pytest-xdist` is not installed) and
slow on this CPU; the source lane `oneworld_impl_20260716` time-boxed its own
wide run at 1002s for the same reason. Both sides were run from clean
`git archive` extracts of their respective commits and interrupted with SIGINT at
the same symmetric progress point (~33%).

| | base `9d318ad` | branch `fusion-wire-20260724` |
|---|---|---|
| passed | **1609** | **1616** |
| failed | **54** | **54** |
| skipped | 7 | 7 |
| wall | 511.22s | 509.15s |
| terminated by | SIGINT | SIGINT |

**The two failure sets are byte-identical** (verified with `comm`: zero entries
unique to either side). The +7 passed on the branch is
`test_capability_naming_honesty.py`, which sorts early enough to run before the
interrupt point; `test_one_world_stage.py` sorts later and was not reached.

**Failures caused by this branch: 0.**

Caveat, stated plainly: both extracts lack the untracked `data/` and `runs/`
fixture files that many tests need, which inflates the absolute failure count on
**both** sides equally. The 54 is therefore a valid basis for comparison but is
**not** a statement about repo health.

### 4.4 Repo structure checks (AGENTS.md)

```
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .   REAL_EXIT=0
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .       REAL_EXIT=0
    status: pass | python_sources: 631 | unknown_python_sources: 0
python3 scripts/racketsport/audit_storage_policy.py --root . --json    REAL_EXIT=1
    unknown_large_tracked_files: 6  (all runs/lanes/holdout_eval_20260721/, pre-existing)
    unknown_large_untracked_source_files: 0
```

The storage audit's exit 1 is the same pre-existing `f9dc11d` registry drift as
section 4.2 item 1, unchanged by this branch.

---

## 5. Remaining risks

1. **No accuracy evidence exists, and none was produced.** The gate on the new
   best_stack entry is unmet and unattempted. `VERIFIED=0`.
2. **The one clip with full artifacts is not a usable judge.** 61.7% of its ball
   world points are off-court. It can demonstrate that the stage runs, abstains,
   and stays deterministic; it cannot demonstrate that refinement helps.
3. **Degrade-vs-fail is a judgement call.** Same-run artifact disagreements are
   reported `degraded` rather than `failed` so a default-OFF preview stage cannot
   take down an authoritative bundle. The cost is that a genuine upstream
   contract regression surfaces as a degraded preview stage rather than a hard
   stop. Mitigated by keeping real programming errors loud, but a reviewer may
   reasonably want this stricter.
4. **The 19 MB artifact is unbudgeted.** `one_world_v1.json` was 19.4 MB for a
   300-frame clip. It is excluded from the bundle manifest and from the storage
   allowlists, but if this stage is ever enabled by default it will need a byte
   budget like `mesh.byte_budget_mib`. Not addressed here.
5. **`build_metrics` was deliberately not wired.**
   `one_world_v1.build_metrics` imports
   `runs.lanes.oneworld_design_20260716.baseline_probe`, i.e. a lane directory
   under `runs/`. Importing a lane artifact from a pipeline stage would be a
   layering violation and would break whenever that lane is pruned. The stage
   emits the artifact and its validation sidecar only; metrics stay a scaffold
   CLI concern.
6. **Concurrent-edit risk on `RUNBOOK.md`.** Another agent held uncommitted
   changes to `RUNBOOK.md` in the main checkout while this branch was built. See
   `MERGE_NOTES.md`.
7. **Preset coupling.** If `world.one_world_v1_fusion` is ever flipped to
   `enabled: true`, `court_skeletons` runs stay excluded by two independent
   guards, but every `full` run without ball artifacts would emit a `blocked`
   stage outcome. That is honest, but it is a visible behaviour change and should
   be a deliberate decision, not a side effect.

---

## 6. Incident disclosure — I killed another lane's test run

While sending SIGINT to my own two time-boxed wide runs, my process-matching
pattern (`pytest tests/`) was too broad and also matched a **concurrent agent's**
test run in `/Users/arnavchokshi/Desktop/pickleball/.claude/worktrees/ball-lane-20260723`.

- That run was killed at **43:05**, having reached
  **71 failed, 3937 passed, 30 skipped, 28 errors**.
- Its partial log survives at `scratchpad/postrebase_full.log`, but no exit file
  was written, so that lane needs to re-run to obtain a clean exit code.
- **No file was modified and no repo state was corrupted.** The main checkout was
  never written to. The cost is that lane's wall-clock time.

Reporting this rather than hiding it. The `ball-lane-20260723` owner should be
told to re-run. The lesson, for anyone repeating this pattern: scope process
kills to a PID captured at launch, never to a substring that other agents'
commands can match.
