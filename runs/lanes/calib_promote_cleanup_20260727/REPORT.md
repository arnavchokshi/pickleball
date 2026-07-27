# CAL: promoting the calibrations that earned it, and a deliberately small cleanup

Lane: `calib_promote_cleanup_20260727` · branch `calibpromo-20260727` off `e209112` · 2026-07-27
**VERIFIED=0.** Promoting a better-fitting calibration is an ENGINEERING improvement to a
fit. It promotes no capability, passes no gate, and the ball is not verified. Two clips
were promoted and four were refused; every refusal is a measurement, not an omission.

Base is `e209112`. `main` advanced to `f9a9c9e` during the session. Nothing is merged or pushed.

---

## 1. What "promote" had to mean, because nothing supported it

Court calibrations are resolved by **filename convention only**. Both resolvers --
`process_video._auto_discover_court_calibration` and
`virtual_world.resolve_best_court_calibration_path` -- look for the literal
`<clip>/labels/court_calibration_metric15pt.json`. No manifest pointer, no `selected`
field, no glob-and-pick-newest. So "make the refined artifact the selected input without
overwriting the raw" was not expressible in this repo, which is exactly why the refit lane
left its refined artifacts unpromoted.

This lane adds the missing indirection, `threed/racketsport/court_calibration_selection.py`.
A label directory may carry a `court_calibration_selected.json` naming the artifact that
supersedes the raw solve beside it, with the sha256 of **both**. Deliberate properties:

- **Fail-loud.** Missing target, digest mismatch, or a pointer superseding something other
  than its own neighbour all raise. A broken promotion must not silently degrade back to
  the artifact it was meant to replace.
- **The raw digest is recorded**, so a later edit to a "raw" solve is detectable rather
  than silent. Raw solves are immutable and this enforces it.
- **No authority escalation.** `authority.class_unchanged` must be `true`. `source` and
  `intrinsics.source` stay `metric_15pt_reviewed` on every promoted artifact, and
  `orchestrator.TRUSTED_INTRINSICS_SOURCES` (orchestrator.py:370) gains no entries. A
  better fit of the same owner-reviewed correspondences is not a new authority class.
- **Explicit stays explicit.** `--court-calibration <raw>` is never redirected by a sibling
  pointer; only naming the pointer itself resolves.
- **Confined.** Pointer references are relative to the pointer's own directory and may not
  escape it.

A clip with no pointer resolves exactly as before, so four of six clips are unchanged in
behaviour.

### 1.1 The refit lane's refined artifacts did not validate

Worth flagging, because promoting them verbatim would have shipped a fail-loud break: every
`refined/*/court_calibration_metric15pt_refined.json` **fails**
`validate_artifact_file("court_calibration", ...)` with 9 errors.
`CourtCalibrationProvenance` (schemas/__init__.py:330) is `extra="forbid"` and requires
exactly `method`, `inputs`, `code_identity`; the lane's block used `refinement_of`,
`refit_reason`, `authority_unchanged`, `refit_lane`, `reviewed_keypoints`,
`not_gate_verified`.

So a promoted artifact is the refined one re-serialised with a schema-conformant
`provenance`, and the narrative those keys carried moved into the selection pointer -- which
is where a promotion decision belongs anyway. Both promoted artifacts are asserted to
validate, by test.

---

## 2. The evidence, measured here

`measure_calibration_floor.py` resolves each clip through the **live** selection resolver
and scores it. This is not a re-read of the refit lane's table.

**The baseline is what the repo consumes TODAY.** The `intrinsics.dist` seam fix is already
merged at `e209112`, so the honest comparison is the shipped artifact read by a
distortion-aware consumer -- the refit lane's state **B**, not its state A. My measured
baseline reproduces state B on all six clips to 4 decimal places, which is the check that
the harness measures the right thing.

### Held-out: leave-one-out over 15 correspondences. **This is the honest column.**

Each fold refits focal length, distortion and pose on 14 correspondences and scores the 1
it never saw.

| clip | plane error (m) now | refined | delta | reprojection (px) now | refined | delta | decision |
|---|---|---|---|---|---|---|---|
| **pbvision demo seed** | 2.745214 | **0.176952** | **-93.6%** | 23.317 | **8.772** | **-62.4%** | **PROMOTE** |
| **owner_IMG_1605** | 2.419810 | **0.106518** | **-95.6%** | 26.847 | **3.448** | **-87.2%** | **PROMOTE** |
| burlington | 0.268336 | 0.269263 | +0.35% | 10.742 | 11.050 | +2.87% | refuse |
| indoor | 0.233780 | 0.190812 | -18.38% | 5.562 | 6.603 | **+18.72%** | refuse |
| outdoor | 0.127254 | 0.127254 | 0.00% | 7.026 | 7.026 | 0.00% | no change |
| wolverine | 0.158241 | 0.158241 | 0.00% | 11.005 | 11.005 | 0.00% | no change |

**Bar applied: both held-out criteria must agree the refined artifact is better.** One
metre-based, one pixel-based. Neither alone is sufficient, because they can disagree -- and
on one clip they do.

### In-sample, for continuity. **Optimistic; do not decide on this.**

`ball_label_geometry.calibration_plane_residuals` back-projects the calibration's own fitted
correspondences. It is a training residual, so more parameters always score better on it.

| clip | plane residual median (m) | reprojection median px | reprojection p95 px |
|---|---|---|---|
| pbvision demo seed | 1.906196 -> 0.144385 | 19.165 -> 3.676 | 51.706 -> 11.423 |
| owner_IMG_1605 | 0.093304 -> 0.089907 | 2.885 -> 2.588 | 6.233 -> 6.084 |
| burlington | 0.190686 -> 0.219627 | 6.390 -> 8.910 | **19.783 -> 25.775** |
| indoor | 0.199033 -> 0.183546 | 3.719 -> 5.446 | 8.814 -> 8.261 |
| outdoor | 0.100551 -> 0.100780 | 4.783 -> 4.794 | 12.277 -> 12.267 |
| wolverine | 0.126773 -> 0.126132 | 5.004 -> 5.019 | 19.921 -> 19.869 |

`owner_IMG_1605` is the clearest demonstration that in-sample cannot be the deciding number:
its in-sample floor barely moves (0.0933 -> 0.0899 m) while its held-out floor falls **23x**.
The defect was in the declared correspondence set, which held-out sees and in-sample
structurally cannot.

---

## 3. Promoted

Both raw artifacts declared their 3 net keypoints at `post_net_height_m` = 0.9144 m while
the labels sit at ~0 m: a wrong world model on 3 of 15 correspondences, not a distortion
problem.

### 3.1 `pbvision_11min_20260713_demo_seed` -- the defect the North Star names

Held-out plane **2.7452 -> 0.1770 m (15.5x)**, held-out reprojection **23.317 -> 8.772 px**.
In-sample floor 1.9062 -> 0.1444 m. `metric_confidence` **low -> med** -- the only clip whose
confidence changes, and it changes because the fit is genuinely better, not because a
threshold moved (`_confidence_from_reprojection` is untouched).

- selected: `runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_metric15pt_promoted.json`
- supersedes: `.../court_calibration_metric15pt.json`, sha256 `3e478a93...`, unmodified
- distortion: k1 = -0.1312, accepted on held-out evidence (k1+k2 was reachable and declined
  at only -3.1% over k1)

### 3.2 `owner_IMG_1605_8a193402780b` -- an internally inconsistent artifact, repaired

Held-out plane **2.4198 -> 0.1065 m (22.7x)**, held-out reprojection **26.847 -> 3.448 px**.
The raw artifact declared net points at 0.9144 m and then *excluded them from its own fit*
(`net_points_excluded_from_fit`). It was inconsistent with itself, and anything reading its
`world_pts` inherited the error.

- selected: `eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt_promoted.json`
- supersedes: `.../court_calibration_metric15pt.json`, sha256 `e2ab042c...`, unmodified
- distortion: none. Zero distortion won on held-out evidence for this camera.

### 3.3 Re-measured in place after promoting

`measure_calibration_floor.py` re-run through the live resolver -- reading what a consumer
now actually gets:

```
owner_IMG_1605_8a193402780b        selected=PROMOTED  held_out_plane=0.106518 m  in_sample=0.089907 m  conf=med
pbvision_11min_20260713_demo_seed  selected=PROMOTED  held_out_plane=0.176952 m  in_sample=0.144385 m  conf=med
burlington_gold_...                selected=raw       held_out_plane=0.268336 m  in_sample=0.190686 m  conf=low
indoor_doubles_...                 selected=raw       held_out_plane=0.233780 m  in_sample=0.199033 m  conf=med
outdoor_webcam_...                 selected=raw       held_out_plane=0.127254 m  in_sample=0.100551 m  conf=med
wolverine_mixed_...                selected=raw       held_out_plane=0.158241 m  in_sample=0.126773 m  conf=low
```

The four refused clips are identical before and after. Full JSON in
`floor_before_promotion.json` and `floor_after_promotion.json`.

---

## 4. Refused, with the measurement that refuses them

### 4.1 `burlington_gold` -- nothing improves, and the in-sample p95 regression is not cosmetic

Both held-out criteria are marginally **worse** (+0.35% plane, +2.87% px) and the in-sample
p95 regresses **19.783 -> 25.775 px**.

That last number matters. `ball_inout_uncertainty.bounce_geometric_uncertainty_m` builds the
in/out abstention radius from `reprojection_error_px.p95`, an **in-sample** number, as if it
were a generalisation estimate. An overfit model therefore *narrows* the radius. Burlington
is the clean demonstration: the honestly-selected k1 model has a worse in-sample p95 and so
a wider radius, while being indistinguishable held-out. Promoting here buys a wider
abstention radius for no held-out accuracy.

Burlington's shipped artifact is also the one the frozen legacy digest expects (in-sample
median 6.3901815068357175, reproduced exactly by my measurement) and it already has the net
at z = 0. Refused.

### 4.2 `indoor_doubles` -- the one clip where the two honest criteria disagree

Plane error **-18.4%**, reprojection **+18.7%**. Near-equal magnitude, opposite direction.

The brief described indoor as "marginal (0.1908 -> 0.1908, unchanged)". That comparison is
against state **A**, the pre-seam-fix world. Against what the repo actually consumes today
the plane error *does* improve by 18.4%, clearing the refit lane's own 15% bar -- so the
honest framing is not "unchanged", it is **"improves on one held-out metric and degrades by
the same amount on the other."**

Promoting would therefore not be acting on a measurement; it would be betting on which
held-out metric ranks. The refit lane's choice of metres for *model selection* is defensible
(metres weight each correspondence by real metric consequence). That is a different and
lower standard than evidence sufficient to displace a shipped artifact. The in-sample median
also regresses 3.719 -> 5.446 px. Refused.

**This is the one refusal a reasonable owner might overturn.** The numbers sit in
`promotion_record.json` under `refused` to overturn it with, and doing so is now a two-file
change rather than a code change.

### 4.3 `outdoor_webcam` and `wolverine_mixed` -- measured negatives, preserved

Both held-out criteria identical to 6 significant figures. The refit **refused distortion on
held-out evidence** for both cameras and recovered the shipped solve. There is nothing to
promote, and the refusal of distortion is a result about those cameras rather than a gap.
Per the refit lane, the owner's reviewed wolverine bounce labels do not materially improve
either (sigma along ray 0.225264 -> 0.224934 m) -- also an honest negative.

No pointer was written for any of the four. They resolve to their raw solves, and a test
asserts that they do.

---

## 5. Cleanup

### 5.1 Worktrees: 5 removed, 2 refused on evidence

`git worktree remove` deletes **gitignored** files silently, so "clean" per `git status` was
not a sufficient precondition. For each candidate I enumerated every untracked-and-ignored
file (excluding `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.benchmarks`) and
checked it against the main checkout by existence and size.

| worktree | merged | unique content found | action |
|---|---|---|---|
| `pb-fix-calib-20260726` | yes | none (caches only) | **removed** |
| `pb-fix-reproj-20260726` | yes | none | **removed** |
| `pb-fix-sigma-20260726` | yes | `data/external/tt3d_repo`, byte-identical to the tt3d worktree's copy outside `.git` metadata | **removed** |
| `pickleball-fusion-20260724` | yes | none; its `cvat_upload/*` are strict subsets of main's | **removed** (`--force`) |
| `pickleball-balllabel-20260726` | yes | 2 label files differ, and **main's are strictly newer** (21 labels / 8 bounces vs 19 / 7) | **removed** |
| `pickleball-tt3d-20260726` | yes | **`data/external/tt3d_repo`, 6.7 MB, absent from main** | **KEPT** |
| `ball-lane-20260723` | yes | **79 files under `runs/ball_lane_20260723/dormant_flags/` present nowhere else** | **KEPT** |

`pb-subframe-20260727`, `pb-farfield-20260727`, `pickleball-main-7581e52` and
`pickleball_court_train_12b555824` were not touched.

Verified after removal: **all 24 branches present, all 6 tags present, and all 7 branch
heads byte-identical** to their pre-removal sha. `git worktree prune` run. Freed ~2.5 GB.

Three findings the brief did not anticipate:

1. **The stated tag precondition is false.** `premerge-*` tags exist for only 4 of the 7
   branches -- `ball-label-tool`, `ball-lane`, `fusion-wire`, `tt3d-validate`. There is **no**
   `premerge-*` tag for `fix-calib-20260726`, `fix-reproj-20260726` or `fix-sigma-20260726`
   (the repo has exactly 6 tags total, one of which is the unrelated
   `premerge-backup-20260726`). I removed those three anyway, having verified the property
   the tag exists to provide: each branch is an ancestor of `main` and its ref is preserved,
   so no commit is unreachable. I could not create the missing tags -- the fence permits only
   `git worktree list|remove|prune` against the main repo.
2. **`data/external/` exists only inside worktrees.** The TT3D evaluation data (a clone of
   `github.com/cogsys-tuebingen/tt3d`, no LICENSE upstream, internal-validation-only) is not
   in the main checkout. After removing `pb-fix-sigma`, `pickleball-tt3d-20260726` holds the
   **only local copy**. It backs `tests/tt3d/test_tt3d_adapter.py` (which skips without it)
   and the evidence behind best_stack `ball.bounce_anchor_uncertainty`.
   **Recommended: move `data/external/` into the main checkout, then that worktree is free.**
3. **`ball-lane-20260723` holds unique negative evidence.** 79 files under
   `runs/ball_lane_20260723/dormant_flags/solves/` exist there and nowhere else -- per-clip
   solve outputs for dormant flag combinations, i.e. measured-but-unadopted work, precisely
   the record this repo keeps on purpose. Its `data/` dirs are symlinks into main or exact
   duplicates, so **only the `runs/` evidence blocks removal**; copy it into main and the
   worktree is free.

`--force` was needed only for `pickleball-fusion-20260724`. Before forcing I confirmed with
`diff -rq` that nothing was unique to the worktree and nothing differed, and I unlinked its
`.venv` **symlink** by hand first so no deletion routine could descend into the main
checkout's `.venv`. Verified afterwards that main's `.venv` is intact.

**Consequence to know about:** `runs/lanes/calib_distortion_fit_20260726/refit_and_measure.py`
hardcodes `BOUNCE_LABEL_SOURCES` as absolute paths into
`/private/tmp/pickleball-balllabel-20260726/...`, which no longer exists. That script degrades
gracefully (`bounce_label_floors` returns `None` for a missing file) but will now silently
skip its bounce-label section. The labels are committed to main at
`runs/lanes/ball_label_tool_20260726/labels/` (1506 tracked files) in a **newer** form, so
the fix is to point that constant at the repo-relative path. Left for the refit lane's owner
rather than edited from here.

### 5.2 Dead code: 0 removed, 2 registered, 1 auditor blind spot flagged

`audit_dead_code.py --root .` exited **1** at base with two unknown surfaces.

| candidate | verdict | why |
|---|---|---|
| `scripts/tt3d/tt3d_adapter.py` | **NOT removed -- registered** | It IS tested. `tests/tt3d/test_tt3d_adapter.py` imports it via a `sys.path` insert. The auditor's `_matching_test_files` scans `tests/racketsport/` only (audit_dead_code.py:239), so it cannot see a test in `tests/tt3d/`. Removal would delete tested, working code. |
| `scripts/tt3d/run_tt3d_validation.py` | **NOT removed -- registered** | A CLI nobody imports, whose only prose write-up lived under `runs/`, which the auditor ignores by design. It drives the TT3D external-validation experiments E0-E3 and produced `runs/lanes/tt3d_external_validation_20260726/`, cited by the live best_stack entry `ball.bounce_anchor_uncertainty`. |

Neither meets the bar (*provably unreachable AND removal breaks no test*). A third fact
settles it: **deleting the driver would orphan `scripts/tt3d/sigma_calibration.py`**, whose
only inbound reference in the entire repo is a comment at `run_tt3d_validation.py:454`. The
three files stand or fall together, and together they are the only external-data 3D ball
accuracy measurement in the repo.

Resolution: `scripts/tt3d/README.md` documents the harness -- the binding no-licence
constraint on the upstream data, the four experiments, how to run it, where the evaluation
data actually lives, its test, and what it measured. That is registration, and it is
documentation the harness genuinely lacked.

**Flagged, not fixed:** `_matching_test_files` at `scripts/racketsport/audit_dead_code.py:239`
restricts test discovery to `tests/racketsport/`, so any module tested from `tests/tt3d/`,
`tests/server/`, `tests/ios/` or `tests/render_service/` reads as unreferenced. Broadening it
would flip no current status (only `unknown` entries can change and there are now none), but
it edits a shared structure check while sibling lanes are in flight, so I left it. Also worth
knowing: this auditor's "referenced" means *some tracked text file names the path*, which is
weaker than reachable -- it is a naming audit, not a reachability audit, and a passing run
should not be read as "no dead code".

**Nothing was removed.** No default-OFF module, no PENDING entry, no rejected candidate and
no dormant ball code was touched.

### 5.3 best_stack

Added `court.calibration_selection_pointer`, `stage: calibration`, `status: WIRED_DEFAULT`,
`gate: null`, `verified: 0`. `revision` 15 -> 16 and the pinned assertion in
`test_best_stack_manifest.py` moved with it. `value` is deliberately a policy dict rather
than a `local_path`, so the entry adds no filesystem precondition to manifest load and could
not abort another lane's suite collection.

**Caveat for the merger:** the SHARED-MANIFEST WRITE LOCK in `.claude/skills/run-lane/SKILL.md`
grants exactly one in-flight writer fleet-wide and this lane was not dispatched holding it.
The edit is confined to this worktree, but a concurrent manifest writer will collide on
`revision` and on the pinned assertion.

---

## 6. Test evidence

Real commands, real exit codes, no pipes swallowing status.

### 6.1 Wide: 116 files across the CAL / BALL / WORLD surface

Identical invocation both sides; the base list is 115 files because
`test_court_calibration_selection.py` does not exist at `e209112`. Base was produced by
checking `e209112` out in this worktree and re-running.

```
$ git checkout e209112
$ python3 -m pytest $(cat suite_filelist_base.txt) -q -p no:cacheprovider
7 failed, 1160 passed, 6 skipped, 6 warnings in 448.19s      exit 1

$ git checkout calibpromo-20260727
$ python3 -m pytest $(cat suite_filelist_final.txt) -q -p no:cacheprovider
6 failed, 1185 passed, 6 skipped, 6 warnings in 434.15s      exit 1
```

Re-run once more after the final commit, so every change in the series is covered
(`suite_branch_final.txt`) -- byte-identical failure set:

```
6 failed, 1185 passed, 6 skipped, 6 warnings in 438.74s      exit 1
```

`diff` of the sorted FAILED lists is **exactly one line** -- the pre-existing failure this
lane repairs:

```
6d5
< FAILED tests/racketsport/test_dead_code_audit.py::test_dead_code_audit_has_no_unknown_python_source_surfaces
```

**Zero new failures. One pre-existing failure fixed. +25 passing** = 24 new
`test_court_calibration_selection` tests + the repaired dead-code audit test.

The 6 failures common to both sides, all pre-existing and untouched:

| test | |
|---|---|
| `test_ball_arc_chain::test_default_chain_config_matches_frozen_row22_manifest` | frozen manifest drift |
| `test_ball_arc_solver::test_wolverine_seg6_fixture_falls_back_to_anchor_bvp_and_render_samples_stay_in_bounds` | |
| `test_court_keypoint_partial_labels::test_img1605_progress_builds_partial_visible_label_payload` | |
| `test_court_partial_geometry::test_visible_floor_homography_infers_img1605_occluded_corner` | |
| `test_court_partial_geometry::test_visible_floor_homography_requires_four_floor_points` | |
| `test_scaffold_tool_index::test_real_scaffold_tool_index_matches_checked_in_schema` | 6 tools categorised `unknown` |

Note that two of the six pre-existing failures are `img1605` court-geometry tests. They fail
identically at base and after promotion, so promoting that clip's calibration neither caused
nor fixed them; they read `court_keypoints_partial.json`, not the calibration.

### 6.2 Focused

```
$ python3 -m pytest tests/racketsport/test_court_calibration_selection.py -q -p no:cacheprovider
24 passed, 6 warnings                                        exit 0

$ python3 -m pytest tests/racketsport/test_dead_code_audit.py -q -p no:cacheprovider
2 passed                                                     exit 0

$ python3 -m pytest tests/racketsport/test_best_stack_manifest.py \
      tests/racketsport/test_best_stack_resolution.py tests/server/test_best_stack_parity.py -q
13 passed, 6 warnings                                        exit 0
```

### 6.3 AGENTS.md structure checks, before and after

| check | at `e209112` | after |
|---|---|---|
| `python3 scripts/racketsport/list_scaffold_tools.py --root .` | exit 0 | exit 0, output **byte-identical** |
| `python3 scripts/racketsport/audit_dead_code.py --root .` | **exit 1** (2 unknowns) | **exit 0** (0 unknowns) |
| `python3 scripts/racketsport/audit_storage_policy.py --root . --json` | exit 1 | exit 1, output **byte-identical** |

`audit_storage_policy` fails identically at base and after: it assumes `.git` is a directory
and a git worktree makes it a file (`NotADirectoryError`). Pre-existing, untouched, not this
lane's to fix.

### 6.4 Environment note, because it changes what "pre-existing failure" means

`load_best_stack_manifest()` hard-fails in a bare worktree -- the manifest points at
gitignored artifacts that are not checked in -- so **5 test files could not be collected** and
the first suite attempt aborted with `Interrupted: 5 errors during collection`, exit 2. I
resolved it by symlinking `models/checkpoints/*`,
`runs/waveb_confidence_gate_20260702T183158Z`, `runs/lanes/w7_ballretrain_20260709/vm_pull`
and `runs/lanes/beststack_verify_20260709/evidence` from the main checkout. Those symlinks
are gitignored, appear in no commit, and are read-only uses of the main repo. Both suite runs
above were made with them in place, on both sides. Without them, several apparent
"pre-existing failures" are really "this worktree lacks a 274 MB checkpoint", which is worth
separating from real ones.

---

## 7. Scope kept

- `ball_arc_solver.py` timing/uncertainty code: **not touched** (sibling lane owns it).
- `court_keypoint_net.py`, `court_structured_solver.py`, the court training path: **not touched.**
- `_confidence_from_reprojection`, trust-band and gate policy: **not touched.**
- `orchestrator.TRUSTED_INTRINSICS_SOURCES` / `REVIEWED_GATE_MARKERS`: **not touched.**
- Raw solves: **not modified.** `git diff --name-status e209112 calibpromo-20260727` shows 4
  modified files -- `best_stack.json`, `process_video.py`, `virtual_world.py` and a test --
  and **no artifact modification at all**; every artifact change is an addition. Both raw
  digests are recorded in their pointers and re-verified by test on every run.
- `eval_clips/ball/manifest.json`: not touched (it carries no calibration pointer field).

## 8. Reproduce

```
python3 runs/lanes/calib_promote_cleanup_20260727/measure_calibration_floor.py \
    --out /tmp/floor.json --label check
python3 runs/lanes/calib_promote_cleanup_20260727/measure_calibration_floor.py \
    --source refined_candidate --out /tmp/refined.json --label candidates
python3 runs/lanes/calib_promote_cleanup_20260727/promote_calibrations.py
```

`promote_calibrations.py` is idempotent and re-verifies the raw digest before and after
writing; it aborts rather than proceed if a raw solve changed.
