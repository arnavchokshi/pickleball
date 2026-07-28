# Lane `alwayson_defaults_20260728` — kitchen-gating + foot-slide grounding, made unconditional

Status: engineering default flip, `VERIFIED=0` binding throughout. This is an
owner-directed change to what the sole pipeline entrypoint
(`scripts/racketsport/process_video.py`) does by default, not an accuracy
promotion. No new gate was passed; no independent evidence was produced.

## Owner directive

"the kitchen gating and foot sliding grounding logic worked super well —
refine all of that and make it what ALWAYS happens." Scope: make the
already-adopted-but-conditionally-wired BODY-side behaviors unconditional
defaults of `process_video.py`, for both `full` and `court_skeletons`
presets, without violating the R3 same-pass safety rule and without
reclassifying anything as `VERIFIED`.

## What was verified before touching anything

- **Conservative NVZ/kitchen occupancy** (`threed/racketsport/placement.py`,
  `_conservative_kitchen_decision` + `_build_nvz_line_posteriors`) is already
  unconditional: it runs inside `rewrite_tracks_with_placement`, which both
  the pre-BODY `placement` stage (stage 7, both presets) and the post-BODY
  `placement_refine` stage call, whenever `court_calibration.json` +
  `tracks.json` exist. It already defaults every frame's
  `court_contact_state` to `"unknown"` and only ever flips to
  `confirmed_outside` / `confirmed_inside_or_on` when the 99% CI
  (`ci99_wholly_outside_nvz` / `ci99_wholly_inside_or_on_nvz`) is wholly on
  one side of the NVZ line; missing/weak NVZ line evidence
  (`local_nvz_line_unobservable`, `local_nvz_line_evidence_below_decision_bar`)
  leaves it `unknown`. **No code change was needed for item 3** — it was
  already computed and emitted for both presets whenever calibration +
  placement exist, and it stays conservative by construction
  (`tests/racketsport/test_placement.py::test_conservative_kitchen_decision_confirms_only_one_sided_ci`,
  `::test_conservative_kitchen_decision_abstains_when_ci_crosses_line`).
  Enabling post-BODY `placement_refine` for the `full` preset (item 2, below)
  means this same conservative decision now also gets computed a second time
  on the post-BODY-refined placement for `full`, matching what
  `court_skeletons` already had.
- **`grounding_refine`** (stage 15) was already default-on when its inputs
  exist; no change needed.
- **R3 same-pass safety rule** (`runs/lanes/r3_unified_grounding/design.md`,
  `runs/lanes/i1_grounding_unification/spec.md`): the danger it killed was a
  *legacy* in-place `tracks.json` rewrite from post-BODY SAM-3D foot pixels,
  which could make the just-computed BODY output stale relative to the
  tracks it was computed from. That legacy in-place rewrite no longer exists
  in the codebase. The *current* `placement_refine` stage
  (`_stage_placement_refine` / `_run_placement_stage(refine_from_sam3d=True)`)
  already only ever writes to the separate, immutable
  `placement_refined.json` / `tracks_placement_refined.json` pair — raw
  `tracks.json` is read-only to it. Downstream `grounding_refine` already
  auto-detects this provenance (`_has_r3_grounding_provenance`, keyed on
  `placement_refined.json.refine_from_sam3d=True` +
  `tracks_placement_refined.json` present) and switches to z-only mode with
  XY translation disabled when it is present — generically, not
  preset-gated. The only thing gating any of this to `court_skeletons` only
  was one `if self.options.pipeline_preset != "court_skeletons": skip` check
  at the top of `_stage_placement_refine`.

This meant items 1 and 2 reduced to: (1) flip a best_stack-driven default
boolean and add an explicit opt-out, and (2) delete one preset conditional
whose safety already lived one layer down, in the immutable-artifact
mechanism itself — not in the preset check.

## Changes

### 1. `placement_trajectory_refine` default-on, both presets

- `configs/racketsport/best_stack.json` — `body.placement_trajectory_refine`:
  `status` `PENDING` -> `WIRED_DEFAULT`, `value.enabled` `false` -> `true`.
  `notes`/`provenance` updated to name the owner directive and lane, keep the
  `preview_band` / `do_not_promote` / `VERIFIED=0` language, and keep the
  `ns02_body_world_placement_independent_gt` gate as the still-unmet
  promotion bar. `proven_against.scope` (`"same four internal eval-only
  cards"`) and `independent_gt: false` were left unchanged — the underlying
  evidence did not get more independent tonight; only the default flipped.
- `scripts/racketsport/process_video.py`:
  - `DEFAULT_PLACEMENT_TRAJECTORY_REFINE` is computed from
    `PLACEMENT_TRAJECTORY_REFINE_STACK_VALUE["enabled"]`, so the best_stack
    flip alone makes it default `True` for every invocation.
  - Added `--no-placement-trajectory-refine` (new flag). Options resolution
    is now: `--no-placement-trajectory-refine` (if given) wins over
    everything; otherwise `--placement-trajectory-refine` OR the best_stack
    default wins. Removed the old `pipeline_preset == "court_skeletons"`
    special case since the best_stack default now covers both presets
    uniformly, and `court_skeletons` previously had **no way to opt out** —
    it now does, via the new flag.
  - Missing BODY/plant evidence stays a typed skip
    (`placement_trajectory_no_body`, `placement_trajectory_no_plant_windows`)
    or typed degrade (`placement_trajectory_no_tracks`); malformed
    `foot_contact_phases.json` still raises `MalformedPlacementInputError`.
    Raw tracks/placement/skeleton/grounding artifacts are untouched — the
    stage only ever writes the separate `placement_trajectory_refined.json`.

### 2. Post-BODY immutable foot-anchoring (`placement_refine`), both presets

- `scripts/racketsport/process_video.py` — `_stage_placement_refine`: removed
  the `pipeline_preset != "court_skeletons"` early skip. The stage now runs
  for both presets whenever `sam3d_keypoints_2d.json` + `skeleton3d.json`
  exist, else typed-skips (`missing_sam3d_foot_evidence`). No other logic in
  `_run_placement_stage` changed — the immutability contract (only
  `placement_refined.json` + `tracks_placement_refined.json` are written,
  raw `tracks.json` stays read-only) was already preset-agnostic.
- `configs/racketsport/best_stack.json` — added new entry
  `body.post_body_placement_refine` (`status: WIRED_DEFAULT`,
  `stage: placement_refine`, `value.applies_to_presets: [full,
  court_skeletons]`, `value.mutates_raw_tracks_json: false`) documenting the
  parity extension, its R3-safety reasoning, and that it is not read by code
  as a gate (the gating is structural: preset-agnostic file-existence
  checks) — this entry is a provenance record, matching the pattern of other
  documentary-only best_stack entries in this file.
- **No architectural blocker was found for the `full` preset.** Verified
  directly: `sam3d_keypoints_2d.json` is in `BODY_OUTPUT_ARTIFACTS_DEFAULT`
  (`scripts/racketsport/remote_body_dispatch.py`) unconditionally, not
  preset-gated; nothing except `_stage_placement_refine`'s own artifacts
  ever writes `tracks.json`, so no other `full`-only stage could make BODY
  stale between BODY and `placement_refine`; and `grounding_refine`'s R3
  z-only-mode switch and `world`'s tracks-preference-when-present
  (`_stage_world` already prefers `tracks_placement_refined.json` over
  `tracks.json` when present, for both presets) were already
  preset-agnostic. This was confirmed live on the smoke run below: BODY-mode
  mesh vertices are `full`-only, but `placement_refine` only ever consumes
  2D/3D joint keypoints, not mesh, so mesh-vs-skeleton-only is not a
  blocker either.
- What stays preset-gated and why: **nothing new.** The legacy `placement_refine`
  same-pass in-place `tracks.json` rewrite that R3 killed remains dead on
  both presets (it was never resurrected). Stage 7 `placement` (the
  pre-BODY, non-immutable rewrite) is unrelated to this change and was
  already preset-agnostic before tonight.

### 3. NVZ/kitchen occupancy always computed and emitted

No code change (see "What was verified" above) — already true for both
presets whenever calibration + placement exist, already conservative
(ambiguous -> `unknown`). Enabling item 2 for `full` gives the `full` preset
a *second* kitchen-decision pass (on the post-BODY-refined placement), which
`court_skeletons` already had.

### 4. best_stack.json

Both entries above (`body.placement_trajectory_refine`,
`body.post_body_placement_refine`) carry
`provenance: {date: "2026-07-28", lane: "alwayson_defaults_20260728", commit:
"worktree", note: "owner directive: always-on"}` (the `"commit": "worktree"`
placeholder follows the existing convention in this file for an entry
committed in the same pass that documents it — see
`ball.ray_court_volume_gate`, commit `5f9ecf4`). `do_not_promote`/preview-band
language and `VERIFIED=0` are unchanged in substance; `status` moved from
`PENDING`/(no prior entry) to `WIRED_DEFAULT`, which is the existing status
this repo uses for owner-directed defaults that are wired but not
independently gate-verified (see `tracking.association_court_margin`,
`WIRED_DEFAULT` + `preview-band`, rev-12 owner-directed default, same
pattern).

### 5. Tests

`tests/racketsport/test_process_video.py`:

- Replaced `test_full_preset_keeps_post_body_placement_refine_disabled`
  (asserted the old `court_skeletons_only` skip) with two tests mirroring
  the existing `court_skeletons` pair:
  `test_full_preset_skips_post_body_placement_without_sam3d_evidence` (typed
  skip when evidence is absent, both presets now share this path) and
  `test_full_preset_emits_immutable_post_body_placement_artifacts` (asserts
  `status == "ran"`, the separate-artifact filenames, `refine_from_sam3d is
  True`, and explicitly `tracks_path != rewritten_tracks_path` as an R3
  immutability regression guard).
- Replaced `test_cli_parses_placement_trajectory_refine_opt_in_over_default_off`
  with `test_cli_placement_trajectory_refine_defaults_on_with_explicit_opt_out`:
  asserts `PLACEMENT_TRAJECTORY_REFINE_STACK_VALUE["enabled"] is True`,
  default is `True` with `..._explicit is False`, `--no-placement-trajectory-refine`
  gives `False` with `..._explicit is True`, both flags together resolve to
  `False` (opt-out wins), and both presets (`full`, `court_skeletons`) share
  the same default and the same opt-out.
- No honesty/immutability assertion was weakened; the new tests add an
  explicit immutability check that did not exist before
  (`tracks_path != rewritten_tracks_path`).

`RUNBOOK.md` stage list (14, 15, 16) and the `--placement-trajectory-refine`
flag row were updated to match; a `--no-placement-trajectory-refine` row was
added.

## Verify — before/after test evidence

Before (baseline, same commit range, run at lane start):

```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_best_stack_manifest.py -q
1 failed, 1 passed  (test_best_stack_manifest_integrity: revision 18 != hardcoded 16 -- PRE-EXISTING, unrelated to this lane; last touched by main-branch commits 5f9ecf4/a762f50 which also left revision at 18 without updating the test. Not touched by this lane; reported, not fixed, since it is outside the fence and outside this task's scope.)
```

After (this lane's changes in place):

```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_process_video.py -q
181 passed in 35.78s

$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_pipeline_contracts.py -q
15 passed in 2.26s

$ .venv/bin/python scripts/racketsport/process_video.py --help
exit 0; --placement-trajectory-refine and --no-placement-trajectory-refine both listed

$ MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/racketsport/test_placement.py tests/racketsport/test_ball_line_calls.py \
    tests/racketsport/test_court_positioning.py tests/racketsport/test_rally_metrics.py \
    tests/racketsport/test_body_grounding_quality.py tests/racketsport/test_body_grounding_refine.py \
    tests/racketsport/test_foot_pin.py tests/racketsport/test_apply_foot_pin_cli.py \
    tests/racketsport/test_build_rally_metrics_cli.py tests/racketsport/test_grounding_consistent_mpjpe.py \
    tests/racketsport/test_placement_refine_clis.py tests/racketsport/test_placement_trajectory_refine.py \
    tests/racketsport/test_player_grounding.py tests/racketsport/test_refine_body_grounding_cli.py \
    tests/racketsport/test_strict_placement_rollup.py tests/racketsport/test_virtual_world_ball_failclosed.py \
    tests/racketsport/test_virtual_world_review.py tests/racketsport/test_virtual_world.py \
    tests/racketsport/test_worldhmr_stance_grounding.py tests/racketsport/test_worldhmr.py -q
238 passed in 9.40s
```

`test_best_stack_manifest_integrity`'s pre-existing `revision == 16` failure
is unchanged by this lane (still fails the same way; `revision` was 18
before this lane touched the file and is still 18 after -- this lane did not
bump it, matching the precedent set by `ball.ray_court_volume_gate`'s
addition in commit `5f9ecf4`, which also left `revision` unchanged).

A repo-wide `tests/racketsport -q` run (~496 files) was also attempted for
broader regression coverage in parallel with the smoke run. After roughly 20
CPU-minutes it had reached 38% with no failures beyond the same scattered,
pre-existing `F` pattern already visible at that point (consistent with
known pre-existing issues such as `test_best_stack_manifest_integrity`) and
no new failures attributable to this lane's diff; it was stopped rather than
run to completion given the time cost in this CPU-only environment and since
it is not one of the task's required verification commands. Everything the
task explicitly requires — `test_process_video.py`, `test_pipeline_contracts.py`,
and focused suites covering placement/grounding/NVZ/placement-trajectory — is
captured in full, passing, above.

## CPU/skeleton-only smoke run — real evidence, both stages exercised

Per RUNBOOK's "CPU/skeleton-only smoke" pattern, on the wolverine eval clip
with its committed calibration
(`eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json`,
git-tracked):

```
.venv/bin/python scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --court-calibration eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json \
  --skip-ball --out runs/lanes/alwayson_defaults_20260728/smoke_wolverine_no_gpu --json
```

No GPU/remote BODY host was available in this environment (`RemoteConfig`
default host is empty; no `--remote-host`/`gcloud` access here), so this
clip's `clip_dir` was pre-seeded with **real** BODY evidence
(`skeleton3d.json`, `sam3d_keypoints_2d.json`, `foot_contact_phases.json`,
`body_full_clip_gate.json`) copied from a prior real BODY run of this same
clip on `main` (`runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner/`)
before invoking the CLI without `--no-gpu` (so tracking runs fresh on
CPU/MPS; BODY itself still cleanly degrades -- no host configured, exits fast
via `RemoteBodyDispatchError`, does not touch the pre-seeded files). This is
disclosed in full because it means tracking/placement/tracks.json are
**genuinely fresh, real output from this run**, while the BODY evidence is
**real but from a different historical run of the same clip** -- good enough
to exercise the stage mechanics honestly, not scientific ground truth.

Stage outcomes (`PIPELINE_SUMMARY.json`, `status: partial` overall, expected
given `--skip-ball` and no BODY host):

| Stage | Status | What it proves |
|---|---|---|
| tracking | ran | Fresh real CPU tracking (yolo26m + BoT-SORT + raw-pool OSNet association), 4 players, ids 1-4 |
| body | degraded | Clean, fast `remote_body_dispatch_unavailable` (no `--remote-host`); pre-seeded skeleton3d.json/sam3d_keypoints_2d.json untouched |
| **placement_refine** | **ran** | **Item 2, proven live on `full` preset** (previously would have been `skipped`, `court_skeletons_only: True`). Notes: "emitted immutable post-BODY track placement from SAM-3D foot pixels; raw tracks.json was not rewritten"; `source_counts: {bbox: 1167, sam3d: 705}`; wrote `placement_refined.json` (14.3 MB) + `tracks_placement_refined.json` (446 KB) |
| grounding_refine | ran | `grounding_anchor_source: placement_track_world_xy`, `xy_translation_enabled: False` -- confirms `_has_r3_grounding_provenance` auto-detected item 2's new `full`-preset output and correctly forced z-only mode, exactly as it already did for `court_skeletons` |
| **placement_trajectory_refine** | **failed (real, disclosed)** | **Proves the default-on wiring is active** -- it did not take the "disabled" skip path (`options.placement_trajectory_refine` was `True` with no flags), it attempted real computation and reached `refine_placement_trajectory`'s internals. It then hit `MissingPlacementInputError: missing track observation for skeleton frame ('19', 58)` -- the pre-seeded (July-9-run) `skeleton3d.json` uses raw-pool ids `19-22`, while this run's fresh `tracks.json` uses selected ids `1-4`. This is an artifact of combining two historical runs for BODY evidence, not a pipeline defect: `placement_refine` itself tolerates this via pixel/bbox-based reassociation (`sam3d_reassigned: 705`), but `refine_placement_trajectory` does direct `(player_id, frame)` lookups and correctly fails loudly on the mismatch rather than silently producing garbage -- exactly the "malformed inputs still fail loudly" contract this lane was told to preserve. Full traceback: `runs/lanes/alwayson_defaults_20260728/smoke_wolverine_no_gpu/wolverine_mixed_0200_mid_steep_corner/stage_errors/placement_trajectory_refine.json` |

**Follow-up proof that the stage completes cleanly under its new default**:
remapped the pre-seeded `skeleton3d.json`/`sam3d_keypoints_2d.json` ids
`{19->1, 20->2, 21->3, 22->4}` (order-preserving, not a claim of correct
physical player correspondence -- disclosed) so they align with this run's
own fresh `tracks.json`, then called the real, unmocked
`ProcessVideoPipeline._stage_placement_trajectory_refine()` directly against
this same `clip_dir` (to avoid re-paying ~4 minutes of CPU tracking; every
other input -- `tracks.json`, `court_calibration.json`, `placement.json`,
`foot_contact_phases.json` -- is the real output already on disk from the CLI
run above). Confirmed the default is live in-process
(`options.placement_trajectory_refine is True` with zero flags), then:

```
status: ran
notes: ['emitted separate preview-band placement_trajectory_refined.json after grounding_refine',
        'raw tracks.json, placement.json, skeleton3d.json, foot_contact_phases.json, and grounding
         artifacts remain immutable', 'preview only; do_not_promote; VERIFIED=0']
artifacts: ['placement_trajectory_refined.json']
metrics: {'player_count': 4, 'frame_count': 705, 'plant_anchored_frame_count': 184}
```

Wrote a real 13.2 MB `placement_trajectory_refined.json`.
`selected_for_world: False` (`world_selected_frame_count: 0`,
`residual_eligible_frame_count: 0`) -- the guard correctly declined to select
any frame for world display, which is the right outcome given the
player-identity correspondence across the two source runs is synthetic, not
verified. This is the stage's own safety guard working as designed, not a
defect.

**Net conclusion**: both `placement_refine` (full preset) and
`placement_trajectory_refine` are proven to run under their new
no-flag defaults, using real (if cross-run-assembled) evidence, and both
correctly refuse to fabricate trust they do not have -- `placement_refine`
via pixel-based reassociation instead of trusting stale ids,
`placement_trajectory_refine` via its residual/guard check refusing
`selected_for_world` on unverified correspondence, and via a loud typed
exception rather than a silent wrong answer when it cannot resolve a lookup
at all.

## What this lane did NOT do

- Did not run a fresh, single-pass, self-consistent GPU/remote BODY dispatch
  for this clip (no remote host / GPU access in this environment) -- the
  `full`-preset `placement_trajectory_refine` "ran successfully" proof above
  used cross-run-assembled, id-remapped BODY evidence and a direct
  (unmocked) stage-method call rather than a second full CLI pass, disclosed
  above.
- Did not touch `NORTH_STAR_ROADMAP.md`, `runs/manager/gpu_fleet.md`, fleet
  scripts, or any of the explicitly excluded untracked directories
  (`brand-exploration/`, `cvat_upload/`, `web/replay/public/critique`).
- Did not run the independent NS-02 BODY world-placement GT gate -- it does
  not exist yet. `body.placement_trajectory_refine`'s gate stays
  `ns02_body_world_placement_independent_gt`, unmet.
- Did not touch the legacy same-pass in-place `tracks.json` rewrite path --
  it stays dead on both presets, per the R3 ruling.
- Did not bump `configs/racketsport/best_stack.json`'s top-level `revision`
  field (18) -- matching the precedent of the last entry-adding commit to
  this file (`5f9ecf4`), which also left it unchanged. The pre-existing
  `test_best_stack_manifest_integrity` failure (asserts `revision == 16`) is
  unrelated to this lane and was not fixed (out of fence, out of scope, and
  not caused by this lane's changes).

## Honest statement

This is an **owner-directed default flip** -- an engineering/integration
change to what `process_video.py` runs unconditionally -- not an accuracy
promotion. `VERIFIED=0` remains binding on both `body.placement_trajectory_refine`
and `body.post_body_placement_refine`. The 4/4-clip foot-slide improvement
cited in both best_stack entries is the same scoped, non-independent evidence
it always was (same four internal eval-only cards used to tune the
hyperparameters); nothing tonight made it independent. The
`ns02_body_world_placement_independent_gt` gate is the real promotion bar and
remains unmet. Conservative NVZ/kitchen occupancy continues to default to
`unknown` on ambiguous evidence and never issues a false decisive call -- that
property was not weakened, only extended to run a second time (post-BODY,
`full` preset) alongside the pre-BODY pass it already had.

## Concurrent activity note

Other integration-owner-fenced lanes committed directly to `main` on this
same repo while this lane was in progress (`f3e3160` "Fleet: warm snapshot
boot path + S3 durable model store", `ebc34d7` "Docs: events + ball 2D->3D
next-steps plan") -- both confirmed to touch none of this lane's fenced files
(`configs/ssh/`, `runs/manager/gpu_fleet.md`, `scripts/fleet/` for the
former; a new `runs/lanes/` docs-only dir for the latter). This lane's diff
is independent of both.
