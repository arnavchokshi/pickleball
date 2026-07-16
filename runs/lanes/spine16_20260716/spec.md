# Lane spine16_20260716 — NS-01.6: one authoritative stage graph + typed expected-optional failures + frame-schedule completeness

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: NORTH_STAR_ROADMAP.md (§4 NS-01.6 row + §5 queue row 1, §6 standing rules),
  AGENTS.md, RUNBOOK.md Stage Order, runs/manager/inflight_lanes.md (live-lane fences).
- CONCURRENT-LANE FENCE (hard): Track A lane `ballarc_scale_guard_20260715` is LIVE editing
  `threed/racketsport/ball_arc_solver.py` + `ball_arc_chain.py` + their tests. You may NOT
  touch those files or their tests, may not read their in-progress diffs as truth, and must
  NOT change the degrade semantics of the two ball_arc caller catch blocks in
  process_video.py (~L2939 and ~L3576): leave both converting chain exceptions to
  `degraded` this window (the arc is render-only and self-kill gated; Track A is adding
  typed timeouts that must keep degrading, not failing). Preserve their dirty worktree state.
- Preserve unrelated dirty work (configs/ssh/a100_known_hosts, ios/* Swift edits,
  scripts/racketsport/build_event_review_session.py, brand-exploration/, cvat_upload/,
  data/, runs/manager/gpu_fleet.md, other lanes' run dirs).
- PYTEST EXIT-CODE TRAP (commit 3b639768c): pytest redirected to a file, `$?` captured on
  the pytest command directly, no pipes; report literal numeric exit codes.
- Wide suite mandatory before claiming anything:
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` (full). Managers's fresh
  baseline: 3684 passed / 24 skipped / 1 failed EXIT 1 where the single failure
  (test_real_scaffold_tool_index_matches_checked_in_schema) is a CONCURRENT session's
  untracked scripts/racketsport/build_event_review_session.py — NOT yours to fix; attribute
  and move on. Also expect transient failures from Track A's live ball_arc edits — attribute
  per-file, never "fix" their files.
- Artifacts under runs/lanes/spine16_20260716/. No new root .md. Any new CLI ships its
  direct-CLI reference test same-lane. Raw observations immutable; schema/programming errors
  fail loudly (rule 10).

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- scripts/racketsport/process_video.py (you are the ONLY lane on this file this window)
- threed/racketsport/pipeline_cli.py (deletion candidate — see Deliverable 2)
- scripts/racketsport/validate_pipeline_artifacts.py (only the --public-contracts migration)
- threed/racketsport/pipeline_contracts.py (only if folding public-tier metadata is needed)
- threed/racketsport/process_video_body_frames.py (only the validation-default hardening)
- AGENTS.md + RUNBOOK.md (ONLY the pipeline_cli reference lines and stage-graph prose made
  stale by your changes; nothing else)
- Tests: tests/racketsport/test_process_video.py, test_pipeline_cli.py (delete WITH its
  module if Deliverable 2 completes), test_truthful_capabilities.py (only the
  pipeline_cli string pin + any stage-order pin your changes make stale),
  test_pipeline_contracts.py (only if metadata folding touches it), new test files named
  test_spine_*.py.
- FORBIDDEN: threed/racketsport/ball_arc_*.py and their tests; threed/racketsport/
  {timebase,io_decode,audio_onsets,coordinates,court_calibration,placement,ball_court_filter,
  ball_physics3d,ball_inout_uncertainty,virtual_world}.py (just-landed Track C work — read
  only); server/; ios/; NORTH_STAR_ROADMAP.md (manager-owned).

## OBJECTIVE (North Star NS-01.6, quote in your report)
"Remove duplicate legacy stage graph, type expected optional failures, fail on
programming/schema errors, validate complete frame schedules."
Gate: "One authoritative stage graph and tests for cold, reused, partial, and failure paths."
Stop rule: "Do not hide arbitrary exceptions as degraded stages."

Manager-verified current state (ground your work on this, re-verify locations):
- Stage graph assembled in three places: _build_prefix_stage_fns (:900), _middle_stage_fns
  (:913, naming asymmetry), _build_suffix_stage_fns (:921), re-listed independently in
  _run_serial (:949) and _run_overlap (:962/:969/:994/:1010).
- threed/racketsport/pipeline_cli.py = the legacy duplicate 9-stage public-contract graph;
  real non-test importer: ONLY scripts/racketsport/validate_pipeline_artifacts.py:14 behind
  the opt-in --public-contracts flag; plus its own test file and a RUNBOOK string pin at
  test_truthful_capabilities.py:455.
- No typed expected-optional-absence exception exists. _run_stage_safely :1175
  `except Exception` converts EVERYTHING (KeyError, schema errors, ZeroDivisionError) to
  status="degraded" — codified by test_pipeline_never_raises_on_unexpected_stage_exception
  (:4247) and test_frames_stage_degrades_when_video_unreadable (:3956).
- Frame-schedule silent-pass gaps: :2485 generic except → degraded; the
  `result.get("validation", {"equal": True})` default at :2519 and :2465; no cross-stage
  check that the written schedule covers the events-stage frame_compute_plan requirements.

## DELIVERABLES (numbered; report PASS/FAIL each; honest PARTIAL allowed)
1. ONE AUTHORITATIVE STAGE GRAPH: a single canonical ordered stage-graph definition
   (one module-level constant/factory in process_video.py) from which serial AND overlap
   paths derive; normalize the naming asymmetry; add one authoritative-graph test that the
   runtime order, the RUNBOOK prose pin, and the executed stage list all derive/agree.
   Preserve the current 20/21/22-stage behavior exactly (order proven unchanged by the
   existing order tests).
2. REMOVE THE LEGACY DUPLICATE: migrate build_public_contract_readiness off
   pipeline_cli.STAGES (fold the public tier/schema metadata into pipeline_contracts.py or
   a data-only module), update validate_pipeline_artifacts.py --public-contracts to the
   migrated source, DELETE threed/racketsport/pipeline_cli.py + tests/racketsport/
   test_pipeline_cli.py, update the RUNBOOK "Legacy Contract CLI" section + AGENTS.md line
   referencing it + the test_truthful_capabilities.py string pin coherently (the RUNBOOK
   section should now say the CLI was removed and where readiness lives). Run the repo
   hygiene checks after deletion (audit_dead_code.py, list_scaffold_tools.py,
   audit_storage_policy.py --ignore-generated-artifacts) and the truthful/scaffold/dead-code
   test quartet. If migration reveals pipeline_cli is MORE load-bearing than stated, STOP
   this deliverable, keep the file, and report the exact blocker honestly.
3. TYPED EXPECTED-OPTIONAL FAILURES: introduce a typed exception (e.g.
   ExpectedOptionalAbsence(stage_status, reason_code, note)) raised by stages for
   expected/optional absences; rewrite _run_stage_safely: typed absence → degraded/blocked
   with the typed reason recorded; ANY OTHER exception → status="failed" (loud, stops the
   run per existing _HardStageFailure semantics) with the full traceback persisted to a
   stage-error artifact. Then reconcile the per-stage broad catches so bugs are not
   swallowed BEFORE reaching the wrapper — convert AT MINIMUM these to typed-or-reraise:
   frames :2485 (generic → keep typed schedule errors loud; unexpected → raise),
   camera_motion :2030 (schema validation failure of a cached artifact = typed
   invalid-cache absence, but a bug inside the validator must raise), placement :2143,
   tracking :1656, _valid_artifact :6292 (a schema-validator crash must not read as
   "artifact invalid" — distinguish typed jsonschema ValidationError from other exceptions).
   EXPLICITLY LEAVE AS-IS (justify in report): the two ball_arc caller catches (:2939,
   :3576 — fence), match_stats :4696 and coaching_facts :4780/:4804 (designed fail-open
   consumer boundaries), body remote dispatch catches :3968/:3982/:4137 and local-body
   :3856/:3859 (remote infra absence is an expected optional — but route them through the
   typed class with reason codes rather than bare Exception where practical), verify :4973
   (advisory stage). Update the two pivotal tests: unexpected ZeroDivisionError now FAILS
   the run; add a paired test that a typed absence still degrades and the run continues.
   Keep the summary-status mapping unchanged (failed > partial > complete).
4. FRAME-SCHEDULE COMPLETENESS: remove the silent `{"equal": True}` defaults (missing
   validation key = typed error); add a runner-side test that BodyFrameScheduleError/
   BodyFrameMaterializationError surfaces as _HardStageFailure (the :2484 loud path); add a
   bounded cross-stage consistency check that the written process_video_frame_schedule.json
   covers every frame index required by the current frame_compute_plan.json (typed loud
   failure on shortfall, honest note when the plan is absent).
5. COLD/REUSE/PARTIAL/FAILURE COVERAGE: after 1-4, ensure the four named path families each
   have at least one test exercising the NEW contract (cold mocked-heavy full run; reuse;
   partial bundle; loud failure incl. the new unexpected-exception-fails behavior). Reuse
   existing tests where they already cover a family; add only what is missing.

## MANDATORY VERIFICATION (literal exit codes, no pipes)
- Focused: test_process_video.py, test_pipeline_contracts.py, test_truthful_capabilities.py,
  test_scaffold_tool_index.py, test_dead_code_audit.py, test_storage_policy_audit.py, your
  new test_spine_*.py — all EXIT 0.
- Full wide suite with per-failure attribution vs the manager baseline above.
- .venv/bin/python scripts/racketsport/process_video.py --help EXIT 0.

## MANDATORY STRUCTURED REPORT (report.json per docs/racketsport/lane_report.schema.json —
write the file yourself at runs/lanes/spine16_20260716/report.json; do not rely on harness
flags)
- objective_result per deliverable 1-5; full_suite exact counts + literal exit codes with
  attribution; HONEST ISSUES; artifacts; BEST-STACK DELTA (expected "(c) no stack delta" —
  state and justify); a dated one-paragraph inflight note; session_id if known.
