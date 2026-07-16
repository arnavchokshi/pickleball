# Lane refinedstage_20260716 — NS-01.6/01.7 remainder: explicit timed post-BODY refined stages + booked dependency-hashing hunks

DISPATCH GATE (manager-enforced): this lane runs only AFTER Track A's ballarc_scale_guard
ruling commit has landed on main. Reconcile with their landed guard semantics before
editing (git log -- threed/racketsport/ball_arc_chain.py + their lane report at
runs/lanes/ballarc_scale_guard_20260715/).

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: NORTH_STAR_ROADMAP.md (§2.1 P0-G row, §5 queue row 1 remainder, §6),
  runs/manager/trackC_20260716/RULINGS.md, runs/manager/inflight_lanes.md (Tracks A/D/F/G/H
  fences), the spine16 commit ffb7e0975 (the authoritative stage graph you extend), the
  evidence17 report.json (its DEFERRED runner hunks — re-derive, do not blindly apply), and
  Track A's landed guard commit + report.
- FENCES (hard): threed/racketsport/ball_arc_*.py + their tests = Track A (READ-ONLY —
  reconcile at the CALLER side only); ios/ = Track D; research dirs = Track F; event-head
  training scaffolding = Track G; web/replay = Track H. Preserve all unrelated dirty work
  (configs/ssh/a100_known_hosts, scripts/racketsport/build_event_review_session.py, etc.).
- PYTEST EXIT-CODE TRAP (3b639768c): no pipes; literal `$?`; report numbers.
- Wide suite mandatory with per-failure attribution (manager will supply/verify the fresh
  baseline; the known environment-only failures are the codex-sandbox socket binds).
- Artifacts under runs/lanes/refinedstage_20260716/. Raw observations immutable; refined
  outputs stay separate artifacts (standing rule 6).

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- scripts/racketsport/process_video.py (SOLE owner this window)
- RUNBOOK.md (stage-order block + counts made stale by the new stages)
- Tests: tests/racketsport/test_process_video.py, test_truthful_capabilities.py (order
  pin + counts), test_spine_stage_contract.py (authoritative-graph test), new
  test_refinedstage_*.py.
- FORBIDDEN: everything else, notably ball_arc_*, event_fusion.py, timebase/io_decode,
  server/, NORTH_STAR_ROADMAP.md.

## OBJECTIVE (quote in report)
North Star queue row 1 remainder: "make post-BODY refined events/arc explicit timed stages
(~122s hidden in `world`)" + "apply the booked contact-dependency-hashing runner hunks".
P0-G gate slice: "Explicit timed refined stages".

Manager-verified ground truth:
- _stage_world (process_video.py ~:4315-4317) calls _refine_events_after_body (~:3613-3730,
  ALWAYS wall_seconds=0.0) then _stage_refined_ball_arc (~:3488-3611, internal timing folded
  into world metrics only); world reports wall_seconds=0.0 and absorbs ~122s. The proper
  pattern exists one function up: _stage_ball_arc (~:2881) is a real timed stage.
- spine16 landed ONE canonical stage-graph definition — extend it, do not fork it.
- evidence17's report.json contains the deferred _contact_dependency_paths hunks
  (frame_times.json, tracks.json fps source, wrist_velocity_peaks*.json,
  ball_candidates.json, coarse ball_inflections.json, audio-config identity). Re-derive
  them against the current file and git apply --check your own diff.

## DELIVERABLES
1. EXPLICIT TIMED STAGES: lift the two refinement functions into first-class stages in the
   canonical graph — `events_refined` then `ball_arc_refined` — positioned exactly where
   they execute today (immediately before `world`; world's behavior/artifacts unchanged,
   its wall time now honest). Each stage: real wall_seconds via the standard harness, its
   existing reuse gating preserved (contact_refinement_v1 / ball_arc_refined_dependencies_v1
   dependency hashing), its existing typed degrade semantics preserved — including Track A's
   landed guard timeouts surfacing as typed degraded outcomes with timing visible, NOT
   failures (the arc is render-only, self-kill gated). Update RUN_IDENTITY_DEPENDENCIES /
   RUN_IDENTITY_OUTPUTS coherently; note honestly in the report that new stage identities
   rebuild the refined artifacts once (NS-01.3 exact-closure semantics — expected, not a
   defect).
2. STAGE-COUNT/DOC COHERENCE: default serial count changes (20 -> 22; with rally_gating/
   verify 23/23/24 — DERIVE the true numbers from the code, do not trust this spec).
   Update the RUNBOOK numbered block + counts, the truthful expected_order pin (insert the
   two markers), the authoritative-graph test, and any order test in test_process_video.py.
   All doc tests green.
3. DEPENDENCY-HASHING HUNKS: apply the evidence17 deferred additions to
   _contact_dependency_paths (both coarse and refined paths as appropriate), including the
   audio-config identity question — resolve it the honest way (hash the onset-extraction
   config/version knobs that change outputs, or document why the output artifact hash
   suffices; do not hand-wave). Add a reuse-refusal test: touching frame_times.json or the
   wrist peaks artifact invalidates contact reuse.
4. COLD/REUSE/TIMING TESTS: cold run produces both new stages with nonzero-capable
   wall_seconds fields and world no longer absorbing them; reuse run keeps refined
   artifacts current w/o rebuild (given unchanged deps); a guard-timeout path test (mock
   the chain call raising Track A's typed timeout) asserting typed degraded + timing.

## MANDATORY VERIFICATION (literal exit codes, no pipes)
- Focused: test_process_video.py, test_truthful_capabilities.py, test_spine_stage_contract.py,
  your new tests — EXIT 0. process_video.py --help EXIT 0.
- Full wide suite + attribution.

## MANDATORY STRUCTURED REPORT at runs/lanes/refinedstage_20260716/report.json (write it
yourself; schema docs/racketsport/lane_report.schema.json): objective_result per
deliverable; full_suite counts + literal exit codes + attribution; HONEST ISSUES (one-time
rebuild, anything Track A-adjacent); BEST-STACK DELTA (expected "(c) none"); dated inflight
note paragraph.
