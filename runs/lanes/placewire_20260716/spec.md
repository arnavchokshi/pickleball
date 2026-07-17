# Lane placewire_20260716 — wire Track I's adopted placement-trajectory fusion as an opt-in runner stage (default OFF)

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: Track I's adopted commit 0ec239325 + their full report
  runs/lanes/trackI_placefuse_20260716/report.json (contains the wiring PROPOSAL — you must
  RE-DERIVE it against the live post-refinedstage runner, never blind-apply), their
  SCHEMA.md, threed/racketsport/placement_trajectory_refine.py (landed, READ-ONLY),
  NORTH_STAR_ROADMAP.md §6, runs/manager/inflight_lanes.md fences,
  runs/manager/trackC_20260716/RULINGS.md (waves 1-3 context).
- FENCES: threed/racketsport/placement_trajectory_refine.py and the two Track I CLIs are
  LANDED Track I code — READ-ONLY (wiring only; if the seam genuinely requires a change in
  their module, STOP that sub-slice and report). ball_arc_* = Track A (live anchorfusion
  lane). ios/ = Track D. orchestrator.py/schemas = just-landed calpolicy (read-only).
  configs/racketsport/best_stack.json rev-13 PENDING entry ALREADY EXISTS (enabled:false,
  do_not_promote) — do NOT edit best_stack.json; your stage reads the existing entry.
  Preserve all unrelated dirty work.
- PYTEST EXIT-CODE TRAP (3b639768c): no pipes; literal `$?`; report numbers.
- Wide suite mandatory w/ attribution. KNOWN repo-wide failures NOT yours: the two
  storage-policy tests (Track G's unregistered event_head_scaffold manifests, flagged) and
  any Track A anchorfusion worktree noise.
- Artifacts under runs/lanes/placewire_20260716/.

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- scripts/racketsport/process_video.py (SOLE owner this window)
- RUNBOOK.md (stage order/flags additions only)
- Tests: tests/racketsport/test_process_video.py, test_truthful_capabilities.py (pins),
  test_spine_stage_contract.py (authoritative graph), new test_placewire_*.py.

## OBJECTIVE
Wire `placement_trajectory_refine` as an OPT-IN stage (default OFF) positioned after
grounding_refine in the canonical stage graph, per Track I's adopted design:
- Explicit flag (e.g. --placement-trajectory-refine) AND/OR best_stack entry enablement
  path (entry is enabled:false; explicit flag wins; absent both = stage skipped with a
  typed skip note). Default behavior byte-parity: with the flag absent, every artifact of a
  default run is unchanged (parity test).
- The stage consumes TRK footpoints/BODY placement/planted-foot windows per Track I's
  landed API, emits placement_trajectory_refined.json (their SCHEMA.md), preview trust
  band, covariance/weights passthrough, raw placement artifacts immutable (standing rule 6).
- Typed failure semantics per the spine16 contract: expected-optional absences (no BODY,
  no plant windows) = typed degrade/skip; programming/schema errors FAIL loudly.
- Stage registered in the canonical graph + RUN_IDENTITY dependencies/outputs (inputs:
  tracks/placement/skeleton3d/grounding artifacts + the trackI config identity) + stage
  counts/doc pins updated coherently (RUNBOOK numbered block, truthful expected_order,
  authoritative-graph test — derive counts from code).
- Downstream: Track K's fusion will PREFER placement_trajectory_refined.json when present —
  you wire PRODUCTION of the artifact only; do not touch world/fusion consumers.

## MANDATORY TESTS (literal exit codes)
- Default-off byte-parity; flag-on cold run produces the artifact w/ preview band + typed
  provenance (mock or small fixture per Track I's test utilities); expected-optional
  absence degrades typed; schema error fails loudly; reuse/identity coherence.
- Focused: test_process_video.py, test_truthful_capabilities.py, test_spine_stage_contract.py,
  your new tests — EXIT 0 (minus the two attributed Track-G storage failures);
  process_video.py --help EXIT 0. Full wide suite + attribution.

## MANDATORY STRUCTURED REPORT at runs/lanes/placewire_20260716/report.json (write it
yourself; schema docs/racketsport/lane_report.schema.json): objective_result per
deliverable; full_suite + literal exit codes + attribution; HONEST ISSUES; BEST-STACK
DELTA = "(b) consumes the existing rev-13 PENDING entry, no edit" — state it; dated
inflight note paragraph.
