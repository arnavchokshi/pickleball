# Lane calpolicy_20260716 — implement the ruled `line_evidence_solved_preview` external-calibration source class

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## RULING YOU IMPLEMENT (Track C manager, 2026-07-16 — do not re-litigate, implement)
A new external-calibration source class `line_evidence_solved_preview` is ADOPTED:
ingestible by the orchestrator, PERMANENTLY preview-band, and NEVER satisfying any gate or
consumer that requires `metric_15pt_reviewed`. Grounds: trust contract §1.4 two-axis design;
the existing `--allow-auto-court-corners-preview` preview-seed precedent; §1.2 import
disclosure + L2 preview tier; standing rule 12 honored by the band, not by refusal;
NS-03.CAL "never court authority" untouched.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: NORTH_STAR_ROADMAP.md §1.4/§3.1 (court/camera row)/§2.2 CAL row/§6;
  orchestrator.py:327 allowlist and its surroundings; the real solved artifact at
  runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_solved.json
  (READ-ONLY fixture — do not modify or move).
- FENCES: refinedstage_20260716 owns scripts/racketsport/process_video.py + RUNBOOK.md +
  test_process_video.py + test_truthful_capabilities.py + test_spine_stage_contract.py —
  do NOT touch them. Track D owns ios/, Track F research dirs, Track G event-head training
  scaffolding, Track H web/replay. ball_arc files: read-only. NORTH_STAR_ROADMAP.md is
  manager-owned (the ruling note is the manager's edit, not yours).
- PYTEST EXIT-CODE TRAP (3b639768c): no pipes; literal `$?`; report numbers.
- Wide suite mandatory with attribution (expect refinedstage noise in runner files —
  attribute per-file, never edit theirs).
- Artifacts under runs/lanes/calpolicy_20260716/.

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- threed/racketsport/orchestrator.py (the external-calibration ingestion allowlist seam)
- threed/racketsport/schemas/__init__.py ONLY if the calibration source enum/validators
  live there (additive; check first)
- Tests: tests/racketsport/test_orchestrator_spine.py (additive) and/or new
  tests/racketsport/test_calpolicy_line_evidence_preview.py

## DELIVERABLES
1. INGESTION: `line_evidence_solved_preview` accepted at the orchestrator external-
   calibration seam, but ONLY when the artifact carries (a) coordinate-space and
   distortion-state declarations, (b) residual diagnostics (per-correspondence or summary
   residuals), and (c) full provenance (method, inputs, code identity). Missing any of
   those = typed loud refusal (never silent downgrade to ingestion). Reuse the §3.1 rule:
   never publish a homography without space, distortion state, and covariance/diagnostics.
2. BAND ENFORCEMENT: everything derived from this source carries trust band `preview`
   permanently — it must be structurally impossible for this source string to satisfy any
   check that names `metric_15pt_reviewed` (search every consumer of the reviewed-source
   check and pin them). Explicit opt-in only; no default behavior change for existing
   source classes (parity test).
3. TESTS: (a) the real banked pbv11 solved artifact ingests successfully as a fixture
   (read-only) and comes out preview-banded; (b) an artifact missing residual diagnostics
   or space declarations is refused loudly; (c) `line_evidence_solved_preview` never
   satisfies a reviewed-calibration gate (adversarial test attempting to sneak it through);
   (d) existing `metric_15pt_reviewed` ingestion byte-parity.
4. Report the exact allowlist diff and every consumer you pinned, so the manager can write
   the North Star CAL-row note precisely.

## MANDATORY VERIFICATION (literal exit codes, no pipes)
- Focused: your test files + test_orchestrator_spine.py + test_pipeline_contracts.py EXIT 0.
- Full wide suite + attribution.

## MANDATORY STRUCTURED REPORT at runs/lanes/calpolicy_20260716/report.json (write it
yourself; schema docs/racketsport/lane_report.schema.json): objective_result per
deliverable; full_suite counts + literal exit codes + attribution; HONEST ISSUES;
BEST-STACK DELTA (expected "(c) none" — this is an ingestion-policy class, not a stack
selection; state it); dated inflight note paragraph.
