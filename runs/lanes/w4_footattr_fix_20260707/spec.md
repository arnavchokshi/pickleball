# LANE w4_footattr_fix_20260707 — implement Design A: BODY-skeleton-direct per-foot contact phases (upstream producer)

## OBJECTIVE
Implement the RULED design from the wave-4 foot-attribution diagnosis: a BODY-skeleton-direct
per-foot contact-phase producer that emits CONFIDENT left/right phases upstream (before
`grounding_refine`), so refine un-kills from its honest no-op — with the FROZEN foot-slide gate
staying green. The diagnosis's measured predictions you are implementing against: confident phases
Burlington ~127 (702 frames), Outdoor ~186 (917), Wolverine ~33 (242), IMG1605 ~10 (39); offline
refine predictor kill=false on all 4. This is GATE-ADJACENT work: an independent adversarial
verifier will attack your change after you return; every acceptance below is measured through
UNMODIFIED existing harnesses.

## EVIDENCE TO READ FIRST (the design is already ruled — do not re-design)
1. `runs/lanes/w4_footattr_diag_20260707/REPORT.md` — Design A is THE spec: signals, confidence
   definition, producer placement. Its `analysis/` scripts + `footattr_measurements.json` are your
   reference implementation of the attribution logic.
2. `runs/lanes/w3_phasefix_20260707/` (spec.md + report.json) — the CONSUMER contract: per-foot
   confidence-bearing phase schema, weak-bilateral demotion rules, `foot_lock_gate_stream`
   instrumentation, companion metric `max_candidate_phase_slide_m`. Your producer must emit
   schema-compatible phases; the consumer rules are FROZEN (do not weaken demotion to make your
   phases pass).
3. The wave-3 offline verifier harnesses (locate in/around `w3_phasefix_20260707` + its verify
   round dirs — grep runs/lanes for the phase-verify / candidate-replay harness): the replay
   predictor and the phaseverify attack harness. Both are UNMODIFIABLE acceptance instruments.
4. Freshest 4-clip artifacts: `runs/lanes/w3_freshworlds_20260707/` @ ad75c875c (what the
   diagnosis measured against).

## ACCEPTANCE (exact; every harness UNMODIFIED — if you believe a harness is wrong, STOP and report)
1. `.venv/bin/python runs/lanes/w4_footattr_diag_20260707/analysis/defect_proof.py
   --post-fix-assert` PASSES with your change, UNMODIFIED. (It fails today by design.)
2. Offline replay predictor (wave-3 harness, unmodified) with your phases consumed predicts
   `grounding_metrics.max_foot_lock_slide_m` <= 0.030 on ALL 4 clips — report the predicted value
   per clip. The bar is FROZEN; a predicted breach you cannot fix without weakening confidence
   rules = STOP: needs-validation with banked evidence. NEVER tune 0.030, never exclude phases
   because they would breach it (rejection reasons referencing the gate threshold are
   definitionally circular and forbidden — grep-assert none of your code does this).
3. grounding_refine offline: engages (predictor kill=false) on all 4 clips; report confident-phase
   counts per clip vs the diagnosis predictions (127/186/33/10 — deviations explained, not hidden).
4. phaseverify attack harness green against your producer.
5. Companion telemetry intact: `max_candidate_phase_slide_m` + per-reason demotion counts still
   emitted (exclusion is never silent).
6. Confidence is a measurable statistic per Design A (document the exact formula in code + report);
   fail-closed: uncertain frames yield NOT-confident, never fabricated confidence.
7. Full blast radius: grep the test files covering every file you touch (expect at least
   `tests/racketsport/test_placement.py`, `tests/racketsport/test_strict_placement_rollup.py`, the
   body-grounding-quality tests, and the w3_phasefix-added tests — list ALL in the report) and run
   them ALL green. Plus your new producer unit tests (attribution logic on fixture skeletons: clear
   left, clear right, ambiguous → not-confident).

## CONSTRAINTS
- Schema-compatible with the w3_phasefix consumer; do NOT change consumer demotion rules, gate
  code, or solver/gate constants.
- Producer placement per Design A (post-BODY, pre-grounding_refine). If wiring requires editing
  `scripts/racketsport/process_video.py` or `threed/racketsport/orchestrator.py` (FENCED): deliver
  that part as a proposed diff in the report and STOP that sub-item — the manager's integration
  micro-lane composes fenced edits.
- DO NOT TOUCH: any `ball_*` file (live lanes), `remote_body_dispatch.py`, `virtual_world.py`,
  `camera_motion.py` (pending separate ruling), `ios/**`, `runs/manager/**`, eval labels, ledger.
- The GPU fresh-worlds proof is the MANAGER's wave-close job — you prove offline only, through the
  real code path (no lane-local surrogate that bypasses production consumption).

## SELF-ITERATION
Iterate until acceptance 1-7 pass or a genuine wall (predicted gate breach per #2; harness
contradiction; fenced seam) — then STOP with evidence. Do not paper over; do not widen scope.

## DISCIPLINE
`.venv/bin/python`; `pytest.importorskip("torch")` for torch tests; no git branch/commit/push; no
network; no new root-level .md; pre-existing failures proven at HEAD; sandbox-environmental
failures classified with proof.

## STRUCTURED REPORT
Acceptance table with per-clip predicted slide max + confident-phase counts vs diagnosis
predictions; CHANGES file:line; full_suite (all listed files) with named failures; HONEST ISSUES
(esp. anywhere your implementation deviates from Design A and why); NEXT (what the adversarial
verifier should attack first — candor here is to your credit).
