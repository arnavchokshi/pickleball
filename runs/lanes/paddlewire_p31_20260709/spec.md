# LANE: paddlewire_p31_20260709 — P3-1: wire the fused paddle 6-DOF estimator into the default E2E

## HARD RULES
- NO branches, NO commits. Read BUILD_CHECKLIST.md last ~15 bullets + NORTH_STAR Part IV (incl. rule
  15 best-stack doctrine) first.
- 4 protected clips EVAL-ONLY; Outdoor/Indoor labels NEVER (this lane needs none).
- WIDE blast-radius suite at end (MPLBACKEND=Agg, no fail-fast), census reported honestly; new CLI ->
  scaffold direct-reference test same-lane; no new root .md.
- Artifacts under runs/lanes/paddlewire_p31_20260709/ only; other lanes' dirs read-only.
- No .patch deliverables; fenced-file needs = inline hunks in report.

## FILE OWNERSHIP
YOU OWN: scripts/racketsport/process_video.py (stage insertion — you are the ONE integration lane
for this change, per TECH_BLUEPRINTS B.1.1), threed/racketsport/paddle_pose_fused.py (+ any paddle
helper modules it needs), configs/racketsport/best_stack.json (paddle entry flip ONLY),
tests/racketsport/test_paddle_* (new/updated), minimal deterministic-fixture test updates.
DO NOT TOUCH: court fence files (court_detector_v2*.py, court_model_infer.py + their tests),
web/replay/**, ios/**, server/**, gate thresholds/keys, heldout ledger, boards. virtual_world.py
stays UNMODIFIED (the artifact contract already renders — that is the proof of clean wiring).

## OBJECTIVE (NORTH_STAR P3-1; TECH_BLUEPRINTS PADDLE §0 + DATED RESEQUENCING NOTE 2026-07-09 R6 — binding)
The fused estimator (paddle_pose_fused.py, source wrist_palm_grip_fused) has been BUILT-NOT-WIRED
for 4 waves (manual artifact injection only; process_video.py only READS a pre-existing
racket_pose_estimate.json at ~:2982/:2996/:3053/:3078 — re-grep at HEAD). Wire it so a plain
default E2E run PRODUCES racket_pose_estimate.json.
Evidence to read first: runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json
(+ STATUS.md), the best_stack.json paddle PENDING/BUILT_NOT_WIRED entry, TECH_BLUEPRINTS PADDLE
pillar §0-§1, runs/lanes/beststack_core_20260708/report.json (how stages resolve through best_stack).

### Requirements
1. New default pipeline stage (paddle_pose) inserted at the correct point in
   _build_suffix_stage_fns (after BODY joints + tracking outputs it consumes exist; before viewer
   packaging that consumes the artifact). Selection resolved THROUGH best_stack.py (doctrine).
2. FAIL-CLOSED semantics: when evidence is absent/insufficient (no MHR palm frames, no wrist track,
   no detector boxes), emit NO artifact for that segment/clip + a LOUD structured summary block
   (paddle_pose: {status, reason, coverage}) in PIPELINE_SUMMARY — never a crash, never a fabricated
   pose. Trust band stays ESTIMATED/preview (patch _paddle_estimate_trust_band wording if the audit's
   noted wording gap exists at HEAD); NO RKT promotion claims anywhere (RKT VERIFIED needs owner
   marker GT — B.1.5).
3. best_stack.json: flip the paddle entry to WIRED_DEFAULT (revision bump per doctrine), provenance =
   this lane + acceptance_record_v2.json numbers; keep the reflection-cone factor DORMANT (activates
   at P1-4 velocities per §0 — do NOT wire it).
4. An explicit opt-out flag (--no-paddle-pose or equivalent) per the bypass-allowed doctrine.

## ACCEPTANCE
1. Deterministic no-flag CPU fixture: runs green with the paddle stage present; fixture diff limited
   to the sanctioned paddle additions (enumerate fields). If the fixture lacks paddle evidence, the
   fail-closed path must itself be exercised + asserted (status block present, no artifact, no crash).
2. A CPU-runnable internal-val path (Wolverine/Burlington artifacts under existing run dirs may be
   used READ-ONLY as inputs if a full re-run is not CPU-feasible): produced racket_pose_estimate.json
   matches the banked contract (ARTIFACT_TYPE racketsport_racket_pose_estimate, source
   wrist_palm_grip_fused) and scoring-only comparison vs the banked acceptance_record_v2.json numbers
   shows parity (IoU within ±0.02 of 0.2356 Wolverine / 0.3424 Burlington on same inputs) — wiring
   must not silently change estimator behavior.
3. Contract test at the pipeline entry point (not a replica) proving default-run produces/fail-closes
   correctly + the opt-out flag works.
4. Wide suite census; failures proven pre-existing at HEAD only. Scaffold refs 0 missing.
## KILL
If the stage insertion point conflicts with in-flight edits to process_video.py (check git status
first — a beststack reconciliation may still be settling): STOP and report; do not rebase around a
moving tree. If parity (acceptance 2) fails by >±0.02 IoU: STOP with the diff analysis — do not tune
the estimator to pass.

## BEST-STACK DELTA: promotes paddle.fused_estimator to WIRED_DEFAULT (evidence attached); reflection
cone stays DORMANT (gate: P1-4 3D velocities).

## STRUCTURED REPORT (schema via --output-schema): objective_result, acceptance table w/ measured
numbers, changes file:line, full_suite census, HONEST ISSUES, draft BUILD_CHECKLIST bullet.
