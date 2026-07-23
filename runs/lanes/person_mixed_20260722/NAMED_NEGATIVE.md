# PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED — 2026-07-22

- **Named negative:** `PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED`, recorded 2026-07-22 per manager
  directive. This supersedes the plain `HONEST_MISS` shorthand for this lane. Evidence:
  `runs/handoff_20260722/ORCHESTRATOR_STATE.md` section 5 and
  `runs/lanes/person_mixed_20260722/gpu_phase_report.json`.
- **Result recap:** The preregistered two-family non-negative bar was not met. `hemel_test` was
  all-positive (F1 +0.0460), but `od8al_val` precision was -0.1924 and F1 was -0.0842. Evidence:
  `runs/lanes/person_mixed_20260722/vm_pull/`,
  `runs/lanes/person_mixed_20260722/gpu_phase_report.json`, and the
  `runs/manager/gpu_fleet.md` close entry at 2026-07-22T14:08Z.
- **Why undercontrolled:** The arms were not exposure-matched: `anchor_train.txt` had 1,066 lines
  versus 14,400 lines in `mixed_train.txt`; both ran 20 epochs at AutoBatch=6. The mixed arm
  therefore received about 13.5x more gradient updates and 6.75x more human-anchor exposures than
  control (control 21,320 total exposures versus mixed 288,000). The `od8al_val` precision collapse
  is therefore unattributable among (a) harmful pseudo-labels, (b) 13.5x update overfit, and (c)
  6.75x anchor repetition overfit. Evidence and analysis:
  `runs/handoff_20260722/ORCHESTRATOR_STATE.md` section 5; source counts and run configuration:
  `runs/lanes/person_mixed_20260722/gpu_phase_report.json`.
- **Consequences:** No follow-up GPU arm on this design. Any PERSON reopen must lead with an
  exposure-matched control. This precedent is standing method rule #1 in
  `.claude/skills/run-lane/SKILL.md`. Evidence: `runs/handoff_20260722/ORCHESTRATOR_STATE.md`
  section 5 and `.claude/skills/run-lane/SKILL.md`.
- **Ledger ownership:** Ledger recording is handled separately by `trackE_datadebt_20260722`; this
  lane does not touch the ledger. Evidence: `runs/lanes/trackE_methodrules_20260722/spec.md`.
