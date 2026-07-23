# trackE_methodrules_20260722 — codify standing method rules in the run-lane skill + record the person named negative

Codex gpt-5.6-sol, effort high (spec-complete: exact text provided; mechanical placement).
Dispatched by Track-E per PROGRAM.md Part 0 item 4 + Track E item 4. CPU-only.

## HARD RULES
- NO branches, NO commits (Track E commits after gpt-5.6-sol ultra review — method rules are
  gate-adjacent).
- Owned files, exhaustive: `.claude/skills/run-lane/SKILL.md`,
  `runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md` (new),
  `runs/lanes/trackE_methodrules_20260722/**`. NOTHING else. The data ledger is explicitly NOT
  yours (trackE_datadebt lane owns it concurrently).
- Do not reword or restructure existing SKILL.md sections; ADD the new section only, keeping the
  file's voice and formatting.
- Report per docs/racketsport/lane_report.schema.json.

## Task 1 — add to .claude/skills/run-lane/SKILL.md
Insert a new section AFTER the "## CROSS-SIGNAL MANDATE" section and BEFORE "## The spec", titled:

## STANDING METHOD RULES (owner-approved SOTA program 2026-07-22 — checked at ultra review)
1. **Exposure-matched arms MANDATORY.** Every A/B training comparison matches arms on total
   optimizer steps, human-row exposure per step, and caps auxiliary/pseudo loss share (the
   `--sst-batch-size` + `--sst-loss-cap` pattern). An arm receiving more gradient updates or more
   human-anchor exposures than its control is invalid BY CONSTRUCTION — reject at review, before
   GPU. Precedent: PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED
   (runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md — 13.5x update / 6.75x anchor-exposure
   asymmetry left the od8al precision collapse unattributable among 3 hypotheses).
2. **Per-domain-family metrics MANDATORY.** Every scored result reports per source/venue family
   alongside pooled; pooled-only numbers are rejected at review (precedent: WASB pooled 0.5670
   hides indoor 0.7395 vs outdoor-night 0.2933).
3. **Ensemble teachers for pseudo-labeling.** Any pseudo-label pass uses >=2 independent teachers
   with an explicit consensus/agreement rule, never a single teacher (single-teacher self-training
   reinforces its own blind spots; CoTracker3-style recipe per runs/research_sota_20260722).
4. **License = FYI only** (owner directive 2026-07-22, internal use). Lanes record license as
   metadata; no lane blocks, quarantines, or gates on license grounds. Protected-eval and
   compare-only quarantines are PROTOCOL, not license, and stand unchanged.

(Adjust the insertion point only if that anchor text is absent; then place it immediately before
the "## The spec" section and say so in the report.)

## Task 2 — write runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md
A short dated record (facts only, no new analysis):
- Named negative: `PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED` (recorded 2026-07-22 per manager
  directive; supersedes the plain "HONEST_MISS" shorthand for this lane).
- Result recap: preregistered two-family non-negative bar NOT met — hemel_test all-positive
  (F1 +0.0460) but od8al_val precision -0.1924 / F1 -0.0842 (runs/lanes/person_mixed_20260722/
  vm_pull/, gpu_phase_report.json; fleet ledger close entry 2026-07-22T14:08Z).
- WHY "undercontrolled": arms were not exposure-matched — anchor_train 1,066 lines vs mixed_train
  14,400 lines, both 20 epochs at AutoBatch=6 => mixed received ~13.5x more gradient updates and
  ~6.75x more human-anchor exposures than control (control 21,320 total exposures vs mixed
  288,000). The od8al precision collapse is therefore unattributable among: (a) harmful
  pseudo-labels, (b) 13.5x update overfit, (c) 6.75x anchor repetition overfit. Analysis:
  runs/handoff_20260722/ORCHESTRATOR_STATE.md section 5.
- Consequences: no follow-up GPU arm on this design; any PERSON reopen must lead with an
  exposure-matched control; this precedent is now standing method rule #1 in the run-lane skill.
- Ledger recording handled separately by trackE_datadebt_20260722 (do not touch the ledger here).

## Acceptance
- SKILL.md contains the new section verbatim (minus the insertion-note paragraph), existing
  content byte-unchanged elsewhere (show `git diff --stat`).
- NAMED_NEGATIVE.md exists with the facts above, each with its evidence path.
- Wide test suite unaffected (`MPLBACKEND=Agg` full run; failures must be proven pre-existing).

## CROSS-SIGNAL row
Consumes: person_mixed evidence + ORCHESTRATOR_STATE section 5. Feeds: every future lane spec
(rules enforced at ultra review), Track C reopen criteria.
BEST-STACK DELTA: (c) none — method governance; no stack change.
