# LANE abc_audiofix_20260721 — E0 repair: audio-only rows never eligible in arm B

## HARD RULES
- No branches, no commits (orchestrator commits after ultra review). Read NORTH_STAR_ROADMAP.md §2
  and runs/regroup_20260721/EXACT_PLAN.md §2.1 + §3.1 E0 first.
- 4 protected clips EVAL-ONLY; protected 50-row event seed NEVER read. No GPU, no network.
- Honest reporting. Run the WIDE test suite (`MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -x -q`
  is NOT wide enough — run the full tests/racketsport suite without -x and report exact pass/fail counts).
- Artifacts under runs/lanes/abc_audiofix_20260721/. Other lanes' run dirs are READ-ONLY evidence.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/build_abc_arm_manifests.py
- tests/racketsport/test_abc_arm_manifests.py (extend, do not fork a new file)
Nothing else. Do NOT touch finetune_event_head.py, eval_event_head.py, abc_decision_gate.py,
datasets.py, or any runs/ evidence.

## CONTEXT (verified facts, 2026-07-21)
The seed-20260720 B manifest accepted 292 rows whose ONLY independent agreement family is
`audio_onset` (weight 0.25). E0 verdict `METHOD_INVALID_AUDIO_ONLY=292` — see
runs/lanes/abc_experiment_20260721/E0_VERDICT.md. Ground evidence:
runs/lanes/abc_experiment_20260721/vm_pull/abc_out/agreement_decisions.jsonl
(sha256 3a3463565e57a5cd909eaad01f2ddf6fa66f23468396f7162a94c85f8b1bf4f1; 2,192 decisions:
1,481 accepted = 292 audio-only + 773 kink-only + 416 both; 585 zero-agreement, 126 low-conf).
Defect: in build_abc_arm_manifests.py the rule is `accepted = needs_agreement_pass and count > 0`
over SIGNAL_FAMILIES = ("audio_onset", "ball_velocity_kink").

## OBJECTIVE (binding ruling, EXACT_PLAN §2.1)
1. **Eligibility:** a row is accepted into arm B ONLY if it has a `ball_velocity_kink`
   agreement. audio_onset alone -> `accepted_into_arm_b=false`, `pseudo_weight=0.0`,
   `rejection_reason="audio_only_no_physical_cue"`.
2. **Audio weight gating:** audio_onset may lift a row from the 1-agreement weight (0.25) to
   the 2-agreement weight (0.5) ONLY when that video's audio family beats its preregistered
   time-shift null. Otherwise the audio agreement is recorded in `independent_agreements` but
   is weight-inert (row stays 0.25). Field `audio_weight_eligible: bool` on every decision row.
3. **Time-shift null (preregistered, deterministic):** per video, observed_rate = fraction of
   eligible events with an audio match at max_delta_s (0.035s). Null: K=20 deterministic
   circular shifts of the audio onset times (offsets derived from the run seed, each with
   |offset| >= 1.0s, modulo video duration), recompute the match rate per shift.
   `beats_null` iff observed_rate > max(null rates). Record per-video block
   `audio_time_shift_null = {observed_match_rate, shift_offsets_s, null_match_rates,
   null_max_rate, beats_null}` in the arm-B manifest metadata AND input_bindings.
4. C must keep mirroring the CORRECTED B rows exactly (same rows/pixels/classes/weights,
   shuffled focal time). No change to C logic beyond inheriting B's corrected row set.
5. Determinism: identical inputs -> byte-identical manifests and decisions.

## ACCEPTANCE (all required)
- New/extended tests in test_abc_arm_manifests.py covering AT MINIMUM: audio-only rejection;
  kink-only -> 0.25; both + beats_null -> 0.5; both + fails-null -> 0.25 with audio recorded;
  null-field presence/shape; C mirrors corrected B row count; determinism (two builds byte-equal).
- Offline recount cross-check: apply the NEW eligibility rule to the pulled real
  agreement_decisions.jsonl (families are recorded per row; no media needed) and assert the
  corrected eligible count is exactly 1,189 (= 773 + 416). Put the recount code in the test file
  or a small script under the lane dir; report the number.
- Full suite: no new failures (report exact counts; any failure must be proven pre-existing).
- Report the exact VM rebuild command line (build_abc_arm_manifests.py invocation) unchanged
  in flags — the fix must not change the CLI contract.

## EVIDENCE TO READ FIRST
- runs/lanes/abc_experiment_20260721/E0_VERDICT.md
- runs/lanes/w1b_abc_loader_20260721/VM_ABC_RUN.md §5 (materializer contract)
- runs/lanes/abc_experiment_20260721/vm_pull/abc_out/agreement_decisions.jsonl (sample rows)

## DATA CONTRACT
- Inputs: the pulled decisions file above (hash pinned); synthetic fixtures only for tests.
  No protected/compare asset readable. Ledger row: pending data-steward bootstrap
  (data_steward_ledger_20260721); this lane's input hash is pinned here.
- Utilization delta: arm-B eligibility corrected 1,481 -> 1,189 rows (292 audio-only excluded).
- No GPU. Effort cap ~4h.
- End-of-lane number: corrected eligible-row recount (must equal 1,189) + full-suite counts.

## CROSS-SIGNAL
Consumes: audio onsets (bounded, null-gated), BALL 2D velocity kinks, pb.vision teacher events.
Feeds: EVENT head A/B/C causal experiment (E1/E2), future typed-wrist cue slot.

## BEST-STACK DELTA
None — experiment-infrastructure correctness; no best_stack.json entry exists for the event head.

## MANDATORY STRUCTURED REPORT (report.json via output schema)
objective_result PASS/FAIL vs the acceptance items; full_suite passed/failed counts; HONEST
ISSUES; artifact paths; the 1,189 recount; the unchanged VM rebuild command.
