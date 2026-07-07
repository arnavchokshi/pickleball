# LANE w4_footattr_diag_20260707 — READ-ONLY diagnosis+design: upstream per-foot contact attribution

## OBJECTIVE
Design confident per-foot contact-phase attribution AT SOURCE. Measured state (two independent
wave-3 diagnoses converged): 100% of contact phases consumed by grounding are confidence-free
`bilateral_from_player_stance` placeholders with `source_phase_foot: unknown`; exact-foot BODY
agreement is only 0.363–0.651. Wave-3's `w3_phasefix` fixed the CONSUMER side honestly: per-foot
confidence-bearing phase schema + weak-bilateral demotion — so today `grounding_refine` is an
honest NO-OP (0 confident phases exist on the eval clips; it self-kills rather than consume junk).
The un-kill requires an UPSTREAM PRODUCER that emits confident per-foot phases. Your job: pin why
attribution is unknown-foot today, inventory what per-foot evidence actually exists at source, and
deliver a ruled-ready design. **You change NO production code.**

## EVIDENCE TO READ FIRST
- `runs/lanes/w3_slidediag_20260707/REPORT.md` §6 and `runs/lanes/w3_groundref_diag_20260707/REPORT.md` §5
  — the converged root-cause evidence (read these FIRST; do not re-derive what they proved).
- `runs/lanes/w3_phasefix_20260707/` (spec.md + report.json) — the phase schema, confidence rules,
  weak-bilateral demotion, `foot_lock_gate_stream` instrumentation, and the non-gated companion
  metric `max_candidate_phase_slide_m`.
- Producer code: grep `bilateral_from_player_stance` and `source_phase_foot` across
  `threed/racketsport/` to find where phases are born (stance detection / placement); read the
  stance detector and what per-foot signals it sees.
- Consumer code: where confident phases WOULD be consumed (`grounding_refine`, foot pin, the slide
  gate `body_grounding_quality.py` — gated key `grounding_metrics.max_foot_lock_slide_m`, bar 0.030 FROZEN).

## DELIVERABLES (all under runs/lanes/w4_footattr_diag_20260707/ — your only writable area)
1. `REPORT.md` with:
   a. Signal inventory: every per-foot signal available at phase-production time (per-foot world
      height, vertical velocity, horizontal speed, stance geometry, mesh/skeleton foot joints,
      visibility/confidence), each MEASURED on the banked eval-clip artifacts (read-only analysis of
      existing runs/ outputs — pick the freshest 4-clip run set; name the run dirs you used): what
      fraction of frames could be attributed left/right at what agreement, using that signal alone
      and in combination.
   b. Root cause: why the current producer emits `unknown` (design gap vs signal gap — is the
      information present but discarded, or genuinely absent?).
   c. ≥2 candidate designs with trade-offs, ONE recommendation. For each: exact files/functions,
      the attribution algorithm, the CONFIDENCE definition (what makes a phase "confident" — must
      be a measurable statistic, not a vibe), predicted confident-phase counts per eval clip
      (computed from your read-only analysis, not guessed), failure modes.
   d. Risk analysis: the slide gate is GREEN and FROZEN (`max_foot_lock_slide_m` ≤ 0.030 on all 4
      clips, fresh-GPU proof @ ad75c875c). The design must state its predicted effect on the gate
      and HOW the offline predictor (the wave-3 verifier's replay harness — locate it in the
      w3_phasefix/verify lane dirs) validates the fix BEFORE any GPU spend. A design that cannot be
      offline-predicted must say so and propose the cheapest safe validation.
   e. The executable defect proof a future adversarial verifier would use (e.g. an assertion on a
      banked clip's artifacts that fails today — 0 confident phases — and must pass post-fix with
      the gate still green).
2. `analysis/` — the scripts + intermediate JSON backing (a).

## HARD CONSTRAINTS
- READ-ONLY on production/test files; write only in your lane dir.
- Never touch eval labels, `runs/manager/heldout_eval_ledger.md`, `ios/`, `runs/manager/`.
- No rejection reason may reference the gate threshold (the wave-3 unfailable-gate lesson —
  circularity is forbidden); flag any existing circularity you find as a defect.
- No git operations. No network. `.venv/bin/python`.

## SELF-VERIFICATION
Analysis scripts run clean twice, deterministic. `git status --porcelain` shows only lane-dir adds.

## SELF-ITERATION
Iterate until (a)-(e) are complete with numbers OR a genuine wall (e.g. required artifacts absent
locally) — then STOP and state exactly what a GPU capture must produce to unblock the analysis.

## STRUCTURED REPORT
objective_result PASS = deliverables complete with measured (not guessed) numbers. Include
per-clip attributable-fraction numbers in the acceptance table. honest_issues: unsoftened.
full_suite: not applicable (read-only) — 0 failed, note "read-only diagnosis".
