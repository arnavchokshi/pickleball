# LANE w3_groundref_diag_20260707 — grounding_refine 4/4 self-kill DIAGNOSIS (wave-3 #2; diagnosis ONLY)

## HARD RULES
- You are Codex lane `w3_groundref_diag_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- DIAGNOSIS LANE: you own NO repo files; edit nothing outside runs/lanes/w3_groundref_diag_20260707/. No git commit/branch/push.
- Protected data: 4 eval clips EVAL-ONLY; reading pipeline artifacts is fine; Outdoor/Indoor labels never. Held-out videos pwxNwFfYQlQ / vQhtz8l6VqU appear nowhere.
- Honest reporting; every number carries its artifact path + repro snippet. Threshold changes may be PROPOSED (clearly labeled needs-manager-ruling) but the default posture is fix-at-source or kill-the-stage.
- Always `.venv/bin/python`. Read first: BUILD_CHECKLIST.md last ~15 bullets.

## CONTEXT (established)
- `grounding_refine` reports `kill_recommended=True` on ALL 4 eval clips, INCLUDING runs untouched by the 30fps fix — the wave-2 verify lane proved this PRE-DATES wave 2 (pre-existing default characteristic, not a wave-2 regression).
- Wave-2 landed `foot_contact_phases` producers; the closing run shows phases produced+consumed per clip (44/116/16/12 across the 4 clips) and grounding_refine consumes them and still self-kills honestly.
- Manager hypothesis (test, don't assume): phase QUALITY/sparsity — the refine stage's internal acceptance criterion may be starved by too-few/too-noisy contact phases.

## OBJECTIVE — answer with numbers
1. **The criterion itself**: extract the exact self-kill decision logic from the stage code (conditions, values, where computed) — cite file:line. What inputs does it evaluate?
2. **Per-clip measured values**: for each of the 4 clips in runs/lanes/wave2_freshworlds_20260707/ (and p06_freshworlds_20260706 where useful), tabulate the actual values the criterion saw vs its cutoffs. WHICH condition trips, per clip?
3. **Upstream quality**: are the foot_contact_phases (counts 44/116/16/12) sufficient in count/duration/confidence for what refine needs? Quantify (phases per second, coverage of stance frames, agreement with stance detection).
4. **Root cause ruling material**: which is it — (a) phases too sparse/noisy (upstream fix), (b) refine consumes them wrongly (consumption bug), (c) criterion miscalibrated for real clips (threshold proposal → manager ruling), (d) refine genuinely adds no value on these clips (kill-the-stage proposal: what would be lost — quantify refine's delta when force-enabled vs killed on the same world, if artifacts allow offline comparison; if force-enable requires a GPU run, say so).
5. **Recommendation**: exactly one recommended path with evidence, plus what a fix lane's acceptance numbers should be and how the wave-end fresh GPU run proves it.

## ACCEPTANCE
Criterion documented with file:line; per-clip value-vs-cutoff table for all 4 clips; upstream phase-quality quantified; one recommended path with the decisive comparison; explicit list of anything unmeasurable offline. No code changes anywhere.

## EVIDENCE
- runs/lanes/wave2_freshworlds_20260707/ + runs/lanes/p06_freshworlds_20260706/ (grounding_refine outputs, kill records, phases artifacts)
- runs/lanes/wave2_rootjump_verify_20260706/ (its grounding_refine pre-existing finding)
- Code READ-ONLY: scripts/racketsport/refine_body_grounding.py + the threed/ grounding refine implementation + foot_contact_phases producer code.

## STRUCTURED REPORT
objective_result PASS/PARTIAL/BLOCKED; the 5 answers; recommendation section; artifact index; HONEST ISSUES; NEXT; state "no code changed" explicitly.
