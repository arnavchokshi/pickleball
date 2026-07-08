# LANE w6_instrudocs_20260708 — instrument/docs debt, VERIFY-FIRST (wave-6 queue #5 + one banked fix candidate)

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes. Working-tree changes only in your OWNED FILES. Manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or runs/manager/ boards — proposed bullet text goes in your report.
- Protected eval clips EVAL-ONLY; Outdoor/Indoor labels never without a ledger row.
- Honest reporting; PASS with full_suite.failed>0 not proven pre-existing = rejected.
- .venv/bin/python; importorskip("torch"); MPLBACKEND=Agg on wide runs.
- Artifacts under runs/lanes/w6_instrudocs_20260708/ ONLY. Other lanes' run dirs READ-ONLY (you may COPY a harness into your ownership if that is where it should live long-term — state the move in the report).
- VERIFY-FIRST DISCIPLINE (this is the whole point of this lane): for EVERY item below, first establish current state with grep/read evidence. If an item is ALREADY satisfied (wave-5 closedocs 26a96b11e may have covered it), close it with the evidence pointer — a verified already-done is a PASS for that item, not wasted work. Never re-do done work; never assume undone work is done.

## FILE OWNERSHIP (exclusive this wave)
- OWNED: CAPABILITIES.md, the BVP verify-harness file(s) you locate via runs/lanes/w5_bvpspan_verify_20260707/ (plus their tests), scripts/racketsport/train_ball_stage2.py + its tests, runs/lanes/w6_instrudocs_20260708/**.
- DO NOT TOUCH: process_video.py + remote_body_dispatch.py (w6_gate1b_knob lane), threed/racketsport/ball_arc_solver.py + tests (w6_magnus lane), cvat_upload/** (w6_labelpack lane), web/replay/**, ios/**.

## ITEMS (each gets: current-state evidence -> action or close-with-pointer)
1. **BVP verify-harness whole-span revision.** The w4 harness's axis-4 check was manager-RULED STALE during wave-5 (BUILD_CHECKLIST [W5 BVP SPAN v2 ACCEPTED 2026-07-08]; the v2 whole-span policy with frozen-baseline protected-span priors is the landed semantics, 792fa5fc6). Revise the verify harness so its whole-span policy matches the landed v2 semantics; PRESERVE the w4 version as a clearly-marked historical instrument (do not delete it — it is cited by wave-4 evidence). The revised harness must still be able to reproduce the v2 acceptance numbers (5/5 protected spans delta 0.0; floors B 0.7727272727 / W 0.8750) — run it once to prove.
2. **CAPABILITIES.md truth-claims for wave-5 landings.** w5_closedocs (26a96b11e) already did a truth-up. VERIFY-ONLY-THEN-PATCH-RESIDUALS: grep CAPABILITIES.md for claims covering (a) OFFICIAL preprocessing contract in training c1f707d6f, (b) BVP span v2 792fa5fc6, (c) P2-2 phase-1 latent decode 62d785ce3 (smoother UNWIRED, lambda_foot=0), (d) transport hardening baa7c911c, (e) stage1_official retrain result (internal-val only, NON-promotable, VERIFIED=0 unchanged). Fix ONLY residual gaps/overclaims; cite each edit against its evidence commit. Zero edits is a legitimate outcome if the truth-up was complete.
3. **Frame-scheduling nondeterminism watch item — ADJUDICATE, do not fix.** Boot prompt queue #5 carries "541 vs 276 frames same clip after body_compute_execution.json delete"; the wave-5 closeout bullet says the nondeterminism watch item was CLEARED (243=243 byte-identical on the close proof). Determine with evidence whether these are the SAME item (cleared) or TWO DISTINCT phenomena (541-vs-276 still unexplained). Search runs/ + logs for the 541/276 occurrence, identify the mechanism if cheap (read-only), and produce a short written adjudication with paths. If a real, distinct, reproducible nondeterminism exists: that is a DIAGNOSIS deliverable for the manager to rule on — do NOT change scheduling code in this lane.
4. **train_ball_stage2.py --resume-checkpoint (banked wave-5 fix candidate).** Wave-5 hit a spot preemption mid-retrain and noted train_ball_stage2.py lacks --resume-checkpoint. Add it, mirroring the PROVEN pattern in train_ball_pretrain.py (which demonstrated resume end-to-end in wave-3: resumed step 10->20, loss continuity, no reset). Acceptance: a CPU-cheap test proving save->resume->loss-continuity on a tiny deterministic run (seeded — the e4bdb4972 smokefix made init deterministic; follow that pattern), plus the scaffold/reference tests for the new flag. This is prep for the critical-path queue #1 retrain (owner labels -> seed re-run on the banked stage1_official base).

## ACCEPTANCE
- Per-item: the current-state evidence + (action taken | closed-with-pointer), with paths/commits.
- Item 1: revised harness reproduces the v2 numbers exactly; w4 historical version preserved + marked.
- Item 4: resume test green; budgets in STEPS (tiny fixed step count), no wall-clock claims.
- FULL wide blast-radius suite green (MPLBACKEND=Agg) since you touch repo source (harness + trainer); failures proven pre-existing at HEAD or fixed.
- Also verify: OWNER_CHECKIN_20260708.md (root, manager-created this session) does not trip the doc-allowlist test — if root-doc allowlist registration is needed for the pattern, register it same-lane (precedent: OWNER_CHECKIN_20260707.md exists — check how it is allowlisted).

## KILL / STOP CRITERIA
- Item 1: if the harness revision cannot reproduce the v2 acceptance numbers, STOP that item and report the discrepancy (needs-validation class) — do not adjust the numbers or the policy to force agreement.
- Item 3 is read-only by definition; any temptation to edit scheduling code = report instead.

## REPORT (schema-enforced)
objective_result vs the 4 items + suite; full_suite line; per-item evidence table; CHANGES file:line; HONEST ISSUES; proposed BUILD_CHECKLIST bullet; NEXT.
