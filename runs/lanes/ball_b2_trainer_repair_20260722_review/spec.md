# LANE SPEC — ball_b2_trainer_repair_20260722_review (ULTRA ADVERSARIAL REVIEW of the trainer repair)

Dispatched 2026-07-22 by the Track-B agent. Codex `gpt-5.6-sol`, effort `ultra` (gate-adjacent:
this repair is a review_r4 minimum_reopen_condition for B2; a defective sampler fix silently
corrupts the flagship experiment). CPU-only, READ-ONLY outside your lane dir.

## WHAT TO REVIEW

`ball_b2_trainer_repair_20260722` (spec, report.json, parity_cpu_fixture_summary.json,
sampler_proof.json) and its diff to `scripts/racketsport/train_ball_stage2.py` (initial sha
9d2d3261..., reported new sha b8e653da...) + its owned test module changes, against:
- its spec's 5 acceptance items, and
- `runs/lanes/ball_b1b2_prep_20260722_review_r4/review_r4.json` blocking facts + minimum_reopen_conditions
  (this repair exists to discharge conditions 1's trainer half).

## ADVERSARIAL CHECKLIST (minimum)

1. **Parity harness**: verify by reading + running that (a) candidate-vs-itself errors (rc=2, no
   artifact) — reproduce it; (b) --baseline-rev + explicit expected-hash are MANDATORY with no
   HEAD default; (c) the baseline source is loaded from the named git blob (not working tree); (d)
   the CPU fixture parity actually exercises the real training step loop, not a stub — trace what
   "7 steps" ran. Re-run the fixture yourself and require identical results.
2. **Sampler Option B**: the claim is duplicate-free-by-construction at permutation splices,
   deterministic from (manifest size, seed), zero additional random draws, all rows preserved per
   epoch-equivalent, and semantic fail-closed assertion retained. Read the implementation for
   subtle bias: does the splice fix change the DISTRIBUTION of pairings in a way that
   systematically disadvantages any sample (e.g., tail rows resampled earlier/later)? Re-run the
   1001/8/20260722 reproduction and the 1,356,000-batch property sweep (or an equivalent
   independent sweep incl. other manifest sizes: 1000, 1007, 1023, 2000+; batch boundaries at
   size%8 != 0 if legal). Historical step-1751 must show 7 unique pre-fix and 8 post-fix.
3. **Human-path invariance**: the report claims AST-identity for human sampler/loader/assertion.
   AST-identity is necessary not sufficient — check imports/globals/constants the human path
   reads that the repair may have touched; then confirm the executable proof story: the
   distinct-baseline CPU fixture showed exact_losses_identical + sample-order identical + model
   state identical between baseline 8118ff0e and candidate b8e653da on the human-only
   configuration. Verify that fixture ran arm-A-equivalent (no SST) — if it ran with SST enabled,
   parity passing would be suspicious, flag it.
4. **Scope**: git status — only the trainer + owned test module + lane dir changed by this lane
   (concurrent drift from other tracks exists: best_stack rev-15 = Track C, court/audio files =
   other B lanes; classify, don't blame). best_stack untouched by THIS diff.
5. **Wide-suite claim**: 42 failures not-all-proven-pre-existing — cross-check against
   `runs/lanes/ball_audio_anchor_20260722/wide_suite_failures.txt` (41 classified) + the manager's
   KNOWN_ATTRIBUTED(trkC_constraints_wire) certification; identify any failure NOT explained by
   (known-list ∪ trkC-drift ∪ other-lane-owned files). Any such residual = potential real
   regression from THIS diff: investigate before verdict.
6. **review_r4 discharge check**: state explicitly, item by item, which review_r4
   minimum_reopen_conditions this repair discharges (parity-harness half, SST-uniqueness half),
   which remain open (B1 gate, real CUDA parity vs NEW candidate hash b8e653da..., builder pin,
   resume review, fresh arming review), and the exact NEW candidate sha the future arming review
   must pin.

## VERDICT

review.json: VERDICT ACCEPT | REJECT; per-item evidence; on ACCEPT include
`recommended_commit_scope` (exact file list) and `new_candidate_trainer_sha256` restated; on
REJECT surgical defect list (repair round 2).

## RULES

No branches/commits; don't modify reviewed files; wide suite optional given documented drift (if
you run it, classify); `VERIFIED=0`; structured report.json.
