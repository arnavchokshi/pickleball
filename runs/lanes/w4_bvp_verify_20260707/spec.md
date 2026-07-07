# LANE w4_bvp_verify_20260707 — ADVERSARIAL ADJUDICATION: is the BVP re-segmentation genuinely equal-or-better?

## OBJECTIVE
The w4_bvp lane (changes UNCOMMITTED in `threed/racketsport/ball_arc_solver.py` +
`tests/racketsport/test_ball_arc_solver.py`) reports: the 5 exact D.3(b) baseline intervals are
0/5 preserved BY IDENTITY — reselection re-segmented them — while claiming the replacement
segments cover the same spans at fit/fallback status with LOWER endpoint errors (except Burlington
[497,543], baseline endpoint_error ZERO, which got SPLIT). Product F1 is exactly unchanged; D.3(a)
improved. The manager must decide whether to RE-RULE the acceptance from exact-identity to
span-equivalence. Your job: adversarially test the span-equivalence claim. The wave-3 lesson
looming over this: an acceptance that cannot fail because the measured object VANISHES
("missing_exact_interval") is the unfailable-gate class. Verdict SPAN-EQUIVALENT-GOOD only if
every attack fails.

## READ FIRST
- `runs/lanes/w4_bvp_20260707/` (report.json + acceptance_audit.json — the per-interval audit)
- `runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json` (the frozen baseline: D.3 gate
  labels, per-interval baseline/after fields)
- `git diff HEAD -- threed/racketsport/ball_arc_solver.py tests/racketsport/test_ball_arc_solver.py`

## ATTACK SURFACE (execute each; add your own)
1. SPAN-EQUIVALENCE, MEASURED: for each of the 5 baseline spans, reconstruct baseline vs
   post-change segment maps (replay both from banked candidates — pre-change replay via a COPY of
   ball_arc_solver.py with the diff reverted in your lane dir). Per span measure: (a) fraction of
   the span covered by `fit*` segments; (b) reprojection residuals over the span (same statistic
   both sides); (c) endpoint_error_m of covering segments vs baseline; (d) junction count +
   junction discontinuity magnitudes (position/velocity deltas at segment joins — MORE junctions
   with discontinuities is a real quality loss even at lower per-segment error); (e) confidence
   tier assigned (fit vs fit_bvp_fallback — fallback carries 0.30 vs 0.92 viewer confidence: a
   downgrade to fallback is NOT equivalence, quantify it). Any span materially worse on (a)-(e) =
   DEFECT with numbers.
2. THE SPLIT [497,543] (Burlington): was the split physically motivated (a real
   bounce/contact/event inside the span in the banked event data?) or noise-chasing (two arcs
   fitting jitter better than one true arc)? Compare the single baseline arc's residual profile vs
   the split pair's, and check junction sanity (velocity continuity at the split point vs a real
   impact signature). Noise-chasing split on a baseline-perfect interval = DEFECT.
3. UNFAILABLE-ACCEPTANCE AUDIT: the audit records `missing_exact_interval` when identity vanishes.
   Prove whether a TRUE DEMOTION (interval survives with same identity but drops below
   fit_bvp_fallback) is still detectable and would still FAIL the audit — construct the case with
   a mutated COPY (force-demote one interval) and run the lane's audit path against it. If the
   audit passes anyway, the acceptance is unfailable = DEFECT.
4. ANCHOR-PRESERVATION RULE NON-VACUOUS: find or construct (mutated-copy harness) a case where the
   new anchor-preservation rejection diagnostics actually PREVENT a demotion that the pre-change
   code performed. If the rule never fires on any real clip replay and cannot be made to fire on a
   constructed case, it is dead code = DEFECT. Also mutation-check the new tests (revert the rule
   in a copy → new tests must fail).
5. INDEPENDENT NUMBERS: re-run the D.3(e) eval-suite scoring and the D.3(a)/(c)/(d) replay
   yourself from banked artifacts; confirm the reported values (F1 0.772727/0.875000, bad_fit=0,
   violation_fraction 0.0). Divergence = DEFECT.
6. LOO REFIT CORRECTNESS: the LOO now refits per holdout. Spot-verify on one interval that the
   per-holdout refit actually re-solves (different arc params per holdout) rather than reusing the
   segment arc (the original bug) — instrument via the copy, not production files.

## HARD CONSTRAINTS
READ-ONLY on production/test files (all mutations on COPIES inside your lane dir; revert-diff
replays via copies + sys.path shims). No git operations beyond read-only diff. `.venv/bin/python`.
Outdoor artifacts: banked replay ONLY (no held-out label access — the ledger is untouchable).
Never touch eval labels / ios/ / runs/manager/.

## STRUCTURED REPORT
objective_result PASS = your adjudication COMPLETED. Prominent VERDICT = SPAN-EQUIVALENT-GOOD or
DEFECTS-FOUND. Acceptance table: one row per attack with the measured numbers (esp. the per-span
(a)-(e) table and the [497,543] split analysis). Every defect ships a runnable proof + exact
command. honest_issues: what you could not measure locally and what the wave-close fresh-GPU chain
runs must therefore check. next: your recommendation on the re-ruling question (exact-identity vs
span-equivalence vs demand-a-fix), with the evidence that supports it — the manager rules.
