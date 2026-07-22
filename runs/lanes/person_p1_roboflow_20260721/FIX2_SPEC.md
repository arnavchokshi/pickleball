# FIX ROUND 2 for person_p1_roboflow_20260721 — R2 findings (runs/lanes/person_p1_roboflow_20260721_review/review_r2.json)

Same ownership. Read the R2 review first. ORCHESTRATOR RULING (fail-closed, holdout-side-wins, per
EXACT_PLAN §2.2 family rule): source families proven to share original footage with the frozen
validation source merge INTO the validation family as WHOLE sources — no frame-level carve-outs of
any source, no threshold shopping.

1. P1-SPLIT-LEAK-R2 (CRITICAL): merge pickle-es3fs/pickleball-video and
   nigh-workspace/pickleball-player-object-detection-cc2sw into the od8al validation family
   (whole sources). Implement a CONTENT-LEVEL cross-split leak scan as a mandatory production
   check (the reviewer's method: pHash<=6 candidate pairs verified by SSIM/ORB or equivalent) so
   cross-workspace renames are caught; it must run on the final split and FAIL on any hit.
   Regenerate ALL artifacts/counts.
2. REPORT THE RETENTION GATE HONESTLY: expected post-merge train ~8,887 images across ~7 family
   groups vs the frozen bar (>=5,000 images AND >=8 train family groups). If <8 family groups, the
   verdict is the named negative PERSON_RF_POOL_TOO_THIN and the training-ready gate must be
   permanently CLOSED for this export (P2 = NO_ATTEMPT_PREREQ). Do NOT massage families, invent
   sub-source splits, or relax the bar to pass. State the exact final numbers.
3. P1-HUMAN-GATE-LEGACY-YAML-BYPASS (HIGH): the preserved pre-fix package must not be trainable —
   remove its data.yaml (or rewrite the dir to a *_QUARANTINED_LEAKED name with a REFUSED.md and
   no yaml). No path may expose the unbalanced 9,129-image train set.
4. P1-HUMAN-GATE-PASS-STATE-CONTRADICTION (HIGH): a legitimate future PASS package must bundle the
   COMPLETED review CSV + a non-PENDING audit manifest; tests must assert package-state consistency.
5. P1-REPORT-SCHEMA (MEDIUM): emit the artifacts key properly in report_fix2.json.
No NEW wide-suite failures beyond the known environmental set. Report to report_fix2.json.
