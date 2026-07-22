# ULTRA ADVERSARIAL REVIEW of lane ball_b0_split_20260721 (read-only; do NOT modify any file)

Deciding reviewer before commit. Ground truth: runs/lanes/ball_b0_split_20260721/spec.md (contract),
report.json (claim), owned files per FILE OWNERSHIP, log.txt, artifacts under the lane dir.
Doctrine: AGENTS.md, runs/regroup_20260721/EXACT_PLAN.md §3.2 B0 + §2.2.

Focus hard on: (1) lineage-classification correctness (scratch vs corrected_prelabel vs
confirmed_prelabel — is the comparison against ORIGINAL prelabel/package lineage sound, or could a
corrected row be misfiled as confirmed and leak model-descended labels into the judge?);
(2) parent-source grouping correctness (the whole point of B0 — no per-clip leakage);
(3) the judge's purity claims: 167/167 scratch, zero train/val source intersection, zero protected
collisions over ALL protected frames — spot-verify against artifacts on disk; (4) the real task-87
export ingestion (350/350) — verify counts from the artifacts, not the report; (5) gate-gaming in the
7 synthetic tests; (6) the 31 wide-suite failures — none plausibly caused by owned files (orchestrator
verified sockets pass locally; storage failures trace to pre-session commit f9dc11dfc).

VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES (exact fixes) | REJECT (why). Be adversarial:
this judge will decide the BALL A/B experiment; contamination here poisons everything downstream.
