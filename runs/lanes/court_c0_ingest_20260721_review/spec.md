# ULTRA ADVERSARIAL REVIEW of lane court_c0_ingest_20260721 (read-only; do NOT modify any file)

You are the deciding reviewer before commit. Ground truth: runs/lanes/court_c0_ingest_20260721/spec.md (the contract),
runs/lanes/court_c0_ingest_20260721/report.json (the claim), the lane-owned files named in the spec's FILE OWNERSHIP
block, and runs/lanes/court_c0_ingest_20260721/log.txt. Repo doctrine: AGENTS.md, runs/regroup_20260721/EXACT_PLAN.md.

Review for: (1) correctness bugs in the owned files; (2) spec-compliance gaps (every MUST in the
spec); (3) gate-gaming or self-certification (tests that assert what the code does rather than what
the spec requires; fixtures engineered to pass); (4) honesty of report claims vs actual code/artifacts;
(5) quarantine/protected-data violations (protected 50, eval_clips, IYnbdRs1Jdk, compare-only pb.vision
IDs, NC-licensed source); (6) the 32 wide-suite failures — verify NONE are plausibly caused by the
owned files (the orchestrator locally verified: socket tests pass outside sandbox; the 2 storage
failures trace to files committed in f9dc11dfc before this session; court-fixture drift predates
these lanes — challenge this attribution if the code says otherwise).

VERDICT required in your final JSON: ACCEPT | ACCEPT_WITH_FIXES (list exact fixes) | REJECT (why).
Be adversarial: a wrong ACCEPT costs more than a wrong REJECT.
