# ULTRA ADVERSARIAL REVIEW of lane person_p1_roboflow_20260721 (read-only; do NOT modify any file)

You are the deciding reviewer before commit. Ground truth: runs/lanes/person_p1_roboflow_20260721/spec.md (contract),
runs/lanes/person_p1_roboflow_20260721/report.json (claim), the lane-owned files in the spec's FILE OWNERSHIP block,
runs/lanes/person_p1_roboflow_20260721/log.txt, and the exported artifacts under runs/lanes/person_p1_roboflow_20260721/.
Repo doctrine: AGENTS.md, runs/regroup_20260721/EXACT_PLAN.md (§3.4 P1 + §2.2 quarantines).

Review for: (1) correctness bugs; (2) spec-compliance (fork-family grouping, whole-source splits,
NC + adjacent-sport exclusion, exhaustive protected collision check — is 2,994 descriptors truly
exhaustive over all 4 protected clips' frames?); (3) gate-gaming/self-certification; (4) honesty of
report vs artifacts (spot-check counts on disk); (5) protected-data violations; (6) the 32 wide-suite
failures — verify none are plausibly caused by the owned files (orchestrator verified sockets pass
locally; storage failures trace to pre-session commit f9dc11dfc; challenge if code says otherwise).

VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES (exact fixes) | REJECT (why). Be adversarial.
