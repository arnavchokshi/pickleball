MANAGER RULING r2 — your r1 is ruled PASS w/ attribution: all 13 tests/server tests re-verified
green locally by the manager; the 6 socket-bind failures re-verified LOCALLY GREEN (sandbox-only,
as you claimed); your bundle-policy direction is doctrine-correct (P0-E: exit-success must never
mean complete).

ONE bounded follow-up, fence EXTENDED to tests/render_service/** (no other lane owns it):
MIGRATE the 10 stale tests/render_service tests that encode the retired exit-success=>complete
behavior to the new honest contract. Rules:
- Do NOT restore any exit-code-upgrade path and do NOT weaken your fail-closed changes to make
  tests pass. The tests move to the new contract, not the reverse.
- Where a test legitimately modeled a COMPLETE bundle, give its fixture real policy evidence
  (mandatory artifacts + advertised URLs) so it earns complete honestly; where it modeled bare
  exit-success, assert partial + the missing-capabilities payload.
- Keep each test's original intent documented in-place (what behavior it guards).
- Rerun: tests/render_service + tests/server + the two socket test files
  (tests/racketsport/test_court_keypoint_review_server.py, tests/racketsport/test_review_input_server.py)
  — all green in your sandbox except socket-bind-only failures (list them).
- SELF-WRITE runs/lanes/ns015_statuspack_20260710/report_r2.json (same schema shape as r1).
No branches/commits. Do not idle-wait; end only with the final report or a hard blocker.
