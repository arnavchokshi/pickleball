# LANE w4_syncstamp_20260707 — micro fix: version-stamp hashes git blobs (not working tree) + transfer retry classification

## OBJECTIVE
Two tool bugs found live by the w4_h100body GPU lane (evidence:
runs/lanes/w4_h100body_20260707/REPORT.md — read its HONEST ISSUES #1/#2 first):
1. `remote_body_dispatch.py --sync-remote-code`'s dirty/drift check hashes WORKING-TREE files,
   but the git-bundle sync ships COMMITTED blobs — so unrelated concurrent-lane dirty files
   produce FALSE "drift" and hard-gate the dispatch (the GPU lane had to build a temporary git
   worktree to get a clean stamp). Fix: the verification compares the REMOTE checkout's file
   hashes against the LOCAL COMMITTED blob hashes (git cat-file / git ls-tree of the synced ref),
   never the working tree. A separate, non-gating INFO line may still report local dirty files.
2. Large-transfer failure classification: the local ssh transport bug
   (`ssh_packet_write_poll: Result too large`) makes tar_batch fail with a NON-retryable exit-1
   while the rsync path fails exit-255 (retryable). Locate the tar_batch transfer helper (grep
   "tar_batch" in scripts/ + threed/ — CHECK its real location and current retry logic) and make
   its transport-failure exit path retryable with the same bounded-retry envelope the rsync
   fallback uses; classify `Result too large` explicitly as transport-retryable.

## ACCEPTANCE
- Unit test: a repo fixture where a tracked file is DIRTY locally but the committed blob matches
  the remote hash → verification PASSES (no false drift) with an INFO note; a fixture where the
  COMMITTED blob differs → verification FAILS loud (true drift still caught).
- Unit test: tar_batch transport-failure path retries within the bounded envelope and surfaces
  the final classified error; `Result too large` classified transport-retryable.
- Full blast radius: `.venv/bin/python -m pytest tests/racketsport/test_remote_body_dispatch.py
  <tar_batch's test file — grep it> tests/racketsport/test_scaffold_tool_index.py -q` plus every
  test file importing the modules you touch (list them; run ALL).

## OWNED FILES (anti-collision fence)
`scripts/racketsport/remote_body_dispatch.py`, `tests/racketsport/test_remote_body_dispatch.py`,
the tar_batch helper module + its test (name them in the report after the CHECK). DO NOT TOUCH:
`ball_arc_solver.py` (live repair lane), `foot_contact.py`, `camera_motion.py`,
`process_video.py`/`orchestrator.py` (fenced), `ios/**`, `runs/manager/**`, eval labels, ledger.

## DISCIPLINE
`.venv/bin/python`; no git branch/commit/push (fixture repos created inside your lane dir / tmp
are fine); no network; no new root-level .md; prove pre-existing failures at HEAD; sandbox
failures classified with proof.

## STRUCTURED REPORT
Acceptance table; CHANGES file:line; full_suite; honest_issues; next.
