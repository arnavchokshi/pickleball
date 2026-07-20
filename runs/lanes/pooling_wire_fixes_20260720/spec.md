# pooling_wire_fixes_20260720 — apply the ultra reviewer's 3 exact fixes (nothing else)

Codex gpt-5.6-sol xhigh. The wire review (runs/lanes/ultra_review_wire_20260720/log.txt, verdict COMMIT_WITH_FIXES) prescribed EXACTLY three fixes. Apply them verbatim — no scope creep, no redesign. NO commits (manager commits).

1. scripts/racketsport/process_video.py:860 — the legacy OFF cache-identity translation must apply ONLY when the callable matches a pinned reviewed post-wire hash; otherwise future CAL/input-quality edits could silently reuse stale generations. Pin the current reviewed hash; on mismatch, fall through to normal (non-translated) identity so stale reuse is impossible.
2. tests/racketsport/test_pooling_wire_20260720.py:455 — hash-bind the fixture and COMPUTE + ASSERT the legacy far_centerline=4/96 baseline in the same test that proves the 63-support/0.356912553px/gate-ready recovery (so the test proves the before AND after, bound to the exact fixture bytes).
3. runs/lanes/pooling_wire_20260720/REPORT.md — line 19: correct the overclaim (the fixture proves recovery vs the computed 4/96 baseline; the pulled production baseline was separately 0/7); line 25: relabel wide-suite as "attributed-red (3952 passed / 27 failed, all failures attributed pre-existing)", not PASS.

ACCEPTANCE: the wire test file green; OFF-path byte-identity test still green; a stale-translation unit case (callable hash mismatch ⇒ NO translation) added and green; focused suite for touched files green. Report each fix APPLIED with the diff hunk.
