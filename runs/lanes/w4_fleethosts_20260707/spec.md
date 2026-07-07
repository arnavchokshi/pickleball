# LANE w4_fleethosts_20260707 — micro fix: kill the recycled-IP footgun (wave-3 gotcha, queue #6)

## OBJECTIVE
Fleet VM external IPs RECYCLE across VMs on stop/start (wave-3: fan1 received fleet1's old IP;
`configs/ssh/a100_known_hosts` + `DEFAULT_REMOTE_HOST` in `scripts/racketsport/remote_body_dispatch.py`
pointed at dead/recycled IPs). Standing rule is "always pass --remote-host" — make the code ENFORCE
it and make known-hosts refresh a one-command idempotent operation.

## DESIGN (pinned)
1. `scripts/racketsport/remote_body_dispatch.py`: remove the `DEFAULT_REMOTE_HOST` constant as a
   silent default — `--remote-host` becomes REQUIRED. Omission fails loud with a message pointing
   at `runs/manager/gpu_fleet.md` (the fleet ledger) as where to find the current host. CHECK
   FIRST: grep the whole repo for `DEFAULT_REMOTE_HOST` and for callers of remote_body_dispatch
   (scripts, tests, docs); update every caller/reference you own; if a caller you do NOT own
   (fenced files: `scripts/racketsport/process_video.py`, `threed/racketsport/orchestrator.py`)
   depends on the default, STOP and report the proposed diff instead of editing those two files.
2. NEW `scripts/fleet/refresh_remote_host.sh` (or `.py` — match whatever `scripts/fleet/` already
   uses; CHECK its existing style): given `--host <ip>` (+ optional `--alias`), (a) `ssh-keyscan`
   the host, (b) REPLACE any stale entries for that ip/alias in `configs/ssh/a100_known_hosts`
   (idempotent — running twice yields identical file), (c) verify ssh connectivity with the fleet
   key + `BatchMode=yes` and print a PASS/FAIL stamp. Do NOT run it against a live host in this
   lane (no network in your sandbox) — unit-test the file-manipulation logic with fixture
   known_hosts content, and mark the connectivity check as the part that runs at fleet start.
3. Doc note: a short "fleet restart protocol" header comment in the new script (refresh known_hosts
   → update ledger → always pass --remote-host). Do NOT create any new root-level .md file.

## OWNED FILES (anti-collision fence)
`scripts/racketsport/remote_body_dispatch.py`, `tests/racketsport/test_remote_body_dispatch.py`,
`scripts/fleet/refresh_remote_host.*` (new) + its test, `configs/ssh/` (only if adding a README
comment file — do NOT edit/delete existing known_hosts entries in this lane). DO NOT TOUCH:
process_video.py, orchestrator.py, ios/**, ball_* files, runs/manager/**.

## SELF-VERIFICATION (mandatory — full blast radius, not a subset)
- `.venv/bin/python -m pytest tests/racketsport/test_remote_body_dispatch.py tests/racketsport/test_scaffold_tool_index.py -q`
  plus every test file your repo-wide grep shows importing/patching `remote_body_dispatch` or
  referencing `DEFAULT_REMOTE_HOST` (list them explicitly in the report; run them ALL).
- If the scaffold-index test requires registering the new CLI, register it in the SAME lane (CHECK
  `tests/racketsport/test_scaffold_tool_index.py` for the mechanism).
- Fix every failure you introduce, including adjacent suites. A pre-existing failure must be proven
  pre-existing (show it fails at HEAD before your change).

## SELF-ITERATION & BOUNDED FIX AUTHORITY
Iterate to green. You may refactor internal call sites of remote_body_dispatch freely within owned
files; anything requiring fenced-file edits → STOP + proposed diff in the report.

## KILL
If removing the default breaks >5 call sites in ways needing real design (not mechanical
threading), STOP and report the call-site inventory + proposal instead of a sprawling change.

## DISCIPLINE
`.venv/bin/python` everywhere; `pytest.importorskip("torch")` pattern if any new test touches
torch (unlikely here); no git branch/commit/push; never touch eval labels or the held-out ledger;
sandbox failures (network binds, .git locks) classified with proof they fail at HEAD too.

## STRUCTURED REPORT
Acceptance table: (1) omitting --remote-host exits nonzero with the ledger-pointing message;
(2) refresh helper idempotent on fixtures; (3) full listed suite green. honest_issues unsoftened.
