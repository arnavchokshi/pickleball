# pooling_wire_20260720 final report

Status: `PARTIAL`  
Truth label: `VERIFIED=0`  
Recorded: `2026-07-20T21:32:20Z`

The default-OFF calibration evidence seam is wired and locally verified. The
required single GPU replay was not launched because two fresh GCP
reconciliation attempts failed before API contact on DNS resolution for
`oauth2.googleapis.com`; the browser fallback also had no available browser
session. Fleet policy forbids provisioning without a successful live instance
and disk list.

## Acceptance

| Target | Result | Evidence |
|---|---|---|
| OFF byte-identical behavior | PASS | Frozen reuse and cold external-CAL artifact goldens pass. OFF also preserves the exact HEAD CAL and input-quality `StageSpec` callable hashes, so existing generations are not invalidated. |
| ON fixture proves recovery | PASS | The hash-bound 96-frame no-decode fixture computes the legacy `far_centerline=4/96` baseline, then recovers pooled support `51 optimize + 12 holdout = 63`; geometry p90 `0.35691255314973347 px`; effective readiness `true`; aggregate mean/p95 `3.6498698057003773/10.253094225879789 px`. The pulled production baseline was separately `0/7`. |
| Readiness bar unchanged | PASS | No diff in `threed/racketsport/court_line_evidence.py` or `court_auto_evidence.py`. Selector defaults remain confidence `0.5`, distance `24 px`, visible fraction `0.2`; aggregate defaults remain confidence `0.5`, mean `8 px`, p95 `16 px`. |
| Raw evidence immutable | PASS | `court_line_evidence.json` and `court_calibration.json` remain byte-identical. Raw sampled evidence and accepted pooled evidence are separate hash-bound sidecars; candidate refinement/calibration is never consumed. |
| Static-camera guard | PASS | The unchanged `3 px` dispersion bound typed-abstains on smooth/coherent drift, short or sustained tail motion, partial-line motion, unassigned raw-sample motion, boundary degradation, and zero raw-template coverage. Clean Drill evidence remains accepted. |
| Determinism | PASS | Reversed frame order produces identical canonical pooled bytes; serialized pooled evidence round-trips through typed validation. |
| Focused suite | PASS | `67 passed in 29.22s`; broad CAL/process/identity slice `269 passed in 46.99s`. |
| Wide suite attributed | ATTRIBUTED-RED (`3952 passed / 27 failed`, all failures attributed pre-existing) | `25 skipped in 2858.24s`. Details below. A clean-HEAD isolation suite was not run, so `failures_all_preexisting=false`. |
| Ultra re-review | PASS | Independent final review passed OFF identity, unchanged thresholds, exact Drill metrics, raw immutability, provenance, static guard, determinism, scope, and no-commit/no-push checks. |
| Exact GPU rerun command | PASS | See `RERUN_CMD.md`. |
| One GPU replay | FAIL | Not launched: live fleet reconciliation is DNS-blocked and no browser control session is available. |

## Implementation

- `scripts/racketsport/process_video.py`
  - Adds default-OFF `--court-line-evidence-pooling`.
  - Runs the exact proven 96-frame `seed_guided_paired_edges` pool only when
    enabled.
  - Emits `court_line_evidence_pool_raw_frames.json` and
    `court_line_evidence_pooled.json`.
  - Recomputes pooled readiness through the unchanged selector and aggregate
    functions, validates all provenance/hashes, and exposes the effective
    evidence only in memory to input-quality and the pre-tracking court gate.
  - Preserves exact legacy OFF stage identities.
- `threed/racketsport/court_line_robustness.py`
  - Provides the proven pool entrypoint, typed pooled loader, unchanged-bar
    evidence combiner, and v4 static-consistency guard.
  - Marks accepted additions `source=pooled_static`.
- `tests/racketsport/test_pooling_wire_20260720.py`
  - Adds portable, no-video-decode Drill fixtures and OFF/ON/gate/guard/
    determinism/tamper tests.

## Wide-suite attribution

- 12 court-finding selector/temporal/oracle failures: two pre-existing
  untracked IMG1605 review frames change the fixture from one frame to three
  and make the legacy three-frame median proposal fail. Existing legacy
  keypoint definitions are AST-identical to HEAD.
- 5 IMG1605 partial-label/eval/geometry failures: stale assertions versus the
  pre-existing ignored progress artifact and tracked three-frame partial label.
- 3 court-keypoint review-server failures, 3 review-input-server failures, and
  2 persistent BODY worker failures: restricted sandbox rejects TCP/AF_UNIX
  `socket.bind` before behavior under test.
- 1 overlapping-court report failure: pre-existing untracked
  `eval_clips/ball/pbvision_11min_20260713/` raises the scanned directory count
  from five to six.
- 1 storage-policy test failure: the required untracked Drill source video is
  `61,628,656` bytes and is the audit's sole unknown large source.

## Other checks

- `main`; `origin/main...HEAD = 0 0`.
- No commit or push.
- `git diff --check`: pass.
- Python compilation: pass.
- Scaffold index: 301 tools, 0 missing direct CLI tests.
- Dead-code audit: pass, 594 Python sources, 0 unknown.
- Storage audit: fail only on
  `data/pbv_replay_20260720/xkadsq9bli3h/max.mp4`.

## Honest limitations

- Pulled production evidence was `0/7`; the current local runtime yields `1/7`,
  while the frozen 96-frame evidence reproduces the proven `4 -> 63` recovery.
  The exact VM/runtime discrepancy remains unresolved.
- Static consistency cannot prove motion whose misleading raw samples remain
  inside the unchanged `3 px` robust bound.
- OFF compatibility uses pinned HEAD callable hashes for the two opt-in-touched
  stages; future OFF-reachable edits must deliberately update that
  compatibility contract.
- No GPU replay artifact exists. `VERIFIED=0` remains binding.

## GPU handoff

Run `RERUN_CMD.md` exactly once only after a live instance/disk reconciliation,
two-sided source/video hashes, GPU/decode checks, and Playwright/Chromium
preflight succeed. Current production hashes:

- `process_video.py`: `aca54b3ecd5cb48292cb8192711cece3ff6fae68a4a58c98a0982d07d8c9e126`
- `court_line_keypoints.py`: `1ac1cccfca8b5ae10aee2c7e658e108f845d856f5ab3d9e979228079b15c9081`
- `court_line_robustness.py`: `f5d02cfa65faacc4b86fd0a704819edddd07f53e34440f73e515c60660e1a723`
- Drill video: `5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181`
