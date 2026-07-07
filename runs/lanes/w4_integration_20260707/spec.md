# LANE w4_integration_20260707 — the wave-4 integration micro-lane (ONLY lane authorized on fenced files)

## OBJECTIVE
Compose wave-4's queued fenced-file edits into `scripts/racketsport/process_video.py`, serialized
through this ONE lane per standing rule (TECH_BLUEPRINTS B.1.1). Two items, both small. The
deferred patch files under runs/lanes/*/deferred_patches/ are DESIGN NOTES ONLY (both proved to be
malformed pseudo-diffs) — re-derive the edits from the specs below; never git-apply those files.

## ITEM 1 — camera_motion_auto first-class mismatch keys (REQUIRED)
The landed camera_motion.py (commit cd0b59390) emits decode-orientation telemetry in its probe
result, and on consequential mismatch persists the reason only via the forced-string bridge
(`forced="auto_decode_orientation_untrusted:..."`). Make the pipeline summary carry it first-class:
- In process_video's camera_motion AUTO summary assembly (grep `camera_motion_auto` +
  `estimate_camera_motion_probe` call site ~:1263 — re-grep at HEAD), persist these keys from the
  probe result INTO the persisted camera_motion_auto summary dict when present:
  `decode_orientation_mismatch`, `decode_orientation_consequential_mismatch`,
  `decode_orientation_untrusted`, `decode_orientation_mismatch_reason`.
- Additive only: existing keys byte-compatible; absent-in-probe → keys absent (no fabricated
  defaults). Keep the forced-string bridge as-is (belt and suspenders).
- Acceptance: a test in tests/racketsport/test_process_video.py asserting (a) a probe result
  fixture carrying the 4 fields yields a persisted summary carrying them verbatim, (b) a probe
  result without them yields a summary without them, (c) existing summary keys unchanged.

## ITEM 2 — --remote-host parser posture (OPTIONAL — judge and possibly no-change)
fleethosts made RemoteConfig().host default empty so omitted --remote-host fails loud AT DISPATCH
with the ledger-pointing message. Evaluate whether a parser-level/post-parse validation in
process_video adds real value (clearer, earlier failure ONLY when a remote BODY dispatch would
actually run — never break local/no-GPU paths). If the existing failure is already loud, early
enough, and clearly worded, record NO-CHANGE-NEEDED with the evidence (the exact current failure
output) — that is a PASS outcome for this item.

## ACCEPTANCE (whole lane)
- `.venv/bin/python -m pytest tests/racketsport/test_process_video.py tests/racketsport/test_camera_motion.py tests/racketsport/test_remote_body_dispatch.py -q` green
  plus every test file your grep shows asserting on camera_motion_auto summary contents (list; run ALL).
- The process_video.py diff is MINIMAL (summary assembly + optional arg validation only) — include
  the full diff hunks in your report CHANGES.

## OWNED FILES (exclusive fence — no other lane may touch these right now)
`scripts/racketsport/process_video.py`, `tests/racketsport/test_process_video.py`. DO NOT TOUCH:
`orchestrator.py` (no queued change — leave it), `camera_motion.py`, `foot_contact.py`,
`ball_arc_solver.py`, `remote_body_dispatch.py`, `ios/**`, `runs/manager/**`, eval labels, ledger.

## DISCIPLINE
`.venv/bin/python`; no git branch/commit/push; no network; no new root-level .md; pre-existing
failures proven at HEAD; sandbox failures classified with proof.

## STRUCTURED REPORT
Acceptance table (incl. ITEM 2's judged outcome + evidence); CHANGES with the exact diff hunks;
full_suite; honest_issues; next.
