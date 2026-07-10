MANAGER RULING r2 — your diagnosis is ACCEPTED (all three claims independently re-verified: the
b437b4118 cap, the w7_critique calibration failure disproving the spec premise, the exact-signature
repro). The outside-window kill rule is CLEARED by the manager: the root cause is proven by your
local repro, so a fix is NOT speculative. YOU ARE NOW AUTHORIZED to implement the fix. Same file
ownership as before (process_video.py, orchestrator.py, process_video_body_frames.py, their tests,
your lane dir). Same hard rules (no branches/commits; honest reporting).

DESIGN REQUIREMENT (single source of truth): the defect is TWO independent frame-set selections —
the materializer's uniformly-capped 1200 set vs BODY's independently derived request set. The fix
must make BODY's execution frame set a SUBSET of (ideally identical to) the materialized set by
construction, not by luck:
- Derive the required frame set ONCE (including any max-frame/cap policy) and hand it to BOTH the
  materializer and BODY scheduling; OR make the materializer materialize exactly the BODY execution
  set. Preserve the cap's INTENT (bounding work) — do not simply delete the cap.
- Enforce equality at the FRAMES stage: typed missing-frame error naming the missing frames there,
  never a late FileNotFoundError inside BODY assembly.
- Warm-dir masking: cached JPEGs must be validated against the CURRENT schedule set (your finding
  that _stage_frames skips on any cached JPEG without schedule-set validation) — reused dirs with a
  stale/incomplete frame set must re-materialize or fail typed, not silently pass.
- Wolverine (244-frame) behavior must remain BIT-IDENTICAL: same frames selected, same schedule
  artifacts. Prove it (schedule JSON diff pre/post-fix).
COMPLETE THE ORIGINAL A3/A4/A5:
- A3: your exact repro command completes BODY input assembly with zero missing frames; print
  schedule-vs-materialized equality for zwCtH repro AND wolverine.
- A4: regression test red at pre-fix code / green post-fix, both logs banked in the lane dir.
- A5: wide suite (MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q) — 0 new failures.
BEST-STACK DELTA: if the fix changes which frames BODY runs on long cold clips (it likely adds the
previously-missing 18), that is a SEMANTICS change — describe it precisely for manager ruling; do
not bump any manifest revision yourself.
REPORT: you cannot use --output-schema on resume — SELF-WRITE runs/lanes/ns016_bodyframes_20260710/report_r2.json
with the same schema shape as your r1 report (objective_result, acceptance A1-A5, changes, full_suite,
honest_issues, next). Do not idle-wait on anything; end only with the final report or a hard blocker.
