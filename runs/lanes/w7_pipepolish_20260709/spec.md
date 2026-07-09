# LANE w7_pipepolish_20260709 — three pipeline items on the settled tree (P5-5b + postchain metric + stats wiring)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg; PYTHONDONTWRITEBYTECODE=1. Protected clips rules stand. You now OWN scripts/racketsport/process_video.py + configs/racketsport/best_stack.json (all prior lanes landed; main is at ecb086b48 — start from a git pull/status re-orient). Do NOT touch: threed/racketsport/mhr_decode*.py, hmr_deep.py, gate_check_body_decode.py, web/**, server/**. Artifacts under runs/lanes/w7_pipepolish_20260709/ only. Disk: run df -h before wide suites; keep outputs lean.

## ITEM 1 — P5-5b INPUT-QUALITY GUARDRAIL (first-class pre-launch gate; refresh R6.4)
A pre-flight stage in process_video that scores input suitability BEFORE heavy compute and emits an input_quality block (PIPELINE_SUMMARY + its own artifact): resolution/fps floor, duration sanity, court-visibility/angle heuristic, blur/exposure gross check. BINDING OWNER POLICY DATAPOINT (read cvat_upload/exports/court_keypoints_20260707/PARTIAL_EXPORT_NOTES_20260709.md): courts mostly not fully visible / camera angle too low = BELOW the acceptance bar for user-submitted clips — encode this class as the first named rejection reason (court_not_fully_visible_low_angle) using whatever cheap signal the existing court-calibration/court-map path already produces (reuse, don't invent a new detector). Behavior: default = ADVISORY loud band (process continues, output banded degraded_input) + a strict mode knob that fail-closes; BOTH knobs route through best_stack.json (new input_quality entry, WIRED_DEFAULT advisory). Cadence/thresholds configurable, never hardcoded (speed-cadence doctrine).
## ITEM 2 — postchain_bypassed_stages metric (boot-queue #10 leftover)
pipeline_summary stages[body].metrics.postchain_bypassed_stages is empty on runs where the BODY postchain bypass engaged; populate it from the existing bypass provenance (the w6 raw-knob machinery already records the 6 stages — thread it into the summary). Test with a bypass fixture.
## ITEM 3 — match-stats integration (P6-2 follow-through)
Apply the PROPOSED integration diff from runs/lanes/w7_p62stats_20260709/report.json (re-derive, git apply --check clean, adapt to current tree): wire compute-match-stats as an optional post-stage emitting match_stats.json by default when its inputs exist (fail-open skip w/ loud summary reason when absent — stats are a consumer, never a blocker). best_stack entry (stats.match_stats_v0 WIRED_DEFAULT, opt-out flag). BODY+COURT-only stands.

## SELF-VERIFICATION
Acceptance THROUGH scripts/racketsport/process_video.py (mocked pipeline entry tests per house pattern). Full blast radius: tests/racketsport/test_process_video*.py, test_best_stack_*.py, test_match_stats.py, your new tests + ONE full tests/racketsport census (no fail-fast) at the end — target zero non-preexisting failures. Fix what you introduce.

## REPORT
Self-write runs/lanes/w7_pipepolish_20260709/report.json (lane_report.schema.json structure): acceptance per item (exact summary/metric keys), BEST-STACK DELTA (expected: 2 new WIRED_DEFAULT entries + revision bump — state precisely), full census numbers, honest_issues, next.
