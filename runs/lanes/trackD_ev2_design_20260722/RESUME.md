# D1 resume card (written 2026-07-22 in response to Track A sweep warning)

- Dispatch mode: nohup-detached + disown at dispatch (ppid 1 confirmed at ~64 min elapsed).
  NOT a harness background task; not exposed to the bg-task sweep class that killed Track A's lanes.
- codex session id: 019f8b47-4a37-78c1-8bf9-c4be09c73f57 (pids 13793/13794)
- If it dies anyway (power/OOM/model-capacity): resume with flags BEFORE `resume`:
  codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write \
    -c model="gpt-5.6-sol" -c model_reasoning_effort=ultra -c tools.web_search=true \
    --output-schema /Users/arnavchokshi/Desktop/pickleball/docs/racketsport/lane_report.schema.json \
    -o /Users/arnavchokshi/Desktop/pickleball/runs/lanes/trackD_ev2_design_20260722/report.json \
    resume 019f8b47-4a37-78c1-8bf9-c4be09c73f57
  (nohup-detach it; brief: git-diff-reorient against partial worktree edits and CONTINUE, not restart.)
- Spec unchanged at runs/lanes/trackD_ev2_design_20260722/spec.md; partial edits live in the
  working tree (event_head/datasets.py + new assignment module + tests observed mid-flight).
