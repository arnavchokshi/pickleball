# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| _(none — wave 2 closed 2026-07-07; wave-3 queue in BUILD_CHECKLIST [WAVE-2 COMPLETE] bullet)_ | | | | | | | |
