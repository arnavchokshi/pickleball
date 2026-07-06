# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| online-harvest-batch1 | sonnet agent | task ab9a841aac7a16d5c | SendMessage resume w/ 'continue: bounded loops, finish report' | data/online_harvest_20260706/ only | none (local) | ~30-60min | 2026-07-06 |
| tech-audit-7pillar | workflow | task w70uo7e8m / wf_d2001088-d74 | Workflow resumeFromRunId wf_d2001088-d74 | none (read-only + reports) | none | ~20-30min | 2026-07-06 |
