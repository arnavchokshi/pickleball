# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| wave2_freshworlds_20260707 | sonnet-agent | manager-session agent handle | SendMessage from the dispatching manager session | NO repo edits; runs/lanes/wave2_freshworlds_20260707/** + fresh run dirs; fleet1 BODY dispatches | pickleball-a100-fleet1 (busy) | 2026-07-07 +1-1.5h | 2026-07-07 |
| p01b_prelabel_20260707 | sonnet-agent | manager-session agent handle | SendMessage from the dispatching manager session | NO repo edits; data/online_harvest_20260706/prelabels/** + runs/lanes/p01b_prelabel_20260707/**; fleet2 lifecycle | pickleball-a100-fleet2 (creating) | 2026-07-07 +2h | 2026-07-07 |
