# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| ios_p010a_20260707 | codex | bash (wave-2 manager session, still open for owner support) | `codex exec resume <session_id from report.json>` | ios/** ONLY + runs/lanes/ios_p010a_20260707/** — CROSS-MANAGER FENCE: wave-3 session owns all racketsport files; this lane touches nothing outside ios/ | — | 2026-07-07 +2-3h | 2026-07-07 |
