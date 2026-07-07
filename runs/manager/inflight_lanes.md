# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| ios_ui_dinkvision_20260707 | codex (RUNNING) | wave-2 manager session | spec ready at runs/lanes/ios_ui_dinkvision_20260707/spec.md | ios/** ONLY (same cross-manager fence; sequenced AFTER p010a so never two lanes in ios/ at once) | — | 2026-07-07 evening | queued 2026-07-07 |
