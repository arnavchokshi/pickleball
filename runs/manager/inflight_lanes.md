# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.
Closed-lane rows + dated session notes through 2026-07-12 are preserved verbatim in
`runs/manager/archive/inflight_history_20260709_20260712.md`.

Standing fence: `brand-exploration/` is the OWNER'S untracked brand work — no lane may touch it.
`cvat_upload/court_diversity_20260712/` + `w7_audit_stratum_20260709/` are staged local-only owner
labeling packages (storage-allowlisted, intentionally untracked).

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| ball_anchor_boost_20260712 | Codex xhigh BL-E (last live sprint lane): audio/kinematic/blur/court-proximity anchor-evidence fusion scored vs frozen reviewed event timing (attacks the convergent ball-lift bottleneck; pb.vision reference-only) | sprint bg c7d8cfb2 | codex exec resume (session id in runs/lanes/ball_anchor_boost_20260712/log2.txt) | ball anchor/event evidence modules + tests + runs/lanes/ball_anchor_boost_20260712/** | — | overnight 07-13; verdict + BEST-STACK DELTA in lane REPORT | 2026-07-12 ~18:0x |
| docreview_20260713 | Codex gpt-5.6-sol xhigh READ-ONLY doc/organization/next-steps review (owner-directed) | Fable bg a11f030d | codex exec resume (log in lane dir) | runs/lanes/docreview_20260713/** only | — | ~1h | 2026-07-13 ~00:1x |

_(2026-07-13 ~00:4x, Fable bg a11f030d DOC/ORG session: [1] adopted stranded coords_remainder2
schemas hunk — HEAD referenced coordinate_contract (metric15 emission + 2 committed tests) without
the schema definition, fresh clone was broken; additive, 41/41 schema tests green. [2] adopted the
sprint session's uncommitted close notes (world-perf 122s attribution etc.) + bodyc keyscan lines
into git. [3] archived ledger history: this file + gpu_fleet.md slimmed to live-only; verbatim
history under runs/manager/archive/. [4] OWNER_CHECKIN.md rewritten to the owner's new standing
format: very brief asks + best-results (accuracy+speed) table per capability — see memory
pickleball-owner-checkin-format. [5] North Star Section 2/5 refreshed to 2026-07-13 state.
Fences honored: BL-E lane files untouched; owner dirs untouched.)_
