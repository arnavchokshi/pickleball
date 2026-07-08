# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| w5_ballretrain_20260707 | sonnet-gpu | agent (see session notes) | SendMessage nudge if passive-wait | NO repo files (lane-dir glue only; VM pickleball-h100-w5ball lifecycle) | pickleball-h100-w5ball (H100 spot, self-provision) | ~2-5h incl transfers | 2026-07-07T23:5x |
| w5_p22wiring_20260708 | codex | bg (this session) | resume: log banner session id in its log.txt | NEW scripts/racketsport/eval_latent_smoothing.py + tests | none | ~1-3h | 2026-07-08 |
| w5_fastbody_bench_20260708 | sonnet-gpu | agent (this session) | SendMessage nudge if passive-wait | NO repo files (VM lifecycle only; HARD $15 cap) | pickleball-h100-w5fastbody | ~2-4h | 2026-07-08 |
| _(none — WAVE 4 CLOSED 2026-07-08: all 7 queue items ruled, ~14 landings pushed, decisive fresh-GPU proof GREEN 4/4, all wave-4 VMs DELETED list-confirmed, fleet1 STOPPED disk-intact. Scorecard = BUILD_CHECKLIST [WAVE-4 COMPLETE] bullet e26e435da; wave-5 marching order = runs/manager/wave5_boot_prompt.md; full lane-by-lane audit trail in git history of this file @ worktree-wave4-manager branch. OWNER LADDER standing: captures ~Jul 9 (W4-E→W5); ball-labeling session on the 12,075-row disagreement queue; court-kp relabels HyUqT7zFiwk+zwCtH_i1_S4.)_ | | | | | | | |

MERGE NOTE (2026-07-08, wave-4 close): main's copy of this board listed 4 rows from the concurrent
live-tier/succession session (p63_reference_ranges, live_offline_docs, live_tier_blueprint,
runbook_doctor). All four LANDED on main before this merge (evidence: fb987892f "canonical
live-vs-offline tier split + root doc cleanup + operator doctor CLI" and its siblings; doctor.py's
missing scaffold registration was repaired cross-lane by wave-4's 1b335bba0). Rows cleared as
landed, not lost — if that session believes anything is still running, it should re-add its row.
