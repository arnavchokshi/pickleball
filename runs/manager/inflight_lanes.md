# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| rootjump_slide_fix_20260706 | codex | bash bgc4q4sfc | `codex exec resume <session_id from report.json>` | placement.py, offline_person_authority.py, strict_placement_rollup.py, pose_temporal.py, foot_pin.py, player_court_membership.py, worldhmr.py + scripts apply_placement/refine_body_grounding/run_physics_footlock/process_video/global_associate_person_tracks + their tests | — | 2026-07-06 late | 2026-07-06 |
| p01b_harvest_ingest_20260706 | codex | bash b9dz3zw51 | same mechanism | NEW ingest_online_harvest.py + tests + data/online_harvest_20260706 outputs | — | 2026-07-06 late | 2026-07-06 |
| p08_vfr_pts_20260706 | codex | bash bz4imr6iw | same mechanism | io_decode.py, rally_gating.py, non-fenced ball-timing consumers (process_video via deferred patches) | — | 2026-07-06 late | 2026-07-06 |
| p21_cammotion_20260706 | codex | bash bq0mf3bqx | same mechanism | camera_motion.py, estimate_camera_motion.py, court_motion_mode.py + tests (wiring = deferred patches) | — | 2026-07-06 late | 2026-07-06 |
| p11_visibility_schema_20260706 | codex | bash bo2gksqpf | same mechanism | schemas/__init__.py (visibility only) + CVAT scripts + tests | — | 2026-07-06 late | 2026-07-06 |
| dispatch_hardening_20260706 | codex | bash bxrhgw63c | same mechanism | remote_body_dispatch.py + test_remote_body_dispatch.py | — | 2026-07-06 late | 2026-07-06 |
| p27a_gvhmr_spike_20260706 | sonnet-agent | manager-session agent handle | SendMessage from the dispatching manager session | NO repo files; runs/lanes/p27a_gvhmr_spike_20260706/ + VM ~/gvhmr_spike only | pickleball-a100-fleet1 (restarting; new IP) | 2026-07-06 +3-5h | 2026-07-06 |
