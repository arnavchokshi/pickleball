# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| calv1_distortion_20260708 | codex | court session (bg job c922aa13) | read runs/lanes/calv1_distortion_20260708/{log.txt,report.json} | threed/racketsport/court_calibration.py, threed/racketsport/owner_capture_intake.py, scripts/racketsport/calibrate_charuco_device.py (NEW), tests/racketsport/test_court_calibration_distortion.py (NEW), tests/racketsport/test_calibrate_charuco_device.py (NEW) | none | ~1-2h | 2026-07-08 |
| calv1_profiles_20260708 | codex | court session (bg job c922aa13) | read runs/lanes/calv1_profiles_20260708/{log.txt,report.json} | threed/racketsport/profile_registry.py, threed/racketsport/court_profile_match.py (NEW), tests/racketsport/test_court_profile_match.py (NEW) | none | ~1-2h | 2026-07-08 |
| calv1_net_20260708 | codex | court session (bg job c922aa13) | read runs/lanes/calv1_net_20260708/{log.txt,report.json} | threed/racketsport/net_plane.py, threed/racketsport/external_gt_precomputed_calibration_runner.py, threed/racketsport/court_auto_evidence.py, threed/racketsport/calibration_overlay.py, threed/racketsport/court_corner_review.py, scripts/racketsport/calibrate.py, tests/racketsport/test_net_plane_overrides.py (NEW) | none | ~30-60min | 2026-07-08 |
| w5_critiqueviewer_20260708 | sonnet-local | agent (this session) | SendMessage if stalled | worktree /tmp/critique_viewer_worktree + lane dir only (local viewer for owner critique; no repo edits) | none | ~minutes; leaves dev server running detached | 2026-07-08 |
| w6_labelpack_20260708 | codex | 019f42fb-ac15-7063-8c72-2d73162c34f0 | codex exec resume (cd ROOT first) | cvat_upload/w6_labelpack_20260708/** + lane dir + OWNER_SESSION_W6 doc | none | ~1-2h | 2026-07-08 11:26 |
| w6_gate1b_knob_20260708 | codex | 019f42fb-b28e-7822-b989-2911db21a8b6 | codex exec resume (cd ROOT first) | process_video.py, remote_body_dispatch.py, BODY post-chain modules + tests | none | ~2-4h | 2026-07-08 11:26 |
| w6_magnus_20260708 | codex | 019f42fb-b717-7f33-a37b-b0f04c6eb0ec | codex exec resume (cd ROOT first) | threed/racketsport/ball_arc_solver.py + tests | none | ~2-4h | 2026-07-08 11:26 |
| w6_instrudocs_20260708 | codex | 019f42fb-cab3-76c2-b055-bb73e2701a5b | codex exec resume (cd ROOT first) | CAPABILITIES.md, BVP verify harness, train_ball_stage2.py + tests | none | ~1-3h | 2026-07-08 11:26 |
| w6_playbackdiag_20260708 | codex (read-only) | 019f42fb-d0f8-7200-bd0f-92ec5b676ff8 | codex exec resume (cd ROOT first) | NOTHING (read-only; lane dir only) | none | ~1h | 2026-07-08 11:26 |
| w6_labelingest_20260708 | codex (STAGED, NOT dispatched) | — | dispatch spec when owner export lands (watchdog class G) | reviewed-corpus build outputs, LoSO fold config, converter + tests | none | staged | — |
| _(none — WAVE 4 CLOSED 2026-07-08: all 7 queue items ruled, ~14 landings pushed, decisive fresh-GPU proof GREEN 4/4, all wave-4 VMs DELETED list-confirmed, fleet1 STOPPED disk-intact. Scorecard = BUILD_CHECKLIST [WAVE-4 COMPLETE] bullet e26e435da; wave-5 marching order = runs/manager/wave5_boot_prompt.md; full lane-by-lane audit trail in git history of this file @ worktree-wave4-manager branch. OWNER LADDER standing: captures ~Jul 9 (W4-E→W5); ball-labeling session on the 12,075-row disagreement queue; court-kp relabels HyUqT7zFiwk+zwCtH_i1_S4.)_ | | | | | | | |

MERGE NOTE (2026-07-08, wave-4 close): main's copy of this board listed 4 rows from the concurrent
live-tier/succession session (p63_reference_ranges, live_offline_docs, live_tier_blueprint,
runbook_doctor). All four LANDED on main before this merge (evidence: fb987892f "canonical
live-vs-offline tier split + root doc cleanup + operator doctor CLI" and its siblings; doctor.py's
missing scaffold registration was repaired cross-lane by wave-4's 1b335bba0). Rows cleared as
landed, not lost — if that session believes anything is still running, it should re-add its row.
