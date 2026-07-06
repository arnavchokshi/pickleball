# P0-6 Fresh Worlds — attempt 1 report (2026-07-06, persisted by manager; agent write was guard-blocked)

RESULT: PARTIAL. All 4 clips fresh E2E through every CPU stage; BODY failed identically 4/4 —
ROOT-CAUSED to local Apple openrsync vs GNU-rsync incompat at >~100-244 files (bisected standalone).
FIX APPLIED BY MANAGER: brew GNU rsync 3.4.4 (/opt/homebrew/bin, PATH-first), verified on the exact
245-file wolverine payload. Retry lane dispatched same day.

Per-clip (attempt 1): wolverine 342.8s / burlington 666.4s / img1605 211.9s / outdoor 802.0s — all
status=partial, BODY degraded (rsync), CAL metric15_unverified warn/med, TRK do_not_promote,
ball_aware triggers events/prox/swing/uniform: 0/29/0/71, 0/75/0/25, 0/0/0/0, 0/100/0/0.

KEY FINDINGS:
1. rsync transport bug (fixed as above); tar-batch hardening in remote_body_dispatch.py booked wave-2 (also P5-1 adjacent).
2. NEW systematic bug: placement stage TypeError (int() arg NoneType) at process_video.py:1127 _run_placement_stage/rewrite_tracks_with_placement — 4/4 clips, caught non-fatal. Needs a fix lane.
3. events_selected.json wiring gap NO LONGER reproduces — all 4 runs write it (writer=run_default_ball_arc_chain); entries are rally-endpoint markers (events trigger 0 = detection-confidence outcome, not missing file). Roadmap P0-2/P0-6 notes reconciled.
4. SAM-3D-Body confirmed live backbone; NO silent fallback (each failure logged "no fallback pose skeleton" — honesty system worked). SAT-HMR never imported by pipeline.
5. Reflection factor plumbing verified dormant-with-exact-expected-reason (manual run vs historical skeleton: reflection_enabled=true, no usable contacts). paddle_pose_fused confirmed NOT wired into process_video (P3-1 as planned).
6. img1605 scheduled ZERO mesh frames (likely floor-only calibration) — follow-up.
7. VM-side checkpoint symlink fix (MANIFEST local_path vs coldstart layout); local yolo26m.pt fetched + sha256-verified vs MANIFEST.
8. Declined unverifiable gdown WASB fetch (classifier) — used explicit --ball-track reuse from ball_p4_render lane, self-documented per run.

Attempt-1 preflight also pinned the new VM host key (configs/ssh/a100_known_hosts) and pointed dispatch at 34.143.175.207 via --remote-* flags (RemoteConfig defaults still reference the OLD host — config-default update booked).
