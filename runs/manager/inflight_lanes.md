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

_(2026-07-13 ~00:4x, Fable bg a11f030d DOC/ORG session: [1] adopted stranded coords_remainder2
schemas hunk — HEAD referenced coordinate_contract (metric15 emission + 2 committed tests) without
the schema definition, fresh clone was broken; additive, 41/41 schema tests green. [2] adopted the
sprint session's uncommitted close notes (world-perf 122s attribution etc.) + bodyc keyscan lines
into git. [3] archived ledger history: this file + gpu_fleet.md slimmed to live-only; verbatim
history under runs/manager/archive/. [4] OWNER_CHECKIN.md rewritten to the owner's new standing
format: very brief asks + best-results (accuracy+speed) table per capability — see memory
pickleball-owner-checkin-format. [5] North Star Section 2/5 refreshed to 2026-07-13 state.
Fences honored: BL-E lane files untouched; owner dirs untouched.)_

_(2026-07-13 ~00:4x: SOURCE VIDEO OBTAINED — data/pbvision_11min_20260713/source_video.mp4 (114MiB,
697.4s, 1280x720@30 h264 + AAC audio, sha 272a2132, zero decode errors; world-readable GCS object,
no auth). PROVENANCE NUANCE: it is pb.vision's OWN demo video (uploader admin-ryan, 'Demo Vid',
uploaded 2024-12-11), NOT owner footage -> posture = R&D reference benchmark ONLY (never training/GT,
never redistributed; same competitor-reference rules as the export). video_provenance.json has full
chain. HEAD-TO-HEAD QUEUED: after forensics+workflow synthesis, one H100 lane runs OUR stack
(baseline + surviving candidate flags) on the same 697s -> rally-by-rally compare_vs_pbvision at
scale (41 rallies). Audio present -> BL-E anchor fusion gets a scale test bed too.)_

_(2026-07-13 ~01:2x, Fable bg a11f030d DOC/ORG session CLOSE: docreview_20260713 (sol xhigh, read-
only) DONE — 45-finding currency audit + ranked program + verified best-results table at
runs/lanes/docreview_20260713/REPORT.md. docfix_20260713 (sol xhigh) DONE PASS manager-re-verified
(16/16 truthful+manifest tests): RUNBOOK NS-01.3/calibration-precedence/BODY-naming/stats/exit-0
corrections, BALL_TRACKING artifact contracts + WIRED_DEFAULT status, README P0 summary, best_stack
updated-date + OSNet staging note (no revision bump), MANIFEST OSNet license posture split.
BOOKED FOLLOW-UP for the next spine/integration owner: test_truthful_capabilities expected_order
pins the obsolete `manifest -> match_stats` tail — runner now emits stats/facts BEFORE manifest;
fix test + RUNBOOK numbered stage block together (docfix honestly skipped it as out-of-fence).
NEXT QUEUE (North Star Section 5, refreshed): 1 NS-01.4/01.5 adopt landed coordinate/timebase cores
across real stage consumers + finish status/packaging; 2 NS-01.6/01.7 explicit timed refined-event
stages (~122s now hidden in world); 3 NS-01.2b physical trace after 1-2; owner-gated: labels
(court diversity pack + tasks 88-91 + ball 87) then gold capture; after fresh labels, score the
TRK margin-1.0 candidate ONCE against the frozen full bar (no new association sweeps). BL-E
ball_anchor_boost remains the sprint session's to rule (interim table trends honest-kill).)_
