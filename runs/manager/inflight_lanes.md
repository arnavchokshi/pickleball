# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.
Closed-lane rows + dated session notes through 2026-07-12 are preserved verbatim in
`runs/manager/archive/inflight_history_20260709_20260712.md`.

Standing fence: `brand-exploration/` is the OWNER'S untracked brand work — no lane may touch it.
`cvat_upload/court_diversity_20260712/` + `w7_audit_stratum_20260709/` are staged local-only owner
labeling packages (storage-allowlisted, intentionally untracked).

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| pbv11_headtohead_20260713 | Sonnet GPU ops lane: MOVE 1 baseline head-to-head (our stack vs pb.vision on the 697s demo video) | (this session) | STOPPED at global provision gate, pre-VM — nothing to resume except retrying the gate | none (no VM touched, no repo source edited) | none created | needs one owner `gcloud auth login` for hello@swayformations.com, then re-dispatch from step 1 of runs/lanes/pbv11_headtohead_20260713/spec.md | 2026-07-13 (this session) |
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

_(2026-07-13 ~01:2x: SYNTHESIS COMMITTED 541f89d9a (dual-model, zero material disagreements; their
global-track pipeline decoded; 3-move program supersedes the 07-12 reproduction map; kills stand).
DISPATCHED: [MOVE-1] pbv11_headtohead_20260713 Sonnet GPU lane (H100, full promoted stack on the
697s demo video, per-rally pbv scorecard, owner union event set; wall cap 5h); [MOVE-2]
ball_evidence_q_20260713 codex xhigh (audio order/timing on normal path + below-threshold WASB
candidate persistence + blur-aware proposals; >=0.90 gate PENDING owner union review — no reviewed-
set claims until then); [MOVE-3] ball_globaltrack_20260713 codex xhigh (isolated whole-rally robust
ballistic track candidate w/ membership over ALL candidates, posterior, typed exceptions, radius
residual conf-gated; pre-registered kills incl. physics-100%-emitted + fallback-below-baseline).
BL-E ruled: killed as-built (2/15->0/15 vs reviewed) w/ scope limits (no audio on cards; >=0.5-only
sidecars) — module committed 08bf09216 for reuse. Monitor re-armed date-agnostic (prior one was
*_20260712-pinned — cost ~40min idle on forensics landing; lesson booked).)_

_(2026-07-13, pbv11_headtohead_20260713 Sonnet GPU-ops session: STOP at the mandatory global provision gate before any VM was created. `gcloud compute instances list` and `gcloud auth application-default print-access-token` both failed with 'Reauthentication failed. cannot prompt during non-interactive execution' for hello@swayformations.com (correct fleet account + project gifted-electron-498923-h1 per `gcloud config list`). Checked the alternate credentialed account (swayformations@gmail.com) as a sanity check — its token is live but it lacks compute.instances.list permission on the fleet project, so it is not a usable substitute. No SA key file exists in-repo (consistent with 'SA key creation org-blocked'). Net effect: gpu_fleet.md's 'EMPTY, zero running VMs' claim is now UNVERIFIED, not freshly confirmed — flagged inline there too. Zero cost, zero VMs, zero repo-source edits, zero commits this session. Committed pin recorded for whenever this resumes: HEAD SHA 541f89d9a160eca8498a7b7419a7c2bc7f5b4a0e (the pbv11 synthesis commit). Full evidence: runs/lanes/pbv11_headtohead_20260713/report.json. NEEDS: owner runs `gcloud auth login` (and ideally `gcloud auth application-default login`) once, interactively, for hello@swayformations.com; then any GPU lane — this one or another — can proceed from the provision gate.)_

_(2026-07-13 ~01:4x: MOVE-1 head-to-head lane TYPED STOP at the global provision gate — gcloud auth
DEAD (hello@ reauth interactive-only; gmail account valid but lacks compute perms; no SA fallback,
org-blocked). $0, zero VMs, pin 541f89d9a recorded for exact resume. OWNER ASK #0 staged: one
`gcloud auth login`. Fleet state UNVERIFIABLE until then (last confirmed empty at trkA teardown).
MOVE-2 + MOVE-3 codex lanes unaffected (CPU-local, running).)_

_(2026-07-13 ~02:3x: MOVE-2 RULED PASS-w/-attr + COMMITTED 03a0085ab + fixup 3b639768c (proven_against
field; 2nd piped-exit-code commit slip — chain now gates on real $?; memory strengthened). Auth
RESTORED by owner -> fleet verified EMPTY -> MOVE-1 head-to-head lane RESUMED from pin 541f89d9a
(SendMessage; H100 provisioning). DISPATCHED ball_gt_rescore_20260713 Sonnet lane (task #7 decisive
test): real WASB MPS inference w/ emit_below_threshold_candidates on both internal cards ->
real-inference byte-parity re-proof -> UNMODIFIED ball_global_track re-scored vs same kill bars +
pbv harness; 11-min card scored too if MOVE-1 lands in time. Live: MOVE-1 GPU (resumed), gt_rescore
Sonnet, concurrent court session. All ball evidence infrastructure now committed.)_

_(2026-07-13 ~04:0x: OWNER RULINGS EXECUTED: [1] TRK FLIP dispatched (trk_flip_20260713 codex xhigh —
margin 1.0m + OSNet -> WIRED_DEFAULT per owner directive; anti-paper-flip gate = production-entry-point
reproduction of sweep numbers 0.8516/0.7117 or NO flip; preview band stays; rev bump). [2] Event-gap
attack: ball_hitdetect running (fair kinematics+wrist+audio test w/ miss taxonomy) + event_bootstrap
dispatched (audio-x-track two-signal auto-labels, tier-A/B, owner 5-min spot-check pack, training-lane
handoff; PB events excluded from labels). MOVE-1 head-to-head still on H100 (pickleball-h100-pbv11
RUNNING). gt_rescore verdict committed 0442c253b.)_

_(2026-07-13 ~04:4x: HITDETECT VERDICT — owner kinematics hypothesis REJECT on internal cards w/
decisive taxonomy: top-5 misses 13/15 reviewed hits; 12/13 misses = candidates_too_noisy_for_corner_fit
at contact (same failure mode as global-track membership rejections — consistent), 1 wrist-absent.
Audio arm UNTESTABLE on cards (no audio) — the kinematics+audio fair test happens on the 11-min clip
(MOVE-1 artifacts + owner union review). Confirms the two live paths: audio-first (product captures
have it) + trained event heads (event_bootstrap lane manufacturing tier-A labels now). Lane was codex
= sandbox no-MPS (11-min local inference skipped; MOVE-1 supplies it). No repo changes (lane-dir
experiment). Live: trk_flip, event_bootstrap, MOVE-1 H100.)_

_(2026-07-13 ~05:2x: EVENT-DATA RESEARCH LANDED — owner hypothesis VINDICATED (32 agents, 2-vote
primary-source refutation): jhong93/spot tennis = rank-1 (33,791 frame-precise HIT+BOUNDCE events,
BSD-3 labels + E2E-Spot reference code, live-verified), OpenTTGames + Extended OpenTT CORROBORATED
x2, ShuttleSet MIT ~70k hits, PadelTracker100 CC-BY-4.0 domain cousin; P2ANet/TTStroke-21 rejected
w/ decisive reasons. Committed 378e0ec84. DISPATCHED eventdata_acquire Sonnet lane (Stage-0: labels
+ pilot videos, 25GB cap, license ledger, 2 semantics checks). sol cross-survey still running.
Owner asks +1 (CoachAI form, BFMD email). LIVE: MOVE-1 H100 head-to-head, trk_flip, event_bootstrap,
eventdata_sol, eventdata_acquire.)_

_(2026-07-13 ~2x:xx: EVENT_BOOTSTRAP RULED PASS — tier-A untyped contact windows built w/ full
provenance; honest weaknesses recorded (0.274 chance-excess proxy, 15.4% strict-survival, circularity
warnings -> visual temporal head is the right student, source imbalance); TRAINING SPEND BLOCKED on
owner 50-row spot-check (staged, added to checkin as ask #4). eventdata_acquire RULED PASS earlier
(corrections committed b8a87dbdf; ~130k events on disk). Storage audit: 0 unknown. Task #8 remaining:
owner review -> event-head training lane design (visual temporal head, public pretrain + pickleball
fine-tune). Still live: trk_flip wide census, MOVE-1 H100 head-to-head.)_

_(2026-07-13: TRK FLIP COMMITTED — margin 1.0 + OSNet default (rev 12), production reproduction
exact, preview-band + do_not_promote honesty intact. The owner's #1 visual symptom fix is live in
the default stack. Remaining in flight: MOVE-1 H100 head-to-head (last big deliverable of the pbv11
program). Task #8 waits on owner 50-row spot-check.)_

_(2026-07-14 SESSION CLOSE (Opus, after Fable-5 spend limit hit late 07-13): HANDOFF authored
runs/HANDOFF_20260714.md (full ball/pbv11/event-data detail + exact next steps). North Star updated
to 07-14 truth (497 lines, 14/14 doc tests): TRK row = WIRED_DEFAULT preview flip; BALL/EVENTS rows =
trained-contact-detection diagnosis + ~130k public labels; dated pointers folded in; owner asks +
spot-check; parallel event-head queue row added. OWNER_CHECKIN headline refreshed. FLEET: orphaned
pickleball-h100-pbv11 (head-to-head died mid-run on Fable spend limit, no scorecard) DELETED + disk-
confirmed; fleet EMPTY. OPEN THREADS for next session: (1) RE-RUN 41-rally head-to-head (Sonnet GPU,
pin 541f89d9a, video local); (2) owner 5-min spot-check -> event-head training; (3) NS-01 core wiring.
All work committed+pushed.)_
