# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.
Closed-lane rows + dated session notes through 2026-07-12 are preserved verbatim in
`runs/manager/archive/inflight_history_20260709_20260712.md`.

Standing fence: `brand-exploration/` is the OWNER'S untracked brand work — no lane may touch it.
`cvat_upload/court_diversity_20260712/` + `w7_audit_stratum_20260709/` are staged local-only owner
labeling packages (storage-allowlisted, intentionally untracked).

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| ballarc_scale_guard_20260715 | Codex gpt-5.6-sol high: ball_arc per-segment wall-clock guard (loud typed timeout per trust contract) + segment-7 pool-explosion diagnosis + regression test from pulled real artifacts; CPU-only proof on the salvaged 697s inputs | Track A manager; codex session 019f68e2-4784-7463-af04-ccaa74c5ab09 (died overnight on model capacity + Mac sleep at ~85% done, RESUMED 2026-07-16 ~00:15 PDT as detached nohup, log_resume.txt) | report at runs/lanes/ballarc_scale_guard_20260715/report.json when done; if it dies again: codex exec [flags] resume 019f68e2-4784… with a state brief | threed/racketsport/ball_arc_solver.py + ball_arc_chain.py + its tests + lane dir (fence excludes process_video.py, ball_physics3d.py, timebase files) | none (CPU local) | hours; manager rules on report | 2026-07-16 (coordinator GO, order 1) |
| ~~pbv_harness_v2_20260715~~ | RULED **ADOPT (scoped pass)** 2026-07-16 by Track A manager, manager-verified with real exit codes: frozen original byte-identical to HEAD (md5 4ebd6c53 both sides), regression A all 3 cards BYTE_IDENTICAL to frozen scorecards, manager's independent 3rd full-scale run EXIT 0 md5-identical (59e03035), tests 4/4 EXIT 0. Root cause: PB segment 92 vz 271.9m/s outside ±60m/s bounds → typed fail-closed `physics_fit_skipped` (1/490 segments), no clamp, no silent drop. Full-11-min scoring GREEN → MOVE-1 prerequisite 2/3 met. | bs7v1lnvu CLOSED | — | runs/lanes/pbv_harness_v2_20260715/** | — | DONE | 2026-07-16 |

_(2026-07-16 Track A manager: owner CAL-seed ask STAGED at
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/ (2-min tap flow: tap_corners.html
+ chosen frame t=10s + OWNER_CAL_SEED_ASK.md). ADDED to OWNER_CHECKIN.md as item 0 on 2026-07-16
after Track C's window-close landed (coordinator go); OWNER_CHECKIN "Running right now"/"Money"
also refreshed to 07-16 truth. HARD RULE standing: NO third MOVE-1 GPU attempt
without the coordinator's explicit go; prerequisites = ballarc guard adopted + harness v2 green +
trusted CAL seed.)_

_(2026-07-16 Track A manager: OWNER CAL SEED BANKED — owner completed the 4-corner tap; manager
validated (bounds/ordering/convexity PASS + homography overlay lands every court-model line on the
painted lines, proof owner_seed_overlay_check.jpg) and banked verbatim + runner-shaped seed at
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_corners_seed.json
(corrected_unverified provenance — enables metric output, NOT a CAL promotion). MOVE-1 #3
prerequisites now: guard-lane ruling ONLY (harness v2 ADOPTED, CAL seed BANKED). Owner recorded no
game footage today — demo-video track remains the active ball workstream.)_

_(2026-07-15/16 Track A manager CLOSE, pbv11_headtohead RE-RUN: **partial — 3D head-to-head NO-RESULT
again**, this time with the blocker precisely located: `ball_arc` segment-association stall at
full-game scale (py-spy x3, segment 7; see runs/lanes/pbv11_headtohead_20260713/rerun_20260715/
pyspy_stall_evidence.md). VM pickleball-h100-pbv11r torn down 02:50:53Z, list+disks confirmed, wall
3.93h ≈ $9-15. Salvage: full-697s 2D ball chain artifacts pulled w/ two-sided md5 + a 2D-coverage
scorecard (ours 78.4% vs pb 75.6% in-window presence — detection-vs-emission caveat applies;
scorecard_2d_salvage.json). Owner event pack NO-RESULT (events stage never reached; refusing to
stage a degraded pack that would burn the owner's review time on unfused raw candidates).
Corrections appended (HANDOFF addendum + fleet ledger): 07-13 attempt likely died of the SAME stall,
not primarily the spend limit. Follow-up fix lane SPEC-ONLY staged (ballarc_scale_guard_20260715 —
DO NOT DISPATCH before coordinator sequences vs Track C coordwire/tbwire fences). Also for the next
full-3D attempt: this video needs a trusted CAL seed (auto-preview graded POOR, metric world
fail-closed) and compare_vs_pbvision crashes on the full 11-min export (PB physics pillar) — both
are prerequisites, both in the follow-up spec. Full record:
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/MANAGER_REPORT.md.)_
| ball_anchor_boost_20260712 | Codex xhigh BL-E (last live sprint lane): audio/kinematic/blur/court-proximity anchor-evidence fusion scored vs frozen reviewed event timing (attacks the convergent ball-lift bottleneck; pb.vision reference-only) | sprint bg c7d8cfb2 | codex exec resume (session id in runs/lanes/ball_anchor_boost_20260712/log2.txt) | ball anchor/event evidence modules + tests + runs/lanes/ball_anchor_boost_20260712/** | — | overnight 07-13; verdict + BEST-STACK DELTA in lane REPORT | 2026-07-12 ~18:0x |
| ~~tbwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED bd99c6d11 (2026-07-15, Track C): typed timebase contract wired through ingest/frames/events decode seams, canonical-beside-legacy (Wolverine 300/300 typed vs legacy 299 explicit, legacy values byte-identical); manager re-verified focused 216 EXIT 0, sandbox-bind claims 57 EXIT 0 locally; PENDING: physical 30s/5min (owner), intrinsics/rolling-shutter slices, independent labels | — | — | committed set in bd99c6d11 | — | done (survived external 1h task-cap kill via detached resume) | 2026-07-15 |
| ~~tbcam_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 1685a8878 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, P0-H remainder): typed intrinsics transforms (scale/rotate/crop) in coordinates.py + route the two ad-hoc scalers parity-first; additive optional CaptureSidecar reference_crop + rolling_shutter fields (goldens stay valid, Swift emission PENDING); orientation-mismatch fails loudly at the calibration seam; io_decode populates RollingShutterModel-or-explicit-missing (kills the hardcoded None at :495) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/tbcam_20260716/log.txt); nohup-detached | threed/racketsport/{schemas/__init__,coordinates,court_calibration,io_decode,timebase,sam3d_body_input_prep,court_auto_evidence}.py + docs schemas + their tests; process_video.py FORBIDDEN (deferred hunks inline) | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~evidence17_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 8a282d4db 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.7 non-ball_arc): audio soft evidence (pop_band_ratio et al) into fusion non-gating w/ documented bounded combination (no raw averaging); BOTH IPPE poses retained (alt_pose + carry-ambiguous-instead-of-drop, primary parity-pinned); repaired-confidence markers in ball_temporal_filter/player_id_repair/pose_temporal (values unchanged); contact-dependency-hashing runner hunks DEFERRED inline | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/evidence17_20260716/log.txt); nohup-detached | threed/racketsport/{event_fusion,racket6dof,racket_stage_runner,racket_pose_preview,ball_temporal_filter,player_id_repair,pose_temporal}.py + their tests; audio_onsets/paddle_pose_fused/ball_arc_*/runner FORBIDDEN | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~coordwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED aab8c3098 (2026-07-15, Track C): typed coordinate API adopted in placement/ball_court_filter/ball_physics3d/ball_inout_uncertainty/virtual_world, six SHA-pinned Wolverine digests byte-identical, distorted-synthetic + fail-closed proofs; manager re-verified 22+165+57 tests EXIT 0; its tbwire-regression isolation confirmed and fixed in c4dfb2d8b | — | — | committed set in aab8c3098 | — | done (survived 1h task-cap kill via detached resume) | 2026-07-15 |
| ios_recordpath_20260715 | Codex gpt-5.6-sol high (Track D): dead record button on owner's real iPhone 14 Pro — root-cause the silent no-op (traced primary mechanism: button `.disabled` while status pinned `.requestingAccess`, configure/ARKit/ownership chain failures swallowed by `try?`/silent guards, no timeout, no banner; five-tab sim pass ran the walker FAKE controller), land loud-state contract fixes + tests, stage signed iphoneos Debug build + exact devicectl install commands (device id B03696B6-..., currently unavailable). MANDATORY skill: ios-debugger-agent (plugin build-ios-apps@openai-curated, installed this session). Sandbox workspace-write; CoreSimulator-blocked steps go to MANAGER_VERIFY.md | codex session 019f68dd-984d-72d3-9725-80a9546355cf (PID file runs/lanes/ios_recordpath_20260715/codex.pid) | `codex exec resume 019f68dd-984d-72d3-9725-80a9546355cf` (nohup fire-and-forget) | ios/App/**, ios/Capture/**, ios/AppTests/**, runs/lanes/ios_recordpath_20260715/** (disjoint from coordwire/ball_anchor) | — | same session: report.json + guard audit + staged device build | 2026-07-15 ~20:2x |
| ~~statusdocs_20260715~~ CLOSED | RULED ADOPT + COMMITTED 9bf8eef75 (2026-07-15, Track C): stale stage-order pin fixed (RUNBOOK block + expected_order incl. coaching_facts, honest Status Interpretation split); manager re-verified truthful 14/14 exit 0 + server/render 150 passed exit 0 | — | — | RUNBOOK.md + tests/racketsport/test_truthful_capabilities.py | — | done | 2026-07-15 |
| ~~spine16_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED ffb7e0975 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.6): one authoritative stage graph (3-way assembly consolidated), REMOVE legacy pipeline_cli duplicate (readiness migration + doc/test pins), typed ExpectedOptionalAbsence + unexpected-exception-FAILS rewrite of _run_stage_safely w/ enumerated per-stage catch conversions, frame-schedule completeness (silent equal:True defaults killed, runner-side loud-path test, plan-coverage cross-check), cold/reuse/partial/failure coverage per new contract; HARD FENCE: ball_arc_* files + their two caller catches untouched (Track A live), just-landed Track C threed modules read-only | Track C manager session 2026-07-15/16 | codex exec resume (session id in runs/lanes/spine16_20260716/log.txt); dispatched nohup-detached (1h-cap immune) | scripts/racketsport/process_video.py (SOLE owner) + pipeline_cli.py (deletion candidate) + validate_pipeline_artifacts.py + pipeline_contracts.py (metadata fold only) + process_video_body_frames.py (validation default) + AGENTS/RUNBOOK pipeline_cli lines + named tests | — | same session; structured report.json + wide suite w/ real exit codes (baseline 3684/24/1-external) | 2026-07-16 |
| ~~owner_event_labels_20260715~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED d0ce58bdd (2026-07-15, Track E): scaled owner event-labeling channel — sampler/renderer/ingest CLIs + 15 tests; 300-clip session STAGED at ~/Desktop/event_labels_20260715/START_HERE.html (120 audio-onset / 75 track-discontinuity / 105 uniform-random, all 6 harvest sources, seed 20260715, 50-row eval seed +/-0.75s + pbvision + protected eval hard-excluded, page blind to stratum); manager-verified: exclusion audit 0 violations, same-seed byte-identical, 300/300 clips ffprobed w/ audio, ITEMS join 0 mismatch, node --check 0, 15 tests EXIT 0, scaffold 3/3 EXIT 0, ingest dry-run vs real manifest EXIT 0; wide suite by composition: trackC waveclose 3684p/1f where the 1f = scaffold-index (this lane, FIXED+green) + import-isolation grep; lane report.json NEVER LANDED — resumed codex proc terminated by manager at wind-down (2026-07-15 coordinator directive); ruling rests entirely on the manager verification battery; codex session 019f68df-5f28-7703-ad6e-bea1cf89e4a0 recorded for forensic resume if ever needed. FLAG: storage audit exits 1 repo-wide, PRE-EXISTING stale allowlist (cvat_upload/w5 zips deleted 07-09) — needs owner-package bookkeeping fix. OWNER NEXT: open ~/Desktop/event_labels_20260715/START_HERE.html, label 300 clips (~75-120 min), export; ingest command in runs/lanes/owner_event_labels_20260715/INGEST_README.md | — | — | committed set in d0ce58bdd; pack on Desktop (untracked) | — | done | 2026-07-15 |

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

_(2026-07-15, coordwire_20260715 CLOSE: NS-01.4/P0-D typed coordinate adoption is wired with a scoped pass across placement, ball target-court/in-out, ball arc camera projection, in/out uncertainty, and virtual-world ball lifting; canonical-beside-legacy Wolverine digests stayed byte-identical and distorted-synthetic/wrong-space coverage passed (22/22 coordinate parity, 274/274 broadened focused, both EXIT 0). Mandatory wide suite completed 3670 passed / 12 failed / 24 skipped, literal EXIT 1: 8 failures are managed-sandbox socket-bind denials and 4 ball_physics_fill failures isolate to concurrent tbwire's eager empty-frame-times fallback change (4/4 pass when the pre-tbwire fallback is restored in-memory), not coordwire math. P0-D's stage-adopted distorted-synthetic + real-iPhone slice is wired (scoped pass); NS-01.4 corrected-beats-raw on independent labels remains PENDING under NS-02. VERIFIED=0; BEST-STACK DELTA (c) none; no process_video.py hunk required or applied.)_
