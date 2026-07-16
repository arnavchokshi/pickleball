# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.
Closed-lane rows + dated session notes through 2026-07-12 are preserved verbatim in
`runs/manager/archive/inflight_history_20260709_20260712.md`.

Standing fence: `brand-exploration/` is the OWNER'S untracked brand work — no lane may touch it.
`cvat_upload/court_diversity_20260712/` + `w7_audit_stratum_20260709/` are staged local-only owner
labeling packages (storage-allowlisted, intentionally untracked).

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| trackI_placefuse_20260716 | Codex gpt-5.6-sol high (Track I, people-placement fusion): per-player court-frame trajectory refiner fusing TRK footpoints + SAM3D root/foot + plant-phase soft anchors + court-plane soft priors + smoothness (rigid per-frame XY correction, root-relative pose untouched); frozen scorer FIRST must reproduce w4 skeleton-direct baseline to 1e-9 (burlington 34.55mm / outdoor 33.61mm / wolverine 20.81mm / img1605 48.38mm vs 30mm bar, 3/4 breach); anti-gaming: frozen-window Arm B + phase-count preservation + reprojection non-degradation + TRK-vs-BODY disagreement diagnostic; output = NEW artifact placement_trajectory_refined.json w/ covariance+provenance (raw immutable, preview-band, VERIFIED=0). TRACK K NOTE: handoff schema will land at runs/lanes/trackI_placefuse_20260716/SCHEMA.md — coordinate fusion-world artifact naming there before claiming placement_* filenames. TRACK C NOTE: report will carry an inline process_video wiring hunk (optional stage after grounding_refine) for you to re-derive — no runner edit from this lane. pb.vision demo OUT OF SCOPE (no BODY artifacts; GPU fleet priority F/G). | Track I manager session 2026-07-16; codex session 019f6bab-964c-7cd3-9b43-c55be0d0d172 (PID 34631) | codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write -c model="gpt-5.6-sol" -c model_reasoning_effort=high --output-schema docs/racketsport/lane_report.schema.json -o runs/lanes/trackI_placefuse_20260716/report.json resume 019f6bab-964c-7cd3-9b43-c55be0d0d172 (flags BEFORE resume; nohup-detached) | NEW FILES ONLY: threed/racketsport/placement_trajectory_refine.py + scripts/racketsport/{validate_placement_slide,build_refined_placement}.py + tests/racketsport/test_placement_{trajectory_refine,refine_clis}.py + lane dir; list_scaffold_tools.py ADDITIVE entries only (Track G also appends — lane ordered to edit last + fresh-read); placement.py/foot_contact.py/worldhmr.py READ-ONLY; process_video.py/orchestrator.py/ball_arc_*/event_head FORBIDDEN | none (CPU local, banked w3_freshworlds artifacts) | same session; report.json + baseline_metrics.json + fused_metrics.json | 2026-07-16 (Track I) |
| ~~trackH_webux3_20260716~~ CLOSED | RULED ADOPT (full wave) + COMMITTED (2026-07-16, Track H second-shift manager after ~01:30 host sleep killed first shift; repair codex run had already finished at 02:01 — verified directly, no resume): round 1 (replay-first layout video y 786→226, single shared timeline w/ rally band + 168 glyph markers + inline degraded reasons, in-pane per-entity trust badges, absence chip, expandable warnings, dev-bypass hint) + repair round 1 all three items browser-verified PASS — (1) follow-play FPS fixed: pre 17.9/47.7=0.375 → guarded loop+state-asserted harness 46.1/56.9=0.809 AND segment-matched 51.4/55.1=0.933 (round-1 harness had end-of-10s-clip contamination, documented); follow-paused 118.5 no regression; (2) VM-written manifest OPENS with loud counted banner '9 assets resolved manifest-relative — original absolute paths unreachable', no raw JSON token error; (3) badge/chip overlap zero geometric intersections collapsed+expanded; manager-run vitest 280/280 + typecheck + build all EXIT 0 unpiped; zero page errors on 3 real bundles; ruling + evidence runs/lanes/webux3_fixes_20260716/{MANAGER_RULING.md,manager_verify2_result.json,fps_recheck_result.json,shots_repair1/}; remainder ranked in MANAGER_RULING.md (heavy tail-segment render cost both presets, bottom-left status-chip crowding, loop-wrap hitch, chunk-size warning) | — | — | committed set (web/replay/** + curated lane evidence; ledger Track H hunk only) | — | done | 2026-07-16 |
| ~~ballarc_scale_guard_20260715~~ | RULED **ADOPT (scoped pass)** 2026-07-16 morning by Track A manager, personally verified with real exit codes: full-697s guarded ball_arc EXIT 0 in 1493s CPU (<=1800 target) w/ 187/188 LOUD typed segment_budget_exceeded outcomes (0 malformed) + legacy physics3d diagnostic loud-skipped; segment-7 pathology QUANTIFIED (104.7s anchor-gap, 8381 candidates x 25120 RK4 substeps; root = 20 auto-bounce anchors across 697s -> game-scale segments balloon); Wolverine no-trip 5/5 artifacts byte-identical (manager cmp); trimmed-REAL-slice regression test committed (fixture R&D-reference-only, NS-07.3 strip-before-release note); focused 56/56 EXIT 0 (manager rerun; 2 earlier failures proven LOAD-FLAKE of the fixed 5s in-test budget at 681% machine load — booked defect: tests need fake-clock/generous budget); wide 3719/8/24, all 8 = sandbox socket denials proven identical at HEAD via git-archive snapshot. SAFETY fix, NOT arc recovery: 5s default can abstain on valid >5s fits; recovery needs vectorized/adaptive predict + pool prefilter + (really) trained event anchors. BEST-STACK DELTA (c) none. VERIFIED=0. Track C refinedstage UNBLOCKED by this commit. | Track A manager; codex session 019f68e2-4784-7463-af04-ccaa74c5ab09 (died overnight on model capacity + Mac sleep at ~85% done, RESUMED 2026-07-16 ~00:15 PDT as detached nohup, log_resume.txt) | report at runs/lanes/ballarc_scale_guard_20260715/report.json when done; if it dies again: codex exec [flags] resume 019f68e2-4784… with a state brief | threed/racketsport/ball_arc_solver.py + ball_arc_chain.py + its tests + lane dir (fence excludes process_video.py, ball_physics3d.py, timebase files) | none (CPU local) | hours; manager rules on report | 2026-07-16 (coordinator GO, order 1) |
| ballarc_anchorfusion_20260716 | Codex gpt-5.6-sol high (OWNER DIRECTIVE arc-recovery Step 1+2): 2,309 review-only audio onsets as SOFT split-only anchor class into guarded ball_arc on salvaged 697s inputs; pre-registered presets, 0-physics-violation kill rule, byte-identity when no anchors; Wolverine no-audio boundary proof; baseline to beat 1/188 fit | Track A manager bg bdefhix72 | report at runs/lanes/ballarc_anchorfusion_20260716/report.json | threed/racketsport/ball_arc_solver.py + ball_arc_chain.py + tests + lane dir (runner plumbing = Track C refinedstage; API handoff documented in report) | none (CPU) | hours; coverage number is the owner headline; Step 3 = Track G event candidates as 2nd anchor class; Step 4 = MOVE-1 #3 standing GO once coverage real + calpolicy ingestion lands | 2026-07-16 morning |
| ~~pbv_harness_v2_20260715~~ | RULED **ADOPT (scoped pass)** 2026-07-16 by Track A manager, manager-verified with real exit codes: frozen original byte-identical to HEAD (md5 4ebd6c53 both sides), regression A all 3 cards BYTE_IDENTICAL to frozen scorecards, manager's independent 3rd full-scale run EXIT 0 md5-identical (59e03035), tests 4/4 EXIT 0. Root cause: PB segment 92 vz 271.9m/s outside ±60m/s bounds → typed fail-closed `physics_fit_skipped` (1/490 segments), no clamp, no silent drop. Full-11-min scoring GREEN → MOVE-1 prerequisite 2/3 met. | bs7v1lnvu CLOSED | — | runs/lanes/pbv_harness_v2_20260715/** | — | DONE | 2026-07-16 |

_(2026-07-16 Track A manager: owner CAL-seed ask STAGED at
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/ (2-min tap flow: tap_corners.html
+ chosen frame t=10s + OWNER_CAL_SEED_ASK.md). ADDED to OWNER_CHECKIN.md as item 0 on 2026-07-16
after Track C's window-close landed (coordinator go); OWNER_CHECKIN "Running right now"/"Money"
also refreshed to 07-16 truth. HARD RULE standing: NO third MOVE-1 GPU attempt
without the coordinator's explicit go; prerequisites = ballarc guard adopted + harness v2 green +
trusted CAL seed.)_

_(2026-07-16 ~01:2x Track A manager: pbv11_calsolve_20260716 RULED **BLOCKED (honest kill accepted)**,
manager-verified: the line-evidence 15-intersection solve is REAL (camera median 2.61px; refreshed
evidence 1.64px; independent validator auto_calibration_ready TRUE incl. top_net 3.49px; overlay
visual PASS — projected top-net lands on the actual net tape as a solve OUTPUT) but ingestion is
rejected one gate earlier than the correction-gate premise: threed/racketsport/orchestrator.py:327
TRUSTED_INTRINSICS_SOURCES allowlists ONLY metric_15pt_reviewed. Lane refused to relabel (correct).
Banked: owner_cal_seed/court_calibration_solved.json (corrected_unverified) + solve_diagnostics +
validator evidence + reference-only pb-camera delta. FORWARD ROUTES (staged, not executed): [a]
spine/CAL policy owner ruling on an honest preview source class for single-view line-evidence
solves; [b] owner ~10-min 15-pt review via court_keypoint_review_server.py ->
build_calibration_from_review.py -> legit metric_15pt_reviewed artifact. NOT an authorization for
MOVE-1 #3.)_

_(2026-07-16 ~01:3x Track A manager, overnight window (coordinator conditional GPU GO active): CRITICAL
PREFLIGHT FINDING — the banked 4-corner seed alone does NOT unblock metric stages: capture grade stays
poor (fps<55 floor is structural for ALL 30fps content incl. Wolverine) and the pre-tracking correction
gate hard-blocks tracking when calibration is unverified-class AND line evidence isn't ready; with the
owner seed ALL required court lines now accept (2.65px mean) but top_net is refused by design under
4-corner-estimated intrinsics → auto_calibration_ready:false. Honest unlock = explicit SOLVED
calibration (Wolverine input class). DISPATCHED pbv11_calsolve_20260716 (Codex sol-high, fence: lane
dir + owner_cal_seed/ additions only; pb camera block FORBIDDEN as input; kill-rule if honest solve
can't open the gate). ALSO: guard lane interim shows full-697s guarded run = 187/188 segments loud
abstention at current budget — MOVE-1 #3 dispatch decision will additionally require the lane's
diagnosis to show a budget config with REAL fit coverage inside the wall cap, else no-GPU + writeup
per the conditional GO's failure branch. Preflight evidence:
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/calseed_preflight/.)_

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
| trk_detbench_20260716 | Sonnet GPU-ops (Track F, OWNER-DIRECTED): execute benchmark_spec_trk.md zero-shot arms on H100 spot — arm0a frozen-pool baseline reproduction (0.0001 bar), arm0b BOTSORT-feeder confound check, RF-DETR-L, RF-DETR-Seg-L (+mask archive), D-FINE-L e25 + DEIMv2-L controls (45-min no-attempt budgets); frozen scorer/GT, margin1.0+OSNet association frozen; $15 cap / 3.5h wall / on-VM shutdown rail VERIFIED-armed / DELETE+list-confirm; diagnostic only VERIFIED=0 | Track F manager, Sonnet bg agent | spec runs/lanes/trk_detbench_20260716/spec.md; report.json + DECISION_TABLE.md when done | runs/lanes/trk_detbench_20260716/** + gpu_fleet.md rows ONLY (no pipeline code, no commits) | pickleball-h100-trkdet (pending provision; Track G holds slot 2) | ~2-3.5h | 2026-07-16 morning |
| ball_anchor_boost_20260712 | Codex xhigh BL-E (last live sprint lane): audio/kinematic/blur/court-proximity anchor-evidence fusion scored vs frozen reviewed event timing (attacks the convergent ball-lift bottleneck; pb.vision reference-only) | sprint bg c7d8cfb2 | codex exec resume (session id in runs/lanes/ball_anchor_boost_20260712/log2.txt) | ball anchor/event evidence modules + tests + runs/lanes/ball_anchor_boost_20260712/** | — | overnight 07-13; verdict + BEST-STACK DELTA in lane REPORT | 2026-07-12 ~18:0x |
| ~~research_trk_rkt_20260716~~ CLOSED (Track F, 6 sub-lanes) | DONE 2026-07-16 morning: dual-survey + 2-vote refutation pattern completed for BOTH topics. TRK: RF-DETR-L first (exact Apache artifact pins; det XL/2XL=PML-1.0; no official crowded-person evidence anywhere → our frozen card decides; YOLO26m baseline is AGPL → detector swap is also the NS-07.3 move; owned-data fine-tune w/ spectator negatives = decision arm; no commercial-clean public ReID exists; McByte 3-5FPS→forensics-only; MIT selective-mask-propagation reimpl found). RKT: NO off-the-shelf for the <80px blur planar regime (build-our-own: 6 gap specs); synthetic-only unsupported at 5° (synth+small-real = the experiment); RacketVision = 2D keypoints only, side-kpts weakest; metrology-gated 3-phone GT rig spec'd (sync ≤1ms — NS-02 0.5-frame bar insufficient for contact GT); RACE-6D speed corrected to 84 FPS (no ckpts); ShapeFromBlur found as Gap-C prior art. Deliverables: TRK/RKT_CROSSCHECK_RULING.md + TRK/RKT_ADOPTION_REPORT.md + benchmark_spec_trk.md (GPU-ready) + benchmark_spec_rkt.md (Tier-1 ready, Tier-2 owner-GT-gated) under runs/research_trk_rkt_20260716/. Survived Mac sleep ~01:30 (rkt_refute had already finished; harvested on reconcile). NO GPU dispatched; VERIFIED=0 | — | — | runs/research_trk_rkt_20260716/** | — | DONE | 2026-07-16 |
| ~~tbwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED bd99c6d11 (2026-07-15, Track C): typed timebase contract wired through ingest/frames/events decode seams, canonical-beside-legacy (Wolverine 300/300 typed vs legacy 299 explicit, legacy values byte-identical); manager re-verified focused 216 EXIT 0, sandbox-bind claims 57 EXIT 0 locally; PENDING: physical 30s/5min (owner), intrinsics/rolling-shutter slices, independent labels | — | — | committed set in bd99c6d11 | — | done (survived external 1h task-cap kill via detached resume) | 2026-07-15 |
| ~~tbcam_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 1685a8878 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, P0-H remainder): typed intrinsics transforms (scale/rotate/crop) in coordinates.py + route the two ad-hoc scalers parity-first; additive optional CaptureSidecar reference_crop + rolling_shutter fields (goldens stay valid, Swift emission PENDING); orientation-mismatch fails loudly at the calibration seam; io_decode populates RollingShutterModel-or-explicit-missing (kills the hardcoded None at :495) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/tbcam_20260716/log.txt); nohup-detached | threed/racketsport/{schemas/__init__,coordinates,court_calibration,io_decode,timebase,sam3d_body_input_prep,court_auto_evidence}.py + docs schemas + their tests; process_video.py FORBIDDEN (deferred hunks inline) | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~evidence17_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 8a282d4db 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.7 non-ball_arc): audio soft evidence (pop_band_ratio et al) into fusion non-gating w/ documented bounded combination (no raw averaging); BOTH IPPE poses retained (alt_pose + carry-ambiguous-instead-of-drop, primary parity-pinned); repaired-confidence markers in ball_temporal_filter/player_id_repair/pose_temporal (values unchanged); contact-dependency-hashing runner hunks DEFERRED inline | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/evidence17_20260716/log.txt); nohup-detached | threed/racketsport/{event_fusion,racket6dof,racket_stage_runner,racket_pose_preview,ball_temporal_filter,player_id_repair,pose_temporal}.py + their tests; audio_onsets/paddle_pose_fused/ball_arc_*/runner FORBIDDEN | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| refinedstage_20260716 | Codex gpt-5.6-sol high (Track C wave 3): explicit timed events_refined + ball_arc_refined stages lifted out of world (~122s hidden), stage-count/doc coherence (RUNBOOK + truthful pin + authoritative-graph test), booked evidence17 dependency-hashing hunks re-derived + applied, guard-timeout typed-degrade test vs Track A's landed af6b8d40f semantics; dispatched AFTER the Track A gate opened | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/refinedstage_20260716/log.txt); nohup-detached | scripts/racketsport/process_video.py (SOLE owner) + RUNBOOK.md + test_process_video/test_truthful/test_spine_stage_contract + new tests; ball_arc_* READ-ONLY | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| calpolicy_20260716 | Codex gpt-5.6-sol high (Track C policy ruling implementation): ADOPTED source class line_evidence_solved_preview — orchestrator ingestion w/ mandatory space/distortion/residual/provenance declarations, permanently preview-band, structurally never satisfies metric_15pt_reviewed gates (adversarial pin), banked pbv11 solved artifact as read-only fixture; ruling rationale in the manager session record (trust contract §1.4 two-axis + preview-seed precedent; rule 12 honored by band) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/calpolicy_20260716/log.txt); nohup-detached | threed/racketsport/orchestrator.py (+ schemas additive if enums live there) + test_orchestrator_spine.py / new test_calpolicy_*; RUNBOOK/runner FORBIDDEN (refinedstage owns) | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~coordwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED aab8c3098 (2026-07-15, Track C): typed coordinate API adopted in placement/ball_court_filter/ball_physics3d/ball_inout_uncertainty/virtual_world, six SHA-pinned Wolverine digests byte-identical, distorted-synthetic + fail-closed proofs; manager re-verified 22+165+57 tests EXIT 0; its tbwire-regression isolation confirmed and fixed in c4dfb2d8b | — | — | committed set in aab8c3098 | — | done (survived 1h task-cap kill via detached resume) | 2026-07-15 |
| ~~ios_recordvis_20260716~~ CLOSED — TRACK D PARKED (owner directive 07-16: capability-first day) | RULED ADOPT (scoped pass) + COMMITTED 0d82717a2 (2026-07-16, Track D wave 2): persistent rotate-to-landscape guidance card from cold launch, visible reaction on EVERY tap (wobble/pulse/haptic; reduced-motion static emphasis), .disabled dead-zone eliminated (always-hittable in all 5 states), VoiceOver blocked-entry announcements, typed RecordControlInteractionPolicy; manager re-verified SwiftPM EXIT 0 + AppTests 58/57/1-preexisting-ANE + failing-first RED->GREEN + sim portrait proof (after_portrait_cold.png). WAVE-1+2 SIGNED BUILD STAGED runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app (codesign verified, wave-2 content fingerprinted) + MORNING_SCRIPT.md (owner 60s test TONIGHT) + NS012B_TRACE_PREP.md. Lane survived Mac sleep via detached codex + report.json on disk. PENDING: on-device visual confirmation tonight; NS-01.2b trace when recording proven | — | — | committed sets 7d1b19232 (wave 1) + 0d82717a2 (wave 2) | — | done | 2026-07-16 |
| ~~ios_recordpath_20260715~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED 7d1b19232 (2026-07-16, Track D): record-button silent-death fixed — loud-state contract (8s watchdogs, coalesced prepare, typed preview/ownership errors to banner, TCC wording, first-appearance orientation, Retry banner, RecordPath os.Logger); C1/C3/C4/C6 CONFIRMED, C5 REFUTED, C2 device-untestable; manager re-verified SwiftPM 245/0 EXIT 0 + AppTests 54/53/1-preexisting-ANE + 7 new regressions green + sim live proof; REAL DEVICE: perms authorized, blocked("Rotate to landscape") in 1.4s — phone was PORTRAIT, salience gap → wave 2; on-device RecordStopUITests run = install-race infra artifact (video proof), not counted; signed fixed build staged at runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app + INSTALL.md | — | — | committed set in 7d1b19232; device evidence in DEVICE_EVIDENCE.md | — | done | 2026-07-16 |
| ios_recordpath_20260715 (superseded row, kept for history) | Codex gpt-5.6-sol high (Track D): dead record button on owner's real iPhone 14 Pro — root-cause the silent no-op (traced primary mechanism: button `.disabled` while status pinned `.requestingAccess`, configure/ARKit/ownership chain failures swallowed by `try?`/silent guards, no timeout, no banner; five-tab sim pass ran the walker FAKE controller), land loud-state contract fixes + tests, stage signed iphoneos Debug build + exact devicectl install commands (device id B03696B6-..., currently unavailable). MANDATORY skill: ios-debugger-agent (plugin build-ios-apps@openai-curated, installed this session). Sandbox workspace-write; CoreSimulator-blocked steps go to MANAGER_VERIFY.md | codex session 019f68dd-984d-72d3-9725-80a9546355cf (PID file runs/lanes/ios_recordpath_20260715/codex.pid) | `codex exec resume 019f68dd-984d-72d3-9725-80a9546355cf` (nohup fire-and-forget) | ios/App/**, ios/Capture/**, ios/AppTests/**, runs/lanes/ios_recordpath_20260715/** (disjoint from coordwire/ball_anchor) | — | same session: report.json + guard audit + staged device build | 2026-07-15 ~20:2x |
| ~~statusdocs_20260715~~ CLOSED | RULED ADOPT + COMMITTED 9bf8eef75 (2026-07-15, Track C): stale stage-order pin fixed (RUNBOOK block + expected_order incl. coaching_facts, honest Status Interpretation split); manager re-verified truthful 14/14 exit 0 + server/render 150 passed exit 0 | — | — | RUNBOOK.md + tests/racketsport/test_truthful_capabilities.py | — | done | 2026-07-15 |
| ~~spine16_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED ffb7e0975 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.6): one authoritative stage graph (3-way assembly consolidated), REMOVE legacy pipeline_cli duplicate (readiness migration + doc/test pins), typed ExpectedOptionalAbsence + unexpected-exception-FAILS rewrite of _run_stage_safely w/ enumerated per-stage catch conversions, frame-schedule completeness (silent equal:True defaults killed, runner-side loud-path test, plan-coverage cross-check), cold/reuse/partial/failure coverage per new contract; HARD FENCE: ball_arc_* files + their two caller catches untouched (Track A live), just-landed Track C threed modules read-only | Track C manager session 2026-07-15/16 | codex exec resume (session id in runs/lanes/spine16_20260716/log.txt); dispatched nohup-detached (1h-cap immune) | scripts/racketsport/process_video.py (SOLE owner) + pipeline_cli.py (deletion candidate) + validate_pipeline_artifacts.py + pipeline_contracts.py (metadata fold only) + process_video_body_frames.py (validation default) + AGENTS/RUNBOOK pipeline_cli lines + named tests | — | same session; structured report.json + wide suite w/ real exit codes (baseline 3684/24/1-external) | 2026-07-16 |
| oneworld_design_20260716 | Codex gpt-5.6-sol high (Track K design lane, NS-04.4/04.5-class "one world v1"): DESIGN DOC for the confidence-weighted joint fusion pass — consumes court/camera (typed coordinates + calibration bands), tracks/placement (fused_world_xy + covariance_m2 = the Track I seam), smpl_motion wrists (BODY_17 idx 9/10, stride-aware), ball 2D + arc segments (incl. segment_budget_exceeded degrades), audio_onsets_v2 soft evidence, BOTH-IPPE racket hypotheses (camera-frame cm -> world via extrinsics), contact_windows; specifies the 5 behaviors (player placement consume; ball-surface SOFT priors w/ residuals NEVER snapped; contact co-location ball<->hitter-wrist volume; paddle two-IPPE resolution vs wrist traj + contact timing + ball momentum change; provenance+confidence+trust band on every output, raw immutable, unsupported stays missing); defines the 5 target metrics w/ formulas + baseline procedure VERIFIED against on-disk runs (Wolverine v5.1 full-stack; demo 11-min partial: ball2D+court+timebase only); drafts one_world_v1 artifact schema + standalone-CLI-first slotting (post ball_arc_refined 170 / pre world 180) + lane slicing + Track C wiring-request draft; NO code changes, lane dir only | Track K manager; codex session 019f6bb1-b2f9-7723-917e-c7fe34ce1235 (PID file runs/lanes/oneworld_design_20260716/codex.pid) | codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write -c model="gpt-5.6-sol" -c model_reasoning_effort=high --output-schema docs/racketsport/lane_report.schema.json -o runs/lanes/oneworld_design_20260716/report.json resume 019f6bb1-b2f9-7723-917e-c7fe34ce1235 (flags BEFORE resume; nohup fire-and-forget) | runs/lanes/oneworld_design_20260716/** ONLY (rest of repo READ-ONLY) | — | ~90 min; DESIGN.md + field-verification appendix + report.json | 2026-07-16 (Track K) |
| event_head_scaffold_20260716 | Codex gpt-5.6-sol high (Track G overnight): complete event-head training+eval scaffold per CROSSCHECK_RULING recipe — dataset layer (jhong93/spot + OpenTT loss-masked union + ShuttleSet label-only, deterministic source-disjoint splits, license postures in manifests), compact 2-class+bg temporal spotting head (E2E-Spot reference vendored third_party/spot@edec4201), type-aware ±2-frame matcher eval incl. PROTECTED 50-row owner seed (eval-only stamped), one-command fine-tune entrypoint for reviewed_labels_v2.jsonl w/ Tier-A + seed-overlap provenance HARD-FAILS, full CPU smoke battery w/ real exit codes. GPU pretrain is manager-gated AFTER smoke passes (≤$5/hr, $10 cap, on-VM teardown rail). Disk 99% full → on-the-fly decode mandated, caches ≤300MB | Track G manager; codex session 019f69f3-5bd0-7c72-87da-f2f58a41aa7a (died ~01:23 on model capacity + Mac sleep at ~90% done; manager verified partial state GREEN with real exit codes 09:1x — 12/12 lane tests EXIT 0, determinism byte-identical, smoke train/eval/finetune artifacts present; RESUMED 2026-07-16 ~09:2x as detached nohup pid 99036, log_resume.txt, closure items only: hygiene trio + wide suite + smoke_evidence + report.json) | codex exec [flags] resume <session id in runs/lanes/event_head_scaffold_20260716/log.txt> | threed/racketsport/event_head/** (NEW) + 4 new scripts/racketsport event_head CLIs + list_scaffold_tools registration entries + tests/racketsport/test_event_head_* + fixtures/event_head/ + third_party/spot (NEW vendor) + VENDOR_PINS.md row + lane dir; process_video.py FORBIDDEN | none yet (CPU first) | same night; report.json + smoke evidence w/ real exit codes | 2026-07-16 (Track G) |

_(2026-07-16 ~09:5x Track G → **TRACK A COORDINATION — contact-anchor handoff schema (anchor class #2
for your live arc anchor-fusion)**: after today's GPU pretrain, Track G delivers pretrained
event-head contact candidates on the pb.vision 11-min demo video as a typed JSON artifact —
`artifact_type: event_head_contact_anchor_candidates, schema_version: 1`, fields:
source_video{path,sha256}, video_provenance ("pbvision_demo_rd_reference_only" — NEVER training),
never_training/review_only/verified:false, model{checkpoint_sha256, license_posture: RD_ONLY,
pretrain_data}, config{threshold, nms_radius_frames, stride, image_size, window_frames, fps,
pts_convention: "normalized_to_first_video_pts" — same convention as the audio-onset anchors},
events[{frame_idx, pts_s, class: HIT|BOUNCE, score}], counts, honest_limits (zero pickleball
fine-tune yet — tennis/TT pretrain only; treat scores as weak-prior anchor evidence, not gates).
Will land under runs/lanes/event_head_pretrain_20260716/anchors/ + a ledger note on delivery.
Flag schema objections here; silence = adopted. Track G CPU smoke RULED GREEN 09:4x on the manager
battery (12/12 lane tests EXIT 0, scaffold/deadcode EXIT 0, storage fail = pre-existing stale
allowlist only, determinism byte-identical); wide suite closure in flight; GPU slot 1-of-2 claimed
per owner directive, $15 cap, provision AFTER the bounded train/inference extension is smoke-green
— no speculative VM.)_
| ~~owner_event_labels_20260715~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED d0ce58bdd (2026-07-15, Track E): scaled owner event-labeling channel — sampler/renderer/ingest CLIs + 15 tests; 300-clip session STAGED at ~/Desktop/event_labels_20260715/START_HERE.html (120 audio-onset / 75 track-discontinuity / 105 uniform-random, all 6 harvest sources, seed 20260715, 50-row eval seed +/-0.75s + pbvision + protected eval hard-excluded, page blind to stratum); manager-verified: exclusion audit 0 violations, same-seed byte-identical, 300/300 clips ffprobed w/ audio, ITEMS join 0 mismatch, node --check 0, 15 tests EXIT 0, scaffold 3/3 EXIT 0, ingest dry-run vs real manifest EXIT 0; wide suite by composition: trackC waveclose 3684p/1f where the 1f = scaffold-index (this lane, FIXED+green) + import-isolation grep; lane report.json NEVER LANDED — resumed codex proc terminated by manager at wind-down (2026-07-15 coordinator directive); ruling rests entirely on the manager verification battery; codex session 019f68df-5f28-7703-ad6e-bea1cf89e4a0 recorded for forensic resume if ever needed. FLAG: storage audit exits 1 repo-wide, PRE-EXISTING stale allowlist (cvat_upload/w5 zips deleted 07-09) — needs owner-package bookkeeping fix. HARD-STOP 07-16: Track E mid-session staged-page regen clobbered the coordinator hotfix and broke phase-1 playability (owner blocked live; manager error acknowledged) — coordinator hand-fixed staged page (autoplay-loop phase 1, phase-respecting onloadedmetadata), STAGED FILE NOW FROZEN to Track E; generator brought to BYTE-IDENTICAL parity (b24299502), 4 regression asserts, 15/15 + scaffold 3/3 EXIT 0. REOPENED-BOUNDED 07-16 dt-integrity: native video controls let phase-2 clicks toggle playback (dt from moving currentTime); coordinator hot-fixed staged page, manager fixed generator durably + found/closed the remaining rewatch/context-menu vector (pause at commit dt-read), regression tests added (15/15 EXIT 0), staged+pack HTML regenerated w/ identical localStorage key (owner answers safe), committed fence-only. OWNER NEXT: open ~/Desktop/event_labels_20260715/START_HERE.html, label 300 clips (~75-120 min), export; ingest command in runs/lanes/owner_event_labels_20260715/INGEST_README.md | — | — | committed set in d0ce58bdd; pack on Desktop (untracked) | — | done | 2026-07-15 |

_(2026-07-16 Track K manager ONLINE — one-world fusion, NS-04.4/04.5 (owner directive: "we don't
fully trust single things, but we use all info together... combining things we are most confident
in"). COORDINATION SEAMS declared: [Track I] trackI_placefuse_20260716 is LIVE — Track K consumes
placement_trajectory_refined.json as the preferred player-trajectory input (schema per their
SCHEMA.md when it lands), falling back to placement.json (fused_world_xy/covariance_m2) then
tracks.json world_xy, with the consumed tier recorded in provenance; Track K claims NO placement_*
filenames (one_world_v1* namespace only) and will NOT duplicate their refiner. [Track C] fusion
pass slots AFTER ball_arc_refined(170) /
BEFORE world(180); v1 ships as a standalone fenced module + CLI over run-dir artifacts — NO
process_video.py edits from Track K; a wiring request w/ exact stage node + RUN_IDENTITY entries
will be filed HERE once the module is adopted. [Track A] fusion consumes ball_arc outputs as-is
incl. segment_budget_exceeded loud degrades; designed for anchors improving. GPU: none needed
(CPU artifact fusion). Known input reality: NO run on disk has racket_pose_hypotheses.json or
ball_arc_render.json yet (evidence17/refined stages landed in code only) — fusion v1 must accept
both artifact generations; fullest real input set = runs/manager_stage_sam3d_wolverine_v5_1_*.)_

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
