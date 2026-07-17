# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.
Closed-lane rows + dated session notes through 2026-07-12 are preserved verbatim in
`runs/manager/archive/inflight_history_20260709_20260712.md`.

Standing fence: `brand-exploration/` is the OWNER'S untracked brand work — no lane may touch it.
`cvat_upload/court_diversity_20260712/` + `w7_audit_stratum_20260709/` are staged local-only owner
labeling packages (storage-allowlisted, intentionally untracked).

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
<<<<<<< Updated upstream
| ~~trackI_placefuse_20260716~~ CLOSED | RULED **ADOPT (scoped pass)** + COMMITTED (2026-07-16, Track I manager, personally verified w/ real unpiped exit codes + an independent manager harness with NO lane code in the loop): skeleton-direct accepted-phase max foot-slide burlington 34.55→6.72mm, outdoor 33.61→5.60mm, wolverine 20.81→6.26mm, img1605 48.38→6.75mm = **4/4 ≤30mm (baseline 1/4)**; anti-gaming frozen-window Arm B 9.48/11.78/6.26/13.36mm all strictly below baseline w/ per-phase medians also improved 4/4; phase counts 11/8/2/3 (all within ±20%, boundary deltas itemized, frozen-window arm makes comparison independent of producer re-selection); reprojection deltas <1px vs 2D evidence (median+p95, 4/4); determinism manager-reproven (2 fresh rebuilds byte-identical to banked md5 356692b3); no-snap NS-04.4 manager-verified (0 clamp flags, all 34 zero-soles pre-existing in raw, 0 newly zeroed; max correction 66.2mm, median 0.03mm); raw-input sha256 pins verified (raw artifacts untouched); sensitivity 24 rows worst ArmA 13.92 / ArmB 20.90mm, 0 over bar (not knife-edge); focused 6/6 + 51/51 + BODY-slice 143/143 EXIT 0 manager-run; wide 3909/11/24 — all 11 failures manager-attributed pre-existing (8 sandbox socket-binds pass locally, 1 Track A dirty ball_arc, 2 Track G HEAD manifest-allowlist); scaffold diff additive 4 lines, both CLIs indexed w/ reference tests; best_stack rev 13 PENDING body.placement_trajectory_refine (enabled:false, do_not_promote); North Star §2.2 BODY row updated. SCOPE HONESTY: hyperparameters chosen on the same 4 internal eval-only cards; preview-band, VERIFIED=0 stands; pb.vision NOT evaluated (no BODY artifacts — needs a GPU pipeline run first). TRACK C HANDOFF: inline process_video wiring hunk (optional stage 145 after grounding_refine, default OFF) in report.json `next` — re-derive against your live file. TRACK K HANDOFF: SCHEMA.md landed; skeleton3d-shaped + rigid_correction_xyz_m + 3x3 covariance + per-term Huber weights; consume as candidate factor, never overwrite raw. Refined artifacts (94MB) + sensitivity intermediates local-only untracked (deterministically reproducible, md5s recorded; 561MB sensitivity intermediates deleted for disk). POST-CLOSE ADDENDA (Track I manager, same session): [1] FIXUP af13d4299 — the PENDING best_stack entry initially lacked provenance.commit, which BestStackManifestError'd the manifest loader at import (body_compute imports it → repo-wide collection failures); caught by my own post-commit sanity run (Track G's G2 manager found it concurrently and routed to coordinator — already fixed by then); manifest loads + body_compute imports + truthful 12 passed verified post-fix. [2] RECORD CORRECTION to Track G's 2ad9e739b note: trackI log.txt was NEVER committed/tracked (0 hits in 0ec239325 and HEAD tree — I unstaged it pre-commit precisely for storage policy); it was a worktree-only audit item, now gzipped to 4.8MB (log.txt.gz), storage audit shows 0 trackI mentions; remaining storage failures = pre-existing cvat_upload stale allowlist (flagged 07-15) + ambient caches, not Track I, not Track G manifests (registered). ORIGINAL SPEC: per-player court-frame trajectory refiner fusing TRK footpoints + SAM3D root/foot + plant-phase soft anchors + court-plane soft priors + smoothness (rigid per-frame XY correction, root-relative pose untouched); frozen scorer FIRST must reproduce w4 skeleton-direct baseline to 1e-9 (burlington 34.55mm / outdoor 33.61mm / wolverine 20.81mm / img1605 48.38mm vs 30mm bar, 3/4 breach); anti-gaming: frozen-window Arm B + phase-count preservation + reprojection non-degradation + TRK-vs-BODY disagreement diagnostic; output = NEW artifact placement_trajectory_refined.json w/ covariance+provenance (raw immutable, preview-band, VERIFIED=0). TRACK K NOTE: handoff schema will land at runs/lanes/trackI_placefuse_20260716/SCHEMA.md — coordinate fusion-world artifact naming there before claiming placement_* filenames. TRACK C NOTE: report will carry an inline process_video wiring hunk (optional stage after grounding_refine) for you to re-derive — no runner edit from this lane. pb.vision demo OUT OF SCOPE (no BODY artifacts; GPU fleet priority F/G). | Track I manager session 2026-07-16; codex session 019f6bab-964c-7cd3-9b43-c55be0d0d172 (PID 34631) | codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write -c model="gpt-5.6-sol" -c model_reasoning_effort=high --output-schema docs/racketsport/lane_report.schema.json -o runs/lanes/trackI_placefuse_20260716/report.json resume 019f6bab-964c-7cd3-9b43-c55be0d0d172 (flags BEFORE resume; nohup-detached) | NEW FILES ONLY: threed/racketsport/placement_trajectory_refine.py + scripts/racketsport/{validate_placement_slide,build_refined_placement}.py + tests/racketsport/test_placement_{trajectory_refine,refine_clis}.py + lane dir; list_scaffold_tools.py ADDITIVE entries only (Track G also appends — lane ordered to edit last + fresh-read); placement.py/foot_contact.py/worldhmr.py READ-ONLY; process_video.py/orchestrator.py/ball_arc_*/event_head FORBIDDEN | none (CPU local, banked w3_freshworlds artifacts) | same session; report.json + baseline_metrics.json + fused_metrics.json | 2026-07-16 (Track I) |
| ~~trackH_webux3_20260716~~ CLOSED | RULED ADOPT (full wave) + COMMITTED (2026-07-16, Track H second-shift manager after ~01:30 host sleep killed first shift; repair codex run had already finished at 02:01 — verified directly, no resume): round 1 (replay-first layout video y 786→226, single shared timeline w/ rally band + 168 glyph markers + inline degraded reasons, in-pane per-entity trust badges, absence chip, expandable warnings, dev-bypass hint) + repair round 1 all three items browser-verified PASS — (1) follow-play FPS fixed: pre 17.9/47.7=0.375 → guarded loop+state-asserted harness 46.1/56.9=0.809 AND segment-matched 51.4/55.1=0.933 (round-1 harness had end-of-10s-clip contamination, documented); follow-paused 118.5 no regression; (2) VM-written manifest OPENS with loud counted banner '9 assets resolved manifest-relative — original absolute paths unreachable', no raw JSON token error; (3) badge/chip overlap zero geometric intersections collapsed+expanded; manager-run vitest 280/280 + typecheck + build all EXIT 0 unpiped; zero page errors on 3 real bundles; ruling + evidence runs/lanes/webux3_fixes_20260716/{MANAGER_RULING.md,manager_verify2_result.json,fps_recheck_result.json,shots_repair1/}; remainder ranked in MANAGER_RULING.md (heavy tail-segment render cost both presets, bottom-left status-chip crowding, loop-wrap hitch, chunk-size warning) | — | — | committed set (web/replay/** + curated lane evidence; ledger Track H hunk only) | — | done | 2026-07-16 |
| ~~trackH_evidpack_20260716~~ CLOSED | RULED SHIPPED (owner-facing visual evidence pack, 2026-07-16 evening): ~/Desktop/visual_evidence_20260716/index.html — (1) detbench YOLO26m-vs-RF-DETR-L side-by-side frame + clip + coverage timeline (burlington cov4 0.712 vs 0.997, recount matched scorer 427/598 of 600) + honest wolverine weakness frame (5 non-player FPs + drifted track, IoU-grounded labels); (2) placefuse slide bars 34.6/33.6/20.8/48.4->6.7/5.6/6.3/6.8 mm + 4 worst-plant-phase zooms (baseline reconstructed = fused - rigid_correction, matches scored to ~1e-13); (3) ballarc mechanism timeline 1/188 vs 85/361 fitted with 18 red-hatched flight-sanity windows + kill-reasons panel (KILLED preset labeled, in-rally 0.27%->29.65%, nothing promoted); (4) live viewer link verified: 5199 server was flagless (sign-in) -> restarted nohup w/ VITE_REPLAY_VERIFY_DEV_BYPASS=1 (PID 88184), opens straight to replay, 9 badges; browser-verified 15/15 images + mp4 playing + zero page errors (verify_pack.py PASS); all visuals CPU-only from committed artifacts, honesty banner (internal cards, preview band, VERIFIED=0); scripts + verification in runs/lanes/trackH_evidpack_20260716/ | Track H manager session 2026-07-16 | — | committed set (lane dir only; Desktop pack outside repo by design) | — | done | 2026-07-16 |
| ~~trackH_oneworld_render~~ CLOSED | RULED SHIPPED (partial fused view, browser-verified, 2026-07-16 wrap-up): Track K's one_world_v1 landed on Wolverine (valid:true 8/8 checks, preview_only/render_only/VERIFIED=0) and Track H rendered it in the REAL viewer with ZERO viewer-code changes — adapter runs/lanes/trackH_evidpack_20260716/adapt_one_world.py transcodes one_world_v1 -> strict racketsport_virtual_world (fused values ONLY, no baseline mixing; strict parser caught a missing trust_band.stage, fixed): 4 players at fused positions (290/300/273/244 of 300 frames — coverage gaps PRESERVED, never carried forward), ball 295 arc_measured + 5 physics_predicted w/ tier carried into provenance, per-entity CAL:PREVIEW badges; HONEST OMISSIONS (fusion's own refusals, surfaced not faked): paddles emitted EMPTY because paddle_resolution.resolved_fraction=0.0 (1102/1102 unresolved legacy wrist proxy) and marker sources NULLed because all 24 paddle contacts ABSTAINED at confidence 0 (rendering either as solved would have been a lie); manager browser-verified at :5199 (video readyState 4, canvas, 5 badges, zero page errors); shipped as evidence-pack section 5 + a state-of-the-world section (detection gains + 4-ghost weakness, foot-slide 34->7mm, ball-arc mechanism-proven-but-physics-gated, event head trained-but-weak RGB-only, VERIFIED=0 everywhere); pack re-verified 16/16 images + mp4 + zero errors | Track H manager session 2026-07-16 | — | committed set (lane dir + ledger; Desktop pack outside repo by design; web/replay UNTOUCHED) | — | done | 2026-07-16 |
| ~~ballarc_scale_guard_20260715~~ | RULED **ADOPT (scoped pass)** 2026-07-16 morning by Track A manager, personally verified with real exit codes: full-697s guarded ball_arc EXIT 0 in 1493s CPU (<=1800 target) w/ 187/188 LOUD typed segment_budget_exceeded outcomes (0 malformed) + legacy physics3d diagnostic loud-skipped; segment-7 pathology QUANTIFIED (104.7s anchor-gap, 8381 candidates x 25120 RK4 substeps; root = 20 auto-bounce anchors across 697s -> game-scale segments balloon); Wolverine no-trip 5/5 artifacts byte-identical (manager cmp); trimmed-REAL-slice regression test committed (fixture R&D-reference-only, NS-07.3 strip-before-release note); focused 56/56 EXIT 0 (manager rerun; 2 earlier failures proven LOAD-FLAKE of the fixed 5s in-test budget at 681% machine load — booked defect: tests need fake-clock/generous budget); wide 3719/8/24, all 8 = sandbox socket denials proven identical at HEAD via git-archive snapshot. SAFETY fix, NOT arc recovery: 5s default can abstain on valid >5s fits; recovery needs vectorized/adaptive predict + pool prefilter + (really) trained event anchors. BEST-STACK DELTA (c) none. VERIFIED=0. Track C refinedstage UNBLOCKED by this commit. | Track A manager; codex session 019f68e2-4784-7463-af04-ccaa74c5ab09 (died overnight on model capacity + Mac sleep at ~85% done, RESUMED 2026-07-16 ~00:15 PDT as detached nohup, log_resume.txt) | report at runs/lanes/ballarc_scale_guard_20260715/report.json when done; if it dies again: codex exec [flags] resume 019f68e2-4784… with a state brief | threed/racketsport/ball_arc_solver.py + ball_arc_chain.py + its tests + lane dir (fence excludes process_video.py, ball_physics3d.py, timebase files) | none (CPU local) | hours; manager rules on report | 2026-07-16 (coordinator GO, order 1) |
| ballarc_anchorfusion_20260716 | Codex gpt-5.6-sol high (OWNER DIRECTIVE arc-recovery Step 1+2): 2,309 review-only audio onsets as SOFT split-only anchor class into guarded ball_arc on salvaged 697s inputs; pre-registered presets, 0-physics-violation kill rule, byte-identity when no anchors; Wolverine no-audio boundary proof; baseline to beat 1/188 fit. INTERIM (manager-read from locked metrics, 09:49): preset_conservative COMPLETE and **KILLED per pre-registration** (flight_sanity_violation x16, 2662 frames demoted) despite real recovery signal underneath (53/371 segments fit vs 1/188; 18.77% in-rally frame coverage; median segment 8.5s vs 33.8s; wall 39.7min bounded). Codex proc externally killed ~09:56 mid-preset_balanced; session RESUMED (019f6ba4-9064, log_resume.txt) w/ taxonomy-first brief: classify every sanity failure (mid-flight split vs bridged direction-change vs weak-fit vs anchor-semantics structural) — taxonomy decides fixable-now vs wait-for-Track-G typed anchors. UPDATE post-TCC-freeze reconcile: preset_balanced COMPLETE + KILLED (85/361 fit, **29.65% in-rally coverage**, 18 violations, wall 38.3min); preset_broad crash proven ENVIRONMENTAL (PermissionError in _sha256 at the freeze instant, stderr evidence, no code bug); conservative→balanced trend = coverage up 18.77→29.65% AND violations up 16→18. Session RESUMED post-all-clear (log_resume2.txt): rerun broad exactly as registered, mandatory violation taxonomy (the decisive deliverable: fixable-in-preset vs needs-Track-G-typed-anchors), Wolverine no-audio boundary, focused tests, final report. | Track A manager; resumed session 019f6ba4-9064-70b0-b8d7-697e20662bea | report at runs/lanes/ballarc_anchorfusion_20260716/report.json | threed/racketsport/ball_arc_solver.py + ball_arc_chain.py + tests + lane dir (runner plumbing = Track C refinedstage; API handoff documented in report) | none (CPU) | hours; coverage number is the owner headline; Step 3 = Track G event candidates as 2nd anchor class; Step 4 = MOVE-1 #3 standing GO once coverage real + calpolicy ingestion lands | 2026-07-16 morning |
| ~~pbv_harness_v2_20260715~~ | RULED **ADOPT (scoped pass)** 2026-07-16 by Track A manager, manager-verified with real exit codes: frozen original byte-identical to HEAD (md5 4ebd6c53 both sides), regression A all 3 cards BYTE_IDENTICAL to frozen scorecards, manager's independent 3rd full-scale run EXIT 0 md5-identical (59e03035), tests 4/4 EXIT 0. Root cause: PB segment 92 vz 271.9m/s outside ±60m/s bounds → typed fail-closed `physics_fit_skipped` (1/490 segments), no clamp, no silent drop. Full-11-min scoring GREEN → MOVE-1 prerequisite 2/3 met. | bs7v1lnvu CLOSED | — | runs/lanes/pbv_harness_v2_20260715/** | — | DONE | 2026-07-16 |

_(2026-07-16 Track A manager: owner CAL-seed ask STAGED at
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/ (2-min tap flow: tap_corners.html
+ chosen frame t=10s + OWNER_CAL_SEED_ASK.md). ADDED to OWNER_CHECKIN.md as item 0 on 2026-07-16
after Track C's window-close landed (coordinator go); OWNER_CHECKIN "Running right now"/"Money"
also refreshed to 07-16 truth. HARD RULE standing: NO third MOVE-1 GPU attempt
without the coordinator's explicit go; prerequisites = ballarc guard adopted + harness v2 green +
trusted CAL seed.)_

_(2026-07-16 evening Track A manager: **CAL LEG CLOSED AT THE REVIEWED CLASS** — owner completed the
15-pt review (semantics: unmarked = out-of-frame, honored by the metric15 loader which uses only
marked points); wrapped via the designed `metric_calibration_from_reviewed_keypoints_15pt` path into
eval_clips/ball/pbvision_11min_20260713/labels/court_calibration_metric15pt.json (source
metric_15pt_reviewed, committed 1075cee57). GATE PREFLIGHT PROVEN: orchestrator ingests it, NO
correction task, calibration stage RAN; tracking 'blocked' in the smoke was --no-gpu semantics only.
CROSS-CHECK EVIDENCE: reviewed solve fx 719.3 vs line solve fx 743.0 (~3% agreement, principal
points identical) — two independent solutions agree on the camera; reviewed fit residual 19.16px
median/51.7px p95 is dominated by its designed zero-distortion choice (line solve fitted k1=-0.28 →
2.61px). Consequences carried honestly: metric_confidence low + reprojection_unusable grade →
in/out line calls abstain (too_close_to_call) while metric stages EXECUTE. OPEN QUESTION routed to
the CAL owner (not a #3 blocker): whether the metric15 fit should model distortion for this camera
class (designed distortion_improvement_threshold knob exists; manager did not touch it — fit-config
decision belongs to the CAL policy owner). STEP 4 STATUS: auth ✓, CAL ✓ (reviewed class), coverage
leg = the ONLY remaining blocker for the MOVE-1 #3 standing GO. Dispatch checklist addendum: include
Track I placement-refiner as explicit opt-in PREVIEW ARM if Track C has it wired at dispatch; don't
hold the run otherwise.)_

_(2026-07-16 late Track A manager — OWNER WIN CONDITION folded into the MOVE-1 #3 run plan (co-equal
with the head-to-head scorecard): the run must ALSO produce the first fused-demo artifact set —
[1] full BODY on ALL rally windows: requires overriding the default mesh budget
(--target-mesh-frame-budget 0 or raised, RUNBOOK controlled-run override) — WALL/COST MATH CHANGES:
BODY at full-rally cadence on 697s is the dominant new cost; size at dispatch against the 5h cap and
raise the cap request explicitly if the estimate exceeds it, do NOT silently trim mesh coverage;
[2] both-IPPE paddle artifacts (evidence17 landed — verify flags at pin); [3] explicit refined
stages + G2 typed anchors in the arc pass — REQUIRES Track C refinedstage runner plumbing for the
soft/typed anchor chain input (my chain API is default-off; runner wiring is theirs); [4] Track I
placement-refiner opt-in preview arm (wiring permitting); [5] MANDATORY dispatch-time step: confirm
with Track C which placewire/one-world runner hunks are LANDED, pin main at a SHA that includes
them, record the hunk inventory in the dispatch spec. Coverage ruling (anchorfusion taxonomy) still
gates everything. ENVELOPE (coordinator, post-1e8fbd842, AMENDED d97571f84+1): wall cap <=8h, lane
spend <=$35, H100 spot <=$5/hr or cheaper SKU (A100 ok if H100 thin). SCOPE — this is NOT a
free-standing authorization: "applies ONLY to the single MOVE-1 #3 dispatch by Track A; conditional
on the recorded sizing math with >=25% margin; requires coordinator confirmation of the dispatch
spec BEFORE provisioning; no other lane or track may cite this entry as spend authority." PROCESS:
Track A sends the coordinator the sized dispatch spec (cost estimate, wall estimate, SKU, Track C
hunk inventory, trim decision if any) and WAITS for explicit confirm before any provision call.
Sizing math = BODY frames x measured per-frame cost from prior runs + stage overheads, >=25% margin
under the cap. If honest sizing exceeds the envelope: trim to FULL mesh cadence on a contiguous
representative rally block (first N rallies end-to-end), never sparse cadence across all —
watchable complete segment beats threadbare whole. Boot-armed on-VM rail, idle self-stop, ledgers,
delete+list-confirm unchanged. Coverage gate precedes all of it.)_

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
| ~~trk_detbench_20260716~~ CLOSED (RULED 2026-07-16 by Track F manager, spot-verified vs pulled scorer JSON — 6/6 cells exact, arm0a reproduction ~3e-11 vs pins) | OWNER-DIRECTED zero-shot detector card COMPLETE on H100 usc1-a (dispatch 1 = 6/6 stockout NO-ATTEMPT $0; AMENDMENT 1 SKU ladder → dispatch 2 landed H100; wall 0.67h ≈ $1.5-2.5; rail verified-armed; DELETE+list-confirm+disks 0). RESULTS (historical-internal card, diagnostic, VERIFIED=0): arm0a PASS; arm0b FEEDER_DRIFT (paired protocol invoked; NOTE feeder YOLO26m itself hits cov4 0.97/0.93 w/ wolverine costs — pool CONSTRUCTION is a coverage variable, bounded root-cause booked for the fine-tune lane, NOT an association sweep). Zero-shot verdicts per pre-registered stop rules: RF-DETR-L REJECT-as-drop-in (wolverine 1sw/16spectFP) but burlington 0.9204 IDF1 / 0.9967 cov4 CLEAN + faster than incumbent (31 vs 35-38 ms/f) + Apache → **ADOPT-NEXT-STEP as fine-tune base, decision arm = GO**; Seg-L REJECT boxes (cov4 collapse, 7.4x wall; masks ARCHIVED for mask-cue lane); D-FINE-L REJECT (no gain); DEIMv2-L REJECT (5sw/18FP wolverine). Controls confirmed the crosscheck rank-1 on our data. Evidence: runs/lanes/trk_detbench_20260716/{DECISION_TABLE.md,report.json,vm_pull/}. FOLLOW-UP RULINGS (same day): [1] fine-tune arm data-feasibility = OUTCOME (c) BLOCKED-ON-DATA — zero owned/reviewed person boxes exist (only the 4 protected clips, players-only, no negative class; Roboflow corpus rejected on unverifiable eval-disjointness; machine prelabels not GT); exact requirement + collection plan at runs/lanes/trk_detbench_20260716/FINETUNE_DATA_RULING.md — supply probed (39 landscape owner clips ~3193s, 7 long games), tooling lane spec'd (adapt owner clip-review channel), owner ask = ONE ~75-115 min box-review session; NO fine-tune GPU spent. [2] pool-construction diagnostic booked SPEC-ONLY at runs/lanes/trk_pooldiag_20260716/spec.md (M1-M5 attribution, artifact-forensics-first, not an association sweep, NOT dispatched) | — | — | runs/lanes/trk_detbench_20260716/** | VM deleted | DONE | 2026-07-16 |
| ~~trk_rfdetr_prod_20260716~~ CLOSED-RULED 2026-07-17 (VM rerun COMPLETE on A100 usc1-a, 0.2h ~$0.3-0.5, teardown+disks list-confirmed, Track G T4 untouched): env-fidelity PASS both clips ~3e-11 (Mac CPU confirmed the outlier — STANDING RULE: frozen-card scoring on GPU-class VM only); POOLDIAG SOLVED = M4 CONFIRMED TOTAL (frozen pools were benchmark_person_trackers DEFAULTS conf 0.18/imgsz 960, NOT orchestrator 0.05/1536 as the detbench spec wrongly asserted — FEEDER_DRIFT was 100% operating point; exact-zero-delta reproduction through the feeder proves construction path innocent; trk_pooldiag_20260716 spec CLOSED BY EVIDENCE, do not dispatch). OWNER HYPOTHESIS HALF-CONFIRMED: burlington RF-DETR-L(P) 0.9220 IDF1/0.9933 cov4 ALL-CLEAN (best row ever on the clip); wolverine FPs cut 16→4 but NOT zeroed, 1 switch, IDF1/cov4 below baseline. Manager judgment: not egregious, not a clean blanket flip — decision input w/ two options (one preregistered conf-0.30 single-shot attempt, or ship-for-demo w/ FP counts on the table) at runs/lanes/trk_rfdetr_prod_20260716/FLIP_DECISION_INPUT.md — COORDINATOR RULED 2b 2026-07-17: conf-0.30 preregistered single shot FAILED (wolverine worse on EVERY axis; VM pickleball-gpu-conf030 A100 0.15h $0.22-0.38, deleted+list-confirmed) => DECISIVE NEGATIVE: wolverine spectator ghosts are HIGH-CONFIDENCE, detector-side threshold suppression EXHAUSTED (both preregistered attempts spent). Owner ruling: high-conf spectator detection is CORRECT behavior; fix belongs downstream (court-footpoint + 4-slot identity pruning = the build-our-own layer Track F's TRK research identified) — new track owns it, ghost forensics feed it. FLIP_PROPOSAL.md written (ship-for-demo @conf 0.18, preview band, do_not_promote, verbatim wolverine-regression notes text) + integration spec RESOLVED to branch 2b w/ exact reproduction targets at runs/lanes/trk_rfdetr_integrate_20260717/spec.md — **HANDED OFF, NOT DISPATCHED** (needs multi-hour orchestrator change + GPU-class reproduction; no GPU in wrap-up window; half-integrated stack rejected per ruling). NEXT SESSION: dispatch that spec. (CPU leg detail: honest typed STOP at env-fidelity gate — local Mac association NOT score-faithful on wolverine: same inputs/pins, byte-identical burlington, but gap-fill synthesized 230-vs-213 frames, +2sw/+9spectFP; NEW STANDING FINDING: frozen-card scoring must pin the PLATFORM, association gap-fill is platform-sensitive. VM RERUN in flight via the proven-faithful detbench GPU agent: pickleball-gpu-rfdetrflip, ≤$5/1.5h wall, arm0a reproduction → bounded pooldiag Phase-1 → RF-DETR-L production-equivalent pool variant P → frozen card both clips; manager writes flip proposal from its numbers) | Codex gpt-5.6-sol high (Track F, OWNER DIRECTIVE "much better at finding people — get ready to use it"): RF-DETR-L through the PRODUCTION-equivalent pool + frozen association (margin1.0+OSNet) on the frozen 2-clip card — gates in order: local env-fidelity (reproduce arm0a within 0.0001) → pool-construction attribution (EXECUTES trk_pooldiag spec Phase 1, M1-M5) → production-equivalent RF-DETR-L pool (both variants if unresolved) → frozen scorer both clips, all axes vs YOLO26m baseline; PASS ⇒ FLIP_PROPOSAL.md (preview band, do_not_promote, code-seam named — NO code/best_stack edits in-lane); CPU/MPS local, lane venv, no GPU. License: owner waived + Apache-2.0 anyway (beats AGPL incumbent) | Track F manager; codex session in log.txt (nohup) | codex exec [flags] resume <session-id in log.txt> w/ state brief | runs/lanes/trk_rfdetr_prod_20260716/** ONLY | — (local) | ~2.5h; report.json + decision table + POOLDIAG_PHASE1.md | 2026-07-16 evening |
| ~~owner_person_labels_20260716~~ CLOSED (RULED STAGED 2026-07-16 evening by Track F manager after MANDATORY independent browser verification: Playwright headless Chromium, 15/15 functional checks PASS x2 runs — load/no-video/stratum-blind-DOM/box-draw-autosave/keys/nav-603ms/empty-confirm/reload-restore/export-valid-JSON/big-Save; proof runs/lanes/owner_person_labels_20260716/manager_verify/verify_run2_proof.txt; verification used isolated browser profile, owner storage + staged files untouched) | Lane result: 450 frames (300 gameplay/100 spectator-rich/50 empty-sparse), 81 scratch (18%) proposals-withheld, session-disjoint 359/91 split, 12/12 lane tests exit 0, deterministic byte-identical rebuilds, ingest dry-run + partial-refusal proven. PACK **REVOKED + DELETED** 2026-07-16 night (owner: frames were NOT pickleball — content audit contact sheet confirms the whole owner_footage_intake_20260702 universe is dance/personal footage, zero pickleball; runs/lanes/owner_person_labels_20260716/content_audit/). LESSON (binding for any future owner-facing pack): metadata/UX verification is NOT enough — CONTENT verification (decoded thumbnails, owner confirmation of the source-clip list) is a mandatory staging gate. Tooling itself remains sound + reusable | Codex session 019f6c03-cdfb (survived mid-build TCC freeze via typed STOP + resume) | — | committed-later set: lane dir tools/tests; Desktop pack untracked (owner package) | — | DONE | 2026-07-16 |
| ball_anchor_boost_20260712 | Codex xhigh BL-E (last live sprint lane): audio/kinematic/blur/court-proximity anchor-evidence fusion scored vs frozen reviewed event timing (attacks the convergent ball-lift bottleneck; pb.vision reference-only) | sprint bg c7d8cfb2 | codex exec resume (session id in runs/lanes/ball_anchor_boost_20260712/log2.txt) | ball anchor/event evidence modules + tests + runs/lanes/ball_anchor_boost_20260712/** | — | overnight 07-13; verdict + BEST-STACK DELTA in lane REPORT | 2026-07-12 ~18:0x |
| ~~research_trk_rkt_20260716~~ CLOSED (Track F, 6 sub-lanes) | DONE 2026-07-16 morning: dual-survey + 2-vote refutation pattern completed for BOTH topics. TRK: RF-DETR-L first (exact Apache artifact pins; det XL/2XL=PML-1.0; no official crowded-person evidence anywhere → our frozen card decides; YOLO26m baseline is AGPL → detector swap is also the NS-07.3 move; owned-data fine-tune w/ spectator negatives = decision arm; no commercial-clean public ReID exists; McByte 3-5FPS→forensics-only; MIT selective-mask-propagation reimpl found). RKT: NO off-the-shelf for the <80px blur planar regime (build-our-own: 6 gap specs); synthetic-only unsupported at 5° (synth+small-real = the experiment); RacketVision = 2D keypoints only, side-kpts weakest; metrology-gated 3-phone GT rig spec'd (sync ≤1ms — NS-02 0.5-frame bar insufficient for contact GT); RACE-6D speed corrected to 84 FPS (no ckpts); ShapeFromBlur found as Gap-C prior art. Deliverables: TRK/RKT_CROSSCHECK_RULING.md + TRK/RKT_ADOPTION_REPORT.md + benchmark_spec_trk.md (GPU-ready) + benchmark_spec_rkt.md (Tier-1 ready, Tier-2 owner-GT-gated) under runs/research_trk_rkt_20260716/. Survived Mac sleep ~01:30 (rkt_refute had already finished; harvested on reconcile). NO GPU dispatched; VERIFIED=0 | — | — | runs/research_trk_rkt_20260716/** | — | DONE | 2026-07-16 |
| ~~tbwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED bd99c6d11 (2026-07-15, Track C): typed timebase contract wired through ingest/frames/events decode seams, canonical-beside-legacy (Wolverine 300/300 typed vs legacy 299 explicit, legacy values byte-identical); manager re-verified focused 216 EXIT 0, sandbox-bind claims 57 EXIT 0 locally; PENDING: physical 30s/5min (owner), intrinsics/rolling-shutter slices, independent labels | — | — | committed set in bd99c6d11 | — | done (survived external 1h task-cap kill via detached resume) | 2026-07-15 |
| ~~tbcam_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 1685a8878 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, P0-H remainder): typed intrinsics transforms (scale/rotate/crop) in coordinates.py + route the two ad-hoc scalers parity-first; additive optional CaptureSidecar reference_crop + rolling_shutter fields (goldens stay valid, Swift emission PENDING); orientation-mismatch fails loudly at the calibration seam; io_decode populates RollingShutterModel-or-explicit-missing (kills the hardcoded None at :495) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/tbcam_20260716/log.txt); nohup-detached | threed/racketsport/{schemas/__init__,coordinates,court_calibration,io_decode,timebase,sam3d_body_input_prep,court_auto_evidence}.py + docs schemas + their tests; process_video.py FORBIDDEN (deferred hunks inline) | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~evidence17_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 8a282d4db 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.7 non-ball_arc): audio soft evidence (pop_band_ratio et al) into fusion non-gating w/ documented bounded combination (no raw averaging); BOTH IPPE poses retained (alt_pose + carry-ambiguous-instead-of-drop, primary parity-pinned); repaired-confidence markers in ball_temporal_filter/player_id_repair/pose_temporal (values unchanged); contact-dependency-hashing runner hunks DEFERRED inline | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/evidence17_20260716/log.txt); nohup-detached | threed/racketsport/{event_fusion,racket6dof,racket_stage_runner,racket_pose_preview,ball_temporal_filter,player_id_repair,pose_temporal}.py + their tests; audio_onsets/paddle_pose_fused/ball_arc_*/runner FORBIDDEN | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~placewire_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 02982d358 2026-07-16; manager re-verified focused 197 EXIT 0, only failures = Track F bundle; runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 4): wire Track I's adopted placement-trajectory fusion (0ec239325) as opt-in default-OFF stage after grounding_refine — re-derive their report hunk vs the post-refinedstage canonical graph; byte-parity when off; typed spine16 failure semantics; preview band; consumes existing best_stack rev-13 PENDING entry (no edit); artifact production only (Track K consumer comes later) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/placewire_20260716/log.txt); nohup-detached | scripts/racketsport/process_video.py (SOLE owner) + RUNBOOK + test_process_video/test_truthful/test_spine_stage_contract + new test_placewire_*; placement_trajectory_refine.py READ-ONLY | — | same session; report.json + wide suite w/ attribution | 2026-07-16 |
| ~~refinedstage_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED d941b0d7d 2026-07-16 post-thaw; manager re-verified w/ real exit codes; runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 3): explicit timed events_refined + ball_arc_refined stages lifted out of world (~122s hidden), stage-count/doc coherence (RUNBOOK + truthful pin + authoritative-graph test), booked evidence17 dependency-hashing hunks re-derived + applied, guard-timeout typed-degrade test vs Track A's landed af6b8d40f semantics; dispatched AFTER the Track A gate opened | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/refinedstage_20260716/log.txt); nohup-detached | scripts/racketsport/process_video.py (SOLE owner) + RUNBOOK.md + test_process_video/test_truthful/test_spine_stage_contract + new tests; ball_arc_* READ-ONLY | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~calpolicy_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED 5cb556fd2 2026-07-16 post-thaw; manager re-verified w/ real exit codes; runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C policy ruling implementation): ADOPTED source class line_evidence_solved_preview — orchestrator ingestion w/ mandatory space/distortion/residual/provenance declarations, permanently preview-band, structurally never satisfies metric_15pt_reviewed gates (adversarial pin), banked pbv11 solved artifact as read-only fixture; ruling rationale in the manager session record (trust contract §1.4 two-axis + preview-seed precedent; rule 12 honored by band) | Track C manager session 2026-07-16 | codex exec resume (session id in runs/lanes/calpolicy_20260716/log.txt); nohup-detached | threed/racketsport/orchestrator.py (+ schemas additive if enums live there) + test_orchestrator_spine.py / new test_calpolicy_*; RUNBOOK/runner FORBIDDEN (refinedstage owns) | — | same session; report.json + wide suite w/ real exit codes | 2026-07-16 |
| ~~coordwire_20260715~~ CLOSED | RULED ADOPT (scoped pass, wired) + COMMITTED aab8c3098 (2026-07-15, Track C): typed coordinate API adopted in placement/ball_court_filter/ball_physics3d/ball_inout_uncertainty/virtual_world, six SHA-pinned Wolverine digests byte-identical, distorted-synthetic + fail-closed proofs; manager re-verified 22+165+57 tests EXIT 0; its tbwire-regression isolation confirmed and fixed in c4dfb2d8b | — | — | committed set in aab8c3098 | — | done (survived 1h task-cap kill via detached resume) | 2026-07-15 |
| ~~ios_recordvis_20260716~~ CLOSED — TRACK D PARKED (owner directive 07-16: capability-first day) | RULED ADOPT (scoped pass) + COMMITTED 0d82717a2 (2026-07-16, Track D wave 2): persistent rotate-to-landscape guidance card from cold launch, visible reaction on EVERY tap (wobble/pulse/haptic; reduced-motion static emphasis), .disabled dead-zone eliminated (always-hittable in all 5 states), VoiceOver blocked-entry announcements, typed RecordControlInteractionPolicy; manager re-verified SwiftPM EXIT 0 + AppTests 58/57/1-preexisting-ANE + failing-first RED->GREEN + sim portrait proof (after_portrait_cold.png). WAVE-1+2 SIGNED BUILD STAGED runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app (codesign verified, wave-2 content fingerprinted) + MORNING_SCRIPT.md (owner 60s test TONIGHT) + NS012B_TRACE_PREP.md. Lane survived Mac sleep via detached codex + report.json on disk. PENDING: on-device visual confirmation tonight; NS-01.2b trace when recording proven | — | — | committed sets 7d1b19232 (wave 1) + 0d82717a2 (wave 2) | — | done | 2026-07-16 |
| ~~ios_recordpath_20260715~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED 7d1b19232 (2026-07-16, Track D): record-button silent-death fixed — loud-state contract (8s watchdogs, coalesced prepare, typed preview/ownership errors to banner, TCC wording, first-appearance orientation, Retry banner, RecordPath os.Logger); C1/C3/C4/C6 CONFIRMED, C5 REFUTED, C2 device-untestable; manager re-verified SwiftPM 245/0 EXIT 0 + AppTests 54/53/1-preexisting-ANE + 7 new regressions green + sim live proof; REAL DEVICE: perms authorized, blocked("Rotate to landscape") in 1.4s — phone was PORTRAIT, salience gap → wave 2; on-device RecordStopUITests run = install-race infra artifact (video proof), not counted; signed fixed build staged at runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app + INSTALL.md | — | — | committed set in 7d1b19232; device evidence in DEVICE_EVIDENCE.md | — | done | 2026-07-16 |
| ios_recordpath_20260715 (superseded row, kept for history) | Codex gpt-5.6-sol high (Track D): dead record button on owner's real iPhone 14 Pro — root-cause the silent no-op (traced primary mechanism: button `.disabled` while status pinned `.requestingAccess`, configure/ARKit/ownership chain failures swallowed by `try?`/silent guards, no timeout, no banner; five-tab sim pass ran the walker FAKE controller), land loud-state contract fixes + tests, stage signed iphoneos Debug build + exact devicectl install commands (device id B03696B6-..., currently unavailable). MANDATORY skill: ios-debugger-agent (plugin build-ios-apps@openai-curated, installed this session). Sandbox workspace-write; CoreSimulator-blocked steps go to MANAGER_VERIFY.md | codex session 019f68dd-984d-72d3-9725-80a9546355cf (PID file runs/lanes/ios_recordpath_20260715/codex.pid) | `codex exec resume 019f68dd-984d-72d3-9725-80a9546355cf` (nohup fire-and-forget) | ios/App/**, ios/Capture/**, ios/AppTests/**, runs/lanes/ios_recordpath_20260715/** (disjoint from coordwire/ball_anchor) | — | same session: report.json + guard audit + staged device build | 2026-07-15 ~20:2x |
| ~~statusdocs_20260715~~ CLOSED | RULED ADOPT + COMMITTED 9bf8eef75 (2026-07-15, Track C): stale stage-order pin fixed (RUNBOOK block + expected_order incl. coaching_facts, honest Status Interpretation split); manager re-verified truthful 14/14 exit 0 + server/render 150 passed exit 0 | — | — | RUNBOOK.md + tests/racketsport/test_truthful_capabilities.py | — | done | 2026-07-15 |
| ~~spine16_20260716~~ CLOSED (RULED ADOPT scoped pass + COMMITTED ffb7e0975 2026-07-16; manager re-verified w/ real exit codes; details runs/manager/trackC_20260716/RULINGS.md) | Codex gpt-5.6-sol high (Track C wave 2, NS-01.6): one authoritative stage graph (3-way assembly consolidated), REMOVE legacy pipeline_cli duplicate (readiness migration + doc/test pins), typed ExpectedOptionalAbsence + unexpected-exception-FAILS rewrite of _run_stage_safely w/ enumerated per-stage catch conversions, frame-schedule completeness (silent equal:True defaults killed, runner-side loud-path test, plan-coverage cross-check), cold/reuse/partial/failure coverage per new contract; HARD FENCE: ball_arc_* files + their two caller catches untouched (Track A live), just-landed Track C threed modules read-only | Track C manager session 2026-07-15/16 | codex exec resume (session id in runs/lanes/spine16_20260716/log.txt); dispatched nohup-detached (1h-cap immune) | scripts/racketsport/process_video.py (SOLE owner) + pipeline_cli.py (deletion candidate) + validate_pipeline_artifacts.py + pipeline_contracts.py (metadata fold only) + process_video_body_frames.py (validation default) + AGENTS/RUNBOOK pipeline_cli lines + named tests | — | same session; structured report.json + wide suite w/ real exit codes (baseline 3684/24/1-external) | 2026-07-16 |
| ~~oneworld_impl_20260716~~ CLOSED | RULED **ADOPT (scoped pass)** + COMMITTED **a54b7c451** 2026-07-16 (Track K manager, wrap-up). Manager-verified w/ REAL unpiped exit codes: independent rebuild BYTE-IDENTICAL to lane output (sha256 2c58c4a8bc84…), validate EXIT 0 valid:true 0 warnings, focused 18 passed EXIT 0, **scaffold index 3 passed REAL_EXIT=0 (A13 red CLOSED — the repo-wide breakage this lane caused is fixed; other tracks may stop attributing it)**, raw immutability 14/14. DELIVERABLE (owner win condition, Wolverine 300 frames): 4 players placed every rally frame (placement_fused, conf .82-.92), 4 paddles ALWAYS carrying display_pose_world w/ honest band (all `unresolved_legacy_wrist_proxy` — gen-1 proxy is all this bundle has), ball 295 arc_measured + 5 physics_predicted + 0 absent, 28 typed events (24 paddle_contact + 4 floor_bounce). **HEADLINE FINDING: fusion CONFIRMED ZERO of 24 declared contacts (22 unsupported >1.2m, 2 too_close_to_call) — at frame 78 the declared hitter's wrist is 11.17m from the arc ball; the pass refused to move the ball and left raw events immutable. Bounded abstention worked as designed; the bundle's contacts/arc/players genuinely disagree.** M1 <=0.60m NOT met + NOT claimed (0 supported); M3 coverage_measured 0.39 = baseline; M5 non-regressing (player p90 61.2683->61.2658px); M2 residuals honestly 0 (upstream anchors land exactly at z=r_b; synthetic no-snap proofs pass); M4 denominator 0. Demo 697s = honest partial (10776 ray_court_projection + 10146 true absences, 666 bounce previews, 2309 review-only onsets, 0 players/paddles/contacts, corrected_unverified). Lane-honest PARTIAL accepted: A9 wide suite time-boxed at 34% w/ 15 unrelated dirty-tree failures (attributed, not HEAD-proven — the ONE debt of this ruling); gen-2 regen EXIT 1 environmental (tracking reuse refused w/o migration attestation — a real finding for Track C). Artifacts 18M/27M stay on disk (storage policy): runs/lanes/oneworld_impl_20260716/{wolverine,demo}/one_world_v1.json. DEFERRED: schemas/__init__.py ARTIFACT_MODELS hunk (inline in report.json) for the integration window. Ruling: runs/manager/trackK_20260716/RULINGS.md | — | — | committed set in a54b7c451 (15 files, fence-only by pathspec) | — | done | 2026-07-16 (Track K) |
| ~~oneworld_impl_20260716 (dispatch row, superseded)~~ | Codex gpt-5.6-sol high (Track K implementation, slices 1+2 of the ADOPTED design): threed/racketsport/one_world_v1.py (models IN-MODULE — schemas/__init__.py FORBIDDEN, calpolicy live; ARTIFACT_MODELS registration = deferred inline hunk in report) + 3 CLIs (build_/report_/validate_one_world_v1) + docs schemas + fenced tests + additive scaffold-index dict entries ONLY (Track I/G uncommitted lines untouched, manager commits hunk-selectively); acceptance: build+validate EXIT 0 on lane-dir Wolverine clone, determinism byte-identical, raw-input sha immutability, metrics table vs FROZEN design baselines (M1 wrist-volume residual supported-median <=0.60m vs 7.9737m + frame-78-class declared-hitter refusals; M3 coverage >=0.39; M5 reprojection fused <= baseline + max(1px,5%) w/ kill->suppress; M4 synthetic-only, denominator 0 honest), 24-contact hitter audit, honest demo partial (ball+court+derived audio only, coverage 0), gen-2 regen attempt NON-blocking (runner volatile), A8b owner-directive tests (contact_evidence_vector emission, audio-only-cannot-confirm, neighbor-bleed-creates-nothing, co-location-discount), wide suite w/ attribution; search sweeps MUST exclude eval_clips/ball/{outdoor,indoor}*; OWNER WIN-CONDITION ADDENDUM A10-A12 injected mid-flight (session killed+resumed w/ spec_addendum_A10.md ~25min in): A10 paddle ALWAYS emits display_pose_world w/ honest band even unresolved (both hypotheses retained, resolve bar untouched), A11 first-class typed events[] (paddle_contact/floor_bounce/net_contact/net_cross w/ t+locations+bands) for the viewer, A12 ball continuity chain estimate_tier (arc_measured -> physics_predicted <=0.5s-from-support -> ray_court_projection altitude_unknown -> absent only if no 2D; metrics tier-1-only; M3 stratified coverage_measured vs coverage_with_predicted); priority = demo-visible completeness w/ honest bands > per-metric polish | Track K manager; codex session 019f6e23-5d6a-73d2-b9c7-821f022f62b4 (PID file runs/lanes/oneworld_impl_20260716/codex.pid holds the RESUME pid) | codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write -c model="gpt-5.6-sol" -c model_reasoning_effort=high --output-schema docs/racketsport/lane_report.schema.json -o runs/lanes/oneworld_impl_20260716/report.json resume 019f6e23-5d6a-73d2-b9c7-821f022f62b4 (flags BEFORE resume; nohup fire-and-forget) | NEW: threed/racketsport/one_world_*.py + scripts/racketsport/{build_one_world_v1,report_one_world_metrics,validate_one_world_v1}.py + docs/racketsport/one_world_v1*.schema.json + tests/racketsport/test_one_world_*.py + lane dir; SHARED-ADDITIVE: list_scaffold_tools.py (3 dict entries/CLI only); FORBIDDEN: process_video.py, orchestrator.py, schemas/__init__.py, ball_arc_*, placement*, event_head/**, other run dirs (read-only) | — | hours; report.json + wide suite w/ real exit codes | 2026-07-16 (Track K) |
| ~~oneworld_design_20260716~~ CLOSED | RULED **ADOPT (as-design, scoped pass)** 2026-07-16 by Track K manager, lane-honest PARTIAL accepted — personally verified w/ real exit codes: baseline_probe re-run EXIT 0 BYTE-IDENTICAL to banked (Wolverine v5.1 baselines CONFIRMED: ball-at-contact wrist-volume residual median 7.9737m/p90 11.1651m over 24/24 events; world coverage@0.5 = 0.39 = 117/300); forensic root-cause: frame 78 2D ball (conf 0.94) sits INSIDE player 4's bbox while contact_windows declares player_id=1 across the image — baseline incoherence REAL (attribution + sparse-anchor arc), probe join correct; design compliant w/ NS-04.4/04.5 kill language (soft priors w/ caps, never snap, both IPPE poses to an independent-evidence resolver that never chooses by reprojection, bounded abstention >1.2m, absence semantics, preview/VERIFIED=0 permanent); label incident BOUNDED (broad rg printed protected label paths into transcript ONLY; deliverables clean, probes attest outdoor_indoor_labels_read:false; log.txt NOT committed; lesson = mandatory search-sweep exclusion globs in all Track K specs); Track I SCHEMA.md consumption ALIGNED as-is (no change requests); Track H schema asks satisfied + Track K answer: DISTINCT "fused" provenance class, physics_predicted NOT overloaded; OWNER contact-directive compliance VERIFIED (two-layer fused contacts, audio non-gating; impl spec A8b makes it testable). Full ruling: runs/manager/trackK_20260716/RULINGS.md | — | — | committed set: DESIGN.md + FIELD_VERIFICATION.md + probes + outputs + report.json + spec.md (log.txt untracked by design) | — | done | 2026-07-16 (Track K) |
| ~~oneworld_design_20260716 (dispatch row, superseded)~~ | Codex gpt-5.6-sol high (Track K design lane, NS-04.4/04.5-class "one world v1"): DESIGN DOC for the confidence-weighted joint fusion pass — consumes court/camera (typed coordinates + calibration bands), tracks/placement (fused_world_xy + covariance_m2 = the Track I seam), smpl_motion wrists (BODY_17 idx 9/10, stride-aware), ball 2D + arc segments (incl. segment_budget_exceeded degrades), audio_onsets_v2 soft evidence, BOTH-IPPE racket hypotheses (camera-frame cm -> world via extrinsics), contact_windows; specifies the 5 behaviors (player placement consume; ball-surface SOFT priors w/ residuals NEVER snapped; contact co-location ball<->hitter-wrist volume; paddle two-IPPE resolution vs wrist traj + contact timing + ball momentum change; provenance+confidence+trust band on every output, raw immutable, unsupported stays missing); defines the 5 target metrics w/ formulas + baseline procedure VERIFIED against on-disk runs (Wolverine v5.1 full-stack; demo 11-min partial: ball2D+court+timebase only); drafts one_world_v1 artifact schema + standalone-CLI-first slotting (post ball_arc_refined 170 / pre world 180) + lane slicing + Track C wiring-request draft; NO code changes, lane dir only | Track K manager; codex session 019f6bb1-b2f9-7723-917e-c7fe34ce1235 (PID file runs/lanes/oneworld_design_20260716/codex.pid) | codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write -c model="gpt-5.6-sol" -c model_reasoning_effort=high --output-schema docs/racketsport/lane_report.schema.json -o runs/lanes/oneworld_design_20260716/report.json resume 019f6bb1-b2f9-7723-917e-c7fe34ce1235 (flags BEFORE resume; nohup fire-and-forget) | runs/lanes/oneworld_design_20260716/** ONLY (rest of repo READ-ONLY) | — | ~90 min; DESIGN.md + field-verification appendix + report.json | 2026-07-16 (Track K) |
| event_head_scaffold_20260716 | Codex gpt-5.6-sol high (Track G overnight): complete event-head training+eval scaffold per CROSSCHECK_RULING recipe — dataset layer (jhong93/spot + OpenTT loss-masked union + ShuttleSet label-only, deterministic source-disjoint splits, license postures in manifests), compact 2-class+bg temporal spotting head (E2E-Spot reference vendored third_party/spot@edec4201), type-aware ±2-frame matcher eval incl. PROTECTED 50-row owner seed (eval-only stamped), one-command fine-tune entrypoint for reviewed_labels_v2.jsonl w/ Tier-A + seed-overlap provenance HARD-FAILS, full CPU smoke battery w/ real exit codes. GPU pretrain is manager-gated AFTER smoke passes (≤$5/hr, $10 cap, on-VM teardown rail). Disk 99% full → on-the-fly decode mandated, caches ≤300MB | Track G manager; codex session 019f69f3-5bd0-7c72-87da-f2f58a41aa7a (died ~01:23 on model capacity + Mac sleep at ~90% done; manager verified partial state GREEN with real exit codes 09:1x — 12/12 lane tests EXIT 0, determinism byte-identical, smoke train/eval/finetune artifacts present; RESUMED 2026-07-16 ~09:2x as detached nohup pid 99036, log_resume.txt, closure items only: hygiene trio + wide suite + smoke_evidence + report.json) | codex exec [flags] resume <session id in runs/lanes/event_head_scaffold_20260716/log.txt> | threed/racketsport/event_head/** (NEW) + 4 new scripts/racketsport event_head CLIs + list_scaffold_tools registration entries + tests/racketsport/test_event_head_* + fixtures/event_head/ + third_party/spot (NEW vendor) + VENDOR_PINS.md row + lane dir; process_video.py FORBIDDEN | none yet (CPU first) | same night; report.json + smoke evidence w/ real exit codes | 2026-07-16 (Track G) |

| ~~event_head_ext_20260716~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED 40b013ab2 (2026-07-16 ~09:5x, Track G2 manager): train_event_head.py --full manifest-driven pretrain mode (deterministic, val-F1@±2 best/last ckpts, --max-wall-minutes honest partial + --init-checkpoint resume, typed exit-3 protected/owner-input rejection, --smoke byte-compatible) + NEW build_event_head_anchor_candidates.py emitting the FROZEN Track A anchor schema + tests + registration. G2-manager-verified real unpiped exit codes: focused 7/7 EXIT 0, scaffold index EXIT 0, independent anchor-CLI smoke EXIT 0 schema-valid on tiny fixture (3 HIT/0 BOUNCE from overfit proof ckpt — mechanics only). Codex 019f6bc6 exited clean after report (all 8 acceptance PASS). CODE_GREEN created w/ 9 md5 pins → GPU lane unblocked | — | — | committed set in 40b013ab2 | — | done | 2026-07-16 (Track G ext, ruled by G2) |
| event_head_pretrain_20260716 | Sonnet GPU-ops bg agent (Track G2): TRAINING LIVE on pickleball-t4-eventhead (T4 usc1-b SPOT ~$0.2-0.4/hr, boot-armed rail poweroff 08:59Z) — TRAIN_STARTED 2026-07-17T03:49Z, probe 0.4396 steps/s, cap-formula 3956 steps @224/win64/b4 imagenet-init, 8 val evals, ETA train ~06:18Z + eval/anchors ~06:50Z. SURVIVED: 12 spot stockouts (2 ladders + AMENDMENT 1), rail-arm-race fail-closed DELETE (AMENDMENT 2 = boot-armed rail via startup script — ops lesson booked), ~9.5h Mac freeze (VM idle-watchdog TERMINATED as designed, disk+staging intact), AV1 decode wall (5 pilot videos transcoded h264 Mac-side frame-count-verified, VM copies only). Spend so far ~$0.1-0.3 vs $10 HARD cap. OWNER DIRECTIVE folded in: anchor artifact carries per-candidate EVIDENCE VECTOR (audio onset proximity/strength from review-only pbvision onsets + 2D ball-track kink + wrist=unavailable-with-reason + head score; head = SOLE emitter, no audio-only candidates) via lane-dir enrich_anchors_evidence.py (tested both modes EXIT 0); NOTE model is RGB-only as committed — track/wrist conditioning channels are a booked follow-up design, not in tonight's checkpoint | Track G2 manager, Sonnet bg agent | status.log + report.json in lane dir | runs/lanes/event_head_pretrain_20260716/** + fleet row (manager-written) | pickleball-t4-eventhead usc1-b | Mac-side protected-seed + public eval by manager after two-sided-md5 pull; enriched anchor handoff to Track A tonight; teardown DELETE+confirm | 2026-07-16 ~09:47, recovered post-freeze 2026-07-17 ~03:3xZ (Track G2) |

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

_(2026-07-16 ~10:0x Track G2 → **TRACK A converter seam, anchor class #2**: the frozen anchor JSON
schema is UNCHANGED (committed working CLI emits it — see 40b013ab2; sample at
runs/lanes/event_head_ext_20260716/anchor_cpu_proof/anchors.json). One integration fact for your
fusion side: your `SoftSegmentBoundary.__post_init__` hard-pins `anchor_class == "audio_onset_soft"`
(ball_arc_solver.py:311-329), so event-head candidates CANNOT be injected as SoftSegmentBoundary
directly. Field mapping is trivial and lossless — corrected_time_s=pts_s (same
normalized_to_first_video_pts convention as your audio onsets), frame=frame_idx,
boundary_id=f"event_head_{class}_{i:04d}", score available for your floor/rank rules — but you need
either (a) an allowed `event_head_soft` anchor_class or (b) an adapter on your side; Track G will
NOT relabel event-head candidates as audio_onset_soft (dishonest provenance). Delivery target
unchanged: runs/lanes/event_head_pretrain_20260716/anchors/pbvision_11min_event_head_anchors.json
this afternoon (GPU lane provisioning now; L4/A100 stockout ladder in progress). Scores are
weak-prior evidence from tennis/TT pretrain — zero pickleball fine-tune; treat as split-only
candidates, never gates.)_

_(2026-07-17 ~05:4x Track G2 → **TRACK A: anchor QUALITY LABEL — read before weighting**. The real
pretrained checkpoint exists (T4, 3956 steps, best_val_f1 0.3631 @±2f, md5
654ec44d0752529ece8d1712ecc07347) and its demo-video anchors land at
runs/lanes/event_head_pretrain_20260716/anchors/. HONEST POSTURE, measured not guessed: the head is
**HIGH-PRECISION / LOW-RECALL on public tennis**, NOT noise and NOT strong. Manager's matched-window
eval (16 clips, 42 GT events, thr 0.5, tol ±2f): HIT tp6/fp0 (recall ~22%), BOUNCE tp3/fp0 (recall
~20%) — 9 predictions, ZERO false positives. IMPORTANT: the lane's earlier "0 TP" public-eval
headline is RETRACTED as a measurement artifact — the committed eval CLI scores 15-frame windows
against a 64-frame-context head (eval_event_head.py:68-69); the identical checkpoint on the identical
clips scores 9 TP / 0 FP at matched 64f and 0 TP at 15f (evidence: eval/matched_window64_eval.json vs
eval/control_window15_eval.json). **RETRACTION + FINAL VERDICT 2026-07-17 06:0x — DO NOT INGEST
these anchors. My own "expect sparse-but-clean HITs on the demo video" guidance above is WITHDRAWN;
the measured demo-video output disproves it.** Zero-shot transfer to pickleball FAILED (not "weak
success"): 4,990 HIT candidates over 697.4s = 7.16/s (real games have ~200-400 contacts total),
median inter-HIT gap 4 frames at stride 2, **98% of all seconds contain a HIT** (the video has 41
rallies WITH dead time) — a near-uniform activation carpet, not discrete events; wider NMS merging
does not rescue it (327 clusters @5f, 103 @10f, 49 @15f). The audio cross-check cannot rescue it
either and is itself informative for you: 70.2% of candidates have an onset within ±0.15s but the
CHANCE baseline is ~99.3% (2,309 onsets / 697.4s = 0.302s mean spacing → any arbitrary timestamp
co-locates) — i.e. co-location at/below chance, ZERO discriminative information. That independently
corroborates the owner's audio-bleed observation and is direct evidence for why an audio-only anchor
class trips physics. Full numbers + method:
runs/lanes/event_head_pretrain_20260716/ANCHOR_VERDICT.md. The lane's real deliverables are the
checkpoint + SCALE_UP_SPEC.md (root cause: 2.4% label reach / 18.1% media coverage /
one-window-per-row — the head saw a rounding error of the corpus and zero pickleball). Re-attempt
anchors ONLY from a pickleball-fine-tuned checkpoint that first passes a cheap pre-flight:
candidates/second must land ~0.3-1.0, not 7. Per OWNER DIRECTIVE
2026-07-16 (neighboring-court audio bleed), each candidate carries an EVIDENCE VECTOR — event_head
score + audio onset proximity/strength (review-only pbvision onsets, ±0.15s window) + 2D ball-track
kink (direction-change deg from the salvaged 697s chain) + wrist_swing_proximity marked
unavailable-with-reason (no BODY artifacts exist for this video) — so you can weight per signal
rather than on one scalar. The visual head is the SOLE emitter: no audio-only candidates are typed as
contacts (that class is yours and already trips physics). Converter seam unchanged: your
SoftSegmentBoundary pins anchor_class=="audio_onset_soft" (ball_arc_solver.py:311-329), so you need
an allowed `event_head_soft` class or an adapter — Track G will not relabel these as audio_onset_soft
(dishonest provenance).)_

_(2026-07-16 ~11:0x TRACK G CLOSE-OUT — handed off to G2 manager by coordinator order. DONE +
manager-verified w/ real exit codes: full event-head scaffold in working tree (uncommitted) —
threed/racketsport/event_head/{datasets,model,matcher}.py, 4 registered CLIs, 12/12 lane tests
EXIT 0, builder determinism byte-identical, CPU smoke train (loss 0.926→0.458, RD_ONLY manifest),
protected-seed eval correctly stamped eval_only/never_training (28 typed + 1 other + 21 neg,
honest zero-shot F1 0.0), finetune fixture EXIT 0 + Tier-A/seed-overlap/missing-file hard-fails
tested (typed exit 22/3), third_party/spot vendored @ edec4201 + VENDOR_PINS row, scaffold-index
EXIT 0, deadcode EXIT 0, storage audit fail = pre-existing stale allowlist ONLY. IN FLIGHT at
handoff: [a] closure codex pid 99036 (session 019f69f3…) finishing the FULL wide suite (pytest pid
99471, ~30.5 CPU min) then writes report.json + smoke_evidence.md into
runs/lanes/event_head_scaffold_20260716/; [b] extension codex pid 75411 (session 019f6bc6…,
HANDOFF_NOTE.md in runs/lanes/event_head_ext_20260716/) building --full pretrain mode + the
Track A anchor CLI, mid-test-writing. NEVER DISPATCHED: the GPU pretrain lane — Agent-tool
dispatch was DENIED twice by the permission system (user-authorized cap is $10 not the relayed
$15; precondition = a LANDED report.json as verifiable CPU-smoke evidence, my in-session ruling
insufficient per the denial). Spec staged dispatch-ready at
runs/lanes/event_head_pretrain_20260716/spec.md ($10 cap, rail-at-provision, reuse-A100-else-L4,
VM-side OpenTT+pbvision fetch, CODE_GREEN code-sync gate — marker never created). Also never done:
commit (was gated on wide suite; NOTE for committer — Track I also edits list_scaffold_tools.py,
stage hunks selectively), fleet ledger row (no VM ever existed, $0 spent), protected-seed eval of
a REAL pretrained checkpoint, pbvision anchors, owner fine-tune run (one-command entrypoint is
built + fixture-proven; real reviewed_labels_v2.jsonl not yet ingested). Monitors: all disarmed.)_
| ~~owner_event_labels_20260715~~ CLOSED | RULED ADOPT (scoped pass) + COMMITTED d0ce58bdd (2026-07-15, Track E): scaled owner event-labeling channel — sampler/renderer/ingest CLIs + 15 tests; 300-clip session STAGED at ~/Desktop/event_labels_20260715/START_HERE.html (120 audio-onset / 75 track-discontinuity / 105 uniform-random, all 6 harvest sources, seed 20260715, 50-row eval seed +/-0.75s + pbvision + protected eval hard-excluded, page blind to stratum); manager-verified: exclusion audit 0 violations, same-seed byte-identical, 300/300 clips ffprobed w/ audio, ITEMS join 0 mismatch, node --check 0, 15 tests EXIT 0, scaffold 3/3 EXIT 0, ingest dry-run vs real manifest EXIT 0; wide suite by composition: trackC waveclose 3684p/1f where the 1f = scaffold-index (this lane, FIXED+green) + import-isolation grep; lane report.json NEVER LANDED — resumed codex proc terminated by manager at wind-down (2026-07-15 coordinator directive); ruling rests entirely on the manager verification battery; codex session 019f68df-5f28-7703-ad6e-bea1cf89e4a0 recorded for forensic resume if ever needed. FLAG: storage audit exits 1 repo-wide, PRE-EXISTING stale allowlist (cvat_upload/w5 zips deleted 07-09) — needs owner-package bookkeeping fix. HARD-STOP 07-16: Track E mid-session staged-page regen clobbered the coordinator hotfix and broke phase-1 playability (owner blocked live; manager error acknowledged) — coordinator hand-fixed staged page (autoplay-loop phase 1, phase-respecting onloadedmetadata), STAGED FILE NOW FROZEN to Track E; generator brought to BYTE-IDENTICAL parity (b24299502), 4 regression asserts, 15/15 + scaffold 3/3 EXIT 0. REOPENED-BOUNDED 07-16 dt-integrity: native video controls let phase-2 clicks toggle playback (dt from moving currentTime); coordinator hot-fixed staged page, manager fixed generator durably + found/closed the remaining rewatch/context-menu vector (pause at commit dt-read), regression tests added (15/15 EXIT 0), staged+pack HTML regenerated w/ identical localStorage key (owner answers safe), committed fence-only. OWNER NEXT: open ~/Desktop/event_labels_20260715/START_HERE.html, label 300 clips (~75-120 min), export; ingest command in runs/lanes/owner_event_labels_20260715/INGEST_README.md | — | — | committed set in d0ce58bdd; pack on Desktop (untracked) | — | done | 2026-07-15 |

_(2026-07-16 WRAP-UP — **TRACK K -> TRACK A (+ Track G): THE ANCHOR CLASS YOU ARE STARVING FOR,
MEASURED.** Your arc failure is anchor sparsity (20/697s); pb.vision wins on anchor DENSITY from
trained event heads. But they have no 3D players and we do — so a paddle contact at the hitter's
hand is a MEASURED 3D ball anchor available with ZERO trained event heads. Manager probe on
Wolverine (runs/lanes/oneworld_impl_20260716/anchor_window_probe.py, EXIT 0, chance-baselined):
declared contacts' windowed closest ball-to-wrist approach median **1.167m vs 4.499m at chance**;
**6/24 within 0.50m (paddle band) vs 0/24 at chance**; 15/24 vs 6/24 within 1.20m. The signal is
REAL (~4x chance). It yields ZERO anchors today ONLY because co-location is evaluated at the
DECLARED event frame and those frames are mistimed (offsets -15..+13 frames, several at the
window edge; attribution 9/24 correct). UNLOCK (v2, specified DESIGN.md §8.5.2, not applied):
bounded closest-approach search inside the event window w/ per-clip chance-margin gate +
wrist-speed agreement -> proposed measured anchor + honest dt timing correction, raw events
immutable. Track A: this is a candidate anchor source for your solver that needs no event head.
Track G: your event head compounds it (better proposals -> more co-locations), and your contact
anchors + this class are complementary, not competing. VERIFIED=0; diagnostic only.)_

_(2026-07-16 WRAP-UP — **TRACK K -> TRACK H: THE FUSED WORLD EXISTS, RENDER IT.** Your
oneworld_render row is UNGATED: schema + module + CLIs committed **a54b7c451**
(docs/racketsport/one_world_v1_schema.json is the contract; `racketsport_one_world_v1`).
ARTIFACTS ON DISK (18M/27M, intentionally untracked per storage policy — render from these paths):
runs/lanes/oneworld_impl_20260716/wolverine/one_world_v1.json (THE watchable one: 300 frames,
4 players placed every rally frame, 4 paddles w/ display_pose_world every frame, ball 295
arc_measured + 5 physics_predicted + 0 absent, 28 typed events = 24 paddle_contact + 4
floor_bounce) and runs/lanes/oneworld_impl_20260716/demo/one_world_v1.json (697s pb.vision
partial: ball ray-projection chain only, 0 players — honest). Rebuild any time:
`.venv/bin/python scripts/racketsport/build_one_world_v1.py --run-dir <run> --out <path>`
(EXIT 0, deterministic, manager-verified byte-identical).
YOUR FIVE ASKS, ANSWERED: (1) per-entity per-frame numeric confidence + trust_band + provenance
w/ input refs = present on every player/ball/paddle/event; (2) explicit absence sentinels =
`frames[].missing[]` + summary.missing_counts (rows never dropped); (3) artifact-level trust_band
= present (preview); (4) provenance class question RULED: use the DISTINCT ball
`estimate_tier` {arc_measured|physics_predicted|ray_court_projection} + paddle `display_tier`
{resolved|unresolved_best_evidence|unresolved_legacy_wrist_proxy} — do NOT overload
physics_predicted; a tier>=2 sample must never render as measured; (5) manifest-relative URLs =
your call, artifact carries no URLs. DISPLAY HONESTY CONTRACT (binding): Wolverine paddles are
ALL `unresolved_legacy_wrist_proxy` — render them as visibly provisional (they are wrist-proxy
orientation, NOT solved 6DoF); the 5 physics_predicted ball frames must look predicted; and the
24 paddle_contact events carry hitter_id=null (22 unsupported / 2 too_close_to_call) — the
viewer must NOT draw a confident hitter attribution. VERIFIED=0; everything preview band.)_

_(2026-07-16 ~11:5x Track K ATTRIBUTION NOTE for concurrent wave closes: the repo-wide
test_scaffold_tool_index RED (JsonSchemaAssertionError $.tools[..].matching_schema pattern
'^docs/racketsport/.+_schema\.json$') is CAUSED BY the live oneworld_impl_20260716 lane — its
three new one_world docs schemas were named dot-style (one_world_v1.schema.json et al) vs the
house underscore pattern. FIX INJECTED as spec_addendum_A13 (session killed+resumed ~11:5x,
renames + SCHEMA_OVERRIDES + reference updates + REAL-unpiped-exit-code verification now
acceptance item A13; lane ordered to claim this failure as its own in wide-suite attribution,
never as concurrent-lane volatility). Track K manager re-verifies the scaffold test personally
before ruling; registration + green land in the same fence-only commit as the CLIs. Other
tracks: attribute this failure to Track K until that commit lands; do not fix it yourselves.)_

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

_(2026-07-16 night, Track F: CVAT person-label thread CLOSED WITH EVIDENCE — independent API sweep of
the live local CVAT (85 tasks / 7 projects, admin token from data/credentials/cvat_local.txt): the only
person-ish label (`player_box`, harvest project 2) has ZERO uses across all 6 tasks; all person-labeled
export zips in ~/Downloads + cvat_upload match the four protected clips exactly (frame+track counts);
0 new person boxes exist. Corroborates the owner ruling. SIDE FINDING for the ball-label owner: CVAT
tasks 7 and 11 show 1 live ball track each vs 49 in their exported harvest_review_20260707 zips —
possible live-instance annotation loss/rollback, not investigated (out of Track F scope). CVAT + Docker
left RUNNING (persistent owner dev instance at localhost:8080). Full inventory in the Track F session
transcript; no repo report file written by the sweep agent.)_

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

_(2026-07-16, placewire_20260716 lane close: Track I placement-trajectory refinement is wired as canonical stage order 145 immediately after `grounding_refine`, opt-in via `--placement-trajectory-refine` or the existing rev-13 best-stack enablement value (currently false); disabled runs typed-skip without writing or mutating payload artifacts, enabled runs emit only `placement_trajectory_refined.json` with preview/VERIFIED=0 provenance, covariance, robust weights, immutable-input hashes, and content-addressed reuse identity. Focused current-HEAD result: 197 passed / 2 failed EXIT 1, both storage-policy failures introduced by concurrent Track F commit d1b536b3f's unregistered 154,743,315-byte bundle; the same focused suite was 199/199 EXIT 0 before that mid-wide commit. Wide: 3920 passed / 35 failed / 24 skipped EXIT 1 in 2827.94s; all failures attributed outside placewire (1 stale rev-12 best-stack test pin vs landed rev 13, 25 pre-existing owner/CAL dirty-label/data-count failures, 8 managed-sandbox socket binds, 1 mid-run Track F bundle storage failure). `process_video.py --help`, scaffold index, dead-code audit, py_compile, and all 185 placewire/process/spine tests EXIT 0. BEST-STACK DELTA = "(b) consumes the existing rev-13 PENDING entry, no edit". Scoped wiring pass only; preview band, do_not_promote, VERIFIED=0.)_

_(2026-07-17 trkL_selection_20260717 CLOSE (Track L manager, wrap-up mode per coordinator): GHOST
DIAGNOSIS SOLVED with numbers — wolverine's 4 surviving "spectator" FPs under RF-DETR-L variant P are
NOT spectators and NOT detector output: they are 4 frames of a 42-frame SYNTHETIC interpolation bridge
(conf exactly 0.3500, `_interpolate_detection` cap) manufactured when global association stitched GT
player 1's tracklet (f0-44) to GT player 4's (f87+) and linearly bridged 10.4m across the court through
the net at 7.4 m/s; interpolated footpoints are ON-COURT by construction → structurally invisible to
ANY footpoint margin gate; production `max_gap_fill_frames: 48` + dt-scaled speed allowance (11.28m at
43 frames) let it through; provenance stamp (`conf_source="interpolated_endpoint_min_capped_0_35"`)
exists in player_id_repair but DIES at tracks.json export; the bridge also PADS cov4 (~32 fake frames
of the 0.7233) and causes the near-miss-rate breach (0.1244) AND the f58 switch. conf030's 16 FPs =
IDENTICAL bridge signature (forensically corrected: its "high-confidence spectators" interpretation is
wrong — threshold attacks were doomed by mechanism, vindicating the owner's selection-layer directive).
OSNet probe (production ckpt, CPU): stitch pair 0.448 = cross-identity band (0.42-0.55) vs legitimate
re-bind 0.304 vs within-identity 0.10-0.16 → open-set veto separates on this clip. Counterfactuals
(frozen scorer, fixed VM tracks, CPU diagnostic): stitch veto alone → spectFP 4→0, near-miss 0.0986
(clears gate), IDF1 0.8141; + slot re-bind → sw 1→0, IDF1 0.8519 ≥ YOLO baseline 0.8516, HOTA 0.8614;
honest cov4 floor 0.6167 (padding removed) — cov4 rebuild = identity-conditioned RAW-POOL recovery
(min_conf 0.0 pool already exported), NOT geometric synthesis. Burlington structurally untouched (6
synth frames, max run 3/1.14m). DELIVERED: runs/lanes/trkL_selection_20260717/{GHOST_DIAGNOSIS.md,
DESIGN_selection_layer.md} — full layers A (soft court presence, SIGMA 0.5m/EMA 2s) + B (4-slot OSNet
enrollment, open-set accept ≤0.35 / reject ≥0.42 / defer band, two-evidence-class stitch veto: ≥0.42
AND >2.5m-or-net-crossing-without-real-support) + C (interpolation ban beyond 12f/2.5m, provenance
survives export, pool recovery) w/ PRE-REGISTERED thresholds + frozen-card acceptance table (wolverine
FULL PASS = 0/0/0 FP-axes + near-miss ≤0.10 + IDF1 ≥0.8036 + cov4 ≥0.7233-via-recovery; cov4
0.6167-0.7233 all-else-green = PARTIAL coordinator ruling; burlington no-degradation) + invariants
(OFF ⇒ byte-identical; VM env-fidelity first; bridge-fixture unit test) + exact dispatch instruction
(build lane CPU Codex sol-high, 4 new files fenced, association modules READ-ONLY; then ONE ≤$2 A100
micro eval session). NO code/best_stack changes; no GPU spent; VERIFIED=0. Diagnosis harness +
counterfactuals + probe JSONs committed in lane dir (148K; extracted artifact trees deleted,
reproducible from trk_rfdetr_prod tarballs). Coordinate future impl lane w/
trk_rfdetr_integrate_20260717 (holds orchestrator/MANIFEST/best_stack when live; conf030 prereg FAILED
→ its branch = 2b ship-at-0.18).)_

_(2026-07-16 CLOSE — Track A FINAL (wrap-up; coordinator withdrew the MOVE-1 #3 envelope; NO GPU fired,
$0 spend this window). **ballarc_anchorfusion_20260716 RULED PARTIAL**: mechanism ADOPTED (typed
SoftSegmentBoundary split-only API in ball_arc_solver/chain, DEFAULT-OFF, byte-identical when unused
— Wolverine 5/5 + demo-slice 5/5; manager re-verified focused 80 passed EXIT 0), ALL THREE
pre-registered audio presets **REJECTED per the 0-violation kill rule**: conservative 53/371 fit
(18.77% in-rally coverage, 16 violations), balanced 85/361 (29.65%, 18), broad 123/367 (43.69%, 18)
vs frozen baseline 1/188 (~0%). Coverage rose with split density AND so did violations — splits alone
buy coverage, not physics. **TAXONOMY VERDICT: needs-typed-anchors** (52/52 classified: 42
anchor-semantics-structural, 9 split-landed-mid-flight, 1 bridged-direction-change, 0
weak-fit-passed-through) — review-only audio cannot supply contact semantics, flight resets, or
z=radius. Corroborated from two independent directions: Track K's fusion refused 24/24 Wolverine
contacts (ball 7.97m median from declared hitter's wrist), and G2's audio-density finding (~99.3% of
+-0.15s windows contain an onset) proves NO single-signal anchor class can work here.
**MOVE-1 #3 REOPEN CONDITIONS (3):**
 [1] ARC COVERAGE still unproven — no preset has achieved honest zero-violation coverage; do not fire
     a GPU run for a scorecard the arc cannot fill.
 [2] TRAINED ANCHORS **KILLED — DO NOT INGEST** runs/lanes/event_head_pretrain_20260716/anchors/*
     per G2's binding ANCHOR_VERDICT.md (zero-shot tennis->pickleball transfer failed: >=0.9 tail
     carries audio agreement 70.2% vs ~99.3% chance = zero discriminative information; manager's
     independent corroboration: the >=0.9 tail is 123/123 HIT with ZERO BOUNCE — no flight-reset
     class at all). Reopen needs (a) G2's SCALE_UP_SPEC (~68x label reach, $2.2-4.5) AND (b)
     pickleball fine-tune on the owner's clip-review labels to close the domain gap; THEN re-run
     anchor fusion (CPU, salvaged inputs, same frozen gate, same 0-violation bar).
 [3] CAL LEG **CLOSED** at the strongest class: owner's 15-pt review ->
     runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_metric15pt.json
     (source metric_15pt_reviewed; orchestrator ingests it, no correction task, calibration RAN;
     cross-checks the 2.61px line solve at ~3% focal agreement; metric_confidence low => in/out
     abstains). NOTE: manager MOVED this seed OUT of eval_clips/ball/ — committing it there had put
     the pb.vision competitor demo video into the CAL EVALUATION corpus (posture violation: R&D
     reference only, never GT) and broke court-corpus tests expecting 4 full-label clips. Corpus
     membership for pbvision is a COURT/CAL OWNER decision, not Track A's.
**KNOWN-RED, unresolved at close (honest):** the lane's wide run showed 37 failures; manager
attributed: 8 = known sandbox socket denials; 3 court_keypoint_review_server = transient port
contention with the owner's LIVE :8777 review server (re-ran clean, 14 passed); 2 storage-policy =
Track G2's event_head.bundle (not Track A); ~17 court_finding_technology_benchmark = caused by the
eval_clips corpus insertion described above (fixed by the move; discover_court_finding_samples now
returns the expected 5). test_overlapping_court_calibration_eval still asserted 6==5 at close —
NOT re-verified after the move (hard stop); next session must confirm the court family is green.
Withdrawn-unscored typed-anchor attempt retained as an audit trail only (PRESET_REGISTRATION.json
status REGISTERED_THEN_WITHDRAWN_UNSCORED; preset_typed_hi.jsonl = killed at 39 segments, NOT a
result). Envelope <=8h/<=$35: **WITHDRAWN/CLOSED — no lane or track may cite it.** Fleet: zero Track A
VMs, list-confirmed; $0 GPU this window.)_

_(2026-07-17 Track A close ADDENDUM — #1 INVESTIGATION FOR THE BALL PROGRAM (cheap, CPU-only, no lane):
Track L (c0004e405, runs/lanes/trkL_selection_20260717/GHOST_DIAGNOSIS.md) proved the wolverine
"spectator FPs" are the ASSOCIATION's own fabrication — a 42-frame linear bridge (frames 45-86, conf
exactly 0.35) marching a footpoint 10.4m across the court THROUGH THE NET at 7.4 m/s, with
`interpolated: true` stamped internally (player_id_repair.py:550) but STRIPPED at tracks.json export,
so fabricated positions reach downstream indistinguishable from measured. Track K's fused-world
refusal cites **frame 78** (declared hitter player 1; ball 11.17m from their wrist, actually inside
player 4's box) — frame 78 sits INSIDE that fabricated bridge window, and the bridge marches from
player 1 toward player 4. HYPOTHESIS: a meaningful share of K's 24 contact refusals — and possibly
arc/hitter disagreement generally — is caused by FABRICATED PLAYER POSITIONS, not ball error. TEST:
cross-reference K's 24 refusal frames against L's bridge frame ranges (both artifacts on disk,
CPU-only). If it holds, the ball-3D program REORDERS: fix the fabrication (and restore the stripped
interpolated provenance) FIRST, then judge arc/contact quality — my taxonomy's 42/52
anchor-semantics-structural verdict may be partly measuring a TRK defect, not a ball defect.)_
=======
| p63_reference_ranges_20260707 | codex | bg task (Fable succession session) | codex exec resume <session_id from report.json> | ONLY NEW FILES: docs/racketsport/reference_ranges_{schema,v0}.json, scripts/racketsport/validate_reference_ranges.py, tests/racketsport/test_reference_ranges.py (+scaffold-index line) | none (no GPU) | ~1-2h from 2026-07-07 dispatch | 2026-07-07 by Fable final session |
| live_offline_docs_20260707 | codex | bg task (live-vs-offline session 2026-07-07) | codex exec resume <session_id from report.json> | CAPABILITIES.md, TIER_MAP.md, NORTH_STAR_ROADMAP.md, EDGE_PLAYBOOK.md, MASTER_PLAN.md, ios/README.md, tests/racketsport/test_truthful_capabilities.py, BUILD_CHECKLIST.md (append) | none | ~1h from 2026-07-07 dispatch | 2026-07-07 live-tier manager session |
| live_tier_blueprint_20260707 | codex | detached nohup (live-tier session 2026-07-07) | codex exec resume <session_id from report.json> | TECH_BLUEPRINTS.md (additive pillar) | none | ~1h | 2026-07-07 live-tier manager session |
| runbook_doctor_20260707 | codex | detached nohup (live-tier session 2026-07-07) | codex exec resume <session_id from report.json> | RUNBOOK.md, NEW scripts/racketsport/doctor.py, NEW tests/racketsport/test_doctor.py | none | ~1-2h | 2026-07-07 live-tier manager session |
>>>>>>> Stashed changes
