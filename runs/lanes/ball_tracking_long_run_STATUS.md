# BALL Tracking Long-Run Manager — STATUS (mission: best single-camera video-to-3D ball trajectory)

## OWNER RETARGET 2026-07-04 ~01:35: optimize the FULL CHAIN, not isolated detector metrics.
Chain: candidate 2D detections -> fusion/selection -> confidence gates -> hidden-FP rejection ->
bounce/contact/event discovery -> calibrated physics lift -> plausibility checks -> trust-banded replay.
Human-reviewed bounces = recovery/debug/internal-val only; PRODUCTION requires automatic anchors/events
or another fail-closed segmentation+lift path.

## CHAIN WORK ORDER (manager ruling, dependency-ordered, evidence-based)
- W1 top-K candidate persistence: **PASS, manager-ACCEPTED** (runs/lanes/ball_topk_candidates_20260704/).
  racketsport_ball_candidates schema + loader; wasb/tracknet adapters emit top-K sidecars opt-in
  (--emit-candidates, top-k 5 default, NMS 10px for tracknet); default outputs byte-identical (proven);
  499/0 blast radius. BlurBall has no candidate path (third_party, out of scope).
  -> CANDIDATE-EMISSION lane: **PASS, manager-ACCEPTED** (runs/lanes/ball_candidates_emission_20260704/).
  8/8 sidecars, primary invariance 7/8 byte-identical (indoor TNv3: conf-only GPU float drift, 32 frames,
  none near cutoff — functionally invariant). HEADROOM: any-of-top-5 hit@20 WASB 0.7825 (+12.1pp vs
  primary), TNv3 0.8448 (+22.2pp) — single-detector top-5 nearly matches the old 3-detector oracle
  (0.8594). UNION of both detectors' top-5: 0.8793 (burl 0.8935 / wolv 0.8502) — EXCEEDS the old
  3-detector primary oracle. Cost $0.98; emission is runtime-free (+/-7s). Notes for W3: WASB blob
  scores often tie at 1.0 (top-candidate ordering is xy tie-break, not confidence); TNv3 candidates are
  pre-InpaintNet (3-4px median offset vs primary); pipeline is not bit-deterministic run-to-run (conf
  drift 1e-3 scale) — never gate W3 on byte-exact reproduction. BlurBall candidates unbuilt (would raise
  ceiling further; possible W3.5).
- W2 automatic anchors: PROVEN 2026-07-04 (runs/lanes/ball_tracking_anchor_diag_20260704/).
  ROOT CAUSE: _select_event_subset seeds ONLY from mandatory anchors (rally endpoints / human bounces /
  immovable); contacts + solver bounces are always optional -> without human bounces: 0 segments, and
  status stays "ran" (silent footgun). Contact anchors themselves are fine.
  LABEL-FREE AUTO-BOUNCES (cusp detection on fused track, geometry only) fed as seeding anchors:
  burlington status ran, 478/600 coverage, LOO 0.0234m (n=276), 12.5% violations — BEATS old
  human-bounce baseline (0.0313). Reprojected median 8.0px, F1 0.601, teleports 4. Wolverine improves
  but honestly self-kills (50% viol): 15-frame 2D hole exactly at its main bounce (f103) — fix =
  gap ballistic-intersection candidates (+ W1 top-K may recover the 2D hole itself).
  Human-agreement sanity: 4/6 historical bounce frames matched within ±2 frames.
  -> W2.5 CODEX LANE: **PASS, manager-ACCEPTED** (runs/lanes/ball_auto_anchor_promotion_20260704/).
  Landed (all red->green proven, 486/0 blast radius): auto_bounce_candidate anchor class w/ honest
  provenance + separate report counts; provenance firewall (fake human_reviewed payloads raise);
  threed/racketsport/ball_bounce_candidates.py + scripts/racketsport/propose_ball_bounce_candidates.py
  (cusp + gap_ballistic_intersection); solve_ball_arcs --auto-bounce-candidates; degenerate_zero_segments
  status. LEGITIMATE-PATH RESULTS: burlington ran 508/600 LOO 0.0226 viol 0.125; wolverine ran 290/300
  LOO 0.0379 viol 0.200 (AT the gate edge — fragile) via gap candidate f104. E1 fps rescore done
  (burlington fused cd033 median 10.79px corrected). NOTE: "reprojection hidden-FP 1.0/0.41" in its
  honest-issues is largely a SCORING-VIEW artifact (arc interpolates hidden spans by design) — W4
  band-aware scorecard separates detection-grade (anchored_measured) from render-only views.
  E1 CORRECTION: E1's reprojection wrapper hardcoded fps=30 (burlington is 59.94) — E1 burlington px
  rows invalid (corrected fused median 10.79px, not 254.8). Cd-second-order ruling unaffected
  (relative). "Burlington floor calibration-limited" claim UNDER RE-VERIFICATION via the rescore.
- W3 round 1: PARTIAL — infrastructure ACCEPTED, association policy iterating
  (runs/lanes/ball_w3_multihyp_association_20260704/, 502 tests green, no gate loosening).
  3D HEALTH WAY UP: burl LOO 0.0142 (was 0.0226); wolv coverage 299/300, LOO 0.0221, sanity 0.100
  (fragility gone). 2D REGRESSED: measured P 0.837 (was 0.898), hFP 0.192 (was 0.130), product F1
  0.765 < fused 0.785. MANAGER DIAGNOSIS: associator trades pixel-exactness for arc-consistency
  (TNv3 candidates pre-InpaintNet ~3-4px bias; WASB tie noise) and TNv3's always-full top-5 lets
  distractors near the interpolated arc self-confirm to measured status during hidden spans.
  W3.1 RESULT: PARTIAL, near-miss (runs/lanes/ball_w3_1_rescue_association_20260704/, 495 blast green).
  rescue_tn05: measured P 0.8799 / hFP 0.1507 / R 0.5053 (missed bars by ~1 frame each; recall +4.7pp
  over W2.5); PRODUCT VIEW BEATS FUSED FOR THE FIRST TIME (rescue_tn02 F1 0.7921, tn05 0.7877 vs
  0.7852) but product hFP 0.37-0.40 (inherited from fused fallback 0.349 — rescue can't remove kept
  frames). Rescue LOO a bit above W3-free (expected: keeps noisier fused pts).
  MANAGER RULING: dual-artifact chain — 3D replay artifact = W3 FREE association (best 3D: LOO
  0.0142/0.0221); 2D product track = rescue_tn05 + NEW arc-veto on weakly-supported fallback frames
  far from the solved trajectory (the physics veto, bounded, with a >1pp-recall-drop kill rule per the
  banned-veto precedent).
  W3.2 RESULT: PARTIAL on stretch, ACCEPTED as convergence (artifacts: runs/lanes/ball_w3_2_arc_veto_chain_20260704/,
  508 tests green). veto_v40_weak: product F1 0.7871 / hFP 0.3562 / recall drop 0.27pp. Aggressive veto
  reaches hFP 0.20-0.23 but violates the pre-declared >1pp recall kill rule — stays dead (no post-hoc
  relitigation). Dual runner + heldout guard + manifests work on both internal-val clips.
  HELD-OUT SHOT EXECUTED PER PROTOCOL (ledger rows 22->23; runs/lanes/ball_heldout_chain_run_20260704/).
  **RESULT: BAR NOT MET.** Outdoor product F1@20 0.6969 < 0.7248 -> WASB-tennis zero-shot REMAINS the
  standing best BALL M1 candidate. Honest texture: chain BEATS the anchor on precision (0.878 vs 0.861),
  hidden-FP (0.021 vs 0.063, 3x cleaner), P95 (34.3 vs 42.2px), teleports (1 vs 7); loses ONLY recall
  (0.578 vs 0.626) — the 4th internal->held-out inversion (consensus fusion tuned on BlurBall-favoring
  internal clips under-recalls on WASB-dominant Outdoor). Indoor first measurement: product F1 0.680 /
  hFP 0.225; both solvers HONESTLY SELF-KILLED at the frozen sanity gate — fail-closed machinery passed
  its first held-out test. Outdoor 3D: status ran, sanity 0.167, LOO 0.0147m, 7 auto anchors, 0 contact.
  FOLLOW-UP FIX LANDED: **PASS, manager-ACCEPTED** (runs/lanes/ball_killflag_product_degrade_20260704/):
  product views degrade to fused-only (veto disabled) when solver status != ran, product_view_mode +
  kill provenance in manifests, red/green regression proof, internal-val behavior proven identical,
  500 blast-radius green.

## FINAL RANKED PROMOTE/DEFER/KILL (owner-required output, 2026-07-04)
- WASB-tennis zero-shot 0.7248: STANDING BEST M1 candidate (unchanged). M1 gate still unpassed; VERIFIED=0.
- Fused+arc dual-artifact CHAIN: DEFER — held-out F1 miss; dominates anchor on every cleanliness metric;
  recall-limited by internal-val-overfit consensus. Revisit only with in-domain data or a label-free
  recall fix. (Its measured-grade view — P 0.88-0.89, 0 teleports — is the cleanest track this project
  has produced; review/replay value stands.)
- Auto-bounce anchors (W2.5) + candidate sidecars (W1) + association modes (W3.x) + honesty fixes:
  KEEP as landed repo infrastructure (all uncommitted; commits deferred per owner joint-commit rule).
- TOTNet zero-shot: KILL (0.218). TrackNetV4 / PB-MAT: BLOCKED-NO-WEIGHTS. Point-tracker gap-fill,
  WASB-FPFilter, VballNet, Cd-0.60, BlurBall-blur-prior: DEFER (evidence/priority). Rally-gating for
  hidden-FP: NULL on current eval data (no dead time).

## NEXT GPU/TRAINING JOB (exact, owner-required)
No GPU job changes the blockers. The next real move is DATA: owner-captured clips -> CVAT labels ->
WASB fine-tune on owner data (never eval clips) -> fresh prereg row. Zero-shot recombination has hit
its measured wall (recall 0.578-0.626 on Outdoor across all honest configs).

## REOPENED BY OWNER 2026-07-05 ~04:0x: watched the 3 worlds — ball "barely shown, badly placed" vs
pb.vision. MANAGER INVESTIGATION (pbv-public/insights schema, insights_shot.schema.json): PB Vision
renders a PARAMETRIC ARC through 3 anchors per shot (start/peak/end + speed/height stats) — no
per-frame tracked positions at all; their confidence = start/end event confidence. Our per-frame
fail-closed dot rendering is the perceptual gap, not detection ML. RULING: presentation-layer rebuild —
P2 RESULT: code PASS / screenshots sandbox-blocked (expected; manager verifies in Phase D).
Landed: ball_arc_render.json dense artifact + schema + manifest URL; viewer trail v2 (dense curves,
interpolated sphere); ?view=courtmap SVG panel; verifier --view courtmap; 724 py + 172 web tests green;
uncommitted pending Phase-A merge + visual gate. Session id for cheap resume: 019f3407-6ba5....

OWNER-APPROVED PLAN (2026-07-05): /Users/arnavchokshi/.claude/plans/okay-this-direction-looks-gentle-barto.md
— P3 anchor-first solver (Phase A CODEX LANE — round 1 PARTIAL, manager-ruled + resumed to finish:
BVP shooting + endpoint-delta corridors + fit-validity gates -> fit_bvp_fallback + court-volume sanity
+ suppression policy v2 + contact sigma hardening + render-path closure + wolverine-seg6 fixture +
3-clip acceptance). Phase B (anchor coverage recovery: bounce-bounce discovery relaxation + wolverine
contact-truncation diagnosis) queued after A. Phase D: manager browser verify all 3 clips incl.
wolverine t=6.6s ball-near-net check, then owner review.

## ============ PRIOR FINISH LINE (superseded by reopen above) 2026-07-05 ~03:0x ============
3-CLIP OWNER VERIFICATION COMPLETE (all manager-browser-verified, screenshots on disk):
- burlington: ok, 22.9 FPS, Ball KPI 436/600 measured · 66 predicted · 98 hidden; ball+trail+skeletons
  clean at t6s (runs/lanes/ball_v2_viewer_polish_20260705/manager_verify_burlington/)
- wolverine: ok, 26.3 FPS, 214/300 measured · 76 predicted · 10 hidden; complete E2E 161s
- outdoor: ok, 21.1 FPS, 360/1151 measured · 92 predicted · 699 hidden (honest hidden spans; ball+court
  world, no body by design) — solver ran/no-kills after V3 config alignment
  (runs/lanes/ball_final_outdoor_20260705/manager_verify/)
DEFAULT-PIPELINE STATE: ball_arc stage default-on (frozen row-22 config, fixture-guarded), candidate
emission default-on (runtime-free), flight-sanity parabolic gate enforced, fail-closed everywhere,
viewer trail + honesty KPI + impact layers live. Future user uploads get ALL of this automatically.
OUTSTANDING (non-blocking, handed off): 16 ball tests broken by the OTHER session's in-flight
orchestrator.py wall_seconds edit (flagged twice in BUILD_CHECKLIST); ALL ball work uncommitted per
owner joint-commit rule (~25 files; hashes in ledger row 22 + lane reports).

## P4 VERIFIED — DELIVERED TO OWNER 2026-07-05 (committed e5789028, manager browser-verified)
ALL 3 CLIPS RENDER HONESTLY NOW:
- wolverine: status ran, 214/300 measured ball, ball in-court+airborne (was: parking-lot then nothing).
- burlington: ran, 471/600 measured. outdoor: ran, ball+court.
- COURT MAP VIEW (P2+P4, PB-Vision-style top-down) WORKS: shot-path lines, bounce dots, P3/P4 + ball
  markers. This is the view that makes PB Vision placement look good — now ours.
- Ball never off-court; honest measured/predicted/hidden KPI; continuous trail.
Screenshots: runs/lanes/ball_p4_render_fix_20260706/{wolverine,burlington}/mgr_verify + wolverine/mgr_courtmap.
REMAINING HONEST LIMIT (owner decision pending): confident-AT-NET on occluded/no-detection stretches
(e.g. some of wolverine 6.66s) needs a CONTACT anchor = SAM-3D 70-joint skeleton = GPU body run (cached
skeleton is RTMW 65-joint, wrist-cue-blocked). Options for owner: (a) GPU SAM-3D body run per clip for
contact anchors, (b) accept current in-court low-confidence containment, (c) owner captures for detection.

## FINAL-VERIFY FINDINGS + P4 FIX LANE (2026-07-05)
Final-verify surfaced 3 render bugs (why the viewer looked bad):
1. CRITICAL: my Phase A commit f1865300 bumped ball_arc_solver SCHEMA_VERSION 1->2 but the web viewer's
   ball_track_arc_solved consumer still requires ==1 -> ball trail fails to load for ALL clips. My regression.
2. Wolverine self-kills (experimental_off) -> renders NOTHING: whole-artifact kill is obsolete vs
   per-segment fallback; seg5/6 have net-clearance/speed violations that don't trigger demotion so they
   stay confident-implausible AND trip the kill; denominator bug (violations/total vs violations/eligible).
3. Court-map view empty (no markers) — P2 CourtMapPanel data-flow bug.
Fresh-contact facts: ball_inflections NOW regenerates at correct 30fps past frame 148 (candidate at
frame 201/6.7s = the hit!) BUT contact_windows=0 because wrist cues need a 70-joint SAM-3D skeleton;
cached skeleton is RTMW3D (65-joint, pre-SAM-3D-migration). So confident-at-net 6.66s needs a GPU SAM-3D
body run (contact anchor) — a STRATEGIC decision after P4. Burlington 466/600 + outdoor 346/1151 measured
(good). -> P4 CODEX LANE RUNNING (runs/lanes/ball_p4_render_unblock_20260705/): schema unblock +
per-segment containment replacing whole-artifact kill (wolverine renders) + court-map markers. Then
manager browser-verify + owner delivery + SAM-3D decision.

## FULL STACK COMMITTED (2026-07-05): Phase A f1865300 + Phase B (winddown 14b68c68 + tests d96790a6)
+ P2 continuous-trail/court-map (winddown). Working tree clean, all pushed.
KEY DIAGNOSIS: wolverine 6.66s low-confidence = STALE contact_windows (60fps time_s vs 30fps track)
truncated contacts at frame 148 -> no contact anchor to pin the hit. Ball IS detected there (~29/30f
bin) so RECOVERY IS POSSIBLE with fresh contacts. -> FINAL-VERIFY LANE RUNNING
(runs/lanes/ball_final_verify_20260705/, Sonnet): regenerate 3 clips with FRESH contacts (no stale
reuse), report whether 6.66s recovers to confident-near-net vs contained, browser-verify all 3 +
court-map with screenshots for manager inspection. Then manager rules + owner delivery.

## OWNER FINISH-LINE ORDER 2026-07-05 ~00:5x: clean up stale agents (DONE — T5 MPS lane + orphan
process killed; its value expired when T6 settled training); then: verify on the 3 CVAT videos that the
implemented 3D-viewer ball looks as good as we can make it, runs fast/efficiently/accurately, and runs
BY DEFAULT in the pipeline -> then END THE RUNNER. Otherwise keep driving to that goal.
F1 3-CLIP RESULTS: burlington ran (433/600 measured ball, LOO 0.0226, viewer ok 0 errors);
wolverine COMPLETE E2E 161s (207/300 measured, 27.5FPS, viewer ok); outdoor self-kills under the
DEFAULT chain config with either track (fused: sanity 0.333; wasb-single: 0.429) — ROOT CAUSE
(F1 diagnostics): default chain diverges from the FROZEN row-22 config (no candidate sidecars;
discovery/subset-selection ON vs frozen OFF). MANAGER RULING: align the pipeline default to the frozen
validated config + default-on candidate emission (runtime-free) -> V3 CODEX LANE RUNNING
(runs/lanes/ball_v3_chain_config_align_20260705/). V2 POLISH: CODE PASS (Codex sandbox can't bind localhost — expected) + MANAGER BROWSER VERIFICATION
DONE: burlington ok=true, FPS 5.3->22.9 (4x, splatter gone), Ball KPI "436/600 measured · 66 predicted
· 98 hidden", calm notices, mid-rally t3s/t6s screenshots show BALL + TRAIL + skeletons cleanly
(runs/lanes/ball_v2_viewer_polish_20260705/manager_verify_burlington/). wolverine ok=true, 26.3 FPS,
KPI "214/300 measured · 76 predicted · 10 hidden".
Manager screenshot review done: wolverine viewer good; burlington had point-cloud splatter + 5.3FPS +
"Ball coverage n/a" -> V2 scope.
EXECUTION: I1 CODEX LANE RUNNING (runs/lanes/ball_i1_default_integration_20260705/) — default ball_arc
stage in process_video (placement session quiet ~10h -> files claimable, extend-never-revert), apply
both deferred patches (virtual_world arc-status gate; viewer App wiring), manifest exposure, full
suites, wolverine smoke. THEN: 3-clip E2E runs (burlington/wolverine/outdoor — labels untouched) +
manager browser verification of the ball trail + speed/accuracy readout -> END.

## OWNER END-STATE DIRECTIVE 2026-07-04 ~15:1x (BINDING DEFINITION OF DONE)
1. Best-achievable ball finding/prediction (training push, in flight).
2. WIRED INTO THE PRODUCTION PIPELINE: future user uploads + our testing automatically run the best
   version (process_video.py ball stage -> auto-bounce candidates -> arc solver dual artifacts ->
   product view; the deferred virtual_world arc-status patch must land as part of this).
3. PHYSICS SANITY IN RENDER: no mid-air direction changes; parabolic flight segments — new
   ball_flight_sanity render gate (per airborne segment: one apex max, no horizontal reversals;
   failing segments render as predicted/low-confidence, never as measured).
4. 3D VIEWER: visible pickleball + trail; band-styled honesty UI (measured=solid, predicted/
   interpolated=dashed-translucent ghost, hidden=gap); impact markers for floor bounce / paddle
   contact / net; legend. Owner cares most about: see the ball, its path/trail, landings, impacts,
   and KNOW when we're predicting vs seeing.
PHASES: D data (T1a2 downloading w/ owner key now) -> T train schedule matrix (A100) -> S select on
non-held-out evidence -> H one prereg'd held-out run per survivor -> I pipeline integration + parabolic
gate -> V viewer trail UI -> E E2E verify on the user-upload path. I+V build against the CURRENT chain
output in parallel with training (checkpoint swap later, not a rebuild).
ROBOFLOW KEY: received, stored gitignored (~/.roboflow_key), masked in all logs.

## OWNER RULING 2026-07-04 ~14:4x: ALL WORK IS PRIVATE/INTERNAL-USE ONLY — never publicly shared.
License gate widened accordingly: research-only / NC / GPL / unclear-license datasets+tools are
train-eligible for internal R&D (license recorded verbatim per source; only truly inaccessible assets
stay blocked). Relayed live to T1a/T1b. Held-out discipline unaffected. Footnote for a future
commercialization decision: NC-trained weights would need license revisit before shipping.

## OWNER RE-STEER 2026-07-04 ~14:15 (/goal): TRAINING-BASED BALL CANDIDATES
Menu: runs/lanes/ball_training_candidate_handoff_20260704/BALL_TRAINING_OPTIONS.md. Goal: >=1 honestly
scored training candidate beating the current best baseline, OR an evidence report proving current
data/model paths can't yet. Heartbeats RETIRED per owner (goal hook drives the loop now; one stale
queued wakeup will fire once and be absorbed).
MANAGER RULING ON THE MENU: primary path = WASB fine-tune with a real data mix (row-14's 0.4618 failure
was owner-CVAT-only/754-positives/underpowered — the missing ingredient is data volume; Roboflow
pickleball sets are the untapped source). Secondary: TrackNetV4 weight re-check (cheap, doc requests),
V5 SDK = scaffold-only reality check. Everything gated on the data/provenance lane.
WAVE T1 DISPATCHED (parallel):
- T1a DATA-PROVENANCE: DONE (runs/lanes/ball_t1a_data_provenance_20260704/data_manifest.json).
  TRAIN-ELIGIBLE NOW: 754 owner positives + 146 hidden (B/W only — the pool that already failed at
  0.4618). ROBOFLOW RANKED (all verified ones CC BY 4.0): racket-ai/pickleball-iiv9m 2441 ball inst;
  liberin/pickleball-vision 6202 imgs; pickleball-seg 1035 masks; +2 small. REJECTED: pickleball-5pshr
  (suspected mislabeled tennis). Leakage: 0 pHash collisions (sample-level; full dedup pending downloads).
  **BLOCKER: Roboflow API key required (site bot-walled; API 401 reproduced). OWNER ASK #1.**
  IMG_1605 unreviewed -> ineligible. VM staging dir ready (87G free).
- T1a2 ROBOFLOW DOWNLOAD: **PASS** (runs/lanes/ball_t1a2_roboflow_download_20260704/data_manifest_v2.json).
  FINAL CLEAN POOL: 10,656 raw imgs / 9,268 ball instances, 6 sources, all CC BY 4.0 API-verified,
  FULL-CORPUS eval-clip leakage 0/41,866 imgs. 5pshr CONFIRMED contaminated (padel/squash + gamechangerv1
  fork) — rejected. Cross-set dupes mapped (drop nigh's 156 liberin dupes). Corpus on VM
  /home/arnavchokshi/ball_training_data/extracted/ (2.4GB). Manager eyeballed sample overlay (genuine
  PPA broadcast). DATA BLOCKER BROKEN: ~13x the data behind the 0.4618 failure.
  MANAGER DATA RULINGS: raw basis (own augmentation only); video-level splits (burlington->train,
  wolverine->val primary selection domain + per-set held-out video secondary); per-source sequence
  reconstruction (stills = pseudo-sequences, documented); negatives only from sample-verified
  genuinely-ball-free sources.
- T3-ASSEMBLY: **DONE** (runs/lanes/ball_t3_assembly_20260704/; VM assembled_v1/ 5.6GB, checksummed).
  8,179 train / 452 val frames, Roboflow-only (owner clips excluded by design, guard-verified 0 refs),
  poisoned negatives dropped w/ visual proof (iiv9m + seg confirmed, liberin conservative), cross-dedup
  applied, sequences reconstructed (200 real-sequence rallies -> 5,442 stock TrackNet train windows;
  stills dead weight for temporal trainers — accepted, stock configs unchanged). Loader-proven both
  splits. CORRECTION: iiv9m raw ball count 1,767 (manifest's 2,441 doesn't reconcile). NOTE: owner's
  Roboflow key found REVOKED mid-lane (post-download; matters only for future re-pulls).
## !! OWNER ORDER 2026-07-04 ~15:4x: CPU-ONLY UNTIL JOINT FINAL RERUNS FINISH — A100 LOCK OFF-LIMITS !!
T4 PAUSED (report filed: runs/lanes/ball_t4_train_20260704/PAUSED_STATE.md). CORRECTION: ZERO epochs ran —
S1 waited its whole life on the exclusive lock behind the placement loop's back-to-back shared-lock eval
jobs. INFRA FINDING: exclusive flock waiters STARVE behind continuous shared holders on this host — when
resuming, coordinate a JOINT-quiet window (or add lock fairness later). FULLY STAGED FOR RESUME (2 cmds):
both datasets on VM (assembled_v1 + wasb_v1/blurball layout, guard-clean 0 owner refs), S1 bootstrap +
launch_s2.sh ready, eval-template + scorecard builder ready. Resume = PAUSED_STATE.md commands after JOINT
confirms done.
Meanwhile (local CPU only):
- V1 VIEWER TRAIL: **PASS** (runs/lanes/ball_v1_viewer_trail_20260704/): BallTrailLayer + ImpactMarkers +
  BallHonestyHud + pure ballTrail logic, 6 new files, tests green (141 pass; 2 pre-existing failures =
  tests referencing purge-deleted run fixtures), deferred_app_wiring.patch verified apply-clean (waits
  on placement session landing App.tsx). Browser verify = manager, post-wiring.
- P1 FLIGHT-SANITY GATE: **PASS, manager-ACCEPTED** (runs/lanes/ball_p1_flight_sanity_gate_20260704/).
  Owner req #3 DONE: ball_flight_sanity.py (one-apex, heading-reversal, speed-continuity) wired into
  run_ball_chain as fail-closed band demotion; REAL-DATA effect: burlington solver_a 77 frames demoted
  (the render excursion class), product 0; wolverine 62 frames/1 seg demoted in product view -> renders
  as predicted styling. Web suite 143/0 (deleted fixtures regenerated). 1 foreign blast failure:
  monitor_process_resources.py scaffold test = placement session's missing reference test (flagged to
  them via BUILD_CHECKLIST). Codex session_id captured for cheap resume: 019f2f51-e4b9-7571-a260-23260d33af9d.

## END-STATE SCOREBOARD (owner directive, 2026-07-04 ~16:15)
1. Best ball finding: training FULLY STAGED, blocked ONLY on owner GPU clearance (JOINT reruns).
2. Pipeline wiring: run_ball_chain = production chain entry w/ flight sanity inside; process_video +
   virtual_world hookups = deferred patches pending placement-session landing.
3. Parabolic motion: DONE + enforced on real clips.
4. Viewer trail/honesty UI: components DONE + tested; deferred_app_wiring.patch apply-clean; browser
   verification = manager, after placement lands.
BLOCKED-ON: (a) owner GPU all-clear -> resume T4 (2 commands) -> scorecard -> held-out decision;
(b) placement session landing -> apply viewer+world patches -> manager browser verify -> E2E.
T5 LOCAL-MPS TRAINING DISPATCHED (runs/lanes/ball_t5_local_mps_train_20260704/): ADC/REST credential
probe DEAD (all GCP auth routes exhausted, evidence in transcript) -> last autonomous lever = local Mac
MPS via run-local patched TrackNetV3 copy (workers0-copy precedent; no vendored edits; VM1 contact =
one bwlimit rsync only). Tiered epoch schedule by measured epoch-1 wall time; pre-declared wolverine
grid; bar = zero-shot TNv3 wolverine 0.626 + 0.02. Outcome either a real scored trained candidate or a
measured-attempt fact for the evidence report.
EVIDENCE REPORT FILED (goal deliverable, pending-training variant):
runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md — no trained candidate yet, reduced to exactly
2 owner-gated blockers w/ unblock commands. Second-GPU route probed and hard-blocked (gcloud token
expired AGAIN ~16:2x; VM1 service account lacks compute.*).

## OWNER UNBLOCK 2026-07-04 ~17:0x: gcloud auth restored + SECOND GPU AUTHORIZED ($20 hard budget,
speed > cost, delete after use). T6 COMPLETE: SCORED-CANDIDATES (runs/lanes/ball_t6_gpu2_train_20260704/; spot A100 7.46h ~$9.70,
VM DELETED + verified; S2 needed 4 real vendored-recipe bug fixes, VM-side only, documented).
MANAGER RULINGS:
- S1 TrackNetV3 fine-tune: wolverine 0.6531 (bar 0.646: technically passed, +0.007) BUT burlington
  reference CRATERED 0.672 -> 0.5018 (-17pt) = the classic narrow-domain trade, caught exactly by the
  reserved reference clip. RULING: DEFER — bar was necessary-not-sufficient; NOT spending the one
  held-out shot on a reference-regressing candidate. Checkpoint preserved:
  models/checkpoints/candidates_t6/tracknetv3_wolverine_winner_best.pt (sha b34226a8...).
- S2 WASB fine-tune RESOLVED by T7 (runs/lanes/ball_t7_se_rescore_20260705/): 0.1479 was an eval
  ARTIFACT (strict=False amputated 36 trained SE keys); real wolverine = 0.6879 (still -4.7pt vs
  zero-shot 0.735); burlington 0.0018 REAL in both paths (static-distractor lock). RULING: FAIL.
- CAMPAIGN VERDICT: both architectures' fine-tunes on the public corpus DEGRADE generalization.
  **FINAL EVIDENCE REPORT (goal deliverable) at runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md**
  — measured proof that available data/model paths don't beat 0.7248 yet; binding constraint =
  owner-domain video-sequence data. T5 MPS row = pending addendum, non-gating.
- Durable fact: blurball<->WASB loaders strict-incompatible (SE keys); never cross-eval without
  key-diff checks.
FYI standing: owner's Roboflow key revoked post-download (matters only for re-pulls).

- T4-TRAIN CAMPAIGN (PAUSED BY OWNER GPU ORDER) -> runs/lanes/ball_t4_train_20260704/: S1 TrackNet (badminton
  seed, stock 30ep) then S2 WASB-via-blurball (tennis seed, LR 3e-4) serial under train lock; then
  pre-declared wolverine selection grids (TNv3 thr {.3-.7}, WASB {.3,.5,.7}) + single burlington
  reference row per winner. Bars: beat matching-arch wolverine zero-shot (TNv3 0.626 / WASB 0.735)
  by >= +0.02 to earn a held-out prereg.
- T2-SHAKEOUT: **PIPELINE-PROVEN, $0.18** (runs/lanes/ball_t2_shakeout_20260704/PIPELINE_READY.md =
  verified parameterized command sequence). Trainer's checkpoint selection honest (pinned epoch 0 on
  synthetic-data regression). GOVERNANCE DISCOVERY: eval_guard = Burlington/Wolverine NEVER gradient-
  train, no override (by design post rows 14/15). MANAGER RULING: respect it — TRAIN SPLIT IS
  ROBOFLOW-ONLY; owner clips eval-only (wolverine = post-training selection via inference+benchmark,
  burlington = untouched internal reference). Cleaner domain-transfer science; sidesteps the guard's
  role-awareness gap (no code fix needed for training phase). T3 re-steered mid-flight accordingly.
  Noted future fixes: role-aware guard opt-in in 2 CLIs; prepare_tracknetv3_finetune_dataset.py dead code.
- T1b TRACKNET-RECHECK: DONE. RULINGS: (1) TrackNetV4 DEAD-AS-SHORTCUT — issue-#4 Drive weights exist
  but the motion-fusion checkpoint fails deserialization (broken get_config, upstream issue #6 open);
  the loadable one is architecturally TrackNetV2 baseline, no edge over WASB 0.7248. (2) TrackNetV5 SDK:
  no weights anywhere, from-our-data-only, Gaussian-heatmap preprocess to unpublished spec — DEFER unless
  T1a finds big data. (3) CORRECTION: upstream WASB-SBDT has NO training code ("TBA") — row 14 trained
  via the third_party/blurball fork (src/main.py --config-name=train_blur); the dataset glue SURVIVES on
  the VM (build_pickleball_wasb_dataset.py, recovered locally to
  runs/lanes/ball_t1b_tracknet_recheck_20260704/build_pickleball_wasb_dataset_VM_RECOVERED.py) —
  contiguous time-segment split, explicit --pair scoping.
- T1d GLUE: PARTIAL-ACCEPTED (converter + additive pickleball dataset classes + CLI + tests landed,
  765 filtered tests green; runs/lanes/ball_t1d_wasb_dataset_glue_20260704/). Gap = vendored registry
  wiring (my no-vendored-edits rule was too strict). -> T1d2 CODEX LANE RUNNING
  T1d2 RESULT: **PASS** — registries wired (marked minimal edits), factory-path CPU loader proofs both
  trainers, converter emits blurball Label.csv layout matching the recovered VM script conventions,
  TRAIN_LAUNCH_README.md VM-ready, 768 tests green. TRAINING IS NOW GATED ONLY ON T3 DATA ASSEMBLY.
  (VM needs omegaconf/pandas/skimage real installs at train time — local proofs stubbed import-only.)

## PRE-DECLARED TRAINING SCHEDULE MATRIX (manager, 2026-07-04 — selection rules fixed BEFORE training)
- S1: TrackNetV3 fine-tune from badminton seed (models/checkpoints/tracknetv3/TrackNet_best.pt) on
  assembled_v1 mixed corpus, codec_motion_v1 augmentation train-only, seq_len 8, ~30 epochs.
- S2: WASB fine-tune via blurball train_blur from tennis seed (wasb_tennis_best.pth.tar), LR 3e-4
  precedent, on the blurball-layout conversion of assembled_v1, ~10-30 epochs.
- S3 (stretch, after S1/S2 read): hard-negative-heavy variant (mine_ball_detector_errors on TRAIN-split
  clips only).
- SELECTION (pre-declared, amended post-T2): training-time val split (in-format, checkpoint selection
  inside trainers) = roboflow held-out videos; final per-arch candidate = best checkpoint by WOLVERINE
  F1@20 post-training (inference + canonical CLI; wolverine internal-val, sweeps legal); burlington =
  untouched internal reference scored once per final candidate. Train data contains ZERO owner frames.
- HELD-OUT BAR (pre-declared): a candidate earns a prereg'd held-out run ONLY if its wolverine F1
  exceeds the best zero-shot wolverine row for the matching arch family (WASB 0.735 / BlurBall 0.702 /
  TNv3 0.626, regen-reproduced) by >= +0.02. Otherwise: evidence report per goal stop-conditions.
- GPU: serialize S1->S2 on the A100 under gpu-train-lock (second spot GPU only if queue congestion
  materially delays; owner cost rules apply). (4) Metadata fix:
  our TrackNetV3 checkpoint is a BADMINTON-trained checkpoint, not tennis. NCTU dataset host is dead;
  Shuttlecock set auth-gated.
- T1c TOOLING SCOUT: DONE. KEY FACTS: TrackNetV3 fine-tune path is TURNKEY (tested CVAT->TrackNet-layout
  builder w/ video-level splits + code-level eval_guard + codec_motion_v1 augmentation + hard-negative
  mining -> vendored third_party/TrackNetV3/train.py consumes it byte-exactly). WASB path: vendored
  trainer exists (hydra, GPU-only, LR 3e-4 precedent) but the CVAT->WASB dataset glue is DELETED and no
  pickleball dataset yaml/class exists -> T1d Codex lane DISPATCHED to rebuild it with the TrackNet
  layout as canonical intermediate (runs/lanes/ball_t1d_wasb_dataset_glue_20260704/). All fine-tuned
  checkpoints from July-1 are gone (only zero-shot seeds survive); wasb_volleyball checkpoint gone;
  prereg is convention-not-code (hand-write per ledger POLICY rules 1-6). gpu-train-lock.sh = full-GPU
  flock for training (distinct from eval lease). blurball checkpoint on disk is unregistered/unwired.
Then: manager designs the training recipe + video-level-split validation protocol; training on the idle
A100 (disk 56%); selection on non-held-out evidence only; ONE prereg'd held-out run per candidate.

## PRIOR LOOP TERMINATION (2026-07-04 ~07:55, superseded by the training re-steer above)
All four mission stop-conditions are now literal: (1) held-out gate FAILED honestly (row 23);
(2) further fusion tuning against held-out = banned label mining; (3) detector accuracy gains =
blocked on missing owner in-domain labels/captures; (4) stronger external checkpoints don't exist
(TrackNetV4/V5/PB-MAT). No in-flight lanes remain. Manager holds at a long heartbeat for owner input.
NOTE FOR NEXT SESSION: all chain code UNCOMMITTED across ~15 ball-owned files (hashes in ledger row 22);
commits deferred per owner joint-commit rule. 5 truthful_capabilities failures are OTHER sessions'
doc/storage state.
  Manager design sketch (refine at dispatch): per-frame observation = candidate set (WASB top-5 +
  TNv3 top-5 + fused point + BlurBall argmax); segment fit does IRLS/EM-style per-frame candidate
  selection against the arc hypothesis, initialized from the fused track; frames with no inlier
  candidate stay hidden (fail-closed); gates/LOO/degenerate status unchanged. TARGETS: measured-grade
  recall up from 0.458 with precision ~0.9 held; product view beats fused F1 0.785 with hidden-FP
  materially under 0.349. Quota note: 4 Codex lanes completed today; resume-on-quota-death is the
  fallback (codex exec resume).
- W4 chain scorecard DONE (runs/lanes/ball_tracking_w4_chain_scorecard_20260704/). Combined internal-val:
  (a) fused raw F1 0.785 / hFP 0.349 / P95 690px / 22 tp
  (c1) arc MEASURED-GRADE: P 0.898, hFP 0.130, median 7.3px, P90/P95 16/19.6px, 0 teleports, R 0.458
  (d1) measured+fused-fallback product view: F1 0.772 — does NOT yet beat raw fused.
  -> Chain's current win = precision/tails/teleports in measured bands; recall is the gap W3 closes
  (top-K candidates -> more frames measured-grade). Floor verdict: burlington calibration-limited claim
  HOLDS (different metric than the fps bug); wolverine never calibration-limited. Classification:
  production-eligible-3D = 0 (fail-closed intact). Overlays + court top-views + trajectory strips
  manager-verified real and honestly labeled. PREREG_CHAIN_DRAFT.md ready with BLANK decision bars.
  MANAGER RULING: held-out shot HELD until W3 lands (product view must beat fused F1 AND measured-grade
  recall must rise materially before spending exposure). Ledger note 21 appended: W4 lane self-reported
  reading only len(frames) from held-out label files (no content; logged for completeness).
- RALLY-GATE lane DONE (runs/lanes/ball_tracking_rally_gate_20260704/): MEASURED NULL ON THIS DATA —
  both internal-val clips have zero dead time (production spans = full clip; gating provably a no-op;
  fused numbers reproduced exactly). Ruling: fused hidden-FP 0.349 is IN-RALLY occlusion hallucination;
  W3 physics association is the remaining mechanism. Rally gating stays valuable only for real match
  footage with dead time (none in current eval set — a data note for owner captures). Also flagged:
  burlington fused tail P90 978px = distractor chains during occlusion — same W3 target.

Updated: 2026-07-04 ~00:45 local. Manager: Fable. Mission re-steered by owner: find + test the strongest PRACTICAL
approach for 3D ball trajectory through the rally — not limited to repo detectors. Web research + GPU
acquisition authorized (spot, <$2/hr take; $2-3/hr only if materially unblocking).

## RESEARCH MEMO (v1 — will be refreshed when research lanes R1/R2/R3 land)

THEORY OF THE PROBLEM: single-camera 3D ball accuracy factors as
  (1) 2D track quality (recall/precision/tail) x (2) rally segmentation (contacts/bounces) x
  (3) calibration quality, lifted through physics/court constraints (gravity+drag arcs, bounce on
  court plane z=0, net plane, contact heights from the 3D player skeletons we already have).
NO 3D ground truth exists -> honest measurables are ONLY: (a) reprojected-2D metrics vs reviewed CVAT
labels (F1@20, hidden-FP, P90/P95, teleports), (b) leave-one-out arc-fit residuals, (c) physical
plausibility (bounce heights, speed continuity), (d) human overlay review. Anything else is vibes.

CANDIDATE TECHNOLOGIES (initial slate; ? = pending research verification):
1. Ensemble fusion of existing detectors (BlurBall+WASB+TNv3 arbiter) — RUNNABLE NOW (lane in flight).
   Why: measured oracle union recall +7.4pt over best single; consensus precision 0.87 vs lone 0.52.
   Ceiling: fusion F1 ~0.82-0.87 max -> M1 gate 0.90 unreachable this way; target = beat 0.7248 anchor.
2. Physics multi-hypothesis 3D lift extending repo arc solver — RUNNABLE NOW (CPU). Repo arc solver
   already achieves LOO 3D residual 0.031-0.066m internal-val (render-only, partial coverage). Extension:
   consume fused 2D + top-K candidates, add drag (pickleball is high-drag), joint association+fit.
3. Point-tracker gap-fill (CoTracker3 / TAPIR class) seeded by high-confidence detections — LIKELY
   RUNNABLE (public checkpoints; needs GPU for speed) — pending R1 facts on small blurred objects.
4. Newer 2025-26 2D small-ball detectors with public weights — pending R1 (TrackNetV4/PB-MAT confirmed
   NO WEIGHTS locally or upstream so far; TOTNet measured dead 0.218).
5. Literature monocular 3D lift methods (monoTrack lineage, tennis/table-tennis 3D, factor-graph /
   candidate-cloud association) — pending R2 for code-today availability.
6. Training/fine-tuning paths — BLOCKED: CVAT-only fine-tunes are a killed pattern (3 held-out
   inversions); owner captures don't exist yet. Any training needs new owner-captured labeled data.

RUNNABLE TODAY (facts): 3 detector checkpoints local+sha256-pinned; existing per-clip tracks for all 4
clips; arc solver + physics fill CPU-runnable; A100 idle this instant but SHARED with placement loop
(serialize via scripts/gpu-train-lock.sh; verify before use); NEW VM acquisition BLOCKED (gcloud auth
needs interactive login — owner one-command unblock: `gcloud auth login`). Local disk 9.1G free (tight);
VM disk 88% (25G free).

FIRST EXPERIMENT CHOSEN (E1): fused-2D -> arc-solver 3D on Burlington+Wolverine (internal-val).
  Why: uses the two strongest measured assets (fusion oracle headroom + working arc solver), all-CPU,
  zero held-out exposure, directly produces a BEFORE/AFTER 3D-relevant table.
  Before = existing single-detector arc runs (coverage 176-295/300 wolverine, 566-588/600 burlington;
  LOO 0.0662/0.0313m). After = arc solver fed by fusion_arbiter_v1 track (+top-K variant if cheap).
  EXPECTED ARTIFACT: runs/lanes/ball_tracking_e1_fused_arc_<date>/ with reprojected-2D benchmark JSON
  (canonical CLI), LOO residuals, coverage, bounce plausibility, overlay PNGs, and an explicit
  2D / review-only-3D / production-eligible-3D classification (production stays fail-closed).

## Truth anchors (all reproduced/verified this session — baseline lane PASS, 0 discrepancies)
- WASB Outdoor anchor F1 0.7248 (thr 0.25 run) / 4-clip leaderboard: BlurBall 0.670 (hFP 0.350),
  WASB 0.637 (0.241), TNv3+Inpaint 0.6254 (0.200; P90/P95 526.9/599.2), TOTNet 0.218 (dead).
- InpaintNet ruling: the 0.6254 track IS InpaintNet-refined (suffix "_heatmap" = confidence source only).
- Internal-val recall@20: BlurBall 0.7851 > WASB 0.6618 > TNv3 0.6233; oracle union 0.8594;
  fusion F1 ceiling 0.8208 naive / 0.8686 perfect-suppression. Consensus precision 0.8715 vs lone 0.52.
  Hidden-FP: 56.3% of hidden-frame predictions have >=2-detector consensus; streaks mean 3.4-4.9 frames.
  MISS/WRONG 72-85% near_player. (runs/lanes/ball_tracking_failure_taxonomy_20260703/)
- M1 gate F1>=0.90/R>=0.75/hFP<=0.05: unreachable by fusing current detectors. VERIFIED stays 0.

## Fail-closed audit (Wave 1 lane 3 — PASS, 278/278 tests green; runs/lanes/ball_tracking_failclosed_audit_20260703/)
GAPS found (evidence + unapplied diffs in proposed_diffs.md): (1) confidence_gate never checks ball
frame `approx` flag -> court-plane-approx positions can badge BAND_MEASURED; (2) arc-solver
`experimental_off` self-kill status has ZERO read sites; (3) 2 bounce-lift writers unmarked at source
(inert downstream today); (4) garbage ball confidence softly blends into contact scores unmarked;
ball_inflections defaults missing world_xyz to (0,0,0) corrupting wrist matching.
-> CODEX FIX LANE dispatched for clean-file diffs + regression tests (dirty-file diffs deferred to
   other session's landing). This hardens the honesty rails BEFORE E1 3D claims.

## Active subagents
- E1-PREP Cd sweep + lift floor (Sonnet) -> runs/lanes/ball_tracking_e1prep_cd_floor_20260704/ — RUNNING.
- FUSION-ARBITER (Sonnet) -> runs/lanes/ball_tracking_fusion_arbiter_20260704/ — RUNNING.

- CODEX-FAILCLOSED-FIXES -> runs/lanes/ball_failclosed_fixes_20260704/ — DISPATCHING (quota risk noted).

## R3 repo 3D deep-dive facts (DONE — full report in agent output; key rulings baked here)
- Arc solver (threed/racketsport/ball_arc_solver.py + scripts/racketsport/solve_ball_arcs.py): camera-ray
  segment fits, gravity + quadratic drag (m=25.5g, d=74.2mm, Cd 0.33 outdoor/0.45 indoor), Huber robust,
  anchors = reviewed bounces (immovable) / solver bounces (s=0.18) / skeleton-wrist contacts (s=0.35);
  net plane = sanity check only. experimental_off self-kill = LOO regression OR >20% sanity violations.
  LOO = held-out ray-distance self-consistency (NOT 3D GT accuracy). Wolverine base run died at 72%
  sanity violations from bad 2D (107 FP sightings pruned) -> E1 thesis confirmed: 2D quality is binding.
- Baseline arc numbers (size-depth run): burlington status=ran, coverage 588/600, LOO 0.0313m;
  wolverine coverage 295/300 only via weak tail, LOO 0.0678m. Inputs recipe in
  runs/ball_size_depth_20260703T0xZ/*/COMMANDS.sh (contact_windows+skeleton3d from
  runs/skeleton_upright_20260703T0018Z/*, reviewed bounces from
  runs/ball_bounce_inout_review_packets_ground_contact_only_20260701T200001Z/*).
- Candidate-cloud BLOCKED reason (recorded): adapters persist argmax only; wasb_adapter.py already
  computes concomp blobs in memory and discards them -> small clean-file change persists top-K and
  unblocks multi-hypothesis 3D association. QUEUED as stage-2 Codex lane (post E1 / quota).
- Blur sidecar measured (angle/length per frame, burlington 488 clear) but ZERO consumers -> stage-2
  arc-solver velocity-direction prior candidate.
- Calibration reality: burlington/wolverine metric_confidence=LOW (reproj p95 19.8/19.9px ~ the 20px F1
  radius); outdoor/indoor = MED (12.3/8.8px). Internal-val 3D fits carry a ~20px calibration noise floor.
- virtual_world ball policy default arc_required_for_midair; arc overlay authoritative over physics fill;
  court-plane lift only via review-only escape hatch. (Audit gap: consumer ignores experimental_off —
  Codex fix lane in flight.)

## R1 2D-survey ruling (DONE — runs/lanes/ball_tracking_research_2d_20260704/R1_2D_SURVEY.md)
- NO better runnable 2D detector exists today: TrackNetV4 dead (placeholder links), TrackNetV5
  proprietary, PB-MAT not found anywhere (treat as nonexistent), monoTrack = TrackNet variant w/ heavy
  deps. -> Fusion + physics lift stays the strategy; detector trio is the practical 2D frontier.
- QUEUED cheap candidates (internal-val first, behind E1): (a) WASB-SBDT-FPFilter — targets hidden-FP,
  weights bundled, but unlicensed 2-star repo -> EVAL-ONLY, owner blessing needed for anything more;
  (b) VballNet ONNX (CPU 100fps) as possible 4th ensemble member — big domain gap, near-zero cost.
- Point-tracker gap-fill (CoTracker3/Track-On2) DEMOTED to stage-2: blur is their shared weakness,
  zero ball evidence, chunking engineering cost. SAM2-mask-propagation stays dead (corroborates SAM3.1 kill).

## R2 3D-survey ruling (DONE — runs/lanes/ball_tracking_research_3d_20260704/R2_3D_SURVEY.md)
- Our arc solver = the field's best-evidenced lift architecture (MonoTrack-style physics segment fit);
  our LOO-on-real-anchors validation is MORE rigorous than most published validation. NO pickleball
  3D-CV paper exists. -> extend our solver; adopt nothing wholesale.
- DRAG FINDING: only pickleball physics paper (arXiv:2409.19000) uses Cd=0.6 (wiffle-ball analogy,
  +-0.5 uncertainty); solver hardcodes 0.33/0.45. -> E1b Cd sensitivity sweep DISPATCHED.
- MonoTrack 3-tier validation adopted: E1c = solver fed reviewed 2D labels (internal-val, diagnostic
  only, never-a-candidate) = physics+calibration lift floor. DISPATCHED with E1b.
- Stage-2 idea bank: TT3D public code (bounce friction linking segments, spin inference);
  trajectory-LSTM hit detection if segmentation ever binds. UNVERIFIED items flagged in survey doc.

## FUSION-ARBITER ruling (lane DONE=PARTIAL per pre-declared rule; manager ACCEPTS candidate, DEFERS held-out)
Canonical internal-val numbers: FUSED F1@20 0.7852 (vs BlurBall 0.7774 best single), precision 0.8107,
recall 0.7613 (-2.4pt vs BlurBall), P95 689.6px (vs 1085.8), teleports 22 (best), hidden-FP 0.3493.
STRUCTURAL FACT: hidden-FP has a hard ~0.349 floor across the whole 432 grid — 33.6% of hidden frames
have >=2 detectors spuriously agreeing; 2D consensus cannot distinguish that. -> The hidden-FP fix must
come from the PHYSICS LIFT (arc solver prunes sightings inconsistent with ballistic segments), which is
why held-out prereg is DEFERRED until E1 rules on the fused+arc CHAIN as one candidate.
Candidate frozen: fusion_arbiter_v1, fuse_arbiter.py sha256 94bfbac0...f08385, config
{r_agree 20, c_blur 0.5, v_max 120, g_max 5, c_other 0.8, s_max 4}; PREREG_DRAFT.md in lane dir (its
held-out detector-track paths 404 post-deletion — regenerate before any held-out run).
Overlay verified by manager (wolverine f33 lone-trusted: fused point tight on label). Small code item
queued: BallTrack.source enum lacks fusion value (artifact uses source="fused").

## CODEX-FAILCLOSED-FIXES ruling (lane DONE=PARTIAL; manager ACCEPTS)
Applied w/ red->green regression proofs: confidence_gate approx-ball never BAND_MEASURED; bounce-lift
review-only markers at source (ball_bounce_2d, ball_manual_court_inout, schema); ball_inflections no
longer fabricates (0,0,0) world_xyz; event_fusion low-trust ball-cue notes. 500 passed / 5 failed —
all 5 proven pre-existing at HEAD (truth-doc/storage tests broken by other sessions' state; theirs to fix).
Diff 2 (arc status gate in virtual_world.py) correctly DEFERRED as patch — file owned by placement
session: runs/lanes/ball_failclosed_fixes_20260704/deferred_virtual_world_arc_status_and_raw_ball_flags.patch
NOTE: threed/racketsport ball files now carry THIS session's uncommitted changes (commits deferred per
owner's joint-commit rule).

## Finished subagents
- Explore inventory scout: DONE. BASELINE-REFRESH: PASS. FAILURE-TAXONOMY: PASS. FAILCLOSED-AUDIT: PASS
  (5 gaps). R3 repo 3D deep-dive: DONE (facts above).

## Current blocker
- None hard. GPU acquisition unblocked; current wave still CPU. A100 idle-but-shared via gpu lock.

## Next GPU job (queued)
- If R1 confirms a point-tracker or new detector worth trying: 4-clip (internal-val first) inference on
  A100 via gpu-eval-run.sh, commands in runs/lanes/ball_tracking_baseline_20260703/gpu_job_queue.md.

## Owner asks (numbered, easiest-first)
1. **ROBOFLOW API KEY (2 min, gates the whole training push):** app.roboflow.com -> Settings ->
   Roboflow API -> copy the key -> paste it here in chat (or save to ~/.roboflow). Unlocks ~4-6k
   labeled pickleball ball instances (all CC BY 4.0) for the fine-tune data mix.
   Alternative if you prefer: manually click Download Dataset (COCO or YOLO format) on:
   universe.roboflow.com/racket-ai/pickleball-iiv9m, /liberin-technologies/pickleball-vision,
   /pickleball-od8al/pickleball-seg — and tell me where you put the zips.
2. OPTIONAL: bounce review kit rerun (validation-only now, not a dependency).
2. Did YOU delete ~26GB local runs/ (00:26-00:31) AND clean the VM disk (88%->41%)? If yes — understood
   (worst casualty: those bounce reviews). If no, a rogue cleanup process is loose on both machines.
3. Local disk now 92% (35G free); VM 41% — both workable.

## GPU acquisition (UNBLOCKED 2026-07-04 — owner completed gcloud auth login as hello@swayformations.com)
- Verified: instances list works. Spot quota asia-southeast1: A100 x16 (1 used), L4 x8, T4 x4.
- Policy armed: acquire only when a lane needs it; prefer L4 spot (~$0.2-0.3/hr) for point-tracker/detector
  inference, A100 spot (~$1.1-1.3/hr) if VRAM-bound; prove real (RUNNING+SSH+nvidia-smi+imports) before
  claiming; separate VM/run dir from placement loop; tear down when queue empties + log cost.

## Standing owner rules added 2026-07-04
- Subagents NEVER on Fable: every Agent call carries explicit model (sonnet=analysis/GPU/browser,
  haiku=trivial scouts); Codex=implementation. Booked to persistent playbook memory.

## Heartbeat integrity baseline (post kill-flag fix, 2026-07-04 09:07)
Row-22 hashes still valid for: ball_arc_solver.py 3dd5f7c802f0602a, ball_bounce_candidates.py
7c3097a212b0c290, propose_ball_bounce_candidates.py e0e5410807e09e0f, schemas/__init__.py 0ff5426190772aa8.
Updated by the ACCEPTED kill-flag lane: solve_ball_arcs.py b484c9b913581d62, run_ball_chain.py
6599de7c66e64ba3. Heartbeats check against THESE values.
