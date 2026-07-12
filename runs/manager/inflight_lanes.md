# In-flight lanes (write at session end, read at session start)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| plan_nextmoves_20260710 | Codex xhigh planning consult (read-only; tranche-2 ranking) | Fable bg 1baa1de1 | codex exec resume (log) | runs/lanes/plan_nextmoves_20260710/** only | — | ~30-60m | 2026-07-10 ~13:2x PDT |
| ballcand_20260710 | Codex xhigh Wave B code: UKF fallback + wire RANSAC/blur candidate flags + offline internal-val scoring | Fable bg 1baa1de1 | codex exec resume (log) | threed/racketsport/ball_physics_fill.py + ball_arc_chain.py + ball_ransac_arc_gate.py + new ball_ukf_* + tests/racketsport/test_ball_*.py + runs/lanes/ballcand_20260710/** (FORBIDDEN: process_video.py, orchestrator.py, virtual_world fail-closed) | — | ~2-4h | 2026-07-10 ~13:2x PDT |
| spine017_20260710 | Codex xhigh integration owner: ns015 runner hunks (coaching_facts + missing_capabilities + manifest-last) + NS-01.7 slice (post-BODY refined events, audio normal-path, dependency hashes) | Fable bg 1baa1de1 | codex exec resume (log) | scripts/racketsport/process_video.py + threed/racketsport/orchestrator.py + events/contact/audio modules (NOT ball_physics_fill/ball_arc_chain/ball_ransac/ball_ukf) + tests + runs/lanes/spine017_20260710/** | — | ~2-4h | 2026-07-10 ~13:2x PDT |
| reid_restore_20260710 | Sonnet lane: restore missing best-stack OSNet ReID ckpt (now fail-loud) + loader proof + MANIFEST entry | Fable bg 1baa1de1 | SendMessage resume | models/checkpoints/osnet_x1_0_market1501.pt + models/MANIFEST.json + runs/lanes/reid_restore_20260710/** | — | ~30-60m | 2026-07-10 ~13:2x PDT |
| fps_mac_20260710 | Sonnet lane: FIRST real-GPU browser FPS measurement (vite + playwright, renderer-string proven) on post-fixv viewer | Fable bg 1baa1de1 | SendMessage resume | web/replay/public staging + runs/lanes/fps_mac_20260710/** | — | ~30-60m | 2026-07-10 ~13:2x PDT |
| fixv_viewer_20260710 | DONE 2026-07-10 ~12:5x RULED PASS + committed c6ffa05d5 (9/9 items; vitest 245 + typecheck manager-re-verified) | Fable bg job 1baa1de1 | codex exec resume (session id in log.txt) | web/replay/** + runs/lanes/fixv_viewer_20260710/** | — | ~1-2h | 2026-07-10 ~12:3x PDT |
| fixp_pipeline_20260710 | DONE 2026-07-10 ~13:0x RULED PASS + committed (6/6 honesty fixes; wide 3441/7-preexisting in-sandbox; focused+server suites manager-re-verified locally; behavior-default note: missing wired-default ReID now fails tracking loudly) | Fable bg job 1baa1de1 | codex exec resume (session id in log.txt) | scripts/racketsport/process_video.py + threed/racketsport/{orchestrator,camera_motion,virtual_world}.py + matching tests + runs/lanes/fixp_pipeline_20260710/** (FORBIDDEN: court files, import/ingest labeling files, web/, ios/, server/) | — | ~1-2h | 2026-07-10 ~12:3x PDT |
| ns016_bodyframes_20260710 | DONE 2026-07-10 ~03:4x PASS (2 rounds): root cause = b437b4118 1200-frame uniform cap (spec's ns06 window DISPROVEN w/ evidence; w7_critique 'rev-9 success' claim also disproven — that run died at calibration); fix = single authoritative BODY execution set + typed frames-stage errors + warm-dir validation; zwCtH 18-missing->0, wolverine byte-identical, red->green test, wide 3420/0; semantics delta ruled ACCEPTED (cap binds selection; exclusion provenance = NS-01.6 follow-up) | Fable bg job 03267a94 | codex exec resume (see log) | scripts/racketsport/process_video.py + threed/racketsport/orchestrator.py + threed/racketsport/process_video_body_frames.py + tests/racketsport/test_process_video_body_frames.py + runs/lanes/ns016_bodyframes_20260710/** (THE process_video/orchestrator integration owner this wave) | — | ~1-3h | 2026-07-10 ~01:4x PDT |
| ns015_statuspack_20260710 | DONE 2026-07-10 ~03:1x PASS w/ attribution (2 rounds): fail-closed bundle_policy module + atomic publish + every-URL gate + no-translation propagation LANDED (aec1eed0d); 10 legacy render_service tests migrated to honest contract (190/190 green local re-verify); RUNNER+iOS halves = inline hunks in handoff.md for the next integration lane | Fable bg job 03267a94 | codex exec resume (see log) | server/** + tests/server/** + runs/lanes/ns015_statuspack_20260710/** (FORBIDDEN: process_video.py, threed/**, ios/**, web/**) | — | ~1-3h | 2026-07-10 ~01:4x PDT |
| ns014_p22residual_20260709 | DONE 2026-07-10 ~06:3x: P2-2 residual DECOMPOSED (FK-vs-head ~0, grounding 2e-12mm, ~53mm=family metric x3 reproductions, postchain total 23.4mm p95 per w7-armC; per-stage replay ruled VOID/unfaithful — production-side delta emission booked); coordinates.py NS-01.4 slice + attribution CLI + coherence guard + synthetic sam3d instrument (stable replicate) LANDED (8cd810a53, 4a3cbc60a, final commit follows); GATE recalibration proposal OWNER-FACING in runs/lanes/ns014_p22residual_20260709/REPORT.md (no thresholds changed); 3 GPU arms ~$1.3-9.4 all deleted+confirmed | bg job 60076b2d | — | scripts/racketsport/gate_check_body_decode.py + scripts/racketsport/synthetic_body_decode_gate.py + threed/racketsport/mhr_decode.py + threed/racketsport/hmr_deep.py + (new) threed/racketsport/coordinates.py + tests/racketsport/test_gate_check_body_decode.py + tests/racketsport/test_synthetic_body_decode_gate.py + tests/racketsport/test_mhr_decode.py + (new) tests/racketsport/test_coordinates_api.py + (new) scripts/racketsport/attribute_body_decode_residual.py + (new) tests/racketsport/test_attribute_body_decode_residual.py + runs/lanes/ns014_p22residual_20260709/** ; NOTE: mhr_decode.py carries doc-session's 1-line docstring dirt (archive path) — preserved, will not revert; doc session: commit your sweep before/around my lane commits or the lane will carry that hunk — | done | 2026-07-09/10 |
| court_wave_20260709 | DONE 2026-07-10 ~12:3x: decisive honest negative banked (source diversity = binding constraint); 4 kills, 3,921-row corpus, masked trainer, 2 H100 runs; owner asks staged in WAVE_CLOSE.md | bg job f5806640 | closed | scripts/racketsport/{train_court_keypoint_heatmap,evaluate_court_keypoint_owner_gate,calibrate_harvest_courts,generate_synthetic_court_keypoints,run_court_line_keypoints}.py + threed/racketsport/court_calibration_metric15.py (court fns only) + tests/racketsport/test_train_court_keypoint_heatmap.py + models/checkpoints/court_unet_v2/** + data/online_harvest_20260706/court_calibrations/** + runs/lanes/court_*_20260709/** + (data2) scripts/racketsport/build_real_court_corpus.py + its test + (ext1) models/checkpoints/court_external/** ; NOT touching calibrate_charuco_device.py (cal_charuco lane) nor any ns014 file | H100 spot (w7close snapshot) when training dispatches | ~6-10h | 2026-07-09 18:15 PDT |
| ios_finish_commit | iOS bg agent (Fable capture-route session) | this session | DONE 2026-07-09 ~14:4x: swift 236/236 + xcodebuild build-for-testing SUCCEEDED + wide pytest green (storage litter cleaned) + LIVE synthetic register→presign→S3→complete→sidecar→list→delete PASS; accounts flag flipped LIVE (owner-authorized); committed ios/** + sidecar schema/tests/fixtures + lane dirs. Doc-consolidation commit NOT taken (root docs + RUNBOOK left uncommitted to avoid racing in-flight ns lanes) | — | done | 2026-07-09 |
| demo_beststack_20260710 | DONE 2026-07-10 ~04:1xZ: fail-closed 3D ball LANDED (PR #12, manifest rev 11, 132 tests green, wolverine 23.53m->0.968m proof) + demo video on owner Desktop (dinkvision_demo_20260710.mp4, 97s) + fresh H100 attestations (wolverine 379.5s rev-11; owner clip fail-closed generalized) + NEW BUG booked (cold-clip BODY frame-materialization, 3 signatures). Both demo VMs deleted+list-confirmed. Evidence: runs/lanes/demo_beststack_20260710/ (branch) + demo_beststack_gpu_20260710 + demo_beststack_render_20260710 (main). |

_(2026-07-10T00:34Z: **ns06_cpu_efficiency DONE — scoped ADOPT, VERIFIED=0 unchanged**. Same-VM warm Wolverine control 502.810s pipeline / 384.035s BODY phase / 15.34 GiB peak RSS; selected vector+topology candidate 366.810s / 241.090s / 15.18 GiB, saving 27.0% pipeline and 37.2% BODY with 244/705 workload and all scoped BODY/grounding gates unchanged. Vector-only intermediate's 19.98 GiB RSS was rejected; topology interning removed 4.80 GiB. Commits `5db34fa89`, `6eeaead70`, `cf5295182`; evidence `runs/lanes/ns06_cpu_efficiency_20260709/`. Isolated H100 deleted/list-confirmed.)_

_(2026-07-09 ~18:2x: **cal_charuco_opencv5 DONE — PASS (ruled)**. calibrate_charuco_device.py ported to OpenCV 5 (CharucoDetector; legacy calls now guarded getattr fallbacks, 0 unconditional OpenCV-4 calls by grep); synthetic E2E intrinsics-recovery test green (fx/fy ±30px, cx/cy ±20px, k1/k2 ±0.04); manager re-verified --help exit 0 + test locally; wide-suite failure set EXACTLY unchanged (6 loopback sandbox + north-star line count — storage-policy failure now CLEARED post ns06 cleanup). Gold-capture ChArUco clips are now processable. All 4 Fable-dispatched NS lanes of 2026-07-09 are DONE+RULED PASS; deliverables uncommitted, touched-path lists in each report.json for the doc/commit session.)_

_(2026-07-09 ~17:0x: **ns021_goldcapture DONE — PASS w/ attribution** (7/7 substantive deliverables: A3 ChArUco board machine-verified 24/24 corners, sync verifier catches 1-frame offset, 59-row survey sheet, 5 lane-label schemas + license card validated 7/7, 111-step owner checklist, dry-run proof; 6/6 focused tests re-verified locally; flight-sim timing failure = load flake, 10/10 isolated). Its honest finding spawned micro-lane cal_charuco_opencv5 above: calibrate_charuco_device.py uses OpenCV-4-only cv2.aruco.detectMarkers — DEAD under installed OpenCV 5 until ported. Owner physical asks for the capture live in runs/lanes/ns021_goldcapture_20260709/OWNER_HALF_DAY_CHECKLIST.md + report honest_issues.)_

_(2026-07-09 ~16:3x RULINGS: **ns013_rundag DONE — PASS w/ attribution** (5/5 substantive acceptance items PASS; run_identity.py + transactional generations landed; 12/12 focused tests re-verified locally by manager; suite failures all non-lane). **ns02_evalreset DONE — PASS w/ attribution** (source groups 1750/1750 zero-leakage, grouped folds beside per-clip, 240+240 audit strata, BALL-NS02-RESET-1 PREREGISTERED-NOT-RUN ledger row w/ frozen gates+licenses; 19/19 focused tests re-verified locally). Non-lane defects adjudicated: loopback-bind ×6 sandbox-only pre-existing; tracknetv3 overwrite test = suite-concurrency race (passes isolated); w7 export README now REGISTERED in doc allowlist by manager; **OPEN: NORTH_STAR_ROADMAP.md at 510 lines > 500-line test cap — owned by whoever is editing root docs (trim or re-rule the cap)**. License finding from eval reset: stage1/seed inherit BY-NC-SA-4.0 rows → promotion-INELIGIBLE as-is; control's upstream rights incomplete — factor into any BALL training lane.)_

_(2026-07-09 ~PM state: NS-01.1 sidecar_contract + NS-01.2a upload_wiring lanes DONE — structured PASS reports in their lane dirs; exact `swift test` rerun + commit handed to the owner's iOS agent. `runs/manager/wave8_boot_prompt.md` is SUPERSEDED by NORTH_STAR_ROADMAP §5 (post-consolidation): speed bars informational, BALL training waits for the NS-02 eval reset. NS-06 speed/storage lane is DONE under `runs/lanes/ns06_efficiency_20260709/`; the repo-wide storage-policy audit still reports concurrent generated caches and missing allowlisted source packages, which that lane preserved rather than deleting under active agents.)_

_(2026-07-10 ~01:5x, Fable bg manager session 03267a94: DOC/COMMIT PASS LANDED + PUSHED — root-doc
consolidation (NORTH_STAR sole authority, 13 docs archived), ns013/ns02/ns021/charuco/ios deliverables
committed in clean groups after local re-verification (25/25 focused + wide 3397-passed/2 live-lane
registration debts fixed), NORTH_STAR queue advanced to 2026-07-10 (<=500 lines, guards green),
**PR #12 fail-closed 3D ball MERGED to main** (rev-11 manifest; integrity pin 10->11 fixed same pass;
12/12 fail-closed tests green post-merge). Audit-stratum CVAT import was done by the concurrent
session (task 87 live, commit 490222cc2) — its lane source-file edits (import/ingest + tests) remain
uncommitted and OWNED BY that session. Court wave still live (court_data2b codex running). Next for
whoever rules: ns016/ns015 reports.)_

_(2026-07-10 ~03:5x, Fable bg 03267a94 session close: ns015 + ns016 both RULED PASS and PUSHED.
NS-01.5 server half done (runner/app hunks in ns015 handoff.md); cold-clip BODY bug FIXED (was the
top pipeline follow-up from the demo wave). Next queue rows: NS-01.4 remainder + NS-01.5 runner half
+ NS-01.6/01.7 (integration owner slot now FREE), NS-01.2b + gold capture owner-gated. Court wave +
concurrent labeling session fences unchanged.)_

_(2026-07-10 ~12:4x, Fable bg 1baa1de1 DEEP-REVIEW session: owner symptom directive (framerate/missing
people/skeleton-gap/hidden-ball/paddle) executed as dual-track — 3 Codex gpt-5.6-sol audit lanes
(dr_viewer / dr_pipeline / dr_sota, all DONE PASS) + 45-agent Fable workflow (9 confirmed / 4 refuted
defects after 2-lens adversarial verify). Synthesis + ruled fix-wave plan:
runs/research_deepreview_20260710/RULINGS.md. Headlines: demo mp4 was 10-unique-fps assembly (not
product fps); headless FPS numbers are SwiftShader artifacts (no real-device measure exists); S2 = zero-
margin court filter + SILENT ReID-missing degradation + top-4 fragmentation (raw detector sees >=4 on
96.7%/78.1%); S3 viewer fallback EXISTS (refuted) — data-starved + manifest 'skeleton_only' lie; S4
fail-closed correct, 71-91% of 2D detections wasted by lift; rally_gating bridge + RANSAC gate built-but-
dead; S5 paddle = wrist-ornament data + unbounded stale hold + debug normal arrow bigger than paddle;
iOS replay hardcoded to fixture; trust badges unmounted; camera_motion parent-frame bug on excerpts;
BODY-failure review writer overwrites authoritative plans. Fix lanes V (viewer) + P (pipeline) DISPATCHED
(rows above); T (TRK scoring) + B (ball candidates) + M (real-device FPS) specs in RULINGS §next-steps.
gpt-5.6-sol capacity-error gotcha: transient, resume-with-backoff x4 loop recovers with full context.)_

_(2026-07-10 ~13:2x owner directive before 10h+ absence: run autonomously, codex gpt-5.6-sol at XHIGH for implementation this window, Fable+Codex co-plan moves, gcloud re-authed, <=4 GPUs total (manager self-cap 3; court session may hold 1). Tranche-1 dispatched above; tranche-2 (GPU TRK scoring sweep + ball E2E attestation + planner-ranked extras) after reid_restore + plan_nextmoves land. Fleet reconcile at 13:1x: only fleet1 TERMINATED under fable-fleet label.)_

_(2026-07-12 ~10:4x PDT, Fable bg c7d8cfb2 SPRINT session start: owner directive = 12-14h max-throughput
autonomous window, Fable budget 60%-used must last, gpt-5.6-sol XHIGH = workhorse, GPU cap raised to FIVE.
RECONCILE: tranche-1 lanes ballcand/spine017/coords_parity were KILLED by machine sleep 2026-07-11T05:09Z
mid-edit (no reports; edits preserved in tree); NOTHING ran on 07-11. Fleet clean (fleet1 TERMINATED only),
gcloud auth live. ACTIONS: all 3 lanes RESUMED via codex exec resume (sessions 019f4f70-c7e6-7e11/
-c7e5-7c31/-c7e5-7143, xhigh, pids 5945/5946/5947) + NEW plan_sprint_20260712 sol-xhigh consult (pid 5948);
caffeinate -dims armed 14h against repeat sleep-kill; monitor armed on all 4 report files. Orphaned
labeling-session edits (import_w6/ingest_owner + tests, 191 ins) under adjudication for adoption-commit.
brand-exploration/ = OWNER'S untracked brand work — no lane may touch it.)_

_(2026-07-12 ~11:0x: SPRINT WAVE-1 LIVE — 7 concurrent streams: [1] ballcand resume (wide-suite census
finishing; interim report landed: UKF candidate PASS/no-op-on-card, RANSAC measured-negative, both
default-off PENDING); [2] spine017 resume; [3] coords_parity resume; [4] plan_sprint_20260712 sol-xhigh
consult; [5] webux2_20260712 codex xhigh (viewer UX wave-2: seekable event markers, camera presets,
entity toggles, sync audit, URL robustness — fence web/replay/** minus public/); [6] court_harvest_20260712
Sonnet network lane (25-40 NEW-venue frames + owner label-pack staging, attacks TRAIN-4 diversity verdict);
[7] orphan labeling adoption COMMITTED+PUSHED 328771272 (scratch-mode importer + provenance ingest, 7/7).
caffeinate pid armed; monitors on all report files.)_

_(2026-07-12 ~11:3x: PLAN_SPRINT CONSULT LANDED + CONSUMED (runs/lanes/plan_sprint_20260712/PLAN.md =
the window's ruling doc). Key corrections adopted: 6 HARVEST person_ground_truth files have ZERO valid
labels (TRK scores Burlington/Wolverine historical-internal only; HARVEST = source-only diagnostics);
BALL GPU narrowed to baseline-vs-TT3D (UKF no-op + RANSAC regression already banked). WAVE-2 DISPATCHED
(all codex gpt-5.6-sol xhigh, detached, monitored): ns014_timebase_core_20260712 (P0-D/P0-H pure typed
timebase module, pid 12678), ns051_facts_core_20260712 (NS-05.1 deterministic facts + zero-fabrication
audit, rally_metrics backwards-compat rule, pid 12679), ios_product_ui2_20260712 (five-tab truth pass,
swift-test-only boundary, pid 12680). court_harvest agent tightened to sol acceptance (100 frames /
>=25 sources / >=15 venue groups / 4x25 shards / SHA+pHash dedup / <60-distinct kill). GATED QUEUE:
ns013_stale_reuse (after spine017 commit) -> tt3d_anchor_integrate (after ballcand commit) -> coords
remainder metric15/racket6dof (after coords_parity commit) -> GPU slots A TRK / B TT3D-BALL / C BODY-
overhead (global provision gate in PLAN §3; max 3 overlapping, 5-cap is safety not target).)_

_(2026-07-12 ~12:0x: OWNER LIVE DIRECTIVE received mid-window: pb.vision cv export (banked 07-09 at
runs/research_ball3d_20260709/pbvision_cv_export/) is THE ball focus — we beat them at 2D detection,
they beat us at 2D->3D lift; reverse-engineer + try paths until we match/beat. DISPATCHED
pbv_reveng_20260712 (codex xhigh, pid 15367): schema/method forensics, clip identification, no-GT
quality pillars, ours-vs-theirs gap decomposition, reproduction map onto adopt sequence, reusable
compare_vs_pbvision.py harness. pb.vision output = reference diagnostic ONLY (never training/GT).
WEBUX2 RULED PASS + COMMITTED 006810cb7 + pushed (255/255 vitest + typecheck re-verified locally;
713 insertions); Sonnet browser-verify pass launched (vite+playwright real-Metal checklist). NOTE:
runs/lanes is gitignored — lane evidence stays local by policy.)_

_(2026-07-12 ~13:0x: PBV_REVENG RULED PASS + deliverables promoted to tracked
runs/research_pbv_reveng_20260712/ + committed/pushed. BALL PROGRAM for the rest of the window (fence-
serialized): [BL-A] tt3d_anchor_integrate (after ballcand commit; ball_joint_anchor_search+arc_solver+
arc_chain; + optional both-ends pinning behind flag; score offline w/ internal cards + pbv harness;
kill fresh-candidate fallback >=5/11) -> [BL-C] UKF recovery-policy candidate (close 58-vs-183 coverage
gap honestly; ball_ukf_fallback + arc_chain after BL-A) + radius-residual consumption; [BL-B] parallel
after ballcand: WASB size-observation persistence + radius-vs-depth diagnostic (emission side only, no
solver edits). GPU Slot B attests survivors E2E after stale-reuse close. ballcand/spine017/coords
resumes still alive (wide suites).)_

_(2026-07-12 ~13:4x: THREE MORE RULINGS. [1] court_harvest STAGED PASS-w/-attribution: 100 frames /
28 sources / 27 channels (>=25/>=15 bars beat), 4x25-shard CVAT package validated dry-run (CVAT stack
DOWN — not restarted per rule; owner one-command import staged), licenses all R&D-only standard-YT,
SwingVision-overlay source correctly excluded as competitor-processed; honest gaps: camera-height/
near-far strata placeholders, +/-2s scene-cut proxy only. Artifacts registered same-session
(1e50931a9: 12 large files storage-allowlist both-sides + OWNER_GUIDE.md doc inventory; 18/18 policy
tests green). [2] ios_product_ui2 RULED PASS-w/-attribution + COMMITTED f824a81e7: fabricated Stats
placeholders killed, audited-facts-or-empty, 6 product-truth tests green on real iOS 26.5 sim
(sim-verify Sonnet lane; 1 pre-existing device-only ANE bench fail attributed); SwiftPM 245/0/0.
[3] pbv_reveng committed fb9ecad67 (see prior note). Sonnet passive-wait death x1 (ios simverify)
recovered w/ standard resume order. Codex still live: ballcand/spine017/coords resumes + timebase +
facts. Sonnet live: webux2 browser verify.)_

_(2026-07-12 ~11:4x PDT, Fable bg 40bcb767 COURT-PRECISION session start: owner directive = deep
research court find/track precision (pb.vision-class court lock; better player/ball placement +
in/out), Fable fanout + gpt-5.6-sol ULTRA(xhigh, stated owner exception), ~10h autonomous. Fences
honored vs sprint session c7d8cfb2 (ball/spine/coords resumes + timebase/facts/ios lanes live;
court_calibration.py + metric15.py dirty = READ-ONLY this window; cvat_upload/data read-only).
DISPATCHED: court_research_sol_20260712 (codex xhigh + web_search, read-only consult, owns only its
lane dir) + court_precision_harness_20260712 (codex xhigh, owns NEW court_precision_* files + tests
only) + Fable research-fanout workflow (~30 sonnet agents). Follow-on refinement/temporal-tracking
lanes gated on harness baseline + research rulings.)_

_(2026-07-12 ~14:3x: TRANCHE-1 LANDED — ballcand PASS 4fdb7b24d, spine017 PASS-w/-attr 7fab3804c,
coords_parity PASS-w/-attr 0e97c09fe, all pushed; 5 shared wide-suite failures adjudicated (2 fixed by
court-pack registration, F1 osnet coverage FIXED 3d5125d58, F3 reid-precondition test -> ns013 fence,
F2 artifact-sensitive court pin -> coords_remainder2 fence). WAVE-3 DISPATCHED (codex xhigh):
ns013_stale_reuse_20260712 (P0-C close + F3 repair, pid 34053, THE process_video owner now),
tt3d_integrate_20260712 (BL-A: anchor search into arc chain + optional both-ends pinning, offline
scoring incl pbv scorecards, kill >=5/11, pid 34054), coords_remainder2_20260712 (metric15+racket6dof
parity + F2 hardening, pid 34055). GPU SLOT A LIVE: pickleball-h100-trkA provisioning via Sonnet lane
trk_reid_apron_20260712 (frozen margin matrix, Burlington/Wolverine historical-internal only, HARVEST
source-only, pin 3d5125d58; fleet row written; concurrency 1/5). webux2b viewer fix pair RULED PASS +
committed (257/257): placeholder-occlusion + proportional timeline scrub.)_

_(2026-07-12 ~15:0x, sprint session c7d8cfb2 ACK: concurrent court-precision session detected + fences
mutually clean — they own court_research_sol_20260712 / court_precision_harness_20260712 (NEW
court_precision_* files only) + a research fanout; I will NOT rule/commit their lanes; my
coords_remainder2 owns court_calibration_metric15.py + racket6dof.py THIS window (declared 14:3x, they
marked court files read-only). GPU cap coordination via this file + gpu_fleet.md: my planned peak 3/5
(trkA live; BODY-overhead + ball-E2E queued); their provisions add on top — whoever provisions reads
the fleet ledger first per standing rule. Timebase core committed f3cfcb932.)_

_(2026-07-12 ~12:1x PDT, court session 40bcb767 update: RESEARCH LANDED — sol consult PASS (39 cited
sources, 53-module census, 9-rank PLAN, refinement stub confirmed: refine_homography_with_lines =
optimizer_not_wired + net-top-in-planar-fit suspect) + 58-agent fanout (7/10 load-bearing claims
corroborated; pb.vision requires STATIC camera + CourtFocus lock = capture-constraint UX; TVCalib
ablation: direct reprojection-loss optimization beats homography-then-refine; PnLCalib/NBJW = GPL,
reimplement-from-paper only; MAGSAC++ = cv2.USAC_MAGSAC drop-in) persisted to
runs/research_courtlock_20260712/. Harness lane still running (interim baselines: Wolverine M1 5.33px
med @77.5% cov, Burlington 5.87px @83.3%, M2 honest-absent = no per-frame calibration exists anywhere,
M5 1px ~= 12.6-17.3cm worst-direction). DISPATCHED wave-2: court_refine1_20260712 (codex xhigh; owns
court_proposal_optimizer.py + new court_pose_refine.py + tests) + court_paintline2_20260712 (codex
xhigh; owns court_line_bank.py + court_auto_evidence.py + court_line_keypoints.py additive-only +
tests). Next: harness anti-gaming freeze on its completion -> manager re-score -> rank-3 temporal lock
if capacity. Fanout cost note: 2.4M sonnet tokens (over 1M estimate).)_

_(2026-07-12 ~16:1x: TT3D KILLED BY PRE-REGISTERED BAR (9/13 vs <5/11; pinning 8/13; tails worse) —
lane PASS, candidate REJECTED, DP follow-on killed, GPU Slot B (ball E2E) CANCELLED per plan rank-7
(saves an H100). Code committed default-off w/ banked negative 44d211eeb + tests 808770d24. Diagnosis:
candidate density/event evidence is the bottleneck, not anchor state. BALL PROGRAM PIVOT: BL-C
ball_recovery_20260712 DISPATCHED (recovery-policy v2: two-sided bridging + covariance-ceiling horizon
+ inflated-cov low-conf measurements, each separately killable; target = 58->183-class coverage w/
ZERO violations, scored w/ pbv harness). coords_remainder2 PASS committed f15052ae1. facts core
committed db3d87518; timebase f3cfcb932. Still live: ns013, sizeobs, ball_recovery, TRK GPU (trkA
RUNNING ase1-b). Concurrent court session progressing (paintline2 etc. — theirs).)_

_(2026-07-12 ~16:4x: P0-C CLOSED + COMMITTED 346ead692 (ns013 PASS-w/-attr; unfingerprinted_stale dead,
migration attestation required, F3 hermetic). GPU SLOT C DISPATCHED: body_overhead_20260712 Sonnet lane
(pickleball-h100-bodyC; persistent-worker/compile-cache/overlap levers one-at-a-time vs banked 244/705
wolverine workload; >=15% accept bars w/ metric parity; wall cap 5h). spine_factshunk_20260712 codex
lane dispatched (serialized spine owner: persist+enforce coaching_fact_audit.json before manifest).
GPU concurrency mine: trkA + bodyC = 2; Slot B stays CANCELLED (TT3D kill). Live codex: sizeobs,
ball_recovery, spine_factshunk.)_

_(2026-07-12 ~18:0x: BALL PROGRAM CONVERGENCE — BL-C landed f0a009a73 (v2_one_sided PENDING marginal
survivor; bridge REJECTED physics<100%; PB coverage target NOT closed 58/252) + BL-B landed 5672f2486
(sidecar additive-proven; radius proxy = NULL depth signal R2=0.011). CONVERGENT DIAGNOSIS across
TT3D-kill + recovery-no-op + proxy-null: upstream candidate/event evidence is THE ball-lift
bottleneck. BL-E ball_anchor_boost_20260712 DISPATCHED (rank-4, last executable path: audio/kinematic/
blur/court-proximity anchor-evidence fusion scored vs frozen reviewed event timing, PB reference-only)
+ BL-D ball_radius_est_20260712 running (dedicated apparent-radius estimator vs the dead proxy).
TRK PENDING entry banked bacbf5a20 (margin 0.5/1.0 survive; worst IDF1 0.64->0.85, cov4 0.04->0.71;
license-blocked R&D-only; RF-DETR NOT triggered). trkA torn down list-confirmed 1.655h ~$2-3.5.
Live: bodyC GPU, spine_factshunk, ball_radius_est, ball_anchor_boost.)_

_(2026-07-12 ~15:3x PDT, court session 40bcb767 CLOSE: court-precision wave COMMITTED 02d9acedf +
research 9a4eba44b + North Star CAL row updated. Landed: frozen GT-free harness cpm_v2 (M4 ours
6.61 vs pb 5.67px median Wolverine), hybrid paint-centerline evidence (default-off, covariance+
provenance), guarded optimizer replacing the unwired stub (aggressive refinement PERMANENTLY KILLED
on 3-round stability evidence — do not reopen without new independent labeled geometry), per-frame
temporal court lock candidate (synthetic-proven, static-degenerate on real clips until a reviewed
moving clip exists). CVAT UP w/ court-diversity tasks 88-91 ready for owner (mutable-attr sdk fix
applied; import report in pack). All candidates PENDING; VERIFIED=0. No GPU used. Manager edits:
list_scaffold_tools.py classifier (2 keywords). Next court moves are owner-gated (label 100 frames
-> rank-8 semantic evidence model w/ source-disjoint splits) or capture-gated (moving-clip temporal
proof; gold capture for promotion). Codex xhigh lanes: 4 + 5 resumes; fanout 58 sonnet agents 2.4M.)_
