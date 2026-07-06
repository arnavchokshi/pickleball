# Build Checklist

Last updated: 2026-07-05.

This is the operational board. It should stay short enough that a new agent can
read it before touching code. For final goal and truth boundaries, read
`MASTER_PLAN.md`; for commands, read `RUNBOOK.md`; for tier placement, read
`CAPABILITIES.md`.

No row is `VERIFIED`.

The master phased plan is `NORTH_STAR_ROADMAP.md` (owner summary + phases P0-P7 with task IDs);
edge tactics + exact stack/data are `EDGE_PLAYBOOK.md`. This board is the live operational channel.

## BUILT vs. LEFT (2026-07-05 — the honest done/not-done split)

"Built" = code/app runs and emits artifacts. "VERIFIED" = passed its promotion gate on real labels.
**`VERIFIED=0` — so everything is "built, not yet gate-passed."** That is assembly complete, accuracy
+ proof remaining — not green-field. Full detail: `NORTH_STAR_ROADMAP.md` §I.0.

**✅ BUILT & RUNNING (server):** E2E orchestrator (`process_video.py`, 17 stages, trust bands) ·
ball 3D chain default (ensemble → auto-bounce anchors → arc solver → flight-sanity → court-map view →
viewer trail+KPI) · SAM-3D body/world (MHR70 skeleton+mesh, placement, stance foot-pin + smoothing,
mesh index, contact-dense scheduling) · fused 6-DOF paddle (render-only) · manual/metric-15pt court
cal (+ Wave A auto-find on a branch) · speed 2141→~532-565s (3.8×) · data-engine ingest+prelabel+
guards · web viewer (trust bands, honesty KPIs, mesh, 2×-FPS, court-map) · eval ledger + gate scripts
+ ~2,900 tests.

**✅ BUILT (our iOS app — it EXISTS):** 110 Swift files, 7 modules (`ios/`) · capture sidecar contract
already carrying intrinsics + ARKit pose + gravity + court plane + locked exposure + LiDAR refs +
240/120fps modes · CoreMotion gravity, live overlays/guidance, on-device person track + CoreML ball
heatmap, camera-roll import, upload manifest · server ingest already consumes the sidecar
(intrinsics/provenance/taps).

**⬜ LEFT (the roadmap P0-P7):** `VERIFIED=0` (no gate passed) · in-domain training data (P0) ·
physical-device capture proof + server consumption of ARKit-pose/gravity (P0-10) · profile registry
(P0-9) · ball to bar + true 3D flight/spin (P1) · body raw-noise/handheld/GT (P2) · paddle
wired+hi-def (P3) · court auto-find landed+robust (P4) · speed levers + cost (P5) · coaching product
(P6) · accounts/onboarding/pricing/legal (P7).

## Status Board

| ID | Area | Status | Current blocker | Next useful action |
|---|---|---|---|---|
| DOCS-1 | Documentation | IN-PROGRESS | full cleanup proof is still incomplete | Keep docs small; continue truth/dead-code/storage audits without adding new narrative docs. |
| CAL-1 | Court calibration | SCAFFOLD/PREVIEW | no no-tap solver has passed reviewed PCK/reprojection gates | Keep v1 tap-assisted/metric seed; score any new solver fail-closed. |
| TRK-1 | Person tracking | IN-PROGRESS | pre-registered candidates still fail coverage/identity/spectator gates | Improve detector/data leverage; do not repeat exhausted association-only sweeps. |
| BALL-1 | Ball tracking/events | SCAFFOLD | reviewed F1/contact/in-out gates not passed | Use reviewed data and model-side candidates; preserve gray-zone behavior. |
| BODY-1 | 3D body | SCAFFOLD | independent-GT world-MPJPE gate missing/failing | Use external/independent GT; never promote candidate-label reviews. |
| PHYS-1 | Foot/physics | INTERNAL-VAL DONE | Wolverine internal-val proof is not protected-clip/product proof | Reverify on protected/representative clips after upstream gates improve. |
| RKT-1 | Paddle pose | SCAFFOLD | no true paddle-face corner/reference GT | Collect/consume true-corner or marker/reference data before pose claims. |
| IOS-1 | Native iOS/live tier | SCOPED PASS | full physical capture/import/live overlay/replay proof still incomplete | Run real device capture/import/live tier and report exact evidence. |
| RPL-1 | Replay/scrubber | SCOPED PASS | review viewer and scoped assets are not production replay verification | Verify native/web playback, size, FPS, and visual QA against a current bundle. |
| E2E-1 | Full pipeline | SCAFFOLD/SCOPED PASS | no clean clip meets all component gates plus replay SLA | Rerun `process_video.py` only after component gates improve. |
| DATA-1 | Data/eval policy | IN-PROGRESS | protected eval/training boundaries need constant enforcement | Keep guards/tests active; pre-register held-out evals. |

## Count Summary

| status | count |
|---|---:|
| IN-PROGRESS | 3 |
| INTERNAL-VAL DONE | 1 |
| SCAFFOLD | 3 |
| SCAFFOLD/PREVIEW | 1 |
| SCOPED PASS | 2 |
| SCAFFOLD/SCOPED PASS | 1 |

## Dated Lane Handoffs

- [P0-2 TREE HYGIENE 2026-07-06, Codex] Vendor pin hygiene restored:
  SAT-HMR/WASB-SBDT/blurball/TrackNetV4 internal statuses clean at
  `third_party/VENDOR_PINS.md` SHAs; SAT-HMR pin row added; WASB/blurball
  pickleball additions remain backed in `third_party/pickleball_vendor_additions/`
  and locally restored for tests without subrepo status noise. Loader-side
  legacy shim accepts IMG_1605 `court_keypoints.json` as single-frame metadata
  without touching protected labels. `gpu_cold_start.sh` fixes: ensurepip
  venv detection, explicit body_venv failure returns, and count-agnostic pytest
  smoke gate. Default ball arc chain now writes provenance-marked
  `events_selected.json` on solver `ran` and omits it on solver self-kill.
  Evidence: overlapping-calibration 7 passed; ball_arc_chain/shell 39 passed,
  8 skipped; WASB vendor dataset 7 passed; `bash -n
  scripts/racketsport/gpu_cold_start.sh` pass. Wide suite
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` =>
  2901 passed / 18 skipped / 7 failed: 6 local TCP bind `EPERM` review-server
  tests in this sandbox plus 1 unrelated A100 known_hosts/default-host mismatch.
  Report: `runs/lanes/p02_hygiene_20260706/report.md`. VERIFIED=0 unchanged.

- [P0-9 PROFILE REGISTRY 2026-07-06] Registry/schema/storage slice implemented in the working tree,
  with no pipeline wiring: five profile types, flat per-account JSON under `runs/profiles/<account_id>/`,
  versioned snapshots, opportunistic court lookup, and non-owner biometric consent enforcement.
  Files: `threed/racketsport/profile_registry.py`, `docs/racketsport/profile_registry_schema.json`,
  `tests/racketsport/test_profile_registry.py`; artifacts:
  `runs/lanes/p09_registry_20260706/`. Tests: focused
  `.venv/bin/python -m pytest tests/racketsport/test_profile_registry.py -q` => 4 passed; doc/schema
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_profile_registry.py tests/racketsport/test_truthful_capabilities.py -q`
  => 19 passed. Mandated wide suite
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` is NOT GREEN:
  2897 passed / 11 failed / 18 skipped / 6 warnings in 1643.70s. Failures are outside P0-9:
  4 WASB/blurball vendored-pickleball-file failures, 6 sandbox TCP-bind `EPERM` review-server
  failures, and 1 unrelated `configs/ssh/a100_known_hosts` host mismatch. No protected eval labels
  touched; VERIFIED=0 unchanged.

- [P0-7 FLIGHT-SIM PHASE-1 2026-07-06, Codex] Pure-numpy simulator added with
  drag core reused from `ball_arc_solver._rk4_step`/`PhysicsParameters`, simulator-only
  fixed-axis Magnus (`Cl=0.195*S`), unmeasured H13-pending bounce defaults, real
  Wolverine calibration projection, deterministic detector noise, and JSONL corpus
  CLI. Artifacts: `runs/lanes/p07_flightsim_20260706/report.json` and
  `runs/lanes/p07_flightsim_20260706/flight_corpus_1000.jsonl` (1000 rows, 80MB).
  Evidence: focused simulator/scaffold tests
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_flight_simulator.py tests/racketsport/test_scaffold_tool_index.py -q`
  => 13 passed; 1k corpus report: flight sanity failed_segments=0/demoted_frames=0,
  noise p95=34.21px recall=0.5771 hidden-FP=0.02148 (within 20%), round-trip
  p95=0.0636m over 20 clean fits, full measured CLI wall=40.715s. Mandated
  wide suite
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q`
  => 2901 passed / 18 skipped / 7 failed: 6 local TCP bind `EPERM` review-server
  tests in this sandbox plus 1 unrelated A100 known_hosts/default-host mismatch.
  VERIFIED=0 unchanged.

- [WORLDHMR-SPLIT COMPLETE + LIVE-VERIFIED 2026-07-06 ~00:2xZ, synergy-audit session] The BODY
  speed-vs-accuracy tradeoff is RESOLVED at the root. Split build_body_artifacts_from_fast_sam into
  shared compute_body_skeleton_and_metrics + separate assemble_body_monolith_payloads; fast path now
  calls the shared compute (deleted its divergent reimpl). LIVE PROOF (restarted spot VM, manager
  Bash-driven): post-split fast-default skeleton vs pre-split legacy = ALL 82390 joints <1mm, MAX
  0.0009mm (float noise); the prior 84.6mm arm/wrist regression is GONE. E2E 565.3s / BODY 516.9s
  (assembly 0.0), gate PASS cov 1.0, foot-slide 0.0201m, lock-free. Fast path is DEFAULT again, now
  accuracy-clean. Day: 2141s->565s = 3.8x, wrists correct. VP-A2: worldhmr smoothing PRESERVED
  (foot-slide 0.0201m unchanged); worldhmr.py synced to VM both repo paths (md5 e3fe1ea5). VM was
  spot-PREEMPTED + restarted (NEW IP 35.240.205.82, disk intact). Remaining lever to 6-8min:
  array_native_gate_feed 145s is now the biggest single BODY chunk. No schema change.

- [WORLDHMR-SPLIT LANE EDITING worldhmr.py 2026-07-05 ~23:5xZ, synergy-audit session — VP-A2 COORDINATION]
  Now editing threed/racketsport/worldhmr.py (your VP-A2 file, landed 15:00, idle 8h). Lane
  worldhmr_compute_assemble_split_20260705 splits build_body_artifacts_from_fast_sam into shared
  joint-compute + separate monolith-assemble so the fast BODY path gets byte-identical joints + ~473s
  (this is the TRUE fix for the array-native arm/wrist regression). Your reset/jitter/centered-smoothing
  changes are PRESERVED (extend-only; lane STOPs if you re-touch worldhmr mid-flight). If your session
  needs worldhmr.py in the next ~40min, ping here and I'll pause. The split does NOT change joint math —
  only factors compute vs assembly apart. VM sync of worldhmr still pending for BOTH our lanes; I'll
  Bash-drive a combined live A100 verify after the split lands (Sonnet spend-reset but GPU verify stays
  manager-driven).

- [ARRAY-NATIVE BODY = REGRESSION, default reverted to legacy 2026-07-05 ~23:2xZ, synergy-audit session]
  Decisive same-worldhmr isolation (manager Bash diff): the array-native slim BODY path is a ~100-145s
  speed win BUT diverges upper-body joints_world[9..20] (arms/WRISTS) up to 84.6mm vs the legacy path
  (feet/gates byte-identical). Cause: body_array_native.py reimplements wrist/stance finalization with
  shallow copies instead of sharing worldhmr.build_body_artifacts_from_fast_sam. NOT accuracy-clean ->
  Codex lane body_array_native_gate_20260705 defaulting BODY back to legacy byte-identical joints
  (accuracy-clean 702s, 3x this morning), keeping monolith-skip, gating array-native experimental.
  TRUE FIX needs splitting build_body_artifacts_from_fast_sam in worldhmr.py (VP-A2 OWNS IT) into
  joint-compute + payload-assemble so both paths share joint math AND keep the speed. VP-A2 session:
  this is the cross-session coordination item — that split is the clean way to get BODY to ~473s
  without moving wrists. Evidence: runs/lanes/e2e_synergy_audit_20260705/STATUS.md 23:2xZ + the two
  run dirs (payload_collapse_isolation_20260705T223802Z legacy, body_payload_collapse_verify_20260705T221027Z array).

- [PAYLOAD-COLLAPSE SPEED PROVEN + ENTANGLEMENT FLAG 2026-07-05 ~22:5xZ, synergy-audit session] BODY
  payload-collapse live-verified: BODY stage_wall 618.5->473.0s (-145.5s), wrapper 142->14.6s,
  deconfounded E2E ~533s (matches VP session's 532s). ACCURACY isolation pending. ENTANGLEMENT FLAG
  for the VISUAL-POLISH session (VP-A2 owns worldhmr.py): my payload-collapse lane's new
  threed/racketsport/body_array_native.py HARD-IMPORTS worldhmr.py smoothing symbols
  (DEFAULT_SMOOTHING_GAP_CARRY_FRAMES etc.) that exist only in your uncommitted worldhmr rewrite. So
  BODY assembly now depends on your lane's code — when you commit/change worldhmr smoothing constants,
  my array-native path picks them up. This is likely FINE (both legacy + array-native call the same
  worldhmr fns, so they stay consistent), but it couples the two lanes: please keep those symbol NAMES
  stable, and ping here before renaming/removing them. The live skeleton divergence vs the old 702s
  baseline (foot-slide 14.7->20mm) is attributed to YOUR grounding change, not the payload refactor —
  confirming via a same-worldhmr legacy-vs-array isolation run now. Files I touched:
  orchestrator.py + run_sam3dbody_batch.py + body_array_native.py (NEW) + schemas/__init__.py (VM-synced).

- [VISUAL-POLISH LIVE-VERIFIED 2026-07-05 ~23:4xZ, manager] Combined E2E Wolverine
  (runs/visual1_wolverine_20260705T220517Z): 532.3s total (4.0x vs 2141 baseline), gates green,
  0 jumps. VP-A/A2 smoothing verified LIVE: temporal resets 14->2, feet jitter RMS 101-231 ->
  41-78 mm/f^2, wrists -40-46%, worst root-step p95 0.267->0.100 m/frame, stance slide p95 improved
  3/4 players (P2 30->47, within gate). VM synced: worldhmr 15148bb4, body_world_label_packet
  e6cf3ad2, visual_quality 764152cb, schemas 37b6cce2, foot_pin 1a164d7f, pose_temporal 54ccbb16.
  REQUEST to ball session: contact_dense hitter-mesh scheduling (VP-B, landed) fell back to uniform
  because the default ball chain reports "no current BALL runtime/source available" in this
  environment — please share/land the ball runtime config (WASB repo+checkpoint flags or default
  wiring) that your browser-verified runs used, so contact evidence exists and hitter-dense meshes
  engage. VP-C2 mesh-in-floor/render-perf/UI cleanup is booked below.

- [VP-C2 MESH ALIGNMENT/FPS UX 2026-07-05, lane VP-C2] Web replay follow-up fixed
  the VP-C regression: chunked `body_mesh_index.json` has no mesh `joint_names`,
  so 70-joint mesh frames now reuse world MHR70 joint names for hip-root alignment
  and never fall back to `transl_world`/first joint/centroid roots. Mesh alignment
  is now an object-level transform with a floor-penetration guard and debug count;
  cached geometry is no longer rewritten during held playback. The 2x FPS toggle
  moved into the Layers panel with a compact readout, and Debug layer controls are
  behind a default-closed expander. Tests: full web Vitest 181 passed; typecheck
  clean; build clean with existing Vite chunk-size warning. Browser verification
  remains manager-owned due no-localhost sandbox. VERIFIED=0 unchanged.

- [VP-A2 WORLDHMR RESET/JITTER 2026-07-05, lane VP-A2] Offline CPU proof
  landed for the worldhmr chain only. Residual stance-anchor resets are now
  carried with honest metadata, long sparse output gaps remain resets, and
  output-stage centered joint smoothing protects stance lower-body samples while
  restoring wrist peak frames. Local Wolverine proof: speed1 resets 14->2,
  vp1 resets 10->2; feet jitter reduction 56.7%/59.4%; wrist jitter reduction
  40.4%/41.2%; lag 0 frames; wrist peak delta 0 frames; root jitter not worse;
  stance slide p95 unchanged; root_motion_temporal_jump_count 0. No VERIFIED
  claim; manager still owns VM sync and live E2E rerun. Required sync set:
  `threed/racketsport/worldhmr.py`,
  `threed/racketsport/body_world_label_packet.py`,
  `threed/racketsport/visual_quality.py`, plus additive schema validation file
  `threed/racketsport/schemas/__init__.py`. Report:
  `runs/lanes/visual_polish_20260705/lane_VPA2_resets_jitter/REPORT.md`.

- [VP-C MESH RENDER CORRECTNESS 2026-07-05, lane VP-C] Web replay mesh render
  lane landed scoped viewer-only fixes: sparse same-window meshes now hold the
  previous computed frame instead of snapping/popping through 10 Hz gaps, mesh
  honesty readout counts held gaps, solid meshes are rigidly translated per
  player/current frame to the rendered skeleton root with debug deltas, and an
  additive "2x FPS (interpolated)" control idle-schedules doubled display data
  for world skeletons plus eligible mesh midpoint frames (user mesh ceiling
  150 ms). Forbidden ball/HUD/python files untouched; browser QA remains
  manager-owned due no-localhost sandbox. Tests: focused viewerData/App Vitest
  110 passed; full web Vitest 178 passed; typecheck clean; build clean with
  existing Vite chunk-size warning. VERIFIED=0 unchanged. Report:
  `runs/lanes/visual_polish_20260705/lane_VPC_mesh_render/REPORT.md`.

- [BODY-PAYLOAD-COLLAPSE (P1/P2) LAUNCHED + P3 FEEDER NOTE 2026-07-05 ~21:5xZ, synergy-audit session,
  Codex xhigh] Codex lane runs/lanes/body_payload_collapse_20260705/ OWNS threed/racketsport/
  orchestrator.py + scripts/racketsport/run_sam3dbody_batch.py + BODY gate/splice helper modules +
  BODY tests (these were my chunkfix lane's files, now free). Goal: feed BODY gates/readiness/splice/
  skeleton3d/mesh-index from per-chunk ARRAYS, skip the 171s smpl/mesh payload assembly + 142s wrapper
  handoff in slim mode (the ONLY remaining lever: inference is 18s). Accuracy guard = byte-identical
  golden diff vs runs/body_chunkfix_verify_20260705T204618Z/source outputs. Live A100 verify is a
  follow-up Sonnet lane, NOT in this Codex lane. Do NOT edit orchestrator.py / run_sam3dbody_batch.py
  until this lands — shout here if you must.
  P3 FEEDER FINDING for the VISUAL-POLISH session: your VP-B contact_dense profile starves because
  contact_windows.json is EMPTY on cold Wolverine (events runs AFTER frames, so no same-run contacts
  feed the mesh scheduler). The fix = events-before-frames reordering in process_video.py (my audit's
  P3/B14). That file is YOURS right now (VP-A2). Either you take P3, or ping here and I'll run it once
  process_video.py is free — it's the missing half that makes VP-B's "mesh near ball" fire on first runs.

- [CAL-SYNTH LANDED 2026-07-05 ~15:0xZ, court lane, worktree — manager-accepted DONE]
  Synthetic court corpus engine: 7 mixture-weighted families (incl. tennis_overlay with dual
  line-family masks, adjacent_multi_court, portrait_phone w/ distortion + off-frame keypoints,
  harsh_shadow), streaming zero-disk API court_synth_stream.iter_synthetic_court_samples
  (deterministic; consumed by CAL-MODEL trainer 13/13), keypoint reprojection self-consistency
  0.000000px over 2000 probe samples, throughput 64-87/s @640x360. Probe corpus + per-family
  contact sheets: runs/lanes/cal_synth_20260705/samples/ (worktree; sheets copied to main).
  Renders are stylized-procedural by design — texture realism is a booked follow-up if training
  demands it. 24/24 lane tests green; suite failures all stash-proven pre-existing/other-lane.

- [CAL-GEO ROUND 2 FINAL 2026-07-05 ~14:4xZ, court lane, worktree — manager-accepted PARTIAL]
  Manager's 6 fixes implemented+measured. Cross-line slot assignment under perspective FIXED where
  evidence supports it: Outdoor 356->4.4px floor median (beats old 12.7px; overlay pixel-accurate),
  aggregate 300.5->213.3px mean median (26.3% under 289.5 baseline; <=200 hard bar missed by 13.3).
  Root causes found by measurement: affine spacing ranking excluded true perspective fits (replaced
  with cross-ratio invariant); garbage net-ROI tie-break vetoed correct courts (confidence-gated);
  self-confirming verify metric now scores vs full persistent bank (wrong fits FAIL gates).
  Remaining: Burlington/Wolverine adjacent-court lock-on (correctly BLOCKED, not promoted; top-3
  cross-frame vote = next geometric idea), Indoor 46->93 median trade (net-positive, reported),
  fallback trigger saturates (temporal-median trigger proposed, unactivated — not predeclared),
  IMG_1605 single-frame tennis-overlay needs the neural track. Zero false-confident promotions.
  Artifacts: runs/lanes/cal_geo_20260705/{benchmark_r2_final_v2,overlays_r2,report.md} (worktree).

- [VP-A2 CLAIM 2026-07-05 ~22:2xZ, manager session] Visual-polish round 2 additionally OWNS
  threed/racketsport/worldhmr.py + body_world_label_packet.py (+ visual_quality.py extension):
  VP-A proved the 10-14 temporal smoothing resets/clip are emitted there (outside VP-A ownership) and
  world-joint jitter RMS feet 100-235 mm/f^2 is the owner-visible shake. VP-A landed: root-step
  redistribution (worst-player p95 0.267->0.100 m/frame), foot-pin hysteresis/soft-anchor plumbing,
  visual_quality harness (48+4 tests green). Conflicts: shout here.

- [VP-A VISUAL SMOOTHNESS PARTIAL 2026-07-05, lane VP-A] Added CPU-only
  `visual_quality` harness/CLI plus foot-pin hysteresis+soft-static anchors and
  placement visual root-step redistribution. Baselines and offline copies live
  under `runs/lanes/visual_polish_20260705/lane_VPA_smoothness/`. Touched
  tests green: 83 passed. Reset reduction is BLOCKED by file ownership:
  `temporal_smoothing_reset` is emitted in `threed/racketsport/worldhmr.py`,
  not VP-A's allowed `pose_temporal.py`; no VM/GPU rerun attempted.

- [CAL-MODEL LANDED 2026-07-05 ~14:2xZ, court lane, worktree court-autofind-20260705 — manager-accepted PASS]
  court_unet_v2 (23.9M params, resnet34 U-Net @640x360; 15 kp heatmaps stride-4 + 5-class
  line-family seg + visibility; sub-pixel decode; geometric loss enabled) + new trainer
  scripts/racketsport/train_court_model_v2.py (AMP/cosine/resume, consumes CAL-SYNTH
  court_synth_stream contract + eval-guarded real tiers) + court_model_infer.py stable adapter +
  evaluate_court_model_v2.py 32-row real gate harness (PCK@5>=0.95 gate unweakened). CPU smoke:
  loss 9.68->8.41, PCK@40 0.075->0.200 <5min. A100 run staged NOT run:
  `bash scripts/gpu-train-lock.sh bash runs/lanes/cal_model_20260705/train_a100.sh` (worktree).
  Suite 2672P/36F — all 36 attributed to other lanes' in-flight files/missing worktree fixtures
  (incl. known pre-existing monitor_process_resources scaffold-index gap). Legacy trainer zero-diff.

- [VP-B MESH FPS LANDED 2026-07-05, lane VP-B] BODY execution now applies a
  `contact_dense` profile behind existing `--mesh-coverage-mode ball_aware`
  plumbing: when ball-aware/contact-attributed frame-plan triggers exist, the
  attributed hitter is scheduled for `world_mesh` on every tracked frame within
  the default +/-0.5s pad; existing selected ball-aware/uniform windows remain
  the sparse continuity floor for other players. No new CLI enum was added
  because `process_video.py` owns the choices list and this lane was forbidden
  to touch it. Viewer solid meshes now interpolate client-side between
  same-player, same-window computed frames only when gap <=66ms and vertex/joint
  counts match; the mini readout exposes computed/interpolated status. Real
  Wolverine vp1 read-only proof found `contact_windows.json` present but empty,
  so the profile honestly fell back to the current uniform-fill schedule (200
  frames, 785 player-frames). Tests: pytest body/body_mesh/ball-stage 37 passed,
  1 skipped; web Vitest 172 passed; typecheck/build clean. VERIFIED=0 unchanged.

- [BODY-CHUNKFIX LANDED 2026-07-05 ~22:0xZ, synergy-audit session, manager-verified] Wolverine E2E
  1144.0 -> 702.4s live (runs/body_chunkfix_verify_20260705T204618Z), gates green, slide bit-identical.
  Root causes: S4 async-writer GIL contention (pickle now synchronous) + a PRE-EXISTING silent
  cross-venv numpy pickle fallback to legacy monolithic JSON that invalidated all prior "pickle"
  handoff measurements (fixed w/ version-agnostic array descriptors). handoff 489 -> 0.83s,
  prep 131 -> 12.9s. Files: scripts/racketsport/run_sam3dbody_batch.py + its tests only; VM synced
  both repo paths (md5 9271c9e6). NEXT measured targets: mesh/smpl payload assembly 171.65s +
  subprocess wrapper 142.0s (P1/P2 in the handoff doc) -> booked 6-8min/clip is in reach; then the
  combined --body-schedule=overlap live proof (command in runs/lanes/sched_parallel_body_20260705/
  REPORT.md). Report: runs/lanes/body_chunkfix_20260705/REPORT.md.

- [SYNERGY SESSION CLOSED BY OWNER 2026-07-05 ~21:5xZ] Landed today by this session (all uncommitted,
  joint-commit rule): viewer fail-closed honesty gate (independently re-verified, pixel-proven) +
  acceptance-tool real coverage; B08 net_plane->default arc solve; SCHED-A --body-schedule=overlap
  (CPU||GPU, serial default byte-identical, 201 tests) + B12 camera-motion threading. Audit canon:
  runs/lanes/e2e_synergy_audit_20260705/ (stage graph, dead/redundant, 18-candidate synergy matrix,
  parallelism DAG) — reconciled 1:1 with runs/lanes/pipeline_speed_accuracy_handoff_20260705/HANDOFF.md.
  IN FLIGHT AT CLOSE: BODY-CHUNKFIX (P0/P1) live A100 verify #2 running
  (runs/body_chunkfix_verify_20260705T204618Z); ruling criteria + next steps in
  runs/lanes/e2e_synergy_audit_20260705/STATUS.md tail. NEXT QUEUE (in order): finish CHUNKFIX ruling ->
  combined overlap live proof -> P3 accuracy-first ball-aware frame plan + P5 membership BODY
  exclusions (claim process_video.py + process_video_body_frames.py + frame_rating.py) -> P2 mmap
  handoff (use CHUNKFIX's measured numbers) -> P7 freshness keys. Codex quota returns Jul 9 ~1:31PM.

- [VISUAL-POLISH LANES OPEN 2026-07-05 ~21:4xZ, manager session] Owner watched the four previews:
  approved a dedicated visual-polish effort. Lane VP-A OWNS threed/racketsport/pose_temporal.py +
  foot_pin.py + placement.py + NEW visual_quality.py/measure_visual_quality.py + their tests
  (smoothing resets 14->≤4, extended foot anchoring w/ hysteresis, visual root-step bound; the
  27cm/frame root steps + 14 resets/300f are the measured targets). Lane VP-B OWNS
  threed/racketsport/body_compute.py + body_mesh_index.py + web/replay MESH-layer components + tests
  (owner directive: hitter-only dense mesh ~30fps in contact windows, sparse elsewhere; client-side
  midpoint interpolation to double display rate, honest labeling; NOT touching ballTrail/shotTrails/
  courtReview/uploadApi or today's HUD honesty logic). Also rebuilding body_mesh_index for the four
  vp1 clips from their existing monoliths (no GPU) so all previews get the mesh layer. Conflicts:
  shout here. Lane home: runs/lanes/visual_polish_20260705/.

- [SCHED-A LANDED 2026-07-05 ~21:1xZ, synergy-audit session] process_video.py now has
  --body-schedule={serial(default)|overlap}: overlap dispatches remote BODY on a background thread
  after frames and runs ball/ball_arc/events/ball_fill on CPU underneath it, hard join before
  placement_refine/world; PIPELINE_SUMMARY gains a parallel_body honesty block (incl.
  body_inputs_missing_due_to_overlap); sha256 input-mutation guard fails the body stage closed if any
  BODY dispatch input changes mid-overlap; serial default is byte-identical (201 tests green).
  B12 landed: explicit --camera-motion now reaches remote BODY dispatch. Live verification pending
  (one combined GPU run with the chunk fix). Report: runs/lanes/sched_parallel_body_20260705/REPORT.md.

- [RACKET-6DOF PHASE 1 LANDED 2026-07-05 ~20:3xZ] Fused wrist+palm+box paddle estimator shipped
  (threed/racketsport/paddle_pose_fused.py + build_paddle_pose_fused.py CLI + 31 tests, all green;
  1 known-unowned scaffold fail pre-existing). Internal-val bars ALL MET: Wolverine IoU 0.258
  (proxy baseline 0.111), center 20.3px, face-normal jitter 5.4/28.2 deg/f med/p95 (was 23-53 med);
  Burlington identical config IoU 0.355 (13x baseline). Renders through UNMODIFIED
  virtual_world/viewer (racket_pose_estimate.json contract, source=wrist_palm_grip_fused,
  render-only/estimated band). Rectangle-to-6DoF kill respected; CVAT rects scoring-only;
  Outdoor/Indoor labels untouched. NOT promotion evidence (RKT-1 stays SCAFFOLD; face-angle GT
  still the gate). Ball-reflection factor dormant until arc-stage 3D contacts exist (ball session:
  your landed default ball_arc stage is exactly what activates it — coordinate at integration).
  Pipeline integration ships as deferred patch AFTER ball/speed sessions' process_video/
  virtual_world changes are verified+committed. Goal doc: RACKET_6DOF_GOAL.md; trail:
  runs/lanes/racket_6dof_20260705/STATUS.md.

- [BALL-3D-CHAIN-DEFAULT 2026-07-05, manager-verified 3 clips] The ball 3D chain (candidate sidecars ->
  auto-bounce anchors -> frozen-config arc solver -> flight-sanity gate -> viewer trail/honesty UI) now
  runs BY DEFAULT in `process_video.py`; browser-verified on burlington (22.9FPS, 436/600 measured),
  wolverine (26.3FPS, 214/300), outdoor (21FPS, 360/1151); commands in RUNBOOK.md, contract in
  BALL_TRACKING_PIPELINE.md section 10, session state in runs/lanes/ball_tracking_long_run_STATUS.md.
  Also fixed: BallStageRun.wall_seconds contract for the orchestrator timing wrap (16 tests green again).

- [SCHEDULING + BODY-INTERIOR FILES CLAIMED 2026-07-05 ~20:0xZ, synergy-audit session] Speed lane is
  CLOSED (final 1144s, its FINAL_REPORT.md written) -> this session claims its released files for the
  owner's new directives (CPU||GPU overlap, GPU utilization, selective fidelity): Lane SCHED-A
  (runs/lanes/sched_parallel_body_20260705/) OWNS scripts/racketsport/process_video.py +
  scripts/racketsport/remote_body_dispatch.py + test_process_video.py/test_remote_body_dispatch.py.
  Lane BODY-CHUNKFIX (runs/lanes/body_chunkfix_20260705/) OWNS scripts/racketsport/run_sam3dbody_batch.py
  + threed/racketsport/orchestrator.py + their test files (incl. the 16 currently-failing
  test_ball_stage_runner.py) + VM sync of owned files. Other sessions: shout here if this conflicts.
  A100 use serialized via the normal gpu-eval lock.

- [VIEWER FAIL-CLOSED GATE LANDED 2026-07-05 ~20:4xZ, synergy-audit session] The fail-open honesty bug
  (self-killed arc solve rendering as "ball: measured") is fixed: trusted-status allowlist {ran} in
  web/replay ballTrail.ts + shotTrails.ts parsers, new HUD solver_off state naming the kill reason,
  App.tsx hasArcSource requires trusted, verify_process_video_viewer.py updated to the REAL status-grid
  labels (was validating only "Players") + new assert_ball_honesty gate (fails any run whose HUD claims
  measured while the artifact status is untrusted). Red-then-green proven on real pre/post HUD
  snapshots; live check on the running viewer: ok:true, HUD="ball: solver off — physical_sanity_
  violation_fraction 0.400000 exceeds 0.200000", KPI/HUD agree. 159/159 vitest + typecheck + build
  clean; 14 new tests + 9 new pytest. Evidence: runs/lanes/ball_viewer_failclosed_fix_20260705/
  live_check/. NOTE: missing status key defaults trusted (producer always writes it; keeps legacy
  fixtures intact). Independent re-verification in flight; will be booked here if it contradicts.

- [CAL-PRODUCT LANDED 2026-07-05 ~10:0xZ, court lane, in worktree court-autofind-20260705 — manager-accepted]
  Upload flow now: predict -> preview frame + 15 draggable keypoints (new CourtReviewCanvas) ->
  explicit Confirm gates the TRUSTED court_corners upload (unconfirmed guesses ship only as
  court_review + court_assist_seed — closes the template-corners-as-trusted fail-open without
  touching process_video/orchestrator). render_app now accepts iOS court_assist_seed + serves
  preview frames. Predictor default mode "proposals" (falls back to template until CAL-GEO lands
  propose_court_from_video). 31 pytest + 156 vitest green, 8 vitest failures proven pre-existing
  (missing runs/ fixtures in worktree). NOTE: lane installed requirements-render.txt deps
  (fastapi/httpx/uvicorn/python-multipart/opencv-headless) into the shared .venv. Diff lives on
  worktree branch worktree-court-autofind-20260705 (987 insertions, 11 files + 3 new) pending
  owner apply / unblock. Browser QA pass = manager, at integration.

- [B08 NET-PLANE->ARC LANDED 2026-07-05 ~20:1xZ, synergy-audit session] Default ball arc solve now
  consumes net_plane.json fail-closed (was hardcoded None in ball_arc_chain.py main solve; only the
  seed prepass used it). Additive provenance fields net_plane_provenance{consumed_net_plane,reason} on
  solved artifact/manifest/summary. 9 new tests; regression on real internal run: solver_status +
  frame count unchanged, one legitimately-implausible weak-tail segment now dropped (render honesty).
  Report: runs/lanes/ball_arc_netplane_20260705/REPORT.md. SPEED SESSION FYI: your in-flight
  orchestrator.py edit currently breaks tests/racketsport/test_ball_stage_runner.py (16 failures,
  BallStageRun has no attribute wall_seconds) — observed 20:0xZ during this lane's blast radius.

- [SYNERGY AUDIT RULED + HANDOFF PACKAGE 2026-07-05 ~19:4xZ, synergy-audit session] Both audit lanes
  landed (runs/lanes/e2e_synergy_audit_20260705/: stage_graph.json 17 stages, DEAD_AND_REDUNDANT.md,
  B_SYNERGY_MATRIX.md 18 graded candidates, B_PARALLELISM_PLAN.md). Manager rulings:
  (1) IMPLEMENTING NOW (Sonnet, Codex quota dead until Jul 9): viewer fail-closed gate
  (runs/lanes/ball_viewer_failclosed_fix_20260705/, owns web/replay trail parsers + App.tsx wiring +
  verify_process_video_viewer.py) and B08 net_plane->default arc solve
  (runs/lanes/ball_arc_netplane_20260705/, owns threed/racketsport/ball_arc_chain.py only).
  (2) HANDOFF TO SPEED SESSION (your files, your queue — full designs + wiring points in
  B_PARALLELISM_PLAN.md 'Recommended Implementation Units' + B_SYNERGY_MATRIX.md): async remote-BODY
  dispatch with local BALL/arc/events overlap (worth ~0s on eval today, architecture win for owner
  captures); events-before-frames accuracy mode (B14, unblocks ball-aware mesh scheduling that
  speed1 run warned about); B12 SMALL CONCRETE GAP — process_video._dispatch_body_remote
  (process_video.py:2219-2230) never threads the resolved --camera-motion path to remote BODY even
  though remote_body_dispatch supports camera_motion_path (483-570) — one-liner-ish, prevents
  placement/BODY divergence on explicit-path runs; B11 cross-clip reuse guards; cold-run triple video
  decode (AUDIT-A rank 2, decode sites cited).
  (3) PARKED with reasons: B06/B15 body-cue ball rescue (4 measured internal->held-out inversions say
  no fusion tuning without owner data), B01 grounding_refine wiring (placement lane just closed
  all-green; wire-or-remove when BODY accuracy reopens), B03/B05/B09/B13/B17/B18 research.
  VERIFIED=0 unchanged everywhere. Evidence for the live fail-open viewer bug (HUD says measured while
  KPI says 0/300 on the same page): runs/lanes/e2e_synergy_audit_20260705/browser_verify/.

- [SPEED LANE CLOSED 2026-07-05 ~19:3xZ, manager] Wolverine E2E 2141->1144s (1.87x), zero quality
  change (6 verification runs, gates green, slide bit-identical). BODY transfers 1.96GB->76MB;
  monoliths now OPT-IN (`--fetch-body-monoliths`) — remote BODY no longer writes/ships
  smpl_motion.json/body_mesh.json by default (BODY reuse accepts skeleton3d+gate; contract validates
  smpl only-when-present); mesh index built on the VM in-memory and fetched (viewer-consumable).
  VM synced: orchestrator.py, schemas/__init__.py, body_mesh_index.py, run_sam3dbody_batch.py (md5s
  in lane STATUS). A100 dispatch dirs were 88GB / disk 100% full (ENOSPC killed one run) — cleaned
  to 55% with owner authorization; dispatch auto-cleanup booked. BALL session: your dispatches now
  get batched-rsync + slim behavior; pass --fetch-body-monoliths if you need the monoliths. Full
  report: runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md. VERIFIED=0 unchanged.

- [VIEWER FAIL-OPEN FIX DISPATCHED 2026-07-05 ~19:1xZ, synergy-audit session] Live browser verification
  of the ball_i1 smoke found a fail-open honesty bug: ball_track_arc_solved.json status=experimental_off
  (solver self-killed) but web/replay trail parsers (ballTrail.ts parseBallTrailArtifact, shotTrails.ts
  parseBallArcSolved) ignore status/kill_reasons -> 73 anchored_measured frames render as a confident
  "ball: measured" solid trail while the Ball KPI (confidence_gated_world, correctly gated) says 0/300
  measured on the SAME page. Also verify_process_video_viewer.py ENTITY_COUNT_LABELS are stale (only
  "Players" still exists) so the acceptance tool validated ~nothing. Codex fix lane
  runs/lanes/ball_viewer_failclosed_fix_20260705/ OWNS: web/replay/src/components/modules/ballTrail.ts,
  web/replay/src/shotTrails.ts, App.tsx (minimal wiring), verify_process_video_viewer.py (+ tests).
  Design: trusted-status ALLOWLIST at parse (unknown -> suppressed), explicit HUD fail-closed state,
  KPI/HUD agreement, acceptance tool gains the honesty assertion. Evidence:
  runs/lanes/e2e_synergy_audit_20260705/browser_verify/. Ball session: this closes your I1 report's
  "browser verification" next-step with a fix; shout here if you want the gate shape changed.

- [COURT-AUTOFIND LANE OPEN 2026-07-05 ~08:4xZ, court manager session] Owner directive: automatic
  court guess on app-open/upload + best-possible calibration (incl. tennis-overlay courts) +
  measured downstream 3D impact. Design: runs/lanes/court_autofind_20260705/DESIGN.md. FILE
  OWNERSHIP (this lane): threed/racketsport/court_detector_v2*.py, court_line_bank.py,
  court_template_competition.py, court_proposals.py, court_proposal_optimizer.py,
  court_assist_seed.py, court_motion_mode.py, net_anchor_court.py, court_keypoint_net.py,
  overlapping_court_calibration.py, court_finding_technology_benchmark.py, server/court_review.py,
  server/render_app.py, web/replay/src/{UploadPanel.tsx,courtReview.ts,uploadApi.ts,new court
  components}, ios Upload/Calibration court files, scripts/racketsport/{build_court_proposals.py,
  generate_synthetic_court_keypoints.py,train_court_keypoint_heatmap.py,new court CLIs}, matching
  tests. EXPLICITLY NOT TOUCHING: scripts/racketsport/process_video.py +
  threed/racketsport/orchestrator.py (in-flight edits 07-05 02:14/04:40 by another session — please
  ping here when landed; small calibration-consumption request queued for after). Not touching
  App.tsx (ball session's deferred patches). GPU: VM1 via train-lock, yields to BODY jobs.

- [E2E-SYNERGY-AUDIT LANE OPEN 2026-07-05 ~18:50Z] New manager session (owner directive: verify every
  stage's data helps other stages' accuracy+speed across the whole E2E workflow; fastest results,
  highest accuracy). Lane home: runs/lanes/e2e_synergy_audit_20260705/. Phase 1 is READ-ONLY (two
  Codex audit lanes: stage graph/dead-data + synergy matrix/parallelism); touches NO source files.
  Acknowledged hot files owned elsewhere: process_video.py/orchestrator.py/remote_body_dispatch.py/
  run_sam3dbody_batch.py/body_mesh_index.py/pipeline_contracts.py (speed session, rerun live),
  virtual_world.py+web/replay (ball integration), paddle/wrist (racket lane), court_*.py+server/
  (court session). Implementation lanes will be posted here with explicit file ownership BEFORE
  dispatch; A100 stays with the speed session — this session provisions its own GPU if needed
  (owner-authorized).

- [WIND-DOWN COMPLETE 2026-07-05 ~18:2x PDT] Everything committed+pushed on main (this commit);
  A100 VM powered off 17:00 PDT (delete staged for owner); RESET_HANDOFF_20260705.md is the
  canonical restart doc. Suites at close: wide pytest green except 2 booked pre-existing
  court-eval failures (IMG_1605 label schema drift — see handoff §8); vitest 182/182; doc/storage/
  scaffold consistency all green. P3-A BVP solver committed as documented WIP; array-native BODY
  path opt-in OFF (stance-protection gap, fix booked); court Wave A on pushed branch
  worktree-court-autofind-20260705. NO further GPU jobs — the VM is down.

- [WIND-DOWN MANAGER SESSION OPEN 2026-07-05 ~16:4xPDT] Owner directive: verify all five
  sessions' work, produce the reset handoff doc, get EVERYTHING committed+pushed, then terminate
  the A100 VM (new GPU + fresh agent later). Canonical doc being built: RESET_HANDOFF_20260705.md
  (repo root). Lanes running: wiring_audit_20260705 (integration truth table + deferred-patch
  ledger, READ-ONLY on source), docs_recon_winddown_20260705 (owns MASTER_PLAN/CAPABILITIES/
  RUNBOOK/TECH_STACK/RACKET_6DOF_GOAL only). Court Wave A branch pushed to origin
  (worktree-court-autofind-20260705). VM archive of unique checkpoints in flight
  (models/checkpoints/vm_archive_20260705/). TO THE TWO LIVE SESSIONS (ball P3-A Codex lane;
  payload-collapse isolation): land + commit your own files per your specs; the final sweep
  commit here will wait for you and will NOT touch ball_arc_solver/ball_flight_sanity/
  ball_arc_chain or PIPELINE_STATUS.md until your lanes report. A100 SHUTDOWN happens at the end
  of this session — dispatch NO new GPU jobs after your current queue empties.

- [RACKET-6DOF GOAL OPENED 2026-07-05 ~08:1xZ] Owner directive: new goal — full 6-DOF paddle
  rendering in the world whenever possible, driven by wrist + ball direction. Goal doc:
  RACKET_6DOF_GOAL.md (root). Lane home: runs/lanes/racket_6dof_20260705/. Does NOT reopen the
  killed rectangle-to-6DoF promotion; box-only world suppression stays. RKT-1 board row unchanged
  (SCAFFOLD) until evidence lands. FENCING: racket lanes ship NEW FILES ONLY for now — they will
  not touch process_video.py / virtual_world.py / web/replay (ball_i1 lane owns those) nor the
  speed lane's instrumentation; world/viewer integration ships later as deferred patches applied
  after ball_i1 lands.

- [S4 BODY SPEED 2026-07-05] Landed local binary subprocess handoff plumbing
  plus slim BODY monolith mode. `run_sam3dbody_batch.py` now defaults stream
  chunks to binary numpy `.npy` sidecars with a JSON chunk index while preserving
  monolithic conversion for old consumers; `orchestrator.py` wraps the
  FastSAM subprocess path to request binary sidecars/no monolithic result and
  falls back with an explicit note when an old runner lacks the flags.
  `BodyStageRunner(write_body_monoliths=False)` is now the default: it skips
  writing `smpl_motion.json`/`body_mesh.json`, records both entries as skipped
  in `body_serialization_timing.json`, still writes `skeleton3d.json`,
  `body_mesh_index/`, gates, readiness, and contact splice, and adds readiness
  notes saying monoliths were not built; `remote_body_dispatch.py` threads
  `fetch_body_monoliths` through as `write_body_monoliths`, so
  `--fetch-body-monoliths` restores current monolith behavior. Expected live
  targets for manager rerun only: handoff+wrapper 376.3+54.6s -> <40s,
  slim-mode monolith assembly 343s -> approximately 0s, slim serialization
  35s -> approximately 0s. Tests:
  `.venv/bin/python -m pytest tests/racketsport/test_run_sam3dbody_batch.py
  tests/racketsport/test_body_mesh_index.py tests/racketsport/test_remote_body_dispatch.py
  tests/racketsport/test_orchestrator_spine.py tests/racketsport/test_schemas.py
  tests/racketsport/test_body_serialization_timing.py -q` => 155 passed,
  1 skipped. VM repo sync needed:
  `scripts/racketsport/run_sam3dbody_batch.py`,
  `threed/racketsport/orchestrator.py`,
  `threed/racketsport/body_mesh_index.py`,
  `scripts/racketsport/remote_body_dispatch.py`,
  `threed/racketsport/schemas/__init__.py`. Subprocess contract change:
  binary chunk format `racketsport_sam3dbody_batch_binary_chunk` v1 plus
  legacy fallback; VERIFIED=0 unchanged and live A100 timing proof still belongs
  to manager.

- [S3 BODY SPEED 2026-07-05] Split the FastSAM batch timing contract without
  changing inference calls: `run_sam3dbody_batch.py` now emits a stdlib-only
  `SAM3DBODY_BATCH_TIMING_JSON` stdout marker plus `<out>.timing.json` with
  true model setup, compile warmup, steady bucket inference, crop/bucket/tensor
  prep, result handoff, per-bucket timing, person-frame count, and steady
  ms/person. `orchestrator.py` merges that sidecar into
  `body_stage_phase_timing.json`, keeps the outer subprocess wall, attributes
  local input prep, SMPL/body-mesh payload assembly, wrapper handoff, and builds
  `body_mesh_index/` directly from the in-memory body_mesh payload after
  writing `body_mesh.json`. `remote_body_dispatch.py`'s generated runner now
  skips the file-based mesh-index fallback when
  `body_mesh_index/body_mesh_index.json` already exists and prints an
  `orchestrator_in_memory` marker. Tests:
  `.venv/bin/python -m pytest tests/racketsport/test_run_sam3dbody_batch.py
  tests/racketsport/test_body_mesh_index.py tests/racketsport/test_remote_body_dispatch.py
  tests/racketsport/test_orchestrator_spine.py tests/racketsport/test_schemas.py
  tests/racketsport/test_body_serialization_timing.py tests/racketsport/test_process_video.py -q`
  => 267 passed, 1 skipped. VM repo sync needed:
  `scripts/racketsport/run_sam3dbody_batch.py`,
  `threed/racketsport/orchestrator.py`, `threed/racketsport/body_mesh_index.py`,
  `threed/racketsport/schemas/__init__.py`; local dispatch script also changed:
  `scripts/racketsport/remote_body_dispatch.py`. VERIFIED=0 unchanged; only a
  live A100 Wolverine rerun can prove whether `other_s` is now below 60s.

- [S2 BODY SPEED 2026-07-05] Landed BODY interior instrumentation and rsync
  batching locally. `pipeline_run.json` stage records now carry optional
  `wall_seconds`; BODY writes `body_stage_phase_timing.json` with runner-local
  phase walls, `person_frame_count`, reused compact-serialization timing, and
  explicit NOT_INSTRUMENTABLE notes for subprocess-internal model-load/compile
  warmup boundaries. Remote BODY default downloads now include
  `body_stage_phase_timing.json`; single-file uploads/downloads are batched
  with `--files-from`, and openrsync 2.6.9's lack of `--ignore-missing-args`
  is handled by one SSH output-existence precheck before the download batch.
  `body_mesh_index.py` now exposes deterministic gzip `compresslevel` with
  fixed `mtime=0`; sandbox 50MB synthetic benchmark showed no sharp CPU drop
  at lower levels (level 9: 3.04 MB/s, 0.119MB chunks; level 4: 3.09 MB/s,
  1.244MB chunks), so default stayed 9. Tests: `.venv/bin/python -m pytest
  tests/racketsport/test_remote_body_dispatch.py tests/racketsport/test_process_video.py
  tests/racketsport/test_orchestrator_spine.py tests/racketsport/test_schemas.py
  tests/racketsport/test_body_mesh_index.py tests/racketsport/test_body_serialization_timing.py -q`
  => 243 passed, 1 skipped; slow benchmark command
  `BODY_MESH_INDEX_BENCH_MB=50 .venv/bin/python -m pytest
  tests/racketsport/test_body_mesh_index.py::test_benchmark_body_mesh_index_compresslevels_prints_sandbox_throughput -q -s`
  => 1 passed. VM repo sync needed: `threed/racketsport/orchestrator.py`,
  `threed/racketsport/schemas/__init__.py`, `threed/racketsport/body_mesh_index.py`.
  New remote artifact: `body_stage_phase_timing.json`. VERIFIED=0 unchanged;
  manager still owns the VM sync and live Wolverine rerun.

- [S1 BODY SPEED 2026-07-05] Landed compact JSON writes for BODY monoliths with
  `body_serialization_timing.json`; remote BODY default downloads now skip
  `smpl_motion.json`/`body_mesh.json` unless `--fetch-body-monoliths`, fetch
  `body_mesh_index/` when present, persist `remote_body_stdout.log`, and write
  `remote_body_dispatch_timing.json` with phase/byte counts. No-force BODY reuse
  accepts `skeleton3d.json` + `body_full_clip_gate.json` when the monolith was
  not fetched. Tests: `.venv/bin/python -m pytest tests/racketsport/test_remote_body_dispatch.py tests/racketsport/test_process_video.py tests/racketsport/test_body_serialization_timing.py -q`
  => 169 passed; mandated pair => 168 passed. VM repo sync needed:
  `threed/racketsport/orchestrator.py`; confirm/sync existing
  `threed/racketsport/body_mesh_index.py` if absent or stale on the VM (import
  list read: stdlib only: gzip/json/math/mmap/platform/resource/shutil/sys/time/array/dataclasses/pathlib/typing).
  Local checkout dispatch files changed: `scripts/racketsport/remote_body_dispatch.py`
  and `scripts/racketsport/process_video.py`.
  VERIFIED=0 unchanged; live VM timing rerun still belongs to manager.

- [PLACEMENT-LANE CLOSED + SPEED LANE OPEN 2026-07-05] Owner closed joint_visual_placement_20260704
  as-is: Wolverine/Burlington/Outdoor all-green (body gate PASS, 0 root jumps, slide 18.4/8.3/23.2mm),
  IMG_1605 one accepted attributed FAIL (foot slide 0.330m, edge-of-frame zero-distortion calibration).
  Standing status doc with highlighted failure cases: PIPELINE_STATUS.md (repo root). NEW ACTIVE
  PRIORITY (owner): pipeline SPEED — baseline walls 1521–3163s/clip; lane home
  runs/lanes/pipeline_speed_20260705/. VERIFIED=0 unchanged. Ball session: your deferred viewer
  patches can proceed against the landed placement/world state (see PIPELINE_STATUS.md §4).

- [BALL->JOINT NOTE 2026-07-04 ~16:1x] scripts/racketsport/monitor_process_resources.py (your session's
  new CLI) lacks the mandatory direct-CLI reference test — test_scaffold_tool_index now fails repo-wide
  (only failure in the ball blast radius). Please add the reference test with your landing.

- [BALL-SESSION COORDINATION 2026-07-04 ~15:30] BALL training campaign running on A100 (T4: TrackNet +
  WASB fine-tunes on 8k-frame Roboflow-only corpus, owner clips excluded by eval_guard design). Viewer
  lane V1 building ADDITIVE ball-trail components under web/replay/src/ (new files only) — App.tsx/
  viewState/styles are recognized as the placement session's dirty files; ball integration ships as
  deferred patches in runs/lanes/ball_v1_viewer_trail_20260704/ + runs/lanes/ball_failclosed_fixes_20260704/
  (virtual_world arc-status patch). Request: ping via BUILD_CHECKLIST when placement lands its viewer/
  world changes so ball patches can apply. Ball status file: runs/lanes/ball_tracking_long_run_STATUS.md.

- [PLACEMENT-STAGE 2026-07-03, scoped Wolverine internal-val] Foot-keypoint placement rewrite passed all run-local acceptance targets in `runs/placement_stage_20260703T1938Z/` (far wobble p90 0.000m, kitchen bias 0.009m, near native-2D p50 0.136m, far speed p90 0.725m/s, coverage unchanged, zero introduced bounds violations); P1 p90 regression fixed to 2.4598 -> 2.4265m/s, still not global `VERIFIED`.
- [SAM3D-WORLD-PRECEDENCE 2026-07-03] `virtual_world.py` now renders `skeleton3d.json` joints before `smpl_motion` fills and emits MHR70 `joint_names` plus per-player `joints_source`; offline Wolverine copy `runs/world_precedence_20260703T0956Z/` has 1102/1102 world joint frames equal to skeleton3d and 0 equal to raw smpl, lower-arm canonical diff 0.0%, foot-pin p95 18.74mm under strict speed-threshold restage, and schema validation passing.
- [SAM3D-FOOT-PIN 2026-07-03, scoped Wolverine render audit] Post-hoc `apply_foot_pin.py` generated `runs/foot_pin_20260703T0924Z/`: rendered-world stance slide p95 37.7mm -> 18.9mm, root p90 improved for all 4 players, max correction 0.049m, limb-length delta ~0; headless viewer verify is blocked in this sandbox by local TCP bind `EPERM` (`viewer_verify_foot_pin/bind_blocker.json`).
- [SAM3D-WRIST-BONE-LOCK 2026-07-03] Direction-preserving lower-arm wrist lock added after SAM3D refine and final contact splice; Wolverine offline copy locks 2204/2204 wrist frames, lower-arm CV=0.0 and median diff=0.0% for all players, with coverage and non-lower-arm metrics unchanged. Report/artifacts: `runs/sam3d_wrist_bone_lock_20260703T0906Z/`. Manager-verified + ACCEPTED 2026-07-03: 170 tests green after manager updated one stale contact-splice test to assert the real invariants (direction preserved + lock provenance) instead of the pre-lock wrist constant; swing-peak timing exact (0-frame delta / 40 peaks); lock is the final skeleton writer post-splice. Locked skeleton awaits restage COMPOSED WITH the in-flight foot-pin output.
- [A100-SESSION-3 2026-07-03, manager-accepted] All SAM3D Phase D gates PASS on shipped defaults: steady 32.23 ms/person (≤55), first call 0.564s (≤1.0, warm-2). Wolverine ball_aware_100 dispatch succeeded, zero Skeleton3D validation errors: 4 players / 1102 annotated frames / 0 implausible / 184 mesh frames; BODY GPU 311s ≈ $0.117/clip. Artifacts: `runs/a100_sam3d_validation2_20260703T0647Z/production_remesh/wolverine_ball_aware_100/`. In flight: viewer staging of new skeletons + 4-clip wall-to-wall E2E timing with reproducibility packets.
- [SAM3D-FOOT-WANDER 2026-07-03] Found and fixed the SAM3D refine-chain foot-slide bug: heel/toe-tip joints were silently smoothed as "core_body" (laggy) instead of "feet" due to a canonical-name/raw-name mismatch in `_joint_smoothing_group`, not bone-length or grounding as suspected; per-stage measurement isolated the damage entirely to `_apply_one_euro` (37.7mm -> 377.4mm p95 stance slide at that stage alone). Fix (flag-gated, default ON, `pose_temporal.py` only): corrected heel/toe canonical-name resolution + dedicated near-pass-through "feet" one-euro params. Real-Wolverine result: pre-pin p95 37.69mm (bar <=40mm), default-threshold foot-pin accepts 97/98 phases with post-pin p95 18.92mm (bar <=20mm), lower-arm rendered error still exactly 0.0%, wrist swing-peak timing exact 0-frame delta. Report/artifacts: `runs/sam3d_foot_wander_20260703T1024Z/`.

## Rules For Updating This Board

- Keep one row per active area. Do not append chronological narratives here.
- Every status upgrade must name the command, run path, test result, device run,
  or label gate that proves it.
- If a row is scoped, include the scope in the handoff or run artifact. Do not
  let scoped evidence become a global claim.
- If a lane generates a long report, store it under `runs/` and summarize only
  the actionable result here.

## Active Priorities

1. **CAL:** maintain tap-assisted/metric seed path for v1 and fail closed on
   unverified automatic proposals.
2. **TRK:** improve detector/data and strict spectator/background handling.
3. **BALL:** pursue reviewed-label ball quality and contact/in-out gates without
   hiding uncertainty.
4. **BODY:** get independent GT for world-MPJPE and keep candidate labels out of
   promotion paths.
5. **iOS/RPL:** prove real-device capture/import/live overlay and current replay
   playback from the same artifact chain.
- [HARVEST BATCH-1 DONE 2026-07-06 ~13:1x, manager-verified] 8 videos / ~88min gameplay / 1.3GB, ALL verified genuine fixed-camera full-court doubles (ffprobe + screening spot-check; 7x1080p + 1x720p; 12 distinct rec channels). Manifest: data/online_harvest_20260706/manifest.json (25 probed entries w/ full provenance; manifest-vs-disk verified 8/8). 32 screening frames. 6 vetted next-batch candidates + 2 retries queued IN the manifest. KEY FINDING: this sandbox shapes googlevideo.com to ~200-280KB/s (28x slower than generic CDN; verified 3 ways incl. raw curl) — future bulk harvest belongs on fleet VMs, not the Mac. NEXT: P0-1b clip-to-rallies + prelabel + role assignment + combine-with-CVAT (wave 2). Owner ruled 8 videos sufficient for batch 1.
- [WAVE-1 DISPATCHED 2026-07-06 ~12:1xPT, manager] IMPLEMENTATION STARTED. GPU fleet1 LIVE: pickleball-a100-fleet1 (A100-40GB spot asia-southeast1-a ~$1.2/hr) cold-start 257s + SAM-3D smoke PASS (27/27 GPU tests; 3 real gpu_cold_start.sh bugs found -> P0-2 owns fixes; evidence runs/lanes/gpu_coldstart_20260706/). Old A100 DELETED (billing stopped). Codex quota BACK (probe passed). Wave 1 in flight, file-disjoint: P0-6 fresh 4-clip worlds on fleet1 (Sonnet; M1 milestone; --force purges stale RTMW caches; co-proves contact-dense scheduling + reflection plumbing + live backbone); P0-2 hygiene (Codex: vendor pins, calib-eval loader shim [RULED], gpu_cold_start 3 bug fixes, events_selected.json wiring); P0-7 flight-sim phase 1 (Codex: numpy, reuses _rk4_step, Magnus sim-side, corpus CLI); P0-9 profile registry (Codex: 5 schemas + per-account JSON storage + consent/retention enforcement). Harvest batch-1 still downloading (~5 videos, throttled). PART 0 GPU item ticked w/ evidence. P0-8 VFR audit deferred to wave 2 (would edit files P0-6 is executing). VERIFIED=0 unchanged.
- [TECH-AUDIT RULINGS + HARVEST + DOCS-COMMIT 2026-07-06, manager] Owner round-2 directives executed. (1) DOCS COMMITTED: abae0756 pushed (30 files — the full North Star system; PART 0 gate satisfied). (2) BROAD HARVEST APPROVED by owner (copyright waived, private use) — ruling folded into PART 0/P0-1b; harvest agent running, first mp4s landing in data/online_harvest_20260706/ (dir gitignored + storage-audit clean). (3) DEEP TECH AUDIT (7 pillar red-teams + synthesis, 9 agents incl body re-run; evidence runs/research_sota_20260705/tech_audit.md): ALL SEVEN = keep-with-modifications; manager ruled ACCEPT on the full change set, folded into NORTH_STAR. Headlines: ensemble VOTING is a measured liability (hFP 0.349 vs single-WASB 0.063) — detectors re-framed as candidate pool, physics-consistency selection A/B (P1-1), SST teacher = physics-gated chain never raw fusion (P1-2); 4-level visibility schema prerequisite + exact TOTNet recipe (P1-1); ball-3D re-SEQUENCED (stabilize P1-4a -> fixed-axis Magnus -> P0-7 on _rk4_step -> learned lift as RESCUE only; Cd DONE / Cl 0%; size-depth residual already default-on; spin gated on H13 friction); paddle re-sequenced (P3-1 now, P3-3 WiLoR next, P3-5 reflection fast-track via P2c; keypoints/masks/nvdiffrast DEFERRED-PENDING-GT-GAP w/ pixel-math gate; ForeHOI corrected ~1min+FoundationPose-bolted); court two-error-budgets split (19.8px = auto-find path; v1 floor = edge distortion ~53px -> P4-4 ChArUco k1/k2 = highest-leverage v1 fix + verify back_project applies dist; P4-2/P4-3 re-tagged UNKNOWN-COURT EPIC out of DONE-v1; net_plane already has linear sag -> P4-6.0 = tape-measured heights, catenary = v2); P0-10 corrected (zero import ARKit exists — ARSession is a BUILD; server consumer already exists); PF-2 solver RULED torch-LM/scipy-TRF (no ceres/GTSAM), gated on P0-10 (0/20 sidecars have pose), latent-MHR variable + confidence-weights + coordinate-descent fallback; speed corrections (S4 mmap attempt was a reverted REGRESSION not partially-proven; TensorRT scoped to WASB+YOLO26 only; warm-worker ceiling 65.6s/clip -> pair with batched multi-clip + compile-cache ~42s); BODY re-run verified: MHR codes already emitted (P2-2 gap = vendored MHRHead decode), camera_motion.py exists (upgrade not greenfield), MAD bone-length detector exists (wire to smoothing), input hard-capped 384-512px, SAM-3D confirmed live backbone, GVHMR accepts external pose+gravity but single-person-hardcoded -> NEW P2-7a (GVHMR gravity-view spike — runnable TODAY on tripod clips: known R + court-normal gravity) + P2-7b (far-player conditioning probe) split out of the all-or-nothing gate. NEW Part IV rule 10: signal-adoption discipline (ablation-first + pixel-math conditioning gate). Doc suites green. VERIFIED=0.
- [FINAL-PASS 2026-07-06, manager] Owner-ordered final pass COMPLETE. Personal edits: Part IV rule order fixed (7,9,8->7,8,9) + rule-7 paren balance; CLAUDE.md-first read order; Map/TOC; NEW I.7 (Definition of Done v1 + critical path + can-start-today + demo milestones M1-M5); P1-2 nightly flywheel; FABLE manual re-pointed at NORTH_STAR (title/§7-line/§9), §14 step 9 session-cadence, NEW CLAUDE.md bootstrap (registered). Then 5-lens verification (coherence B+, day-1 Fable-5 simulation, manual audit, claude-code-guide hooks feasibility, cross-doc congruence; 32 findings) — 30 apply-now fixes APPLIED: new tasks P3-4b/P5-7/P5-8 (EDGE deltas now real), P1-4a/b labels, PART 0 evidence-verification preamble + consent-scope split + CC-only harvest default, P0-2 label-schema STOP surfaced, P0-6 backbone-verify, P0-7 numpy-first/MuJoCo-prestage note, P0-9 storage RULED (per-account JSON + profile_registry_schema.json), disk-preflight trap, 'P4 ball lane' disambiguated, manual §4/§14 reconciliation + §17 Codex-quota fallback + Sonnet-not-Codex for gcloud + tool-name portability + §16 hook verdicts baked (Notification auto-resume = INFEASIBLE as specced; SessionStart inject + PostToolUse audit = feasible), run-lane skill dispatch block now §10-verbatim (--output-schema/-o report.json, no Monitors), CREATED runs/manager/{gpu_fleet.md,inflight_lanes.md} + scripts/fleet/{lane_vm_startup.sh,reconcile.sh} scaffolds. Review verdict: content thorough, structural blocker = docs-of-record UNCOMMITTED (PART 0 vs joint-commit rule contradiction) — OWNER DECISION #1; plus harvest consent/copyright scope (safe CC-only default now encoded). Evidence: runs/research_sota_20260705/final_pass_review.md. Doc suites green. VERIFIED=0.
- [BALL-P4-CLOSURE 2026-07-06, manager, closing the loop for e5789028] HEAD commit e5789028 (ball viewer render unblock + court-map markers) landed with 'manager browser verification next' noted in its lane REPORT; per runs/lanes/ball_tracking_long_run_STATUS.md the P4 delivery WAS manager-browser-verified (3 clips, screenshots in runs/lanes/ball_p4_render_fix_20260706/). This bullet backfills the missing same-commit coordination entry (final-pass finding #8); standing rule: every landing commit adds its own BUILD_CHECKLIST bullet in the same change.
- [PASS-3 + HARSH REVIEW + MANAGER SETUP 2026-07-06, manager] Court/net + fusion + production research (87 agents) folded: NORTH_STAR PART II-C + new PHASE F (global fusion PF-1..4 = the combine-everything pillar; JOSH contact-coupled joint optimization) + P4-6 net-3D + P5-5b/P5-6 QA. 7-lens harsh review (32 agents, 26 fixes, 23/24 severe survived) APPLIED: grades tech-correctness B/already-built B+/self-contained B-/consistency C+/feasibility C+/production D+/pillar-coverage B-; self-contained verdict = 'mostly, with real blocking exceptions'; weakest pillar = PRODUCTION-READINESS. Fixes: P4-0 real bullet, P1-3(e) RIFE, P5 gate rescoped to evidenced floor (≤2×/≤1× was mathematically unreachable), P0-6 circular dep removed, P7-4b biometric privacy/retention/consent (blocking before non-owner footage), pre-flight QA, fully-loaded cost, PF-2 net-occlusion+grip residuals+ARKit fallback, P7-1 server auth/durable-storage, [VERIFIED]->[CORROBORATED] tag rename, IDF1/SLA/citation corrections, train_a100.sh path fixed. OWNER 4 DIRECTIVES done: (1) multi-GPU fleet rule (Part IV rule 7 replaced; FABLE manual §12); (2) online-video harvest P0-1b + EDGE §4.2; (3) stop-and-ask rule (Part IV rule 9; manual §13) + PART 0 owner-setup block atop NORTH_STAR; (4) FABLE_OPERATING_MANUAL extended §12-16 (fleet/stop-ask/manager-loop/anti-patterns) + 3 skills drafted (.claude/skills/{research-fanout,run-lane,gpu-fleet-provision}) + hooks documented. Reports: runs/research_sota_20260705/{pass3_*,harsh_review,fable5_manager_setup}.md. Doc suites green (17). VERIFIED=0.
- [PASS-2 DEEP DIVE 2026-07-06, manager] Owner caught a missed sub-task (RacketVision TrajPred); ran a 159-agent citation-graph deep dive (enumerate every seed's sub-tasks + backward/forward refs + cutting-edge recency sweep, verify, synth). Reports: runs/research_sota_20260705/pass2_{ball,body,paddle,product}_report.md; consolidated into NORTH_STAR PART II-B. TWO CORRECTIONS to our own plan: (1) pickleball aerodynamics DO exist (Lindsey TWU 2025 outdoor Cd~0.33/indoor 0.45 + asymmetric topspin lift; Steyn arXiv:2501.00163 Cd 0.30/Cl 0.195*S — corroborated) -> seed P0-7/P1-4 instead of 'no prior exists'; (2) SAM-3D latent-smoothing (P2-2) now has a published blueprint on our exact backbone (arXiv 2512.21573). New tasks/notes: P1-8 ball forecasting (RacketVision cross-attn + LATTE-MV); adoptions mapped to tasks — SOMA-X (MHR<->SMPL-X), RF-DETR, Grounding DINO zero-shot paddle, Uplifting-TT lift (code), Human3R/DuoMo/JOSH3R body challengers, OnePoseViaGen/RGBTrack/Image-as-IMU paddle, CoachMe/BioCoach coaching, AnyCalib/BroadTrack court, WASB-HLSM/TrackNetV5-SDK detectors, SoccerNet-v3D diameter-depth (cites our H23). Competitive: Owl AI x MLP automated officiating live 2026-05-22 (broadcast multi-cam). Doc suites green.
- [ROBOFLOW-PRECISION 2026-07-06, manager] Owner caught the overstated "no pickleball dataset exists" phrasing (Roboflow has many). Corrected NORTH_STAR §I.1 + §II.1 to the precise, MEASURED claim: no academic TRACKING benchmark exists, but Roboflow DETECTION datasets do — and we already built an 8,631-frame Roboflow corpus + fine-tuned both ball archs → BOTH degraded held-out (TNv3 -17pt, WASB 0.0018 distractor-lock; runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md). Three measured reasons: wrong domain (broadcast stills vs our phone-tripod), wrong format (84% temporally-dead vs TrackNet's 3-frame need), too few sources (overfit). NOT useless → new P1-0 (aggregate ALL Roboflow pickleball projects, dedup, diversity = antidote to distractor-lock) as PRETRAIN/aux; P1-1 reworded: public data primes, owner data finishes, never ship a public-only student. Doc suites green.
- [DONE-VS-TODO + APP-AS-GIVEN 2026-07-05, manager] Owner: treat our own iPhone app as a GIVEN and make done-vs-todo crystal clear. Verified `ios/` reality (110 Swift files, 7 modules; CaptureSidecar contract ALREADY carries per-frame intrinsics/ARKit-pose/gravity/court-plane/locked-exposure/LiDAR/240fps; server ingest already consumes intrinsics/provenance/taps). Added NORTH_STAR §I.0 "What is ALREADY BUILT vs what is LEFT" ledger + per-phase "Already built / To build" lines + concrete P0-8/P0-9/P0-10 task blocks; reframed the app from "spike to build" to "prove on-device + wire server consumption of ARKit-pose/gravity (currently dropped)". Added this board's "BUILT vs LEFT" section. Key honest gap surfaced: app writes rich geometry sidecar, SERVER largely ignores it (only intrinsics→fingerprint) — P0-10 wires it. Doc suites green.
- [EDGE-PLAYBOOK-IPHONE 2026-07-05, manager] Owner: ALL video comes from iPhone. Added EDGE_PLAYBOOK §2b hacks H27-H34, two tiers: Tier 1 = any stock-camera file (H27 QuickTime GPS/timestamp/HDR/lens metadata harvest at ingest -> auto court+device profile; H27b MANDATORY PTS/VFR-correctness audit — iPhone is variable-frame-rate, latent timing-bug class; H27c stock-camera capture guidance). Tier 2 = our capture-logger app (new roadmap delta P0-10; ios/ scaffolding is the seed): H28 per-frame sidecar (intrinsics+exposure+ARKit pose+gravity+PTS+GPS+thermal), H29 ARKit pose+gravity solves handheld AT SOURCE (gravity = the exact quantity world-grounded HMR fights for; masked-SLAM becomes stock-video fallback), H30 per-frame exposure makes blur-speedometer exact + AE/AF/WB/EIS lock protects color+geometry, H31 stereo-audio side cue (spike), H32 two-iPhone timestamp-synced GT rig (independent GT we lack — P2-6/P1-4), H33 LiDAR setup scans (Pro), H34 capture-mode playbook. BOM + capture-protocol tables updated. Doc suites green (15).
- [EDGE-PLAYBOOK-MULTIUSER 2026-07-05, manager] Owner refinement applied to EDGE_PLAYBOOK.md + NORTH_STAR_ROADMAP.md: NOT owner-hardcoded — all person/court/gear specificity now flows through a per-account PROFILE REGISTRY (new hack H0: court/device/player/gear/session profiles populated by one-time setup phases, opportunistic consumption, generic-path fallback when absent; owner = profile #1, friends onboard the same way; data accumulated per user). New roadmap delta P0-9 (registry+fallback plumbing; wizard UI at P7-1). License stance reworded: private-for-now, non-constraint, revisit only IF expansion. Doc suites green.
- [EDGE-PLAYBOOK 2026-07-05, manager] Owner second pass landed: `EDGE_PLAYBOOK.md` (registered in doc allowlist, suites green) — N=1 profile-once advantages, pickleball-rulebook logic hacks H1-H26 (court-profile library, line-color family separation, double-bounce free GT, rally-grammar Viterbi events, composite ball data on real empty-court backgrounds, blur-speedometer, contact-window RIFE interpolation, whole-rally factor-graph polish), exact per-stage tech BOM, exact data-source tables, owner capture protocol v2. OWNER RULING recorded: internal-only forever -> licenses NOT a constraint (held-out discipline unchanged). NORTH_STAR_ROADMAP.md cross-linked + task deltas listed in playbook §5 (new P4-0 court profiles, P3-4b texture anchors, H17/H18 lanes). VERIFIED=0 unchanged.
- [NORTH-STAR-ROADMAP 2026-07-05, manager] Owner-requested master to-do roadmap landed at `NORTH_STAR_ROADMAP.md` (registered in the doc allowlist; doc/storage/scaffold suites green: 15+5 passed). Grounded in a 4-domain adversarially-verified SOTA research sweep persisted at `runs/research_sota_20260705/` (ball/body/paddle/product reports, ~114 Sonnet agents, primary-source cites + refutation verdicts). Phases P0-P7 with stable task IDs (P0-1..P7-5) — reference those IDs in future lane bullets. VERIFIED=0 unchanged; roadmap is planning-only and defers to CAPABILITIES.md.
- [DOCS-RECON-WINDDOWN 2026-07-05] Owned root docs reconciled against run evidence with `VERIFIED=0` unchanged; report artifacts: `runs/lanes/docs_recon_winddown_20260705/RECON_NOTES.md` and `runs/lanes/docs_recon_winddown_20260705/report.json`.
- [WIRING-AUDIT 2026-07-05] Pipeline integration truth table and deferred-patch ledger written to `runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md`, `runs/lanes/wiring_audit_20260705/DEFERRED_PATCH_LEDGER.md`, and `runs/lanes/wiring_audit_20260705/report.json`. Top gaps: dirty/untracked BODY-viewer-placement work needs reset-safe packaging, `monitor_process_resources.py` still lacks a direct CLI reference test, and SAM3D subprocess status text still says binary while default transport is pickle.
- [WINDDOWN-SWEEP 2026-07-05] Final mechanical sweep covered doc inventory, storage index cleanup, monitor CLI reference coverage, and SAM3D subprocess chunk-status wording before the manager reset commit.
