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

- [ROOTJUMP SLIDE FIX 2026-07-06, Codex PARTIAL] Source attribution: p06 placement.json was stable, but final tracks.json kept pre-visual-bound outliers because null-frame fresh tracks were rewritten through a 30fps visual-bound index while the clips are 60fps. Fixes landed in placement/process path: visual-bound rewrites now use source fps, final tracks materialize non-null frame_idx, and placement emits foot_contact_phases.json for grounding_refine. Offline lane replay at `runs/lanes/rootjump_slide_fix_20260706/fixed_chain/measurement_summary.json` clears BODY joint blockers for Burlington/Outdoor (`quality_blockers=[]`, max root speed ~8.0m/s) and preserves Wolverine/IMG1605 gate pass; slide p95 metrics remain Burlington 26.36mm, Outdoor 22.58mm, Wolverine ~0mm, IMG1605 11.37mm. Caveats: no BODY re-inference/smpl_motion local replay; fixed skeletons are rigidly re-grounded to corrected anchors; grounding_refine consumed generated phases but sanity-gate restored originals; full `tests/racketsport` was interrupted after 25:01 with 1035 passed/4 skipped and no failures before `court_finding_technology_benchmark.py` slow path. VERIFIED=0 unchanged.

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
- [P2-1 CAMMOTION 2026-07-07, p21_cammotion_20260706] PASS (module-level slice, no promotion): upgraded `camera_motion.py` default to hardened LK+RANSAC with person masking, flow-track MAD filtering, two-pass MAD+Gaussian camera-path smoothing, and scaled CPU processing; RAFT remains `not_enabled_pending_weights` (torchvision adapter/synthetic dense-flow unit test only, no downloads). Evidence/report: `runs/lanes/p21_cammotion_20260706/lane_report.json` and `proxy_metrics.json`. Acceptance proxies: IMG_1605 handheld 3/3 wins vs legacy (inlier 0.767540->0.894909, jerk 2.620541->2.453140, court stability 14.837807->11.752930); Wolverine static no hallucination (drift p95 0.563397->0.426674, max 9.485480->0.676153); masked beats unmasked on both clips; default runtime 50.22/48.52 ms/frame. Deferred process/placement patches under `runs/lanes/p21_cammotion_20260706/deferred_patches/` both `git apply --check` clean. Verification: targeted camera-motion 12 passed, placement camera-motion 2 passed, scaffold/dead-code/storage checks pass; wide `MPLBACKEND=Agg` suite = 2929 passed / 18 skipped / 6 SANDBOX-SUSPECT localhost-bind failures in review-server tests (lane forbids localhost binds). `VERIFIED=0` unchanged.
- [WAVE-1 COMPLETE 2026-07-06 ~16:5x, manager-verified closeout] ALL SIX LANES RULED. GPU-coldstart PASS (fleet1 257s + smoke; 3 script bugs -> fixed by P0-2). Harvest PASS (8 verified games, 88min, manifest). P0-2 hygiene PASS / P0-7 flight-sim PASS / P0-9 registry PASS (clean local wide suite adjudicated sandbox artifacts; 2 long-booked calib failures now FIXED; code landed 04c8da21). P0-6 fresh-worlds PASS after two manager-fixed blockers: (1) Apple-openrsync->GNU-rsync transport (brew 3.4.4; deeper ssh/VPN flake survives via lane's bounded-retry driver — runs/lanes/p06_freshworlds_20260706/body_dispatch_driver.py; tar-batch hardening booked wave-2); (2) placement frame_idx=None crash (fix+regression 90b19289). DEFINITIVE CLOSING TABLE (grounding wave, stance-aware ACTIVE 4/4, anchor=placement_track_world_xy): wolverine complete/BODY-gate PASS/slide ~0m PASS; img1605 (HANDHELD, was 330mm attributed-FAIL) complete/PASS/25.6mm PASS <- headline; burlington 46.9mm FAIL + outdoor 40.5mm FAIL (21-33x improved; sole blockers = root_motion_temporal_jump, an honest new measurement the ungrounded path couldn't see). Manager browser-verified wolverine world via verify_process_video_viewer (0 assertion errors, solver ran, honest HUD). M1 MILESTONE: MET with two attributed quality items. Fleet1 POWERED OFF via ssh (gcloud reauth challenge — see manual §12 auth note; disk persists, restart = gcloud start after owner login). WAVE-2 QUEUE: (1) burlington/outdoor root_motion_temporal_jump + 10-17mm slide overshoot; (2) P0-8 VFR/PTS audit; (3) P2-1 camera-motion upgrade; (4) P0-1b harvest ingest (clip+prelabel+roles, corpus at data/online_harvest_20260706); (5) P2-7a GVHMR spike; (6) P1-0/P1-1 prep (4-level visibility schema prereq); (7) tar-batch dispatch hardening + frame_idx-null producer cleanup + foot_contact_phases producer (grounding_refine); (8) keyless SA impersonation. VERIFIED=0 unchanged (gates are internal-val, promotion needs held-out ledger rows).
- [HARVEST BATCH-1 DONE 2026-07-06 ~13:1x, manager-verified] 8 videos / ~88min gameplay / 1.3GB, ALL verified genuine fixed-camera full-court doubles (ffprobe + screening spot-check; 7x1080p + 1x720p; 12 distinct rec channels). Manifest: data/online_harvest_20260706/manifest.json (25 probed entries w/ full provenance; manifest-vs-disk verified 8/8). 32 screening frames. 6 vetted next-batch candidates + 2 retries queued IN the manifest. KEY FINDING: this sandbox shapes googlevideo.com to ~200-280KB/s (28x slower than generic CDN; verified 3 ways incl. raw curl) — future bulk harvest belongs on fleet VMs, not the Mac. NEXT: P0-1b clip-to-rallies + prelabel + role assignment + combine-with-CVAT (wave 2). Owner ruled 8 videos sufficient for batch 1.
- [P0-1B HARVEST INGEST 2026-07-06, Codex p01b_harvest_ingest_20260706] Batch-1 ingest plumbing landed: 8/8 downloaded games processed into 43 stream-copy rally clips + per-clip provenance under data/online_harvest_20260706/rallies/ and runs/lanes/p01b_harvest_ingest_20260706/. Corpus card: corpus_card.{json,md}; role split train=29 / internal-val=11 / heldout-candidate-proposed=3; proposed held-out games are pwxNwFfYQlQ (Schwarz Pickleball, 720p Mega Courts) + vQhtz8l6VqU (Jack Munro, 1080p/25fps) for manager ledger registration only (heldout_eval_ledger.md not written). Dedup: eval collisions 0; cross-source dHash collision table flags 9 same-channel Rich Pickleball collisions between wBu8bC4OfUY and _L0HVmAlCQI for review. Spot-check: 10/10 sampled segments contain real rally play; active-gameplay coverage proxy on two videos = 84.7% and 95.6%. Prelabel: 40 one-clip shards excluding proposed held-outs; CPU smoke passed through run_wasb_ball.py on 50 frames; biometric rule encoded as session-only / no persistent ReID-gallery or face/appearance embeddings. CVAT export deferred because the 4-level visibility schema is not present. VERIFIED=0 unchanged.
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
- [WAVE-2 DISPATCHED 2026-07-06 ~18:xx, manager] 6 file-fenced Codex lanes + 1 Sonnet GPU lane in flight. (A) rootjump_slide_fix: burlington/outdoor root_motion_temporal_jump = per-frame ABAB flip between two self-consistent positions (burlington P5 0.85m frames 105-111; outdoor P1 3.0m pure-Y 648-654) — manager-ruled selection/identity bug, fix at source (no smoothing, thresholds frozen), slide→≤30mm, + frame_idx-null PRODUCER + foot_contact_phases producer (P2-8 wire path) + optional MAD bone-length smoothing wiring (owns pose_temporal.py). (B) p01b_harvest_ingest: rally clips + roles + dedup + corpus card from 8 games; held-out = PROPOSALS only (ledger manager-owned); PART 0 biometric default ruling baked in (session-only, no persistent ReID). (C) p08_vfr_pts: PTS frame-time tables + repo-wide constant-fps audit; process_video.py untouchable → deferred patches. (D) p21_cammotion: module-level masked LK+RANSAC+MAD upgrade + static-clip no-hallucination guard; pipeline wiring deferred-patch; RAFT flag-gated (weights need network). (F) p11_visibility_schema: 4-level occlusion taxonomy schema+CVAT round-trip, back-compat mandatory (P1-1 prereq). (G) dispatch_hardening: tar-batch transport + bounded retry adopting p06 body_dispatch_driver.py pattern (DEFAULT_REMOTE_HOST already 34.143.175.207). Sonnet p27a_gvhmr_spike restarting fleet1 for GVHMR external-pose+gravity on wolverine+burlington ×4 players + burlington-flip triangulation (flip in GVHMR-on-our-tracks ⇒ upstream tracks; else our placement/grounding). AUTH: gcloud token alive AND pickleball-fleet@ impersonation VERIFIED → wave-1 queue item 8 satisfied; fleet ops use --impersonate-service-account. PART 0 Roboflow key still blank → typed needs-decision STOP leads OWNER_CHECKIN_20260706.md (blocks only P1-0 aggregation). Specs at runs/lanes/<lane>/spec.md; lane fences recorded in specs + inflight_lanes.md. VERIFIED=0 unchanged.
- [ROBOFLOW KEY IN + P1-0 DOWNLOAD LANE 2026-07-06, manager] Owner answered the wave-2 STOP with the key -> stored data/credentials/roboflow.env (gitignored+600; .gitignore rules added for data/credentials/ + data/roboflow_universe*/); NORTH_STAR PART 0 item ticked with evidence. Sonnet network lane p10_roboflow_universe_20260706 dispatched: enumerate ALL Roboflow Universe pickleball projects, download every dataset with annotations to data/roboflow_universe_20260706/<project>/ + manifest (id/url/license-as-recorded/classes/counts/format/status), 30GB cap w/ ball/temporal-first priority, bounded retries, eval clips + online_harvest untouched, no repo-code edits. Offline aggregation/dedup/normalize/leakage-check (P1-0 gate: corpus card, dedup rate, temporal-vs-still split, 0-leakage vs eval clips) = follow-up Codex lane when downloads land. VERIFIED=0 unchanged.
- [ROOTJUMP LANE RULED 2026-07-06, manager] rootjump_slide_fix landed, CONDITIONAL-ACCEPT after personal spot-check. ROOT CAUSE (manager-verified in diff): hardcoded 30fps in placement.py visual root-step rewrite frame-index map (_frame_index(frame,30.0)->fps) — on 60fps burlington/outdoor the rewrite fetched ~half-timestamp frames => the ABAB root alternation; attribution_report.json shows fixed track==placement==skeleton delta 0.0 (was <=0.95m divergence). 7.9999 max_root_speed CLEARED as pre-existing designed write-speed cap (written_speed_capped_frames counters exist at HEAD — not lane-added gate-ducking). Also landed: frame_idx-null PRODUCER fix (1785/2653/680/359 nulls -> 0 across 4 clips + regression test) + foot_contact_phases in-pipeline producer (44/88/36/12 phases; grounding_refine consumes then self-kills on its own sanity gate — honest no-op, revisit post-P2-1 stance work). MAD bone-length wiring skipped by lane (bounded authority, primary-fix safety) -> wave-2 followup micro-lane candidate. UNPROVEN pending wave2_rootjump_verify lane (Sonnet, dispatched): REAL slide p95 via full-fidelity replay w/ smpl_motion.json scp'd read-only from fleet1 disk + reconciliation of 46.9/40.5mm (wave-1 closing metric) vs 26.36/22.58mm (body_grounding_quality) discrepancy. Wide suite: 157 focused tests green; lane's wide run interrupted in unrelated slow court benchmark at 1035-passed/0-fail — single clean local adjudication run booked for wave end. Lane code UNCOMMITTED until verify returns. VERIFIED=0 unchanged.
- [DISPATCH-HARDENING 2026-07-07, Codex dispatch_hardening_20260706] remote_body_dispatch now defaults to tar_batch transport (single tar.gz upload -> remote untar; single remote tar.gz -> download -> local safe untar) with bounded retry/backoff on retryable transport exits, while GNU rsync remains selectable fallback via RemoteConfig/`--transport rsync`. Local mocked-ssh proof only: >=245-file tar upload/download hash round-trip, transient rc=255 retry success, permanent retry bound, rsync fallback/timing tests all pass; required focused suites passed 215/215. Wide suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q`: 2925 passed / 9 failed / 18 skipped, classified in `runs/lanes/dispatch_hardening_20260706/report.json` as sandbox socket-bind failures, loaded-run timing flake, and VFR cross-lane/concurrent failures that passed focused rerun. live VM unproven -- sandbox; manager must do GPU-wave proof. VERIFIED=0 unchanged.
- [P11 VISIBILITY SCHEMA PARTIAL 2026-07-07, Codex p11_visibility_schema_20260706] Additive ball `visibility_level` landed for schema/CVAT export/CVAT import/dataset sidecars/gate inputs with explicit WBCE weights clear=1, partial=2, full=3, out_of_frame=3 and legacy bools preserved as ambiguous `legacy_visible`/`legacy_hidden` (no invented precision). Evidence: targeted round-trip/back-compat/focused visibility suites passed 105/105; real legacy reviewed artifacts validated under `runs/cvat_imports/2026_06_30/*/reviewed_boxes.json`; full wide suite `MPLBACKEND=Agg .venv/bin/python ... tests/racketsport -q` reached 2924 passed / 10 failed / 18 skipped, classified in `runs/lanes/p11_visibility_schema_20260706/lane_report.json` as sandbox bind/timing, fenced VFR io_decode cross-lane, and here-doc runner artifact. Storage policy audit still fails on missing allowlisted untracked fixture assets unrelated to this lane. VERIFIED=0 unchanged.
- [DISPATCH-HARDENING RULED 2026-07-06, manager] Lane G ACCEPTED. tar_batch transport now RemoteConfig DEFAULT (rsync selectable via --transport rsync), bounded retry/backoff typed-error after 3 attempts (fault-injection tested, no retry storm), 245-file mocked-boundary tar round-trip hash-identical, timing artifacts record transport+tar phases, process_video call sites untouched (215/215). Wide 2925/9-classified (6 socket-bind sandbox, 1 load-flake rerun-green, 2 io_decode cross-lane rerun-green) — wave-end clean local run adjudicates. LIVE-VM tar_batch proof rides the wave-end fresh-worlds run per lane NEXT. Commit STAGED for owner (auto-mode classifier blocks code commits under joint-commit rule; manifest at runs/manager/wave2_commit_manifest.md). VERIFIED=0 unchanged.
- [VISIBILITY-SCHEMA RULED 2026-07-06, manager] Lane F ACCEPTED (P1-1 prereq DONE). BallVisibilityLevel clear/partial/full/out_of_frame ADDITIVE end-to-end: schemas + CVAT export/import round-trip (test green) + dataset sidecars w/ wbce_weight (1/2/3/3) + honest legacy-ambiguity mapping (no invented precision); back-compat proven on 5 real artifacts incl runs/cvat_imports/2026_06_30/burlington reviewed_boxes.json; helper-module touches (label_review/cvat_video/ball_tracknet_cvat_dataset/cvat_gate_inputs) = wrappers' implementation, no fence collision. Wide 2924/10 same classified families as lane G (sandbox binds, load timing, io_decode cross-lane, heredoc-runner artifact) — wave-end clean run adjudicates. FLAG for closeout: audit_storage_policy exit-1 (caches + allowlist gaps, possibly new data/credentials/) — manager checks at wave end. Unblocks harvest lane's cvat_export probe + P1-1 WBCE training consumption. Commit staged in manifest. VERIFIED=0 unchanged.
- [HARVEST-INGEST RULED 2026-07-06, manager] Lane B ACCEPTED (P0-1b gate met). 8/8 games -> 43 stream-copy rally clips + per-clip provenance; spot-check 10/10 rally-play precision; coverage 84.7/95.6% on the 2 audit videos; dedup vs 4 eval clips = 0 collisions (dHash<=3, 2s sampling); corpus card + 40 one-clip GPU prelabel shards + CPU smoke green; CVAT export correctly self-deferred (schema probe predated lane F landing). MANAGER DATA RULING applied personally: 9 hamming-0 cross-source collisions between the two Rich Pickleball videos (channel furniture + same venue) STRADDLED train/internal_val -> whole channel same-sided to TRAIN (5 provenance role flips w/ role_note; internal_val now 6 clips / 4 sources; corpus_card manager_ruling_20260706 records the standing rule: harvest role granularity = per channel when same court). Held-out proposals REGISTERED as ledger rows HARVEST-1 (pwxNwFfYQlQ) + HARVEST-2 (vQhtz8l6VqU) in runs/manager/heldout_eval_ledger.md — zero prelabel exposure, excluded from shards. FOLLOW-UPS queued: CVAT-export rerun now that 4-level schema landed; GPU prelabel of 40 shards on fleet (after GVHMR/verify GPU work). Wide 2929/6 all socket-bind sandbox family — wave-end adjudication. Commit staged in manifest. VERIFIED=0 unchanged.
- [P2-7a GVHMR SPIKE RULED 2026-07-06, manager] Lane PASS, 8/8 runs (2 clips x 4 players), ~$1.30 GPU / 5.8min compute, first inference 53min from VM-up. HEADLINE FINDINGS: (1) roadmap premise CORRECTED — get_R_c2gv accepts external pose+gravity but is TRAINING-dataloader-only; shipped static-cam inference collapses to identity; lane's validated override shows injected-true-gravity ~= GVHMR-own on EVERY player of both tripod clips -> external-gravity value unproven on tripod; stress extreme-tilt/handheld BEFORE further investment (P2-7b/P2-7 planning input). (2) TRIANGULATION: GVHMR-on-our-crops shows ZERO flips at burlington p5 f105-111 (steps 0.01-0.13m) where our root ABAB'd 0.74-0.92m -> fault independently confirmed UPSTREAM in our placement/tracks fusion, corroborating rootjump lane's 30fps-bug attribution with a second instrument. (3) GVHMR global-root jitter competitive-to-smoother per-player (caveats: our numbers include the now-fixed bug + our path has designed speed-caps, theirs raw) — P2-7 bracket material only; single-person hardcoding (no scene coherence) = separate engineering budget in any challenger decision. (4) NEW GOTCHA: Mac->VM binary transfers flake non-deterministically >~10MB (fresh-random-data repro; not IPQoS; undiagnosed) — chunk+retry workaround used; watch tar_batch live proof for this; --transport rsync fallback stands; chunked-tar = wave-3 candidate. Fleet1 RUNNING, IP unchanged 34.143.175.207, ~/gvhmr_spike env retained, reserved for wave-end GPU work. Artifacts runs/lanes/p27a_gvhmr_spike_20260706/ (report.md, metrics.json, triangulation_burlington_p5.json, reusable harness scripts). VERIFIED=0 unchanged.
- [P2-1 CAMMOTION RULED 2026-07-06, manager] Lane D ACCEPTED. Hardened default camera-motion module: person-masked LK + MAD flow-track filter + temporal MAD+Gaussian smoothing + processing_scale 0.6; img1605 handheld beats legacy on ALL 3 proxies (inlier 0.7675->0.8949, jerk 2.621->2.453, court-line stability 14.84->11.75px); wolverine static guard IMPROVED (drift p95 0.563->0.427, max 9.485->0.676 — legacy was hallucinating motion spikes); masked>unmasked everywhere; runtime HALVED 98->50ms/frame (under 60ms bar). RAFT-small honestly not_enabled_pending_weights (needs one network prefetch — wave-3 opportunistic; LK+MAD already wins). 2 deferred patches (default_stage_wiring, placement_consumption_hook) git-apply-check clean -> wave-end integration micro-lane (with C's patches + B's CVAT-export rerun + MAD bone-length wiring on now-free pose_temporal.py). Wide 2929/6 all socket-bind sandbox family. Commit staged in manifest. VERIFIED=0 unchanged.
- [P0-8 VFR/PTS PARTIAL 2026-07-07, Codex p08_vfr_pts_20260706] Additive frame_times.json path landed in io_decode plus owned BALL/contact/rally/arc/fill consumers and CLI options; synthetic VFR contact proof shows PTS frame-2 contact about 0.240s vs wrong constant-30fps 0.0667s (>1 real-frame drift). Audit artifact complete: runs/lanes/p08_vfr_pts_20260706/fps_audit.{json,md} (1363 hits, unexamined=0; FIXED 201 / JUSTIFIED 1067 / DEFERRED 95). process_video.py remains fenced; deferred patch applies clean at runs/lanes/p08_vfr_pts_20260706/deferred_patches/process_video_frame_times.patch. Focused suites 188/188 + cli-help 274/274 + scaffold/dead-code/storage green; full wide suite interrupted after 30:03 at 1038 passed / 4 skipped / 0 observed failures in unrelated court_finding_technology_benchmark.py:3267, so P0-8 gate stays PARTIAL until a clean full-suite run completes. Report: runs/lanes/p08_vfr_pts_20260706/report.json. VERIFIED=0 unchanged.
- [P1-0 DOWNLOAD RULED + AGGREGATION DISPATCHED 2026-07-06, manager] Download lane PASS: 75 Universe projects enumerated (17 terms, saturation observed; universe web UI is a Cloudflare 403 bot-wall — authenticated api.roboflow.com/universe/search is the working path), 65 downloaded COCO-first-try, 6.91GB/~35min no throttling (Roboflow CDN fine from this Mac, unlike googlevideo), 51/65 likely video-sequential (temporal fuel), 10 failures all Roboflow-side (9 no-exported-version incl 5.6k-raw t-vqtdd/pickle-y2oed, 1 dead 404), key grep-verified never leaked. MANAGER CURATION RULED: 2 confirmed padel-contaminated + 1 ambiguous -> adjacent_sport_aux bucket (P1-1 multi-sport aux fuel, never pickleball diversity); fork/mirror pairs -> content-hash dedupe keep-most-complete; 9 no-version projects skipped permanently. Aggregation Codex lane p10_roboflow_aggregate dispatched (index-based, NO image copying — Mac volume at 96%/12GB free, flagged as owner quick-win): normalize COCO->our point-label conventions (visibility_level honestly absent for public data), dHash dedup within-corpus + vs 4 eval clips (T4 0-leakage discipline), temporal-vs-still split, corpus card. VERIFIED=0 unchanged.
- [P1-0 ROBOFLOW AGGREGATE 2026-07-07, Codex p10_roboflow_aggregate_20260706] PASS (pretrain/aux corpus only; no training, no promotion): 65/65 downloaded COCO datasets parsed with 0 parse failures into index-only artifacts under `data/roboflow_universe_20260706/aggregated/` (no copied images). Buckets: core_pickleball 59, adjacent_sport_aux 3, excluded_duplicate 3, excluded_dead 10 (9 no-version + 1 dead 404). Merged index keeps 61,260 samples after dHash 8x8 dedup (110,003 considered; 48,743 duplicates; rate 0.443106); temporal split 84,459 sequence / 25,903 still; protected eval hash check 0 collisions across 35 eval hashes. Fork/mirror mappings and the 12-frame ambiguous tennis spot-check are recorded in `corpus_card.json`; du delta 339,928 KiB <=1GB. Verification: Roboflow tests 5/5, scaffold/dead/storage checks pass after cache cleanup; full wide `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` = 2948 passed / 6 failed / 18 skipped, all 6 SANDBOX-SUSPECT localhost-bind review-server tests (lane forbids localhost binds). VERIFIED=0 unchanged.
- [ROOTJUMP VERIFY RULED 2026-07-06, manager — CLAIM DOWNGRADED] Independent replay lane returned PARTIAL with three load-bearing findings. (1) smpl_motion.json NEVER MATERIALIZED anywhere (S1 speed lane set write_body_monoliths=False; in-memory only) — no offline replay can be full-fidelity, incl the fix lane's. (2) METRIC RECONCILIATION: 46.9/40.5mm = grounding_metrics.max_foot_lock_slide_m = THE GATED metric (foot_slide_gate threshold 0.03 keys off MAX; blockers:[foot_slide_gate_failed]); 26.36/22.58 = foot_lock_slide_p95_m, gated by NOTHING and already <30mm pre-fix — the fix lane's acceptance (and the manager's spec bar) tracked the wrong statistic; the gated MAX is computed INSIDE GPU BODY on never-persisted camera-frame samples (worldhmr.py:280 via orchestrator) — §18 lesson recurrence: only a fresh GPU dispatch can measure it. (3) BLOCKER-GONE DISPUTED: fix-lane counterfactual (root:=anchor snap) clears blockers BY CONSTRUCTION (its 7.9999 sits exactly on the pre-existing 8.0 smoothing clamp = saturation signature); verify counterfactual (preserve BODY root-to-anchor residual) shows LARGE-BUT-INCOMPLETE reduction (burlington 24->5 jumps 54.97->11.25mps; outdoor 55->18 188.6->30.5mps) with residuals AT the originally-cited frames; verify also found+corrected 2 real bugs in the fix lane's validation (undistort=True mismatch — ablated, not explanatory; provenance-path bug that made outdoor validate bbox-only) and the residual SURVIVED. RULING: 30fps fix = real + correct + commits as a FIX (manifest flipped to GO with scoped wording); gate-clearance claim = PENDING the wave-end fresh GPU run on corrected tracks (now the sole decisive proof for blocker + max-slide<=30mm; composed with all wave-2 landings + tar_batch live proof). Also flagged: stale CLIs (apply_placement.py lacks --foot-contact-phases-out/--camera-motion; refine_body_grounding.py lacks --xy-translation-enabled) -> integration micro-lane; smoothing_flag vs temporal_smoothing_reset schema-name mismatch in stand-in replays (logged); grounding_refine kill_recommended=True on all 4 clips incl untouched 30fps = pre-existing default characteristic. Evidence: runs/lanes/wave2_rootjump_verify_20260706/ (replay_summary.json, smpl_motion_fetch_attempt.md, cross-check + ablation dirs). VERIFIED=0 unchanged.
- [P0-8 VFR RULED + INTEGRATION LANE DISPATCHED 2026-07-06, manager] Lane C ACCEPTED: frame_times.json PTS table + constant_fps_assumed provenance fallback in io_decode; 1363-hit constant-fps audit with ZERO unexamined (95 DEFERRED in then-fenced files); whole BALL timing chain threaded VFR-correct (arc solver/chain/adapters/bounce/physics-fill/rally-gating/event-fusion; 188/188 focused); synthetic-VFR proof: constant-fps off by 0.173s (>4 frame durations) where PTS exact; process_video wiring = apply-clean deferred patch. Wide run interrupted in the same slow court benchmark as lane A's (2nd occurrence — benchmark file gets split handling in adjudication). ALL SIX wave-2 Codex lanes now RULED (A fix-GO/gate-pending, B/C/D/F/G accepted). INTEGRATION lane dispatched (all fences lifted except roboflow-aggregate): applies C's process_video patch + D's default-stage/placement patches, MAD bone-length wiring w/ wrist-harness proof, CVAT export rerun w/ 4-level visibility + held-out exclusion assert, stale CLI flags (apply_placement/refine_body_grounding), resolves the 95 now-unfenced audit hits. After it: manager clean wide-suite adjudication -> fresh-worlds GPU closing run (decisive for rootjump gate + gated max_foot_lock_slide + tar_batch live). VERIFIED=0 unchanged.
- [WAVE2-INTEGRATION 2026-07-07, Codex wave2_integration_20260706] Integration lane PASS with scoped caveats: P0-8 process_video frame_times patch was clean before P21 context then faithfully hand-ported; P21 default-stage/placement patches applied; camera_motion remains stage-ordered before placement but is default-OFF behind --enable-camera-motion after copied p06 placement stability found Wolverine jitter_after_p90_mean 2.2440->2.2670 (kill criterion). MAD bone-length smoothing wired default-ON after p06 Wolverine+Burlington-60fps measurement showed no regressions and wrist peak delta 0 frames; CVAT harvest export rerun wrote 6 validated 4-level visibility task packages with pwxNwFfYQlQ/vQhtz8l6VqU excluded; stale CLIs gained apply_placement --foot-contact-phases-out/--camera-motion and refine_body_grounding --xy-translation-enabled; fps audit delta resolved 37/37 owned deferred hits (3 FIXED, 34 JUSTIFIED-CONSTANT). Focused touched-area suites 199/199 green; scaffold/dead-code/storage/torch checks green after generated-cache cleanup; wide split: non-slow suite 2909 passed / 7 classified failures / 15 skipped (6 sandbox localhost-bind, 1 p10 Roboflow fenced markdown inventory), excluded court benchmark 41/41 green in 22m23s. Artifacts: runs/lanes/wave2_integration_20260706/. VERIFIED=0 unchanged.
- [INTEGRATION RULED + CLOSING RUN DISPATCHED 2026-07-07, manager] Integration lane ACCEPTED: all 3 deferred patches composed (C's hand-ported, stated); MAD bone-length -> smoothing weights DEFAULT-ON (wrist delta 0 on wolverine+burlington-60fps); CVAT export = 6 schema-validated 4-level task packages, held-out excluded by assertion; apply_placement/refine_body_grounding parity flags; 37/37 owned audit hits resolved (3 FIXED/34 JUSTIFIED). KILL-CRITERION FIRED HONESTLY on camera-motion default stage: wolverine placement jitter p90 2.244->2.267 (~1% regression) -> wired flag-gated --enable-camera-motion DEFAULT-OFF; wave-3 design note: motion-CONDITIONAL auto-enable (on for handheld, off static). Slow court benchmark quantified: 41 tests / 1343s standalone (adjudication will reuse this green + run wide-minus-benchmark on final tree). Wide 2909/7 standard classified families (+1 CROSS-LANE doc-inventory hit from still-running aggregation lane). Fresh-worlds CLOSING run dispatched on fleet1 (Sonnet): 4 clips E2E --force, tar_batch LIVE proof w/ rsync fallback A/B on flake, DECISIVE for burlington/outdoor root-jump blocker + gated max_foot_lock_slide_m<=30mm + regression clips. VERIFIED=0 unchanged.
- [P1-0 COMPLETE 2026-07-07, manager] Aggregation lane ACCEPTED -> P1-0 done end-to-end (download 65 datasets + index-based aggregation). Corpus: 110,003 frames considered, 48,743 dHash dupes removed (44.3% — honest: video-sequential + fork near-identicals), 61,260 kept samples across buckets core_pickleball=59 sources / adjacent_sport_aux=3 / excluded_duplicate=3 (fork maps recorded) / dead=10; LEAKAGE vs 4 protected eval clips = 0 (T4 discipline held); loader smoke green; 339MB delta (index-only, no image copies); corpus_card.md registered in allowlist same-lane. Wide 2948/6 all socket-bind sandbox family. P1-1 can now consume: public pretrain corpus + 4-level visibility schema + WBCE weights — owner in-domain data remains the finisher per standing rule. Wave-2 queue item 6 FULLY DONE. Manager clean-suite adjudication launched on final tree. Commit staged in manifest. VERIFIED=0 unchanged.
- [WAVE-2 SUITE ADJUDICATION GREEN 2026-07-07, manager-run] Clean local wide suite on the final composed tree: 2916 passed / 0 failed / 15 skipped in 5m39s (+ court benchmark 41 passed standalone from integration lane = 2957 total). EVERY lane-classified failure confirmed sandbox-only (6 socket-bind review-server tests, io_decode cross-lane hits, flight-sim load timing, append_lock heredoc artifact — ALL pass locally). Zero real failures from 8 concurrent lanes. Suite condition CLOSED on all wave-2 rulings. audit_storage_policy.py: status=pass exit 0 on settled tree (lane F's mid-wave exit-1 = transient concurrent-lane state). Remaining open: fresh-worlds closing run (in flight on fleet1) -> manager browser verify -> prelabel -> teardown/handoff. VERIFIED=0 unchanged.
- [CLOSING RUN RULED 2026-07-07, manager] Fresh-worlds on composed wave-2 code, 4/4 clips, 29.6min total. ROOT-JUMP FIX VINDICATED: outdoor 55->0 jumps (fully clean, was 188mps spikes), burlington 24->1 (survivor 10.04 vs 10.0 review floor = 0.4%-over marginal at f57/p1); wolverine+img1605 regression-clean (0 jumps). TAR_BATCH LIVE-PROVEN: 4/4 first-attempt, 18-67MB payloads (above the documented ~10MB flake threshold), zero retries/fallback. Plumbing confirmed: phases produced+consumed (44/116/16/12), frame_times all clips, camera-motion default-OFF, wrist 0-drift 4/4, MAD active, GPU 15.4-15.6ms/person-frame. FOOT-SLIDE GATE STILL OPEN: burlington max 46.9->40.6mm (improved, over 30 bar), outdoor 40.5->56.0mm (WORSE), wolverine 0->18.4 (PASS but shifted) — p95 under bar everywhere => outlier-frame driven; shift-on-3-of-4 implicates a wave-2 interaction; PRIME SUSPECT = MAD bone-length default-ON (its offline validation could not see the real footlock chain — monolith hard wall). MAD A/B lane dispatched on fleet1 (wolverine+outdoor MAD OFF, ~20min): guilty => default flips OFF one-line; innocent => wave-3 diagnosis with fresh per-frame evidence. grounding_refine 4/4 self-kill PREDATES wave-2 (verify-lane evidence) — booked as wave-3 linked diagnosis. img1605 0-mesh-frames pre-existing follow-up stands. COST HONESTY: fleet1 uptime 4h10m ~$5.00 total; ~$2.5 of that was manager idle-reserve between GVHMR and closing run — logged as a real dent in the idle-spend rule (call: restart-risk protection for the decisive run; next time STOP the VM for gaps >1h). Wave closes after MAD A/B + prelabel return: root-jump WON, slide overshoot (pre-existing) = wave-3 priority 1. VERIFIED=0 unchanged.
- [BROWSER VERIFY PASS 2026-07-07, manager-run] Fresh burlington + wolverine worlds verified in the real viewer (verify_process_video_viewer, headless chromium): assertion_errors=[] both; burlington renders 4 correctly-placed skeletons + ball arc + honest HUD (471/600 measured / 117 predicted / 12 hidden; 2 notices: missing mesh vertices, missing paddle pose; contacts 0); screenshots at runs/manager/wave2_browser_verify/{burlington2,wolverine2}/. LESSON (booked): the verify CLI takes replay_viewer_manifest.json, NOT PIPELINE_SUMMARY.json — wrong file yields a fail-closed artifact_type rejection with an empty out-dir and a bare canvas-timeout (cost 3 misdiagnosis probes; the fail-closed viewer behaved exactly as designed). Remaining before wave close: MAD A/B (fleet1) + prelabel (fleet2) reports -> STOP fleet1 -> handoff. VERIFIED=0 unchanged.
- [PRELABEL RULED 2026-07-07, manager] p01b_prelabel lane ACCEPTED — PASS with zero failures: fleet2 provisioned first-try (70s ssh-ready, EXCLUSIVE_PROCESS confirmed), 40/40 shards prelabeled (WASB zero-shot-tennis fallback per dispatch doc = the documented P1-2 SST seed, terminology corrected honestly), held-out assertion CLEAN at every checkpoint (0 refs to pwxNwFfYQlQ/vQhtz8l6VqU anywhere), 46.7 GPU-min ~0.79x realtime, sidecars 40/40 schema-valid (visible-fraction 48.5-73.9%), checkpoint sha256-verified via on-VM gdown (6.1MB), 1.09GB clip sync in ONE clean rsync pass — the >10MB Mac->VM flake did NOT reproduce here either (2nd independent non-repro; downgrading gotcha severity to intermittent/VPN-state in handoff). VM DELETED + list-confirmed ~$1.3 total. FLAGGED: WASB undocumented deps (hydra-core/omegaconf/matplotlib/pandas/Pillow/torchvision/tqdm — future lanes note), harvest manifest duration_s sometimes wrong vs decoded (harvest-lane bug, harmless here), gdown --id flag removed upstream. P0-1b COMPLETE THROUGH PRELABELS: rally clips + roles + ledger rows + dedup + corpus card + CVAT 4-level task packages + review selection (480 frames/6 clips) + prelabels for those exact 6 clips — labeling factory (P0-4) is go for wave 3. Sidecars = direct P1-2 SST seed. VERIFIED=0 unchanged.
- [MAD A/B RULED 2026-07-07, manager] Lane ACCEPTED w/ praise for structural pushback: discovered fleet1 remote checkout 16 COMMITS STALE (pinned 5b9f132ee, missing 1041b4465 — BODY dispatch ships DATA, never threed/ code), making the requested baseline meaningless; lane synced real code and ran TRUE ON/OFF arms (4 dispatches, 31.9 GPU-min, disclosed). VERDICT: MAD INNOCENT 3 ways (baseline never ran MAD; true-ON engages 0 frames both clips — no bone crosses 6-sigma; ON-vs-OFF delta ~1e-7m). MAD stays DEFAULT ON. Closing-run conclusions VERIFIED intact despite drift (worldhmr/orchestrator md5-identical remote-vs-local => slide numbers valid; root-jump computed locally post-placement). Slide-shift attribution moves to the placement 30fps fix's error redistribution — guard counters scale w/ regression (outdoor divergence_snap 478 + speed_cap 718 vs wolverine 22/83) => WAVE-3 #1 diagnosis trail. WAVE-3 #2: BODY dispatch code-sync/version-stamp (fail-loud on drift). Remote+local git state verified restored clean.
- [WAVE-2 COMPLETE 2026-07-07, manager-verified closeout] ALL LANES RULED (10 dispatched + 3 verify/AB), fleet DOWN (fleet1 STOPPED disk-intact, fleet2 DELETED), tracked tree committed clean (8 commits: 21154883/450348763/a80c1eaee/e4d314cd2/4763d4703/1041b4465/bf902c592 + settings). SCORECARD vs owner queue: (1) root-jump WON — outdoor 55->0, burlington 24->1 @10.04 vs 10.0 floor (96-100% cut; 30fps frame-index root cause, GVHMR-triangulated + adversarially verified); slide gate still FAIL (burlington 40.6 / outdoor 56.0 vs 30mm bar; pre-existing overshoot + placement-fix redistribution; per-frame evidence + guard-counter trail banked -> wave-3 #1); worlds browser-verified (assertion_errors=[], honest HUD). (2) P0-1b COMPLETE through prelabels: 43 rally clips, roles (Rich-channel same-sided ruling), HARVEST-1/2 ledger rows, corpus card, CVAT 4-level packages + 480-frame review selection, 40/40 WASB prelabels (P1-2 SST seed). (3) P0-8 DONE: 1363-hit audit 0-unexamined, PTS tables, BALL chain VFR-correct, synthetic VFR proof. (4) P2-1: module hardened (handheld all-3-proxies win, 2x faster, static guard) + MAD wired (proven no-op on eval clips) + stage flag-gated default-OFF per kill-criterion (wave-3: motion-conditional enable). (5) P2-7a DONE: gravity-injection premise CORRECTED (no tripod benefit; train-dataloader-only at source), single-person cost quantified, spike harness banked. (6) P1-0+P1-1-prereq DONE: 4-level visibility end-to-end; 65 Universe datasets 6.9GB; aggregated corpus 61,260 samples, 44.3% dedup, 0 eval leakage. (7) tar_batch DEFAULT + LIVE-PROVEN 4/4 (18-67MB, 0 retries); frame_idx + foot_contact_phases producers landed (grounding_refine consumes, self-kills honestly — wave-3 linked diagnosis). EXTRAS: SA impersonation confirmed (queue item 8), commit-permission encoded (.claude/settings.json), suite 2957 green + storage audit pass, fleet1 CODE-DRIFT discovered/contained/booked, disk cleanup script staged for owner (~48GB; CV_pipeline KEEP per owner). COST: ~$7-8 GPU total (fleet1 ~5.2h incl ~$2.5 manager idle-reserve dent — logged; fleet2 $1.3). M-milestone posture: M1 intact; slide gate = the sole carried blocker. WAVE-3 QUEUE (priority): (1) slide-max outlier diagnosis on placement-fix redistribution (guard-counter trail); (2) BODY dispatch code-sync/version-stamp hardening; (3) grounding_refine 4/4-self-kill diagnosis (phase quality); (4) P0-4 labeling factory launch (CVAT packages + prelabels ready); (5) P1-1 pretrain warm-start on aggregated corpus; (6) camera-motion motion-conditional default; (7) img1605 0-mesh-frames; (8) RAFT weights prefetch. VERIFIED=0 unchanged (gates internal-val; promotion needs held-out ledger rows).
- [HARVEST REVIEW LABELS COMPLETE 2026-07-07, owner+manager] Owner labeled ALL 6 review clips (~480 frames) across cloud CVAT (2) + the w3_labelfactory local CVAT (4); manager verified every export (parse/counts/prelabel-concordance) BEFORE filing: ~274 human-verified ball boxes, machine-agreement med 0-7px, 60 machine-missed balls captured, held-out untouched. Systematic UI trap caught+fixed: visibility_level 'full' read as 'fully visible' -> deterministic full->clear remap on 4 clips (raws + MANAGER_NOTEs preserved); 5 dup frames flagged for import dedupe; partial under-used (noted honestly — clear-vs-absent signal dominates, fine for WBCE seed). Set filed at cvat_upload/exports/harvest_review_20260707/ w/ README (import rules inside). This is P1-1's first human-verified in-domain-adjacent seed + P1-2 teacher-eval material; owner throughput ~6 tasks/~2h incl tooling friction (P0-4 labels/hour datum). Court-keypoint task set (1 frame/source, metric-15pt) queued as owner's next 10-min session. VERIFIED=0 unchanged.
- [IOS P0-10a DISPATCHED 2026-07-07, wave-2 manager session (owner-support), CROSS-MANAGER NOTICE] Owner-directed disjoint-domain lane ios_p010a_20260707 dispatched from the still-open wave-2 manager session: P0-10 APP-SIDE slice only — real ARKit session (ARSession/ARWorldTrackingConfiguration; ARKitSetupPassSidecar becomes a real producer; PTS-timestamped, CoreMotion gravity kept as cross-check), capture-policy ENFORCEMENT (EIS off/AE-AF-WB lock/landscape + enforcement outcome into sidecar), minimal H0 profile-capture flows (5-step guided checklist), all behind a device-free fake-AR-provider abstraction with XCTest coverage. FENCE: ios/** + its lane dir ONLY — zero overlap with wave-3's racketsport world; P0-10(b) server wiring explicitly excluded (wave-3+ owns it). Codex sandbox cannot xcodebuild -> lane self-verifies via swiftc -parse + authored tests; manager runs the local xcodebuild/simulator verify leg after landing; owner device recording remains the final gate step (unblocked whenever owner has the phone, no court needed for the smoke). VERIFIED=0 unchanged.
- [ROADMAP AUDIT + PART VI 2026-07-07, manager docs pass] Owner-requested North-Star pass-through complete. DIRECTION AUDIT: strategy CONFIRMED (data-first validated by 2 waves of measured evidence; wave-3's two diagnoses converging on ONE root cause — weak bilateral contact phases explain BOTH the slide-max outliers AND grounding_refine 4/4 self-kill — is the root-cause-composition thesis working). Named risks booked in the doc: BALL training not yet started (wall 0.6969 vs 0.7248 untouched — wave 4 = first training wave, internal-val only), owner in-domain data = THE critical-path bottleneck, P0-10 ARKit slipped 3 waves (now hard-scheduled ≤ wave 8), slide-gate statistic question (max vs p99+outlier-cap) framed as a conditional needs-decision STOP — never a silent re-tune. NORTH_STAR_ROADMAP.md updated on main (docs-only; file-disjoint from live w3 lanes): PART III checkboxes reconciled with dated evidence (P0-1/P0-1b/P0-2/P0-6/P0-8/P1-0/P2-7a ticked; P0-7 phase-1/P0-9 schema/P2-1 module/P1-1-P1-2 staging noted; wave-3 STATUS lines explicitly UNRULED), PART 0 SA-impersonation confirmation, II-B GVHMR gravity-premise correction, Part IV NEW RULES 11-14 (remote code-sync/version-stamp before trusting VM metrics; exact gated metric key in acceptance; lane-isolation reality — file-fenced local lanes + wave-end clean-suite adjudication; MANDATORY wave-end docs reconciliation), PART V wave-2/3 evidence pointers, and NEW PART VI WAVE EXECUTION PLAYBOOK (VI.0 invariant lifecycle; VI.1 wave-3 exit contract; VI.2 wave-4 first-training-wave lanes W4-A..F; VI.3-VI.6 waves 5-8 mapped to M2-M5; VI.7 per-wave invariants + critical-path guard). MASTER_PLAN.md header gains the doc-role supersession note. CONFIRMED STALE: CAPABILITIES.md + PIPELINE_STATUS.md (both 2026-07-05, pre-wave-2) — refresh is now a wave-3 close requirement (rule 14). Fact base: 3 read-only audit lanes (dirty-tree, canonical-truth, task-evidence map). No code touched; in-flight w3 lane files untouched. VERIFIED=0 unchanged.
- [DINKVISION UI LANE QUEUED 2026-07-07, wave-2 manager session] Owner delivered the brand board (DinkVision; paddle-eye mark; cream/courtGreen/ink/ballYellow + trail palette; playful-clean). Full UI spec WRITTEN and QUEUED (runs/lanes/ios_ui_dinkvision_20260707/spec.md) — dispatches immediately after ios_p010a lands + local xcodebuild verify passes (sequenced: never two lanes in ios/ simultaneously). Scope: DesignSystem tokens; signature animations (splash eye-BLINK -> iris circular-mask expand into app; reusable BallTrailLoadingView from the board's speed-streak motif); 4-tab shell (Record default w/ ONE-TAP capture law composing p010a's ARKit+policy+sidecar pipeline + policy chips; Replays via existing playback; Stats as honest sample-data placeholder cards; Settings w/ H0 profile flows); app icon assets; ios/README design-system section (doc currency per owner). NORTH_STAR alignment: P0-10 app-side + early P7-1 shell; server-fed stats explicitly NOT faked. VERIFIED=0 unchanged.
- [DOC-INVENTORY CROSS-LANE NOTE 2026-07-07, manager docs pass] `test_markdown_doc_inventory_stays_small_and_explicit` currently FAILS on 6 unregistered .md files created by live wave-3 work: `cvat_upload/court_keypoints_20260707/OWNER_COURT_KP_GUIDE.md`, `cvat_upload/exports/harvest_review_20260707/README.md` + 4 per-clip `MANAGER_NOTE.md`. Adjudicated PRE-EXISTING vs the docs pass (docs pass touched only 3 already-registered root .md; verified 14/15 pass with this sole failure). Wave-3 closeout must register them same-wave (allowlist entry or relocate notes under runs/) per the drift lesson + new Part IV rule 14. All other doc-consistency tests green after the roadmap edits.
- [IOS P0-10A 2026-07-07, Codex ios_p010a_20260707] SCOPED PASS, app-side only: ARKit producer, PTS-aligned AR frame sidecar samples, policy enforcement/reporting, and minimal H0 profile checklist landed under `ios/` with XCTest sources authored. Verification in this sandbox: `xcrun swiftc -parse` on every touched Swift file passed; `xcodebuild`/simulator and physical-device recording are explicitly deferred per lane instructions. Evidence: `runs/lanes/ios_p010a_20260707/lane_report.json`. P0-10(b) server wiring untouched; owner device recording remains the gate. VERIFIED=0 unchanged.
- [IOS P0-10a RULED + DINKVISION UI DISPATCHED 2026-07-07, wave-2 manager session] p010a ACCEPTED after manager local verify: real ARKit producer (ARSession/ARWorldTrackingConfiguration, PTS-aligned arkit_frame_samples, fail-closed unavailable-reasons), requested-vs-achieved CapturePolicyEnforcer, H0 ProfileCaptureFlowState, device-free DeterministicARSessionProvider — swift build + FULL swift test 174/174 GREEN after 2 manager one-liner fixes the lane's parse-only gate could not catch (missing return CaptureSidecarWriter.swift:85; missing `import PickleballCore` in ProfileCaptureFlowTests) — LESSON booked: swiftc -parse is syntax-only; local build leg is mandatory for ios lanes. Zero iOS simulators were installed on this Mac — platform download kicked (disk now 45GB free post owner cleanup); full xcodebuild sim pass rides the UI lane's verify. UI lane ios_ui_dinkvision DISPATCHED (gate passed): manager-authored pixel-law mockups (8 app screens + website, HTML/CSS + rendered PNGs at runs/lanes/ios_ui_dinkvision_20260707/mockups/, owner copy at ~/Desktop/DinkVision_Mockups/) — eye-blink splash, BallTrail loader, one-tap record law, 4-tab shell, honest sample-data stats. VERIFIED=0 unchanged.
- [IOS UI DINKVISION 2026-07-07, Codex ios_ui_dinkvision_20260707] SCOPED PASS, app-UI lane only: DinkVision SwiftUI shell landed under `ios/` with mockup-token design system, paddle-eye app icon set, splash blink + iris transition with reduced-motion fallback, reusable BallTrail loader, Record-default tab composing `CaptureViewModel`/P0-10a policy/profile/sidecar flow, real local-capture Replays list via `CaptureLibrary`, sample-watermarked Stats, and H0 Profile checklist/settings. Verification here: 6/6 touched Swift files `xcrun swiftc -parse` clean; `swift test --package-path ios --disable-sandbox` = 174 passed / 0 failed / 1 skipped for package modules. App-hosted XCTests for splash/record-flow/policy/replay datasource are authored but require manager xcodebuild/simulator because this lane forbids that leg. Storage audit still fails on pre-existing wave-3 court-keypoint untracked large files, not this lane. Evidence: `runs/lanes/ios_ui_dinkvision_20260707/report.json`. VERIFIED=0 unchanged; physical iPhone first launch/recording remains the gate.
- [TECH_BLUEPRINTS SHIPPED 2026-07-07, manager (Fable final session)] Owner-ordered succession handoff: NEW root doc `TECH_BLUEPRINTS.md` (2,886 lines, registered in allowlist) = PART A successor-manager primer (doc precedence, judgment-heuristics-as-rules, bright-line STOP test, boot ritual, dispatch cheat-sheet) + PART B cross-pillar rulings (BALL VERIFIED bar clarified: 0.7248 = promotion milestone, M1 0.90 = the VERIFIED flip; ONE contact producer via event_fusion; paddle artifact BUILT-NOT-WIRED until P3-1; process_video stage inserts serialize through ONE integration lane; profile_registry schema single-owner = DATA pillar; v1_done_harness owned by PF-4; rally-end-cause owned by BALL chain; dependency spine: confident-phases→P3-1→P1-4 arcs→{P3-5, PF-1, coaching-S1}→PF-2; successor top-12 mistakes w/ binding guardrails) + PART C nine pillar blueprints (DATA+identity, BALL2D, BALL3D, BODY, PADDLE, COURT/NET, FUSION, COACHING, SPEED/PROD — each: final ruling, measured state, exact build plan w/ grep-verified file targets + recipes + exact gated keys, decision trees ending in typed STOPs, DO-NOT lists, web-verified external bets 2026-07-07). Method: 34-agent workflow (8 opus drafters + 16 adversarial verifiers (repo-reality + successor-confusion) + 8 fixers + xhigh completeness critic; 62 verifier findings applied, 0 rejected) + 1 data-pillar agent; ~3M tokens. Reading order wired: CLAUDE.md step 2, NORTH_STAR PART VI header, wave3 boot prompt. Known open items booked IN the doc: owner manifest lacks role field (P0-3), corpus_dashboard.py to-build (P0-4 gate), ReID acceptance key pre-registration, P7-4b consent still blank. Staging: runs/manager/blueprints_20260707/. VERIFIED=0 unchanged.
- [DINKVISION UI RULED + LIVE-VERIFIED 2026-07-07, wave-2 manager session] UI lane ACCEPTED, full verification chain complete: (1) lane report PASS on all 6 acceptance items; (2) manager local build caught 1 Swift-6 strict-concurrency error (Sendable self-capture in CameraCaptureController closure -> capture the @unchecked-Sendable recorder instead; parse-gate-invisible class, 3rd such catch today); (3) app LAUNCHED live on DinkVision-Test sim (iOS 26.5/iPhone 17 Pro, platform+device installed this session): splash iris-expand caught mid-animation, Record home matches manager pixel-law mockups (court stage, policy chips honest-yellow on sim, permission primer card, dimmed ring = correct no-camera state), real screenshots delivered to ~/Desktop/DinkVision_Mockups/real_app_*.png beside the mockups; (4) app-hosted suite: ALL 20 DinkVision tests green (capture flow 6, splash state machine 5 incl reduced-motion, live overlay 6, app flow 3) — xcodebuild's TEST FAILED verdict traced to ONE PRE-EXISTING device-dependent ANELatencyBenchmark test that cannot run on simulator (predates today; classified, not ours); SPM suite 174/174 x2. Landings committed+pushed (8abf694f8 ARKit, 4790b571e UI). iOS arc lessons: swiftc -parse is syntax-only (3 type/concurrency bugs slipped it — local build leg is MANDATORY for ios lanes, now proven); bg-task kill-sweep recurred once (foreground retry worked). REMAINING for P0-10 gate: owner 5-min on-device recording smoke (no court needed) -> sidecar with real ARKit pose/gravity -> server world using them. VERIFIED=0 unchanged.
- [WAVE-3 COMPLETE 2026-07-07, manager-verified closeout] ALL LANES RULED (12 dispatched + 3 repair/verify rounds + 3 micro); fleet DOWN (fleet1 STOPPED disk-intact + snapshot pickleball-fleet1-snap-20260707 READY as fan template; fan1 + both trainer VMs DELETED list-confirmed). HEADLINE — FOOT-SLIDE GATE GREEN 4/4 on fresh GPU proof @ ad75c875c: grounding_metrics.max_foot_lock_slide_m burlington 40.6→20.25mm / outdoor 56.0→22.50mm / wolverine 17.98 / img1605 16.66 (bar 30mm FROZEN; p95 all <12mm; blockers empty 4/4; root jumps 0/0/0/0 incl. burlington's wave-2 marginal gone; body_full_clip_gate 4/4). Root cause per two independently-converged diagnoses = weak bilateral unknown-foot contact phases; fix hardened through THREE adversarial-verify rounds (r1: vacuous surrogates + stance-fallback leak; r2: unfailable-gate-by-exclusion caught w/ failing defect test; r3: fail-closed gate + non-gated max_candidate_phase_slide_m companion — fresh-run candidates 17-23mm, ZERO phase_slide_exceeds_lock_gate rows, predictor confirmed incl. outdoor exact-to-0.1mm). Browser-verified assertion_errors=[] ×3; burlington ball coverage 471→575/600. SCORECARD vs owner queue: (0) BODY-dispatch version-stamp hardening LANDED+LIVE-PROVEN (73 files, 0 drift, stamp echo 4/4, --sync-remote-code both VMs first-try); (1) slide gate CLOSED; (2) grounding_refine → honest no-op posture (0 confident phases on eval clips; un-kill needs upstream foot-attribution = wave-4); (3) P0-4 LAUNCHED + FIRST HUMAN LABELS (CVAT 2.69.0 live; owner 6 clips/~480 frames/~2h = 240 frames/hr; 268 boxes imported via NEW sparse-review semantics + 3 import bugs fixed; registered p11_seed+p12_teacher_eval); (4) P1-1 WARM-START TRAINED on first-fleet-H100 (internal_val 2640: f1@20 0.0615→0.6104 ~10×, median 167.9→2.73px, precision@20 0.848, recall 0.477 = P1-2 target; protected-hash 35/0 both VMs; cycle-caching + output-channels harness bugs fixed in repo); (5) P1-2 teacher RULED BY MEASUREMENT: 2D-gated teachers LOSE to raw WASB on human GT (pooled F1 0.395 vs 0.680 — gates cut ~60% of true balls for +0.03 precision) → raw single-WASB = blessed SST seed (consensus-fusion ban intact), gated sets archived as measured negative, physics-gated teacher deferred to P4 auto-cal (3D chain CLIs hard-require court cal — structural dependency); (6) camera-motion motion-CONDITIONAL default LANDED, PARTIAL-carried: static path bit-exact + 3/4 auto-decisions correct; img1605 in-pipeline probe 0.329 vs 53.7 offline → auto-OFF (ran uncompensated, STILL passed all gates) — probe-context diagnosis = wave-4 #1; (7) img1605 mesh fallback LANDED+LIVE-PROVEN (100 uniform-stride windows, non-promotional, others byte-identical); (8) RAFT-small prefetched sha-recorded (opt-in). EXTRAS: owner quota unlock actioned same-hour (first-fleet H100 validated ~8 steps/s ≈2.3× A100-40 → recorded default heavy worker for TRAINING; BODY stays A100 until separately validated); snapshot→fan pattern proven (env byte-identical; 4-clip makespan 20m54s vs 29.6min serial); 34 pruned sidecars + raw clips regenerated from sha256-verified sources w/ RECONSTRUCTION_NOTEs; court-kp task set ready (6 legal sources; held-out excluded honestly — 8-source ask self-corrected); sparse reviewed-only benchmark = canonical for harvest sets. GOTCHAS: codex exec resume rejects exec-level flags + inherits CALLING cwd (§10 correction due); GCP describe-quota lags admission control (create = definitive test); known_hosts + DEFAULT_REMOTE_HOST point at recycled/dead IPs (refresh next fleet cycle; always --remote-host); manifest stage needs --vite-allow-root off-root; compile_warmup ~2× post cache-clear. COST ~$9-13 GPU total, all VMs down/confirmed. Suites: wide green minus benchmark (banked 41/41 standalone) + truthful-capabilities 15/15 + storage audit pass. PUSH PENDING owner one-liner (classifier wants machine-readable grant: add "Bash(git push *)" to settings.json permissions.allow). WAVE-4 QUEUE (priority): (1) cammotion probe-context diagnosis (0.329-vs-53.7 repro banked); (2) upstream foot-attribution (per-foot evidence at source → un-kill refine); (3) P1-2 fine-tune on owner labels + threshold sweep (recall 0.477↑; warm-start ckpts runs/lanes/w3_p11_train_20260707/checkpoints/); (4) P4 court auto-cal (unlocks physics-gated teacher; court-kp labels next owner session); (5) burlington virtual_world missing-mesh-vertices notice; (6) known_hosts/DEFAULT_REMOTE_HOST + fleet-restart IP protocol; (7) H100 BODY-stage compat validation. VERIFIED=0 unchanged (all gates internal-val; promotion needs pre-registered held-out rows + owner go).
- [BRAND V2 DISPATCHED 2026-07-07, wave-2 manager session] Owner delivered brand taste board (saved to memory: pickleball-dinkvision-brand): ink-on-cream mark w/ pickleball iris + striped grip = APP ICON; splash SIMPLIFIED per owner (zoomed closed eye -> lids open revealing live app, NO other layers — iris-expand removed); speed-streak trail motif retained + replay-open transition; perforation panels + hand-drawn accents (slashes/dot-grids/curvy arrow) as empty-state/onboarding-only system. Manager personally designed the master icon through 4 render-review iterations (final: clip-path lid-cut construction, 7-hole hex pickleball iris, contained grip stripes — legible at 60px; masters at runs/lanes/ios_brand_v2_20260707/mockups/, owner copies incl app_icon_1024.png at ~/Desktop/DinkVision_Mockups/). Codex lane ios_brand_v2 dispatched w/ pixel-law spec (icon asset from master PNG untouched, PaddleEyeMark v4 geometry, lid-reveal splash state machine + tests, DinkVisionAccents.swift at 4 sites only, README brand section). Manager build+sim+device-redeploy verify leg follows landing. ios/ fence: this session only (wave-3 closed; wave-4 not started). VERIFIED=0 unchanged.
- [BRAND V2 IOS 2026-07-07, Codex ios_brand_v2_20260707] SCOPED PASS in sandbox: AppIcon is now the manager master single-1024 asset; PaddleEyeMark uses v4 clipped-iris geometry; splash state machine is zoomedClosed -> lidsOpening -> done with reduced-motion crossfade; accents live only at Replays empty, Stats sample watermark, Profile completed rows, and permission primer; replay-open gets a <=450ms BallTrail sweep; README brand v2 updated. Verification here: `xcrun swiftc -parse` on touched Swift files clean, old iris-expand phase grep clean, icon SHA matches mockup master. Manager still owns xcodebuild/sim/device visual pass. VERIFIED=0 unchanged.
- [P6-3 LANE DISPATCHED 2026-07-07, Fable final session] Third parallel stream opened per owner ask: Codex lane `p63_reference_ranges_20260707` (web-search enabled, xhigh) building the P6-3 pickleball reference-range library v0 — NEW FILES ONLY (docs/racketsport/reference_ranges_{schema,v0}.json + validate_reference_ranges.py CLI + test + scaffold-index line), zero overlap with the wave-4 manager queue files or ios/; provenance-honesty rule hard-coded (unsourced ranges ship as placeholder_unverified, fabricated citations = failed lane). Acceptance: ≥12 schema-valid metric×band entries across ≥4 families/≥3 bands, ≥8 at trade_benchmark+ tier, validator CLI + direct-CLI test, wide suite classified-green. Spec + report: runs/lanes/p63_reference_ranges_20260707/. Booked in inflight_lanes.md; whichever manager session is live when it returns should rule the report if this session is gone. VERIFIED=0 unchanged.
- [BRAND V2 RULED + ON DEVICE 2026-07-07, wave-2 manager session] Brand-v2 lane ACCEPTED after full manager verify chain: lane PASS 6/6 (icon SHA-matched to manager master, iris-expand grep-proven removed, accents at 4 sanctioned sites); manager local leg caught + fixed 6 parse-invisible issues (Sendable self-capture, MainActor isolation on pure-geometry types x2, SwiftUI type-check timeout split, lid-curve geometry NOT coincident at closed [38%-open 'closed' eye = black flash root cause], closed-hold 140->280ms); 22/22 app tests green (sole red = pre-existing sim-incompatible ANELatency, classified); splash sequence LIVE-VERIFIED frame-by-frame in sim (cream zoomed closed eye w/ lashes -> lids part revealing live record home -> home; burst captures) + new v4 mark renders on record screen. Committed 27502d2c0 + pushed; DEPLOYED TO OWNER IPHONE (build+install+launch OK). POLISH NITS booked (non-blocking): tab bar peeks during splash; closed-frame cream slightly dimmed mid-fade. Owner P0-10 smoke still pending: 30s recording -> manager pulls sidecar -> verify arkit_frame_samples/gravity/PTS/policy. VERIFIED=0 unchanged.
- [BRAND V3 IOS 2026-07-07, Codex ios_brand_v3_20260707] SCOPED PASS in sandbox: AppIcon now byte-identical to owner `app_icon_1024.png`, owner `mark_master.png` and `lockup_master.png` are template-off asset-catalog images, large in-app logo placements use the raster mark while tab-bar glyphs remain vector, and splash v3 runs settle -> blink -> openUp with AppTest source covering the owner eye-geometry constants. Verification here: touched Swift files `xcrun swiftc -parse` clean, package `env SWIFTPM_MODULECACHE_OVERRIDE=/private/tmp/pickleball-swiftpm-modulecache swift test --package-path ios --cache-path /private/tmp/pickleball-swiftpm-cache --scratch-path /private/tmp/pickleball-swiftpm-scratch --disable-sandbox` passes for package modules, old zoomed-eye splash shapes/phases grep-clean, asset SHAs match owner masters. App-hosted xcodebuild/sim/device visual pass remains manager-owned. VERIFIED=0 unchanged.
- [IOS CAMFIX 2026-07-07, Codex ios_camfix_20260707] SCOPED PASS in sandbox: Record now runs ARKit only as a bounded pre-record setup pass, pauses/releases ARSession before AVCapture preview/record ownership, writes `setup_pass` plus preserved top-level `arkit_camera_pose`/`court_plane` when available, and records per-frame `arkit_frame_samples` as `provenance=coremotion_only` during AVCapture with no fabricated AR pose. Sequencing/state tests cover setupPass->stopped->capturing ownership, unavailable-reason sidecar path, CoreMotion-only provenance, and stale/gravity-delta refresh; Record chip flow is `Aligning…` -> `Aligned ✓` or `Align skipped`, then recording remains unblocked. Verification here: `xcrun swiftc -parse` on all `ios/**/*.swift` clean; `swift test --disable-sandbox --package-path ios` = 181 passed / 0 failed / 1 skipped. App-hosted xcodebuild/sim/device preview-freeze proof remains manager-owned because CoreSimulator/SwiftPM cache writes are blocked in this sandbox. VERIFIED=0 unchanged.
- [IOS ARC 2026-07-07, wave-2 manager session: brand v2/v3 + camfix RULED, v4 in flight] BRAND V2 accepted then superseded where owner corrected: owner ruled the vector mark 'weird' -> BRAND V3 uses owner's ACTUAL artwork byte-pinned (icon/mark/lockup SHAs verified; splash reworked to owner spec: icon-aligned settle -> SLOW blink (260/120/340ms, lids aligned to measured eye at (0.50,0.361) of mark) -> openUp zoom; deployed to device). Standing rule in brand memory: NEVER re-vector the owner's mark. CAMFIX accepted: root cause = ARKit ARSession vs AVCaptureSession camera exclusivity (preview froze on first frames); fix = bounded ARKitSetupPassRunner (Aligning... chip, skippable, fail-closed unavailable-reasons) + CameraResourceOwnership TOKENS making simultaneous ownership unrepresentable + per-frame CoreMotion gravity w/ provenance=coremotion_only; 181/0 package + 24/0 app tests; ON DEVICE (owner to verify live camera + run the 30s P0-10 smoke). Manager verify-fix tally this arc: 7 parse-invisible catches (Sendable captures x2, MainActor isolation x3, type-check split, lid-geometry+timing) — swiftc-parse-only lanes REQUIRE the manager build leg, now standing. Device hygiene: stray UITests-Runner app uninstalled (wildcard-glob deploy bug, mine); deploys now name Pickleball.app exactly. BRAND V4 DISPATCHED (owner direction): center-docked CODE-DRAWN yellow-emboss record button + hand-drawn motion system (draw-on accents, sketch underline, sticker transitions, wobble) + Coach coming-soon tab + Stats/Replays flush-out + TRAIL SWOOSH replay-open (diagonal bottom-left->up-right, ball becomes play button) + 3D viewer overhaul (branded chrome, one-tap camera presets Broadcast/Behind/Top/Ball-follow, auto-play, tap-player-follow, once-only hand-drawn coach-mark). VERIFIED=0 unchanged.
- [PRODUCT-INFRA SPEC APPROVED 2026-07-07, manager] Owner directive supersedes the TECH_BLUEPRINTS P7-1 ruling (SQLite+Render-disk+Key-Value): product backbone = MongoDB Atlas (system of record + job queue) + AWS S3 (presigned direct upload/download) + pull-based GPU worker daemon on the existing fleet VMs (kills SSH/IP-sync push fragility) + own-account auth (argon2id + 15min JWT + rotating refresh, invite-gated) + Stripe scaffold (live later, P7-3). Budget granted $30/mo (est $18-20); purchase-approval STOP on paid tiers RESOLVED. Spec: docs/superpowers/specs/2026-07-07-product-infra-design.md (branch worktree-product-infra-spec, draft PR). Rollout INFRA-1..5 lanes (Mongo/auth/S3 -> worker cutover -> React screens -> iOS wiring -> delete-cascade/P0-9 port); pipeline/fleet/eval rules untouched. NO lane builds against the old P7-1 plan. VERIFIED=0 unchanged.
- [INFRA-1 LANDED 2026-07-07, manager] Accounts layer merged dark (PICKLEBALL_ACCOUNTS_ENABLED=0 in prod): Mongo (users/refresh_tokens/jobs/clips/entitlements + idempotent indexes), argon2id + 15min JWT + rotating refresh (reuse revokes chain), invite-gated register, S3 presigned multipart clip flow, jobs_v2 (inline-exec until INFRA-2 queue), stripe webhook stub, slowapi limits, .env gitignore rules. render_service 54/54 (manager re-verified); racketsport wide baseline 15F/3029P all attributed pre-existing (missing local runs/ artifacts + doc-lane-owned inventory tests; lane fence = server-only). INFRA-0 cloud backbone COMPLETE + verified (Atlas Flex live+ping, S3 bucket pickleball-prod-939490550128 + 2 scoped IAM users + round-trip PASS, Render Starter + 23 env keys, allowlist = Render egress CIDRs + owner). Next: INFRA-2 pull-worker daemon + queue cutover. VERIFIED=0 unchanged.
