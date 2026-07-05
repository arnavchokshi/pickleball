# Build Checklist

Last updated: 2026-07-05.

This is the operational board. It should stay short enough that a new agent can
read it before touching code. For final goal and truth boundaries, read
`MASTER_PLAN.md`; for commands, read `RUNBOOK.md`; for tier placement, read
`CAPABILITIES.md`.

No row is `VERIFIED`.

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

## Recent Handoffs

- [CAL-SYNTH LANDED 2026-07-05, court-autofind lane] Synthetic court corpus generator v2 shipped:
  NEW threed/racketsport/court_synth_scenes.py (shared procedural render engine, PIL/numpy only, no
  torch, no eval-clip reads) + NEW threed/racketsport/court_synth_stream.py (zero-disk streaming
  trainer contract iter_synthetic_court_samples(config, seed) -> image_bgr/keypoints_xy(15x2)/
  keypoints_vis{0=off_frame,1=occluded,2=visible}/line_family_mask{0=other,1=pickleball,2=tennis,
  3=net}/surface_mask{0=bg,1=apron,2=interior}/meta{homography,distortion,scenario,image_size};
  deterministic per (config, seed)). 7 mixture-weighted scenario families incl. tennis-overlay
  (both line families, distinct colors/widths/wear), adjacent 2-4 courts, portrait-phone 9:16 +
  radial/tangential distortion + off-frame keypoints, harsh directional shadows (one shared light
  direction, person-cast), portable-net/clutter. Self-consistency: meta replays the exact
  pinhole+Brown-Conrady projection -> 0.000px max error over the full 2000-sample probe (bar was
  <0.5px), net keypoints at regulation post height included. Throughput ~65-87 samples/s at
  640x360 on the dev Mac idle (33-44/s under full-suite CPU contention; bar >=25/s).
  generate_synthetic_court_keypoints.py CLI kept backward-compatible (same court_keypoints.json
  envelope, synthetic status, manifest schema_version 2 adds scenario_counts + per-sample
  scenario; new optional --scenarios/--scenario-weights). Probe corpus + per-family
  contact_sheet.jpg (keypoints color-coded by visibility) under
  runs/lanes/cal_synth_20260705/samples/. Tests: test_generate_synthetic_court_keypoints.py
  extended (11) + NEW test_court_synth_stream.py (13), all green. CAL-MODEL: consume the stream
  contract as documented in court_synth_stream.py's docstring; do not renumber class ints.

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
