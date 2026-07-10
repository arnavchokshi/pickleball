# LANE ns014_p22residual_20260709 — P2-2 residual attribution + NS-01.4 typed coordinates + synthetic sam3d adapter + gate decomposition (Mac-side)

You are a Codex implementation lane. Fable (the manager) designed this mission; execute it
exactly, self-verify, and return a structured report. Big self-iterating lane: fix your own
failures; do not stop early.

## HARD RULES
- NO git branches, NO commits, NO git index operations. Leave all changes in the working tree.
- Read `NORTH_STAR_ROADMAP.md` §2.2 BODY row + §4 NS-01.4 and `AGENTS.md` before coding.
- 4 protected eval clips are EVAL-ONLY (Burlington/Wolverine internal scoring allowed; Outdoor/
  Indoor labels NEVER). This lane touches no labels at all.
- HONEST reporting. A blocked or failed acceptance item is a first-class result — report it, never
  fake it. Visual plausibility and smaller residuals are not accuracy proof.
- FENCED FILES — READ-ONLY, do not edit under any circumstance: `ios/**`, `.gitignore`,
  `scripts/racketsport/calibrate_charuco_device.py`, EVERY root-level `*.md` (all mid-commit by
  another session — includes RUNBOOK.md/NORTH_STAR_ROADMAP.md/AGENTS.md/README.md),
  `scripts/racketsport/process_video.py`, `scripts/racketsport/remote_body_dispatch.py`,
  `threed/racketsport/worldhmr.py`, `threed/racketsport/body_postchain.py`,
  `threed/racketsport/orchestrator.py`, `threed/racketsport/pose_temporal.py`,
  `threed/racketsport/foot_pin.py`, `threed/racketsport/contact_splice.py`.
  If the mission proves impossible without editing one of these, STOP that deliverable and report
  the exact blocking symbol/line in HONEST ISSUES; deferred fenced-file changes = inline diff
  hunks in the report, never applied, never a .patch file.
- `threed/racketsport/mhr_decode.py` currently carries ONE uncommitted line of another session's
  docstring dirt (line ~3: `TECH_BLUEPRINTS.md` -> `runs/archive/root_docs_20260709/TECH_BLUEPRINTS.md`).
  PRESERVE it exactly; never revert it.
- Preserve all other unrelated dirty working-tree changes. Touch ONLY your owned files.
- All artifacts under `runs/lanes/ns014_p22residual_20260709/`. Other lanes' run dirs are
  READ-ONLY evidence.
- Every new CLI ships its direct-CLI-reference test same-lane (scaffold index convention).
- No new root .md files. No new .md files at all (register nothing; put docs in module docstrings
  + the lane report).

## FILE OWNERSHIP (this lane owns exactly these; everything else read-only)
- `threed/racketsport/coordinates.py` (NEW)
- `threed/racketsport/mhr_decode.py`
- `threed/racketsport/hmr_deep.py` (adoption-only edits, see D1; keep minimal)
- `scripts/racketsport/gate_check_body_decode.py`
- `scripts/racketsport/synthetic_body_decode_gate.py`
- `scripts/racketsport/attribute_body_decode_residual.py` (NEW)
- `tests/racketsport/test_coordinates_api.py` (NEW)
- `tests/racketsport/test_mhr_decode.py`
- `tests/racketsport/test_gate_check_body_decode.py`
- `tests/racketsport/test_synthetic_body_decode_gate.py`
- `tests/racketsport/test_attribute_body_decode_residual.py` (NEW)
- `runs/lanes/ns014_p22residual_20260709/**`

## EVIDENCE TO READ FIRST (ground truth for this mission)
- `runs/lanes/w7_p22gate_20260709/arm_c_field_quant_report.json` — persisted joints_world vs
  grounded-raw pred_keypoints_3d: mean 15.8mm / p95 23.4mm / max 26.7mm (production run).
- `runs/lanes/w7_speedgate_20260709/results.json` key `arm2_gate1b_fixed_harness` — corrected
  canonical gate: per-player joints_world p95 22-58mm; 262.17mm = single worst-frame outlier
  (player 20, p95 24.2mm); mesh_skeleton_divergence ~50-53mm p95 uniformly.
- `runs/lanes/w7_p22gate_20260709/gate1b_raw_arm_report.json` — field mapping + decoder provenance.
- Banked CPU fixture (READ-ONLY): `runs/lanes/w7_p22gate_20260709/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/`
  — a RAW-postchain-preset run: `body_raw_grounded_joints.json` (680 player-frames, all 4 players),
  `skeleton3d.json` (byte-near-identical to raw grounded — postchain was fully bypassed),
  `court_calibration.json`, `tracks.json`, `placement.json`, etc. `body_mesh.json` is ABSENT
  (VM-only); do not expect it.
- Banked raw records (READ-ONLY): `runs/lanes/w7_p22gate_20260709/fast_sam_subprocess_sample/`
  — chunk `index.json` describing 680 records across 43 buckets, but ONLY buckets 000000/000001
  present = 32 records (players 18/20/21, frame_idx 56-79). Your CPU acceptance runs use these 32.
  The index's `chunk_dir`/paths point at a dead VM path — your loader invocation must tolerate
  loading just the present buckets for the requested request_ids (restrict requested ids to the
  32 present ones in tests).

## ESTABLISHED FACTS (manager-verified; build on these, do not re-derive)
The production BODY chain from raw Fast-SAM-3D-Body records to persisted artifacts is:
1. `hmr_deep.py:908-917` `normalize_fast_sam_body_output` — pred_cam_t added exactly-once via
   `mhr_decode.apply_pred_cam_t_once` -> joints_camera/vertices_camera.
2. `worldhmr.py:2697,2745-2774` `_camera_offsets_to_world` — root-relative rotation into court
   orientation (row-vector `offset @ R`, = R^T x column convention).
3. `worldhmr.py:2698-2703` camera-motion xy correction (OFF for wolverine; statics-off).
4. `worldhmr.py:2709-2736` placement anchoring: low-joint-cluster xy -> track_world_xy,
   dz = -min_z floors lowest point to court Z0. World frame born here. Output = the
   `body_raw_grounded_joints.json` payload when the raw preset is active.
5. `worldhmr.py:3011-3107` `_smooth_grounded_frames` EMA alpha=0.65 + step-limit + identity reset
   (flag temporal_smoothing).
6. `worldhmr.py:1666-1877` `_apply_footlock_to_player_frames` — z-snap contacts, rigid xy delta
   cap 0.02m, second step-limit (flag foot_lock).
7-8. skeleton3d-only: `apply_sam3d_temporal_refine_gate` + wrist bone lock CALL SITE #1
   (`pose_temporal.py:562-711`).
9. `worldhmr.py:1585-1639` `_apply_root_phase_median_lock_to_payload` — BOTH payloads (mesh
   vertices translated too; only with stance_index + foot_pin on).
10. `foot_pin.py:119-202` — skeleton3d ONLY.
11. `worldhmr.py:1953-2027,2113-2193` `_apply_world_joint_visual_smoothing` — 3-tap FIR
   (0.30/0.40/0.30) on BOTH payloads' joints_world; mesh_vertices_world detached and NEVER
   smoothed (spatial offset between mesh and joints is OUR OWN doing here).
12. `worldhmr.py:2029-2071` `_restore_wrist_peak_timing_windows` — wrist/elbow selective reverts.
13. `contact_splice.py` — skeleton3d <- body_mesh named-joint override at hitter contacts.
14. wrist bone lock CALL SITE #2 (`orchestrator.py:1360-1361`) — skeleton3d, post-splice.
- `scale`/scale_params is PASS-THROUGH metadata (players[].scale); never applied to joints_world
  at grounding time.
- Gate-1b today (`gate_check_body_decode.py:417-516`): re-decodes persisted euler params through
  MHR forward (FK), grounds via `mhr_decode.ground_decoded_camera_frame` (-> the REAL
  `worldhmr._ground_fast_sam_sample`), compares vs persisted body_mesh joints_world. Its residual
  therefore mixes THREE things: (a) grounding determinism, (b) intentional postchain mutations
  (steps 5-12), (c) FK-vs-keypoint-head model divergence. The <=1mm bar can only ever hold for (a).
- Known hole: when no frame has vertices_world, `worst_vertices_mm` defaults 0.0 and the vertices
  half of the pass-AND is VACUOUS.
- `body_raw_grounded_joints.json` is persisted ONLY under the raw postchain preset
  (`body_postchain.py:9`, `worldhmr.py:839`).

## MISSION (4 deliverables)

### D1 — `threed/racketsport/coordinates.py`: NS-01.4 typed transform module (slice)
A dependency-light module (numpy + stdlib only; NO torch/cv2 imports at module top) that makes the
repo's implicit conventions explicit and gives the decode/gate surfaces one shared vocabulary.
Contents:
- `CoordinateSpace` (StrEnum or Literal set) naming at minimum: `pixels_raw_native`,
  `pixels_undistorted_native`, `pixels_preview_scaled`, `camera_m`,
  `body_camera_root_relative_m`, `world_court_netcenter_z_up_m`, `world_xy_homography_m`.
  Module docstring documents each space precisely (origin, axes, units) — source the definitions
  from `docs`/code, e.g. schemas' `court_netcenter_z_up_m` literal, `net_plane.py` normal [0,1,0].
- `HOMOGRAPHY_PIXEL_CONVENTIONS = ("raw_pixels", "undistorted_pixels")` typed constant.
- `invert_extrinsics(R, t) -> (R_camera_to_world, camera_center_world)` — THE canonical
  world_to_camera inversion (`camera = R @ world + t`; `C = -R.T @ t`), vectorized numpy,
  docstring stating the convention and the row-vs-column equivalence note for
  `skeleton_upright.rotate_camera_offsets_row_times_R`.
- `world_to_camera_points(points_world, R, t)` and `camera_to_world_points(points_camera, R, t)`
  — canonical forward/inverse application.
- `apply_translation_once(points, translation, already_applied=False)` — the generic
  exactly-once translation policy. `mhr_decode.apply_pred_cam_t_once` becomes a thin delegating
  wrapper (public name/signature/behavior UNCHANGED — its tests must pass unmodified).
- Re-export/wrap `court_calibration.camera_matrix_from_intrinsics` as the blessed K-builder
  (import inside the function to stay light).
ADOPTION (this lane only): `mhr_decode.py` (delegate apply_pred_cam_t_once),
`gate_check_body_decode.py` + `synthetic_body_decode_gate.py` (use the typed constants/helpers
where they currently inline them), `hmr_deep.py` ONLY if a trivial delegation exists (else leave —
keep hmr_deep edits minimal to nothing). ZERO numeric behavior change anywhere: prove via A3.
Do NOT refactor placement/ball/racket/court files (other lanes' surfaces); the consolidation map
for those goes in your report as booked follow-up, citing the 6 independent `invert_extrinsics`
sites (`court_calibration.py:247`, `court_calibration_metric15.py:661-664`,
`ball_inout_uncertainty.py:216`, `racket6dof.py:333-340`, `paddle_pose_fused.py:857`,
`ball_arc_solver.py:2337-2338`) and 7+ inline K constructions.

### D2 — `scripts/racketsport/attribute_body_decode_residual.py`: the P2-2 attribution instrument
CLI that decomposes the GATE-1b residual into named components from ONE run dir. No edits to
worldhmr — import its functions and, where per-stage snapshots are needed, wrap/monkeypatch the
stage functions FROM THE CLI at call time around the real entry points.
Inputs: `--run-dir` (containing `body_mesh.json` and/or `skeleton3d.json` + sibling
`fast_sam_subprocess/batch_outputs-*.json.chunks/index.json`), `--calibration` (default:
run-dir court_calibration.json), optional `--raw-grounded` (a `body_raw_grounded_joints.json`),
`--out`, `--max-frames-per-player` (default 0 = all available).
Report (`artifact_type: racketsport_body_decode_residual_attribution`, `schema_version: 1`):
- `grounding_determinism`: re-ground raw records through the REAL path
  (`hmr_deep.normalize_fast_sam_body_output` -> `worldhmr._ground_fast_sam_sample`, cam_t
  exactly-once) and compare vs `--raw-grounded` when provided: per-player + overall
  mean/p95/max mm, `passed_1mm` bool. When absent: status `no_reference`.
- `postchain_attribution`: replay the postchain (stages 5,6,9,11,12 for the mesh payload; note
  skeleton-only stages separately if skeleton3d present) via the REAL worldhmr entry point with
  snapshot capture between stages; per-stage per-player delta stats (mean/p95/max mm vs previous
  stage). Read the run's actual knob values from run-dir artifacts where recorded; document every
  assumption you must make in the report's `assumptions` list. If the entry point cannot be driven
  without fenced edits, fall back to calling the documented stage functions in the mapped order
  and SAY SO in `method`.
- `replay_validation`: replayed final joints_world vs persisted (body_mesh and/or skeleton3d):
  per-player mean/p95/max mm + `chain_reproduced_1mm` bool. >1mm is NOT a lane failure — name the
  first diverging stage in `first_divergent_stage` and report honestly.
- `fk_vs_head_divergence` (OPTIONAL stage, fail-soft): decode persisted euler params via
  mhr_decode MHR runtime and compare decoded camera-frame keypoints vs raw `pred_keypoints_3d`
  (camera frame, pre-grounding; report both raw and root-aligned stats). On Mac the MHR runtime is
  typically unavailable -> status `blocked_mhr_runtime_unavailable` with the import error string.
  INFORMATIONAL metric — no 1mm pass bool on this component.
Tests (CPU, must run in this repo's venv):
- On the banked fixture: restrict to the 32 present raw records; `grounding_determinism` vs the
  banked `body_raw_grounded_joints.json` must report p95 <= 1.0mm AND max <= 2.0mm.
- All-stages-disabled replay == identity (<= 1e-6 m).
- Snapshot-capture mechanics on synthetic tmp_path payloads; CLI --help + scaffold reference test.

### D3 — `synthetic_body_decode_gate.py`: wire the real sam3d adapter (R1(e) instrument)
- Add `--checkpoint`, `--mhr-asset`, `--device` args (defaults mirroring
  `gate_check_body_decode.py`'s).
- Replace the stub branch (lines 65-73) with a real adapter honoring `_mock_decode`'s exact return
  contract `{"joints_world": [[x,y,z]...], "vertices_world": [[x,y,z]...]}`:
  1. Author geometry as an actual MHR mesh: use the mhr_decode MHR runtime to pose a neutral/
     simple-pose body at known scale + known synthetic pred_cam_t (keep the authored-truth
     bookkeeping identical to the mock path).
  2. Render it recognizably human: filled-triangle rasterization with simple lambertian shading on
     a plain background, >= 384px on the long side (numpy/matplotlib-Agg only; no new deps, no
     neural rendering). Two-step realism ladder ONLY: attempt-1 plain shaded mesh; attempt-2 add a
     simple floor + background gradient. NOTHING further — kill rule.
  3. Run SAM-3D-Body inference on the render (the Fast-SAM runtime import path used by the
     production subprocess); apply cam_t exactly-once; return the contract dict.
  Honest statuses: `measured` | `blocked_sam3d_runtime_unavailable` (Mac default; include import
  error) | `blocked_synthetic_render_not_detectable` (no valid detection after the 2-attempt
  ladder; include per-attempt evidence). The mock path must stay byte-identical (existing tests
  unmodified and green).
- CPU tests: adapter selection, blocked statuses, CLI help, report shape. No GPU claims — the VM
  arm runs it later; your job is that `--decoder sam3d` is fully wired and fails-soft correctly
  on this Mac.

### D4 — `gate_check_body_decode.py`: decomposition fields + fail-closed vertices hole
- ADDITIVE report fields only; ALL existing thresholds/constants UNCHANGED (recalibration is a
  manager/owner decision, not this lane's):
  - `residual_decomposition`: optional `--attribution-report` input; when provided, embed the
    attribution summary (grounding_determinism / postchain totals / fk_vs_head) beside gate_1b so
    a reader sees WHICH component carries the millimeters.
  - Vertices hole: when no sampled frame contributed vertices, set
    `gate_1b_world_round_trip.vertices_status = "absent_not_measured"` and make the overall
    `passed` EXCLUDE the vacuous vertices half only in the direction of MORE strictness: joints
    must still pass AND vertices absence must be explicitly surfaced (`passed` stays what joints
    say, but the report can never silently claim a measured-vertices pass). Fail-closed, additive.
- Tests for both behaviors.

## ACCEPTANCE (all measured, reported as objective_result)
- A1 focused suite green: `MPLBACKEND=Agg .venv/bin/python -m pytest
  tests/racketsport/test_coordinates_api.py tests/racketsport/test_mhr_decode.py
  tests/racketsport/test_gate_check_body_decode.py tests/racketsport/test_synthetic_body_decode_gate.py
  tests/racketsport/test_attribute_body_decode_residual.py tests/racketsport/test_smooth_body_mhr_latent.py
  tests/racketsport/test_hmr_deep_primitives.py tests/racketsport/test_worldhmr.py
  tests/racketsport/test_worldhmr_stance_grounding.py tests/racketsport/test_dead_code_audit.py
  tests/racketsport/test_scaffold_tool_index.py tests/racketsport/test_truthful_capabilities.py -q`
- A2 attribution CLI on the banked fixture: grounding_determinism p95 <= 1.0mm and max <= 2.0mm on
  the 32 banked records; all-stages-disabled replay identity <= 1e-6 m. Save the actual report to
  `runs/lanes/ns014_p22residual_20260709/attribution_banked_fixture_report.json`.
- A3 zero-numeric-drift proof: run `gate_check_body_decode.py --self-check` (or its existing
  stub-decoder test path) and `synthetic_body_decode_gate.py --decoder mock` BEFORE your first edit
  (save outputs under `runs/lanes/ns014_p22residual_20260709/pre/`) and AFTER (under `post/`);
  diff-prove numerically identical metric values + unchanged threshold constants
  (GATE_1A_MAX_ABS_ERROR_DEG=0.1, GATE_1B_MAX_ABS_ERROR_MM=1.0, MESH_SKELETON_DIVERGENCE_P95_MM=5.0).
- A4 WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` — capture the
  pre-lane baseline FIRST (save both logs); no NEW failures vs baseline. Pre-existing failures:
  list them verbatim, do not fix, do not hide.
- Kill criteria: fenced-file edit required -> STOP that deliverable, inline diff hunk in report.
  MHR/Fast-SAM runtime required for a CPU acceptance item -> the item reports blocked-status
  honestly; never mock a `measured` status.

## BEST-STACK DELTA (mandatory)
(c) No stack delta: measurement/API lane; no model, weights, or promoted-policy change; all gate
thresholds byte-unchanged. State this verbatim in the report and confirm
`configs/racketsport/best_stack.json` untouched.

## STRUCTURED REPORT
Write the schema-validated report (harness supplies --output-schema). Include: objective_result
per deliverable (PASS/FAIL/BLOCKED + numbers), full_suite counts (baseline vs post), HONEST
ISSUES, artifacts list with paths, assumptions, the booked NS-01.4 consolidation follow-up map,
and a one-paragraph dated handoff summary for the manager (NOT written to any root .md).
