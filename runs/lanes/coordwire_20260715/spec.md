# Lane coordwire_20260715 — NS-01.4/P0-D: adopt the typed coordinate API in the remaining real stage consumers

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. Work only inside that repo. VERIFIED=0 is binding:
no promotion language anywhere — "wired" and "scoped pass" at most.

## HARD RULES
- NO branches, NO commits, NO pushes. Leave changes in the working tree on `main`.
  The manager rules and commits.
- Read first: `NORTH_STAR_ROADMAP.md` (§2.1 row P0-D, §4 NS-01.4, §6 standing rules),
  `runs/HANDOFF_20260714.md` §5, the two prior parity commits `f15052ae1` and `0e97c09fe`
  (`git show`) — your work continues exactly their discipline, and
  `runs/manager/inflight_lanes.md` (last note).
- Protected data: Outdoor/Indoor labels NEVER. Wolverine internal-val fixtures OK (the
  SHA-pinned parity fixture under `runs/lanes/w7_critique_20260709/wolv_world/...` is your
  real-iPhone-clip proof bed; READ-ONLY).
- Preserve unrelated dirty work: do not touch `configs/ssh/a100_known_hosts`,
  `brand-exploration/`, `cvat_upload/`, `data/`, `runs/manager/gpu_fleet.md`, other lanes'
  run dirs.
- Raw observations immutable; schema/programming errors fail loudly; every stage declares
  coordinate space (standing rules 6/8/10).
- PYTEST EXIT-CODE TRAP (commit 3b639768c — twice bitten): run pytest redirected to a file,
  capture `$?` of the pytest command directly, no pipes. Report literal numeric exit codes.
- Wide blast-radius suite mandatory: `MPLBACKEND=Agg .venv/bin/python -m pytest
  tests/racketsport -q` (full). Report exact counts; every failure proven pre-existing.
- Lane artifacts under `runs/lanes/coordwire_20260715/`. No new root .md. No new CLI without
  a same-lane direct-CLI reference test.

## EXPLICIT FILE OWNERSHIP (you may edit ONLY these)
- `threed/racketsport/coordinates.py` (additive helpers only; existing API frozen)
- `threed/racketsport/court_calibration.py` (typed adapters beside legacy only)
- `threed/racketsport/placement.py`
- `threed/racketsport/ball_court_filter.py`
- `threed/racketsport/ball_physics3d.py`
- `threed/racketsport/ball_inout_uncertainty.py`
- `threed/racketsport/virtual_world.py`
- Tests: `tests/racketsport/test_coordinates_api.py`, `test_coords_parity_real_fixture.py`
  (additive), `test_placement*.py`, `test_ball_court_filter*.py`, `test_ball_arc_solver.py`,
  `test_ball_inout_uncertainty.py`, `test_virtual_world*.py`, new `test_coords_*` files.
- FORBIDDEN (hard fence — concurrent lanes / frozen surfaces / other owners):
  `scripts/racketsport/process_video.py` (another lane owns it THIS WINDOW — if you believe a
  runner hunk is required, put the exact diff hunk INLINE in your report as a deferred change;
  do NOT apply it), `threed/racketsport/io_decode.py`, `threed/racketsport/timebase.py`,
  `threed/racketsport/audio_onsets.py`, `threed/racketsport/camera_motion.py` (out of scope),
  `threed/racketsport/court_precision_metrics.py` (FROZEN harness cpm_v2_frozen_20260712),
  `threed/racketsport/overlapping_court_calibration.py` and `court_proposal_optimizer.py`
  (CAL-lane territory), `RUNBOOK.md`, `tests/racketsport/test_truthful_capabilities.py`.

## OBJECTIVE
Commits f15052ae1 + 0e97c09fe put the typed coordinate vocabulary (canonical-beside-legacy,
parity-proven) at the person/paddle/body-decode/calibration seams. Five real stage consumers
still do raw pixel<->court/world math outside the canonical API. Adopt the typed API there,
parity-first, so P0-D's "stages do not consume it yet" defect is materially closed for the
placement, ball-in/out, ball-arc, ball-uncertainty, and world stages.

North Star acceptance gates you serve (quote in your report):
- P0-D exit gate: "Stage-adopted transform API passes distorted synthetic and real iPhone tests."
- NS-01.4 gate: "Distorted synthetic and real iPhone tests; corrected event/geometry error
  beats raw path on independent labels" — the independent-labels clause is PENDING (no
  independent label set exists yet; NS-02 owner-gated). NEVER fabricate labels; state PENDING.
- NS-01.4 stop rule: "Do not mix coordinate spaces, align latest samples, or correct audio
  destructively; preserve raw values."

Exact raw-math seams to adopt (manager-verified locations):
1. `placement.py`: `undistort_pixel` (~L1699, raw `cv2.undistortPoints(...,P=k)`),
   `_unproject_pixel` (~L2576, raw H^-1 @ [u,v,1]), `_project_pixel_to_court` (~L1871-72),
   `_homography_pixel_convention` (~L749) + `undistort_applied` (~L253) — route through the
   canonical typed API / `HomographyPixelConvention`; the pixel space
   (raw vs undistorted native) must be an explicit typed declaration at each call.
2. `ball_court_filter.py`: `build_target_court_polygon` (~L40), `filter_ball_track_to_target_court`
   (~L72), `_target_image_xy_to_world_xy` (~L210) — typed project/unproject with declared spaces.
3. `ball_physics3d.py`: `_project_world_array` (~L387 raw pinhole), `project_image_points_to_world`
   (~L317), `reconstruct_bounce_arcs_from_image_track` (~L101), residual `_project_world_point`
   (~L805) — typed camera-model seam (world_court_netcenter_z_up_m -> pixels, declared).
4. `ball_inout_uncertainty.py`: solvePnP/projectPoints seams (~L150/153/214) — declare spaces
   through the typed vocabulary (the cv2 calls stay; the seam contract becomes typed).
5. `virtual_world.py`: replace the raw `project_image_points_to_world` import (L12) with the
   typed adapter, spaces declared, used in `build_virtual_world_state` (~L81).

## REQUIRED DISCIPLINE (same as the two landed parity commits)
- Canonical-beside-legacy. NO numerical behavior change claimed or made. Missing declarations
  decode to the historical default; conflicting explicit declarations fail closed.
- Parity proof per seam on frozen fixtures:
  (a) REAL IPHONE CLIP: extend `test_coords_parity_real_fixture.py` — byte-stable digests on
      the SHA-pinned Wolverine artifacts for placement/ball/world outputs you touched (build
      the digest pins from HEAD before your edits; show both runs in your log).
  (b) DISTORTED SYNTHETIC: for every touched seam, a synthetic camera with NONZERO distortion
      coefficients round-trips/projects identically through legacy vs typed paths, and the
      declared-space plumbing rejects a wrong-space input loudly (fail-closed test).
- The typed path must be the one the stage code actually CALLS after this lane (adoption,
  not a parallel shim nobody invokes). Legacy free functions may remain for external callers
  but the five modules above route through the canonical API.
- If exact byte parity is impossible at some seam (e.g. float association differences),
  STOP that seam and report honestly rather than shipping a silent numerical change.

## MANDATORY TESTS
- All existing coordinate tests green (35 focused from f15052ae1 + parity file from 0e97c09fe).
- New per-seam parity + distorted-synthetic + fail-closed tests as above.
- Full wide suite: exact counts + literal exit code.

## MANDATORY STRUCTURED REPORT (report.json via output schema)
- `objective_result`: per-seam PASS/FAIL; which P0-D/NS-01.4 clauses are now "wired (scoped
  pass)" vs PENDING (independent-labels clause = PENDING, say it explicitly).
- `full_suite`: exact counts + literal pytest exit codes; failures proven pre-existing.
- HONEST ISSUES; artifacts under runs/lanes/coordwire_20260715/; any DEFERRED
  process_video.py hunks as inline diffs.
- BEST-STACK DELTA: expected "(c) no stack delta" — state and justify.
- A dated one-paragraph note for runs/manager/inflight_lanes.md.
