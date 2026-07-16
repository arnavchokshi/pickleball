# Lane tbcam_20260716 — P0-H remainder: native-intrinsics scaling, crop/orientation, rolling shutter into the wired contracts

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: NORTH_STAR_ROADMAP.md (§2.1 P0-H row as updated 2026-07-15, §4 NS-01.1/NS-01.4
  rows, §6), runs/manager/inflight_lanes.md (live fences), git show bd99c6d11 (the wired
  timebase seam you extend).
- CONCURRENT-LANE FENCES (hard): spine16_20260716 owns scripts/racketsport/process_video.py,
  process_video_body_frames.py, pipeline_cli/pipeline_contracts, validate_pipeline_artifacts,
  AGENTS.md/RUNBOOK.md, tests/racketsport/test_process_video.py and
  test_truthful_capabilities.py — you may NOT touch ANY of those. Track A owns
  threed/racketsport/ball_arc_*.py + tests. A third lane owns event_fusion/racket6dof/
  racket_stage_runner/ball_temporal_filter/player_id_repair/pose_temporal. Preserve all
  unrelated dirty work (configs/ssh/a100_known_hosts, ios/*, scripts/racketsport/
  build_event_review_session.py, brand-exploration/, cvat_upload/, data/).
- If you conclude a process_video.py change is required, put the exact diff hunk INLINE in
  your report as a deferred change — do NOT apply it.
- Cross-language sidecar caution: the CaptureSidecar schema is a Swift/Python contract
  (NS-01.1; golden fixtures under tests/racketsport/fixtures/capture_sidecar/ must remain
  valid; unknown keys stay rejected). New fields must be OPTIONAL with explicit
  absent-semantics; Swift-side emission is out of scope and stays PENDING (say so).
- PYTEST EXIT-CODE TRAP (3b639768c): no pipes; literal `$?` captured; report numbers.
- Wide suite mandatory; manager baseline 3684/24/1 EXIT 1 (single failure = concurrent
  session's untracked build_event_review_session.py — attribute, don't fix). Expect
  concurrent-lane noise; attribute per-file, never edit other lanes' files.
- Artifacts under runs/lanes/tbcam_20260716/. Raw values immutable; fail loudly on schema
  errors.

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- threed/racketsport/schemas/__init__.py (CaptureSidecar additive optional fields only)
- threed/racketsport/coordinates.py (additive typed intrinsics-transform helpers)
- threed/racketsport/court_calibration.py (sidecar-consumer seam declarations)
- threed/racketsport/io_decode.py (rolling-shutter population + evidence reasons)
- threed/racketsport/timebase.py (additive only; frozen contract semantics unchanged)
- threed/racketsport/sam3d_body_input_prep.py + threed/racketsport/court_auto_evidence.py
  (route their ad-hoc intrinsics scaling through the typed helper, parity-first)
- docs/racketsport/timebase_schema.json + the capture-sidecar JSON schema if one exists
  under docs/racketsport/ (additive)
- Tests: test_timebase_contract.py, test_io_decode.py, test_coordinates_api.py,
  test_court_calibration*.py, test_capture_sidecar*.py / schema tests,
  test_sam3d_body_input_prep*.py, test_court_auto_evidence*.py, new test_tbcam_*.py.

## OBJECTIVE (quote in report)
P0-H row (North Star, post-2026-07-15): "native-intrinsics scaling, crop/orientation, and
rolling-shutter device slices remain unwired (NS-01.6/01.7)". NS-01.1 requires "native
intrinsics with reference crop/orientation ... and rolling shutter". Close the
REPRESENTATION + TYPED-TRANSFORM + CONTRACT-POPULATION slices; physical-device evidence
stays owner-gated PENDING.

Manager-verified ground truth:
1. No typed intrinsics scaler exists. Ad-hoc scalers: sam3d_body_input_prep.py:78-97
   static_camera_intrinsics_k; court_auto_evidence.py:228-252. calibration_image_size
   inference (court_calibration.py:312-340) even guesses size from cx*2/cy*2.
2. CameraIntrinsics (schemas:78-86) has no reference resolution; CaptureSidecar has NO
   reference_crop field at all; orientation/video_rotation_angle_degrees (schemas:204-206)
   have ZERO intrinsics-transforming consumers (only a warn check in ball_capture_protocol).
3. Rolling shutter: RollingShutterModel exists (timebase.py:525-555) w/ schema slot
   (timebase_schema.json:39-49); the sidecar has no readout field; io_decode.py:495
   hardcodes rolling_shutter_model=None; row_time has zero callers.

## DELIVERABLES (numbered; honest PARTIAL allowed)
1. TYPED INTRINSICS TRANSFORMS in coordinates.py: scale_intrinsics (native->processed
   raster, paired with scale_raster_points' declared spaces), rotate_intrinsics (0/90/180/
   270 with width/height semantics), crop_intrinsics (principal-point offset). Fail-closed
   on undeclared/conflicting spaces. Unit tests incl. round-trips and synthetic nonzero-
   distortion cameras (distortion coefficients pass through unchanged and that is DECLARED,
   not silent — document the pinhole-only scope honestly).
2. ROUTE THE AD-HOC SCALERS through (1), parity-first: static_camera_intrinsics_k and
   court_auto_evidence produce byte-identical outputs on their existing tests/fixtures
   (pin digests before/after in your lane log).
3. SIDECAR REPRESENTATION (additive, optional, cross-language-safe):
   reference_crop (rect in native pixels) and rolling_shutter (readout seconds + direction)
   fields on CaptureSidecar with strict validators; all three golden fixtures stay valid;
   explicit typed absent-semantics. Swift emission = PENDING (report it).
4. ORIENTATION HONESTY AT THE CALIBRATION SEAM: where court_calibration builds calibration
   from sidecar intrinsics+resolution, detect a rotation-implying orientation
   (video_rotation_angle_degrees 90/270 vs landscape intrinsics) and FAIL LOUDLY with a
   typed error instead of silently proceeding (current behavior would be silently wrong);
   apply rotate_intrinsics only where a test proves correctness synthetically. No silent
   behavior change for rotation-0/None captures (parity).
5. ROLLING-SHUTTER POPULATION: io_decode build_timebase_artifacts populates
   RollingShutterModel from the sidecar field when present; when absent, records an explicit
   missing reason in timebase_decode_evidence.json (never silent None). Synthetic test both
   ways; row_time consumers remain out of scope (declaration slice only — say so).

## MANDATORY VERIFICATION (literal exit codes, no pipes)
- All owned-file focused tests EXIT 0; the three golden sidecar fixtures validate; Wolverine
  timebase artifacts regenerate byte-identical for the no-new-fields case (digest pin).
- Full wide suite with per-failure attribution vs the manager baseline.

## MANDATORY STRUCTURED REPORT at runs/lanes/tbcam_20260716/report.json (write it yourself,
schema docs/racketsport/lane_report.schema.json): objective_result per deliverable;
full_suite counts + literal exit codes + attribution; HONEST ISSUES (Swift PENDING, physical
PENDING, pinhole-only scope); deferred process_video.py hunks inline if any; BEST-STACK
DELTA (expected "(c) none"); dated inflight note paragraph.
