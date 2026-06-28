# Automatic Court Tracking Design

## Goal

Build court tracking so the normal user flow requires no court taps. The user should be able to show a pickleball court, and the pipeline should automatically detect the playing surface, kitchen/NVZ lines, center service lines, and visible top of net, then feed those measured court observations into calibration, 3D world grounding, overlays, and evaluation gates.

Manual corner taps remain available only as a fallback, debugging tool, or supervised label source for improving the automatic detector.

## Implementation Status — 2026-06-28

The current runtime now writes `court_line_evidence.json` as part of the calibration stage whenever it can inspect a video or frame. With a trusted sidecar/manual seed, the runner samples the source video before still frames and requires semantic NVZ/kitchen, center service-line, and trusted top-net evidence before video-backed runs proceed to tracking. Without a trusted calibration seed, a random video still produces fail-closed court evidence plus `court_zones.json` and `net_plane.json`, then stops before creating `court_calibration.json`.

This is not yet a verified no-tap calibration solve. The remaining CAL-3 work is to produce a trusted court calibration directly from video-only line/keypoint evidence, then pass the real-clip gates below without manual taps.

## Current Evidence

The current prototype path creates `court_calibration.json`, `court_zones.json`, and `net_plane.json` from human-reviewed four-corner corrections. The resulting overlay is often reasonable for the outer border, but it does not prove interior-line accuracy:

- `court_calibration.json` contains only the point correspondences that built the solve, so four-corner reprojection can be `0px` while kitchen lines and net are still wrong.
- `CourtTemplate.line_segments_m` includes baselines, sidelines, net, and NVZ lines, but pickleball center service-line segments are missing.
- `net_plane.json` is regulation geometry, not a detected top-net observation.
- Existing eval gates check artifact validity and corner reprojection. They do not score kitchen-line residuals, centerline residuals, top-net residuals, or no-tap auto-detection success.
- Local validation assets are under `runs/eval0/prototype_gate_h100_v2/` and include four calibrated clips, raw review/CVAT frames, calibration overlays, and short smoke videos.

The current `court_keypoint_net.py` scaffold already defines many useful court dots: corners, NVZ intersections, net intersections, and center NVZ points. It needs baseline-center dots and explicit line evidence to become a no-tap system.

## Product Requirements

1. The default pipeline must attempt automatic court detection before any manual tap path.
2. The automatic detector must produce semantic evidence, not just a fitted homography:
   - Outer boundary: `near_baseline`, `far_baseline`, `left_sideline`, `right_sideline`.
   - Kitchen/NVZ lines: `near_nvz`, `far_nvz`.
   - Center service lines: `near_centerline`, `far_centerline`.
   - Top net: `net_top_left`, `net_top_center`, `net_top_right`, plus a fitted top-net polyline/curve.
   - Keypoint dots: corners, NVZ/sideline intersections, NVZ/center intersections, baseline/center intersections, net/sideline intersections, and net center.
3. The solver may use regulation math as a prior, but it must also align to measured image evidence for kitchen lines, centerlines, and net top.
4. Manual labels from existing and future review artifacts become training and validation standards. The existing four-corner `court_corners.json` format should be extended or complemented with line/keypoint labels rather than replaced by a separate vocabulary.
5. Downstream 3D modules must receive a court/world frame that includes quality/confidence for interior lines and top net. Metrics that depend on kitchen, centerline, or net-plane accuracy must be confidence-gated.

## Architecture

### 1. Court Template Completeness

Extend `CourtTemplate.line_segments_m` for pickleball to include:

- `near_centerline`: from baseline center `(x=0, y=-22ft)` to near NVZ center `(x=0, y=-7ft)`.
- `far_centerline`: from far NVZ center `(x=0, y=7ft)` to baseline center `(x=0, y=22ft)`.

Extend the court keypoint taxonomy with:

- `near_baseline_center`
- `far_baseline_center`

These complete the semantic map for line/dot alignment.

### 2. Court Evidence Artifact

Add a new schema-valid artifact, `court_line_evidence.json`, produced before or during calibration. It records per-frame and aggregated evidence:

- `line_observations`: semantic line ID, image segment endpoints, confidence, frame indexes, residual to template projection, visible fraction, and source detector.
- `keypoint_observations`: semantic dot name, image point, confidence, frame indexes, and source detector.
- `net_observations`: top-net endpoints/center, fitted top-net geometry, confidence, frame indexes, and residual to regulation net projection.
- `aggregate`: accepted/rejected line IDs, mean/p95 residuals, temporal stability, and `auto_calibration_ready`.

This artifact becomes the bridge between raw video pixels, label standards, calibration, overlays, and eval gates.

### 3. Automatic Evidence Detection

Implement a CPU-safe first version that can run on local frames without model checkpoints:

1. Seed with ARKit sidecar if present; otherwise seed with coarse image/court heuristics from white-line candidates and rectangular court topology.
2. Run line detection in multiple frames using contrast normalization, edge detection, probabilistic Hough/LSD-style candidate extraction, and color/brightness masks for court lines.
3. Score candidates against semantic template projections using orientation, perpendicular distance, endpoint overlap, in-court ordering, visible length, and temporal agreement.
4. Detect top-net evidence separately from ground lines, using a narrow ROI near the projected net, horizontal/near-horizontal bright-tape and dark-edge candidates, post proximity, and center sag prior.
5. Aggregate 20 to 40 static frames by robust median/RANSAC, rejecting player occlusion and transient shadows.

The first implementation should be deterministic and testable. Later CAL-3 training can replace or augment the candidate detector with a learned keypoint/line model.

### 4. Point And Line Calibration Refinement

Keep the existing `CourtCalibration` artifact, but build it from a richer set of evidence:

- Ground-plane points from semantic dots where confidence is high.
- Ground-plane line residuals for baselines, sidelines, NVZ lines, and centerlines.
- A net-top residual that validates or adjusts the projected top-net line without corrupting the ground-plane solve.

The solver should run in tiers:

1. Auto line/keypoint calibration: preferred, no taps.
2. ARKit-seeded auto calibration: preferred when iOS sidecar exists.
3. Manual review/tap calibration: fallback only.

The calibration summary should record the chosen tier and why any fallback was used.

### 5. Net Model Integration

Keep regulation net dimensions as the physical prior: 22 ft post spacing, 36 in at sidelines, 34 in at center. Add measured top-net observations as an evidence layer:

- `NetPlane` remains the physical world plane.
- `court_line_evidence.json` records observed top-net image evidence.
- Overlay and eval compare projected `NetPlane` top against observed top-net evidence.
- Ball/net crossing and player net-plane logic must use the calibrated `NetPlane`, but should expose low confidence when observed top-net residual is high.

### 6. Review And Label Standards

Extend review labels so the "lines and dots" standard is explicit:

- Keep `court_corners.json` for compatibility.
- Add or extend with `court_lines.json` containing semantic line segments and keypoint dots.
- Accepted labels should include at least one clear frame per clip with: four corners, both NVZ lines, near/far centerlines when visible, and top-net left/center/right.
- Use these labels to score automatic detection and to create future training data for the court-keypoint/line model.

### 7. Evaluation Gates

Add no-tap calibration gates before claiming this is fixed:

- `auto_calibration_ready=true` on the accepted static prototype clips without using manual taps.
- Outer-border median residual at or below the current visual quality.
- NVZ/kitchen p95 residual below a configured threshold against labels.
- Centerline detected and rendered when visible, with residual reported separately.
- Top-net residual reported separately against labels or high-confidence pixel evidence.
- Overlay index must include all semantic line IDs and residual summaries, not just line counts.

Until those pass, the system should say "auto calibration attempted but not verified" rather than presenting manual-corner artifacts as successful automatic court tracking.

## Files To Change

- `threed/racketsport/court_templates.py`: add centerline segments.
- `threed/racketsport/court_keypoint_net.py`: add baseline-center dots and preserve solvePnP ordering.
- `threed/racketsport/schemas/__init__.py`: add court evidence schema types and register `court_line_evidence`.
- `threed/racketsport/court_line_evidence.py`: new CPU-safe detector/scorer/aggregator module.
- `threed/racketsport/court_calibration.py`: add calibration builder from line/keypoint evidence and richer residual summaries.
- `threed/racketsport/net_plane.py`: add comparison helpers between regulation net projection and observed top-net evidence.
- `threed/racketsport/calibration_overlay.py`: render centerlines, evidence lines/dots, top-net residuals, and per-line confidence.
- `threed/racketsport/court_corner_review.py`: keep old corner import but support richer review payloads.
- `threed/racketsport/eval/calib_eval.py`: add no-tap and semantic-line gates.
- `threed/racketsport/orchestrator.py`: add automatic calibration runner before manual fallback.
- `scripts/racketsport/auto_calibrate_court.py`: CLI for running automatic court evidence and calibration on local frames/videos.
- `tests/racketsport/`: focused TDD coverage for each behavior.

## Testing Strategy

Use TDD for implementation:

1. Template/keypoint tests fail first for missing centerline segments and baseline-center dots.
2. Schema tests fail first for missing `court_line_evidence` artifact validation.
3. Evidence scoring tests use synthetic projected segments with jitter, occlusion, distractor parallel lines, and top-net candidates.
4. Calibration tests verify richer correspondences lower or report interior-line residuals and never silently use four-corner `0px` as proof.
5. Overlay tests require centerline IDs and evidence residual fields.
6. CLI/eval tests run on local sample frames without writing into checked-in run artifacts.

Verification commands:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --basetemp=/tmp/pickleball_court_tests \
  tests/racketsport/test_court_geometry.py \
  tests/racketsport/test_court_keypoint_net.py \
  tests/racketsport/test_court_calibration.py \
  tests/racketsport/test_calibration_overlay.py \
  tests/racketsport/test_court_corner_review_calibration.py
```

After implementation, also render fresh overlays to `/tmp` from the accepted prototype clips and inspect the no-tap residual report before updating any committed run artifact.

## Open Assumptions

- The "lines and dots" standard the user referenced is represented by existing review conventions plus prior visual guidance, not by a checked-in interior-line schema. This design makes it explicit as `court_lines.json` and `court_line_evidence.json`.
- The first no-tap implementation should be CPU-safe and deterministic. Learned model training is a second milestone after labels and gates exist.
- Manual taps are allowed only as fallback and labeling support, never as the default product path.
