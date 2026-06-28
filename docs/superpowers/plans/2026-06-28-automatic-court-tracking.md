# Automatic Court Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-tap court calibration path that detects and scores court outline, kitchen/NVZ lines, centerlines, and top-net evidence before falling back to manual review.

**Architecture:** Add semantic court completeness first, then introduce a schema-valid `court_line_evidence.json` artifact. Use deterministic CPU-safe line/keypoint scoring against the regulation template, then feed accepted evidence into calibration overlays, eval summaries, and the orchestrator's calibration stage.

**Tech Stack:** Python 3.11, Pydantic, pytest, OpenCV when available, existing `threed.racketsport` calibration/orchestrator modules.

**Status after 2026-06-28 pass:** Tasks 1-3 and the overlay/top-net trust portions of Task 4 are implemented. Task 5 is partially implemented: the default calibration stage now samples video/frame evidence, writes `court_line_evidence.json`, and fail-closes video-backed runs when semantic court evidence is not ready. The true no-tap calibration solver and `auto_calibrate_court.py` CLI remain open; do not mark CAL-3 verified until a random video can create a trusted `court_calibration.json` without manual taps and pass the real-clip gates.

---

## File Structure

- `threed/racketsport/court_templates.py`: add pickleball centerline world segments.
- `threed/racketsport/court_keypoint_net.py`: add baseline-center keypoint dots and preserve deterministic solvePnP order.
- `threed/racketsport/schemas/__init__.py`: add `CourtLineObservation`, `CourtKeypointObservation`, `NetLineObservation`, `CourtLineEvidence`, and register schema name `court_line_evidence`.
- `threed/racketsport/court_line_evidence.py`: new detector/scorer helpers. It owns semantic line scoring, temporal aggregation, and optional OpenCV frame candidate extraction.
- `threed/racketsport/court_calibration.py`: add calibration-from-evidence builder and residual helpers without breaking existing manual calibration.
- `threed/racketsport/calibration_overlay.py`: render centerlines and expose evidence residuals when evidence is provided.
- `threed/racketsport/eval/calib_eval.py`: include no-tap evidence readiness and semantic residuals in metrics.
- `threed/racketsport/orchestrator.py`: add automatic calibration runner before manual fallback.
- `scripts/racketsport/auto_calibrate_court.py`: CLI for local no-tap evidence generation/calibration.
- `tests/racketsport/test_court_geometry.py`: template centerline tests.
- `tests/racketsport/test_court_keypoint_net.py`: baseline-center dot tests.
- `tests/racketsport/test_court_line_evidence.py`: new evidence schema/scoring/aggregation tests.
- `tests/racketsport/test_court_calibration.py`: evidence calibration tests.
- `tests/racketsport/test_calibration_overlay.py`: centerline/evidence overlay tests.
- `tests/racketsport/test_orchestrator_spine.py`: automatic calibration runner selection tests.

### Task 1: Semantic Court Completeness

**Files:**
- Modify: `tests/racketsport/test_court_geometry.py`
- Modify: `tests/racketsport/test_court_keypoint_net.py`
- Modify: `threed/racketsport/court_templates.py`
- Modify: `threed/racketsport/court_keypoint_net.py`

- [ ] **Step 1: Write failing template test**

Add an assertion that pickleball `line_segments_m` includes `near_centerline` from `(0, -22ft, 0)` to `(0, -7ft, 0)` and `far_centerline` from `(0, 7ft, 0)` to `(0, 22ft, 0)`.

- [ ] **Step 2: Run failing template test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_geometry.py::test_pickleball_template_matches_regulation_dimensions
```

Expected: FAIL because the centerline IDs are absent.

- [ ] **Step 3: Implement centerline segments**

In `CourtTemplate.line_segments_m`, add pickleball-only centerlines when `sport == "pickleball"` and `non_volley_zone_ft` exists.

- [ ] **Step 4: Write failing keypoint taxonomy test**

Update `test_pickleball_keypoint_taxonomy_includes_corners_nvz_and_centerline_intersections` to require `near_baseline_center` and `far_baseline_center`.

- [ ] **Step 5: Run failing keypoint test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_keypoint_net.py::test_pickleball_keypoint_taxonomy_includes_corners_nvz_and_centerline_intersections
```

Expected: FAIL because baseline-center dots are absent.

- [ ] **Step 6: Implement baseline-center dots**

Add `near_baseline_center` at `(0, -22ft, 0)` and `far_baseline_center` at `(0, 22ft, 0)` to `PICKLEBALL_KEYPOINTS` in deterministic court traversal order.

- [ ] **Step 7: Verify Task 1**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_geometry.py tests/racketsport/test_court_keypoint_net.py
```

Expected: PASS.

### Task 2: Court Evidence Schema

**Files:**
- Modify: `tests/racketsport/test_schemas.py`
- Modify: `threed/racketsport/schemas/__init__.py`

- [ ] **Step 1: Write failing schema test**

Add a test validating a `court_line_evidence` artifact with one line observation, one keypoint observation, one top-net observation, and an aggregate readiness block.

- [ ] **Step 2: Run failing schema test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_schemas.py::test_court_line_evidence_schema_validates_semantic_lines_and_net
```

Expected: FAIL because the schema classes and registry entry are missing.

- [ ] **Step 3: Implement schema models**

Add strict Pydantic models for line, keypoint, net, residual summary, and `CourtLineEvidence`. Register `ARTIFACT_MODELS["court_line_evidence"]`.

- [ ] **Step 4: Verify Task 2**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_schemas.py
```

Expected: PASS.

### Task 3: Deterministic Evidence Scoring

**Files:**
- Create: `tests/racketsport/test_court_line_evidence.py`
- Create: `threed/racketsport/court_line_evidence.py`

- [ ] **Step 1: Write failing scoring tests**

Cover:

- A projected semantic line accepts a close, same-orientation candidate.
- A nearby parallel distractor with poor overlap loses to the correct candidate.
- A top-net candidate is scored separately from ground court lines.
- Aggregation marks `auto_calibration_ready` false when required NVZ/centerline evidence is missing.

- [ ] **Step 2: Run failing scoring tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_line_evidence.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement scorer primitives**

Implement pure-Python geometry helpers:

- `score_line_candidate(expected, candidate)`.
- `select_best_line_observation(line_id, expected, candidates)`.
- `build_evidence_from_projected_lines(...)`.
- `aggregate_court_line_evidence(...)`.

Do not require OpenCV for these tests.

- [ ] **Step 4: Verify Task 3**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_line_evidence.py
```

Expected: PASS.

### Task 4: Evidence-Based Calibration And Overlay

**Files:**
- Modify: `tests/racketsport/test_court_calibration.py`
- Modify: `tests/racketsport/test_calibration_overlay.py`
- Modify: `threed/racketsport/court_calibration.py`
- Modify: `threed/racketsport/calibration_overlay.py`

- [ ] **Step 1: Write failing calibration test**

Add a test that builds calibration from semantic evidence with more than four points and asserts interior-line residuals are reported separately from corner reprojection.

- [ ] **Step 2: Run failing calibration test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_calibration.py::test_calibration_from_court_line_evidence_reports_interior_residuals
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement evidence calibration builder**

Add a builder that extracts high-confidence keypoints from `CourtLineEvidence`, calls existing `_build_calibration`, and returns calibration plus residual summary for all semantic lines.

- [ ] **Step 4: Write failing overlay test**

Require overlay `court_line_ids` to include `near_centerline` and `far_centerline`, and include an evidence summary when supplied.

- [ ] **Step 5: Implement overlay support**

Render centerline segments from template automatically. Add optional evidence summary fields without changing existing callers.

- [ ] **Step 6: Verify Task 4**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_court_calibration.py tests/racketsport/test_calibration_overlay.py
```

Expected: PASS.

### Task 5: Automatic Calibration Runner And CLI

**Files:**
- Modify: `tests/racketsport/test_orchestrator_spine.py`
- Create: `tests/racketsport/test_auto_calibrate_court_cli.py`
- Modify: `threed/racketsport/orchestrator.py`
- Create: `scripts/racketsport/auto_calibrate_court.py`

- [ ] **Step 1: Write failing runner test**

Assert the calibration stage tries automatic evidence when frames are present, writes `court_line_evidence.json`, and falls back to manual sidecar only when automatic evidence is not ready.

- [ ] **Step 2: Run failing runner test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_orchestrator_spine.py::test_calibration_runner_prefers_auto_evidence_before_manual_sidecar
```

Expected: FAIL because no automatic runner exists.

- [ ] **Step 3: Implement automatic runner**

Add an `AutomaticCalibrationRunner` that consumes local frame images, uses deterministic evidence scoring, writes `court_line_evidence.json`, and produces calibration artifacts only if `auto_calibration_ready` is true. Keep `ManualCalibrationRunner` as fallback.

- [ ] **Step 4: Write failing CLI test**

Assert `scripts/racketsport/auto_calibrate_court.py` can read a frame directory and write artifacts to a temp output directory.

- [ ] **Step 5: Implement CLI**

Add a small CLI with `--frames`, `--sidecar`, `--sport`, and `--out` arguments. It must never write into `runs/` unless the caller points `--out` there.

- [ ] **Step 6: Verify Task 5**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_orchestrator_spine.py tests/racketsport/test_auto_calibrate_court_cli.py
```

Expected: PASS.

### Task 6: Evaluation And Local Clip Check

**Files:**
- Modify: `tests/racketsport/test_eval_metrics.py`
- Modify: `threed/racketsport/eval/calib_eval.py`

- [ ] **Step 1: Write failing eval test**

Add a test that calibration eval reads `court_line_evidence.json` and reports `auto_calibration_ready`, missing semantic lines, NVZ residuals, centerline residuals, and net residuals.

- [ ] **Step 2: Run failing eval test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/racketsport/test_eval_metrics.py::test_calib_eval_reports_court_line_evidence_readiness
```

Expected: FAIL because eval ignores evidence.

- [ ] **Step 3: Implement eval reporting**

Extend calibration eval metrics without changing existing required artifacts. Missing evidence should be explicit, not silently successful.

- [ ] **Step 4: Verify focused suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --basetemp=/tmp/pickleball_court_tests \
  tests/racketsport/test_court_geometry.py \
  tests/racketsport/test_court_keypoint_net.py \
  tests/racketsport/test_schemas.py \
  tests/racketsport/test_court_line_evidence.py \
  tests/racketsport/test_court_calibration.py \
  tests/racketsport/test_calibration_overlay.py \
  tests/racketsport/test_orchestrator_spine.py \
  tests/racketsport/test_eval_metrics.py
```

Expected: PASS.

- [ ] **Step 5: Render local no-tap overlays to `/tmp`**

Run the new CLI against accepted prototype frames and output to `/tmp/pickleball_auto_court/<clip>`. Inspect generated evidence summaries and overlays before claiming visual success.
