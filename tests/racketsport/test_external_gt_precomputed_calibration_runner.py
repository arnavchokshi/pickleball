from __future__ import annotations

from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.external_gt_precomputed_calibration_runner import PrecomputedCalibrationRunner
from threed.racketsport.orchestrator import StageContext
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    CourtLineEvidence,
    ReprojectionError,
    validate_artifact_file,
)


def _calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="external_gt"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=1.25, p95=2.5),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def test_precomputed_calibration_runner_copies_trusted_calibration_and_fails_closed_evidence(
    tmp_path: Path,
) -> None:
    inputs_dir = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs_dir.mkdir()
    run_dir.mkdir()
    calibration = _calibration()
    (inputs_dir / "court_calibration.json").write_text(calibration.model_dump_json() + "\n", encoding="utf-8")

    stage_run = PrecomputedCalibrationRunner(source_note="trusted external GT camera").run(
        StageContext(clip="aspset_510_sample", inputs_dir=inputs_dir, run_dir=run_dir, sport="pickleball")
    )

    assert stage_run.stage == "calibration"
    assert stage_run.status == "ran"
    assert stage_run.real_model is False
    assert stage_run.source_mode == "precomputed_external_gt_calibration"
    assert stage_run.produced_artifacts == (
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
    )
    assert stage_run.metrics == {"reprojection_median_px": 1.25, "reprojection_p95_px": 2.5}
    assert stage_run.notes == (
        "trusted external GT camera",
        "court_zones.json/net_plane.json are schema-compatibility formalities, not real "
        "measurements of this non-pickleball scene",
        "court_line_evidence.json is deliberately fail-closed (zero observations): there is "
        "no real pickleball court to detect lines/net on in this footage",
    )

    copied_calibration = validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    assert isinstance(copied_calibration, CourtCalibration)
    assert copied_calibration.model_dump(mode="json") == calibration.model_dump(mode="json")
    validate_artifact_file("court_zones", run_dir / "court_zones.json")
    validate_artifact_file("net_plane", run_dir / "net_plane.json")
    line_evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    assert isinstance(line_evidence, CourtLineEvidence)
    assert line_evidence.source == "external_gt_precomputed_calibration_no_video_court"
    assert line_evidence.aggregate.auto_calibration_ready is False
    assert line_evidence.aggregate.missing_required_line_ids
    assert line_evidence.aggregate.missing_required_net_ids
    assert "external_ground_truth_footage_has_no_pickleball_court_to_detect" in line_evidence.aggregate.reasons


def test_precomputed_calibration_runner_requires_seed_calibration(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs_dir.mkdir()
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="requires a pre-built court_calibration.json"):
        PrecomputedCalibrationRunner(source_note="trusted external GT camera").run(
            StageContext(clip="missing", inputs_dir=inputs_dir, run_dir=run_dir, sport="pickleball")
        )
