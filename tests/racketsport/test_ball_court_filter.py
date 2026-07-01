from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_court_filter import (
    build_target_court_polygon,
    filter_ball_track_to_target_court_metric_margin,
    filter_ball_track_to_target_court,
    point_in_polygon_with_margin,
)
from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.schemas import BallTrack, CourtCalibration, validate_artifact_file


def _calibration_for_image_court(width: float = 100.0, height: float = 200.0) -> CourtCalibration:
    template = get_court_template("pickleball")
    image_corners = [[0.0, height], [width, height], [width, 0.0], [0.0, 0.0]]
    homography = homography_from_planar_points(template.corners_m, image_corners)
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "intrinsics": {
                "fx": width * 1.2,
                "fy": width * 1.2,
                "cx": width / 2.0,
                "cy": height / 2.0,
                "dist": [],
                "source": "test",
            },
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 10.0],
                "camera_height_m": 10.0,
            },
            "homography": homography,
            "image_pts": image_corners,
            "world_pts": template.corners_m,
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "warn", "reasons": ["test"]},
        }
    )


def _write_ball_track(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "tracknet",
                "frames": [
                    {"t": 0.0, "xy": [50.0, 100.0], "conf": 1.0, "visible": True},
                    {"t": 1 / 30.0, "xy": [160.0, 100.0], "conf": 1.0, "visible": True},
                    {"t": 2 / 30.0, "xy": [170.0, 100.0], "conf": 0.0, "visible": False},
                ],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def _write_ball_track_for_world_points(path: Path, calibration: CourtCalibration, world_points: list[list[float]]) -> None:
    image_points = project_planar_points(calibration.homography, [[point[0], point[1], 0.0] for point in world_points])
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "tracknet",
                "frames": [
                    {"t": index / 30.0, "xy": xy, "conf": 1.0, "visible": True}
                    for index, xy in enumerate(image_points)
                ],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def test_target_court_polygon_scales_from_calibration_image_to_video_size() -> None:
    polygon = build_target_court_polygon(
        _calibration_for_image_court(width=100.0, height=200.0),
        target_size=(200, 400),
    )

    expected = [[0.0, 400.0], [200.0, 400.0], [200.0, 0.0], [0.0, 0.0]]
    for actual_point, expected_point in zip(polygon, expected, strict=True):
        assert actual_point == pytest.approx(expected_point)


def test_point_in_polygon_with_margin_accepts_near_edge_and_rejects_background() -> None:
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]

    assert point_in_polygon_with_margin([50.0, 50.0], polygon, margin_px=0.0) is True
    assert point_in_polygon_with_margin([105.0, 50.0], polygon, margin_px=6.0) is True
    assert point_in_polygon_with_margin([130.0, 50.0], polygon, margin_px=6.0) is False


def test_filter_ball_track_marks_visible_background_points_invisible(tmp_path: Path) -> None:
    ball_track_path = tmp_path / "ball_track.json"
    _write_ball_track(ball_track_path)

    payload, summary = filter_ball_track_to_target_court(
        ball_track_path=ball_track_path,
        calibration=_calibration_for_image_court(width=100.0, height=200.0),
        target_size=(100, 200),
        margin_px=10.0,
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[0].conf == pytest.approx(1.0)
    assert filtered.frames[1].visible is False
    assert filtered.frames[1].conf == pytest.approx(0.0)
    assert filtered.frames[2].visible is False
    assert summary["visible_before"] == 2
    assert summary["visible_after"] == 1
    assert summary["rejected_outside_target_court"] == 1
    assert summary["status"] == "TESTED-ON-REAL-DATA"


def test_metric_court_margin_uses_regulation_meters_not_fixed_pixels(tmp_path: Path) -> None:
    calibration = _calibration_for_image_court(width=100.0, height=200.0)
    template = get_court_template("pickleball")
    half_width = template.width_m / 2.0
    ball_track_path = tmp_path / "ball_track.json"
    _write_ball_track_for_world_points(
        ball_track_path,
        calibration,
        [
            [0.0, 0.0],
            [half_width + 0.4, 0.0],
            [half_width + 0.7, 0.0],
        ],
    )

    payload, summary = filter_ball_track_to_target_court_metric_margin(
        ball_track_path=ball_track_path,
        calibration=calibration,
        target_size=(100, 200),
        margin_m=0.5,
    )

    filtered = BallTrack.model_validate(payload)
    assert [frame.visible for frame in filtered.frames] == [True, True, False]
    assert summary["court_margin_m"] == pytest.approx(0.5)
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["visible_before"] == 3
    assert summary["visible_after"] == 2
    assert summary["rejected_outside_target_court"] == 1


def test_filter_ball_track_cli_writes_filtered_track_and_summary(tmp_path: Path) -> None:
    ball_track_path = tmp_path / "ball_track.json"
    calibration_path = tmp_path / "court_calibration.json"
    out = tmp_path / "filtered_ball_track.json"
    summary_path = tmp_path / "summary.json"
    _write_ball_track(ball_track_path)
    calibration_path.write_text(
        json.dumps(_calibration_for_image_court(width=100.0, height=200.0).model_dump(mode="json")),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/filter_ball_track_to_court.py",
            "--ball-track",
            str(ball_track_path),
            "--calibration",
            str(calibration_path),
            "--target-size",
            "100",
            "200",
            "--margin-px",
            "10",
            "--out",
            str(out),
            "--summary-out",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["rejected_outside_target_court"] == 1
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["visible_after"] == 1
    assert written["status"] == "TESTED-ON-REAL-DATA"
