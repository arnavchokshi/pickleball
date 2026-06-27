from __future__ import annotations

import json
import subprocess
import sys

import pytest

from threed.racketsport.calibration_overlay import build_calibration_overlay
from threed.racketsport.net_plane import build_net_plane
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError


def _synthetic_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[20.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )


def test_overlay_emits_projected_court_lines_and_net_points():
    overlay = build_calibration_overlay(_synthetic_calibration(), net_plane=build_net_plane("pickleball"))

    assert overlay["schema_version"] == 1
    assert overlay["sport"] == "pickleball"
    assert {line["id"] for line in overlay["court_lines"]} >= {
        "near_baseline",
        "far_baseline",
        "left_sideline",
        "right_sideline",
        "near_nvz",
        "far_nvz",
        "net",
    }
    near_baseline = next(line for line in overlay["court_lines"] if line["id"] == "near_baseline")
    assert near_baseline["image"][0] == pytest.approx([899.04, 674.112])
    assert near_baseline["image"][1] == pytest.approx([1020.96, 674.112])
    assert set(overlay["net_points"]) == {"left_post", "right_post", "center"}
    assert overlay["net_points"]["left_post"][0] < 960.0
    assert overlay["net_points"]["right_post"][0] > 960.0
    assert overlay["net_points"]["center"] == pytest.approx([960.0, 540.0])
    assert overlay["summary"]["court_line_count"] == len(overlay["court_lines"])
    assert overlay["summary"]["net_point_count"] == 3


def test_overlay_rejects_mismatched_net_plane_sport():
    with pytest.raises(ValueError, match="net plane endpoints do not match calibration sport"):
        build_calibration_overlay(_synthetic_calibration(), net_plane=build_net_plane("tennis"))


def test_render_calibration_overlay_cli_writes_svg_and_json_summary(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    net_path = tmp_path / "net_plane.json"
    svg_path = tmp_path / "calibration_overlay.svg"
    summary_path = tmp_path / "calibration_overlay.json"
    calibration_path.write_text(_synthetic_calibration().model_dump_json(), encoding="utf-8")
    net_path.write_text(build_net_plane("pickleball").model_dump_json(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(calibration_path),
            "--net-plane",
            str(net_path),
            "--out",
            str(svg_path),
            "--summary-out",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    svg = svg_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert completed.stdout == f"wrote {svg_path}\nwrote {summary_path}\n"
    assert "<svg" in svg
    assert 'data-line-id="near_baseline"' in svg
    assert 'data-net-point-id="center"' in svg
    assert summary["summary"]["court_line_count"] >= 7
    assert summary["summary"]["net_point_count"] == 3


def test_render_calibration_overlay_cli_fails_cleanly_for_missing_input(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "calibration_overlay.svg"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "missing calibration artifact" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_render_calibration_overlay_cli_fails_cleanly_for_invalid_input(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(calibration_path),
            "--out",
            str(tmp_path / "calibration_overlay.svg"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Field required" in completed.stderr
    assert "Traceback" not in completed.stderr
