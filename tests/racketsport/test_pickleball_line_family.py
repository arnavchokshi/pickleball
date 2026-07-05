from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.pickleball_line_family import analyze_frame_line_family


SCRIPT_PATH = "scripts/racketsport/diagnose_pickleball_line_family.py"


def test_pickleball_centerline_outranks_tennis_line_crossing_kitchen() -> None:
    image, calibration = _synthetic_court(
        pickleball_bgr=(0, 230, 255),
        tennis_bgr=(255, 0, 255),
        draw_tennis_crossing=True,
        tennis_x_m=0.18,
    )

    result = analyze_frame_line_family(image, calibration, frame_id="synthetic")

    assert result["verified"] is False
    assert result["not_cal3_verified"] is True
    assert result["dominant_pickleball_color_family"]["coherent"] is True
    assert result["auto_centerline_evidence_ready"] is True
    assert result["centerline_verdicts"]["near_centerline"]["support_fraction"] >= 0.70
    assert result["centerline_verdicts"]["far_centerline"]["support_fraction"] >= 0.70
    assert result["centerline_verdicts"]["near_centerline"]["kitchen_crossing_violation"] is False
    assert result["centerline_verdicts"]["far_centerline"]["kitchen_crossing_violation"] is False

    tennis_candidates = [
        candidate
        for candidate in result["candidate_lines"]
        if candidate["role"] == "tennis_artifact" and candidate["kitchen_violation"]
    ]
    assert tennis_candidates, result["candidate_lines"]
    assert max(candidate["raw_support_fraction"] for candidate in tennis_candidates) >= 0.85
    assert result["centerline_verdicts"]["near_centerline"]["score"] > max(
        candidate["score"] for candidate in tennis_candidates
    )


def test_same_color_crossing_line_keeps_centerline_evidence_not_ready() -> None:
    image, calibration = _synthetic_court(
        pickleball_bgr=(0, 230, 255),
        tennis_bgr=(0, 230, 255),
        draw_tennis_crossing=True,
        tennis_x_m=0.0,
    )

    result = analyze_frame_line_family(image, calibration, frame_id="same_color")

    assert result["auto_centerline_evidence_ready"] is False
    assert result["centerline_verdicts"]["near_centerline"]["kitchen_crossing_violation"] is True
    assert result["centerline_verdicts"]["far_centerline"]["kitchen_crossing_violation"] is True
    assert "same_color_centerline_crosses_kitchen" in result["reasons"]


def test_no_lines_image_returns_empty_candidates_without_crashing() -> None:
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    calibration = _calibration_payload()

    result = analyze_frame_line_family(image, calibration, frame_id="blank")

    assert result["detected_segment_count"] == 0
    assert result["selected_lines"] == []
    assert result["candidate_lines"] == []
    assert result["auto_centerline_evidence_ready"] is False
    assert result["verified"] is False
    assert result["not_gate_verified"] is True


def test_frame_analysis_is_deterministic() -> None:
    image, calibration = _synthetic_court(
        pickleball_bgr=(0, 230, 255),
        tennis_bgr=(255, 0, 255),
        draw_tennis_crossing=True,
        tennis_x_m=0.18,
    )

    first = analyze_frame_line_family(image, calibration, frame_id="deterministic")
    second = analyze_frame_line_family(image, calibration, frame_id="deterministic")

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_diagnose_pickleball_line_family_cli_writes_diagnostic_artifacts(tmp_path: Path) -> None:
    image, calibration = _synthetic_court(
        pickleball_bgr=(0, 230, 255),
        tennis_bgr=(255, 0, 255),
        draw_tennis_crossing=True,
        tennis_x_m=0.18,
    )
    frame_path = tmp_path / "frame.png"
    calibration_path = tmp_path / "court_calibration.json"
    out_dir = tmp_path / "out"
    cv2.imwrite(str(frame_path), image)
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH,
            "--frame",
            str(frame_path),
            "--calibration",
            str(calibration_path),
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out_dir / "line_family_diagnostic.json").read_text(encoding="utf-8"))
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["auto_centerline_evidence_ready"] is True
    assert (out_dir / "line_family_diagnostic.md").is_file()
    assert list(out_dir.glob("overlay_*.png"))


def test_diagnose_pickleball_line_family_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Diagnose pickleball line color-family and centerline topology" in completed.stdout


def _synthetic_court(
    *,
    pickleball_bgr: tuple[int, int, int],
    tennis_bgr: tuple[int, int, int],
    draw_tennis_crossing: bool,
    tennis_x_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    image[:, :] = (35, 45, 45)
    calibration = _calibration_payload()

    if draw_tennis_crossing:
        _draw_court_line(image, calibration, (tennis_x_m, -6.7056), (tennis_x_m, 6.7056), tennis_bgr, 5)

    for y in (-6.7056, -2.1336, 2.1336, 6.7056):
        _draw_court_line(image, calibration, (-3.048, y), (3.048, y), pickleball_bgr, 5)
    for x in (-3.048, 3.048):
        _draw_court_line(image, calibration, (x, -6.7056), (x, 6.7056), pickleball_bgr, 5)
    _draw_court_line(image, calibration, (0.0, -6.7056), (0.0, -2.1336), pickleball_bgr, 5)
    _draw_court_line(image, calibration, (0.0, 2.1336), (0.0, 6.7056), pickleball_bgr, 5)
    return image, calibration


def _calibration_payload() -> dict[str, object]:
    return {
        "coordinate_frame": "court_netcenter_z_up_m",
        "homography": [
            [80.0, 0.0, 480.0],
            [0.0, -45.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _draw_court_line(
    image: np.ndarray,
    calibration: dict[str, object],
    p1_m: tuple[float, float],
    p2_m: tuple[float, float],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    h = calibration["homography"]
    assert isinstance(h, list)
    x1, y1 = _project(h, p1_m)
    x2, y2 = _project(h, p2_m)
    cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), color_bgr, thickness, cv2.LINE_AA)


def _project(h: list[list[float]], point_m: tuple[float, float]) -> tuple[float, float]:
    x, y = point_m
    scale = h[2][0] * x + h[2][1] * y + h[2][2]
    if scale == 0.0:
        raise AssertionError("degenerate synthetic homography")
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / scale,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / scale,
    )
