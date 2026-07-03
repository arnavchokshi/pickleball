from __future__ import annotations

import subprocess
import sys

import numpy as np

from threed.racketsport.ball_blur_sidecar import estimate_blur_from_points


def test_estimate_blur_from_points_returns_midpoint_angle_and_extent() -> None:
    points = np.array([[10.0 + i, 20.0 + i] for i in range(15)], dtype=np.float64)

    estimate = estimate_blur_from_points(points)

    assert estimate is not None
    assert estimate["center_xy"] == [17.0, 27.0]
    assert estimate["blur_angle_deg"] == 45.0
    assert estimate["blur_length_px"] > 18.0
    assert estimate["blur_width_px"] >= 0.0
    assert estimate["quality"] == "clear"


def test_build_ball_blur_sidecar_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_blur_sidecar.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Estimate blur sidecar attributes" in completed.stdout
    assert "--video" in completed.stdout
    assert "--ball-track" in completed.stdout
    assert "--out-json" in completed.stdout
