from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_line_calls import classify_ball_line_calls
from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.schemas import BallLineCalls, validate_artifact_file


def _xy_ft(x_ft: float, y_ft: float) -> list[float]:
    return [x_ft * FT_TO_M, y_ft * FT_TO_M]


def _ball_track_payload(*, bounces: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [100.0, 100.0], "conf": 0.9, "visible": True, "approx": False},
        ],
        "bounces": bounces,
    }


def test_classifies_inside_outside_and_kitchen_bounces() -> None:
    payload = _ball_track_payload(
        bounces=[
            {"t": 0.10, "world_xy": _xy_ft(-5.0, -12.0)},
            {"t": 0.25, "world_xy": _xy_ft(10.8, 1.0)},
            {"t": 0.40, "world_xy": _xy_ft(2.0, 3.0)},
        ],
    )

    result = classify_ball_line_calls(
        payload,
        sport="pickleball",
        uncertainty_radius_m=0.02,
        input_ball_track="ball_track.json",
    )

    artifact = BallLineCalls.model_validate(result)
    assert artifact.summary.total_bounces == 3
    assert artifact.summary.court_call_counts == {"in": 2, "out": 1, "unknown": 0}
    assert artifact.summary.kitchen_call_counts == {"non_nvz": 1, "nvz": 1, "unknown": 1}
    assert artifact.calls[0].court_call == "in"
    assert artifact.calls[0].kitchen_call == "non_nvz"
    assert artifact.calls[0].zone == "near_left_service"
    assert artifact.calls[1].court_call == "out"
    assert artifact.calls[1].nearest_boundary_line_id == "right_sideline"
    assert artifact.calls[2].court_call == "in"
    assert artifact.calls[2].kitchen_call == "nvz"
    assert artifact.calls[2].zone == "far_nvz"


def test_boundary_lines_are_in_when_uncertainty_does_not_cross_decision_band() -> None:
    payload = _ball_track_payload(bounces=[{"t": 0.5, "world_xy": _xy_ft(10.0, 0.0)}])

    result = classify_ball_line_calls(payload, sport="pickleball", uncertainty_radius_m=0.0)

    call = BallLineCalls.model_validate(result).calls[0]
    assert call.court_call == "in"
    assert call.nearest_boundary_line_id == "right_sideline"
    assert call.boundary_margin_m == pytest.approx(0.0)
    assert "on_boundary_line" in call.reasons


def test_uncertainty_band_abstains_near_sideline_and_nvz_line() -> None:
    payload = _ball_track_payload(
        bounces=[
            {"t": 0.5, "world_xy": _xy_ft(10.05, 0.0)},
            {"t": 0.6, "world_xy": _xy_ft(2.0, 7.05)},
        ],
    )

    result = classify_ball_line_calls(payload, sport="pickleball", uncertainty_radius_m=0.05)

    artifact = BallLineCalls.model_validate(result)
    assert artifact.calls[0].court_call == "unknown"
    assert artifact.calls[0].nearest_boundary_line_id == "right_sideline"
    assert "boundary_within_uncertainty" in artifact.calls[0].reasons
    assert artifact.calls[1].court_call == "in"
    assert artifact.calls[1].kitchen_call == "unknown"
    assert artifact.calls[1].nearest_kitchen_line_id == "far_nvz"
    assert "nvz_boundary_within_uncertainty" in artifact.calls[1].reasons


def test_tennis_court_calls_keep_boundary_confidence_without_nvz() -> None:
    payload = _ball_track_payload(bounces=[{"t": 0.5, "world_xy": _xy_ft(0.0, 0.0)}])

    result = classify_ball_line_calls(payload, sport="tennis", uncertainty_radius_m=0.05)

    call = BallLineCalls.model_validate(result).calls[0]
    assert call.court_call == "in"
    assert call.kitchen_call == "unknown"
    assert "no_nvz_for_sport" in call.reasons
    assert call.confidence > 0.0


def test_empty_bounces_fail_closed_with_no_calls() -> None:
    payload = _ball_track_payload(bounces=[])

    result = classify_ball_line_calls(payload, sport="pickleball")

    artifact = BallLineCalls.model_validate(result)
    assert artifact.calls == []
    assert artifact.summary.total_bounces == 0
    assert artifact.summary.status == "blocked"
    assert "ball_track has no bounces" in artifact.summary.reasons


def test_classify_ball_line_calls_cli_writes_schema_valid_artifact(tmp_path: Path) -> None:
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_line_calls.json"
    ball_track.write_text(
        json.dumps(_ball_track_payload(bounces=[{"t": 0.1, "world_xy": _xy_ft(-2.0, -8.0)}])),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/classify_ball_line_calls.py",
            "--ball-track",
            str(ball_track),
            "--sport",
            "pickleball",
            "--uncertainty-radius-m",
            "0.02",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["summary"]["status"] == "ready"
    artifact = validate_artifact_file("ball_line_calls", out)
    assert isinstance(artifact, BallLineCalls)
    assert artifact.calls[0].court_call == "in"
