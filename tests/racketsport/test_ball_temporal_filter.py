from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_temporal_filter import (
    filter_ball_track_temporal_outliers,
    filter_ball_track_temporal_path,
)
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _write_track(path: Path) -> None:
    frames = [
        {"t": 0 / 30.0, "xy": [0.0, 0.0], "conf": 0.9, "visible": True},
        {"t": 1 / 30.0, "xy": [10.0, 0.0], "conf": 0.9, "visible": True},
        {"t": 2 / 30.0, "xy": [500.0, 500.0], "conf": 0.9, "visible": True},
        {"t": 3 / 30.0, "xy": [30.0, 0.0], "conf": 0.9, "visible": True},
        {"t": 4 / 30.0, "xy": [40.0, 0.0], "conf": 0.0, "visible": False},
        {"t": 5 / 30.0, "xy": [50.0, 0.0], "conf": 0.9, "visible": True},
        {"t": 6 / 30.0, "xy": [1000.0, 1000.0], "conf": 0.9, "visible": True},
        {"t": 7 / 30.0, "xy": [70.0, 0.0], "conf": 0.9, "visible": True},
    ]
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def test_temporal_filter_keeps_longest_motion_chain_and_interpolates_short_gap(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    _write_track(track_path)

    payload, summary = filter_ball_track_temporal_path(
        ball_track_path=track_path,
        max_speed_px_per_second=900.0,
        base_jump_px=20.0,
        max_link_gap_frames=4,
        max_interpolate_gap_frames=2,
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[2].visible is True
    assert filtered.frames[2].approx is True
    assert filtered.frames[2].xy == pytest.approx([20.0, 0.0])
    assert filtered.frames[4].visible is True
    assert filtered.frames[4].approx is True
    assert filtered.frames[4].xy == pytest.approx([40.0, 0.0])
    assert filtered.frames[6].visible is True
    assert filtered.frames[6].approx is True
    assert filtered.frames[6].xy == pytest.approx([60.0, 0.0])
    assert filtered.frames[7].visible is True
    assert summary["uses_human_clicks"] is False
    assert summary["rejected_off_path_count"] == 2
    assert summary["interpolated_count"] == 3


def test_temporal_filter_cli_writes_schema_valid_output(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    out = tmp_path / "ball_temporal.json"
    summary_out = tmp_path / "ball_temporal_summary.json"
    _write_track(track_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/filter_ball_temporal.py",
            "--ball-track",
            str(track_path),
            "--max-speed-px-per-second",
            "900",
            "--base-jump-px",
            "20",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["uses_human_clicks"] is False
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)


def test_temporal_outlier_filter_removes_only_isolated_impossible_jumps(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    _write_track(track_path)

    payload, summary = filter_ball_track_temporal_outliers(
        ball_track_path=track_path,
        max_speed_px_per_second=900.0,
        base_jump_px=20.0,
        max_neighbor_gap_frames=4,
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[1].visible is True
    assert filtered.frames[2].visible is False
    assert filtered.frames[3].visible is True
    assert filtered.frames[4].visible is False
    assert filtered.frames[5].visible is True
    assert filtered.frames[6].visible is False
    assert filtered.frames[7].visible is True
    assert summary["rejected_isolated_outlier_count"] == 2
    assert summary["uses_human_clicks"] is False
