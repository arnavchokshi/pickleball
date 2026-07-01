from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_ransac_arc_gate import filter_ball_track_ransac_arcs
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _write_arc_track(path: Path) -> None:
    frames = []
    for index in range(7):
        x = 20.0 + 8.0 * index
        y = 30.0 + 2.0 * index + 0.45 * index * index
        if index == 4:
            y += 24.0
        frames.append({"t": index / 30.0, "xy": [x, y], "conf": 0.8, "visible": True})
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "fused",
                "frames": frames,
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def test_ransac_arc_gate_rejects_outlier_over_residual_threshold(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    _write_arc_track(track_path)

    payload, summary = filter_ball_track_ransac_arcs(
        ball_track_path=track_path,
        max_residual_px=5.0,
        min_fit_points=4,
        max_gap_frames=6,
    )

    filtered = BallTrack.model_validate(payload)
    assert [frame.visible for frame in filtered.frames] == [True, True, True, True, False, True, True]
    assert filtered.frames[4].conf == pytest.approx(0.0)
    assert summary["artifact_type"] == "racketsport_ball_ransac_arc_recovery"
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["max_residual_px"] == pytest.approx(5.0)
    assert summary["evaluated_segment_count"] == 1
    assert summary["rejected_ransac_outlier_count"] == 1
    assert summary["uses_human_clicks"] is False
    assert summary["not_ground_truth"] is True


def test_ransac_arc_gate_cli_writes_schema_valid_track_and_summary(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    out_path = tmp_path / "ransac_ball_track.json"
    summary_path = tmp_path / "ransac_summary.json"
    _write_arc_track(track_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/filter_ball_ransac_arc.py",
            "--ball-track",
            str(track_path),
            "--max-residual-px",
            "5",
            "--min-fit-points",
            "4",
            "--max-gap-frames",
            "6",
            "--out",
            str(out_path),
            "--summary-out",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["rejected_ransac_outlier_count"] == 1
    assert isinstance(validate_artifact_file("ball_track", out_path), BallTrack)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["artifact_type"] == "racketsport_ball_ransac_arc_recovery"
    assert written["status"] == "TESTED-ON-REAL-DATA"
