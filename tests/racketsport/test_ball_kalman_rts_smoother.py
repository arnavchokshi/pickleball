from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_kalman_rts_smoother import smooth_ball_track_kalman_rts
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _write_gap_track(path: Path) -> None:
    frames = []
    for index in range(8):
        visible = index in {0, 1, 4, 5, 6, 7}
        frames.append(
            {
                "t": index / 30.0,
                "xy": [10.0 * index, 40.0 + (0.6 if index in {1, 5} else -0.4)],
                "conf": 0.82 if visible else 0.0,
                "visible": visible,
            }
        )
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


def test_kalman_rts_smoother_fills_short_gaps_and_reports_jitter(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    _write_gap_track(track_path)

    payload, summary = smooth_ball_track_kalman_rts(
        ball_track_path=track_path,
        max_gap_fill_frames=6,
        measurement_variance_px=1.0,
        process_variance=0.05,
    )

    smoothed = BallTrack.model_validate(payload)
    assert smoothed.frames[2].visible is True
    assert smoothed.frames[2].approx is True
    assert smoothed.frames[3].visible is True
    assert smoothed.frames[3].approx is True
    assert smoothed.frames[2].conf == pytest.approx(0.41)
    assert smoothed.frames[2].xy[0] == pytest.approx(20.0, abs=4.0)
    assert summary["artifact_type"] == "racketsport_ball_kalman_rts_smoother"
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["max_gap_fill_frames"] == 6
    assert summary["filled_gap_frame_count"] == 2
    assert summary["max_filled_gap_frames"] == 2
    assert summary["jitter_px_std"] < 2.0
    assert summary["uses_human_clicks"] is False
    assert summary["not_ground_truth"] is True


def test_kalman_rts_smoother_cli_writes_schema_valid_track_and_summary(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    out_path = tmp_path / "kalman_ball_track.json"
    summary_path = tmp_path / "kalman_summary.json"
    _write_gap_track(track_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/smooth_ball_kalman_rts.py",
            "--ball-track",
            str(track_path),
            "--max-gap-fill-frames",
            "6",
            "--measurement-variance-px",
            "1.0",
            "--process-variance",
            "0.05",
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
    assert json.loads(completed.stdout)["filled_gap_frame_count"] == 2
    assert isinstance(validate_artifact_file("ball_track", out_path), BallTrack)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["artifact_type"] == "racketsport_ball_kalman_rts_smoother"
    assert written["status"] == "TESTED-ON-REAL-DATA"
