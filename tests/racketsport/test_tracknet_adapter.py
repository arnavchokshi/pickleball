from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack, validate_artifact_file
from threed.racketsport.tracknet_adapter import tracknet_csv_to_ball_track


def test_tracknet_csv_to_ball_track_converts_official_prediction_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,321,240\n1,0,0,0\n3,1,330,250\n", encoding="utf-8")

    payload = tracknet_csv_to_ball_track(csv_path, fps=60.0)

    ball_track = BallTrack.model_validate(payload)
    assert ball_track.source == "tracknet"
    assert [frame.t for frame in ball_track.frames] == pytest.approx([0.0, 1 / 60.0, 3 / 60.0])
    assert ball_track.frames[0].xy == [321.0, 240.0]
    assert ball_track.frames[0].visible is True
    assert ball_track.frames[0].conf == pytest.approx(1.0)
    assert ball_track.frames[1].visible is False
    assert ball_track.frames[1].conf == pytest.approx(0.0)


def test_tracknet_csv_to_ball_track_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Frame,X,Y\n0,321,240\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing TrackNet column"):
        tracknet_csv_to_ball_track(csv_path, fps=60.0)


def test_run_tracknet_ball_cli_converts_existing_predictions(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "ball_track_run.json"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,321,240\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_tracknet_ball.py",
            "--predictions-csv",
            str(csv_path),
            "--fps",
            "60",
            "--out",
            str(out),
            "--metadata-out",
            str(meta),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["frame_count"] == 1
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert metadata["source_mode"] == "tracknet_csv"
    assert metadata["confidence_semantics"] == "official visibility mapped to conf 1.0/0.0"


def test_run_tracknet_ball_cli_refuses_missing_tracknet_runtime(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_tracknet_ball.py",
            "--video",
            str(tmp_path / "clip.mp4"),
            "--tracknet-file",
            str(tmp_path / "TrackNet_best.pt"),
            "--inpaintnet-file",
            str(tmp_path / "InpaintNet_best.pt"),
            "--tracknet-repo",
            str(tmp_path / "TrackNetV3"),
            "--fps",
            "60",
            "--out",
            str(tmp_path / "ball_track.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "missing TrackNetV3 predict.py" in completed.stderr
    assert not (tmp_path / "ball_track.json").exists()
