from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack, validate_artifact_file
from threed.racketsport.wasb_adapter import (
    WASB_CONFIDENCE_SEMANTICS,
    wasb_csv_to_ball_track,
    write_ball_track_from_wasb_predictions,
)


def test_wasb_csv_to_ball_track_uses_real_heatmap_peak_confidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321.5,240.25,0.873\n"
        "1,0,0,0,0.318\n",
        encoding="utf-8",
    )

    payload = wasb_csv_to_ball_track(csv_path, fps=60.0)

    ball_track = BallTrack.model_validate(payload)
    assert ball_track.source == "wasb"
    assert [frame.t for frame in ball_track.frames] == pytest.approx([0.0, 1 / 60.0])
    assert ball_track.frames[0].xy == pytest.approx([321.5, 240.25])
    assert ball_track.frames[0].conf == pytest.approx(0.873)
    assert ball_track.frames[0].visible is True
    assert ball_track.frames[1].conf == pytest.approx(0.318)
    assert ball_track.frames[1].visible is False


def test_wasb_csv_to_ball_track_thresholds_visibility_without_binarizing_confidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321,240,0.499\n"
        "1,1,322,241,0.500\n",
        encoding="utf-8",
    )

    payload = wasb_csv_to_ball_track(csv_path, fps=60.0)

    ball_track = BallTrack.model_validate(payload)
    assert [frame.conf for frame in ball_track.frames] == pytest.approx([0.499, 0.5])
    assert [frame.visible for frame in ball_track.frames] == [False, True]


def test_wasb_csv_to_ball_track_rejects_missing_confidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,321,240\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing WASB column\\(s\\): Confidence"):
        wasb_csv_to_ball_track(csv_path, fps=60.0)


def test_write_ball_track_from_wasb_predictions_writes_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "wasb_run.json"
    csv_path.write_text("Frame,Visibility,X,Y,Confidence\n0,1,321,240,0.60\n", encoding="utf-8")

    summary = write_ball_track_from_wasb_predictions(
        predictions_csv=csv_path,
        fps=60.0,
        out=out,
        metadata_out=meta,
    )

    assert summary["frame_count"] == 1
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert metadata["status"] == "TESTED-ON-REAL-DATA"
    assert metadata["confidence_semantics"] == WASB_CONFIDENCE_SEMANTICS
    assert metadata["source_mode"] == "wasb_csv"
    assert metadata["official_repo_url"] == "https://github.com/nttcom/WASB-SBDT"


def test_run_wasb_ball_cli_converts_existing_predictions(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    out = tmp_path / "ball_track.json"
    csv_path.write_text("Frame,Visibility,X,Y,Confidence\n0,1,321,240,0.60\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_wasb_ball.py",
            "--predictions-csv",
            str(csv_path),
            "--fps",
            "60",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["source_mode"] == "wasb_csv"
    assert BallTrack.model_validate_json(out.read_text(encoding="utf-8")).source == "wasb"


def test_run_wasb_ball_cli_refuses_missing_official_runtime(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_wasb_ball.py",
            "--video",
            str(tmp_path / "clip.mp4"),
            "--checkpoint",
            str(tmp_path / "wasb.pth.tar"),
            "--wasb-repo",
            str(tmp_path / "WASB-SBDT"),
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
    assert "missing WASB-SBDT official src/models" in completed.stderr
    assert not (tmp_path / "ball_track.json").exists()
