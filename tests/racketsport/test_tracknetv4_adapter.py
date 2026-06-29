from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack, validate_artifact_file
from threed.racketsport.tracknetv4_adapter import (
    run_tracknetv4_or_convert,
    tracknetv4_csv_to_ball_track,
)
from threed.racketsport import tracknetv4_adapter


def test_tracknetv4_schema_constant_is_not_public_api() -> None:
    assert "TRACKNETV4_COLUMNS" not in tracknetv4_adapter.__all__
    assert not hasattr(tracknetv4_adapter, "TRACKNETV4_COLUMNS")


def test_tracknetv4_csv_to_ball_track_accepts_official_prediction_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_predictions.csv"
    csv_path.write_text("Frame, Visibility, X, Y\n0,1,321,240\n2,0,-1,-1\n3,1,330,250\n", encoding="utf-8")

    payload = tracknetv4_csv_to_ball_track(csv_path, fps=60.0)

    ball_track = BallTrack.model_validate(payload)
    assert ball_track.source == "tracknet"
    assert [frame.t for frame in ball_track.frames] == pytest.approx([0.0, 2 / 60.0, 3 / 60.0])
    assert ball_track.frames[0].xy == [321.0, 240.0]
    assert ball_track.frames[0].visible is True
    assert ball_track.frames[0].conf == pytest.approx(1.0)
    assert ball_track.frames[1].xy == [-1.0, -1.0]
    assert ball_track.frames[1].visible is False
    assert ball_track.frames[1].conf == pytest.approx(0.0)


def test_tracknetv4_csv_to_ball_track_accepts_simple_lowercase_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "simple.csv"
    csv_path.write_text("frame,x,y\n0,12,34\n1,\n2,-1,-1\n", encoding="utf-8")

    payload = tracknetv4_csv_to_ball_track(csv_path, fps=30.0)

    ball_track = BallTrack.model_validate(payload)
    assert [frame.visible for frame in ball_track.frames] == [True, False, False]
    assert ball_track.frames[0].xy == [12.0, 34.0]
    assert ball_track.frames[1].xy == [0.0, 0.0]
    assert ball_track.frames[2].xy == [-1.0, -1.0]


def test_run_tracknetv4_or_convert_validates_runtime_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called for invalid runtime paths")

    monkeypatch.setattr("threed.racketsport.tracknetv4_adapter.subprocess.run", fail_if_called)

    with pytest.raises(FileNotFoundError, match="missing TrackNetV4 repo"):
        run_tracknetv4_or_convert(
            out=tmp_path / "ball_track.json",
            fps=60.0,
            video=tmp_path / "clip.mp4",
            checkpoint=tmp_path / "model_final.keras",
            tracknetv4_repo=tmp_path / "TrackNetV4",
        )


def test_run_tracknetv4_predict_uses_configurable_command_and_writes_unverified_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "TrackNetV4"
    src = repo / "src"
    src.mkdir(parents=True)
    predict_py = src / "predict.py"
    predict_py.write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True)
parser.add_argument("--model_weights", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--queue_length", required=True)
parser.add_argument("--tag", required=True)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "predictions.csv").write_text("Frame,Visibility,X,Y\\n0,1,10,20\\n1,0,-1,-1\\n", encoding="utf-8")
(output_dir / "args.json").write_text(json.dumps(vars(args), sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    checkpoint = tmp_path / "model_final.keras"
    checkpoint.write_bytes(b"weights")
    prediction_dir = tmp_path / "predictions"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "ball_track_run.json"

    metadata = run_tracknetv4_or_convert(
        out=out,
        fps=60.0,
        metadata_out=meta,
        video=video,
        checkpoint=checkpoint,
        tracknetv4_repo=repo,
        prediction_dir=prediction_dir,
        command=[
            "{python}",
            "{predict_py}",
            "--video_path",
            "{video}",
            "--model_weights",
            "{checkpoint}",
            "--output_dir",
            "{output_dir}",
            "--queue_length",
            "{queue_length}",
            "--tag",
            "custom",
        ],
        queue_length=9,
    )

    assert metadata["frame_count"] == 2
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
    predict_args = json.loads((prediction_dir / "args.json").read_text(encoding="utf-8"))
    assert predict_args["queue_length"] == "9"
    assert predict_args["tag"] == "custom"
    written_metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert written_metadata["not_ground_truth"] is True
    assert written_metadata["verified"] is False
    assert written_metadata["runtime"]["run_succeeded"] is True
    assert written_metadata["runtime"]["tracknetv4_checkpoint"]["path"] == str(checkpoint)


def test_run_tracknetv4_ball_cli_converts_existing_predictions_with_unverified_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_predictions.csv"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "ball_track_run.json"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,321,240\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_tracknetv4_ball.py",
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
    assert metadata["source_mode"] == "tracknetv4_csv"
    assert metadata["not_ground_truth"] is True
    assert metadata["verified"] is False
