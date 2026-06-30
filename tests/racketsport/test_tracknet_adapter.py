from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack, validate_artifact_file
from threed.racketsport.tracknet_adapter import (
    run_official_tracknet_predict,
    run_tracknet_or_convert,
    tracknet_csv_to_ball_track,
)


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


def test_run_tracknet_ball_cli_passes_video_range_to_official_predict(tmp_path: Path) -> None:
    repo = tmp_path / "TrackNetV3"
    repo.mkdir()
    predict_py = repo / "predict.py"
    predict_py.write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video_file", required=True)
parser.add_argument("--tracknet_file", required=True)
parser.add_argument("--inpaintnet_file", required=True)
parser.add_argument("--save_dir", required=True)
parser.add_argument("--batch_size", required=True)
parser.add_argument("--large_video", action="store_true")
parser.add_argument("--video_range")
args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
video_stem = Path(args.video_file).stem
(save_dir / f"{video_stem}_ball.csv").write_text("Frame,Visibility,X,Y\\n0,1,10,20\\n", encoding="utf-8")
(save_dir / "args.json").write_text(json.dumps(vars(args), sort_keys=True), encoding="utf-8")
""",
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    tracknet = tmp_path / "TrackNet_best.pt"
    inpaintnet = tmp_path / "InpaintNet_best.pt"
    tracknet.write_bytes(b"tracknet")
    inpaintnet.write_bytes(b"inpaintnet")
    prediction_dir = tmp_path / "predictions"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "ball_track_run.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_tracknet_ball.py",
            "--video",
            str(video),
            "--tracknet-file",
            str(tracknet),
            "--inpaintnet-file",
            str(inpaintnet),
            "--tracknet-repo",
            str(repo),
            "--prediction-dir",
            str(prediction_dir),
            "--video-range",
            "10",
            "20",
            "--fps",
            "60",
            "--out",
            str(out),
            "--metadata-out",
            str(meta),
            "--large-video",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["frame_count"] == 1
    predict_args = json.loads((prediction_dir / "args.json").read_text(encoding="utf-8"))
    assert predict_args["video_range"] == "10,20"
    assert predict_args["large_video"] is True
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert metadata["runtime"]["video_range_seconds"] == [10, 20]
    assert metadata["runtime"]["video_range_semantics"] == (
        "official TrackNetV3 background median sampling range; does not trim prediction frames"
    )
    assert metadata["runtime"]["batch_size"] == 16
    assert metadata["runtime"]["large_video"] is True
    assert metadata["runtime"]["processed_frame_count"] == 1
    assert metadata["runtime"]["effective_fps"] > 0.0
    assert metadata["runtime"]["realtime_factor"] > 0.0
    assert metadata["runtime"]["wall_seconds"] > 0.0


def test_run_official_tracknet_predict_patches_cuda_only_predict_in_temp_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "TrackNetV3"
    repo.mkdir()
    source_predict = """
import torch
tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
inpaintnet = get_model('InpaintNet').cuda()
x = x.float().cuda()
coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
"""
    (repo / "predict.py").write_text(source_predict, encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    tracknet = tmp_path / "TrackNet_best.pt"
    inpaintnet = tmp_path / "InpaintNet_best.pt"
    tracknet.write_bytes(b"tracknet")
    inpaintnet.write_bytes(b"inpaintnet")
    save_dir = tmp_path / "predictions"
    observed: dict[str, Path] = {}

    def fake_subprocess_run(cmd, *, cwd, check):
        observed["cwd"] = Path(cwd)
        observed["predict_py"] = Path(cmd[1])
        patched_predict = observed["predict_py"].read_text(encoding="utf-8")
        assert check is True
        assert observed["cwd"] != repo
        assert observed["predict_py"].parent == observed["cwd"]
        assert "DEVICE = torch.device" in patched_predict
        assert ".cuda()" not in patched_predict
        assert ".cuda(" not in patched_predict
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "clip_ball.csv").write_text("Frame,Visibility,X,Y\n0,1,10,20\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("threed.racketsport.tracknet_adapter.subprocess.run", fake_subprocess_run)

    csv_path = run_official_tracknet_predict(
        tracknet_repo=repo,
        video=video,
        tracknet_file=tracknet,
        inpaintnet_file=inpaintnet,
        save_dir=save_dir,
        batch_size=1,
    )

    assert csv_path == save_dir / "clip_ball.csv"
    assert (repo / "predict.py").read_text(encoding="utf-8") == source_predict


def test_run_tracknet_temp_prediction_metadata_does_not_reference_deleted_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "TrackNetV3"
    repo.mkdir()
    (repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    tracknet = tmp_path / "TrackNet_best.pt"
    inpaintnet = tmp_path / "InpaintNet_best.pt"
    tracknet.write_bytes(b"tracknet")
    inpaintnet.write_bytes(b"inpaintnet")

    def fake_predict(*, save_dir, **_kwargs):
        csv_path = Path(save_dir) / "clip_ball.csv"
        csv_path.write_text("Frame,Visibility,X,Y\n0,1,10,20\n", encoding="utf-8")
        return csv_path

    monkeypatch.setattr("threed.racketsport.tracknet_adapter.run_official_tracknet_predict", fake_predict)

    metadata = run_tracknet_or_convert(
        out=tmp_path / "ball_track.json",
        fps=30.0,
        metadata_out=tmp_path / "ball_track_run.json",
        video=video,
        tracknet_file=tracknet,
        inpaintnet_file=inpaintnet,
        tracknet_repo=repo,
        prediction_dir=None,
    )

    predictions_csv = Path(metadata["predictions_csv"])
    assert predictions_csv.is_file()
    assert predictions_csv.parent == tmp_path
