from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.schemas import BallCandidates, BallTrack, validate_artifact_file
from threed.racketsport.tracknet_adapter import (
    _join_tracknet_confidence_csv,
    _TrackNetVideoIterableDatasetEofGuard,
    _run_nonoverlap_heatmap_confidence,
    _topk_heatmap_local_maxima,
    run_official_tracknet_predict,
    run_tracknet_or_convert,
    tracknet_csv_to_ball_track,
    write_ball_track_from_csv,
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


def test_tracknet_csv_to_ball_track_uses_heatmap_peak_confidence_when_required(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321,240,0.873\n"
        "1,0,0,0,0.318\n",
        encoding="utf-8",
    )

    payload = tracknet_csv_to_ball_track(csv_path, fps=60.0, confidence_mode="heatmap_peak")

    ball_track = BallTrack.model_validate(payload)
    assert ball_track.frames[0].visible is True
    assert ball_track.frames[0].conf == pytest.approx(0.873)
    assert ball_track.frames[1].visible is False
    assert ball_track.frames[1].conf == pytest.approx(0.318)


def test_tracknet_csv_to_ball_track_thresholds_heatmap_visibility(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321,240,0.499\n"
        "1,1,322,241,0.500\n"
        "2,1,323,242,0.501\n",
        encoding="utf-8",
    )

    payload = tracknet_csv_to_ball_track(csv_path, fps=60.0, confidence_mode="heatmap_peak")

    ball_track = BallTrack.model_validate(payload)
    assert [frame.conf for frame in ball_track.frames] == pytest.approx([0.499, 0.5, 0.501])
    assert [frame.visible for frame in ball_track.frames] == [False, True, True]


def test_join_tracknet_confidence_csv_saturates_model_heatmap_overshoot(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "clip_ball.csv"
    confidence_csv = tmp_path / "clip_ball_heatmap_confidence.csv"
    joined_csv = tmp_path / "clip_ball_with_heatmap_confidence.csv"
    predictions_csv.write_text("Frame,Visibility,X,Y\n299,1,487,442\n", encoding="utf-8")
    confidence_csv.write_text(
        "Frame,Visibility,X,Y,Confidence\n299,1,487,446,1.01583052\n",
        encoding="utf-8",
    )

    _join_tracknet_confidence_csv(
        predictions_csv=predictions_csv,
        confidence_csv=confidence_csv,
        out=joined_csv,
    )

    payload = tracknet_csv_to_ball_track(joined_csv, fps=30.0, confidence_mode="heatmap_peak")
    ball_track = BallTrack.model_validate(payload)
    assert ball_track.frames[0].conf == pytest.approx(1.0)
    assert ball_track.frames[0].visible is True


def test_join_tracknet_confidence_csv_allows_terminal_hidden_row_without_heatmap(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "clip_ball.csv"
    confidence_csv = tmp_path / "clip_ball_heatmap_confidence.csv"
    joined_csv = tmp_path / "clip_ball_with_heatmap_confidence.csv"
    predictions_csv.write_text(
        "Frame,Visibility,X,Y\n"
        "1149,1,977,226\n"
        "1150,1,1012,161\n"
        "1151,0,0,0\n",
        encoding="utf-8",
    )
    confidence_csv.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "1149,1,977,226,0.11062220\n"
        "1150,1,1012,161,0.14725174\n",
        encoding="utf-8",
    )

    _join_tracknet_confidence_csv(
        predictions_csv=predictions_csv,
        confidence_csv=confidence_csv,
        out=joined_csv,
    )

    payload = tracknet_csv_to_ball_track(joined_csv, fps=60.0, confidence_mode="heatmap_peak")
    ball_track = BallTrack.model_validate(payload)
    assert ball_track.frames[-1].visible is False
    assert ball_track.frames[-1].conf == pytest.approx(0.0)


def test_join_tracknet_confidence_csv_rejects_empty_heatmap_confidence_rows(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "clip_ball.csv"
    confidence_csv = tmp_path / "clip_ball_heatmap_confidence.csv"
    joined_csv = tmp_path / "clip_ball_with_heatmap_confidence.csv"
    predictions_csv.write_text("Frame,Visibility,X,Y\n0,1,321,240\n", encoding="utf-8")
    confidence_csv.write_text("Frame,Visibility,X,Y,Confidence\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TrackNet heatmap confidence CSV is empty"):
        _join_tracknet_confidence_csv(
            predictions_csv=predictions_csv,
            confidence_csv=confidence_csv,
            out=joined_csv,
        )


def test_tracknet_csv_to_ball_track_supports_custom_heatmap_visibility_threshold(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321,240,0.249\n"
        "1,1,322,241,0.250\n",
        encoding="utf-8",
    )

    payload = tracknet_csv_to_ball_track(
        csv_path,
        fps=60.0,
        confidence_mode="heatmap_peak",
        heatmap_visible_threshold=0.25,
    )

    ball_track = BallTrack.model_validate(payload)
    assert [frame.visible for frame in ball_track.frames] == [False, True]


def test_tracknet_csv_to_ball_track_rejects_missing_heatmap_confidence_in_strict_mode(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,321,240\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing TrackNet column.*Confidence"):
        tracknet_csv_to_ball_track(csv_path, fps=60.0, confidence_mode="heatmap_peak")


def test_tracknet_csv_to_ball_track_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Frame,X,Y\n0,321,240\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing TrackNet column"):
        tracknet_csv_to_ball_track(csv_path, fps=60.0)


def test_tracknet_heatmap_candidate_extraction_uses_source_pixel_nms_radius() -> None:
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[4, 5] = 0.91
    heatmap[5, 7] = 0.89
    heatmap[12, 13] = 0.72

    candidates = _topk_heatmap_local_maxima(heatmap, top_k=2, nms_radius_px=10.0, img_scaler=(2.0, 2.0))

    assert len(candidates) == 2
    assert candidates[0]["xy"] == [10.0, 8.0]
    assert candidates[0]["score"] == pytest.approx(0.91)
    assert candidates[1]["xy"] == [26.0, 24.0]
    assert candidates[1]["score"] == pytest.approx(0.72)
    assert all(candidate["source_detector"] == "tracknet_heatmap_nms" for candidate in candidates)


def test_write_tracknet_candidate_sidecar_is_opt_in_and_preserves_primary_track_bytes(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y,Confidence\n"
        "0,1,321,240,0.60\n"
        "1,0,0,0,0.20\n",
        encoding="utf-8",
    )
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_out = off_dir / "ball_track.json"
    on_out = on_dir / "ball_track.json"

    off_summary = write_ball_track_from_csv(
        predictions_csv=csv_path,
        fps=60.0,
        out=off_out,
        confidence_mode="heatmap_peak",
    )
    on_summary = write_ball_track_from_csv(
        predictions_csv=csv_path,
        fps=60.0,
        out=on_out,
        confidence_mode="heatmap_peak",
        emit_candidates=True,
        candidate_top_k=1,
        candidate_frames={
            0: [
                {"xy": [111.0, 222.0], "score": 0.55},
                {"xy": [321.0, 240.0], "score": 0.95},
            ],
            1: [],
        },
    )

    assert off_out.read_bytes() == on_out.read_bytes()
    assert "candidates_out" not in off_summary
    assert not (off_dir / "ball_candidates.json").exists()
    assert on_summary["candidates_out"] == str(on_dir / "ball_candidates.json")
    sidecar = validate_artifact_file("racketsport_ball_candidates", on_dir / "ball_candidates.json")
    assert isinstance(sidecar, BallCandidates)
    assert sidecar.source == "tracknet"
    assert sidecar.frames[0].candidates[0].xy == [321.0, 240.0]
    assert sidecar.frames[0].candidates[0].score == pytest.approx(0.95)
    assert sidecar.frames[0].candidates[0].source_detector == "tracknet_heatmap_nms"


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


def test_run_tracknet_ball_cli_heatmap_mode_requires_confidence_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    out = tmp_path / "ball_track.json"
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
            "--confidence-mode",
            "heatmap_peak",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "missing TrackNet column(s): Confidence" in completed.stderr
    assert not out.exists()


def test_run_tracknet_ball_cli_writes_heatmap_visibility_threshold_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_ball.csv"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "ball_track_run.json"
    csv_path.write_text("Frame,Visibility,X,Y,Confidence\n0,1,321,240,0.60\n", encoding="utf-8")

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
            "--confidence-mode",
            "heatmap_peak",
            "--heatmap-visible-threshold",
            "0.75",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["visible_frame_count"] == 0
    ball_track = BallTrack.model_validate_json(out.read_text(encoding="utf-8"))
    assert ball_track.frames[0].conf == pytest.approx(0.60)
    assert ball_track.frames[0].visible is False
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert metadata["heatmap_visible_threshold"] == pytest.approx(0.75)


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


def test_run_tracknet_video_heatmap_mode_joins_persisted_confidence_csv(
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

    observed_predict_kwargs = {}

    def fake_predict(*, save_dir, **kwargs):
        observed_predict_kwargs.update(kwargs)
        csv_path = Path(save_dir) / "clip_ball.csv"
        csv_path.write_text("Frame,Visibility,X,Y\n0,1,10,20\n1,0,0,0\n", encoding="utf-8")
        return csv_path

    observed_confidence_kwargs = {}

    def fake_confidence_predict(*, save_dir, **kwargs):
        observed_confidence_kwargs.update(kwargs)
        csv_path = Path(save_dir) / "clip_ball_heatmap_confidence.csv"
        csv_path.write_text(
            "Frame,Visibility,X,Y,Confidence\n0,1,10,20,0.8125\n1,0,0,0,0.247\n",
            encoding="utf-8",
        )
        return csv_path

    monkeypatch.setattr("threed.racketsport.tracknet_adapter.run_official_tracknet_predict", fake_predict)
    monkeypatch.setattr(
        "threed.racketsport.tracknet_adapter.run_tracknet_heatmap_confidence_predict",
        fake_confidence_predict,
    )

    metadata = run_tracknet_or_convert(
        out=tmp_path / "ball_track.json",
        fps=30.0,
        metadata_out=tmp_path / "ball_track_run.json",
        video=video,
        tracknet_file=tracknet,
        inpaintnet_file=inpaintnet,
        tracknet_repo=repo,
        prediction_dir=None,
        confidence_mode="heatmap_peak",
        heatmap_eval_mode="nonoverlap",
        heatmap_large_video=True,
    )

    predictions_csv = Path(metadata["predictions_csv"])
    assert predictions_csv.is_file()
    assert "Confidence" in predictions_csv.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads((tmp_path / "ball_track.json").read_text(encoding="utf-8"))
    assert payload["frames"][0]["conf"] == pytest.approx(0.8125)
    assert payload["frames"][1]["conf"] == pytest.approx(0.247)
    assert metadata["confidence_mode"] == "heatmap_peak"
    assert metadata["confidence_semantics"] == "TrackNet heatmap peak value (0..1)"
    assert metadata["heatmap_visible_threshold"] == pytest.approx(0.5)
    assert metadata["runtime"]["heatmap_eval_mode"] == "nonoverlap"
    assert metadata["runtime"]["heatmap_large_video"] is True
    assert observed_predict_kwargs["large_video"] is False
    assert observed_confidence_kwargs["large_video"] is True
    assert observed_confidence_kwargs["eval_mode"] == "nonoverlap"


def test_nonoverlap_heatmap_confidence_ignores_tracknet_exact_end_large_video_bug() -> None:
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    class FakeX:
        def float(self):
            return self

        def to(self, _device):
            return self

    class ExactEndBugLoader:
        def __iter__(self):
            yield np.array([[[0, 0], [0, 1]]]), FakeX()
            raise IndexError("list index out of range")

    def fake_tracknet(_x):
        heatmaps = np.zeros((1, 2, 4, 4), dtype=np.float32)
        heatmaps[0, 0, 1, 2] = 0.75
        heatmaps[0, 1, 2, 3] = 0.8
        return heatmaps

    rows = _run_nonoverlap_heatmap_confidence(
        data_loader=ExactEndBugLoader(),
        tracknet=fake_tracknet,
        torch=FakeTorch,
        device="cuda",
        img_scaler=(10.0, 20.0),
        expected_frame_count=2,
    )

    assert [row["Frame"] for row in rows] == [0, 1]
    assert [row["Confidence"] for row in rows] == pytest.approx([0.75, 0.8])


def test_tracknet_video_iterable_eof_guard_preserves_partial_dataloader_batch() -> None:
    class BuggyExactEndDataset:
        def __iter__(self):
            yield 0
            yield 1
            raise IndexError("list index out of range")

    guarded = _TrackNetVideoIterableDatasetEofGuard(BuggyExactEndDataset())
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError:
        assert list(guarded) == [0, 1]
        return

    batches = list(DataLoader(guarded, batch_size=16))

    assert len(batches) == 1
    assert batches[0].tolist() == [0, 1]


def test_tracknet_video_iterable_eof_guard_handles_normal_stop_iteration() -> None:
    class NormalDataset:
        def __iter__(self):
            yield 0
            yield 1

    assert list(_TrackNetVideoIterableDatasetEofGuard(NormalDataset())) == [0, 1]
