from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from threed.racketsport.ball_size_observations import (
    WasbBallSizeObservations,
    connected_component_blob_extents,
    load_wasb_ball_size_observations,
)
from threed.racketsport.schemas import BallCandidates, BallTrack, validate_artifact_file
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
    assert metadata["input_preprocessing"] == "official"
    assert metadata["official_repo_url"] == "https://github.com/nttcom/WASB-SBDT"


def test_write_wasb_candidate_sidecar_is_opt_in_and_preserves_primary_track_bytes(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
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

    off_summary = write_ball_track_from_wasb_predictions(predictions_csv=csv_path, fps=60.0, out=off_out)
    on_summary = write_ball_track_from_wasb_predictions(
        predictions_csv=csv_path,
        fps=60.0,
        out=on_out,
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
    sidecar_payload = json.loads((on_dir / "ball_candidates.json").read_text(encoding="utf-8"))
    assert sidecar_payload["input_preprocessing"] == "official"
    assert sidecar_payload["provenance"]["input_preprocessing"] == "official"
    sidecar = validate_artifact_file("racketsport_ball_candidates", on_dir / "ball_candidates.json")
    assert isinstance(sidecar, BallCandidates)
    assert sidecar.source == "wasb"
    assert sidecar.frames[0].candidates[0].xy == [321.0, 240.0]
    assert sidecar.frames[0].candidates[0].score == pytest.approx(0.95)
    assert sidecar.frames[0].candidates[0].source_detector == "wasb_concomp"


def test_wasb_connected_component_extents_capture_every_raw_blob_before_topk() -> None:
    cv2 = pytest.importorskip("cv2")
    from threed.racketsport import wasb_adapter

    heatmap = np.zeros((6, 8), dtype=np.float32)
    heatmap[1:3, 1:4] = np.asarray([[0.51, 0.7, 0.51], [0.6, 0.9, 0.6]])
    heatmap[4, 6] = 0.8
    affine = np.asarray([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0]], dtype=np.float32)

    blobs = connected_component_blob_extents(heatmap, affine, cv2=cv2, np=np)

    assert len(blobs) == 2
    assert blobs[0]["heatmap_peak"] == pytest.approx(0.9)
    assert blobs[0]["width_px"] == pytest.approx(6.0)
    assert blobs[0]["height_px"] == pytest.approx(6.0)
    assert blobs[0]["component_pixel_count"] == 6
    assert blobs[0]["component_area_px2"] == pytest.approx(36.0)
    assert blobs[0]["radius_proxy_px"] == pytest.approx(3.0)
    assert blobs[1]["width_px"] == pytest.approx(2.0)
    assert blobs[1]["height_px"] == pytest.approx(3.0)
    assert len(wasb_adapter._topk_candidate_blobs(
        [{"xy": blob["center_xy_px"], "score": blob["heatmap_peak"]} for blob in blobs],
        top_k=1,
        default_source_detector="wasb_concomp",
    )) == 1
    assert len(blobs) == 2


def test_default_on_wasb_size_sidecar_is_pts_aligned_deterministic_and_track_byte_identical(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from threed.racketsport.wasb_adapter import run_wasb_or_convert

    fake_repo = _write_fake_wasb_repo(tmp_path / "WASB-SBDT")
    video = tmp_path / "moving_ball.mp4"
    _write_moving_dot_video(video, cv2=cv2)
    checkpoint = tmp_path / "lane_latest.pt"
    torch.save({"state_dict": {}, "run_dir": Path("runs/lanes/ball_sizeobs_20260712")}, checkpoint)
    frame_times = {"frames": [{"frame": 0, "pts_s": 0.001}, {"frame": 1, "pts_s": 0.041}, {"frame": 2, "pts_s": 0.082}]}

    off_out = tmp_path / "off" / "ball_track.json"
    on_out = tmp_path / "on" / "ball_track.json"
    run_wasb_or_convert(
        out=off_out,
        fps=30.0,
        frame_times=frame_times,
        video=video,
        checkpoint=checkpoint,
        wasb_repo=fake_repo,
        batch_size=1,
        max_frames=3,
        device="cpu",
        emit_size_observations=False,
        input_preprocessing="harness_v0",
    )
    on_summary = run_wasb_or_convert(
        out=on_out,
        fps=30.0,
        frame_times=frame_times,
        video=video,
        checkpoint=checkpoint,
        wasb_repo=fake_repo,
        batch_size=1,
        max_frames=3,
        device="cpu",
        input_preprocessing="harness_v0",
    )

    assert off_out.read_bytes() == on_out.read_bytes()
    assert (off_out.parent / "ball_track_wasb_predictions.csv").read_bytes() == (
        on_out.parent / "ball_track_wasb_predictions.csv"
    ).read_bytes()
    assert not (off_out.parent / "ball_size_observations.json").exists()
    sidecar_path = on_out.parent / "ball_size_observations.json"
    assert on_summary["runtime"]["size_observations_out"] == str(sidecar_path)
    first_bytes = sidecar_path.read_bytes()
    sidecar = load_wasb_ball_size_observations(sidecar_path)
    assert isinstance(sidecar, WasbBallSizeObservations)
    assert [frame.pts_seconds for frame in sidecar.frames] == pytest.approx([0.001, 0.041, 0.082])
    assert [frame.frame for frame in sidecar.frames] == [0, 1, 2]
    assert all(frame.blob_count == len(frame.blobs) for frame in sidecar.frames)
    assert all(frame.blob_count == 1 for frame in sidecar.frames)
    assert sidecar.radius_proxy_definition.startswith("0.5 * sqrt")

    run_wasb_or_convert(
        out=on_out,
        fps=30.0,
        frame_times=frame_times,
        video=video,
        checkpoint=checkpoint,
        wasb_repo=fake_repo,
        batch_size=1,
        max_frames=3,
        device="cpu",
        input_preprocessing="harness_v0",
    )
    assert sidecar_path.read_bytes() == first_bytes


def test_wasb_official_preprocessing_mode_is_legacy_bit_identical() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from threed.racketsport import wasb_adapter

    base = np.arange(17 * 19 * 3, dtype=np.uint16).reshape(17, 19, 3)
    frames = [((base + i * 37) % 256).astype(np.uint8) for i in range(3)]
    trans_input = np.array([[1.7, 0.13, -2.4], [-0.08, 1.51, 3.2]], dtype=np.float32)

    tensor = wasb_adapter._preprocess_wasb_window(
        frames,
        trans_input,
        cv2=cv2,
        np=np,
        torch=torch,
        input_preprocessing="official",
    )

    assert tuple(tensor.shape) == (9, 288, 512)
    digest = __import__("hashlib").sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
    assert digest == "cf085a40e6e99579a6ad05c08a258eb2a427e7b721fe1e4186dfc04953a19fa4"


def test_wasb_harness_v0_preprocessing_matches_roboflow_image_tensor(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from threed.racketsport import wasb_adapter
    from threed.racketsport import roboflow_corpus

    frame = (np.arange(13 * 11 * 3, dtype=np.uint16).reshape(13, 11, 3) % 256).astype(np.uint8)
    image_path = tmp_path / "frame.png"
    Image.fromarray(frame).save(image_path)

    tensor = wasb_adapter._preprocess_wasb_window(
        [frame, frame, frame],
        np.eye(2, 3, dtype=np.float32),
        cv2=cv2,
        np=np,
        torch=torch,
        input_preprocessing="harness_v0",
    )
    expected = roboflow_corpus._image_tensor(image_path, image_size=(512, 288))

    assert torch.equal(tensor[:3], expected)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_wasb_checkpoint_loader_accepts_lane_posixpath_payload_and_official_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from threed.racketsport import wasb_adapter

    lane_checkpoint = tmp_path / "lane_latest.pt"
    torch.save(
        {
            "state_dict": {"layer.weight": torch.tensor([1.0, 2.0])},
            "run_dir": Path("runs/lanes/w4_ballgpu_20260707"),
        },
        lane_checkpoint,
    )

    payload = wasb_adapter._load_wasb_checkpoint_payload(lane_checkpoint, torch=torch)
    state_dict = wasb_adapter._checkpoint_state_dict(payload)

    assert payload["run_dir"] == Path("runs/lanes/w4_ballgpu_20260707")
    assert torch.equal(state_dict["layer.weight"], torch.tensor([1.0, 2.0]))

    official_checkpoint = Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar")
    if official_checkpoint.is_file():
        official_payload = wasb_adapter._load_wasb_checkpoint_payload(official_checkpoint, torch=torch)
        assert "model_state_dict" in official_payload


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
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["input_preprocessing"] == "official"
    assert BallTrack.model_validate(payload).source == "wasb"


def test_wasb_outputs_stamp_input_preprocessing_on_track_metadata_and_candidates(tmp_path: Path) -> None:
    csv_path = tmp_path / "clip_wasb.csv"
    out = tmp_path / "ball_track.json"
    csv_path.write_text("Frame,Visibility,X,Y,Confidence\n0,1,321,240,0.60\n", encoding="utf-8")

    summary = write_ball_track_from_wasb_predictions(
        predictions_csv=csv_path,
        fps=60.0,
        out=out,
        input_preprocessing="harness_v0",
        emit_candidates=True,
        candidate_frames={0: [{"xy": [321.0, 240.0], "score": 0.95}]},
    )

    track_payload = json.loads(out.read_text(encoding="utf-8"))
    assert track_payload["input_preprocessing"] == "harness_v0"
    assert BallTrack.model_validate(track_payload).source == "wasb"
    candidates_payload = json.loads((tmp_path / "ball_candidates.json").read_text(encoding="utf-8"))
    assert candidates_payload["input_preprocessing"] == "harness_v0"
    assert candidates_payload["provenance"]["input_preprocessing"] == "harness_v0"
    assert summary["input_preprocessing"] == "harness_v0"
    assert summary["non_promotable_measurement_mode"] is True


def test_wasb_harness_v0_checkpoint_inference_writes_non_degenerate_stamped_artifacts(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from threed.racketsport.wasb_adapter import run_wasb_or_convert

    fake_repo = _write_fake_wasb_repo(tmp_path / "WASB-SBDT")
    video = tmp_path / "moving_ball.mp4"
    _write_moving_dot_video(video, cv2=cv2)
    checkpoint = tmp_path / "lane_latest.pt"
    torch.save({"state_dict": {}, "run_dir": Path("runs/lanes/w4_ballgpu_20260707")}, checkpoint)

    out = tmp_path / "ball_track.json"
    metadata = tmp_path / "wasb_run.json"
    summary = run_wasb_or_convert(
        out=out,
        fps=30.0,
        metadata_out=metadata,
        video=video,
        checkpoint=checkpoint,
        wasb_repo=fake_repo,
        batch_size=1,
        max_frames=3,
        device="cpu",
        emit_candidates=True,
        candidate_top_k=3,
        input_preprocessing="harness_v0",
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    visible_xy = [tuple(frame["xy"]) for frame in payload["frames"] if frame["visible"]]
    assert payload["input_preprocessing"] == "harness_v0"
    assert len(set(visible_xy)) > 1
    assert summary["input_preprocessing"] == "harness_v0"
    assert summary["non_promotable_measurement_mode"] is True
    assert json.loads(metadata.read_text(encoding="utf-8"))["input_preprocessing"] == "harness_v0"
    candidates_payload = json.loads((tmp_path / "ball_candidates.json").read_text(encoding="utf-8"))
    assert candidates_payload["input_preprocessing"] == "harness_v0"
    assert candidates_payload["provenance"]["input_preprocessing"] == "harness_v0"


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


def _write_moving_dot_video(path: Path, *, cv2: object) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 18))
    assert writer.isOpened()
    try:
        for index, x in enumerate((5, 13, 24)):
            frame = np.zeros((18, 32, 3), dtype=np.uint8)
            frame[4 + index, x] = [255, 255, 255]
            writer.write(frame)
    finally:
        writer.release()


def _write_fake_wasb_repo(root: Path) -> Path:
    src = root / "src"
    for relative in ("models", "detectors", "trackers", "utils"):
        package_dir = src / relative
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (src / "models" / "__init__.py").write_text(
        """
import torch


class TinyModel(torch.nn.Module):
    def forward(self, batch):
        batch_size, _channels, height, width = batch.shape
        logits = torch.full((batch_size, 3, height, width), -20.0, dtype=batch.dtype, device=batch.device)
        for batch_index in range(batch_size):
            for output_index in range(3):
                channel = batch[batch_index, output_index * 3]
                flat_index = torch.argmax(channel)
                y = flat_index // width
                x = flat_index % width
                logits[batch_index, output_index, y, x] = 20.0
        return [logits]


def build_model(cfg):
    return TinyModel()
""".lstrip(),
        encoding="utf-8",
    )
    (src / "detectors" / "postprocessor.py").write_text(
        """
import torch


class TracknetV2Postprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, logits_by_scale, affine_by_scale):
        logits = logits_by_scale[0].detach()
        affine = affine_by_scale[0]
        results = []
        batch_size, frames_out, height, width = logits.shape
        for batch_index in range(batch_size):
            window = []
            for output_index in range(frames_out):
                flat_index = torch.argmax(logits[batch_index, output_index])
                y = float(flat_index // width)
                x = float(flat_index % width)
                matrix = affine[batch_index]
                xy = [
                    float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
                    float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
                ]
                window.append([{"xys": [xy], "scores": [1.0]}])
            results.append(window)
        return results
""".lstrip(),
        encoding="utf-8",
    )
    (src / "trackers" / "online.py").write_text(
        """
class OnlineTracker:
    def __init__(self, cfg):
        self.cfg = cfg

    def refresh(self):
        pass

    def update(self, detections):
        if not detections:
            return {"x": 0.0, "y": 0.0, "visi": False}
        xy = detections[0]["xy"]
        return {"x": float(xy[0]), "y": float(xy[1]), "visi": True}
""".lstrip(),
        encoding="utf-8",
    )
    (src / "utils" / "image.py").write_text(
        """
import cv2


def get_affine_transform(center, scale, rot, output_size, inv=0):
    src = (
        (float(center[0]) - float(scale) / 2.0, float(center[1]) - float(scale) / 2.0),
        (float(center[0]) + float(scale) / 2.0, float(center[1]) - float(scale) / 2.0),
        (float(center[0]) - float(scale) / 2.0, float(center[1]) + float(scale) / 2.0),
    )
    dst = ((0.0, 0.0), (float(output_size[0]), 0.0), (0.0, float(output_size[1])))
    if inv:
        return cv2.getAffineTransform(__import__("numpy").float32(dst), __import__("numpy").float32(src))
    return cv2.getAffineTransform(__import__("numpy").float32(src), __import__("numpy").float32(dst))
""".lstrip(),
        encoding="utf-8",
    )
    return root
