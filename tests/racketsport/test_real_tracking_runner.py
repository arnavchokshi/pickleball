from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.orchestrator import (
    RealYOLO26BoTSORTReIDTrackingRunner,
    StageContext,
    run_pipeline,
)
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    Tracks,
    validate_artifact_file,
)


def _write_manifest(path: Path, checkpoint: Path, *, sha256: str | None = None) -> None:
    digest = sha256 or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "yolo26m",
                        "stage": "person_detect",
                        "use": "Offline person detector candidate",
                        "source": "https://docs.ultralytics.com/models/yolo26/",
                        "license": "AGPL-3.0",
                        "commercial_posture": "agpl_caveat",
                        "status": "available_on_h100",
                        "local_path": str(checkpoint),
                        "sha256": digest,
                        "fallbacks": ["yolo11m"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_calibration(path: Path) -> None:
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )
    path.write_text(calibration.model_dump_json(), encoding="utf-8")


def _write_sidecar(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device_tier": "B_standard",
                "device_model": "iPhone16,2",
                "fps": 30,
                "format": "hevc",
                "resolution": [1920, 1080],
                "orientation": "landscape",
                "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
                "arkit_camera_pose": None,
                "court_plane": None,
                "manual_court_taps": [],
                "gravity": [0.0, -1.0, 0.0],
                "lidar_depth_refs": [],
                "ondevice_pose_track": None,
                "capture_quality": {"grade": "good", "reasons": []},
            }
        ),
        encoding="utf-8",
    )


class _Scalar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def item(self) -> float | int:
        return self.value


class _XYXY:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> "_XYXY":
        assert index == 0
        return self

    def cpu(self) -> "_XYXY":
        return self

    def tolist(self) -> list[float]:
        return self.values


class _FakeBox:
    def __init__(self, *, track_id: int, xyxy: list[float], conf: float = 0.9, cls: int = 0) -> None:
        self.id = _Scalar(track_id)
        self.xyxy = _XYXY(xyxy)
        self.conf = _Scalar(conf)
        self.cls = _Scalar(cls)


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox], *, orig_shape: tuple[int, int] | None = (1080, 1920)) -> None:
        self.boxes = boxes
        if orig_shape is not None:
            self.orig_shape = orig_shape


def test_real_tracking_runner_invokes_manifest_yolo26m_with_botsort_reid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"registry yolo26m checkpoint")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint)

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"not decoded by fake ultralytics")
    (inputs / "detections.json").write_text("{ this would fail if the real runner read it", encoding="utf-8")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")

    calls: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            calls["model_path"] = model_path

        def track(self, *args: object, **kwargs: object) -> list[_FakeResult]:
            calls["track_args"] = args
            calls["track_kwargs"] = kwargs
            return [
                _FakeResult([_FakeBox(track_id=9, xyxy=[1080.0, 900.0, 1120.0, 1000.0], conf=0.91)]),
                _FakeResult([_FakeBox(track_id=9, xyxy=[1084.0, 900.0, 1124.0, 1000.0], conf=0.89)]),
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    runner = RealYOLO26BoTSORTReIDTrackingRunner(manifest_path=manifest)
    result = runner.run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            device="0",
            max_frames=2,
        )
    )

    assert result.real_model is True
    assert result.source_mode == "yolo26m_botsort_reid"
    assert result.produced_artifacts == (
        "raw_tracked_detections.json",
        "tracked_detections.json",
        "metrics.json",
        "tracks.json",
    )
    assert result.metrics["checkpoint_sha256_verified"] == checkpoint.name
    assert calls["model_path"] == str(checkpoint)
    track_kwargs = calls["track_kwargs"]
    assert isinstance(track_kwargs, dict)
    assert track_kwargs["source"] == str(inputs / "source.mp4")
    assert track_kwargs["classes"] == [0]
    assert track_kwargs["conf"] == 0.05
    assert track_kwargs["imgsz"] == 1536
    assert track_kwargs["stream"] is True
    assert track_kwargs["persist"] is False
    tracker_path = Path(str(track_kwargs["tracker"]))
    tracker_text = tracker_path.read_text(encoding="utf-8")
    assert "tracker_type: botsort" in tracker_text
    assert "with_reid: True" in tracker_text
    assert "model: auto" in tracker_text

    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert [player.id for player in tracks.players] == [9]
    assert [frame.t for frame in tracks.players[0].frames] == [0.0, 1.0 / 30.0]
    assert tracks.players[0].frames[0].world_xy == [1.0, 0.0]

    raw_pool = json.loads((run_dir / "raw_tracked_detections.json").read_text(encoding="utf-8"))
    scaled_pool = json.loads((run_dir / "tracked_detections.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert raw_pool["frames"][0]["detections"][0]["bbox"] == [1080.0, 900.0, 1120.0, 1000.0]
    assert scaled_pool["frames"][0]["detections"][0]["bbox"] == [1080.0, 900.0, 1120.0, 1000.0]
    assert metrics["artifact_type"] == "racketsport_person_tracker_candidate"
    assert metrics["counts"]["source_width"] == 1920
    assert metrics["counts"]["calibration_width"] == 1920


def test_real_tracking_runner_scales_result_bboxes_to_calibration_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"registry yolo26m checkpoint")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint)

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"not decoded by fake ultralytics")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            assert model_path == str(checkpoint)

        def track(self, *args: object, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult([_FakeBox(track_id=9, xyxy=[540.0, 450.0, 560.0, 500.0])], orig_shape=(540, 960))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    runner = RealYOLO26BoTSORTReIDTrackingRunner(manifest_path=manifest)
    result = runner.run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            max_frames=1,
        )
    )

    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert tracks.players[0].frames[0].bbox == (1080.0, 900.0, 1120.0, 1000.0)
    assert tracks.players[0].frames[0].world_xy == [1.0, 0.0]
    assert result.metrics["bbox_scale_x"] == 2.0
    assert result.metrics["bbox_scale_y"] == 2.0
    assert result.metrics["bbox_scale_status"] == "scaled"
    assert result.metrics["tracker_source_width"] == 960
    assert result.metrics["tracker_source_height"] == 540


def test_real_tracking_runner_fails_closed_without_source_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"registry yolo26m checkpoint")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint)

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"not decoded by fake ultralytics")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            assert model_path == str(checkpoint)

        def track(self, *args: object, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult([_FakeBox(track_id=9, xyxy=[540.0, 450.0, 560.0, 500.0])], orig_shape=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    runner = RealYOLO26BoTSORTReIDTrackingRunner(manifest_path=manifest)
    with pytest.raises(ValueError, match="source video dimensions are unavailable"):
        runner.run(
            StageContext(
                clip="clip_001",
                inputs_dir=inputs,
                run_dir=run_dir,
                sport="pickleball",
                max_frames=1,
            )
        )

    assert not (run_dir / "tracks.json").exists()


def test_real_tracking_runner_caps_tracked_people_after_botsort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"registry yolo26m checkpoint")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint)

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"not decoded by fake ultralytics")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            assert model_path == str(checkpoint)

        def track(self, *args: object, **kwargs: object) -> list[_FakeResult]:
            return [
                _FakeResult(
                    [
                        _FakeBox(track_id=1, xyxy=[1080.0, 900.0, 1120.0, 1000.0], conf=0.91),
                        _FakeBox(track_id=2, xyxy=[1180.0, 900.0, 1220.0, 1000.0], conf=0.90),
                        _FakeBox(track_id=3, xyxy=[1080.0, 1120.0, 1120.0, 1400.0], conf=0.89),
                        _FakeBox(track_id=4, xyxy=[1180.0, 1120.0, 1220.0, 1400.0], conf=0.88),
                        _FakeBox(track_id=99, xyxy=[980.0, 720.0, 1020.0, 1000.0], conf=0.99),
                    ]
                ),
                _FakeResult(
                    [
                        _FakeBox(track_id=1, xyxy=[1084.0, 900.0, 1124.0, 1000.0], conf=0.91),
                        _FakeBox(track_id=2, xyxy=[1184.0, 900.0, 1224.0, 1000.0], conf=0.90),
                        _FakeBox(track_id=3, xyxy=[1084.0, 1120.0, 1124.0, 1400.0], conf=0.89),
                        _FakeBox(track_id=4, xyxy=[1184.0, 1120.0, 1224.0, 1400.0], conf=0.88),
                    ]
                ),
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    runner = RealYOLO26BoTSORTReIDTrackingRunner(manifest_path=manifest, max_players=4)
    result = runner.run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            max_frames=2,
        )
    )

    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert [player.id for player in tracks.players] == [1, 2, 3, 4]
    assert result.metrics["extra_players_dropped"] == 1
    assert result.metrics["max_players"] == 4


def test_real_tracking_runner_fails_before_model_load_on_sha_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"actual checkpoint bytes")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint, sha256="0" * 64)
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"video")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")

    class ShouldNotLoadYOLO:
        def __init__(self, model_path: str) -> None:
            raise AssertionError(f"YOLO loaded before sha verification: {model_path}")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=ShouldNotLoadYOLO))

    runner = RealYOLO26BoTSORTReIDTrackingRunner(manifest_path=manifest)
    with pytest.raises(ValueError, match="sha256 mismatch.*yolo26m"):
        runner.run(
            StageContext(
                clip="clip_001",
                inputs_dir=inputs,
                run_dir=run_dir,
                sport="pickleball",
            )
        )


@pytest.mark.h100
@pytest.mark.integration
def test_h100_real_tracking_runner_smoke_from_env(tmp_path: Path) -> None:
    if os.environ.get("RUN_H100_YOLO26_TRACKING") != "1":
        pytest.skip("set RUN_H100_YOLO26_TRACKING=1 plus TRK1_H100_INPUTS/TRK1_H100_VIDEO to run the real H100 smoke")
    inputs = Path(os.environ["TRK1_H100_INPUTS"])
    video = Path(os.environ["TRK1_H100_VIDEO"])
    manifest = Path(os.environ.get("TRK1_H100_MANIFEST", "models/MANIFEST.json"))
    summary = run_pipeline(
        clip=os.environ.get("TRK1_H100_CLIP", inputs.name),
        inputs_dir=inputs,
        run_dir=tmp_path / "trk1_h100_real",
        stage="tracking",
        tracking_mode="real",
        tracking_video=video,
        manifest_path=manifest,
        max_frames=int(os.environ.get("TRK1_H100_MAX_FRAMES", "90")),
    )

    assert summary["status"] == "pass"
    assert summary["stages"][-1]["stage"] == "tracking"
    assert summary["stages"][-1]["real_model"] is True
    assert summary["stages"][-1]["source_mode"] == "yolo26m_botsort_reid"
    assert validate_artifact_file("tracks", tmp_path / "trk1_h100_real" / "tracks.json")
