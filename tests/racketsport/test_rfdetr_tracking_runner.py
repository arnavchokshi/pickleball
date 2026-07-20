from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.racketsport import process_video
from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.orchestrator import (
    PERSON_DETECTOR_KILL_SWITCH_ENV,
    RFDETR_LARGE_MODEL_ID,
    RFDETR_PERSON_DETECTOR_CANDIDATE_STACK_KEY,
    YOLO26M_MODEL_ID,
    PersonDetectorSelection,
    RealRFDETRBoTSORTTrackingRunner,
    RealYOLO26BoTSORTReIDTrackingRunner,
    StageContext,
    _default_runners,
    _load_rfdetr_large,
    _person_detector_selection_from_stack_entry,
    _verified_botsort_no_reid_config,
    _verified_botsort_reid_config,
    resolve_person_detector_selection,
    run_pipeline,
)
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    validate_artifact_file,
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
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 960.0,
                    "cy": 540.0,
                    "dist": [],
                    "source": "manual",
                },
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


def _write_manifest(path: Path, checkpoint: Path, *, include_rfdetr: bool = True) -> None:
    models: list[dict[str, Any]] = [
        {
            "id": "yolo26m",
            "stage": "person_detect",
            "use": "fallback",
            "source": "https://docs.ultralytics.com/models/yolo26/",
            "license": "AGPL-3.0",
            "commercial_posture": "agpl_caveat",
            "status": "available_on_h100",
            "local_path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "fallbacks": [],
        }
    ]
    if include_rfdetr:
        models.append(
            {
                "id": "rfdetr_large_2026",
                "stage": "person_detect",
                "use": "pending preview candidate",
                "source": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
                "license": "Apache-2.0",
                "commercial_posture": "ok",
                "status": "available_on_h100",
                "local_path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "fallbacks": ["yolo26m"],
            }
        )
    path.write_text(json.dumps({"schema_version": 1, "models": models}), encoding="utf-8")


def _runner_inputs(tmp_path: Path, *, include_rfdetr: bool = True) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "rf-detr-large-2026.pth"
    checkpoint.write_bytes(b"rfdetr checkpoint fixture")
    manifest = tmp_path / "MANIFEST.json"
    _write_manifest(manifest, checkpoint, include_rfdetr=include_rfdetr)
    tracker_config = tmp_path / "botsort_no_reid_loose.yaml"
    tracker_config.write_text("tracker_type: botsort\nwith_reid: False\n", encoding="utf-8")
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "source.mp4").write_bytes(b"frame iterator is injected")
    _write_sidecar(inputs / "capture_sidecar.json")
    _write_calibration(run_dir / "court_calibration.json")
    return manifest, tracker_config, inputs, run_dir


def _tracks_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "t": 0.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [1.0, 2.0],
                        "conf": 0.9,
                    }
                ],
            }
        ],
        "rally_spans": [],
    }


def test_rfdetr_runner_filters_conf_and_class_then_writes_unchanged_pool_schema(tmp_path: Path) -> None:
    manifest, tracker_config, inputs, run_dir = _runner_inputs(tmp_path)
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8), np.zeros((1080, 1920, 3), dtype=np.uint8)]
    detector_calls: list[float] = []
    tracker_inputs: list[tuple[list[float], list[float]]] = []

    class Prediction:
        def __init__(self, frame_index: int) -> None:
            if frame_index == 0:
                self.xyxy = np.asarray(
                    [
                        [1080.0, 900.0, 1120.0, 1000.0],
                        [10.0, 10.0, 20.0, 20.0],
                        [1180.0, 900.0, 1220.0, 1000.0],
                    ]
                )
                self.confidence = np.asarray([0.91, 0.99, 0.17])
                self.class_id = np.asarray([1, 2, 1])
            else:
                self.xyxy = np.asarray([[1084.0, 900.0, 1124.0, 1000.0]])
                self.confidence = np.asarray([0.89])
                self.class_id = np.asarray([1])

    class Detector:
        frame_index = 0
        model_config = SimpleNamespace(resolution=704)

        def predict(self, _image: Any, *, threshold: float) -> Prediction:
            detector_calls.append(threshold)
            prediction = Prediction(self.frame_index)
            self.frame_index += 1
            return prediction

    class Tracker:
        track_id = 9

        def update(self, boxes: Any, _frame: Any) -> np.ndarray:
            tracker_inputs.append((boxes.conf.tolist(), boxes.cls.tolist()))
            rows = []
            for bbox, score in zip(boxes.xyxy, boxes.conf):
                rows.append([*bbox.tolist(), self.track_id, float(score), 0.0, 0.0])
            return np.asarray(rows, dtype=np.float32).reshape(-1, 8)

    runner = RealRFDETRBoTSORTTrackingRunner(
        manifest_path=manifest,
        tracker_config_path=tracker_config,
        detector_factory=lambda _path, native_input_size: (
            Detector()
            if native_input_size == 704
            else pytest.fail(f"unexpected RF-DETR resolution {native_input_size}")
        ),
        tracker_factory=lambda _path, _device: Tracker(),
        frame_iterator=lambda _path, _max_frames: frames,
    )
    result = runner.run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            device="cpu",
            max_frames=2,
        )
    )

    assert detector_calls == [0.18, 0.18]
    assert tracker_inputs == [([pytest.approx(0.91)], [0.0]), ([pytest.approx(0.89)], [0.0])]
    assert result.source_mode == "rfdetr_large_2026_botsort_no_reid_loose"
    assert result.produced_artifacts == (
        "raw_tracked_detections.json",
        "tracked_detections.json",
        "metrics.json",
        "tracks.json",
    )
    raw_pool = json.loads((run_dir / "raw_tracked_detections.json").read_text(encoding="utf-8"))
    assert raw_pool["frames"][0]["detections"] == [
        {
            "bbox": [1080.0, 900.0, 1120.0, 1000.0],
            "class": "person",
            "conf": pytest.approx(0.91),
            "track_id": 9,
        }
    ]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    detector_metrics = dict(metrics["detector"])
    runtime_package_version = detector_metrics.pop("runtime_package_version")
    assert runtime_package_version is None or isinstance(runtime_package_version, str)
    assert detector_metrics == {
        "api": "rfdetr.RFDETRLarge.predict",
        "checkpoint_sha256": hashlib.sha256(b"rfdetr checkpoint fixture").hexdigest(),
        "conf_floor": 0.18,
        "model_id": "rfdetr_large_2026",
        "native_input_size_loaded": 704,
        "native_input_size_requested": 704,
        "person_class_id": 1,
        "runtime_package": "rfdetr",
    }
    assert metrics["tracker"]["api"] == "per_frame_update"
    assert metrics["counts"]["detector_wrong_class_boxes"] == 1
    assert metrics["counts"]["detector_below_conf_boxes"] == 1
    assert result.metrics["checkpoint_sha256_verified"] == hashlib.sha256(
        b"rfdetr checkpoint fixture"
    ).hexdigest()
    assert result.metrics["native_input_size"] == 704
    assert validate_artifact_file("tracks", run_dir / "tracks.json")


def test_rfdetr_operating_point_is_frozen() -> None:
    with pytest.raises(ValueError, match="conf floor is frozen at 0.18"):
        RealRFDETRBoTSORTTrackingRunner(conf=0.30)
    with pytest.raises(ValueError, match="person class id is frozen at 1"):
        RealRFDETRBoTSORTTrackingRunner(person_class_id=0)
    with pytest.raises(ValueError, match="native input size is frozen at 704"):
        RealRFDETRBoTSORTTrackingRunner(native_input_size=960)


def test_runner_rejects_selection_provenance_that_disagrees_with_runtime() -> None:
    selection = resolve_person_detector_selection(YOLO26M_MODEL_ID)
    mismatched = PersonDetectorSelection(
        **{
            **selection.__dict__,
            "conf_floor": 0.25,
        }
    )

    with pytest.raises(
        ValueError,
        match="selection provenance does not match runtime",
    ):
        RealYOLO26BoTSORTReIDTrackingRunner(
            detector_selection=mismatched,
        )


@pytest.mark.parametrize(
    ("validator", "first_value", "second_value"),
    [
        (_verified_botsort_no_reid_config, "False", "True"),
        (_verified_botsort_reid_config, "True", "False"),
    ],
)
def test_tracker_config_validation_rejects_conflicting_duplicate_reid_values(
    tmp_path: Path,
    validator: Any,
    first_value: str,
    second_value: str,
) -> None:
    tracker_config = tmp_path / "conflicting_tracker.yaml"
    tracker_config.write_text(
        "tracker_type: botsort\n"
        f"with_reid: {first_value}\n"
        f"with_reid: {second_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'with_reid'"):
        validator(tracker_config)


def test_mock_rfdetr_detector_drives_real_botsort_cpu_smoke(tmp_path: Path) -> None:
    manifest, _tracker_config, inputs, run_dir = _runner_inputs(tmp_path)
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]

    class Detector:
        frame_index = 0
        model_config = SimpleNamespace(resolution=704)

        def predict(self, _image: Any, *, threshold: float) -> Any:
            assert threshold == 0.18
            x1 = 1080.0 + 4.0 * self.frame_index
            self.frame_index += 1
            return type(
                "Prediction",
                (),
                {
                    "xyxy": np.asarray([[x1, 900.0, x1 + 40.0, 1000.0]]),
                    "confidence": np.asarray([0.91]),
                    "class_id": np.asarray([1]),
                },
            )()

    runner = RealRFDETRBoTSORTTrackingRunner(
        manifest_path=manifest,
        tracker_config_path=Path("configs/racketsport/botsort_no_reid_loose.yaml"),
        detector_factory=lambda _path, native_input_size: (
            Detector()
            if native_input_size == 704
            else pytest.fail(f"unexpected RF-DETR resolution {native_input_size}")
        ),
        frame_iterator=lambda _path, _max_frames: frames,
    )
    result = runner.run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            device="cpu",
            max_frames=3,
        )
    )

    assert result.metrics["tracker_frames"] == 3
    assert result.metrics["tracked_person_boxes"] == 3
    assert validate_artifact_file("tracks", run_dir / "tracks.json")


def test_default_yolo_and_explicit_rfdetr_candidate_resolve_full_operating_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = {
        "tracking_mode": "real",
        "tracking_video": None,
        "manifest_path": Path("models/MANIFEST.json"),
        "tracker_config_path": Path("configs/racketsport/botsort_reid.yaml"),
        "max_players": 4,
        "court_margin_m": 1.0,
        "id_strategy": "auto",
        "ball_source_path": None,
    }
    monkeypatch.delenv(PERSON_DETECTOR_KILL_SWITCH_ENV, raising=False)
    default_runners = _default_runners(person_detector=None, **common)
    assert isinstance(default_runners["tracking"], RealYOLO26BoTSORTReIDTrackingRunner)
    default_selection = resolve_person_detector_selection(None)
    assert default_selection.identity_payload() == {
        "model_id": "yolo26m",
        "conf_floor": 0.05,
        "person_class_id": 0,
        "native_input_size": 1536,
        "tracker_config": str(
            Path("configs/racketsport/botsort_reid.yaml").resolve()
        ),
        "stack_entry_key": "tracking.person_detector",
        "stack_entry_status": "WIRED_DEFAULT",
        "stack_revision": 14,
        "override_source": "best_stack_default",
        "tracker_config_source": "best_stack",
    }

    monkeypatch.setenv(PERSON_DETECTOR_KILL_SWITCH_ENV, "yolo26m")
    explicit_yolo_runners = _default_runners(person_detector=None, **common)
    assert isinstance(explicit_yolo_runners["tracking"], RealYOLO26BoTSORTReIDTrackingRunner)

    monkeypatch.setenv(PERSON_DETECTOR_KILL_SWITCH_ENV, "rfdetr_large_2026")
    candidate_runners = _default_runners(person_detector=None, **common)
    assert isinstance(candidate_runners["tracking"], RealRFDETRBoTSORTTrackingRunner)
    candidate_selection = resolve_person_detector_selection(None)
    assert candidate_selection.model_id == RFDETR_LARGE_MODEL_ID
    assert candidate_selection.conf_floor == 0.18
    assert candidate_selection.person_class_id == 1
    assert candidate_selection.native_input_size == 704
    assert candidate_selection.tracker_config_path.name == "botsort_no_reid_loose.yaml"
    assert candidate_selection.stack_entry_key == RFDETR_PERSON_DETECTOR_CANDIDATE_STACK_KEY
    assert candidate_selection.stack_entry_status == "PENDING"
    assert candidate_selection.override_source == (
        f"environment:{PERSON_DETECTOR_KILL_SWITCH_ENV}"
    )

    monkeypatch.setenv(PERSON_DETECTOR_KILL_SWITCH_ENV, "unknown")
    with pytest.raises(ValueError, match="unsupported person detector"):
        _default_runners(person_detector=None, **common)


def test_candidate_stack_entry_cannot_silently_resolve_to_yolo(
    tmp_path: Path,
) -> None:
    tracker_config = tmp_path / "botsort_reid.yaml"
    tracker_config.write_text(
        "tracker_type: botsort\nwith_reid: True\n",
        encoding="utf-8",
    )
    mismatched_candidate = SimpleNamespace(
        key=RFDETR_PERSON_DETECTOR_CANDIDATE_STACK_KEY,
        status="PENDING",
        value={
            "model_id": YOLO26M_MODEL_ID,
            "conf_floor": 0.05,
            "person_class_id": 0,
            "native_input_size": 1536,
            "tracker_config": str(tracker_config),
            "kill_switch_env": PERSON_DETECTOR_KILL_SWITCH_ENV,
        },
    )

    with pytest.raises(
        ValueError,
        match="model_id must equal 'rfdetr_large_2026'",
    ):
        _person_detector_selection_from_stack_entry(
            manifest_path=tmp_path / "best_stack.json",
            manifest_revision=14,
            entry=mismatched_candidate,
            expected_model_id=RFDETR_LARGE_MODEL_ID,
            override_source="test",
        )


def test_rfdetr_manifest_resolution_failure_is_loud_before_detector_load(tmp_path: Path) -> None:
    manifest, tracker_config, inputs, run_dir = _runner_inputs(tmp_path, include_rfdetr=False)
    loaded = False

    def detector_factory(_path: Path, _native_input_size: int) -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("detector must not load before manifest resolution")

    runner = RealRFDETRBoTSORTTrackingRunner(
        manifest_path=manifest,
        tracker_config_path=tracker_config,
        detector_factory=detector_factory,
        tracker_factory=lambda _path, _device: object(),
        frame_iterator=lambda _path, _max_frames: [],
    )
    with pytest.raises(KeyError, match="model id not found in manifest: rfdetr_large_2026"):
        runner.run(
            StageContext(
                clip="clip_001",
                inputs_dir=inputs,
                run_dir=run_dir,
                sport="pickleball",
            )
        )
    assert loaded is False


def test_committed_rfdetr_manifest_and_preview_stack_contract() -> None:
    model_manifest = json.loads(Path("models/MANIFEST.json").read_text(encoding="utf-8"))
    [model] = [entry for entry in model_manifest["models"] if entry["id"] == "rfdetr_large_2026"]
    assert model["license"] == "Apache-2.0"
    assert model["sha256"] == "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"

    stack = json.loads(Path("configs/racketsport/best_stack.json").read_text(encoding="utf-8"))
    default_entry = stack["entries"]["tracking.person_detector"]
    candidate_entry = stack["entries"][RFDETR_PERSON_DETECTOR_CANDIDATE_STACK_KEY]
    assert stack["revision"] == 14
    assert default_entry["status"] == "WIRED_DEFAULT"
    assert default_entry["value"]["model_id"] == "yolo26m"
    assert default_entry["value"]["tracker_config"].endswith("botsort_reid.yaml")
    assert candidate_entry["status"] == "PENDING"
    assert candidate_entry["trust_band"] == "preview"
    assert candidate_entry["do_not_promote"] is True
    assert candidate_entry["value"]["model_id"] == "rfdetr_large_2026"
    assert candidate_entry["value"]["fallback_model_id"] == "yolo26m"
    assert candidate_entry["proven_against"]["production_reproduction_status"] == "NO-ATTEMPT"
    assert "fabricated f45-86 association bridge" in candidate_entry["notes"]
    assert "not detector output" in candidate_entry["notes"]


def test_rfdetr_loader_passes_and_verifies_native_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeRFDETRLarge:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self.model_config = SimpleNamespace(resolution=kwargs["resolution"])

    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        SimpleNamespace(RFDETRLarge=FakeRFDETRLarge),
    )
    checkpoint = tmp_path / "rf-detr-large-2026.pth"
    checkpoint.write_bytes(b"fixture")

    detector = _load_rfdetr_large(checkpoint, 704)

    assert detector.model_config.resolution == 704
    assert calls == [
        {
            "pretrain_weights": str(checkpoint),
            "resolution": 704,
        }
    ]


def _process_pipeline_options(tmp_path: Path) -> process_video.PipelineOptions:
    video = tmp_path / "input.mp4"
    if not video.is_file():
        video.write_bytes(b"content-identity-only video fixture")
    options = process_video.PipelineOptions(
        video=video,
        clip="clip_001",
        run_dir=tmp_path / "process_run",
    )
    options.global_association = False
    options.no_gpu = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    (options.clip_dir / "source.mp4").write_bytes(video.read_bytes())
    _write_calibration(options.clip_dir / "court_calibration.json")
    return options


def _install_fake_process_tracking_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[PersonDetectorSelection],
) -> None:
    def fake_run_pipeline(**kwargs: Any) -> dict[str, Any]:
        selection = kwargs["person_detector"]
        assert isinstance(selection, PersonDetectorSelection)
        calls.append(selection)
        run_dir = Path(kwargs["run_dir"])
        (run_dir / "tracks.json").write_text(
            json.dumps(_tracks_payload()),
            encoding="utf-8",
        )
        source_mode = (
            "rfdetr_large_2026_botsort_no_reid_loose"
            if selection.model_id == RFDETR_LARGE_MODEL_ID
            else "yolo26m_botsort_reid"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_person_tracker_candidate",
                    "source_mode": source_mode,
                    "person_detector": selection.identity_payload(),
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "pass",
            "stages": [
                {
                    "stage": "tracking",
                    "status": "ran",
                    "source_mode": source_mode,
                    "notes": ["fake detector-aware process runner"],
                }
            ],
        }

    monkeypatch.setattr(
        process_video.orchestrator,
        "run_pipeline",
        fake_run_pipeline,
    )


def test_process_tracking_reuse_invalidates_on_detector_and_override_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _process_pipeline_options(tmp_path)
    source_identity = process_video.SourceIdentity.from_path(
        options.video,
        timing={"fps": 30.0, "frame_count": 1},
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_source_content_identity",
        lambda _self: source_identity,
    )
    calls: list[PersonDetectorSelection] = []
    _install_fake_process_tracking_runner(monkeypatch, calls)

    monkeypatch.delenv(PERSON_DETECTOR_KILL_SWITCH_ENV, raising=False)
    first_pipeline = process_video.ProcessVideoPipeline(options)
    first = first_pipeline._run_stage_safely(
        "tracking",
        first_pipeline._stage_tracking,
    )
    assert first.status == "ran"
    assert [call.model_id for call in calls] == [YOLO26M_MODEL_ID]

    same_default_pipeline = process_video.ProcessVideoPipeline(options)
    same_default = same_default_pipeline._run_stage_safely(
        "tracking",
        same_default_pipeline._stage_tracking,
    )
    assert same_default.status == "skipped"
    assert len(calls) == 1

    monkeypatch.setenv(PERSON_DETECTOR_KILL_SWITCH_ENV, YOLO26M_MODEL_ID)
    explicit_yolo_pipeline = process_video.ProcessVideoPipeline(options)
    explicit_yolo = explicit_yolo_pipeline._run_stage_safely(
        "tracking",
        explicit_yolo_pipeline._stage_tracking,
    )
    assert explicit_yolo.status == "ran"
    assert [call.model_id for call in calls] == [
        YOLO26M_MODEL_ID,
        YOLO26M_MODEL_ID,
    ]
    assert calls[-1].override_source == (
        f"environment:{PERSON_DETECTOR_KILL_SWITCH_ENV}"
    )

    monkeypatch.setenv(PERSON_DETECTOR_KILL_SWITCH_ENV, RFDETR_LARGE_MODEL_ID)
    candidate_pipeline = process_video.ProcessVideoPipeline(options)
    candidate = candidate_pipeline._run_stage_safely(
        "tracking",
        candidate_pipeline._stage_tracking,
    )
    assert candidate.status == "ran"
    assert [call.model_id for call in calls] == [
        YOLO26M_MODEL_ID,
        YOLO26M_MODEL_ID,
        RFDETR_LARGE_MODEL_ID,
    ]
    assert calls[-1].tracker_config_path.name == "botsort_no_reid_loose.yaml"

    same_candidate_pipeline = process_video.ProcessVideoPipeline(options)
    same_candidate = same_candidate_pipeline._run_stage_safely(
        "tracking",
        same_candidate_pipeline._stage_tracking,
    )
    assert same_candidate.status == "skipped"
    assert len(calls) == 3

    tampered_tracks = _tracks_payload()
    tampered_tracks["players"][0]["id"] = 99
    (options.clip_dir / "tracks.json").write_text(
        json.dumps(tampered_tracks),
        encoding="utf-8",
    )
    tampered_pipeline = process_video.ProcessVideoPipeline(options)
    rebuilt = tampered_pipeline._run_stage_safely(
        "tracking",
        tampered_pipeline._stage_tracking,
    )
    assert rebuilt.status == "ran"
    assert len(calls) == 4
    rebuilt_tracks = json.loads(
        (options.clip_dir / "tracks.json").read_text(encoding="utf-8")
    )
    assert rebuilt_tracks["players"][0]["id"] == 1
    assert any("quarantined stale tracking outputs" in note for note in rebuilt.notes)

    identity_manifest = (
        tampered_pipeline._run_identity_store.current_manifest("tracking")
    )
    assert identity_manifest is not None
    identity = identity_manifest["identity"]
    detector_identity = identity["config"]["person_detector"]
    assert detector_identity == {
        **calls[-1].identity_payload(),
        "declared_checkpoint_sha256": (
            "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
        ),
    }
    tracker_identity = identity["models"]["tracker_config"]
    tracker_bytes = calls[-1].tracker_config_path.read_bytes()
    assert tracker_identity["sha256"] == hashlib.sha256(tracker_bytes).hexdigest()
    assert "person_detector_checkpoint" in identity["models"]
    tracks_row = [
        row
        for row in identity_manifest["external_artifacts"]
        if row["path"] == "tracks.json"
    ]
    assert len(tracks_row) == 1


def test_process_tracking_rf_failure_quarantines_old_yolo_tracks_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _process_pipeline_options(tmp_path)
    source_identity = process_video.SourceIdentity.from_path(
        options.video,
        timing={"fps": 30.0, "frame_count": 1},
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_source_content_identity",
        lambda _self: source_identity,
    )
    calls: list[PersonDetectorSelection] = []
    _install_fake_process_tracking_runner(monkeypatch, calls)

    monkeypatch.delenv(PERSON_DETECTOR_KILL_SWITCH_ENV, raising=False)
    yolo_pipeline = process_video.ProcessVideoPipeline(options)
    first = yolo_pipeline._run_stage_safely(
        "tracking",
        yolo_pipeline._stage_tracking,
    )
    assert first.status == "ran"
    assert (options.clip_dir / "tracks.json").is_file()
    stale_association = options.clip_dir / "global_association"
    stale_association.mkdir()
    (stale_association / "tracks.json").write_text(
        json.dumps(_tracks_payload()),
        encoding="utf-8",
    )

    def failed_rfdetr_pipeline(**kwargs: Any) -> dict[str, Any]:
        selection = kwargs["person_detector"]
        assert isinstance(selection, PersonDetectorSelection)
        assert selection.model_id == RFDETR_LARGE_MODEL_ID
        return {
            "status": "blocked",
            "stages": [
                {
                    "stage": "tracking",
                    "status": "fail",
                    "source_mode": "rfdetr_large_2026_botsort_no_reid_loose",
                    "notes": ["missing RF-DETR checkpoint"],
                }
            ],
        }

    monkeypatch.setattr(
        process_video.orchestrator,
        "run_pipeline",
        failed_rfdetr_pipeline,
    )
    monkeypatch.setenv(
        PERSON_DETECTOR_KILL_SWITCH_ENV,
        RFDETR_LARGE_MODEL_ID,
    )
    rfdetr_pipeline = process_video.ProcessVideoPipeline(options)
    downstream_called = False

    def forbidden_downstream() -> process_video.StageOutcome:
        nonlocal downstream_called
        downstream_called = True
        return process_video.StageOutcome(
            stage="placement",
            status="ran",
            wall_seconds=0.0,
        )

    hard_failed = rfdetr_pipeline._run_stage_list(
        [
            ("tracking", rfdetr_pipeline._stage_tracking),
            ("placement", forbidden_downstream),
        ],
    )
    failed = rfdetr_pipeline.stage_outcomes[-1]

    assert hard_failed is True
    assert failed.status == "failed"
    assert downstream_called is False
    assert not (options.clip_dir / "tracks.json").exists()
    quarantined_tracks = list(
        (options.clip_dir / ".run_identity" / "quarantine" / "tracking").glob(
            "*/tracks.json.quarantined"
        )
    )
    assert quarantined_tracks
    assert not list(options.run_dir.rglob("tracks.json"))
    assert any("quarantined stale tracking outputs" in note for note in failed.notes)


def test_process_tracking_association_failure_quarantines_partial_rfdetr_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _process_pipeline_options(tmp_path)
    options.global_association = True
    source_identity = process_video.SourceIdentity.from_path(
        options.video,
        timing={"fps": 30.0, "frame_count": 1},
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_source_content_identity",
        lambda _self: source_identity,
    )
    calls: list[PersonDetectorSelection] = []
    _install_fake_process_tracking_runner(monkeypatch, calls)
    monkeypatch.setenv(
        PERSON_DETECTOR_KILL_SWITCH_ENV,
        RFDETR_LARGE_MODEL_ID,
    )

    pipeline = process_video.ProcessVideoPipeline(options)

    def fail_association() -> list[str]:
        association_dir = options.clip_dir / "global_association"
        association_dir.mkdir()
        (association_dir / "tracks.json").write_text(
            json.dumps(_tracks_payload()),
            encoding="utf-8",
        )
        raise process_video._HardStageFailure("association failed after detector success")

    monkeypatch.setattr(
        pipeline,
        "_attempt_global_association",
        fail_association,
    )
    failed = pipeline._run_stage_safely(
        "tracking",
        pipeline._stage_tracking,
    )

    assert failed.status == "failed"
    assert not (options.clip_dir / "tracks.json").exists()
    assert not (options.clip_dir / "metrics.json").exists()
    assert not (options.clip_dir / "global_association").exists()
    assert not list(options.run_dir.rglob("tracks.json"))


def test_process_tracking_identity_binds_global_association_scorer_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _process_pipeline_options(tmp_path)
    options.global_association = True
    source_identity = process_video.SourceIdentity.from_path(
        options.video,
        timing={"fps": 30.0, "frame_count": 1},
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_source_content_identity",
        lambda _self: source_identity,
    )
    calls: list[PersonDetectorSelection] = []
    _install_fake_process_tracking_runner(monkeypatch, calls)

    def fake_association(_self: process_video.ProcessVideoPipeline) -> list[str]:
        association_dir = options.clip_dir / "global_association"
        association_dir.mkdir(exist_ok=True)
        (association_dir / "tracks.json").write_text(
            json.dumps(_tracks_payload()),
            encoding="utf-8",
        )
        return ["fake global association"]

    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_attempt_global_association",
        fake_association,
    )
    monkeypatch.delenv(PERSON_DETECTOR_KILL_SWITCH_ENV, raising=False)

    first_pipeline = process_video.ProcessVideoPipeline(options)
    first = first_pipeline._run_stage_safely(
        "tracking",
        first_pipeline._stage_tracking,
    )
    assert first.status == "ran"
    assert len(calls) == 1
    manifest = first_pipeline._run_identity_store.current_manifest("tracking")
    assert manifest is not None
    assert any(
        row["path"] == "global_association"
        for row in manifest["external_artifacts"]
    )

    tampered = _tracks_payload()
    tampered["players"][0]["id"] = 88
    (options.clip_dir / "global_association" / "tracks.json").write_text(
        json.dumps(tampered),
        encoding="utf-8",
    )

    second_pipeline = process_video.ProcessVideoPipeline(options)
    rebuilt = second_pipeline._run_stage_safely(
        "tracking",
        second_pipeline._stage_tracking,
    )

    assert rebuilt.status == "ran"
    assert len(calls) == 2
    discovered = list(options.run_dir.rglob("tracks.json"))
    assert len(discovered) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["players"][0]["id"] == 1
        for path in discovered
    )


def test_process_tracking_uses_selection_cached_before_reuse_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _process_pipeline_options(tmp_path)
    source_identity = process_video.SourceIdentity.from_path(
        options.video,
        timing={"fps": 30.0, "frame_count": 1},
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_source_content_identity",
        lambda _self: source_identity,
    )
    calls: list[PersonDetectorSelection] = []
    _install_fake_process_tracking_runner(monkeypatch, calls)
    monkeypatch.delenv(PERSON_DETECTOR_KILL_SWITCH_ENV, raising=False)

    pipeline = process_video.ProcessVideoPipeline(options)
    original_decision = pipeline._run_identity_store.decision

    def decision_then_change_environment(*args: Any, **kwargs: Any) -> Any:
        decision = original_decision(*args, **kwargs)
        monkeypatch.setenv(
            PERSON_DETECTOR_KILL_SWITCH_ENV,
            RFDETR_LARGE_MODEL_ID,
        )
        return decision

    monkeypatch.setattr(
        pipeline._run_identity_store,
        "decision",
        decision_then_change_environment,
    )
    outcome = pipeline._run_stage_safely("tracking", pipeline._stage_tracking)

    assert outcome.status == "ran"
    assert len(calls) == 1
    assert calls[0].model_id == YOLO26M_MODEL_ID
    assert calls[0].override_source == "best_stack_default"
    assert outcome.metrics["source_mode"] == "yolo26m_botsort_reid"


@pytest.mark.h100
@pytest.mark.integration
def test_h100_rfdetr_tracking_runner_smoke_from_env(tmp_path: Path) -> None:
    if os.environ.get("RUN_H100_RFDETR_TRACKING") != "1":
        pytest.skip(
            "set RUN_H100_RFDETR_TRACKING=1 plus "
            "TRK_RFDETR_H100_INPUTS/TRK_RFDETR_H100_VIDEO to run"
        )
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    if not torch.cuda.is_available():
        pytest.skip("RF-DETR GPU smoke requires torch.cuda.is_available()")

    inputs = Path(os.environ["TRK_RFDETR_H100_INPUTS"])
    video = Path(os.environ["TRK_RFDETR_H100_VIDEO"])
    manifest = Path(
        os.environ.get("TRK_RFDETR_H100_MANIFEST", "models/MANIFEST.json")
    )
    run_dir = tmp_path / "rfdetr_h100_real"
    summary = run_pipeline(
        clip=os.environ.get("TRK_RFDETR_H100_CLIP", inputs.name),
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="tracking",
        tracking_mode="real",
        person_detector=RFDETR_LARGE_MODEL_ID,
        tracking_video=video,
        manifest_path=manifest,
        device="cuda:0",
        max_frames=int(os.environ.get("TRK_RFDETR_H100_MAX_FRAMES", "90")),
    )

    tracking_stage = [
        stage for stage in summary["stages"] if stage["stage"] == "tracking"
    ][-1]
    assert tracking_stage["status"] == "ran"
    assert tracking_stage["real_model"] is True
    assert (
        tracking_stage["source_mode"]
        == "rfdetr_large_2026_botsort_no_reid_loose"
    )
    assert tracking_stage["metrics"]["checkpoint_sha256_verified"] == (
        "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["detector"]["conf_floor"] == 0.18
    assert metrics["detector"]["person_class_id"] == 1
    assert metrics["detector"]["native_input_size_requested"] == 704
    assert metrics["detector"]["native_input_size_loaded"] == 704
    assert metrics["detector"]["checkpoint_sha256"] == (
        "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
    )
    assert metrics["tracker"]["with_reid"] is False
    assert metrics["tracker"]["config"].endswith(
        "botsort_no_reid_loose.yaml"
    )
    assert validate_artifact_file("tracks", run_dir / "tracks.json")
