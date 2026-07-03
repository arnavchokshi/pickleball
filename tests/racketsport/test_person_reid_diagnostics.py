from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.person_reid_diagnostics import (
    AppearanceDiagnosticConfig,
    ReIDEmbeddingExportConfig,
    _extract_embeddings_in_batches,
    _filter_compatible_state_dict,
    assert_reid_checkpoint_clip_safe,
    build_source_appearance_diagnostic,
    build_source_reid_embedding_export,
    inspect_detection_appearance_inputs,
    reid_checkpoint_training_provenance,
    resolve_reid_device,
)


def _detections_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [2, 2, 12, 18], "class": "person", "conf": 0.9, "track_id": 1},
                    {"bbox": [20, 2, 30, 18], "class": "person", "conf": 0.8, "track_id": 2},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [3, 2, 13, 18], "class": "person", "conf": 0.88, "track_id": 1},
                    {"bbox": [21, 2, 31, 18], "class": "person", "conf": 0.82, "track_id": 2},
                ],
            },
        ],
    }


def test_inspect_detection_appearance_inputs_reports_no_embeddings() -> None:
    summary = inspect_detection_appearance_inputs(_detections_payload())

    assert summary["status"] == "no_persisted_appearance_embeddings"
    assert summary["source_only"] is True
    assert summary["uses_cvat_labels"] is False
    assert summary["person_detection_count"] == 4
    assert summary["embedding_like_keys"] == []
    assert summary["detection_keys"]["bbox"] == 4
    assert summary["detection_keys"]["track_id"] == 4


def test_build_source_appearance_diagnostic_exports_features_and_crop_sheets(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    try:
        for _ in range(2):
            frame = np.zeros((24, 40, 3), dtype=np.uint8)
            frame[:, :20] = (0, 0, 255)
            frame[:, 20:] = (255, 0, 0)
            writer.write(frame)
    finally:
        writer.release()

    out_dir = tmp_path / "diag"
    report = build_source_appearance_diagnostic(
        video_path=video_path,
        detections_payload=_detections_payload(),
        out_dir=out_dir,
        config=AppearanceDiagnosticConfig(max_samples_per_track=2, crop_padding_px=0, tile_size=32),
    )

    assert report["status"] == "source_only_appearance_diagnostic"
    assert report["promote_trk"] is False
    assert report["appearance_feature_type"] == "cpu_color_histogram_not_reid_embedding"
    assert report["sample_count"] == 4
    assert sorted(report["tracks"].keys()) == ["1", "2"]
    assert (out_dir / "source_appearance_diagnostics.json").exists()
    assert (out_dir / "track_appearance_features.json").exists()
    assert (out_dir / "crop_sheets" / "source_track_1.png").exists()
    assert (out_dir / "crop_sheets" / "source_track_2.png").exists()

    features = json.loads((out_dir / "track_appearance_features.json").read_text(encoding="utf-8"))
    assert features["uses_cvat_labels"] is False
    assert len(features["samples"]) == 4
    assert len(features["tracks"]["1"]["mean_feature"]) == 48


def test_build_source_reid_embedding_export_persists_model_vectors_and_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    try:
        for frame_index in range(2):
            frame = np.zeros((24, 40, 3), dtype=np.uint8)
            frame[:, :20] = (0, 0, 255 - frame_index)
            frame[:, 20:] = (255 - frame_index, 0, 0)
            writer.write(frame)
    finally:
        writer.release()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake model bytes")
    seen_crop_shapes: list[tuple[int, int, int]] = []

    def fake_embedder(crops: list[np.ndarray]) -> list[list[float]]:
        seen_crop_shapes.extend(tuple(int(value) for value in crop.shape) for crop in crops)
        return [[float(index), float(index) + 0.25, 1.0] for index, _crop in enumerate(crops)]

    out_path = tmp_path / "learned_reid_embeddings.json"
    report = build_source_reid_embedding_export(
        video_path=video_path,
        detections_payload=_detections_payload(),
        output_path=out_path,
        model_path=model_path,
        command_metadata={"argv": ["export", "--bounded"], "cwd": str(tmp_path)},
        config=ReIDEmbeddingExportConfig(max_detections=3, crop_padding_px=0, batch_size=2, l2_normalize=False),
        feature_extractor=fake_embedder,
    )

    assert report["status"] == "source_only_learned_embedding_export"
    assert report["promote_trk"] is False
    assert report["feature_type"] == "learned_model_embedding"
    assert report["model_path"] == str(model_path)
    assert report["model_sha256"]
    assert report["feature_dim"] == 3
    assert report["command_metadata"]["argv"] == ["export", "--bounded"]
    assert report["detection_count"] == 3
    assert len(report["detections"]) == 3
    assert seen_crop_shapes == [(16, 10, 3), (16, 10, 3), (16, 10, 3)]

    first = report["detections"][0]
    assert first["frame"] == 0
    assert first["detection_index"] == 0
    assert first["source_track_id"] == 1
    assert first["track_id"] == 1
    assert first["bbox"] == [2.0, 2.0, 12.0, 18.0]
    assert first["feature_dim"] == 3
    assert first["embedding"] == [0.0, 0.25, 1.0]

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["feature_dim"] == 3
    assert written["track_sample_counts"] == {"1": 2, "2": 1}


def test_build_source_reid_embedding_export_marks_osnet_backend_with_injected_runner(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((24, 40, 3), dtype=np.uint8))
    finally:
        writer.release()

    model_path = tmp_path / "osnet_x1_0_sportsmot.pth"
    model_path.write_bytes(b"fake osnet bytes")

    report = build_source_reid_embedding_export(
        video_path=video_path,
        detections_payload=_detections_payload(),
        output_path=tmp_path / "osnet_reid_embeddings.json",
        model_path=model_path,
        config=ReIDEmbeddingExportConfig(
            max_detections=2,
            crop_padding_px=0,
            batch_size=2,
            backend="osnet",
            osnet_model_name="osnet_x1_0",
        ),
        feature_extractor=lambda crops: [[1.0, 0.0, 0.0, 0.0] for _crop in crops],
    )

    assert report["feature_type"] == "osnet_reid_embedding"
    assert report["feature_extractor"] == "injected_osnet_feature_extractor"
    assert report["model_family"] == "osnet"
    assert report["config"]["backend"] == "osnet"
    assert report["config"]["osnet_model_name"] == "osnet_x1_0"
    assert report["detections"][0]["embedding"] == [1.0, 0.0, 0.0, 0.0]


def test_build_source_reid_embedding_export_rejects_inconsistent_feature_dimensions(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((24, 40, 3), dtype=np.uint8))
    finally:
        writer.release()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake model bytes")

    with pytest.raises(ValueError, match="consistent feature dimensions"):
        build_source_reid_embedding_export(
            video_path=video_path,
            detections_payload=_detections_payload(),
            output_path=tmp_path / "bad.json",
            model_path=model_path,
            command_metadata={"argv": ["export"]},
            config=ReIDEmbeddingExportConfig(max_detections=2),
            feature_extractor=lambda _crops: [[1.0, 2.0], [3.0]],
        )


def test_build_source_reid_diagnostics_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_source_reid_diagnostics.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "source-only appearance/ReID diagnostics" in completed.stdout


def test_export_person_reid_embeddings_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/export_person_reid_embeddings.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "learned per-detection ReID embeddings" in completed.stdout
    assert "--backend" in completed.stdout
    assert "osnet" in completed.stdout


def test_filter_compatible_state_dict_skips_classifier_shape_mismatches() -> None:
    class TensorLike:
        def __init__(self, *shape: int) -> None:
            self.shape = shape

    compatible, skipped = _filter_compatible_state_dict(
        {
            "conv.weight": TensorLike(2, 3),
            "classifier.weight": TensorLike(4, 512),
            "extra.bias": TensorLike(1),
        },
        {
            "conv.weight": TensorLike(2, 3),
            "classifier.weight": TensorLike(1000, 512),
        },
    )

    assert list(compatible) == ["conv.weight"]
    assert skipped == ["classifier.weight", "extra.bias"]


def test_reid_checkpoint_training_provenance_reports_stock_checkpoint_as_safe(tmp_path: Path) -> None:
    checkpoint = tmp_path / "osnet_x1_0_market1501.pt"
    checkpoint.write_bytes(b"fake stock checkpoint")

    provenance = reid_checkpoint_training_provenance(checkpoint)

    assert provenance["has_local_training_provenance"] is False
    assert provenance["trained_on_clip_ids"] == []
    assert provenance["weights_sha256"]
    assert_reid_checkpoint_clip_safe(provenance, clip_id="burlington_gold_0300_low_steep_corner")


def test_reid_checkpoint_training_provenance_finds_fine_tuned_train_clips(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "reid_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_person_reid_crop_dataset",
                "uses_cvat_labels": True,
                "clip_counts": {
                    "burlington_gold_0300_low_steep_corner": {"train": 480, "query": 0, "gallery": 0},
                    "outdoor_webcam_iynbd_1500_long_high_baseline": {"train": 0, "query": 40, "gallery": 120},
                },
            }
        ),
        encoding="utf-8",
    )
    save_dir = tmp_path / "checkpoints" / "osnet_finetune_v1"
    save_dir.mkdir(parents=True)
    (save_dir / "training_summary.json").write_text(
        json.dumps({"dataset_dir": str(dataset_dir), "manifest_path": str(dataset_dir / "manifest.json")}),
        encoding="utf-8",
    )
    checkpoint = save_dir / "model.pth.tar-20"
    checkpoint.write_bytes(b"fake fine-tuned checkpoint")

    provenance = reid_checkpoint_training_provenance(checkpoint)

    assert provenance["has_local_training_provenance"] is True
    assert provenance["trained_on_clip_ids"] == ["burlington_gold_0300_low_steep_corner"]
    assert provenance["held_out_val_clip_ids"] == ["outdoor_webcam_iynbd_1500_long_high_baseline"]

    # Held-out / unrelated clips remain safe to score.
    assert_reid_checkpoint_clip_safe(provenance, clip_id="outdoor_webcam_iynbd_1500_long_high_baseline")
    assert_reid_checkpoint_clip_safe(provenance, clip_id="wolverine_mixed_0200_mid_steep_corner")

    # The clip that leaked into the train split must fail closed with a clear message.
    with pytest.raises(ValueError, match="training-provenance leak"):
        assert_reid_checkpoint_clip_safe(provenance, clip_id="burlington_gold_0300_low_steep_corner")


def test_offline_authority_refuses_to_score_a_clip_the_reid_checkpoint_trained_on(tmp_path: Path) -> None:
    from threed.racketsport.offline_person_authority import run_offline_authority_candidate
    from threed.racketsport.schemas import PersonGroundTruth, PlayerTrack, TrackFrame, Tracks

    source_run = tmp_path / "source" / "clip_a" / "candidate_a"
    source_run.mkdir(parents=True)
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(2.0, 2.0, 12.0, 18.0), world_xy=(0.0, -1.0), conf=0.9)],
            )
        ],
        rally_spans=[],
    )
    (source_run / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (source_run / "tracked_detections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 10.0,
                "frames": [
                    {"frame": 0, "detections": [{"bbox": [2.0, 2.0, 12.0, 18.0], "class": "person", "conf": 0.9, "track_id": 7}]},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Fine-tuned checkpoint whose local provenance says it trained on "clip_a".
    dataset_dir = tmp_path / "reid_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"clip_counts": {"clip_a": {"train": 10, "query": 0, "gallery": 0}}}),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps({"dataset_dir": str(dataset_dir)}),
        encoding="utf-8",
    )
    model_path = checkpoint_dir / "osnet_finetuned.pth"
    model_path.write_bytes(b"fake fine-tuned checkpoint")

    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 40, 3), dtype=np.uint8))
    writer.release()

    gt_path = tmp_path / "person_ground_truth.json"
    gt_path.write_text(
        json.dumps(
            PersonGroundTruth.model_validate(
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_person_ground_truth",
                    "clip_id": "clip_a",
                    "source_format": "cvat_video_1_1",
                    "source_path": "synthetic",
                    "fps": 10.0,
                    "frames": [
                        {
                            "frame_index": 0,
                            "source_frame_id": 1,
                            "labels": [
                                {
                                    "track_id": 1,
                                    "bbox_xywh": [2.0, 2.0, 10.0, 16.0],
                                    "ignored": False,
                                    "visibility": 1.0,
                                    "confidence": 1.0,
                                    "class_id": None,
                                    "class_name": "player",
                                    "person_class": True,
                                }
                            ],
                        }
                    ],
                    "summary": {
                        "frame_count": 1,
                        "valid_label_count": 1,
                        "ignored_label_count": 0,
                        "track_ids": [1],
                        "max_valid_players_per_frame": 1,
                    },
                }
            ).model_dump(mode="json"),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        run_offline_authority_candidate(
            clip_id="clip_a",
            candidate="candidate_a",
            video_path=video_path,
            source_run_dir=source_run,
            out_dir=tmp_path / "authority",
            reid_model_path=model_path,
            ground_truth_path=gt_path,
            expected_players=1,
            feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
        )
    except ValueError as exc:
        assert "training-provenance leak" in str(exc)
        assert "clip_a" in str(exc)
    else:
        raise AssertionError("expected a training-provenance leak to fail closed")


class _FakeAccelerator:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeAccelerator(mps_available)


class _FakeTorch:
    """Mock-friendly torch stand-in so device auto-detection can be tested
    without requiring the real torch/torchreid stack to be importable."""

    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeAccelerator(cuda_available)
        self.backends = _FakeBackends(mps_available)


def test_resolve_reid_device_explicit_override_always_wins() -> None:
    # Explicit device strings win even when they contradict what auto-detect
    # would otherwise pick (e.g. forcing cpu on a cuda-capable fake torch).
    fake_torch = _FakeTorch(cuda_available=True, mps_available=True)
    assert resolve_reid_device("cpu", torch_module=fake_torch) == "cpu"
    assert resolve_reid_device("mps", torch_module=fake_torch) == "mps"
    assert resolve_reid_device("cuda:1", torch_module=fake_torch) == "cuda:1"


def test_resolve_reid_device_auto_detect_prefers_cuda_over_mps() -> None:
    fake_torch = _FakeTorch(cuda_available=True, mps_available=True)
    assert resolve_reid_device(None, torch_module=fake_torch) == "cuda:0"


def test_resolve_reid_device_auto_detect_falls_back_to_mps_on_apple_silicon() -> None:
    # This is the fix this lane made: previously the auto-detect path only
    # ever checked torch.cuda.is_available() and fell straight to "cpu" on
    # every non-CUDA machine, including Apple Silicon Macs with a working
    # MPS backend -- silently leaving the #1 measured pipeline cost center
    # (OSNet ReID embedding extraction, ~1,665-7,586 s/min-video on CPU) on
    # the slow device by default. A thorough correctness investigation (200
    # real crops: ~1e-11 cosine deviation, byte-identical clustering; a
    # follow-up full-11,095-detection-clip investigation that initially
    # looked like a device-specific divergence but resolved to an unrelated
    # concurrent code change once re-measured with matched, same-session
    # code -- see runs/trk_speed_reid_gpu_20260702T045139Z/) found no
    # device-specific correctness risk, so mps stays in auto-detect.
    fake_torch = _FakeTorch(cuda_available=False, mps_available=True)
    assert resolve_reid_device(None, torch_module=fake_torch) == "mps"


def test_resolve_reid_device_auto_detect_can_opt_out_of_mps_via_allow_mps_auto() -> None:
    fake_torch = _FakeTorch(cuda_available=False, mps_available=True)
    # allow_mps_auto=False overrides the module-level MPS_AUTO_DETECT_ENABLED
    # default per call, e.g. if a future finding reopens this question for a
    # specific caller without having to flip the module-wide default.
    assert resolve_reid_device(None, torch_module=fake_torch, allow_mps_auto=False) == "cpu"
    assert resolve_reid_device(None, torch_module=fake_torch, allow_mps_auto=True) == "mps"


def test_resolve_reid_device_auto_detect_falls_back_to_cpu_with_no_accelerator() -> None:
    fake_torch = _FakeTorch(cuda_available=False, mps_available=False)
    assert resolve_reid_device(None, torch_module=fake_torch) == "cpu"


def test_resolve_reid_device_treats_empty_string_as_unset() -> None:
    fake_torch = _FakeTorch(cuda_available=False, mps_available=True)
    assert resolve_reid_device("", torch_module=fake_torch) == "mps"


def test_resolve_reid_device_tolerates_backend_probe_raising() -> None:
    class _ExplodingAccelerator:
        def is_available(self) -> bool:
            raise RuntimeError("backend not fully initialized")

    class _ExplodingBackends:
        mps = _ExplodingAccelerator()

    class _ExplodingTorch:
        cuda = _ExplodingAccelerator()
        backends = _ExplodingBackends()

    # A backend that raises while probing availability must degrade to cpu,
    # not crash embedding export.
    assert resolve_reid_device(None, torch_module=_ExplodingTorch()) == "cpu"


def test_resolve_reid_device_falls_back_to_cpu_when_torch_is_not_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_reid_device(None) == "cpu"


def test_extract_embeddings_in_batches_calls_extractor_once_per_chunk_not_per_crop() -> None:
    """The OSNet embedder forward pass must run as one batched model call per
    chunk of ``batch_size`` crops (a single stacked tensor forward), not one
    singleton forward per crop -- the latter would be the actual "per-crop
    forward" cost driver CUTS_SPEC.md flagged as worth checking."""

    calls: list[int] = []

    def fake_extractor(crops: list[int]) -> list[list[float]]:
        calls.append(len(crops))
        return [[float(crop)] for crop in crops]

    crops = list(range(200))
    embeddings = _extract_embeddings_in_batches(fake_extractor, crops, batch_size=64)

    # 200 crops at batch_size=64 -> ceil(200/64) = 4 extractor calls, not 200.
    assert calls == [64, 64, 64, 8]
    assert len(embeddings) == 200
    assert embeddings[0] == [0.0]
    assert embeddings[-1] == [199.0]
