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
    build_source_appearance_diagnostic,
    build_source_reid_embedding_export,
    inspect_detection_appearance_inputs,
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
