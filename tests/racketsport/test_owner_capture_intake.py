from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.owner_capture_intake import (
    PROTECTED_OWNER_EVAL_SLUGS,
    OwnerCaptureVideoMetadata,
    ProtectedEvalCaptureError,
    ReviewExportError,
    apply_reviewed_cvat_export,
    ingest_owner_capture,
    load_owner_data_manifest,
    sha256_file,
    validate_owner_data_manifest,
)


def _write_sidecar(path: Path, *, provenance: str = "camera_roll_import") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_id": path.parent.name,
                "provenance": provenance,
                "fps": 30.0,
                "resolution": [64, 48],
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 101.0,
                    "cx": 32.0,
                    "cy": 24.0,
                    "dist": [0.0, 0.0, 0.0, 0.0],
                    "source": "avfoundation_fov_estimate",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_package(root: Path, name: str = "owner_capture_001") -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / "clip.mov").write_bytes(b"owner video bytes")
    _write_sidecar(package / "capture_sidecar.json")
    return package


def _patch_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    import threed.racketsport.owner_capture_intake as intake

    monkeypatch.setattr(
        intake,
        "probe_video_metadata",
        lambda _path: OwnerCaptureVideoMetadata(width=64, height=48, fps=30.0, duration_s=1.0, frame_count=30),
    )


@pytest.mark.parametrize("slug", PROTECTED_OWNER_EVAL_SLUGS)
def test_intake_refuses_each_protected_eval_slug_before_video_probe(tmp_path: Path, slug: str) -> None:
    package = _write_package(tmp_path, name=slug)

    with pytest.raises(ProtectedEvalCaptureError, match=slug):
        ingest_owner_capture(package, manifest_path=tmp_path / "OWNER_DATA_MANIFEST.json")


def test_intake_refuses_video_sha_matching_protected_eval_content(tmp_path: Path) -> None:
    video = tmp_path / "renamed_owner_clip.mov"
    video.write_bytes(b"eval clip bytes under a renamed file")
    protected_sha = sha256_file(video)

    with pytest.raises(ProtectedEvalCaptureError, match="fake_eval_clip"):
        ingest_owner_capture(
            video,
            manifest_path=tmp_path / "OWNER_DATA_MANIFEST.json",
            protected_video_shas={protected_sha: "fake_eval_clip"},
        )


def test_registers_valid_capture_idempotently_and_round_trips_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    package = _write_package(tmp_path)
    manifest_path = tmp_path / "runs" / "owner_data" / "OWNER_DATA_MANIFEST.json"

    first = ingest_owner_capture(package, manifest_path=manifest_path)
    second = ingest_owner_capture(package, manifest_path=manifest_path)

    assert first["status"] == "registered"
    assert second["status"] == "already_registered"
    assert first["capture_id"] == second["capture_id"] == "owner_capture_001"
    manifest = load_owner_data_manifest(manifest_path)
    validate_owner_data_manifest(manifest)
    assert len(manifest["captures"]) == 1
    row = manifest["captures"][0]
    assert row["source"] == "owner_capture"
    assert row["sha256"] == sha256_file(package / "clip.mov")
    assert row["sidecar_provenance"] == "camera_roll_import"
    assert row["camera_fingerprint"].startswith("64x48@30.000:")
    assert row["review_status"] == "unreviewed"
    assert row["train_eligible"] is False


def test_post_review_flip_requires_reviewed_cvat_export_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch)
    package = _write_package(tmp_path)
    manifest_path = tmp_path / "runs" / "owner_data" / "OWNER_DATA_MANIFEST.json"
    registered = ingest_owner_capture(package, manifest_path=manifest_path)
    candidate_export = tmp_path / "candidate_review.json"
    candidate_export.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_cvat_video_annotations",
                "status": "candidate_prediction",
                "clip_id": registered["capture_id"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReviewExportError, match="reviewed"):
        apply_reviewed_cvat_export(
            registered["capture_id"],
            reviewed_export_path=candidate_export,
            manifest_path=manifest_path,
            corpus_manifest_path=tmp_path / "runs" / "training_corpora_20260702" / "owner_capture" / "manifest.json",
        )

    manifest = load_owner_data_manifest(manifest_path)
    assert manifest["captures"][0]["review_status"] == "unreviewed"
    assert manifest["captures"][0]["train_eligible"] is False


def test_reviewed_cvat_export_flips_train_eligibility_and_writes_corpus_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    package = _write_package(tmp_path)
    manifest_path = tmp_path / "runs" / "owner_data" / "OWNER_DATA_MANIFEST.json"
    registered = ingest_owner_capture(package, manifest_path=manifest_path)
    reviewed_export = tmp_path / "reviewed_boxes.json"
    reviewed_export.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_cvat_video_annotations",
                "status": "reviewed",
                "clip_id": registered["capture_id"],
                "source_format": "cvat_video_1_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_manifest_path = tmp_path / "runs" / "training_corpora_20260702" / "owner_capture" / "manifest.json"

    result = apply_reviewed_cvat_export(
        registered["capture_id"],
        reviewed_export_path=reviewed_export,
        manifest_path=manifest_path,
        corpus_manifest_path=corpus_manifest_path,
    )

    assert result["status"] == "reviewed_materialized"
    manifest = load_owner_data_manifest(manifest_path)
    row = manifest["captures"][0]
    assert row["review_status"] == "reviewed"
    assert row["train_eligible"] is True
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    assert corpus_manifest["artifact_type"] == "racketsport_owner_capture_training_corpus_manifest"
    assert corpus_manifest["samples"][0]["capture_id"] == registered["capture_id"]
    assert corpus_manifest["samples"][0]["camera_fingerprint"] == row["camera_fingerprint"]
    assert corpus_manifest["samples"][0]["review_status"] == "reviewed"


def test_existing_cvat_import_payload_without_status_is_accepted_as_reviewed_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    package = _write_package(tmp_path)
    manifest_path = tmp_path / "runs" / "owner_data" / "OWNER_DATA_MANIFEST.json"
    registered = ingest_owner_capture(package, manifest_path=manifest_path)
    imported_export = tmp_path / "reviewed_boxes_from_importer.json"
    imported_export.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_cvat_video_annotations",
                "clip_id": registered["capture_id"],
                "source_format": "cvat_video_1_1",
                "frames": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = apply_reviewed_cvat_export(
        registered["capture_id"],
        reviewed_export_path=imported_export,
        manifest_path=manifest_path,
        corpus_manifest_path=tmp_path / "runs" / "training_corpora_20260702" / "owner_capture" / "manifest.json",
    )

    assert result["status"] == "reviewed_materialized"
