from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.owner_capture_intake import (
    CandidatePredictionStatusError,
    OwnerCaptureVideoMetadata,
    build_review_manifest,
    enforce_candidate_prediction_status,
    ingest_owner_capture,
    prelabel_owner_capture,
)


def _write_registered_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    import threed.racketsport.owner_capture_intake as intake

    monkeypatch.setattr(
        intake,
        "probe_video_metadata",
        lambda _path: OwnerCaptureVideoMetadata(width=64, height=48, fps=30.0, duration_s=1.0, frame_count=30),
    )
    package = tmp_path / "owner_capture_002"
    package.mkdir()
    (package / "clip.mov").write_bytes(b"owner prelabel video")
    (package / "capture_sidecar.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_id": "owner_capture_002",
                "provenance": "camera_roll_import",
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "source": "test"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "runs" / "owner_data" / "OWNER_DATA_MANIFEST.json"
    result = ingest_owner_capture(package, manifest_path=manifest_path)
    return result["capture_id"], manifest_path


def test_prelabel_dry_run_writes_job_spec_without_model_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_id, manifest_path = _write_registered_capture(tmp_path, monkeypatch)

    result = prelabel_owner_capture(capture_id, manifest_path=manifest_path, owner_data_root=tmp_path / "runs" / "owner_data", dry_run=True)

    assert result["status"] == "dry_run_ready"
    prelabels_dir = tmp_path / "runs" / "owner_data" / capture_id / "prelabels"
    job_spec = json.loads((prelabels_dir / "prelabel_job_spec.json").read_text(encoding="utf-8"))
    review_manifest = json.loads((prelabels_dir / "review_manifest.json").read_text(encoding="utf-8"))
    assert job_spec["status"] == "candidate_prediction"
    assert job_spec["dry_run"] is True
    assert job_spec["expected_outputs"]["person_tracks"].endswith("person_tracks.json")
    assert job_spec["expected_outputs"]["ball_track"].endswith("ball_track.json")
    assert "scripts/racketsport/run_offline_person_authority.py" in job_spec["commands"]["person_tracks"]
    assert "scripts/racketsport/run_wasb_ball.py" in job_spec["commands"]["ball_track"]
    assert review_manifest["status"] == "candidate_prediction"
    assert review_manifest["segments"] == []
    assert not (prelabels_dir / "person_tracks.json").exists()
    assert not (prelabels_dir / "ball_track.json").exists()


def test_candidate_status_enforcement_refuses_reviewed_prelabel_payload() -> None:
    with pytest.raises(CandidatePredictionStatusError, match="candidate_prediction"):
        enforce_candidate_prediction_status({"status": "reviewed", "frames": []}, artifact_name="person_tracks.json")


def test_review_manifest_orders_low_confidence_segments_first() -> None:
    manifest = build_review_manifest(
        "capture_abc",
        [
            {"segment_id": "high_conf", "start_frame": 20, "end_frame": 29, "confidence": 0.91},
            {"segment_id": "low_conf", "start_frame": 0, "end_frame": 9, "confidence": 0.12},
            {"segment_id": "mid_conf", "start_frame": 10, "end_frame": 19, "confidence": 0.45},
        ],
    )

    assert [segment["segment_id"] for segment in manifest["segments"]] == ["low_conf", "mid_conf", "high_conf"]
    assert manifest["segments"][0]["review_priority"] == 1
    assert manifest["segments"][0]["reason"] == "lowest_confidence_first"
