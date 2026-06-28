from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.court_corner_review import build_calibration_from_corrections
from threed.racketsport.schemas import CourtCalibration, CourtLineEvidence, validate_artifact_file


def test_build_calibration_from_corrections_reorders_reviewed_corners(tmp_path: Path) -> None:
    clip = "clip_001"
    drafts_root = tmp_path / "runs" / "prototype_gate"
    corrections_root = tmp_path / "review_bundle" / "corrections"
    images_root = tmp_path / "review_bundle" / "images"
    out_root = tmp_path / "runs" / "calibrated"
    _write_draft_manifest(drafts_root, clip)
    _write_jpeg(images_root / clip / "frame_000001.jpg", width=1920, height=1080)
    _write_correction(
        corrections_root / clip / "court_corners.json",
        clip=clip,
        corners={
            "far_left": [200.0, 160.0],
            "far_right": [1720.0, 160.0],
            "near_right": [1820.0, 980.0],
            "near_left": [100.0, 980.0],
        },
    )

    summary = build_calibration_from_corrections(
        drafts_root=drafts_root,
        corrections_root=corrections_root,
        frames_root=images_root,
        out_root=out_root,
    )

    assert summary["status"] == "corrected_unverified"
    assert summary["calibrated_clip_count"] == 1
    sidecar = json.loads((out_root / clip / "capture_sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["manual_court_taps"] == [
        [100.0, 980.0],
        [1820.0, 980.0],
        [1720.0, 160.0],
        [200.0, 160.0],
    ]
    assert sidecar["resolution"] == [1920, 1080]
    calibration = validate_artifact_file("court_calibration", out_root / clip / "court_calibration.json")
    assert isinstance(calibration, CourtCalibration)
    assert calibration.reprojection_error_px.median == pytest.approx(0.0)
    assert validate_artifact_file("court_zones", out_root / clip / "court_zones.json")
    assert validate_artifact_file("net_plane", out_root / clip / "net_plane.json")
    evidence = validate_artifact_file("court_line_evidence", out_root / clip / "court_line_evidence.json")
    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.aggregate.auto_calibration_ready is False


def test_build_calibration_from_corrections_fails_closed_on_empty_template(tmp_path: Path) -> None:
    clip = "clip_001"
    corrections_root = tmp_path / "review_bundle" / "corrections"
    correction_path = corrections_root / clip / "court_corners.json"
    correction_path.parent.mkdir(parents=True)
    correction_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": clip,
                "target_file": "court_corners.json",
                "items": [],
            }
        ),
        encoding="utf-8",
    )

    summary = build_calibration_from_corrections(
        drafts_root=tmp_path / "runs" / "prototype_gate",
        corrections_root=corrections_root,
        frames_root=tmp_path / "review_bundle" / "images",
        out_root=tmp_path / "runs" / "calibrated",
    )

    assert summary["status"] == "blocked"
    assert summary["calibrated_clip_count"] == 0
    assert summary["clips"][0]["status"] == "blocked"
    assert "no reviewed court_corners item" in summary["clips"][0]["notes"][0]
    assert not (tmp_path / "runs" / "calibrated" / clip / "capture_sidecar.json").exists()


def test_build_calibration_from_corrections_cli_writes_summary(tmp_path: Path) -> None:
    clip = "clip_001"
    drafts_root = tmp_path / "runs" / "prototype_gate"
    corrections_root = tmp_path / "review_bundle" / "corrections"
    images_root = tmp_path / "review_bundle" / "images"
    out_root = tmp_path / "runs" / "calibrated"
    _write_draft_manifest(drafts_root, clip)
    _write_jpeg(images_root / clip / "frame_000001.jpg", width=1280, height=720)
    _write_correction(
        corrections_root / clip / "court_corners.json",
        clip=clip,
        corners={
            "far_left": [280.0, 120.0],
            "far_right": [1000.0, 120.0],
            "near_right": [1160.0, 650.0],
            "near_left": [120.0, 650.0],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_calibration_from_review.py",
            "--drafts-root",
            str(drafts_root),
            "--corrections-root",
            str(corrections_root),
            "--frames-root",
            str(images_root),
            "--out-root",
            str(out_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "corrected_unverified"
    assert (out_root / "court_corner_calibration_summary.json").is_file()


def _write_draft_manifest(drafts_root: Path, clip: str) -> None:
    labels = drafts_root / clip / "labels"
    labels.mkdir(parents=True)
    (labels / "prototype_autolabel_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": {"name": clip, "metadata": {"frame_rate_fps": 60}},
            }
        ),
        encoding="utf-8",
    )


def _write_correction(path: Path, *, clip: str, corners: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": clip,
                "target_file": "court_corners.json",
                "items": [
                    {
                        "review_id": "court_corners_manual_seed",
                        "frame": "frame_000001.jpg",
                        "source": "human_review",
                        "court_corners": corners,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_jpeg(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\xff\xd8"
        + b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
    )
