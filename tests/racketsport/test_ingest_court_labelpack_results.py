from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.racketsport.ingest_court_labelpack_results import ingest_labelpack
from scripts.racketsport.train_court_model_v2 import load_real_training_rows
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/racketsport/ingest_court_labelpack_results.py"


def test_ingest_court_labelpack_results_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, CLI, "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--owner-omissions-are-invisible" in result.stdout


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    package_root = tmp_path / "court_labelpack3"
    frames = package_root / "frames"
    frames.mkdir(parents=True)
    positive_frame = frames / "positive.jpg"
    unsupported_frame = frames / "unsupported.jpg"
    Image.new("RGB", (640, 360), color=(25, 115, 80)).save(positive_frame)
    Image.new("RGB", (640, 360), color=(80, 50, 35)).save(unsupported_frame)

    positive_source_sha = "a" * 64
    package = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_labelpack3_owner_click_package",
        "label_order": list(PICKLEBALL_COURT_KEYPOINT_NAMES),
        "protocol_exclusions": {"selected_identity_overlap_count": 0},
        "images": [
            {
                "file_name": positive_frame.name,
                "relative_path": "frames/positive.jpg",
                "frame_sha256": _sha256(positive_frame),
                "source_sha256": positive_source_sha,
                "resolution": [640, 360],
                "venue_id": "venue_supported",
                "venue": "Fixture Supported",
                "workspace": "fixture",
                "source_path": "source/supported.jpg",
            },
            {
                "file_name": unsupported_frame.name,
                "relative_path": "frames/unsupported.jpg",
                "frame_sha256": _sha256(unsupported_frame),
                "source_sha256": "b" * 64,
                "resolution": [640, 360],
                "venue_id": "venue_unsupported",
                "venue": "Fixture Unsupported",
                "workspace": "fixture",
                "source_path": "source/unsupported.jpg",
            },
        ],
    }
    results = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diversity_owner_sequential_labels",
        "authority": "owner_reviewed",
        "label_order": list(PICKLEBALL_COURT_KEYPOINT_NAMES),
        "items": {
            positive_frame.name: {
                "status": "in_progress",
                "keypoints": {
                    "near_left_corner": [45, 330],
                    "near_right_corner": [595, 330],
                    "far_right_corner": [455, 55],
                    "far_left_corner": [185, 55],
                    "net_center": [320, 165],
                },
                "skipped_points": {"near_baseline_center": True},
                "exclusion_reasons": [],
            },
            unsupported_frame.name: {
                "status": "excluded",
                "keypoints": {},
                "skipped_points": {},
                "exclusion_reasons": ["bad_angle"],
            },
        },
    }
    package_path = package_root / "package_manifest.json"
    results_path = tmp_path / "owner_results.json"
    prior_path = tmp_path / "prior_manifest.json"
    _write_json(package_path, package)
    _write_json(results_path, results)
    _write_json(prior_path, {"rows": [{"image_sha256": positive_source_sha}]})
    return {
        "package": package,
        "package_path": package_path,
        "results": results,
        "results_path": results_path,
        "prior_path": prior_path,
        "positive_frame": positive_frame,
        "unsupported_frame": unsupported_frame,
    }


def test_owner_omissions_require_confirmation_then_load_as_unsupervised(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "ingested"

    with pytest.raises(ValueError, match="pass --owner-omissions-are-invisible"):
        ingest_labelpack(
            package_manifest_path=fixture["package_path"],
            results_path=fixture["results_path"],
            out=out,
            owner_omissions_are_invisible=False,
            prior_training_manifest=fixture["prior_path"],
        )
    assert not out.exists()

    report = ingest_labelpack(
        package_manifest_path=fixture["package_path"],
        results_path=fixture["results_path"],
        out=out,
        owner_omissions_are_invisible=True,
        prior_training_manifest=fixture["prior_path"],
    )

    assert report["status"] == "READY"
    assert report["counts"]["package_frames"] == 2
    assert report["counts"]["positive_frames"] == 1
    assert report["counts"]["unsupported_view_frames"] == 1
    assert report["counts"]["positive_venue_groups"] == 1
    assert report["counts"]["trainer_clip_groups"] == 1
    assert report["counts"]["labeled_points"] == 5
    assert report["counts"]["explicitly_skipped_points"] == 1
    assert report["counts"]["omitted_points_interpreted_as_invisible"] == 9
    assert report["counts"]["prior_training_exact_source_overlap_positive_frames"] == 1
    assert report["counts"]["new_source_positive_frames"] == 0
    assert report["counts"]["geometry_audited_positive_frames"] == 1
    assert report["status_counts"] == {"excluded": 1, "in_progress": 1}
    assert report["unsupported_reason_counts"] == {"bad_angle": 1}
    assert report["policies"]["missing_points_fabricated"] is False
    assert report["policies"]["unsupported_views_enter_keypoint_training"] is False
    assert json.loads(fixture["results_path"].read_text(encoding="utf-8")) == fixture["results"]

    label_path = out / "train" / "venue_supported" / "labels" / "court_keypoints.json"
    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    item = label_payload["annotation"]["items"][0]
    assert set(item["keypoints"]) == set(PICKLEBALL_COURT_KEYPOINT_NAMES)
    assert item["keypoints"]["net_center"] == [320.0, 165.0]
    assert item["keypoints"]["near_baseline_center"] is None
    assert item["keypoints"]["far_baseline_center"] is None
    assert item["status"] == "reviewed"
    assert item["provenance"]["explicitly_skipped_points"] == ["near_baseline_center"]
    assert item["provenance"]["omitted_points_interpreted_as"]["meaning"].startswith(
        "occluded_or_not_visible"
    )

    rows = load_real_training_rows([out / "train"])
    assert len(rows) == 1
    assert set(rows[0]["keypoints"]) == {
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
        "net_center",
    }
    assert rows[0]["label_source"] == "reviewed_partial_court_keypoint_labels"
    assert rows[0]["label_status"] == "reviewed"
    assert Path(rows[0]["image_path"]).is_file()

    unsupported = json.loads((out / "unsupported_view_manifest.json").read_text(encoding="utf-8"))
    assert unsupported["authority"] == "owner_reviewed"
    assert unsupported["items"] == [
        {
            "file_name": "unsupported.jpg",
            "frame_sha256": _sha256(fixture["unsupported_frame"]),
            "image": "unsupported/venue_unsupported/frames/unsupported.jpg",
            "reasons": ["bad_angle"],
            "source_sha256": "b" * 64,
            "supported_view": False,
            "venue": "Fixture Unsupported",
            "venue_id": "venue_unsupported",
            "workspace": "fixture",
        }
    ]
    assert not any(path.name == "unsupported.jpg" for path in (out / "train").rglob("*.jpg"))


def test_rejects_frame_hash_mismatch_before_materialization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    package = fixture["package"]
    package["images"][0]["frame_sha256"] = "0" * 64
    _write_json(fixture["package_path"], package)

    out = tmp_path / "ingested"
    with pytest.raises(ValueError, match="frame SHA-256 mismatch: positive.jpg"):
        ingest_labelpack(
            package_manifest_path=fixture["package_path"],
            results_path=fixture["results_path"],
            out=out,
            owner_omissions_are_invisible=True,
        )
    assert not out.exists()


def test_rejects_out_of_bounds_coordinate_before_materialization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    results = fixture["results"]
    results["items"]["positive.jpg"]["keypoints"]["near_left_corner"] = [641, 330]
    _write_json(fixture["results_path"], results)

    out = tmp_path / "ingested"
    with pytest.raises(ValueError, match="outside the declared 640x360 image"):
        ingest_labelpack(
            package_manifest_path=fixture["package_path"],
            results_path=fixture["results_path"],
            out=out,
            owner_omissions_are_invisible=True,
        )
    assert not out.exists()
