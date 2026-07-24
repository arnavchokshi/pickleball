from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.racketsport import court_diversity_label_server as labeler


CLI_PATH = "scripts/racketsport/court_diversity_label_server.py"


def _write_fixture(root: Path) -> None:
    package_root = root / labeler.PACKAGE_ROOT
    package_root.mkdir(parents=True)
    images = [
        {"file_name": "frame0.png", "resolution": [100, 80], "source_id": "source-a", "title": "A"},
        {"file_name": "frame1.png", "resolution": [100, 80], "source_id": "source-b", "title": "B"},
    ]
    (package_root / "package_manifest.json").write_text(
        json.dumps(
            {
                "images": images,
                "shards": [
                    {
                        "shard_name": "shard1",
                        "task_name": "shard1",
                        "file_names": ["frame0.png", "frame1.png"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (package_root / "import_report_20260712_courtsession.json").write_text(
        json.dumps({"tasks": [{"task_id": 88, "task_name": "shard1"}]}),
        encoding="utf-8",
    )
    (package_root / "owner_camera_policy_and_exclusions.json").write_text(
        json.dumps(
            {
                "product_input_policy": {},
                "tasks": {
                    "88": {
                        "task_name": "shard1",
                        "excluded_frames": [{"frame": 1, "reasons": ["fisheye"]}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    suggestions = {
        "frame0.png": {
            "keypoints_px": {
                name: {"xy": [4.0, 5.0]} for name in labeler.LABEL_ORDER
            }
        }
    }
    (package_root / "model_estimated_suggestions.json").write_text(json.dumps(suggestions), encoding="utf-8")


def _write_cvat(path: Path) -> None:
    points = "".join(
        f'<points label="{name}" points="7,9" />' for name in labeler.LABEL_ORDER
    )
    xml = f'<annotations><image id="0" name="frame0.png" width="100" height="80">{points}</image></annotations>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations.xml", xml)


def test_prefill_preserves_cvat_points_and_owner_exclusion(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cvat = tmp_path / "job.zip"
    _write_cvat(cvat)

    progress = labeler.build_initial_progress(tmp_path, cvat_export=cvat)

    assert progress["items"]["frame0.png"]["status"] == "reviewed"
    assert len(progress["items"]["frame0.png"]["keypoints"]) == 15
    assert progress["items"]["frame1.png"]["status"] == "excluded"
    assert progress["items"]["frame1.png"]["exclusion_reasons"] == ["fisheye"]
    assert Path(CLI_PATH).name == "court_diversity_label_server.py"


def test_validation_rejects_out_of_bounds_points(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    payload = {
        "items": {
            "frame0.png": {
                "status": "in_progress",
                "keypoints": {"far_left_corner": [101, 2]},
                "skipped_points": {},
                "exclusion_reasons": [],
            }
        }
    }

    with pytest.raises(ValueError, match="out-of-bounds"):
        labeler.validate_progress(tmp_path, payload)


def test_diagnostic_counts_strict_rows_and_pixel_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    progress = {
        "items": {
            "frame0.png": {
                "status": "reviewed",
                "keypoints": {name: [7.0, 9.0] for name in labeler.LABEL_ORDER},
                "skipped_points": {},
                "exclusion_reasons": [],
            },
            "frame1.png": {
                "status": "excluded",
                "keypoints": {},
                "skipped_points": {},
                "exclusion_reasons": ["fisheye"],
            },
        }
    }

    result = labeler.score_progress(tmp_path, progress)

    assert result["human_labeled_frames"] == 1
    assert result["strict_ingest_eligible_frames"] == 1
    assert result["excluded_frames"] == 1
    assert result["compared_point_count"] == 15
    assert result["median_error_px"] == 5.0
    assert result["pck_at_5px"] == 1.0


def test_export_reconciles_every_frame_and_keeps_exclusions_shape_free(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    progress = {
        "saved_at": "2026-07-23T04:43:55+00:00",
        "items": {
            "frame0.png": {
                "status": "reviewed",
                "keypoints": {name: [7.0, 9.0] for name in labeler.LABEL_ORDER},
                "skipped_points": {},
                "exclusion_reasons": [],
            },
            "frame1.png": {
                "status": "excluded",
                "keypoints": {},
                "skipped_points": {},
                "exclusion_reasons": ["fisheye"],
            },
        },
    }

    report = labeler.export_progress_to_cvat(tmp_path, progress, tmp_path / "exports")

    export_path = tmp_path / "exports/shard1_annotations.zip"
    assert export_path.is_file()
    with zipfile.ZipFile(export_path) as archive:
        root = labeler.ElementTree.fromstring(archive.read("annotations.xml"))
    images = root.findall("image")
    assert [image.attrib["name"] for image in images] == ["frame0.png", "frame1.png"]
    assert len(images[0].findall("points")) == 15
    assert images[1].findall("points") == []
    assert report["reviewed_frame_count"] == 1
    assert report["excluded_frame_count"] == 1
    assert report["point_count"] == 15
