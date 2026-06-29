from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from threed.racketsport.person_mot import import_mot_zip, write_person_ground_truth
from threed.racketsport.schemas import PersonGroundTruth, validate_artifact_file


def _write_mot_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("gt/labels.txt", "player\nspectator\n")
        archive.writestr(
            "gt/gt.txt",
            "\n".join(
                [
                    "1,1,10,20,30,40,1,1,0.90",
                    "1,2,50,60,20,30,1,1,1.00",
                    "2,1,11,21,30,40,1,1,0.80",
                    "2,99,100,100,10,10,0,2,1.00",
                ]
            )
            + "\n",
        )


def _write_mot_zip_with_gt(path: Path, rows: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("gt/labels.txt", "player\n")
        archive.writestr("gt/gt.txt", "\n".join(rows) + "\n")


def test_import_mot_zip_normalizes_cvat_tracks_to_person_ground_truth(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_mot.zip"
    _write_mot_zip(zip_path)

    ground_truth = import_mot_zip(zip_path, clip_id="clip_a", fps=30.0)

    assert ground_truth.artifact_type == "racketsport_person_ground_truth"
    assert ground_truth.clip_id == "clip_a"
    assert ground_truth.fps == pytest.approx(30.0)
    assert ground_truth.summary.frame_count == 2
    assert ground_truth.summary.valid_label_count == 3
    assert ground_truth.summary.ignored_label_count == 1
    assert ground_truth.frames[0].frame_index == 0
    assert ground_truth.frames[0].source_frame_id == 1
    assert ground_truth.frames[0].labels[0].class_name == "player"
    assert ground_truth.frames[0].labels[0].person_class is True
    assert ground_truth.frames[0].labels[0].bbox_xywh == pytest.approx((10.0, 20.0, 30.0, 40.0))
    assert ground_truth.frames[1].labels[1].ignored is True
    assert ground_truth.frames[1].labels[1].person_class is False


def test_import_mot_zip_rejects_fractional_frame_track_and_class_ids(tmp_path: Path) -> None:
    for field_index, message in [(0, "frame"), (1, "track_id"), (7, "class_id")]:
        row = ["1", "1", "10", "20", "30", "40", "1", "1", "0.90"]
        row[field_index] = "1.9"
        zip_path = tmp_path / f"fractional_{field_index}.zip"
        _write_mot_zip_with_gt(zip_path, [",".join(row)])

        with pytest.raises(ValueError, match=f"{message} must be an integer"):
            import_mot_zip(zip_path)


def test_write_person_ground_truth_registers_schema(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_mot.zip"
    out_path = tmp_path / "person_ground_truth.json"
    _write_mot_zip(zip_path)
    ground_truth = import_mot_zip(zip_path, clip_id="clip_a")

    write_person_ground_truth(out_path, ground_truth)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_person_ground_truth"
    parsed = validate_artifact_file("person_ground_truth", out_path)
    assert isinstance(parsed, PersonGroundTruth)
    assert parsed.clip_id == "clip_a"


def test_import_person_mot_cli_writes_normalized_artifact(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_mot.zip"
    out_path = tmp_path / "labels" / "person_ground_truth.json"
    _write_mot_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/import_person_mot.py",
            "--mot-zip",
            str(zip_path),
            "--clip-id",
            "clip_cli",
            "--fps",
            "30",
            "--out",
            str(out_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(out_path) in result.stdout
    parsed = validate_artifact_file("person_ground_truth", out_path)
    assert isinstance(parsed, PersonGroundTruth)
    assert parsed.clip_id == "clip_cli"
    assert parsed.summary.valid_label_count == 3
