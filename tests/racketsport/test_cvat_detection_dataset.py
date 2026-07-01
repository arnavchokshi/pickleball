from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.cvat_detection_dataset import (
    class_map_for_preset,
    yolo_label_line_from_xywh,
)


def test_class_map_for_preset_uses_stable_detector_classes() -> None:
    assert class_map_for_preset("player") == {"player": 0}
    assert class_map_for_preset("paddle") == {"paddle": 0}
    assert class_map_for_preset("ball") == {"ball": 0}
    assert class_map_for_preset("combined") == {"player": 0, "paddle": 1, "ball": 2}


def test_yolo_label_line_from_xywh_clips_to_image_bounds() -> None:
    line = yolo_label_line_from_xywh(
        bbox_xywh=(-10.0, 20.0, 40.0, 80.0),
        image_width=200,
        image_height=400,
        class_id=2,
    )

    assert line == "2 0.075000 0.150000 0.150000 0.200000"


def test_yolo_label_line_from_xywh_rejects_boxes_outside_image() -> None:
    with pytest.raises(ValueError, match="outside image"):
        yolo_label_line_from_xywh(
            bbox_xywh=(300.0, 20.0, 40.0, 80.0),
            image_width=200,
            image_height=400,
            class_id=0,
        )


def test_export_cvat_detection_yolo_dataset_cli_forwards_options(monkeypatch, tmp_path: Path, capsys) -> None:
    import scripts.racketsport.export_cvat_detection_yolo_dataset as export_cli

    captured: dict[str, object] = {}
    video = tmp_path / "video.mp4"
    reviewed = tmp_path / "reviewed_boxes.json"
    video.write_bytes(b"placeholder")
    reviewed.write_text("{}", encoding="utf-8")

    def fake_export_cvat_detection_yolo_dataset(**kwargs):
        captured.update(kwargs)
        return {"image_count": 10, "label_count": 25, "data_yaml": str(tmp_path / "data.yaml")}

    monkeypatch.setattr(export_cli, "export_cvat_detection_yolo_dataset", fake_export_cvat_detection_yolo_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/export_cvat_detection_yolo_dataset.py",
            "--clip",
            f"clip-a={video}={reviewed}",
            "--preset",
            "combined",
            "--out-dir",
            str(tmp_path / "dataset"),
            "--split-mode",
            "by_clip",
            "--val-clip",
            "clip-a",
            "--frame-stride",
            "3",
        ],
    )

    assert export_cli.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["image_count"] == 10
    assert captured["out_dir"] == tmp_path / "dataset"
    assert captured["class_map"] == {"player": 0, "paddle": 1, "ball": 2}
    assert captured["split_mode"] == "by_clip"
    assert captured["val_clips"] == ("clip-a",)
    assert captured["frame_stride"] == 3


def test_export_cvat_detection_yolo_dataset_cli_reports_errors_without_traceback(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_cvat_detection_yolo_dataset.py",
            "--clip",
            "bad-spec",
            "--preset",
            "ball",
            "--out-dir",
            str(tmp_path / "dataset"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "CVAT detection YOLO dataset export failed:" in completed.stderr
    assert "Traceback" not in completed.stderr
