from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.racketsport.export_person_yolo_dataset as export_cli


def test_export_person_yolo_dataset_cli_forwards_options(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_export_yolo_dataset(**kwargs):
        captured.update(kwargs)
        return {"image_count": 12, "label_count": 48, "data_yaml": str(tmp_path / "data.yaml")}

    monkeypatch.setattr(export_cli, "export_yolo_dataset", fake_export_yolo_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/export_person_yolo_dataset.py",
            "--clip",
            "clip-a=video.mp4=gt.json",
            "--out-dir",
            str(tmp_path / "dataset"),
            "--split-mode",
            "alternating",
            "--val-every",
            "5",
            "--frame-stride",
            "2",
        ],
    )

    assert export_cli.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["image_count"] == 12
    assert captured["out_dir"] == tmp_path / "dataset"
    assert captured["split_mode"] == "alternating"
    assert captured["val_every"] == 5
    assert captured["frame_stride"] == 2


def test_export_person_yolo_dataset_cli_reports_export_errors_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_export_yolo_dataset(**_kwargs):
        raise RuntimeError("missing OpenCV")

    monkeypatch.setattr(export_cli, "export_yolo_dataset", fake_export_yolo_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/export_person_yolo_dataset.py",
            "--clip",
            "clip-a=clip.mp4=gt.json",
            "--out-dir",
            str(tmp_path / "dataset"),
        ],
    )

    assert export_cli.main() == 1

    captured = capsys.readouterr()
    assert "person YOLO dataset export failed: missing OpenCV" in captured.err
    assert "Traceback" not in captured.err
