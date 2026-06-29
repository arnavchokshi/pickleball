from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.racketsport.run_person_detector_oracle as oracle_cli


def test_person_detector_oracle_cli_writes_report(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run_detector_oracle(**kwargs):
        captured.update(kwargs)
        return [
            {
                "clip_id": "clip-a",
                "candidate": "model_full",
                "model": "model",
                "mode": "full",
                "imgsz": 640,
                "conf": 0.1,
                "wall_time_s": 1.0,
                "processed_fps": 30.0,
                "avg_detections_per_frame": 4.0,
                "L4_iou_0.50": 0.75,
                "L8_iou_0.50": 1.0,
                "metrics_path": str(tmp_path / "metrics.json"),
            }
        ]

    monkeypatch.setattr(oracle_cli, "run_detector_oracle", fake_run_detector_oracle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/run_person_detector_oracle.py",
            "--out-dir",
            str(tmp_path / "oracle"),
            "--clip",
            "clip-a=clip.mp4=gt.json",
            "--model",
            "model=models/model.pt",
            "--mode",
            "full",
            "--imgsz",
            "640",
            "--conf",
            "0.1",
            "--candidate-limit",
            "4",
            "--candidate-limit",
            "8",
            "--oracle-iou",
            "0.5",
        ],
    )

    assert oracle_cli.main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["row_count"] == 1
    assert Path(summary["out_dir"]) == tmp_path / "oracle"
    assert captured["out_dir"] == tmp_path / "oracle"
    assert captured["candidate_limits"] == (4, 8)
    assert captured["oracle_ious"] == (0.5,)
    assert (tmp_path / "oracle" / "detector_oracle.csv").is_file()
    assert (tmp_path / "oracle" / "REPORT.md").is_file()
