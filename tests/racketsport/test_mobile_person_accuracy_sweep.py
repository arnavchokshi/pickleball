from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from scripts.racketsport.run_mobile_person_accuracy_sweep import (
    _aggregate_rows,
    _candidate_name,
    _parse_clips,
    _parse_models,
    _write_outputs,
    main as run_sweep_main,
)


def _row(candidate: str, clip: str, *, idf1: float, mota: float = 0.5, coverage: float = 0.5) -> dict:
    return {
        "candidate": candidate,
        "clip_id": clip,
        "model": "yolo26n",
        "model_path": "models/checkpoints/yolo26n.pt",
        "imgsz": 416,
        "conf": 0.1,
        "iou": 0.6,
        "tracker": "predict_iou",
        "prune_mode": "confidence",
        "court_margin_m": None,
        "bbox_expand": 1.0,
        "idf1": idf1,
        "mota": mota,
        "precision": 0.7,
        "recall": 0.8,
        "expected_player_coverage": coverage,
        "id_switches": 1,
        "false_positives": 2,
        "false_negatives": 3,
        "matches": 4,
        "gt_detections": 5,
        "pred_detections": 6,
        "processed_fps": 30.0,
        "p50_latency_ms": 5.0,
        "p95_latency_ms": 9.0,
        "tracks_path": f"runs/{clip}/{candidate}/on_device_person_tracks.json",
        "timing_path": f"runs/{clip}/{candidate}/timing.json",
        "metrics_path": f"runs/{clip}/{candidate}/metrics.json",
        "overlay_path": None,
        "cached_detection_runtime": True,
    }


def test_parse_clips_and_models_require_existing_inputs(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    gt = tmp_path / "person_ground_truth.json"
    calibration = tmp_path / "court_calibration.json"
    model = tmp_path / "yolo26n.pt"
    for path in (video, gt, calibration, model):
        path.write_bytes(b"stub")

    clips = _parse_clips([f"clip_a={video}={gt}={calibration}"])
    models = _parse_models([f"named-model={model}"])

    assert clips[0].clip_id == "clip_a"
    assert clips[0].video_path == video
    assert clips[0].ground_truth_path == gt
    assert clips[0].court_calibration_path == calibration
    assert models[0].name == "named_model"
    assert models[0].path == model
    assert _candidate_name("YOLO 26N", 416, 0.1, 0.6, "predict-iou") == "yolo_26n_img416_conf010_iou060_predict_iou"

    with pytest.raises(FileNotFoundError, match="missing video"):
        _parse_clips([f"clip_b={tmp_path / 'missing.mp4'}={gt}"])


def test_accuracy_sweep_aggregates_leaderboard_and_writes_review_outputs(tmp_path: Path) -> None:
    rows = [
        _row("candidate_a", "clip_1", idf1=0.7, mota=0.6, coverage=0.8),
        _row("candidate_a", "clip_2", idf1=0.9, mota=0.8, coverage=0.9),
        _row("candidate_b", "clip_1", idf1=0.6, mota=0.7, coverage=0.7),
    ]
    clips = _parse_clips([])
    leaderboard = _aggregate_rows(rows)

    assert leaderboard[0]["candidate"] == "candidate_a"
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["mean_idf1"] == pytest.approx(0.8)
    assert leaderboard[0]["worst_idf1"] == pytest.approx(0.7)
    assert leaderboard[0]["clip_count"] == 2
    assert leaderboard[0]["cached_detection_runtime"] is True

    _write_outputs(
        tmp_path,
        clips=clips,
        rows=rows,
        leaderboard=leaderboard,
        failures=[{"candidate": "bad", "clip_id": "clip_1", "error": "boom"}],
        args={"out_dir": tmp_path, "force": False},
    )

    summary = json.loads((tmp_path / "sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_type"] == "racketsport_mobile_person_accuracy_sweep"
    assert summary["row_count"] == 3
    assert summary["candidate_count"] == 2
    assert summary["failure_count"] == 1
    assert (tmp_path / "leaderboard.csv").is_file()
    assert "candidate_a" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_accuracy_sweep_partial_failures_are_nonzero_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = _stub_main_dependencies(tmp_path, monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/run_mobile_person_accuracy_sweep.py",
            "--out-dir",
            str(out_dir),
            "--clip",
            f"clip_1={tmp_path / 'clip.mp4'}={tmp_path / 'person_ground_truth.json'}",
            "--model",
            f"fake={tmp_path / 'fake.pt'}",
            "--track-model",
            f"fake={tmp_path / 'fake.pt'}",
            "--tracker",
            "ok",
            "--tracker",
            "boom",
            "--imgsz",
            "416",
            "--conf",
            "0.1",
            "--skip-ultralytics-track",
        ],
    )

    assert run_sweep_main() == 2
    summary = json.loads((out_dir / "sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    assert summary["row_count"] == 1
    assert summary["failure_count"] == 1


def test_accuracy_sweep_allow_partial_keeps_partial_status_but_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _stub_main_dependencies(tmp_path, monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/run_mobile_person_accuracy_sweep.py",
            "--out-dir",
            str(out_dir),
            "--clip",
            f"clip_1={tmp_path / 'clip.mp4'}={tmp_path / 'person_ground_truth.json'}",
            "--model",
            f"fake={tmp_path / 'fake.pt'}",
            "--track-model",
            f"fake={tmp_path / 'fake.pt'}",
            "--tracker",
            "ok",
            "--tracker",
            "boom",
            "--imgsz",
            "416",
            "--conf",
            "0.1",
            "--skip-ultralytics-track",
            "--allow-partial",
        ],
    )

    assert run_sweep_main() == 0
    summary = json.loads((out_dir / "sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    assert summary["failure_count"] == 1


def _stub_main_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "clip.mp4").write_bytes(b"video")
    (tmp_path / "person_ground_truth.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fake.pt").write_bytes(b"model")
    out_dir = tmp_path / "sweep"

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    def fake_load_or_collect_detections(**kwargs: object) -> dict[str, object]:
        return {"frames": []}

    def fake_score_cached_linker(**kwargs: object) -> dict:
        clip = kwargs["clip"]
        candidate_name = str(kwargs["candidate_name"])
        linker_name = str(kwargs["linker_name"])
        if linker_name == "boom":
            raise RuntimeError("simulated candidate failure")
        return _row(candidate_name, clip.clip_id, idf1=0.75)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "scripts.racketsport.run_mobile_person_accuracy_sweep._load_or_collect_detections",
        fake_load_or_collect_detections,
    )
    monkeypatch.setattr(
        "scripts.racketsport.run_mobile_person_accuracy_sweep._score_cached_linker",
        fake_score_cached_linker,
    )
    monkeypatch.setattr("scripts.racketsport.run_mobile_person_accuracy_sweep._render_top_overlays", lambda *_, **__: None)
    return out_dir
