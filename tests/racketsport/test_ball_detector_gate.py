from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_detector_gate import build_ball_detector_gate_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_payload(*, fine_tuned: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "models": [
            {
                "id": "tracknetv3",
                "stage": "ball_tracking",
                "source": "https://github.com/qaz812345/TrackNetV3",
                "license": "MIT",
                "status": "available_on_h100",
                "local_path": "TrackNet_best.pt",
                "sha256": "a" * 64,
                "fine_tuned_on_pickleball": fine_tuned,
                "training_data": ["badminton_tennis_transfer", "roboflow_50k", "self_labeled_2k", "heldout_2k"],
            },
            {
                "id": "tracknetv3_inpaintnet",
                "stage": "ball_tracking",
                "source": "https://github.com/qaz812345/TrackNetV3",
                "license": "MIT",
                "status": "available_on_h100",
                "local_path": "InpaintNet_best.pt",
                "sha256": "b" * 64,
                "fine_tuned_on_pickleball": fine_tuned,
            },
            {
                "id": "wasb_tennis_bmvc2023",
                "stage": "ball_tracking_verifier",
                "source": "https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md",
                "license": "MIT",
                "status": "available_on_h100",
                "local_path": "wasb_tennis_best.pth.tar",
                "sha256": "c" * 64,
                "fine_tuned_on_pickleball": fine_tuned,
            },
        ],
    }


def _ball_track_payload(*, binary_confidence: bool = False) -> dict[str, object]:
    confidences = [1.0, 0.0, 1.0] if binary_confidence else [0.61, 0.22, 0.83]
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": idx / 60.0, "xy": [100.0 + idx, 200.0], "conf": conf, "visible": conf >= 0.5}
            for idx, conf in enumerate(confidences)
        ],
        "bounces": [],
    }


def _benchmark_payload(*, f1: float = 0.92, recall: float = 0.80, hidden_fp: float = 0.03) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_ball_tracker_benchmark",
        "aggregate": {
            "tracknet_finetuned": {
                "micro_label_f1_at_20px": f1,
                "micro_precision_at_10px": 0.93,
                "micro_recall_at_10px": recall,
                "micro_visible_recall_at_20px": recall,
                "micro_hidden_false_positive_rate": hidden_fp,
            }
        },
    }


def _metadata_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_tracknet_ball_run",
        "source_mode": "tracknet_predict",
        "confidence_semantics": "TrackNet heatmap peak value (0..1)",
        "official_repo_url": "https://github.com/qaz812345/TrackNetV3",
    }


def test_ball_detector_gate_passes_official_finetuned_models_heatmap_confidence_and_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    track = tmp_path / "ball_track.json"
    benchmark = tmp_path / "benchmark.json"
    metadata = tmp_path / "metadata.json"
    _write_json(manifest, _manifest_payload())
    _write_json(track, _ball_track_payload())
    _write_json(benchmark, _benchmark_payload())
    _write_json(metadata, _metadata_payload())

    report = build_ball_detector_gate_report(
        model_manifest_path=manifest,
        ball_track_path=track,
        benchmark_path=benchmark,
        metadata_path=metadata,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M1 Offline detector"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["model_manifest"]["tracknetv3"]["source_official"] is True
    assert report["confidence"]["has_non_binary_confidence"] is True
    assert report["metrics"]["best_f1"] == pytest.approx(0.92)
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_detector_gate_fails_closed_for_pretrained_binary_or_low_metric_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    track = tmp_path / "ball_track.json"
    benchmark = tmp_path / "benchmark.json"
    _write_json(manifest, _manifest_payload(fine_tuned=False))
    _write_json(track, _ball_track_payload(binary_confidence=True))
    _write_json(benchmark, _benchmark_payload(f1=0.70, recall=0.65, hidden_fp=0.32))

    report = build_ball_detector_gate_report(
        model_manifest_path=manifest,
        ball_track_path=track,
        benchmark_path=benchmark,
        metadata_path=tmp_path / "missing_metadata.json",
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_detector_gate_failed"
    assert set(report["violations"]) >= {
        "tracknet_not_pickleball_finetuned",
        "inpaintnet_not_pickleball_finetuned",
        "wasb_not_pickleball_finetuned",
        "confidence_values_are_binary_only",
        "missing_detector_run_metadata",
        "detector_f1_below_0_90",
        "detector_recall_below_0_75",
        "detector_hidden_fp_rate_over_0_05",
    }


def test_ball_detector_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    out = tmp_path / "m1_report.json"
    _write_json(benchmark, _benchmark_payload(f1=0.70, recall=0.65, hidden_fp=0.32))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_detector.py",
            "--benchmark",
            str(benchmark),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["gate_result"] == "fail"
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_detector_gate_failed"
