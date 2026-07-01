from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_validation_gate import build_ball_validation_gate_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _milestone_report(milestone: str, *, gate_result: str = "pass", status: str = "TESTED-ON-REAL-DATA") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": f"racketsport_{milestone.lower()}_gate_report",
        "milestone": milestone,
        "status": status,
        "gate_result": gate_result,
        "blocked_reason": None if gate_result == "pass" else f"{milestone.lower()}_failed",
    }


def _benchmark_payload(candidate: str, *, f1: float = 0.92, recall: float = 0.81, hidden_fp: float = 0.03) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_ball_tracker_benchmark",
        "aggregate": {
            candidate: {
                "micro_label_f1_at_20px": f1,
                "micro_visible_recall_at_20px": recall,
                "micro_hidden_false_positive_rate": hidden_fp,
                "total_visible_label_count": 100,
                "total_hidden_label_count": 50,
            }
        },
    }


def _wasb_track_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "wasb",
        "frames": [{"t": 0.0, "xy": [100.0, 200.0], "conf": 0.8, "visible": True}],
        "bounces": [],
    }


def _wasb_metadata_payload(*, out: str, predictions_csv: str, source_mode: str = "wasb_predict") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_wasb_ball_run",
        "status": "TESTED-ON-REAL-DATA",
        "source_mode": source_mode,
        "predictions_csv": predictions_csv,
        "out": out,
        "fps": 60.0,
        "frame_count": 1,
        "visible_frame_count": 1,
        "confidence_semantics": "WASB heatmap peak value (0..1)",
        "visible_threshold": 0.5,
        "not_ground_truth": True,
        "official_repo_url": "https://github.com/nttcom/WASB-SBDT",
        "official_model_zoo_url": "https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md",
        "runtime": {
            "device": "cuda",
            "effective_fps": 32.0,
            "processed_frame_count": 1,
            "processed_window_count": 1,
            "read_frame_count": 3,
            "video": "cvat_upload/clip.mp4",
            "wasb_checkpoint": {
                "path": "models/checkpoints/wasb/wasb_tennis_best.pth.tar",
                "sha256": "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb",
            },
            "wasb_repo": "third_party/WASB-SBDT",
            "wasb_repo_commit": "923462cacdeb3353b84ddebdedb3f4b7a8553b0f",
        },
    }


def _eval_suite_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracking_eval_suite",
        "status": "scored_not_gate_verified",
        "clip_count": 4,
        "benchmark": {
            "artifact_type": "racketsport_ball_tracker_benchmark",
            "aggregate": {
                "tracknet_wasb_fusion": {
                    "category": "generalizable",
                    "micro_visible_hit_recall": 0.91,
                    "micro_hidden_false_positive_rate": 0.02,
                }
            },
        },
    }


def test_ball_validation_gate_passes_when_all_milestones_metrics_wasb_and_suite_pass(tmp_path: Path) -> None:
    milestone_paths: dict[str, Path] = {}
    for key in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        path = tmp_path / f"{key.lower()}_report.json"
        _write_json(path, _milestone_report(key))
        milestone_paths[key] = path
    tracknet_benchmark = tmp_path / "tracknet_benchmark.json"
    wasb_benchmark = tmp_path / "wasb_benchmark.json"
    wasb_track = tmp_path / "wasb_ball_track.json"
    wasb_metadata = tmp_path / "wasb_metadata.json"
    wasb_predictions = tmp_path / "wasb_predictions.csv"
    eval_suite = tmp_path / "eval_suite_summary.json"
    _write_json(tracknet_benchmark, _benchmark_payload("tracknet"))
    _write_json(wasb_benchmark, _benchmark_payload("wasb", f1=0.91, recall=0.80, hidden_fp=0.04))
    _write_json(wasb_track, _wasb_track_payload())
    wasb_predictions.write_text("Frame,Visibility,X,Y,Confidence\n0,1,100,200,0.8\n", encoding="utf-8")
    _write_json(wasb_metadata, _wasb_metadata_payload(out=str(wasb_track), predictions_csv=str(wasb_predictions)))
    _write_json(eval_suite, _eval_suite_payload())

    report = build_ball_validation_gate_report(
        milestone_report_paths=milestone_paths,
        tracknet_benchmark_path=tracknet_benchmark,
        wasb_track_path=wasb_track,
        wasb_metadata_path=wasb_metadata,
        wasb_benchmark_path=wasb_benchmark,
        eval_suite_path=eval_suite,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M8 Verifier + validation"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["milestones"]["pass_count"] == 8
    assert report["offline_detector"]["best_f1"] == pytest.approx(0.92)
    assert report["wasb_verifier"]["track_source"] == "wasb"
    assert report["wasb_verifier"]["metadata"]["source_mode"] == "wasb_predict"
    assert report["wasb_verifier"]["metadata"]["checkpoint_sha256"] == "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
    assert report["wasb_verifier"]["run_evidence_valid"] is True
    assert report["eval_suite"]["has_wasb_fusion_candidate"] is True
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_validation_gate_rejects_wasb_track_without_official_runtime_metadata(tmp_path: Path) -> None:
    tracknet_benchmark = tmp_path / "tracknet_benchmark.json"
    wasb_benchmark = tmp_path / "wasb_benchmark.json"
    wasb_track = tmp_path / "wasb_ball_track.json"
    wasb_metadata = tmp_path / "wasb_metadata.json"
    wasb_predictions = tmp_path / "wasb_predictions.csv"
    eval_suite = tmp_path / "eval_suite_summary.json"
    _write_json(tracknet_benchmark, _benchmark_payload("tracknet"))
    _write_json(wasb_benchmark, _benchmark_payload("wasb"))
    _write_json(wasb_track, _wasb_track_payload())
    wasb_predictions.write_text("Frame,Visibility,X,Y,Confidence\n0,1,100,200,0.8\n", encoding="utf-8")
    metadata = _wasb_metadata_payload(
        out=str(tmp_path / "different_ball_track.json"),
        predictions_csv=str(wasb_predictions),
        source_mode="wasb_csv",
    )
    metadata["runtime"] = {}
    _write_json(wasb_metadata, metadata)
    _write_json(eval_suite, _eval_suite_payload())

    report = build_ball_validation_gate_report(
        milestone_report_paths={},
        tracknet_benchmark_path=tracknet_benchmark,
        wasb_track_path=wasb_track,
        wasb_metadata_path=wasb_metadata,
        wasb_benchmark_path=wasb_benchmark,
        eval_suite_path=eval_suite,
    )

    assert report["status"] == "SCAFFOLD"
    assert report["wasb_verifier"]["run_evidence_valid"] is False
    assert set(report["violations"]) >= {
        "wasb_metadata_source_mode_not_predict",
        "wasb_metadata_out_mismatch",
        "wasb_runtime_device_not_cuda",
        "wasb_runtime_checkpoint_sha256_not_official",
        "wasb_runtime_repo_commit_not_official",
    }


def test_ball_validation_gate_fails_closed_for_missing_reports_bad_metrics_and_missing_wasb_fusion(tmp_path: Path) -> None:
    tracknet_benchmark = tmp_path / "tracknet_benchmark.json"
    wasb_track = tmp_path / "wasb_ball_track.json"
    eval_suite = tmp_path / "eval_suite_summary.json"
    _write_json(tracknet_benchmark, _benchmark_payload("tracknet", f1=0.70, recall=0.65, hidden_fp=0.32))
    _write_json(wasb_track, {**_wasb_track_payload(), "source": "tracknet"})
    _write_json(
        eval_suite,
        {
            **_eval_suite_payload(),
            "clip_count": 4,
            "benchmark": {"aggregate": {"tracknet_only": {"micro_visible_hit_recall": 0.70}}},
        },
    )

    report = build_ball_validation_gate_report(
        milestone_report_paths={"M0": tmp_path / "missing_m0.json"},
        tracknet_benchmark_path=tracknet_benchmark,
        wasb_track_path=wasb_track,
        wasb_benchmark_path=tmp_path / "missing_wasb_benchmark.json",
        eval_suite_path=eval_suite,
    )

    assert report["status"] == "SCAFFOLD"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_validation_gate_failed"
    assert set(report["violations"]) >= {
        "missing_milestone_report:M0",
        "missing_milestone_report:M1",
        "offline_detector_f1_below_0_90",
        "offline_detector_recall_below_0_75",
        "offline_detector_hidden_fp_rate_over_0_05",
        "wasb_track_source_invalid",
        "missing_wasb_benchmark",
        "missing_wasb_fusion_candidate",
    }


def test_ball_validation_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    tracknet_benchmark = tmp_path / "tracknet_benchmark.json"
    out = tmp_path / "m8_report.json"
    _write_json(tracknet_benchmark, _benchmark_payload("tracknet", f1=0.70, recall=0.65, hidden_fp=0.32))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_validation.py",
            "--tracknet-benchmark",
            str(tracknet_benchmark),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["gate_result"] == "fail"
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_validation_gate_failed"
