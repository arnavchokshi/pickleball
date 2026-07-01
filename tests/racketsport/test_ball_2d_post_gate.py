from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_2d_post_gate import build_ball_2d_post_gate_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _m1_report_payload(*, gate_result: str = "pass") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_detector_gate_report",
        "milestone": "M1 Offline detector",
        "status": "TESTED-ON-REAL-DATA",
        "gate_result": gate_result,
        "blocked_reason": None if gate_result == "pass" else "ball_detector_gate_failed",
    }


def _post_summary_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_2d_postprocess_summary",
        "model_consensus": {"primary": "tracknet", "verifier": "wasb", "radius_px_1080p": 60.0},
        "court_gating": {"margin_m": 0.5},
        "max_speed_gate": {"max_world_speed_mps": 30.0},
        "ransac": {"max_residual_px": 5.0},
        "local_search": {"recovery_heatmap_threshold": 0.25},
        "kalman_rts": {"max_gap_fill_frames": 6, "jitter_px_std": 1.4},
    }


def _benchmark_payload(*, f1: float = 0.91, recall: float = 0.80, hidden_fp: float = 0.03) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_ball_tracker_benchmark",
        "aggregate": {
            "m2_full_post": {
                "category": "m2_temporal",
                "micro_label_f1_at_20px": f1,
                "micro_visible_recall_at_20px": recall,
                "micro_hidden_false_positive_rate": hidden_fp,
                "total_teleport_count": 0,
                "mean_max_visible_gap_frames": 5,
                "mean_p95_step_px": 24.0,
            }
        },
    }


def test_ball_2d_post_gate_passes_full_spec_post_summary_and_metrics(tmp_path: Path) -> None:
    m1 = tmp_path / "m1_report.json"
    summary = tmp_path / "m2_post_summary.json"
    benchmark = tmp_path / "m2_benchmark.json"
    _write_json(m1, _m1_report_payload())
    _write_json(summary, _post_summary_payload())
    _write_json(benchmark, _benchmark_payload())

    report = build_ball_2d_post_gate_report(
        m1_detector_report_path=m1,
        postprocess_summary_path=summary,
        benchmark_paths=[benchmark],
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M2 2D post"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["postprocess"]["model_consensus"]["radius_px_1080p"] == pytest.approx(60.0)
    assert report["metrics"]["best_f1"] == pytest.approx(0.91)
    assert report["metrics"]["best_hidden_false_positive_rate"] == pytest.approx(0.03)
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_2d_post_gate_fails_closed_without_m1_pass_summary_or_gate_metrics(tmp_path: Path) -> None:
    m1 = tmp_path / "m1_report.json"
    benchmark = tmp_path / "m2_benchmark.json"
    _write_json(m1, _m1_report_payload(gate_result="fail"))
    _write_json(benchmark, _benchmark_payload(f1=0.62, recall=0.51, hidden_fp=0.25))

    report = build_ball_2d_post_gate_report(
        m1_detector_report_path=m1,
        postprocess_summary_path=tmp_path / "missing_summary.json",
        benchmark_paths=[benchmark],
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_2d_post_gate_failed"
    assert set(report["violations"]) >= {
        "m1_detector_gate_not_passed",
        "missing_postprocess_summary",
        "post_f1_below_0_90",
        "post_recall_below_0_75",
        "post_hidden_fp_rate_over_0_05",
    }


def test_ball_2d_post_gate_names_missing_postprocess_component_evidence(tmp_path: Path) -> None:
    m1 = tmp_path / "m1_report.json"
    summary = tmp_path / "m2_post_summary.json"
    benchmark = tmp_path / "m2_benchmark.json"
    _write_json(m1, _m1_report_payload(gate_result="fail"))
    _write_json(
        summary,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_2d_postprocess_summary",
            "status": "TESTED-ON-REAL-DATA",
            "model_consensus": {
                "evidence_present": True,
                "primary": "tracknet",
                "verifier": "wasb",
                "radius_px_1080p": 60.0,
            },
            "court_gating": {"evidence_present": True, "margin_m": 0.5},
            "max_speed_gate": {"evidence_present": False, "max_world_speed_mps": None},
            "ransac": {"evidence_present": False, "max_residual_px": None},
            "local_search": {"evidence_present": False, "recovery_heatmap_threshold": None},
            "kalman_rts": {
                "evidence_present": False,
                "max_gap_fill_frames": None,
                "jitter_px_std": None,
            },
            "missing_components": ["max_speed_gate", "ransac", "local_search", "kalman_rts"],
            "not_ground_truth": True,
        },
    )
    _write_json(benchmark, _benchmark_payload(f1=0.62, recall=0.51, hidden_fp=0.25))

    report = build_ball_2d_post_gate_report(
        m1_detector_report_path=m1,
        postprocess_summary_path=summary,
        benchmark_paths=[benchmark],
    )

    assert "missing_postprocess_summary" not in report["violations"]
    assert set(report["violations"]) >= {
        "max_speed_gate_evidence_missing",
        "ransac_evidence_missing",
        "local_search_evidence_missing",
        "kalman_rts_evidence_missing",
    }


def test_ball_2d_post_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    benchmark = tmp_path / "m2_benchmark.json"
    out = tmp_path / "m2_report.json"
    _write_json(benchmark, _benchmark_payload(f1=0.62, recall=0.51, hidden_fp=0.25))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_2d_post.py",
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
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_2d_post_gate_failed"
