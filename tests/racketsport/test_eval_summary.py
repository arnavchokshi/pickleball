from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.eval.summary import build_eval_run_summary


def _metric(
    *,
    value: int | float | bool | None,
    passed: bool | None,
    status: str = "measured",
) -> dict:
    return {
        "value": value,
        "unit": None,
        "gate": "test gate",
        "passed": passed,
        "status": status,
    }


def _write_metrics(
    phase_dir: Path,
    *,
    phase: str,
    status: str,
    phase_metric: dict | None = None,
    clip_metrics: list[dict] | None = None,
    clip_statuses: list[str] | None = None,
) -> None:
    clip_metrics = clip_metrics if clip_metrics is not None else [_metric(value=1, passed=True)]
    clip_statuses = clip_statuses if clip_statuses is not None else ["pass"] * len(clip_metrics)
    phase_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "phase": phase,
        "evaluator": f"{phase}_eval",
        "root": str(phase_dir),
        "labels_root": "data/testclips",
        "status": status,
        "required_artifacts": ["artifact.json"],
        "summary": {
            "total_clips": len(clip_statuses),
            "ready_clips": len(clip_statuses),
            "evaluated_clips": sum(1 for clip_status in clip_statuses if clip_status in {"pass", "fail"}),
            "passed_clips": sum(1 for clip_status in clip_statuses if clip_status == "pass"),
            "failed_clips": sum(1 for clip_status in clip_statuses if clip_status == "fail"),
            "blocked_clips": sum(1 for clip_status in clip_statuses if clip_status == "blocked"),
        },
        "metrics": {"artifact_readiness": phase_metric or _metric(value=True, passed=True)},
        "clips": [
            {
                "clip": f"clip_{index:03d}",
                "run_dir": str(phase_dir / f"clip_{index:03d}"),
                "labels_dir": f"data/testclips/clip_{index:03d}/labels",
                "status": clip_status,
                "missing_label_files": [],
                "missing_artifacts": ["artifact.json"] if clip_status == "blocked" else [],
                "metrics": {f"clip_metric_{index}": metric},
                "notes": [],
            }
            for index, (clip_status, metric) in enumerate(zip(clip_statuses, clip_metrics), start=1)
        ],
        "notes": [],
    }
    (phase_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_eval_run_summary_counts_all_pass_metrics_deterministically(tmp_path):
    phase2 = tmp_path / "runs" / "phase2"
    phase1 = tmp_path / "runs" / "phase1"
    _write_metrics(phase2, phase="phase2", status="pass")
    _write_metrics(phase1, phase="phase1", status="pass")

    summary = build_eval_run_summary([phase2, phase1])

    assert summary["artifact_type"] == "racketsport_eval_run_summary"
    assert summary["execution"] == {
        "cpu_only": True,
        "runs_evaluations": False,
        "uses_gpu": False,
        "mutates_metrics": False,
    }
    assert [phase["phase"] for phase in summary["phases"]] == ["phase1", "phase2"]
    assert summary["summary"]["phase_count"] == 2
    assert summary["summary"]["metrics_file_count"] == 2
    assert summary["summary"]["missing_metrics_file_count"] == 0
    assert summary["summary"]["malformed_metrics_file_count"] == 0
    assert summary["summary"]["status_counts"]["pass"] == 2
    assert summary["summary"]["metric_status_counts"] == {"measured": 4, "not_measured": 0}
    assert summary["summary"]["gate_result_counts"] == {"pass": 4, "fail": 0, "not_measured": 0}
    assert summary["highest_risk_phases"][0]["risk_score"] == 0
    assert summary["phases"][0]["metrics_path"].endswith("phase1/metrics.json")


def test_build_eval_run_summary_reports_missing_metrics_file(tmp_path):
    phase1 = tmp_path / "runs" / "phase1"
    phase2 = tmp_path / "runs" / "phase2"
    _write_metrics(phase1, phase="phase1", status="pass")
    phase2.mkdir(parents=True)

    summary = build_eval_run_summary([phase1, phase2])

    assert summary["summary"]["phase_count"] == 2
    assert summary["summary"]["metrics_file_count"] == 1
    assert summary["summary"]["missing_metrics_file_count"] == 1
    assert summary["summary"]["status_counts"]["missing_metrics"] == 1
    assert summary["missing_metrics_files"] == [str(phase2 / "metrics.json")]
    missing_phase = summary["phases"][1]
    assert missing_phase["phase"] == "phase2"
    assert missing_phase["status"] == "missing_metrics"
    assert missing_phase["risk_reasons"] == ["metrics.json missing"]


def test_build_eval_run_summary_counts_mixed_not_measured_and_failed_gates(tmp_path):
    phase1 = tmp_path / "runs" / "phase1"
    phase2 = tmp_path / "runs" / "phase2"
    _write_metrics(
        phase1,
        phase="phase1",
        status="not_measured",
        phase_metric=_metric(value=None, passed=None, status="not_measured"),
        clip_metrics=[_metric(value=None, passed=None, status="not_measured")],
        clip_statuses=["not_measured"],
    )
    _write_metrics(
        phase2,
        phase="phase2",
        status="fail",
        phase_metric=_metric(value=False, passed=False),
        clip_metrics=[_metric(value=0, passed=False)],
        clip_statuses=["fail"],
    )

    summary = build_eval_run_summary([phase1, phase2])

    assert summary["summary"]["status_counts"]["fail"] == 1
    assert summary["summary"]["status_counts"]["not_measured"] == 1
    assert summary["summary"]["metric_status_counts"] == {"measured": 2, "not_measured": 2}
    assert summary["summary"]["gate_result_counts"] == {"pass": 0, "fail": 2, "not_measured": 2}
    assert summary["summary"]["clip_status_counts"]["fail"] == 1
    assert summary["summary"]["clip_status_counts"]["not_measured"] == 1
    assert summary["highest_risk_phases"][0]["phase"] == "phase2"
    assert "phase status fail" in summary["highest_risk_phases"][0]["risk_reasons"]


def test_build_eval_run_summary_rejects_malformed_metrics_by_default(tmp_path):
    phase1 = tmp_path / "runs" / "phase1"
    phase1.mkdir(parents=True)
    (phase1 / "metrics.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse"):
        build_eval_run_summary([phase1])


def test_summarize_eval_runs_cli_writes_summary_json(tmp_path):
    phase1 = tmp_path / "runs" / "phase1"
    out = tmp_path / "summary.json"
    _write_metrics(phase1, phase="phase1", status="pass")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/summarize_eval_runs.py",
            "--phase-dir",
            str(phase1),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["summary"]["status_counts"]["pass"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == 1
