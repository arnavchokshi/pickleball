from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.check_eval_regression import compare_phase_metrics


def _metric(value: object, *, status: str = "measured") -> dict[str, object]:
    return {"value": value, "unit": "count", "gate": ">= 0", "passed": True, "status": status}


def _phase_payload(*, top_metrics: dict[str, object] | None = None, clip_metrics: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase2",
        "evaluator": "track_eval",
        "root": "runs/phase2",
        "labels_root": "data/testclips",
        "status": "pass",
        "required_artifacts": ["tracks.json"],
        "summary": {
            "total_clips": 1,
            "ready_clips": 1,
            "evaluated_clips": 1,
            "passed_clips": 1,
            "failed_clips": 0,
            "blocked_clips": 0,
        },
        "metrics": top_metrics or {},
        "clips": [
            {
                "clip": "clip_001",
                "run_dir": "runs/phase2/clip_001",
                "labels_dir": "data/testclips/clip_001",
                "status": "pass",
                "missing_label_files": [],
                "missing_artifacts": [],
                "metrics": clip_metrics or {},
                "notes": [],
            }
        ],
        "notes": [],
    }


def test_compare_phase_metrics_fails_numeric_measured_drops_over_default_limit() -> None:
    baseline = _phase_payload(clip_metrics={"track_frames": _metric(100), "players_detected": _metric(2)})
    current = _phase_payload(clip_metrics={"track_frames": _metric(97.9), "players_detected": _metric(2)})

    result = compare_phase_metrics(current=current, baseline=baseline)

    assert result.status == "fail"
    assert len(result.failures) == 1
    assert result.failures[0].path == "clips[clip_001].metrics.track_frames"
    assert result.failures[0].baseline == 100
    assert result.failures[0].current == 97.9
    assert result.failures[0].drop_percent > 2.0


def test_compare_phase_metrics_allows_configurable_drop_limit_and_ignores_unmeasured_values() -> None:
    baseline = _phase_payload(
        top_metrics={"artifact_readiness": _metric(True)},
        clip_metrics={
            "track_frames": _metric(100),
            "coverage": _metric(None, status="not_measured"),
            "mode": _metric("fast"),
        },
    )
    current = _phase_payload(
        top_metrics={"artifact_readiness": _metric(False)},
        clip_metrics={
            "track_frames": _metric(97.9),
            "coverage": _metric(0.5),
            "mode": _metric("slow"),
        },
    )

    result = compare_phase_metrics(current=current, baseline=baseline, max_drop_percent=3.0)

    assert result.status == "pass"
    assert result.checked_metrics == 1
    assert result.failures == []


def test_check_eval_regression_cli_exits_nonzero_and_prints_failures(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(50)})), encoding="utf-8")
    current_path.write_text(json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(48)})), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/check_eval_regression.py",
            "--current",
            str(current_path),
            "--baseline",
            str(baseline_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "fail"
    assert payload["max_drop_percent"] == 2.0
    assert payload["failures"][0]["path"] == "clips[clip_001].metrics.track_frames"
