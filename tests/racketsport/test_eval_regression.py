from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.check_eval_regression import compare_phase_metrics


def _metric(value: object, *, status: str = "measured", gate: str = ">= 0") -> dict[str, object]:
    return {"value": value, "unit": "count", "gate": gate, "passed": True, "status": status}


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


def test_compare_phase_metrics_treats_less_than_gates_as_lower_is_better() -> None:
    baseline = _phase_payload(
        clip_metrics={
            "track_frames": _metric(100, gate="presence_check.track_frames: >= 1"),
            "p95_latency_ms": _metric(10.0, gate="runtime_check.p95_latency_ms: <= 15"),
        }
    )
    current = _phase_payload(
        clip_metrics={
            "track_frames": _metric(100, gate="presence_check.track_frames: >= 1"),
            "p95_latency_ms": _metric(8.0, gate="runtime_check.p95_latency_ms: <= 15"),
        }
    )

    result = compare_phase_metrics(current=current, baseline=baseline)

    assert result.status == "pass"
    assert result.checked_metrics == 2
    assert result.failures == []


def test_compare_phase_metrics_fails_lower_is_better_increase_over_limit() -> None:
    baseline = _phase_payload(clip_metrics={"p95_latency_ms": _metric(10.0, gate="runtime_check.p95_latency_ms: <= 15")})
    current = _phase_payload(clip_metrics={"p95_latency_ms": _metric(14.0, gate="runtime_check.p95_latency_ms: <= 15")})

    result = compare_phase_metrics(current=current, baseline=baseline)

    assert result.status == "fail"
    assert result.failures[0].path == "clips[clip_001].metrics.p95_latency_ms"
    assert result.failures[0].drop_percent == 40.0


def test_compare_phase_metrics_infers_operatorless_error_metrics_are_lower_is_better() -> None:
    baseline = _phase_payload(
        clip_metrics={"ball_p90_error_px": _metric(2.0, gate="label_check.ball_p90_error_px_recorded")}
    )
    current = _phase_payload(
        clip_metrics={"ball_p90_error_px": _metric(4.0, gate="label_check.ball_p90_error_px_recorded")}
    )

    result = compare_phase_metrics(current=current, baseline=baseline)

    assert result.status == "fail"
    assert result.failures[0].path == "clips[clip_001].metrics.ball_p90_error_px"
    assert result.failures[0].drop_percent == 100.0


def test_compare_phase_metrics_keeps_positive_operatorless_rates_higher_is_better() -> None:
    baseline = _phase_payload(
        clip_metrics={"visible_coverage_rate": _metric(0.8, gate="label_check.visible_coverage_rate_recorded")}
    )
    current = _phase_payload(
        clip_metrics={"visible_coverage_rate": _metric(0.85, gate="label_check.visible_coverage_rate_recorded")}
    )

    result = compare_phase_metrics(current=current, baseline=baseline)

    assert result.status == "pass"
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
    assert "checked_artifacts" not in payload
    assert payload["failures"][0]["path"] == "clips[clip_001].metrics.track_frames"


def test_check_eval_regression_cli_discovers_and_pairs_metrics_from_roots(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    current_root = tmp_path / "current"

    baseline_phase1 = baseline_root / "phase1"
    current_phase1 = current_root / "phase1"
    baseline_phase2 = baseline_root / "phase2"
    current_phase2 = current_root / "phase2"
    for path in [baseline_phase1, current_phase1, baseline_phase2, current_phase2]:
        path.mkdir(parents=True)

    (baseline_phase1 / "metrics.json").write_text(
        json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(100)})),
        encoding="utf-8",
    )
    (current_phase1 / "metrics.json").write_text(
        json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(100)})),
        encoding="utf-8",
    )
    (baseline_phase2 / "metrics.json").write_text(
        json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(50)})),
        encoding="utf-8",
    )
    (current_phase2 / "metrics.json").write_text(
        json.dumps(_phase_payload(clip_metrics={"track_frames": _metric(48)})),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/check_eval_regression.py",
            "--current-root",
            str(current_root),
            "--baseline-root",
            str(baseline_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "fail"
    assert payload["checked_artifacts"] == 2
    assert payload["checked_metrics"] == 2
    assert payload["failures"][0]["path"] == "phase2/metrics.json:clips[clip_001].metrics.track_frames"


def test_check_eval_regression_cli_rejects_empty_path_arguments(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_phase_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/check_eval_regression.py",
            "--current",
            "",
            "--baseline",
            str(baseline_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "path must not be empty" in completed.stderr
