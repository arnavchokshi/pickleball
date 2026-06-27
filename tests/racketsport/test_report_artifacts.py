from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import threed.racketsport.report_model as report_model
from threed.racketsport.schemas import HabitReport, validate_artifact_file


def _metrics_payload() -> dict:
    return {
        "schema_version": 1,
        "players": [
            {
                "id": 1,
                "shots": [
                    {
                        "t": 0.25,
                        "type": "dink",
                        "type_conf": 0.82,
                        "metrics": {
                            "nvz_margin_ft": {"value": -0.5, "conf": 0.86, "gated": False, "frames": 5},
                            "balance_score": {"value": 0.72, "conf": 0.91, "gated": True},
                        },
                    }
                ],
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_report_artifacts_uses_metric_facts_without_met2_claims(tmp_path: Path) -> None:
    metrics_path = _write_json(tmp_path / "racket_sport_metrics.json", _metrics_payload())

    artifacts = report_model.build_report_artifacts(metrics_path)
    habit_report = artifacts.habit_report
    coach_report = artifacts.coach_report

    assert isinstance(habit_report, HabitReport)
    assert isinstance(coach_report, HabitReport)
    assert habit_report.model_dump(mode="json") == coach_report.model_dump(mode="json")
    assert habit_report.sport == "pickleball"
    assert habit_report.coverage.overall == pytest.approx(0.5)
    assert habit_report.coverage.skipped_reason_counts == {"gated_metric": 1}
    assert habit_report.priority_habit_id == "p1_s000_nvz_margin_ft"
    assert habit_report.habits[0].model_dump(mode="json") == {
        "id": "p1_s000_nvz_margin_ft",
        "title": "P1 dink nvz_margin_ft",
        "summary": "Measured nvz_margin_ft=-0.5 at t=0.250s.",
        "confidence": 0.86,
        "clip_ref": {"t0_sec": 0.0, "t1_sec": 1.25},
        "cue": "Placeholder only: inspect measured nvz_margin_ft before coaching copy.",
        "drill": {"name": "Metric review: nvz_margin_ft", "duration_min": 5.0},
        "source": {
            "player_id": 1,
            "shot_index": 0,
            "shot_type": "dink",
            "shot_time_s": 0.25,
            "metric": "nvz_margin_ft",
            "value": -0.5,
        },
    }


def test_corrections_queue_excludes_metrics_and_records_skipped_reason(tmp_path: Path) -> None:
    metrics_path = _write_json(tmp_path / "racket_sport_metrics.json", _metrics_payload())
    corrections_path = _write_json(
        tmp_path / "corrections_queue.json",
        {
            "schema_version": 1,
            "correction_count": 1,
            "corrections": [
                {
                    "operation": "delete",
                    "artifact": "habit_report.json",
                    "clip_id": "clip_001",
                    "path": "/players/1/shots/0/metrics/nvz_margin_ft",
                    "reason": "coach excluded this candidate",
                }
            ],
        },
    )

    artifacts = report_model.build_report_artifacts(metrics_path, corrections_path=corrections_path)

    assert artifacts.habit_report.habits == []
    assert artifacts.habit_report.priority_habit_id == ""
    assert artifacts.habit_report.coverage.overall == 0.0
    assert artifacts.habit_report.coverage.skipped_reason_counts == {
        "gated_metric": 1,
        "manual_exclusion": 1,
    }


def test_report_artifact_cli_writes_schema_valid_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics_path = _write_json(run_dir / "racket_sport_metrics.json", _metrics_payload())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_report_artifacts.py",
            "--metrics",
            str(metrics_path),
            "--out-dir",
            str(run_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary == {
        "schema_version": 1,
        "habit_report": str(run_dir / "habit_report.json"),
        "coach_report": str(run_dir / "coach_report.json"),
        "habit_count": 1,
        "coverage": {"overall": 0.5, "skipped_reason_counts": {"gated_metric": 1}},
    }
    assert isinstance(validate_artifact_file("habit_report", run_dir / "habit_report.json"), HabitReport)
    assert isinstance(validate_artifact_file("coach_report", run_dir / "coach_report.json"), HabitReport)


def test_report_artifact_cli_fails_closed_on_invalid_metrics(tmp_path: Path) -> None:
    metrics_path = _write_json(tmp_path / "racket_sport_metrics.json", {"schema_version": 1, "players": "bad"})

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_report_artifacts.py",
            "--metrics",
            str(metrics_path),
            "--out-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "ERROR: report artifact build failed:" in completed.stderr
    assert "racket_sport_metrics.json failed validation" in completed.stderr


def test_report_artifact_schema_doc_is_valid_json() -> None:
    schema = json.loads(Path("docs/racketsport/report_artifacts_schema.json").read_text(encoding="utf-8"))

    assert schema["title"] == "Racket-sport report artifacts"
    assert schema["$defs"]["habit_report"]["required"] == [
        "schema_version",
        "sport",
        "coverage",
        "priority_habit_id",
        "habits",
    ]
