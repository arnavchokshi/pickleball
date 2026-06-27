"""Coach and player report models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from threed.racketsport.habit_model import ReportExclusion, build_habit_report
from threed.racketsport.schemas import HabitReport, RacketSportMetrics, validate_artifact_file


REPORT_ARTIFACT_NAMES = ("habit_report.json", "coach_report.json")
REPORT_CORRECTION_ARTIFACTS = {"habit_report.json", "coach_report.json", "habit_report", "coach_report"}


@dataclass(frozen=True)
class ReportArtifacts:
    habit_report: HabitReport
    coach_report: HabitReport

    def summary(self, *, habit_report_path: str | Path, coach_report_path: str | Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "habit_report": str(habit_report_path),
            "coach_report": str(coach_report_path),
            "habit_count": len(self.habit_report.habits),
            "coverage": self.habit_report.coverage.model_dump(mode="json"),
        }


def build_report_artifacts(
    metrics_path: str | Path,
    *,
    corrections_path: str | Path | None = None,
) -> ReportArtifacts:
    metrics = load_metrics_artifact(metrics_path)
    exclusions = load_report_exclusions(corrections_path) if corrections_path is not None else []
    habit_report = build_habit_report(metrics, exclusions=exclusions)

    return ReportArtifacts(habit_report=habit_report, coach_report=HabitReport.model_validate(habit_report.model_dump(mode="json")))


def write_report_artifacts(
    out_dir: str | Path,
    artifacts: ReportArtifacts,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    habit_path = out_path / "habit_report.json"
    coach_path = out_path / "coach_report.json"
    habit_path.write_text(artifacts.habit_report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    coach_path.write_text(artifacts.coach_report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return artifacts.summary(habit_report_path=habit_path, coach_report_path=coach_path)


def load_metrics_artifact(path: str | Path) -> RacketSportMetrics:
    metrics_path = Path(path)
    try:
        artifact = validate_artifact_file("racket_sport_metrics", metrics_path)
    except (OSError, json.JSONDecodeError, ValidationError, KeyError) as exc:
        raise ValueError(f"{metrics_path.name} failed validation: {exc}") from exc
    if not isinstance(artifact, RacketSportMetrics):
        raise ValueError(f"{metrics_path.name} failed validation: parsed artifact had unexpected type")
    return artifact


def load_report_exclusions(path: str | Path) -> list[ReportExclusion]:
    queue_path = Path(path)
    try:
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{queue_path.name} failed validation: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{queue_path.name} failed validation: must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{queue_path.name} failed validation: schema_version must equal 1")
    corrections = payload.get("corrections")
    if not isinstance(corrections, list):
        raise ValueError(f"{queue_path.name} failed validation: corrections must be an array")
    correction_count = payload.get("correction_count")
    if correction_count is not None and correction_count != len(corrections):
        raise ValueError(f"{queue_path.name} failed validation: correction_count does not match corrections length")

    exclusions: list[ReportExclusion] = []
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            raise ValueError(f"{queue_path.name} failed validation: corrections/{index} must be an object")
        operation = correction.get("operation")
        artifact = correction.get("artifact")
        path_value = correction.get("path")
        reason = correction.get("reason")
        if not isinstance(operation, str) or not operation:
            raise ValueError(f"{queue_path.name} failed validation: corrections/{index}/operation must be a string")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError(f"{queue_path.name} failed validation: corrections/{index}/artifact must be a string")
        if not isinstance(path_value, str) or not path_value.startswith("/"):
            raise ValueError(f"{queue_path.name} failed validation: corrections/{index}/path must be a JSON pointer")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{queue_path.name} failed validation: corrections/{index}/reason must be a non-empty string")
        if artifact not in REPORT_CORRECTION_ARTIFACTS:
            continue
        if operation != "delete":
            raise ValueError(
                f"{queue_path.name} failed validation: corrections/{index}/operation must be delete for report exclusions"
            )
        exclusions.append(ReportExclusion(path=path_value, reason=reason))
    return exclusions
