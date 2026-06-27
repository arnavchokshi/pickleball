from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_pipeline_artifact_readiness"
ReadinessStatus = Literal["ready", "not_ready"]
StageStatus = Literal["ready", "not_ready", "blocked"]


class PipelineContractError(ValueError):
    """Raised when a pipeline contract path or stage is invalid."""


@dataclass(frozen=True)
class PipelineStageContract:
    stage: str
    phase: str
    required_artifacts: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


PIPELINE_STAGE_CONTRACTS: tuple[PipelineStageContract, ...] = (
    PipelineStageContract(
        stage="calibration",
        phase="phase1",
        required_artifacts=("court_calibration.json", "court_zones.json", "net_plane.json"),
    ),
    PipelineStageContract(
        stage="tracking",
        phase="phase2",
        required_artifacts=("tracks.json",),
        depends_on=("calibration",),
    ),
    PipelineStageContract(
        stage="body",
        phase="phase3",
        required_artifacts=("smpl_motion.json", "skeleton3d.json"),
        depends_on=("tracking",),
    ),
    PipelineStageContract(
        stage="physics",
        phase="phase4",
        required_artifacts=("smpl_motion.json",),
        depends_on=("body",),
    ),
    PipelineStageContract(
        stage="ball_events",
        phase="phase5",
        required_artifacts=("ball_track.json", "contact_windows.json"),
        depends_on=("physics",),
    ),
    PipelineStageContract(
        stage="racket",
        phase="phase6",
        required_artifacts=("racket_pose.json",),
        depends_on=("physics", "ball_events"),
    ),
    PipelineStageContract(
        stage="metrics",
        phase="phase7",
        required_artifacts=("racket_sport_metrics.json", "habit_report.json"),
        depends_on=("physics", "ball_events", "racket"),
    ),
    PipelineStageContract(
        stage="shot_drill",
        phase="phase8",
        required_artifacts=("racket_sport_metrics.json", "drill_report.json"),
        depends_on=("metrics",),
    ),
    PipelineStageContract(
        stage="copy",
        phase="phase9",
        required_artifacts=("habit_report.json", "coach_report.json"),
        depends_on=("metrics",),
    ),
    PipelineStageContract(
        stage="replay",
        phase="phase10",
        required_artifacts=("replay_scene.json",),
        depends_on=("physics", "ball_events", "racket"),
    ),
    PipelineStageContract(
        stage="e2e",
        phase="phase11",
        required_artifacts=(
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "tracks.json",
            "smpl_motion.json",
            "ball_track.json",
            "contact_windows.json",
            "racket_pose.json",
            "racket_sport_metrics.json",
            "habit_report.json",
            "coach_report.json",
            "drill_report.json",
            "replay_scene.json",
        ),
        depends_on=("metrics", "shot_drill", "copy", "replay"),
    ),
)

PIPELINE_STAGE_ORDER = [contract.stage for contract in PIPELINE_STAGE_CONTRACTS]
_CONTRACTS_BY_STAGE = {contract.stage: contract for contract in PIPELINE_STAGE_CONTRACTS}


def safe_relative_path(value: str | Path) -> Path:
    raw = str(value)
    if not raw or raw == ".":
        raise PipelineContractError("relative path must be non-empty")

    pure = PurePosixPath(raw.replace("\\", "/"))
    if pure.is_absolute():
        raise PipelineContractError(f"relative path must not be absolute: {raw}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise PipelineContractError(f"relative path must not contain empty/current/parent segments: {raw}")

    return Path(*pure.parts)


def build_readiness_report(run_dir: str | Path, *, stage: str = "e2e") -> dict[str, Any]:
    stage = _normalize_stage(stage)
    run_path = Path(run_dir)
    contracts = _ordered_contracts_for(stage)
    stage_reports: list[dict[str, Any]] = []
    ready_stages: set[str] = set()

    for contract in contracts:
        present = [artifact for artifact in contract.required_artifacts if (run_path / safe_relative_path(artifact)).is_file()]
        missing = [artifact for artifact in contract.required_artifacts if artifact not in present]
        blocked_by = [dependency for dependency in contract.depends_on if dependency not in ready_stages]

        if missing:
            status: StageStatus = "not_ready"
        elif blocked_by:
            status = "blocked"
        else:
            status = "ready"
            ready_stages.add(contract.stage)

        stage_reports.append(
            {
                "stage": contract.stage,
                "phase": contract.phase,
                "depends_on": list(contract.depends_on),
                "status": status,
                "blocked_by": blocked_by,
                "required_artifacts": list(contract.required_artifacts),
                "present_artifacts": present,
                "missing_artifacts": missing,
            }
        )

    required_artifacts = _dedupe_artifacts(stage_reports, "required_artifacts")
    missing_artifacts = _dedupe_artifacts(stage_reports, "missing_artifacts")
    status: ReadinessStatus = "ready" if all(item["status"] == "ready" for item in stage_reports) else "not_ready"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_dir": str(run_path),
        "requested_stage": stage,
        "status": status,
        "stage_order": PIPELINE_STAGE_ORDER,
        "required_artifacts": required_artifacts,
        "missing_artifacts": missing_artifacts,
        "stages": stage_reports,
    }


def _normalize_stage(stage: str) -> str:
    normalized = stage.strip()
    if normalized not in _CONTRACTS_BY_STAGE:
        valid = ", ".join(PIPELINE_STAGE_ORDER)
        raise PipelineContractError(f"unknown pipeline stage: {stage}; expected one of: {valid}")
    return normalized


def _ordered_contracts_for(stage: str) -> list[PipelineStageContract]:
    needed = _dependency_closure(stage)
    return [contract for contract in PIPELINE_STAGE_CONTRACTS if contract.stage in needed]


def _dependency_closure(stage: str) -> set[str]:
    needed: set[str] = set()

    def visit(current: str) -> None:
        if current in needed:
            return
        contract = _CONTRACTS_BY_STAGE[current]
        for dependency in contract.depends_on:
            visit(dependency)
        needed.add(current)

    visit(stage)
    return needed


def _dedupe_artifacts(stage_reports: list[dict[str, Any]], field: str) -> list[str]:
    seen: set[str] = set()
    artifacts: list[str] = []
    for report in stage_reports:
        for artifact in report[field]:
            if artifact in seen:
                continue
            seen.add(artifact)
            artifacts.append(artifact)
    return artifacts
