from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .schemas import validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_pipeline_artifact_readiness"
COURT_REVIEW_RETIRED_CLIPS = {"burlington_gold_0300_low_steep_corner"}
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
        required_artifacts=("court_calibration.json", "court_zones.json", "net_plane.json", "court_line_evidence.json"),
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
        required_artifacts=("smpl_motion.json", "skeleton3d.json", "body_compute_execution.json", "body_mesh_readiness.json"),
        depends_on=("tracking",),
    ),
    PipelineStageContract(
        stage="physics",
        phase="phase4",
        required_artifacts=("smpl_motion.json", "physics_refinement.json"),
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
        required_artifacts=("racket_pose.json", "racket_pose_readiness.json", "racket_promotion_audit.json"),
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
            "court_line_evidence.json",
            "tracks.json",
            "smpl_motion.json",
            "skeleton3d.json",
            "body_compute_execution.json",
            "body_mesh_readiness.json",
            "physics_refinement.json",
            "ball_track.json",
            "contact_windows.json",
            "racket_pose.json",
            "racket_pose_readiness.json",
            "racket_promotion_audit.json",
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
_ARTIFACT_SCHEMA_BY_FILENAME = {
    "court_calibration.json": "court_calibration",
    "court_zones.json": "court_zones",
    "net_plane.json": "net_plane",
    "court_line_evidence.json": "court_line_evidence",
    "tracks.json": "tracks",
    "smpl_motion.json": "smpl_motion",
    "skeleton3d.json": "skeleton3d",
    "body_compute_execution.json": "body_compute_execution",
    "body_mesh_readiness.json": "body_mesh_readiness",
    "physics_refinement.json": "physics_refinement",
    "ball_track.json": "ball_track",
    "contact_windows.json": "contact_windows",
    "racket_pose.json": "racket_pose",
    "racket_pose_readiness.json": "racket_pose_readiness",
    "racket_promotion_audit.json": "racket_promotion_audit",
    "racket_sport_metrics.json": "racket_sport_metrics",
    "habit_report.json": "habit_report",
    "coach_report.json": "coach_report",
    "drill_report.json": "drill_report",
    "replay_scene.json": "replay_scene",
}


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
        artifact_validation_errors = _artifact_validation_errors(run_path, present)
        blocked_by = [dependency for dependency in contract.depends_on if dependency not in ready_stages]
        semantic_blockers = _semantic_blockers_for_stage(contract.stage, run_path)

        if missing or artifact_validation_errors:
            status: StageStatus = "not_ready"
        elif blocked_by or semantic_blockers:
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
                "artifact_validation_errors": artifact_validation_errors,
                "semantic_blockers": semantic_blockers,
            }
        )

    required_artifacts = _dedupe_artifacts(stage_reports, "required_artifacts")
    missing_artifacts = _dedupe_artifacts(stage_reports, "missing_artifacts")
    artifact_validation_errors = _dedupe_artifacts(stage_reports, "artifact_validation_errors")
    semantic_blockers = _dedupe_semantic_blockers(stage_reports)
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
        "artifact_validation_errors": artifact_validation_errors,
        "semantic_blockers": semantic_blockers,
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


def _dedupe_semantic_blockers(stage_reports: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    blockers: list[str] = []
    for report in stage_reports:
        stage = str(report["stage"])
        for blocker in report.get("semantic_blockers", []):
            key = f"{stage}:{blocker}"
            if key in seen:
                continue
            seen.add(key)
            blockers.append(key)
    return blockers


def _artifact_validation_errors(run_path: Path, artifacts: list[str]) -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        schema = _ARTIFACT_SCHEMA_BY_FILENAME.get(artifact)
        if schema is None:
            continue
        try:
            validate_artifact_file(schema, run_path / artifact)
        except Exception as exc:
            errors.append(f"{artifact}: {exc}")
    return errors


def _semantic_blockers_for_stage(stage: str, run_path: Path) -> list[str]:
    if stage == "calibration":
        return _court_line_evidence_blockers(run_path)
    if stage == "body":
        return _dedupe([*_body_compute_execution_blockers(run_path), *_body_mesh_readiness_blockers(run_path)])
    if stage == "ball_events":
        return _contact_windows_blockers(run_path)
    if stage == "racket":
        return _dedupe([*_racket_pose_readiness_blockers(run_path), *_racket_promotion_audit_blockers(run_path)])
    return []


def _court_line_evidence_blockers(run_path: Path) -> list[str]:
    if run_path.name in COURT_REVIEW_RETIRED_CLIPS:
        return ["court_line_evidence_retired_for_court_calibration"]
    path = run_path / "court_line_evidence.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"court_line_evidence_{error}"]
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return ["court_line_evidence_missing_aggregate"]

    blockers: list[str] = []
    if aggregate.get("auto_calibration_ready") is not True:
        blockers.append("court_line_evidence_not_ready")
    elif not _ready_court_evidence_has_observations(payload, aggregate):
        line_observations = payload.get("line_observations")
        net_observations = payload.get("net_observations")
        if not isinstance(line_observations, list) or not line_observations:
            blockers.append("court_line_evidence_ready_without_line_observations")
        if not isinstance(net_observations, list) or not net_observations:
            blockers.append("court_line_evidence_ready_without_net_observations")
    for line_id in _string_list(aggregate.get("missing_required_line_ids")):
        blockers.append(f"court_line_evidence_missing_required_line_{line_id}")
    for net_id in _string_list(aggregate.get("missing_required_net_ids")):
        blockers.append(f"court_line_evidence_missing_required_net_{net_id}")
    return _dedupe(blockers)


def _ready_court_evidence_has_observations(payload: dict[str, Any], aggregate: dict[str, Any]) -> bool:
    line_observations = payload.get("line_observations")
    net_observations = payload.get("net_observations")
    if not isinstance(line_observations, list) or not isinstance(net_observations, list):
        return False
    observed_line_ids = {
        str(item.get("line_id"))
        for item in line_observations
        if isinstance(item, dict) and isinstance(item.get("line_id"), str)
    }
    accepted_line_ids = _string_list(aggregate.get("accepted_line_ids"))
    return bool(accepted_line_ids) and bool(net_observations) and all(line_id in observed_line_ids for line_id in accepted_line_ids)


def _body_compute_execution_blockers(run_path: Path) -> list[str]:
    path = run_path / "body_compute_execution.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"body_compute_execution_{error}"]
    scheduled_count = _int_value(_nested_value(payload, "summary", "scheduled_frame_count"))
    if scheduled_count is None:
        scheduled_frames = payload.get("scheduled_frames")
        scheduled_count = len(scheduled_frames) if isinstance(scheduled_frames, list) else None
    if scheduled_count == 0:
        return ["body_compute_execution_has_no_scheduled_frames"]
    if scheduled_count is None:
        return ["body_compute_execution_missing_scheduled_frame_count"]
    scheduled_by_target = _nested_value(payload, "summary", "scheduled_by_target_representation")
    world_mesh_count = _int_value(scheduled_by_target.get("world_mesh")) if isinstance(scheduled_by_target, dict) else None
    if isinstance(world_mesh_count, int) and world_mesh_count == 0:
        return ["body_compute_execution_has_no_world_mesh_frames"]
    return []


def _body_mesh_readiness_blockers(run_path: Path) -> list[str]:
    path = run_path / "body_mesh_readiness.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"body_mesh_readiness_{error}"]
    blockers: list[str] = []
    representation_decision = str(payload.get("representation_decision", ""))
    if representation_decision == "no_world_mesh_requested":
        blockers.append("body_mesh_no_world_mesh_requested")
    elif representation_decision == "world_mesh_required_missing_output":
        blockers.append("body_mesh_world_mesh_required_missing_output")
    elif representation_decision == "world_mesh_required_available_unverified":
        blockers.append("body_mesh_world_mesh_unverified")
    status = str(payload.get("status", ""))
    if status == "missing_body_output":
        blockers.append("body_mesh_readiness_missing_body_output")
    if payload.get("trusted_for_body_promotion") is False:
        blockers.append("body_mesh_not_trusted_for_promotion")
    return _dedupe(blockers)


def _contact_windows_blockers(run_path: Path) -> list[str]:
    path = run_path / "contact_windows.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"contact_windows_{error}"]
    events = payload.get("events")
    if not isinstance(events, list):
        return ["contact_windows_invalid_events"]
    if not events:
        return ["contact_windows_has_no_events"]
    return []


def _racket_pose_readiness_blockers(run_path: Path) -> list[str]:
    path = run_path / "racket_pose_readiness.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"racket_pose_readiness_{error}"]
    blockers = [f"racket_pose_readiness_{blocker}" for blocker in _string_list(payload.get("blockers"))]
    status = str(payload.get("status", ""))
    if status not in {"ready", "ready_for_rkt_promotion", "verified"}:
        blockers.append(f"racket_pose_readiness_status_{status or 'missing'}")
    return _dedupe(blockers)


def _racket_promotion_audit_blockers(run_path: Path) -> list[str]:
    path = run_path / "racket_promotion_audit.json"
    if not path.is_file():
        return []
    payload, error = _read_mapping_json(path)
    if error:
        return [f"racket_promotion_audit_{error}"]
    blockers = [f"racket_promotion_audit_{blocker}" for blocker in _string_list(payload.get("blockers"))]
    if payload.get("trusted_for_rkt_promotion") is not True:
        blockers.append("racket_promotion_audit_not_trusted_for_rkt_promotion")
    return _dedupe(blockers)


def _read_mapping_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, "invalid_json"
    except OSError:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "not_object"
    return payload, None


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if not token:
            continue
        result.append(token)
    return result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
