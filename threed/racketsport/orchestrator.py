"""Fail-closed racket-sport pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from scripts.racketsport.track import build_tracks

from .court_calibration import calibration_from_manual_taps
from .court_templates import Sport
from .court_zones import build_court_zones
from .net_plane import build_net_plane
from .pipeline_contracts import PIPELINE_STAGE_CONTRACTS, PipelineContractError, PipelineStageContract
from .schemas import CourtCalibration, StrictArtifact, validate_artifact_file


ARTIFACT_SCHEMA_BY_FILENAME: dict[str, str] = {
    "court_calibration.json": "court_calibration",
    "court_zones.json": "court_zones",
    "net_plane.json": "net_plane",
    "tracks.json": "tracks",
    "smpl_motion.json": "smpl_motion",
    "skeleton3d.json": "skeleton3d",
    "ball_track.json": "ball_track",
    "contact_windows.json": "contact_windows",
    "racket_pose.json": "racket_pose",
    "racket_sport_metrics.json": "racket_sport_metrics",
    "habit_report.json": "habit_report",
    "coach_report.json": "coach_report",
    "drill_report.json": "drill_report",
    "replay_scene.json": "replay_scene",
}

PIPELINE_STATUS_PASS = "pass"
PIPELINE_STATUS_FAIL = "fail"
PIPELINE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class StageContext:
    clip: str
    inputs_dir: Path
    run_dir: Path
    sport: Sport
    device: str | None = None
    max_frames: int | None = None


@dataclass(frozen=True)
class StageRun:
    stage: str
    status: str
    real_model: bool
    source_mode: str
    produced_artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "real_model": self.real_model,
            "source_mode": self.source_mode,
            "produced_artifacts": list(self.produced_artifacts),
            "notes": list(self.notes),
            "metrics": self.metrics,
        }


class StageRunner(Protocol):
    stage: str
    real_model: bool
    source_mode: str

    def run(self, context: StageContext) -> StageRun:
        """Run one stage and write its artifacts."""


class ManualCalibrationRunner:
    stage = "calibration"
    real_model = False
    source_mode = "manual_sidecar"

    def run(self, context: StageContext) -> StageRun:
        sidecar_path = context.inputs_dir / "capture_sidecar.json"
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"missing calibration sidecar: {sidecar_path}")

        calibration = calibration_from_manual_taps(sidecar_path, sport=context.sport)
        artifacts = {
            "court_calibration.json": calibration,
            "court_zones.json": build_court_zones(context.sport),
            "net_plane.json": build_net_plane(context.sport),
        }
        for filename, artifact in artifacts.items():
            _write_json_artifact(context.run_dir / filename, artifact)

        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=tuple(artifacts),
            notes=(
                "manual 4-corner calibration seed; requires human-reviewed corners for product verification",
            ),
            metrics={
                "reprojection_median_px": calibration.reprojection_error_px.median,
                "reprojection_p95_px": calibration.reprojection_error_px.p95,
            },
        )


class PrecomputedTrackingRunner:
    stage = "tracking"
    real_model = False
    source_mode = "precomputed_detections"

    def run(self, context: StageContext) -> StageRun:
        detections_path = context.inputs_dir / "detections.json"
        calibration_path = context.run_dir / "court_calibration.json"
        if not detections_path.is_file():
            raise FileNotFoundError(f"missing detections: {detections_path}")
        calibration = validate_artifact_file("court_calibration", calibration_path)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        detections_payload = _read_json(detections_path)
        tracks, counts = build_tracks(detections_payload, calibration, max_step_m=2.0)
        if counts["accepted"] <= 0 or not tracks.players:
            raise ValueError(f"tracking failed: no accepted on-court person tracks; counts={counts}")

        _write_json_artifact(context.run_dir / "tracks.json", tracks)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("tracks.json",),
            notes=(
                "uses precomputed detections; not a GPU model invocation",
            ),
            metrics=counts,
        )


DEFAULT_RUNNERS: dict[str, StageRunner] = {
    "calibration": ManualCalibrationRunner(),
    "tracking": PrecomputedTrackingRunner(),
}


def run_pipeline(
    *,
    clip: str,
    inputs_dir: str | Path,
    run_dir: str | Path,
    stage: str = "e2e",
    sport: Sport = "pickleball",
    runners: dict[str, StageRunner] | None = None,
    device: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the pipeline through ``stage`` and stop rather than fabricate artifacts."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    context = StageContext(
        clip=clip,
        inputs_dir=Path(inputs_dir),
        run_dir=run_path,
        sport=sport,
        device=device,
        max_frames=max_frames,
    )
    registry = dict(DEFAULT_RUNNERS)
    if runners:
        registry.update(runners)

    stage_runs: list[dict[str, Any]] = []
    summary_status = PIPELINE_STATUS_PASS

    for contract in _ordered_contracts_for(stage):
        runner = registry.get(contract.stage)
        if runner is None:
            stage_runs.append(_blocked_stage(contract, f"no runner registered for stage: {contract.stage}"))
            summary_status = PIPELINE_STATUS_BLOCKED
            break

        try:
            result = runner.run(context)
            _validate_contract_artifacts(contract, run_path)
        except Exception as exc:
            stage_runs.append(
                StageRun(
                    stage=contract.stage,
                    status=PIPELINE_STATUS_FAIL,
                    real_model=getattr(runner, "real_model", False),
                    source_mode=getattr(runner, "source_mode", "unknown"),
                    notes=(f"{contract.stage} failed: {exc}",),
                ).as_dict()
            )
            summary_status = PIPELINE_STATUS_FAIL
            break
        stage_runs.append(result.as_dict())

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_pipeline_run",
        "clip": clip,
        "requested_stage": stage,
        "status": summary_status,
        "run_dir": str(run_path),
        "inputs_dir": str(context.inputs_dir),
        "stages": stage_runs,
    }
    _write_json(run_path / "pipeline_run.json", summary)
    return summary


def _ordered_contracts_for(stage: str) -> list[PipelineStageContract]:
    contracts_by_stage = {contract.stage: contract for contract in PIPELINE_STAGE_CONTRACTS}
    if stage not in contracts_by_stage:
        valid = ", ".join(contract.stage for contract in PIPELINE_STAGE_CONTRACTS)
        raise PipelineContractError(f"unknown pipeline stage: {stage}; expected one of: {valid}")

    needed: set[str] = set()

    def visit(current: str) -> None:
        if current in needed:
            return
        contract = contracts_by_stage[current]
        for dependency in contract.depends_on:
            visit(dependency)
        needed.add(current)

    visit(stage)
    return [contract for contract in PIPELINE_STAGE_CONTRACTS if contract.stage in needed]


def _validate_contract_artifacts(contract: PipelineStageContract, run_dir: Path) -> None:
    for artifact in contract.required_artifacts:
        schema_name = ARTIFACT_SCHEMA_BY_FILENAME.get(artifact)
        if schema_name is None:
            raise ValueError(f"no schema mapping for required artifact: {artifact}")
        validate_artifact_file(schema_name, run_dir / artifact)


def _blocked_stage(contract: PipelineStageContract, note: str) -> dict[str, Any]:
    return StageRun(
        stage=contract.stage,
        status=PIPELINE_STATUS_BLOCKED,
        real_model=False,
        source_mode="unregistered",
        notes=(note,),
    ).as_dict()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_artifact(path: Path, artifact: StrictArtifact | Any) -> None:
    if hasattr(artifact, "model_dump"):
        payload = artifact.model_dump(mode="json")
    else:
        payload = artifact
    _write_json(path, payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fail-closed racket-sport pipeline spine.")
    parser.add_argument("--clip", required=True, help="Clip identifier for the run summary.")
    parser.add_argument("--inputs", type=Path, required=True, help="Directory containing stage inputs for this clip.")
    parser.add_argument("--out", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--stage", default="e2e", help="Target pipeline stage from pipeline_contracts.py.")
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    parser.add_argument("--device", default=None, help="Optional GPU device hint for future GPU runners.")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    summary = run_pipeline(
        clip=args.clip,
        inputs_dir=args.inputs,
        run_dir=args.out,
        stage=args.stage,
        sport=args.sport,
        device=args.device,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == PIPELINE_STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
