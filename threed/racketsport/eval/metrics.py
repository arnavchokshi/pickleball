from __future__ import annotations

from pathlib import Path

from threed.racketsport.schemas import EvalClipResult, EvalMetric, EvalStatus, EvalSummary, PhaseEvalMetrics
from threed.racketsport.testclips import TestClipDatasetManifest


def metric(
    *,
    value: float | int | bool | str | None,
    unit: str | None,
    gate: str,
    passed: bool | None,
    status: str = "measured",
) -> EvalMetric:
    return EvalMetric(value=value, unit=unit, gate=gate, passed=passed, status=status)


def missing_artifacts(run_dir: Path, required_artifacts: list[str]) -> list[str]:
    return [artifact for artifact in required_artifacts if not (run_dir / artifact).is_file()]


def summarize_clips(dataset: TestClipDatasetManifest, clips: list[EvalClipResult]) -> EvalSummary:
    return EvalSummary(
        total_clips=dataset.total_clips,
        ready_clips=dataset.ready_clips,
        evaluated_clips=sum(1 for clip in clips if clip.status in {"pass", "fail"}),
        passed_clips=sum(1 for clip in clips if clip.status == "pass"),
        failed_clips=sum(1 for clip in clips if clip.status == "fail"),
        blocked_clips=sum(1 for clip in clips if clip.status == "blocked"),
    )


def aggregate_status(clips: list[EvalClipResult]) -> EvalStatus:
    if any(clip.status == "fail" for clip in clips):
        return "fail"
    if any(clip.status == "blocked" for clip in clips):
        return "blocked"
    if clips and all(clip.status == "pass" for clip in clips):
        return "pass"
    return "not_measured"


def build_phase_metrics(
    *,
    phase: str,
    evaluator: str,
    root: Path,
    labels_root: Path,
    required_artifacts: list[str],
    dataset: TestClipDatasetManifest,
    clips: list[EvalClipResult],
    notes: list[str] | None = None,
) -> PhaseEvalMetrics:
    status = aggregate_status(clips)
    artifact_checks = [clip for clip in clips if clip.status in {"pass", "fail", "blocked"}]
    artifact_readiness_passed = bool(artifact_checks) and all(not clip.missing_artifacts for clip in artifact_checks)
    artifact_readiness_status = "measured" if artifact_checks else "not_measured"
    return PhaseEvalMetrics(
        schema_version=1,
        phase=phase,
        evaluator=evaluator,
        root=str(root),
        labels_root=str(labels_root),
        status=status,
        required_artifacts=required_artifacts,
        summary=summarize_clips(dataset, clips),
        metrics={
            "artifact_readiness": metric(
                value=artifact_readiness_passed if artifact_checks else None,
                unit=None,
                gate="all required artifacts exist",
                passed=artifact_readiness_passed if artifact_checks else None,
                status=artifact_readiness_status,
            )
        },
        clips=clips,
        notes=notes or [],
    )


def write_phase_metrics(out: str | Path, payload: PhaseEvalMetrics) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
