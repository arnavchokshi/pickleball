from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threed.racketsport.schemas import EvalClipResult, EvalMetric, EvalStatus, EvalSummary, PhaseEvalMetrics
from threed.racketsport.testclips import TestClipDatasetManifest


NUMERIC_GATE_OPERATORS = {"<", "<=", ">", ">=", "=="}


@dataclass(frozen=True)
class NumericGate:
    name: str
    op: str
    threshold: float | int
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.op not in NUMERIC_GATE_OPERATORS:
            raise ValueError(f"unsupported numeric gate operator: {self.op}")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise TypeError("numeric gate threshold must be an int or float")

    @property
    def label(self) -> str:
        return f"{self.name}: {self.op} {self.threshold}"


def metric(
    *,
    value: float | int | bool | str | None,
    unit: str | None,
    gate: str,
    passed: bool | None,
    status: str = "measured",
) -> EvalMetric:
    return EvalMetric(value=value, unit=unit, gate=gate, passed=passed, status=status)


def evaluate_numeric_gates(
    values: Mapping[str, Any],
    gates: Mapping[str, NumericGate],
) -> dict[str, EvalMetric]:
    gated: dict[str, EvalMetric] = {}
    for name, gate in gates.items():
        value = _extract_numeric_gate_value(values.get(name))
        if value is None:
            gated[name] = metric(value=None, unit=gate.unit, gate=gate.label, passed=None, status="not_measured")
            continue

        gated[name] = metric(
            value=value,
            unit=gate.unit,
            gate=gate.label,
            passed=_numeric_gate_passed(value, gate),
        )
    return gated


def _extract_numeric_gate_value(raw: Any) -> float | int | None:
    if isinstance(raw, EvalMetric):
        raw = raw.value
    elif isinstance(raw, Mapping):
        raw = raw.get("value")

    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"numeric gate value must be an int or float, got {type(raw).__name__}")
    return raw


def _numeric_gate_passed(value: float | int, gate: NumericGate) -> bool:
    if gate.op == "<":
        return value < gate.threshold
    if gate.op == "<=":
        return value <= gate.threshold
    if gate.op == ">":
        return value > gate.threshold
    if gate.op == ">=":
        return value >= gate.threshold
    return value == gate.threshold


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
                gate="artifact_check.all_required_artifacts_exist",
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
