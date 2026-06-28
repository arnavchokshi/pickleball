from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.court_calibration import (
    CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
    CALIBRATION_REPROJECTION_P95_GATE_PX,
)
from threed.racketsport.eval.metrics import (
    NumericGate,
    build_phase_metrics,
    evaluate_numeric_gates,
    metric,
    missing_artifacts,
    write_phase_metrics,
)
from threed.racketsport.schemas import CourtCalibration, CourtLineEvidence, EvalClipResult, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_CALIBRATION_ARTIFACTS = [
    "court_calibration.json",
    "court_zones.json",
    "net_plane.json",
    "court_line_evidence.json",
]
CALIBRATION_GATES = {
    "reprojection_median_px": NumericGate(
        name="calibration_reprojection_median_px",
        op="<",
        threshold=CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
        unit="px",
    ),
    "reprojection_p95_px": NumericGate(
        name="calibration_reprojection_p95_px",
        op="<",
        threshold=CALIBRATION_REPROJECTION_P95_GATE_PX,
        unit="px",
    ),
}


def evaluate(root: str | Path, labels_root: str | Path) -> object:
    root_path = Path(root)
    labels_path = Path(labels_root)
    dataset = build_testclip_manifest(labels_path)
    results: list[EvalClipResult] = []
    notes: list[str] = []

    if not dataset.root_exists:
        notes.append("labels root does not exist")

    for clip in dataset.clips:
        run_dir = root_path / clip.name
        labels_dir = clip.labels_dir
        if not clip.is_ready:
            results.append(
                EvalClipResult(
                    clip=clip.name,
                    run_dir=str(run_dir),
                    labels_dir=str(labels_dir),
                    status="not_measured",
                    missing_label_files=clip.missing_label_files,
                    missing_artifacts=[],
                    metrics={},
                    notes=["DATA-1 labels are incomplete"],
                )
            )
            continue

        missing = missing_artifacts(run_dir, REQUIRED_CALIBRATION_ARTIFACTS)
        if missing:
            results.append(
                EvalClipResult(
                    clip=clip.name,
                    run_dir=str(run_dir),
                    labels_dir=str(labels_dir),
                    status="blocked",
                    missing_label_files=[],
                    missing_artifacts=missing,
                    metrics={},
                    notes=[],
                )
            )
            continue

        results.append(_evaluate_ready_clip(clip.name, run_dir=run_dir, labels_dir=labels_dir))

    return build_phase_metrics(
        phase="phase1",
        evaluator="calib_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_CALIBRATION_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    notes: list[str] = []
    try:
        calibration = validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
        validate_artifact_file("court_zones", run_dir / "court_zones.json")
        validate_artifact_file("net_plane", run_dir / "net_plane.json")
        line_evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    except Exception as exc:
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=[f"artifact validation failed: {exc}"],
        )

    if not isinstance(calibration, CourtCalibration):
        notes.append("court_calibration artifact did not parse as CourtCalibration")
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=notes,
        )
    if not isinstance(line_evidence, CourtLineEvidence):
        notes.append("court_line_evidence artifact did not parse as CourtLineEvidence")
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=notes,
        )

    metrics = evaluate_numeric_gates(
        {
            "reprojection_median_px": calibration.reprojection_error_px.median,
            "reprojection_p95_px": calibration.reprojection_error_px.p95,
        },
        CALIBRATION_GATES,
    )
    metrics["auto_calibration_ready"] = metric(
        value=line_evidence.aggregate.auto_calibration_ready,
        unit=None,
        gate="court_line_evidence.auto_calibration_ready",
        passed=line_evidence.aggregate.auto_calibration_ready,
    )
    metrics["court_line_mean_residual_px"] = metric(
        value=line_evidence.aggregate.mean_residual_px,
        unit="px",
        gate="court_line_evidence.aggregate.mean_residual_px",
        passed=None,
    )
    metrics["court_line_p95_residual_px"] = metric(
        value=line_evidence.aggregate.p95_residual_px,
        unit="px",
        gate="court_line_evidence.aggregate.p95_residual_px",
        passed=None,
    )
    if not line_evidence.aggregate.auto_calibration_ready:
        notes.extend(line_evidence.aggregate.reasons)
    passed = all(gated_metric.passed is True for gated_metric in metrics.values() if gated_metric.passed is not None)
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics=metrics,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 court-calibration artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase1"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase1/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
