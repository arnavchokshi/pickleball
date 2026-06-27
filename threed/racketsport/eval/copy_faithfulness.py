from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import (
    NumericGate,
    build_phase_metrics,
    evaluate_numeric_gates,
    metric,
    missing_artifacts,
    write_phase_metrics,
)
from threed.racketsport.schemas import EvalClipResult, HabitReport, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_COPY_ARTIFACTS = ["habit_report.json", "coach_report.json"]
COPY_GATES = {
    "habit_count": NumericGate(name="copy_habit_count_min", op=">=", threshold=1, unit="habits"),
    "coach_habit_count": NumericGate(name="copy_coach_habit_count_min", op=">=", threshold=1, unit="habits"),
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

        missing = missing_artifacts(run_dir, REQUIRED_COPY_ARTIFACTS)
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
        phase="phase9",
        evaluator="copy_faithfulness",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_COPY_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        habit_report = validate_artifact_file("habit_report", run_dir / "habit_report.json")
        coach_report = validate_artifact_file("coach_report", run_dir / "coach_report.json")
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

    if not isinstance(habit_report, HabitReport) or not isinstance(coach_report, HabitReport):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["copy artifacts did not parse as HabitReport"],
        )

    habit_count = len(habit_report.habits)
    coach_habit_count = len(coach_report.habits)
    priority_habit_match = habit_report.priority_habit_id == coach_report.priority_habit_id
    gated = evaluate_numeric_gates(
        {
            "habit_count": habit_count,
            "coach_habit_count": coach_habit_count,
        },
        COPY_GATES,
    )
    passed = all(gated_metric.passed is True for gated_metric in gated.values()) and priority_habit_match
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "habit_count": gated["habit_count"],
            "coach_habit_count": gated["coach_habit_count"],
            "priority_habit_match": metric(
                value=priority_habit_match,
                unit=None,
                gate="coach copy preserves priority habit",
                passed=priority_habit_match,
            ),
            "coverage_overall": metric(
                value=habit_report.coverage.overall,
                unit="ratio",
                gate="recorded for later fast-tier/report gates",
                passed=True,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 9 report-copy artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase9"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase9/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
