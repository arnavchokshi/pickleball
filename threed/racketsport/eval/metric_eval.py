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
from threed.racketsport.schemas import EvalClipResult, HabitReport, RacketSportMetrics, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_METRIC_ARTIFACTS = ["racket_sport_metrics.json", "habit_report.json"]
METRIC_GATES = {
    "metric_players": NumericGate(name="metric_players_min", op=">=", threshold=1, unit="players"),
    "shots": NumericGate(name="metric_shots_min", op=">=", threshold=1, unit="shots"),
    "metric_values": NumericGate(name="metric_values_min", op=">=", threshold=1, unit="metrics"),
    "habits": NumericGate(name="metric_habits_min", op=">=", threshold=1, unit="habits"),
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

        missing = missing_artifacts(run_dir, REQUIRED_METRIC_ARTIFACTS)
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
        phase="phase7",
        evaluator="metric_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_METRIC_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        metrics_artifact = validate_artifact_file("racket_sport_metrics", run_dir / "racket_sport_metrics.json")
        habit_report = validate_artifact_file("habit_report", run_dir / "habit_report.json")
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

    if not isinstance(metrics_artifact, RacketSportMetrics) or not isinstance(habit_report, HabitReport):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["metric artifacts did not parse as RacketSportMetrics and HabitReport"],
        )

    metric_players = len(metrics_artifact.players)
    shots = sum(len(player.shots) for player in metrics_artifact.players)
    metric_values = sum(len(shot.metrics) for player in metrics_artifact.players for shot in player.shots)
    gated_metrics = sum(
        1
        for player in metrics_artifact.players
        for shot in player.shots
        for value in shot.metrics.values()
        if value.gated is not None
    )
    habits = len(habit_report.habits)
    gated = evaluate_numeric_gates(
        {
            "metric_players": metric_players,
            "shots": shots,
            "metric_values": metric_values,
            "habits": habits,
        },
        METRIC_GATES,
    )
    passed = all(gated_metric.passed is True for gated_metric in gated.values())
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "metric_players": gated["metric_players"],
            "shots": gated["shots"],
            "metric_values": gated["metric_values"],
            "gated_metrics": metric(
                value=gated_metrics,
                unit="metrics",
                gate="recorded for later confidence gates",
                passed=True,
            ),
            "habits": gated["habits"],
            "coverage_overall": metric(
                value=habit_report.coverage.overall,
                unit="ratio",
                gate="recorded for later confidence gates",
                passed=True,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 7 metrics and habit artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase7"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase7/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
