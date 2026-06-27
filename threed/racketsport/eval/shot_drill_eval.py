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
from threed.racketsport.schemas import DrillReport, EvalClipResult, RacketSportMetrics, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_SHOT_DRILL_ARTIFACTS = ["racket_sport_metrics.json", "drill_report.json"]
SHOT_DRILL_GATES = {
    "shots": NumericGate(name="presence_check.shot_drill_shots_min", op=">=", threshold=1, unit="shots"),
    "drill_reps": NumericGate(name="presence_check.shot_drill_reps_min", op=">=", threshold=1, unit="reps"),
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

        missing = missing_artifacts(run_dir, REQUIRED_SHOT_DRILL_ARTIFACTS)
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
        phase="phase8",
        evaluator="shot_drill_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_SHOT_DRILL_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        metrics_artifact = validate_artifact_file("racket_sport_metrics", run_dir / "racket_sport_metrics.json")
        drill_report = validate_artifact_file("drill_report", run_dir / "drill_report.json")
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

    if not isinstance(metrics_artifact, RacketSportMetrics) or not isinstance(drill_report, DrillReport):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["shot/drill artifacts did not parse as RacketSportMetrics and DrillReport"],
        )

    shots = [shot for player in metrics_artifact.players for shot in player.shots]
    shot_types = ",".join(sorted({shot.type for shot in shots}))
    fault_reps = drill_report.reps - drill_report.clean_reps
    gated = evaluate_numeric_gates(
        {
            "shots": len(shots),
            "drill_reps": drill_report.reps,
        },
        SHOT_DRILL_GATES,
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
            "shots": gated["shots"],
            "shot_types": metric(
                value=shot_types,
                unit=None,
                gate="recorded for later shot-class gates",
                passed=bool(shot_types),
            ),
            "drill_reps": gated["drill_reps"],
            "clean_reps": metric(
                value=drill_report.clean_reps,
                unit="reps",
                gate="recorded for later rep-count gates",
                passed=True,
            ),
            "fault_reps": metric(
                value=fault_reps,
                unit="reps",
                gate="recorded for later drill-quality gates",
                passed=fault_reps >= 0,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 8 shot labels and drill reports.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase8"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase8/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
