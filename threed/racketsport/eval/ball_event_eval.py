from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import (
    NumericGate,
    build_phase_metrics,
    evaluate_numeric_gates,
    missing_artifacts,
    write_phase_metrics,
)
from threed.racketsport.schemas import BallTrack, ContactWindows, EvalClipResult, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_BALL_EVENT_ARTIFACTS = ["ball_track.json", "contact_windows.json"]
BALL_EVENT_GATES = {
    "ball_frames": NumericGate(
        name="ball_frames_min",
        op=">=",
        threshold=1,
        unit="frames",
    ),
    "contact_events": NumericGate(
        name="ball_contact_events_recorded",
        op=">=",
        threshold=0,
        unit="events",
    ),
    "bounce_events": NumericGate(
        name="ball_bounce_events_recorded",
        op=">=",
        threshold=0,
        unit="events",
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

        missing = missing_artifacts(run_dir, REQUIRED_BALL_EVENT_ARTIFACTS)
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
        phase="phase5",
        evaluator="ball_event_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_BALL_EVENT_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        ball_track = validate_artifact_file("ball_track", run_dir / "ball_track.json")
        contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
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

    if not isinstance(ball_track, BallTrack) or not isinstance(contact_windows, ContactWindows):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["ball artifacts did not parse as BallTrack and ContactWindows"],
        )

    ball_frames = len(ball_track.frames)
    contact_events = len(contact_windows.events)
    bounce_events = len(ball_track.bounces)
    metrics = evaluate_numeric_gates(
        {
            "ball_frames": ball_frames,
            "contact_events": contact_events,
            "bounce_events": bounce_events,
        },
        BALL_EVENT_GATES,
    )
    passed = all(gated_metric.passed is True for gated_metric in metrics.values())
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics=metrics,
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 ball/event artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase5/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
