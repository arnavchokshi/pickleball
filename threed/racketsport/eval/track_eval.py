from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import build_phase_metrics, metric, missing_artifacts, write_phase_metrics
from threed.racketsport.schemas import EvalClipResult, Tracks, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_TRACK_ARTIFACTS = ["tracks.json"]


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

        missing = missing_artifacts(run_dir, REQUIRED_TRACK_ARTIFACTS)
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
        phase="phase2",
        evaluator="track_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_TRACK_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
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

    if not isinstance(tracks, Tracks):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["tracks artifact did not parse as Tracks"],
        )

    player_count = len(tracks.players)
    frame_count = sum(len(player.frames) for player in tracks.players)
    players_passed = player_count >= 1
    frames_passed = frame_count >= 1
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if players_passed and frames_passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "players_detected": metric(
                value=player_count,
                unit="players",
                gate=">= 1",
                passed=players_passed,
            ),
            "track_frames": metric(
                value=frame_count,
                unit="frames",
                gate=">= 1",
                passed=frames_passed,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 person-track artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase2/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
