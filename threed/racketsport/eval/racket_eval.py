from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import build_phase_metrics, metric, missing_artifacts, write_phase_metrics
from threed.racketsport.schemas import EvalClipResult, RacketPose, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_RACKET_ARTIFACTS = ["racket_pose.json"]


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

        missing = missing_artifacts(run_dir, REQUIRED_RACKET_ARTIFACTS)
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
        phase="phase6",
        evaluator="racket_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_RACKET_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        racket_pose = validate_artifact_file("racket_pose", run_dir / "racket_pose.json")
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

    if not isinstance(racket_pose, RacketPose):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["racket artifact did not parse as RacketPose"],
        )

    player_count = len(racket_pose.players)
    frame_count = sum(len(player.frames) for player in racket_pose.players)
    contact_count = sum(len(player.contacts) for player in racket_pose.players)
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
            "racket_players": metric(value=player_count, unit="players", gate=">= 1", passed=players_passed),
            "racket_frames": metric(value=frame_count, unit="frames", gate=">= 1", passed=frames_passed),
            "racket_contacts": metric(
                value=contact_count,
                unit="contacts",
                gate="recorded for later face/contact gates",
                passed=True,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 racket-pose artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase6"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase6/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
