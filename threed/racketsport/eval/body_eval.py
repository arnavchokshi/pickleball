from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import build_phase_metrics, metric, missing_artifacts, write_phase_metrics
from threed.racketsport.schemas import EvalClipResult, Skeleton3D, SmplMotion, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_BODY_ARTIFACTS = ["smpl_motion.json", "skeleton3d.json"]


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

        missing = missing_artifacts(run_dir, REQUIRED_BODY_ARTIFACTS)
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
        phase="phase3",
        evaluator="body_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_BODY_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        smpl_motion = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
        skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
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

    if not isinstance(smpl_motion, SmplMotion) or not isinstance(skeleton, Skeleton3D):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["body artifacts did not parse as SmplMotion and Skeleton3D"],
        )

    smpl_players = len(smpl_motion.players)
    smpl_frames = sum(len(player.frames) for player in smpl_motion.players)
    skeleton_players = len(skeleton.players)
    passed = smpl_players >= 1 and smpl_frames >= 1 and skeleton_players >= 1
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "smpl_players": metric(value=smpl_players, unit="players", gate=">= 1", passed=smpl_players >= 1),
            "smpl_frames": metric(value=smpl_frames, unit="frames", gate=">= 1", passed=smpl_frames >= 1),
            "skeleton_players": metric(
                value=skeleton_players,
                unit="players",
                gate=">= 1",
                passed=skeleton_players >= 1,
            ),
        },
        notes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 body artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase3/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
