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
from threed.racketsport.schemas import EvalClipResult, SmplMotion, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_PHYSICS_ARTIFACTS = ["smpl_motion.json"]
PHYSICS_GATES = {
    "smpl_players": NumericGate(
        name="presence_check.physics_smpl_players_min",
        op=">=",
        threshold=1,
        unit="players",
    ),
    "smpl_frames": NumericGate(
        name="presence_check.physics_smpl_frames_min",
        op=">=",
        threshold=1,
        unit="frames",
    ),
    "foot_contact_frames": NumericGate(
        name="presence_check.physics_foot_contact_frames_min",
        op=">=",
        threshold=1,
        unit="frames",
    ),
    "skate_free_players": NumericGate(
        name="presence_check.physics_skate_free_players_min",
        op=">=",
        threshold=1,
        unit="players",
    ),
    "grf_frames": NumericGate(
        name="presence_check.physics_grf_frames_min",
        op=">=",
        threshold=1,
        unit="frames",
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

        missing = missing_artifacts(run_dir, REQUIRED_PHYSICS_ARTIFACTS)
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
        phase="phase4",
        evaluator="physics_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_PHYSICS_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        smpl_motion = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
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

    if not isinstance(smpl_motion, SmplMotion):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["physics artifact did not parse as SmplMotion"],
        )

    player_count = len(smpl_motion.players)
    frame_count = sum(len(player.frames) for player in smpl_motion.players)
    foot_contact_frames = sum(
        1
        for player in smpl_motion.players
        for frame in player.frames
        if frame.foot_contact.left or frame.foot_contact.right
    )
    skate_free_players = sum(1 for player in smpl_motion.players if player.skate_free)
    physics_modes = ",".join(sorted({player.physics for player in smpl_motion.players if player.physics}))
    grf_frames = sum(
        1 for player in smpl_motion.players for frame in player.frames if frame.grf is not None and len(frame.grf) > 0
    )
    metrics = evaluate_numeric_gates(
        {
            "smpl_players": player_count,
            "smpl_frames": frame_count,
            "foot_contact_frames": foot_contact_frames,
            "skate_free_players": skate_free_players,
            "grf_frames": grf_frames,
        },
        PHYSICS_GATES,
    )
    passed = all(gated_metric.passed is True for gated_metric in metrics.values())
    metrics["physics_modes"] = metric(
        value=physics_modes,
        unit=None,
        gate="recorded for later physics gates",
        passed=bool(physics_modes),
    )
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
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 foot-lock and physics artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase4/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
