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
from threed.racketsport.replay_export import audit_replay_export_manifest, inspect_glb_file, resolve_replay_glb_path
from threed.racketsport.schemas import EvalClipResult, ReplayScene, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_REPLAY_ARTIFACTS = ["replay_scene.json"]
REPLAY_GATES = {
    "players": NumericGate(name="presence_check.replay_players_min", op=">=", threshold=1, unit="players"),
    "points": NumericGate(name="presence_check.replay_points_min", op=">=", threshold=1, unit="points"),
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

        missing = missing_artifacts(run_dir, REQUIRED_REPLAY_ARTIFACTS)
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
        phase="phase10",
        evaluator="replay_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_REPLAY_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        replay_scene = validate_artifact_file("replay_scene", run_dir / "replay_scene.json")
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

    if not isinstance(replay_scene, ReplayScene):
        return EvalClipResult(
            clip=clip_name,
            run_dir=str(run_dir),
            labels_dir=str(labels_dir),
            status="fail",
            missing_label_files=[],
            missing_artifacts=[],
            metrics={},
            notes=["replay artifact did not parse as ReplayScene"],
        )

    expected_glb_refs = [("court_glb", replay_scene.court_glb), *[
        (f"points/{index}/glb_url", point.glb_url) for index, point in enumerate(replay_scene.points)
    ]]
    expected_glbs = [glb for _, glb in expected_glb_refs]
    missing_glbs: list[str] = []
    invalid_glbs: list[str] = []
    resolved_glbs: list[tuple[str, Path]] = []
    for field, glb in expected_glb_refs:
        try:
            resolved_glbs.append((glb, resolve_replay_glb_path(run_dir, glb, field=field)))
        except FileNotFoundError:
            missing_glbs.append(glb)
        except ValueError:
            invalid_glbs.append(glb)
    for glb, path in resolved_glbs:
        try:
            inspect_glb_file(path)
        except ValueError:
            invalid_glbs.append(glb)
    replay_production_ready = False
    replay_production_blockers: list[str] = []
    if not missing_glbs and not invalid_glbs:
        try:
            replay_audit = audit_replay_export_manifest(run_dir, replay_scene)
            replay_production_ready = bool(replay_audit["production_replay_ready"])
            replay_production_blockers = [str(blocker) for blocker in replay_audit["blockers"]]
        except (FileNotFoundError, ValueError) as exc:
            replay_production_blockers = [f"replay_export_manifest_invalid: {exc}"]
    player_count = len(replay_scene.players)
    point_count = len(replay_scene.points)
    largest_point_glb_mb = max((point.size_mb for point in replay_scene.points), default=0.0)
    gated = evaluate_numeric_gates(
        {
            "players": player_count,
            "points": point_count,
        },
        REPLAY_GATES,
    )
    passed = (
        all(gated_metric.passed is True for gated_metric in gated.values())
        and not missing_glbs
        and not invalid_glbs
        and replay_production_ready
    )
    notes = []
    if missing_glbs:
        notes.append(f"missing referenced GLB files: {', '.join(missing_glbs)}")
    if invalid_glbs:
        notes.append(f"invalid referenced GLB files: {', '.join(invalid_glbs)}")
    if replay_production_blockers:
        notes.append(f"production replay blockers: {', '.join(replay_production_blockers)}")
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "players": gated["players"],
            "points": gated["points"],
            "glb_files_present": metric(
                value=len(expected_glbs) - len(missing_glbs),
                unit="files",
                gate="all referenced GLB files exist",
                passed=not missing_glbs,
            ),
            "glb_files_valid": metric(
                value=len(expected_glbs) - len(missing_glbs) - len(invalid_glbs),
                unit="files",
                gate="all referenced GLB files are structurally valid",
                passed=not missing_glbs and not invalid_glbs,
            ),
            "largest_point_glb_mb": metric(
                value=largest_point_glb_mb,
                unit="MB",
                gate="recorded for later replay size gate",
                passed=True,
            ),
            "replay_production_ready": metric(
                value=replay_production_ready,
                unit=None,
                gate="production replay export requirements met",
                passed=replay_production_ready,
            ),
            "replay_production_blocker_count": metric(
                value=len(replay_production_blockers),
                unit="blockers",
                gate="production replay export blockers must be zero",
                passed=not replay_production_blockers,
            ),
        },
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 10 replay-scene artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase10"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase10/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
