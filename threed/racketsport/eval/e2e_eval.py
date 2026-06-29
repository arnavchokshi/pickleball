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


REQUIRED_E2E_ARTIFACTS = [
    "court_calibration.json",
    "court_zones.json",
    "net_plane.json",
    "court_line_evidence.json",
    "tracks.json",
    "smpl_motion.json",
    "skeleton3d.json",
    "physics_refinement.json",
    "ball_track.json",
    "contact_windows.json",
    "racket_pose.json",
    "racket_sport_metrics.json",
    "habit_report.json",
    "coach_report.json",
    "drill_report.json",
    "replay_scene.json",
]

ARTIFACT_SCHEMA_NAMES = {
    "court_calibration.json": "court_calibration",
    "court_zones.json": "court_zones",
    "net_plane.json": "net_plane",
    "court_line_evidence.json": "court_line_evidence",
    "tracks.json": "tracks",
    "smpl_motion.json": "smpl_motion",
    "skeleton3d.json": "skeleton3d",
    "physics_refinement.json": "physics_refinement",
    "ball_track.json": "ball_track",
    "contact_windows.json": "contact_windows",
    "racket_pose.json": "racket_pose",
    "racket_sport_metrics.json": "racket_sport_metrics",
    "habit_report.json": "habit_report",
    "coach_report.json": "coach_report",
    "drill_report.json": "drill_report",
    "replay_scene.json": "replay_scene",
}

E2E_ARTIFACT_GATES = {
    "required_artifacts_present": NumericGate(
        name="artifact_check.e2e_required_artifacts_present",
        op="==",
        threshold=len(REQUIRED_E2E_ARTIFACTS),
        unit="artifacts",
    ),
    "required_artifacts_total": NumericGate(
        name="artifact_check.e2e_required_artifacts_total",
        op="==",
        threshold=len(REQUIRED_E2E_ARTIFACTS),
        unit="artifacts",
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

        missing = missing_artifacts(run_dir, REQUIRED_E2E_ARTIFACTS)
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
        phase="phase11",
        evaluator="e2e_eval",
        root=root_path,
        labels_root=labels_path,
        required_artifacts=REQUIRED_E2E_ARTIFACTS,
        dataset=dataset,
        clips=results,
        notes=notes,
    )


def _evaluate_ready_clip(clip_name: str, *, run_dir: Path, labels_dir: Path) -> EvalClipResult:
    try:
        parsed = {
            artifact: validate_artifact_file(schema_name, run_dir / artifact)
            for artifact, schema_name in ARTIFACT_SCHEMA_NAMES.items()
        }
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

    replay_scene = parsed["replay_scene.json"]
    referenced_glbs = 0
    missing_glbs: list[str] = []
    invalid_glbs: list[str] = []
    replay_production_ready = False
    replay_production_blockers: list[str] = []
    if isinstance(replay_scene, ReplayScene):
        expected_glb_refs = [("court_glb", replay_scene.court_glb), *[
            (f"points/{index}/glb_url", point.glb_url) for index, point in enumerate(replay_scene.points)
        ]]
        expected_glbs = [glb for _, glb in expected_glb_refs]
        referenced_glbs = len(expected_glbs)
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
        if not missing_glbs and not invalid_glbs:
            try:
                replay_audit = audit_replay_export_manifest(run_dir, replay_scene)
                replay_production_ready = bool(replay_audit["production_replay_ready"])
                replay_production_blockers = [str(blocker) for blocker in replay_audit["blockers"]]
            except (FileNotFoundError, ValueError) as exc:
                replay_production_blockers = [f"replay_export_manifest_invalid: {exc}"]

    present_artifacts = len(REQUIRED_E2E_ARTIFACTS)
    gated = evaluate_numeric_gates(
        {
            "required_artifacts_present": present_artifacts,
            "required_artifacts_total": len(REQUIRED_E2E_ARTIFACTS),
            "referenced_glb_files_present": referenced_glbs - len(missing_glbs),
            "referenced_glb_files_valid": referenced_glbs - len(missing_glbs) - len(invalid_glbs),
        },
        {
            **E2E_ARTIFACT_GATES,
            "referenced_glb_files_present": NumericGate(
                name="artifact_check.e2e_referenced_glb_files_present",
                op="==",
                threshold=referenced_glbs,
                unit="files",
            ),
            "referenced_glb_files_valid": NumericGate(
                name="artifact_check.e2e_referenced_glb_files_valid",
                op="==",
                threshold=referenced_glbs,
                unit="files",
            ),
        },
    )
    gated["replay_production_ready"] = metric(
        value=replay_production_ready,
        unit=None,
        gate="production replay export requirements met",
        passed=replay_production_ready,
    )
    gated["replay_production_blocker_count"] = metric(
        value=len(replay_production_blockers),
        unit="blockers",
        gate="production replay export blockers must be zero",
        passed=not replay_production_blockers,
    )
    passed = all(gated_metric.passed is True for gated_metric in gated.values())
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
        metrics=gated,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 11 end-to-end artifact completeness.")
    parser.add_argument("--root", type=Path, default=Path("runs/phase11"))
    parser.add_argument("--labels", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase11/metrics.json"))
    args = parser.parse_args()

    payload = evaluate(args.root, args.labels)
    write_phase_metrics(args.out, payload)
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if payload.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
