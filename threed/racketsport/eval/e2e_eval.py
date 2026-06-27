from __future__ import annotations

import argparse
import json
from pathlib import Path

from threed.racketsport.eval.metrics import build_phase_metrics, metric, missing_artifacts, write_phase_metrics
from threed.racketsport.schemas import EvalClipResult, ReplayScene, validate_artifact_file
from threed.racketsport.testclips import build_testclip_manifest


REQUIRED_E2E_ARTIFACTS = [
    "court_calibration.json",
    "court_zones.json",
    "net_plane.json",
    "tracks.json",
    "smpl_motion.json",
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
    "tracks.json": "tracks",
    "smpl_motion.json": "smpl_motion",
    "ball_track.json": "ball_track",
    "contact_windows.json": "contact_windows",
    "racket_pose.json": "racket_pose",
    "racket_sport_metrics.json": "racket_sport_metrics",
    "habit_report.json": "habit_report",
    "coach_report.json": "coach_report",
    "drill_report.json": "drill_report",
    "replay_scene.json": "replay_scene",
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
    if isinstance(replay_scene, ReplayScene):
        expected_glbs = [replay_scene.court_glb, *[point.glb_url for point in replay_scene.points]]
        referenced_glbs = len(expected_glbs)
        missing_glbs = [glb for glb in expected_glbs if not (run_dir / glb).is_file()]

    present_artifacts = len(REQUIRED_E2E_ARTIFACTS)
    passed = present_artifacts == len(REQUIRED_E2E_ARTIFACTS) and not missing_glbs
    notes = [f"missing referenced GLB files: {', '.join(missing_glbs)}"] if missing_glbs else []
    return EvalClipResult(
        clip=clip_name,
        run_dir=str(run_dir),
        labels_dir=str(labels_dir),
        status="pass" if passed else "fail",
        missing_label_files=[],
        missing_artifacts=[],
        metrics={
            "required_artifacts_present": metric(
                value=present_artifacts,
                unit="artifacts",
                gate="all required end-to-end artifacts exist",
                passed=present_artifacts == len(REQUIRED_E2E_ARTIFACTS),
            ),
            "required_artifacts_total": metric(
                value=len(REQUIRED_E2E_ARTIFACTS),
                unit="artifacts",
                gate="all required end-to-end artifacts exist",
                passed=True,
            ),
            "referenced_glb_files_present": metric(
                value=referenced_glbs - len(missing_glbs),
                unit="files",
                gate="all replay GLB references exist",
                passed=not missing_glbs,
            ),
        },
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
