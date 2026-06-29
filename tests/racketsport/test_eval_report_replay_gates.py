from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.eval import copy_faithfulness, metric_eval, replay_eval, shot_drill_eval
from threed.racketsport.replay_export import build_replay_review_export_from_virtual_world, write_replay_scene


REQUIRED_LABEL_FILES = (
    "court_corners.json",
    "players.json",
    "feet_nvz.json",
    "ball.json",
    "events.json",
    "racket_pose.json",
    "foot_contact.json",
    "coach_habits.json",
    "manual_metrics.json",
)


def _write_ready_clip(labels_root: Path, name: str) -> None:
    labels_dir = labels_root / name / "labels"
    labels_dir.mkdir(parents=True)
    for label in REQUIRED_LABEL_FILES:
        (labels_dir / label).write_text("{}", encoding="utf-8")


def _write_partial_clip(labels_root: Path, name: str) -> None:
    labels_dir = labels_root / name / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "court_corners.json").write_text("{}", encoding="utf-8")


def _metric_payload(*, players: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "players": players
        if players is not None
        else [
            {
                "id": 1,
                "shots": [
                    {
                        "t": 0.25,
                        "type": "dink",
                        "type_conf": 0.82,
                        "top2": [{"type": "dink", "confidence": 0.82}, {"type": "reset_block", "confidence": 0.11}],
                        "metrics": {
                            "nvz_margin_ft": {"value": -0.5, "conf": 0.86, "frames": 5},
                            "paddle_face_deg": {"value": 4.0, "conf": 0.82, "gated": False},
                        },
                    }
                ],
            }
        ],
    }


def _habit_report_payload(*, habits: list[dict] | None = None, priority_habit_id: str = "kitchen_foot") -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "coverage": {"overall": 0.82, "skipped_reason_counts": {"ball_uncertain": 1}},
        "priority_habit_id": priority_habit_id,
        "replay_ref": {"glb_url": "/replay/clip_001?point=1"},
        "habits": habits
        if habits is not None
        else [
            {
                "id": "kitchen_foot",
                "title": "Kitchen foot",
                "summary": "Foot crossed the NVZ line.",
                "confidence": 0.86,
                "clip_ref": {"t0_sec": 0.2, "t1_sec": 1.2},
                "cue": "Stay balanced before contact.",
                "drill": {"name": "Kitchen reset", "duration_min": 6.0},
            }
        ],
    }


def _write_metric_artifacts(run_dir: Path, *, metrics: dict | None = None, habit_report: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "racket_sport_metrics.json").write_text(
        json.dumps(metrics if metrics is not None else _metric_payload()),
        encoding="utf-8",
    )
    (run_dir / "habit_report.json").write_text(
        json.dumps(habit_report if habit_report is not None else _habit_report_payload()),
        encoding="utf-8",
    )


def _write_shot_drill_artifacts(run_dir: Path, *, metrics: dict | None = None, reps: int = 3) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "racket_sport_metrics.json").write_text(
        json.dumps(metrics if metrics is not None else _metric_payload()),
        encoding="utf-8",
    )
    (run_dir / "drill_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "drill": "kitchen_dinks",
                "reps": reps,
                "clean_reps": 2 if reps >= 2 else reps,
                "per_rep": [
                    {"t": float(idx), "quality": "clean" if idx % 2 == 0 else "fault", "reasons": []}
                    for idx in range(reps)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_reviewed_shot_labels(labels_root: Path, name: str, *, shot_label: str = "dink") -> None:
    labels_dir = labels_root / name / "labels"
    (labels_dir / "events.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "human_reviewed",
                "not_ground_truth": False,
                "annotation": {
                    "target_file": "events.json",
                    "items": [
                        {
                            "id": "truth_shot_001",
                            "status": "accepted",
                            "t": 0.25,
                            "frame_index": 15,
                            "player_id": 1,
                            "shot_label": shot_label,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_copy_artifacts(
    run_dir: Path,
    *,
    habit_report: dict | None = None,
    coach_report: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "habit_report.json").write_text(
        json.dumps(habit_report if habit_report is not None else _habit_report_payload()),
        encoding="utf-8",
    )
    (run_dir / "coach_report.json").write_text(
        json.dumps(coach_report if coach_report is not None else _habit_report_payload()),
        encoding="utf-8",
    )


def _write_replay_artifacts(run_dir: Path, *, players: list[int] | None = None, points: list[dict] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    scene = build_replay_review_export_from_virtual_world(
        _replay_virtual_world_payload(players=players if players is not None else [1, 2]),
        export_root=run_dir,
        point_id=3,
    )
    if points is not None:
        payload = scene.model_dump(mode="json")
        payload["points"] = points
        scene = payload
    write_replay_scene(run_dir / "replay_scene.json", scene)


def _replay_virtual_world_payload(*, players: list[int]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "line_segments": {"baseline": [[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]]},
            "net": {"endpoints": [[-3.05, 6.705, 0.91], [3.05, 6.705, 0.91]]},
        },
        "players": [
            {
                "id": player_id,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "track_world_xy": [float(player_id), 1.0], "joints_world": [[float(player_id), 1.0, 1.2]]},
                    {"t": 1.0, "track_world_xy": [float(player_id), 2.0], "joints_world": [[float(player_id), 2.0, 1.1]]},
                ],
            }
            for player_id in players
        ],
        "ball": {"source": "fixture", "frames": [{"t": 0.0, "world_xyz": [0.0, 6.0, 0.3], "visible": True}]},
        "paddles": [
            {
                "player_id": players[0],
                "frames": [
                    {
                        "t": 0.0,
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.2, 0.0]],
                        "mesh_faces": [[0, 1, 2]],
                    }
                ],
            }
        ],
    }


def _run_eval(evaluator: object, tmp_path: Path) -> object:
    return evaluator.evaluate(tmp_path / "runs", tmp_path / "data" / "testclips")


def test_metric_eval_uses_named_numeric_gates_for_existing_count_thresholds(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_metric_artifacts(root / "clip_001")

    payload = _run_eval(metric_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert metrics["metric_players"].gate == "presence_check.metric_players_min: >= 1"
    assert metrics["shots"].gate == "presence_check.metric_shots_min: >= 1"
    assert metrics["metric_values"].gate == "presence_check.metric_values_min: >= 1"
    assert metrics["habits"].gate == "presence_check.metric_habits_min: >= 1"
    assert all(metrics[name].passed is True for name in ("metric_players", "shots", "metric_values", "habits"))
    assert metrics["coverage_overall"].gate == "recorded for later confidence gates"


def test_metric_eval_fails_named_numeric_gate_when_count_is_below_threshold(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_metric_artifacts(root / "clip_001", metrics=_metric_payload(players=[]))

    payload = _run_eval(metric_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert payload.status == "fail"
    assert metrics["metric_players"].gate == "presence_check.metric_players_min: >= 1"
    assert metrics["metric_players"].passed is False
    assert metrics["shots"].gate == "presence_check.metric_shots_min: >= 1"
    assert metrics["shots"].passed is False


def test_shot_drill_eval_uses_named_numeric_gates_for_shot_and_rep_thresholds(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_shot_drill_artifacts(root / "clip_001")

    payload = _run_eval(shot_drill_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert metrics["shots"].gate == "presence_check.shot_drill_shots_min: >= 1"
    assert metrics["drill_reps"].gate == "presence_check.shot_drill_reps_min: >= 1"
    assert metrics["shots"].passed is True
    assert metrics["drill_reps"].passed is True
    assert metrics["shot_types"].gate == "recorded for later shot-class gates"


def test_shot_drill_eval_reports_reviewed_event_level_shot_metrics(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_reviewed_shot_labels(labels_root, "clip_001", shot_label="dink")
    _write_shot_drill_artifacts(root / "clip_001")

    payload = _run_eval(shot_drill_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert payload.status == "pass"
    assert metrics["shot_label_macro_f1"].gate == "label_check.shot_macro_f1: >= 0.65"
    assert metrics["shot_label_macro_f1"].value == 1.0
    assert metrics["shot_label_top2_accuracy"].gate == "label_check.shot_top2_accuracy: >= 0.85"
    assert metrics["shot_label_top2_accuracy"].value == 1.0
    assert metrics["shot_label_unknown_rate"].value == 0.0
    assert metrics["shot_label_gated_rate"].value == 0.0


def test_copy_faithfulness_uses_named_numeric_gates_for_report_habit_counts(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_copy_artifacts(root / "clip_001")

    payload = _run_eval(copy_faithfulness, tmp_path)
    metrics = payload.clips[0].metrics

    assert metrics["habit_count"].gate == "presence_check.copy_habit_count_min: >= 1"
    assert metrics["coach_habit_count"].gate == "presence_check.copy_coach_habit_count_min: >= 1"
    assert metrics["habit_count"].passed is True
    assert metrics["coach_habit_count"].passed is True
    assert metrics["priority_habit_match"].gate == "coach copy preserves priority habit"


def test_replay_eval_uses_named_numeric_gates_for_player_and_point_counts(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_replay_artifacts(root / "clip_001")

    payload = _run_eval(replay_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert metrics["players"].gate == "presence_check.replay_players_min: >= 1"
    assert metrics["points"].gate == "presence_check.replay_points_min: >= 1"
    assert metrics["players"].passed is True
    assert metrics["points"].passed is True
    assert metrics["glb_files_present"].gate == "all referenced GLB files exist"
    assert metrics["glb_files_valid"].gate == "all referenced GLB files are structurally valid"
    assert metrics["glb_files_valid"].passed is True


def test_replay_eval_fails_when_referenced_glb_is_invalid(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_replay_artifacts(root / "clip_001")
    (root / "clip_001" / "points" / "point_003_review.glb").write_bytes(b"glb")

    payload = _run_eval(replay_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert payload.status == "fail"
    assert metrics["glb_files_present"].passed is True
    assert metrics["glb_files_valid"].passed is False
    assert "invalid referenced GLB files: points/point_003_review.glb" in payload.clips[0].notes[0]


def test_replay_eval_fails_when_referenced_glb_escapes_run_dir(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs"
    _write_ready_clip(labels_root, "clip_001")
    _write_replay_artifacts(root / "clip_001")
    replay_scene_path = root / "clip_001" / "replay_scene.json"
    replay_scene = json.loads(replay_scene_path.read_text(encoding="utf-8"))
    replay_scene["points"][0]["glb_url"] = "../outside.glb"
    replay_scene_path.write_text(json.dumps(replay_scene), encoding="utf-8")

    payload = _run_eval(replay_eval, tmp_path)
    metrics = payload.clips[0].metrics

    assert payload.status == "fail"
    assert metrics["glb_files_present"].passed is True
    assert metrics["glb_files_valid"].passed is False
    assert "invalid referenced GLB files: ../outside.glb" in payload.clips[0].notes[0]


@pytest.mark.parametrize("evaluator", [metric_eval, shot_drill_eval, copy_faithfulness, replay_eval])
def test_incomplete_data1_labels_still_short_circuit_to_not_measured(tmp_path: Path, evaluator: object) -> None:
    labels_root = tmp_path / "data" / "testclips"
    _write_partial_clip(labels_root, "clip_001")

    payload = _run_eval(evaluator, tmp_path)

    assert payload.status == "not_measured"
    assert payload.summary.evaluated_clips == 0
    assert payload.clips[0].status == "not_measured"
    assert payload.clips[0].metrics == {}
    assert "players.json" in payload.clips[0].missing_label_files
