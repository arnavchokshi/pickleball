from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.frame_rating import build_frame_compute_plan


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [-1.0, -2.9], "conf": 0.41},
                ],
            },
            {
                "id": 2,
                "side": "near",
                "role": "right",
                "frames": [
                    {"t": 0.0, "bbox": [500.0, 100.0, 600.0, 300.0], "world_xy": [1.0, -3.0], "conf": 0.9},
                ],
            },
        ],
        "rally_spans": [],
    }


def _ball_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [300.0, 200.0], "conf": 0.94, "visible": True},
            {"t": 1.0 / 30.0, "xy": [310.0, 210.0], "conf": 0.22, "visible": False},
        ],
        "bounces": [],
    }


def _sparse_ball_payload() -> dict:
    payload = _ball_payload()
    payload["frames"] = payload["frames"][:1]
    return payload


def _contact_payload() -> dict:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 1.0 / 30.0,
                "frame": 1,
                "player_id": 1,
                "confidence": 0.88,
                "sources": {"audio": 0.9, "wrist_vel": 0.7, "ball_inflection": 0.65},
                "window": {"t0": 0.02, "t1": 0.08, "importance": 0.9},
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frame_compute_plan_prioritizes_contact_low_confidence_and_ball_uncertainty() -> None:
    plan = build_frame_compute_plan(
        _tracks_payload(),
        ball_track=_ball_payload(),
        contact_windows=_contact_payload(),
        expected_players=2,
    )

    assert plan["artifact_type"] == "racketsport_frame_compute_plan"
    assert plan["fps"] == 30.0
    assert [frame["frame_idx"] for frame in plan["frames"]] == [0, 1]
    assert plan["frames"][1]["score"] > plan["frames"][0]["score"]
    assert plan["frames"][1]["recommended_tier"] == "deep_mesh"
    assert plan["frames"][1]["reasons"] == [
        "contact_window",
        "low_track_confidence",
        "ball_uncertain",
    ]
    assert plan["frames"][1]["target_representation"] == "world_mesh"
    assert plan["frames"][0]["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.92,
            "score": 0.55,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window"],
        },
        {
            "player_id": 2,
            "track_conf": 0.9,
            "score": 0.0,
            "recommended_tier": "baseline",
            "target_representation": "track_only",
            "reasons": [],
        },
    ]
    assert plan["frames"][1]["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.41,
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "low_track_confidence", "ball_uncertain"],
        }
    ]
    assert plan["deep_mesh_windows"] == [
        {
            "frame_start": 0,
            "frame_end": 1,
            "t0": 0.0,
            "t1": 2.0 / 30.0,
            "frame_count": 2,
            "target_representation": "world_mesh",
            "fallback_representation": "skeleton_preview",
            "target_player_ids": [1],
            "reason_counts": {
                "ball_uncertain": 1,
                "contact_window": 2,
                "low_track_confidence": 1,
            },
            "max_score": 1.0,
        }
    ]
    assert plan["summary"]["deep_mesh_window_count"] == 1
    assert plan["summary"]["by_player_target_representation"] == {"track_only": 1, "world_mesh": 2}


def test_frame_compute_plan_marks_missing_expected_players_for_human_review() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]

    plan = build_frame_compute_plan(tracks, expected_players=4)

    assert plan["frames"][0]["active_players"] == 1
    assert "missing_expected_players" in plan["frames"][0]["reasons"]
    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert plan["frames"][0]["player_targets"][0]["target_representation"] == "manual_review_required"
    assert plan["frames"][0]["player_targets"][0]["reasons"] == ["missing_expected_players"]
    assert plan["summary"]["by_player_target_representation"] == {"manual_review_required": 2}


def test_frame_compute_plan_schedules_reviewed_assigned_contact_despite_incomplete_coverage() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]
    contacts = _contact_payload()
    contacts["events"][0]["sources"] = {
        "audio": 0.0,
        "wrist_vel": 0.0,
        "ball_inflection": 0.0,
        "human_review": 1.0,
    }

    plan = build_frame_compute_plan(tracks, contact_windows=contacts, expected_players=4)

    frame_zero = plan["frames"][0]
    assert frame_zero["recommended_tier"] == "deep_mesh"
    assert frame_zero["target_representation"] == "world_mesh"
    assert frame_zero["reasons"] == [
        "contact_window",
        "missing_expected_players",
        "reviewed_contact_targeted_body",
    ]
    assert frame_zero["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.92,
            "score": 0.55,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "reviewed_contact_targeted_body"],
        }
    ]
    assert plan["deep_mesh_windows"] == [
        {
            "frame_start": 0,
            "frame_end": 1,
            "t0": 0.0,
            "t1": 2.0 / 30.0,
            "frame_count": 2,
            "target_representation": "world_mesh",
            "fallback_representation": "skeleton_preview",
            "target_player_ids": [1],
            "reason_counts": {
                "contact_window": 2,
                "low_track_confidence": 1,
                "missing_expected_players": 2,
                "reviewed_contact_targeted_body": 2,
            },
            "max_score": 0.8,
        }
    ]
    assert plan["summary"]["targeted_reviewed_contact_frame_count"] == 2
    assert plan["summary"]["coverage_incomplete_deep_mesh_frame_count"] == 2
    assert plan["summary"]["human_review_frame_count"] == 0


def test_frame_compute_plan_keeps_unassigned_reviewed_contact_fail_closed_when_coverage_incomplete() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]
    contacts = _contact_payload()
    contacts["events"][0]["player_id"] = None
    contacts["events"][0]["sources"] = {
        "audio": 0.0,
        "wrist_vel": 0.0,
        "ball_inflection": 0.0,
        "human_review": 1.0,
    }

    plan = build_frame_compute_plan(tracks, contact_windows=contacts, expected_players=4)

    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert "reviewed_contact_targeted_body" not in plan["frames"][0]["reasons"]
    assert plan["deep_mesh_windows"] == []
    assert plan["summary"]["targeted_reviewed_contact_frame_count"] == 0


def test_frame_compute_plan_keeps_machine_contact_fail_closed_when_coverage_incomplete() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]

    plan = build_frame_compute_plan(tracks, contact_windows=_contact_payload(), expected_players=4)

    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert "reviewed_contact_targeted_body" not in plan["frames"][0]["reasons"]
    assert plan["deep_mesh_windows"] == []


def test_frame_compute_plan_marks_omitted_ball_frames_as_uncertain() -> None:
    plan = build_frame_compute_plan(
        _tracks_payload(),
        ball_track=_sparse_ball_payload(),
        expected_players=2,
    )

    frame_one = next(frame for frame in plan["frames"] if frame["frame_idx"] == 1)
    assert frame_one["ball_conf"] is None
    assert "ball_missing" in frame_one["reasons"]
    assert frame_one["recommended_tier"] == "skeleton_preview"
    assert plan["summary"]["by_reason"]["ball_missing"] == 1


def test_frame_compute_plan_cli_writes_json(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    ball = _write_json(tmp_path / "ball_track.json", _ball_payload())
    contacts = _write_json(tmp_path / "contact_windows.json", _contact_payload())
    out = tmp_path / "frame_compute_plan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_frame_compute_plan.py",
            "--tracks",
            str(tracks),
            "--ball-track",
            str(ball),
            "--contact-windows",
            str(contacts),
            "--expected-players",
            "2",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["frame_count"] == 2
    assert summary["summary"]["deep_mesh_window_count"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["frames"][1]["recommended_tier"] == "deep_mesh"
