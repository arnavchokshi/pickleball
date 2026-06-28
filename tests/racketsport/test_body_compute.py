from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [-1.0, -2.9], "conf": 0.91},
                ],
            }
        ],
        "rally_spans": [],
    }


def _frame_plan_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 2,
        "frames": [
            {
                "frame_idx": 0,
                "t": 0.0,
                "score": 0.8,
                "recommended_tier": "human_review",
                "target_representation": "manual_review_required",
                "reasons": ["missing_expected_players"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.92,
                "ball_conf": None,
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.92,
                        "score": 0.8,
                        "recommended_tier": "human_review",
                        "target_representation": "manual_review_required",
                        "reasons": ["missing_expected_players"],
                    }
                ],
            },
            {
                "frame_idx": 1,
                "t": 1.0 / 30.0,
                "score": 0.75,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["contact_window"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.91,
                "ball_conf": 0.6,
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.91,
                        "score": 0.75,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["contact_window"],
                    }
                ],
            },
        ],
        "deep_mesh_windows": [
            {
                "frame_start": 1,
                "frame_end": 1,
                "t0": 1.0 / 30.0,
                "t1": 2.0 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "skeleton_preview",
                "target_player_ids": [7],
                "reason_counts": {"contact_window": 1},
                "max_score": 0.75,
            }
        ],
        "summary": {
            "by_tier": {"deep_mesh": 1, "human_review": 1},
            "by_reason": {"contact_window": 1, "missing_expected_players": 1},
            "max_score": 0.8,
            "deep_mesh_window_count": 1,
            "deep_mesh_frame_count": 1,
            "human_review_frame_count": 1,
        },
    }


def _targeted_reviewed_contact_frame_plan_payload() -> dict:
    payload = _frame_plan_payload()
    payload["expected_players"] = 4
    payload["frames"] = [
        {
            "frame_idx": 1,
            "t": 1.0 / 30.0,
            "score": 0.75,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "missing_expected_players", "reviewed_contact_targeted_body"],
            "active_players": 1,
            "active_player_ids": [7],
            "missing_players": 3,
            "min_track_conf": 0.91,
            "ball_conf": None,
            "player_targets": [
                {
                    "player_id": 7,
                    "track_conf": 0.91,
                    "score": 0.55,
                    "recommended_tier": "deep_mesh",
                    "target_representation": "world_mesh",
                    "reasons": ["contact_window", "reviewed_contact_targeted_body"],
                }
            ],
        }
    ]
    payload["deep_mesh_windows"] = [
        {
            "frame_start": 1,
            "frame_end": 1,
            "t0": 1.0 / 30.0,
            "t1": 2.0 / 30.0,
            "frame_count": 1,
            "target_representation": "world_mesh",
            "fallback_representation": "skeleton_preview",
            "target_player_ids": [7],
            "reason_counts": {
                "contact_window": 1,
                "missing_expected_players": 1,
                "reviewed_contact_targeted_body": 1,
            },
            "max_score": 0.75,
        }
    ]
    payload["summary"] = {
        "by_tier": {"deep_mesh": 1},
        "by_reason": {
            "contact_window": 1,
            "missing_expected_players": 1,
            "reviewed_contact_targeted_body": 1,
        },
        "by_player_target_representation": {"world_mesh": 1},
        "max_score": 0.75,
        "deep_mesh_window_count": 1,
        "deep_mesh_frame_count": 1,
        "human_review_frame_count": 0,
        "targeted_reviewed_contact_frame_count": 1,
        "coverage_incomplete_deep_mesh_frame_count": 1,
    }
    return payload


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_body_compute_execution_cli_writes_adaptive_manifest(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _frame_plan_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--frame-compute-plan",
            str(frame_plan),
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
    assert summary["summary"]["scheduled_frame_count"] == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_body_compute_execution"
    assert payload["mode"] == "adaptive_frame_compute_plan"
    assert payload["scheduled_frames"][0]["frame_idx"] == 1
    assert payload["scheduled_frames"][0]["player_targets"] == [
        {
            "player_id": 7,
            "track_conf": 0.91,
            "score": 0.75,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window"],
        }
    ]
    assert payload["skipped_frames"][0]["player_targets"] == [
        {
            "player_id": 7,
            "track_conf": 0.92,
            "score": 0.8,
            "recommended_tier": "human_review",
            "target_representation": "manual_review_required",
            "reasons": ["missing_expected_players"],
        }
    ]
    assert payload["summary"]["skipped_by_tier"] == {"human_review": 1}
    assert payload["summary"]["skipped_by_target_representation"] == {"manual_review_required": 1}
    assert payload["summary"]["skipped_by_reason"] == {"missing_expected_players": 1}


def test_body_compute_execution_reports_targeted_reviewed_contact_schedule(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _targeted_reviewed_contact_frame_plan_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--frame-compute-plan",
            str(frame_plan),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scheduled_frames"][0]["target_player_ids"] == [7]
    assert payload["scheduled_frames"][0]["active_player_ids"] == [7]
    assert payload["scheduled_frames"][0]["reasons"] == [
        "contact_window",
        "missing_expected_players",
        "reviewed_contact_targeted_body",
    ]
    assert payload["summary"]["scheduled_by_reason"] == {
        "contact_window": 1,
        "missing_expected_players": 1,
        "reviewed_contact_targeted_body": 1,
    }
    assert payload["summary"]["scheduled_by_target_representation"] == {"world_mesh": 1}
    assert payload["summary"]["scheduled_coverage_incomplete_frame_count"] == 1
    assert payload["summary"]["scheduled_targeted_reviewed_contact_frame_count"] == 1
