from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_failure_taxonomy import build_ball_failure_taxonomy


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ball_track() -> dict:
    return {
        "schema_version": 1,
        "source": "wasb",
        "fps": 10.0,
        "frames": [
            {"t": 0.0, "xy": [15.0, 15.0], "conf": 0.91, "visible": True},
            {"t": 0.1, "xy": [145.0, 105.0], "conf": 0.84, "visible": True},
            {"t": 0.2, "xy": [0.0, 0.0], "conf": 0.12, "visible": False},
            {"t": 0.3, "xy": [55.0, 55.0], "conf": 0.77, "visible": True},
        ],
        "bounces": [],
    }


def _cvat_annotations() -> dict:
    frames = [
        {
            "frame_index": 0,
            "boxes": [
                _box(label="ball", frame=0, xyxy=[10.0, 10.0, 20.0, 20.0], track_id=1),
            ],
        },
        {
            "frame_index": 1,
            "boxes": [
                _box(label="ball", frame=1, xyxy=[100.0, 100.0, 110.0, 110.0], track_id=1),
            ],
        },
        {
            "frame_index": 2,
            "boxes": [
                _box(label="ball", frame=2, xyxy=[200.0, 200.0, 260.0, 260.0], track_id=1, occluded=True),
            ],
        },
        {
            "frame_index": 3,
            "boxes": [
                _box(label="player", frame=3, xyxy=[40.0, 40.0, 70.0, 90.0], track_id=2),
            ],
        },
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": "fixture_clip",
        "source_format": "cvat_video_1_1",
        "source_path": "fixture.zip",
        "task": {
            "task_id": 1,
            "name": "fixture",
            "size": 4,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": 3,
            "original_size": [320, 240],
            "source": "fixture",
            "dumped": "2026-07-03T00:00:00Z",
        },
        "frames": frames,
        "tracks": [
            {
                "track_id": 1,
                "label": "ball",
                "visible_box_count": 3,
                "outside_box_count": 0,
                "keyframe_count": 3,
                "first_visible_frame": 0,
                "last_visible_frame": 2,
            }
        ],
        "summary": {
            "frame_count": 4,
            "visible_box_count": 4,
            "outside_box_count": 0,
            "labels": ["ball", "player"],
            "track_count_by_label": {"ball": 1, "player": 1},
            "visible_box_count_by_label": {"ball": 3, "player": 1},
        },
    }


def _box(*, label: str, frame: int, xyxy: list[float], track_id: int, occluded: bool = False) -> dict:
    x1, y1, x2, y2 = xyxy
    return {
        "track_id": track_id,
        "label": label,
        "frame_index": frame,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "keyframe": True,
        "occluded": occluded,
        "source": "manual",
    }


def test_build_ball_failure_taxonomy_classifies_reviewed_frames(tmp_path: Path) -> None:
    track_path = _write_json(tmp_path / "ball_track.json", _ball_track())
    labels_path = _write_json(tmp_path / "reviewed_boxes.json", _cvat_annotations())

    taxonomy = build_ball_failure_taxonomy(
        ball_track_path=track_path,
        cvat_labels_path=labels_path,
        candidate_name="fixture_wasb",
        f1_radius_px=20.0,
        teleport_px_per_frame=1000.0,
    )

    assert taxonomy["summary"]["class_counts"] == {
        "far_camera": 2,
        "hidden_false_positive": 1,
        "likely_player_or_paddle": 1,
        "near_camera": 1,
        "occluded_or_contact": 1,
        "visible_hit": 1,
        "visible_mislocalized": 1,
        "visible_miss": 1,
    }
    assert taxonomy["summary"]["actionable_failure_counts"] == {
        "hidden_false_positive": 1,
        "visible_mislocalized": 1,
        "visible_miss": 1,
    }
    assert [frame["primary_class"] for frame in taxonomy["frames"]] == [
        "visible_hit",
        "visible_mislocalized",
        "visible_miss",
        "hidden_false_positive",
    ]
    assert taxonomy["frames"][1]["error_px"] > 20.0
    assert taxonomy["frames"][2]["classes"] == ["visible_miss", "near_camera", "occluded_or_contact"]
    assert taxonomy["blocked_classes"]["likely_line_glint"] == "requires court-line geometry or image evidence"


def test_build_ball_failure_taxonomy_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_failure_taxonomy.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a frame-level BALL failure taxonomy" in completed.stdout
    assert "--ball-track" in completed.stdout
    assert "--cvat-labels" in completed.stdout
    assert "--candidate" in completed.stdout
    assert "--out-json" in completed.stdout
