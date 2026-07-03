from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_fp_player_proximity import (
    ClipProximityInput,
    bucket_ball_fps_by_player_proximity,
    bucket_ball_fps_by_player_proximity_from_files,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ball_track_payload(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "source": "wasb",
        "frames": frames,
        "bounces": [],
    }


def _reviewed_boxes_payload(*, size: int, ball_frames: dict[int, tuple[float, float, float, float]]) -> dict:
    frames = []
    for frame_index in range(size):
        boxes = []
        if frame_index in ball_frames:
            x1, y1, x2, y2 = ball_frames[frame_index]
            boxes.append(
                {
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "frame_index": frame_index,
                    "keyframe": True,
                    "label": "ball",
                    "occluded": False,
                    "source": "manual",
                    "track_id": 0,
                }
            )
        frames.append({"frame_index": frame_index, "boxes": boxes})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": "clip_a",
        "source_format": "cvat_video_1_1",
        "source_path": "synthetic",
        "task": {
            "task_id": 1,
            "name": "test",
            "size": size,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": size - 1,
            "original_size": [100, 100],
            "source": "synthetic",
        },
        "frames": frames,
        "tracks": [],
        "summary": {
            "frame_count": size,
            "visible_box_count": len(ball_frames),
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": len(ball_frames)},
        },
    }


def _tracks_payload(*, fps: float, player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]]) -> dict:
    max_players = max((len(boxes) for boxes in player_boxes_by_frame.values()), default=0)
    players = []
    for player_idx in range(max_players):
        frames = []
        for frame_idx, boxes in sorted(player_boxes_by_frame.items()):
            if player_idx >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[player_idx]
            frames.append({"t": frame_idx / fps, "bbox": [x1, y1, x2, y2], "world_xy": [0.0, 0.0], "conf": 0.9})
        players.append({"id": player_idx + 1, "side": "near", "role": "left", "frames": frames})
    return {"schema_version": 1, "fps": fps, "players": players, "rally_spans": []}


def test_ball_fp_player_proximity_buckets_on_near_and_far(tmp_path: Path) -> None:
    fps = 10.0
    # Frame 0: matches a reviewed ball label -> not a FP, excluded.
    # Frame 1: hidden FP sitting inside the player box -> on_player.
    # Frame 2: hidden FP just outside the player box by less than 1.5x diag -> near_player.
    # Frame 3: hidden FP far from every player box -> far_field.
    # Frame 4: hidden FP with no player boxes at all -> no_players_in_frame.
    frames = [
        {"t": 0 / fps, "xy": [50.0, 50.0], "conf": 0.9, "visible": True},
        {"t": 1 / fps, "xy": [15.0, 15.0], "conf": 0.8, "visible": True},
        {"t": 2 / fps, "xy": [35.0, 10.0], "conf": 0.7, "visible": True},
        {"t": 3 / fps, "xy": [500.0, 500.0], "conf": 0.6, "visible": True},
        {"t": 4 / fps, "xy": [10.0, 10.0], "conf": 0.5, "visible": True},
    ]
    ball_track_path = tmp_path / "ball_track.json"
    _write_json(ball_track_path, _ball_track_payload(frames))

    reviewed_boxes_path = tmp_path / "reviewed_boxes.json"
    _write_json(reviewed_boxes_path, _reviewed_boxes_payload(size=5, ball_frames={0: (45.0, 45.0, 55.0, 55.0)}))

    # Player box on frames 0-3 is a 20x20 box at (0,0)-(20,20), diag ~28.28
    # (on_player threshold 0.25x diag ~7.07px, near_player threshold 1.5x
    # diag ~42.43px). Frame 1's point (15,15) is inside the box -> on_player.
    # Frame 2's point (35,10) is 15px past the right edge -> near_player.
    player_box = (0.0, 0.0, 20.0, 20.0)
    tracks_path = tmp_path / "tracks.json"
    _write_json(
        tracks_path,
        _tracks_payload(
            fps=fps,
            player_boxes_by_frame={0: [player_box], 1: [player_box], 2: [player_box], 3: [player_box]},
        ),
    )

    report = bucket_ball_fps_by_player_proximity(
        clips=[
            ClipProximityInput(
                clip="clip_a",
                ball_track_path=ball_track_path,
                reviewed_boxes_path=reviewed_boxes_path,
                tracks_path=tracks_path,
            )
        ],
    )

    assert report["not_ground_truth"] is True
    assert report["source_only"] is True
    assert report["promotion_claimed"] is False
    clip_report = report["clips"]["clip_a"]
    assert clip_report["false_positive_count"] == 4
    assert clip_report["bucket_counts"] == {
        "on_player": 1,
        "near_player": 1,
        "far_field": 1,
        "no_players_in_frame": 1,
    }
    rows_by_frame = {row["frame"]: row for row in clip_report["rows"]}
    assert rows_by_frame[1]["bucket"] == "on_player"
    assert rows_by_frame[1]["distance_to_nearest_player_box_px"] == 0.0
    assert rows_by_frame[2]["bucket"] == "near_player"
    assert rows_by_frame[3]["bucket"] == "far_field"
    assert rows_by_frame[4]["bucket"] == "no_players_in_frame"
    assert rows_by_frame[4]["distance_to_nearest_player_box_px"] is None
    assert 0 not in rows_by_frame  # matched a reviewed label, not a FP

    combined = report["combined"]
    assert combined["false_positive_count"] == 4
    assert combined["bucket_fractions"]["on_player"] == 0.25


def test_ball_fp_player_proximity_excludes_frames_beyond_reviewed_label_horizon(tmp_path: Path) -> None:
    fps = 10.0
    frames = [
        {"t": 0 / fps, "xy": [10.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 5 / fps, "xy": [10.0, 10.0], "conf": 0.9, "visible": True},  # frame 5, beyond reviewed size=3
    ]
    ball_track_path = tmp_path / "ball_track.json"
    _write_json(ball_track_path, _ball_track_payload(frames))

    reviewed_boxes_path = tmp_path / "reviewed_boxes.json"
    _write_json(reviewed_boxes_path, _reviewed_boxes_payload(size=3, ball_frames={}))

    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=fps, player_boxes_by_frame={}))

    report = bucket_ball_fps_by_player_proximity(
        clips=[
            ClipProximityInput(
                clip="clip_a",
                ball_track_path=ball_track_path,
                reviewed_boxes_path=reviewed_boxes_path,
                tracks_path=tracks_path,
            )
        ],
    )

    clip_report = report["clips"]["clip_a"]
    # Only frame 0 is inside the reviewed horizon (size=3 -> frames 0,1,2).
    assert clip_report["false_positive_count"] == 1
    assert clip_report["rows"][0]["frame"] == 0


def test_ball_fp_player_proximity_from_files_matches_direct_call(tmp_path: Path) -> None:
    fps = 10.0
    frames = [{"t": 0 / fps, "xy": [10.0, 10.0], "conf": 0.9, "visible": True}]
    ball_track_path = tmp_path / "ball_track.json"
    _write_json(ball_track_path, _ball_track_payload(frames))
    reviewed_boxes_path = tmp_path / "reviewed_boxes.json"
    _write_json(reviewed_boxes_path, _reviewed_boxes_payload(size=1, ball_frames={}))
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=fps, player_boxes_by_frame={0: [(0.0, 0.0, 20.0, 20.0)]}))

    report = bucket_ball_fps_by_player_proximity_from_files(
        clips=[
            {
                "clip": "clip_a",
                "ball_track": str(ball_track_path),
                "reviewed_boxes": str(reviewed_boxes_path),
                "tracks": str(tracks_path),
            }
        ],
    )

    assert report["combined"]["false_positive_count"] == 1


def test_ball_fp_player_proximity_rejects_invalid_bucket_thresholds(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=10.0, player_boxes_by_frame={}))
    ball_track_path = tmp_path / "ball_track.json"
    _write_json(ball_track_path, _ball_track_payload([]))
    reviewed_boxes_path = tmp_path / "reviewed_boxes.json"
    _write_json(reviewed_boxes_path, _reviewed_boxes_payload(size=0, ball_frames={}))

    clip = ClipProximityInput(
        clip="clip_a",
        ball_track_path=ball_track_path,
        reviewed_boxes_path=reviewed_boxes_path,
        tracks_path=tracks_path,
    )
    try:
        bucket_ball_fps_by_player_proximity(clips=[clip], near_player_diag_fraction=0.1, on_player_diag_fraction=0.25)
    except ValueError as exc:
        assert "near_player_diag_fraction" in str(exc)
    else:
        raise AssertionError("expected rejection of near <= on threshold ordering")


def test_run_ball_fp_player_proximity_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/run_ball_fp_player_proximity.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Bucket ball-detector false positives" in completed.stdout


def test_run_ball_fp_player_proximity_cli_writes_report(tmp_path: Path) -> None:
    fps = 10.0
    ball_track_path = tmp_path / "ball_track.json"
    _write_json(ball_track_path, _ball_track_payload([{"t": 0 / fps, "xy": [10.0, 10.0], "conf": 0.9, "visible": True}]))
    reviewed_boxes_path = tmp_path / "reviewed_boxes.json"
    _write_json(reviewed_boxes_path, _reviewed_boxes_payload(size=1, ball_frames={}))
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=fps, player_boxes_by_frame={0: [(0.0, 0.0, 20.0, 20.0)]}))
    out_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_ball_fp_player_proximity.py",
            "--clip",
            f"clip_a:{ball_track_path}:{reviewed_boxes_path}:{tracks_path}",
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["combined"]["false_positive_count"] == 1
