from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_identity_filter import (
    filter_ball_track_with_click_anchors,
    load_ball_click_review,
)
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _write_track(path: Path) -> None:
    frames = []
    for index in range(6):
        xy = [float(index * 10), 0.0]
        if index == 2:
            xy = [500.0, 500.0]
        visible = index != 1
        frames.append({"t": index / 30.0, "xy": xy, "conf": 0.8 if visible else 0.0, "visible": visible})
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def _write_clicks(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_click_review",
                "status": "human_reviewed",
                "clip": "clip_a",
                "target_file": "ball.json",
                "coordinate_frame": "image_pixels_video_space",
                "items": [
                    {
                        "review_id": "ball_frame_000000",
                        "frame_index": 0,
                        "t": 0.0,
                        "image": "images/frame_000000.jpg",
                        "ball_xy": [0.0, 0.0],
                        "xy_px": [0.0, 0.0],
                        "visible": True,
                        "visibility": "visible",
                        "notes": "",
                    },
                    {
                        "review_id": "ball_frame_000003",
                        "frame_index": 3,
                        "t": 0.1,
                        "image": "images/frame_000003.jpg",
                        "ball_xy": [30.0, 0.0],
                        "xy_px": [30.0, 0.0],
                        "visible": True,
                        "visibility": "visible",
                        "notes": "",
                    },
                    {
                        "review_id": "ball_frame_000004",
                        "frame_index": 4,
                        "t": 4 / 30.0,
                        "image": "images/frame_000004.jpg",
                        "ball_xy": None,
                        "xy_px": None,
                        "visible": False,
                        "visibility": "missing",
                        "notes": "missing",
                    },
                    {
                        "review_id": "ball_frame_000005",
                        "frame_index": 5,
                        "t": 5 / 30.0,
                        "image": "images/frame_000005.jpg",
                        "ball_xy": None,
                        "xy_px": None,
                        "visible": None,
                        "visibility": None,
                        "notes": "",
                    },
                ],
                "not_ground_truth": True,
            }
        ),
        encoding="utf-8",
    )


def test_load_ball_click_review_counts_visible_hidden_and_pending(tmp_path: Path) -> None:
    clicks_path = tmp_path / "ball_points.json"
    _write_clicks(clicks_path)

    review = load_ball_click_review(clicks_path)

    assert review.clip == "clip_a"
    assert len(review.visible_items) == 2
    assert len(review.hidden_items) == 1
    assert len(review.pending_items) == 1


def test_filter_ball_track_with_click_anchors_corrects_clicks_and_rejects_identity_jumps(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    clicks_path = tmp_path / "ball_points.json"
    _write_track(track_path)
    _write_clicks(clicks_path)

    payload, summary = filter_ball_track_with_click_anchors(
        ball_track_path=track_path,
        clicks_path=clicks_path,
        max_identity_error_px=40.0,
        interpolate_max_gap_frames=10,
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[0].xy == pytest.approx([0.0, 0.0])
    assert filtered.frames[1].visible is True
    assert filtered.frames[1].approx is True
    assert filtered.frames[1].xy == pytest.approx([10.0, 0.0])
    assert filtered.frames[2].visible is False
    assert filtered.frames[4].visible is False
    assert filtered.frames[5].visible is True
    assert summary["clicked_visible_count"] == 2
    assert summary["clicked_hidden_count"] == 1
    assert summary["click_pending_count"] == 1
    assert summary["identity_rejected_count"] == 1
    assert summary["interpolated_count"] == 1
    assert summary["metrics"]["source"]["negative_false_positive_rate"] > 0.0
    assert summary["metrics"]["output"]["visible_recall"] == pytest.approx(1.0)
    assert summary["metrics"]["output"]["negative_false_positive_rate"] == pytest.approx(0.0)


def test_ball_identity_filter_cli_writes_schema_valid_output(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    clicks_path = tmp_path / "ball_points.json"
    out = tmp_path / "ball_identity.json"
    summary_out = tmp_path / "ball_identity_summary.json"
    _write_track(track_path)
    _write_clicks(clicks_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/filter_ball_identity_from_clicks.py",
            "--ball-track",
            str(track_path),
            "--clicks",
            str(clicks_path),
            "--max-identity-error-px",
            "40",
            "--interpolate-max-gap-frames",
            "10",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["identity_rejected_count"] == 1
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
    assert json.loads(summary_out.read_text(encoding="utf-8"))["interpolated_count"] == 1
