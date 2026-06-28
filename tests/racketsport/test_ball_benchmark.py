from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_benchmark import (
    BallCandidate,
    benchmark_ball_track_candidate,
    write_ball_tracker_benchmark,
)


def _write_track(path: Path) -> None:
    frames = [
        {"t": 0 / 30.0, "xy": [10.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 1 / 30.0, "xy": [20.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 2 / 30.0, "xy": [500.0, 500.0], "conf": 0.9, "visible": True},
        {"t": 3 / 30.0, "xy": [35.0, 12.0], "conf": 0.9, "visible": True},
        {"t": 4 / 30.0, "xy": [40.0, 12.0], "conf": 0.9, "visible": False},
    ]
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def _write_clicks(path: Path, *, clip: str = "clip_a") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_click_review",
                "status": "human_reviewed",
                "clip": clip,
                "target_file": "ball.json",
                "coordinate_frame": "image_pixels_video_space",
                "items": [
                    {
                        "review_id": "ball_frame_000000",
                        "frame_index": 0,
                        "t": 0.0,
                        "image": "frame_000000.jpg",
                        "ball_xy": [11.0, 10.0],
                        "visible": True,
                        "visibility": "visible",
                    },
                    {
                        "review_id": "ball_frame_000002",
                        "frame_index": 2,
                        "t": 2 / 30.0,
                        "image": "frame_000002.jpg",
                        "ball_xy": [30.0, 11.0],
                        "visible": True,
                        "visibility": "visible",
                    },
                    {
                        "review_id": "ball_frame_000003",
                        "frame_index": 3,
                        "t": 3 / 30.0,
                        "image": "frame_000003.jpg",
                        "ball_xy": None,
                        "visible": False,
                        "visibility": "missing",
                    },
                    {
                        "review_id": "ball_frame_000004",
                        "frame_index": 4,
                        "t": 4 / 30.0,
                        "image": "frame_000004.jpg",
                        "ball_xy": None,
                        "visible": None,
                        "visibility": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_benchmark_candidate_scores_visible_hits_hidden_false_positives_and_jitter(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    clicks_path = tmp_path / "ball_points.json"
    _write_track(track_path)
    _write_clicks(clicks_path)

    summary = benchmark_ball_track_candidate(
        ball_track_path=track_path,
        clicks_path=clicks_path,
        candidate_name="raw",
        hit_radius_px=36.0,
        teleport_px_per_frame=160.0,
    )

    assert summary["category"] == "generalizable"
    assert summary["label_metrics"]["visible_label_count"] == 2
    assert summary["label_metrics"]["visible_prediction_count"] == 2
    assert summary["label_metrics"]["visible_hit_count"] == 1
    assert summary["label_metrics"]["visible_recall_at_5px"] == pytest.approx(0.5)
    assert summary["label_metrics"]["visible_recall_at_20px"] == pytest.approx(0.5)
    assert summary["label_metrics"]["hidden_false_positive_count"] == 1
    assert summary["label_metrics"]["hidden_false_positives_per_minute"] == pytest.approx(600.0)
    assert summary["label_metrics"]["hidden_true_negative_rate"] == pytest.approx(0.0)
    assert summary["label_metrics"]["label_f1_at_20px"] == pytest.approx(0.5)
    assert summary["jitter_metrics"]["teleport_count"] == 2
    assert summary["jitter_metrics"]["max_visible_gap_frames"] == 1
    assert summary["quality_score"] < 0.5


def test_write_benchmark_aggregates_candidates_and_markdown(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    review_root = tmp_path / "review"
    clip_dir = run_root / "clip_a" / "tracks"
    click_dir = review_root / "clip_a"
    clip_dir.mkdir(parents=True)
    click_dir.mkdir(parents=True)
    track_path = clip_dir / "ball_track.json"
    clicks_path = click_dir / "ball_points.json"
    _write_track(track_path)
    _write_clicks(clicks_path)

    summary = write_ball_tracker_benchmark(
        candidates=[BallCandidate(clip="clip_a", name="raw", path=track_path)],
        review_root=review_root,
        out_json=tmp_path / "summary.json",
        out_markdown=tmp_path / "summary.md",
    )

    assert summary["aggregate"]["raw"]["clip_count"] == 1
    assert summary["aggregate"]["raw"]["total_visible_label_count"] == 2
    assert summary["aggregate"]["raw"]["total_visible_hit_count"] == 1
    assert summary["aggregate"]["raw"]["micro_visible_hit_recall"] == pytest.approx(0.5)
    assert summary["aggregate"]["raw"]["total_hidden_label_count"] == 1
    assert summary["aggregate"]["raw"]["total_hidden_false_positive_count"] == 1
    assert summary["aggregate"]["raw"]["micro_hidden_false_positive_rate"] == pytest.approx(1.0)
    assert "# Ball Tracker Benchmark" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_quality_score_penalizes_excessive_visible_gaps(tmp_path: Path) -> None:
    clicks_path = tmp_path / "ball_points.json"
    clicks_path.write_text(
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
                        "image": "frame_000000.jpg",
                        "ball_xy": [11.0, 10.0],
                        "visible": True,
                        "visibility": "visible",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dense_track = tmp_path / "dense.json"
    sparse_track = tmp_path / "sparse.json"
    dense_frames = [
        {"t": index / 30.0, "xy": [11.0, 10.0], "conf": 0.9, "visible": True}
        for index in range(61)
    ]
    sparse_frames = [
        {
            "t": index / 30.0,
            "xy": [11.0, 10.0],
            "conf": 0.9 if index in {0, 60} else 0.0,
            "visible": index in {0, 60},
        }
        for index in range(61)
    ]
    for path, frames in ((dense_track, dense_frames), (sparse_track, sparse_frames)):
        path.write_text(
            json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
            encoding="utf-8",
        )

    dense = benchmark_ball_track_candidate(
        ball_track_path=dense_track,
        clicks_path=clicks_path,
        candidate_name="dense",
    )
    sparse = benchmark_ball_track_candidate(
        ball_track_path=sparse_track,
        clicks_path=clicks_path,
        candidate_name="sparse",
    )

    assert sparse["jitter_metrics"]["max_visible_gap_frames"] == 60
    assert sparse["quality_score"] < dense["quality_score"]


def test_ball_benchmark_cli_expands_candidate_specs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    review_root = tmp_path / "review"
    clip_dir = run_root / "clip_a" / "tracks"
    click_dir = review_root / "clip_a"
    clip_dir.mkdir(parents=True)
    click_dir.mkdir(parents=True)
    _write_track(clip_dir / "ball_track.json")
    _write_clicks(click_dir / "ball_points.json", clip="clip_a")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/benchmark_ball_trackers.py",
            "--run-root",
            str(run_root),
            "--review-root",
            str(review_root),
            "--clip",
            "clip_a",
            "--candidate",
            "raw:generalizable=tracks/ball_track.json",
            "--out-json",
            str(tmp_path / "summary.json"),
            "--out-md",
            str(tmp_path / "summary.md"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "raw" in completed.stdout
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["candidate_count"] == 1
