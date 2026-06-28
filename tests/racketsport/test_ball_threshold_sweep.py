from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_threshold_sweep import _best_threshold_candidate, sweep_prediction_thresholds


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_clicks(path: Path, *, clip: str) -> None:
    _write_json(
        path,
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
                    "ball_xy": [100.0, 200.0],
                    "visible": True,
                    "visibility": "visible",
                },
                {
                    "review_id": "ball_frame_000001",
                    "frame_index": 1,
                    "t": 1.0 / 30.0,
                    "image": "frame_000001.jpg",
                    "ball_xy": None,
                    "visible": False,
                    "visibility": "missing",
                },
            ],
        },
    )


def _write_totnet_predictions(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_totnet_predictions",
            "fps": 30.0,
            "image_size": [1920, 1080],
            "input_size": [512, 288],
            "model": {"id": "totnet_test"},
            "frames": [
                {"frame_index": 0, "xy": [100.0, 200.0], "confidence": 0.91, "visible": True},
                {"frame_index": 1, "xy": [900.0, 500.0], "confidence": 0.2, "visible": True},
            ],
        },
    )


def test_sweep_prediction_thresholds_selects_threshold_that_removes_hidden_false_positive(tmp_path: Path) -> None:
    clip = "clip_a"
    predictions = tmp_path / "predictions" / clip / "totnet_predictions.json"
    review_root = tmp_path / "review"
    _write_totnet_predictions(predictions)
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)

    summary = sweep_prediction_thresholds(
        predictions_by_clip={clip: predictions},
        review_root=review_root,
        out_root=tmp_path / "sweep",
        family="totnet",
        candidate_name_prefix="totnet_test",
        thresholds=[0.0, 0.5],
    )

    assert summary["best_candidate"] == "totnet_test_thr_0_500"
    assert summary["best_threshold"] == pytest.approx(0.5)
    aggregate = summary["benchmark"]["aggregate"]
    assert aggregate["totnet_test_thr_0_000"]["micro_hidden_false_positive_rate"] == pytest.approx(1.0)
    assert aggregate["totnet_test_thr_0_500"]["micro_hidden_false_positive_rate"] == pytest.approx(0.0)
    assert aggregate["totnet_test_thr_0_500"]["micro_visible_hit_recall"] == pytest.approx(1.0)


def test_sweep_prediction_thresholds_rejects_empty_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        sweep_prediction_thresholds(
            predictions_by_clip={},
            review_root=tmp_path / "review",
            out_root=tmp_path / "sweep",
            family="totnet",
            candidate_name_prefix="totnet_test",
            thresholds=[],
        )


def test_best_threshold_candidate_uses_gate_eligible_threshold_before_quality_score() -> None:
    candidate, threshold = _best_threshold_candidate(
        {
            "totnet_thr_0_100": {
                "mean_quality_score": -0.47,
                "micro_visible_hit_recall": 0.6,
                "micro_hidden_false_positive_rate": 0.1,
                "total_teleport_count": 4,
            },
            "totnet_thr_0_900": {
                "mean_quality_score": -0.4,
                "micro_visible_hit_recall": 0.01,
                "micro_hidden_false_positive_rate": 0.0,
                "total_teleport_count": 0,
            },
        },
        [0.1, 0.9],
        "totnet",
    )

    assert candidate == "totnet_thr_0_100"
    assert threshold == pytest.approx(0.1)


def test_best_threshold_candidate_returns_none_when_every_threshold_has_zero_recall() -> None:
    candidate, threshold = _best_threshold_candidate(
        {
            "totnet_thr_0_500": {
                "mean_quality_score": -0.4,
                "micro_visible_hit_recall": 0.0,
                "micro_hidden_false_positive_rate": 0.0,
                "total_teleport_count": 0,
            },
            "totnet_thr_0_900": {
                "mean_quality_score": -0.4,
                "micro_visible_hit_recall": 0.0,
                "micro_hidden_false_positive_rate": 0.0,
                "total_teleport_count": 0,
            },
        },
        [0.5, 0.9],
        "totnet",
    )

    assert candidate is None
    assert threshold is None


def test_sweep_ball_prediction_thresholds_cli(tmp_path: Path) -> None:
    clip = "clip_a"
    predictions = tmp_path / "predictions" / clip / "totnet_predictions.json"
    review_root = tmp_path / "review"
    out_root = tmp_path / "sweep"
    _write_totnet_predictions(predictions)
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/sweep_ball_prediction_thresholds.py",
            "--family",
            "totnet",
            "--candidate-prefix",
            "totnet_test",
            "--review-root",
            str(review_root),
            "--out-root",
            str(out_root),
            "--threshold",
            "0.0",
            "--threshold",
            "0.5",
            "--prediction",
            f"{clip}={predictions}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["best_candidate"] == "totnet_test_thr_0_500"
    assert (out_root / "threshold_sweep_summary.json").is_file()
