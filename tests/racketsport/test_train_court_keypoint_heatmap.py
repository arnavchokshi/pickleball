from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport.train_court_keypoint_heatmap import load_real_corner_labels, run_training, training_cli_summary
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_real_corner_labels_uses_committed_video_frame_when_label_frames_are_absent(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_corners.json",
        {
            "schema_version": 1,
            "annotation": {
                "items": [
                    {
                        "frame": "frame_000002.jpg",
                        "court_corners": {
                            "near_left": [8, 32],
                            "near_right": [56, 32],
                            "far_right": [44, 6],
                            "far_left": [20, 6],
                        },
                    }
                ]
            },
            "frames": {"frame_dir": "runs/label_frames/clip_a", "source_resolution": [128, 72]},
        },
    )

    rows = load_real_corner_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    row = rows[0]
    assert row["clip"] == "clip_a"
    assert row["video_path"] == str(clip_root / "source.mp4")
    assert row["frame_index"] == 2
    assert row["image_path"] is None
    assert set(row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert row["label_coordinate_space"] == [64, 36]
    assert row["source_video_size"] == [128, 72]
    assert row["keypoints"]["near_left_corner"] == pytest.approx([16.0, 64.0])
    assert row["keypoints"]["near_right_corner"] == pytest.approx([112.0, 64.0])


def test_run_training_writes_holdout_predictions_overlay_and_gate_metric(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + idx * 10, 60, 90)
        writer.write(frame)
    writer.release()
    _write_json(
        clip_root / "labels" / "court_corners.json",
        {
            "schema_version": 1,
            "annotation": {
                "items": [
                    {
                        "frame": "frame_000000.jpg",
                        "court_corners": {
                            "near_left": [8, 32],
                            "near_right": [56, 32],
                            "far_right": [44, 6],
                            "far_left": [20, 6],
                        },
                    }
                ]
            },
            "frames": {"frame_dir": "runs/label_frames/clip_a"},
        },
    )

    out = tmp_path / "court_run"
    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=out,
            holdout_clip=["clip_a"],
            epochs=0,
            batch_size=1,
            image_width=32,
            image_height=18,
            sigma=1.5,
            learning_rate=1e-3,
            real_finetune_start_epoch=0,
            eval_every=1,
            seed=13,
            device="cpu",
        )
    )

    assert summary["gate"]["metric"] == "heldout_median_keypoint_reprojection_px"
    assert summary["gate"]["threshold_px"] == 5.0
    assert summary["after"]["real_corner_median_px"] is not None
    assert summary["after"]["real_corner_median_model_input_px"] is not None
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(summary["after"]["real_corner_median_px"])
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(
        summary["after"]["real_corner_median_model_input_px"] * 2.0,
        rel=0.15,
    )
    assert summary["holdout_artifacts"][0]["clip"] == "clip_a"
    assert summary["gate"]["value_px"] == summary["holdout_artifacts"][0]["median_keypoint_reprojection_px"]
    assert summary["holdout_artifacts"][0]["prediction_artifact"].endswith("clip_a_court_keypoints.json")
    assert summary["holdout_artifacts"][0]["overlay_artifact"].endswith("clip_a_court_keypoints_overlay.mp4")
    assert summary["holdout_artifacts"][0]["overlay_frame_count"] == 3
    assert (out / "holdout_predictions" / "clip_a_court_keypoints.json").is_file()
    assert (out / "holdout_overlays" / "clip_a_court_keypoints_overlay.mp4").is_file()


def test_training_cli_summary_prints_gate_metric_and_artifact_paths() -> None:
    summary = training_cli_summary(
        {
            "checkpoint": "run/court_keypoint_heatmap.pt",
            "gate": {
                "metric": "heldout_median_keypoint_reprojection_px",
                "value_px": 12.5,
                "threshold_px": 5.0,
                "passed": False,
            },
            "before": {"real_keypoint_median_px": 80.0},
            "after": {"real_keypoint_median_px": 40.0},
            "holdout_artifacts": [
                {
                    "clip": "clip_a",
                    "prediction_artifact": "run/holdout_predictions/clip_a.json",
                    "overlay_artifact": "run/holdout_overlays/clip_a.mp4",
                    "median_keypoint_reprojection_px": 12.5,
                }
            ],
        }
    )

    assert summary["checkpoint"] == "run/court_keypoint_heatmap.pt"
    assert summary["gate"]["metric"] == "heldout_median_keypoint_reprojection_px"
    assert summary["gate"]["value_px"] == 12.5
    assert summary["holdout_artifacts"][0]["overlay_artifact"] == "run/holdout_overlays/clip_a.mp4"
