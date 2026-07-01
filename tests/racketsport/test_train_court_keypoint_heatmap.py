from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport.train_court_keypoint_heatmap import (
    court_keypoint_heatmap_loss,
    court_keypoint_probabilities,
    load_real_corner_labels,
    load_real_court_keypoint_labels,
    make_court_keypoint_heatmap_model,
    run_training,
    training_cli_summary,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_court_keypoint_heatmap_loss_prioritizes_labeled_peaks() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    target[0, 0, 4, 4] = 1.0
    mask = torch.ones_like(target)

    peak_missed = torch.full_like(target, -6.0)
    background_false_positive = torch.full_like(target, -6.0)
    background_false_positive[0, 0, 4, 4] = 6.0
    background_false_positive[0, 0, 0, 0] = 6.0

    missed_loss = court_keypoint_heatmap_loss(peak_missed, target, mask)
    false_positive_loss = court_keypoint_heatmap_loss(background_false_positive, target, mask)

    assert missed_loss.item() > false_positive_loss.item() * 4.0

    probabilities = court_keypoint_probabilities(torch.tensor([[[[-6.0, 0.0, 6.0]]]]))
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0, 0, 0].tolist() == pytest.approx([0.00000612898, 0.002472608, 0.9975212])


def test_court_keypoint_heatmap_model_uses_encoder_decoder_context() -> None:
    torch = pytest.importorskip("torch")

    model = make_court_keypoint_heatmap_model(3)
    output = model(torch.zeros((2, 3, 90, 160), dtype=torch.float32))

    assert tuple(output.shape) == (2, 3, 90, 160)
    assert any(isinstance(module, torch.nn.Conv2d) and module.stride == (2, 2) for module in model.modules())
    assert any(isinstance(module, torch.nn.Upsample) for module in model.modules())


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


def _reviewed_court_keypoint_label_payload(
    frame: str = "frame_000002.jpg",
    *,
    source_resolution: list[int] | None = None,
    label_coordinate_space: list[int] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "annotation": {
            "items": [
                {
                    "frame": frame,
                    "keypoints": {
                        point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
                    },
                }
            ]
        },
        "frames": {
            "frame_dir": "runs/label_frames/clip_a",
            "source_resolution": source_resolution or [128, 72],
            "label_coordinate_space": label_coordinate_space or [64, 36],
        },
        "review": {"status": "reviewed", "reviewer": "court-label-review"},
    }


def test_load_real_court_keypoint_labels_requires_reviewed_full_15_point_labels(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="reviewed 15-keypoint court labels"):
        load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")


def test_load_real_court_keypoint_labels_reads_reviewed_full_15_point_labels(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(clip_root / "labels" / "court_keypoints.json", _reviewed_court_keypoint_label_payload())

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    row = rows[0]
    assert row["clip"] == "clip_a"
    assert row["video_path"] == str(clip_root / "source.mp4")
    assert row["frame_index"] == 2
    assert row["image_path"] is None
    assert set(row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert row["label_coordinate_space"] == [64, 36]
    assert row["source_video_size"] == [128, 72]
    assert row["keypoints"]["near_left_corner"] == pytest.approx([20.0, 10.0])
    assert row["keypoints"]["far_nvz_right"] == pytest.approx([104.0, 66.0])


def test_load_real_court_keypoint_labels_reads_all_reviewed_items(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    payload = _reviewed_court_keypoint_label_payload("frame_000001.jpg")
    second_item = {
        "frame": "frame_000002.jpg",
        "keypoints": {
            point.name: [float(index * 5 + 20), float(index * 7 + 30)]
            for index, point in enumerate(PICKLEBALL_KEYPOINTS)
        },
    }
    payload["annotation"]["items"].append(second_item)
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 2
    assert [row["frame_index"] for row in rows] == [1, 2]
    assert [row["clip"] for row in rows] == ["clip_a", "clip_a"]
    assert rows[1]["keypoints"]["near_left_corner"] == pytest.approx([40.0, 60.0])
    assert rows[1]["keypoints"]["far_nvz_right"] == pytest.approx([180.0, 256.0])


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
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg",
            source_resolution=[64, 36],
            label_coordinate_space=[64, 36],
        ),
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

    assert summary["gate"]["metric"] == "heldout_pck_at_5px"
    assert summary["gate"]["threshold"] == 0.95
    assert summary["gate"]["pck_threshold_px"] == 5.0
    assert summary["gate"]["value"] == summary["after"]["real_keypoint_pck_at_5px"]
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["keypoint_count"] == 15
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["pck_at_5px"] == summary["after"]["real_keypoint_pck_at_5px"]
    assert summary["after"]["real_corner_median_px"] is not None
    assert summary["after"]["real_corner_median_model_input_px"] is not None
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(summary["after"]["real_corner_median_px"])
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(
        summary["after"]["real_corner_median_model_input_px"] * 2.0,
        rel=0.15,
    )
    assert summary["holdout_artifacts"][0]["clip"] == "clip_a"
    assert summary["holdout_artifacts"][0]["prediction_artifact"].endswith("clip_a_court_keypoints.json")
    assert summary["holdout_artifacts"][0]["overlay_artifact"].endswith("clip_a_court_keypoints_overlay.mp4")
    assert summary["holdout_artifacts"][0]["overlay_frame_count"] == 3
    assert (out / "holdout_predictions" / "clip_a_court_keypoints.json").is_file()
    assert (out / "holdout_overlays" / "clip_a_court_keypoints_overlay.mp4").is_file()


def test_run_training_can_hold_out_frames_per_viewpoint(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(6):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (30 + idx * 5, 70, 100)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000001.jpg",
        source_resolution=[64, 36],
        label_coordinate_space=[64, 36],
    )
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(1, 5)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_frame_split",
            holdout_clip=["clip_b"],
            holdout_frame_stride=2,
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

    assert summary["holdout_strategy"] == {"type": "frame_stride", "stride": 2}
    assert summary["real_train_count"] == 2
    assert summary["real_holdout_count"] == 2
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["keypoint_count"] == 30


def test_training_cli_summary_prints_gate_metric_and_artifact_paths() -> None:
    summary = training_cli_summary(
        {
            "checkpoint": "run/court_keypoint_heatmap.pt",
            "gate": {
                "metric": "heldout_pck_at_5px",
                "value": 0.9,
                "threshold": 0.95,
                "pck_threshold_px": 5.0,
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
    assert summary["gate"]["metric"] == "heldout_pck_at_5px"
    assert summary["gate"]["value"] == 0.9
    assert summary["holdout_artifacts"][0]["overlay_artifact"] == "run/holdout_overlays/clip_a.mp4"
