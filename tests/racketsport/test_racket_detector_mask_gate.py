from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from threed.racketsport.racket_detector_mask_gate import (
    _predict_and_write_masks,
    build_detector_mask_report,
    detections_from_yolo_boxes,
    filter_detections_by_box_geometry,
    merge_tiled_detections,
    match_detections_to_labels,
)


def test_match_detections_to_labels_counts_unique_iou_matches() -> None:
    labels = [
        {"bbox_xyxy": [10.0, 10.0, 30.0, 30.0]},
        {"bbox_xyxy": [60.0, 60.0, 90.0, 90.0]},
    ]
    detections = [
        {"bbox_xyxy": [11.0, 11.0, 29.0, 29.0], "score": 0.9},
        {"bbox_xyxy": [12.0, 12.0, 28.0, 28.0], "score": 0.8},
        {"bbox_xyxy": [62.0, 62.0, 88.0, 88.0], "score": 0.7},
    ]

    result = match_detections_to_labels(labels, detections, iou_threshold=0.5)

    assert result == {
        "label_count": 2,
        "detection_count": 3,
        "match_count": 2,
        "false_positive_count": 1,
        "recall": 1.0,
        "precision": 2 / 3,
    }


def test_detector_mask_report_fails_until_recall_gate_passes() -> None:
    records = [
        {
            "clip_id": "clip_a",
            "frame_index": 0,
            "labels": [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}],
            "detections": [
                {
                    "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                    "score": 0.9,
                    "mask": {"present": True, "area_px": 50},
                }
            ],
        },
        {
            "clip_id": "clip_b",
            "frame_index": 0,
            "labels": [{"bbox_xyxy": [100.0, 100.0, 120.0, 120.0]}],
            "detections": [],
        },
    ]

    report = build_detector_mask_report(
        records,
        model_sources={
            "detector": "IDEA-Research/grounding-dino-tiny",
            "mask": "facebook/sam2-hiera-tiny",
        },
        iou_threshold=0.5,
        recall_gate=0.9,
    )

    assert report["artifact_type"] == "racketsport_racket_detector_mask_gate"
    assert report["status"] == "fail"
    assert report["execution"]["runs_inference"] is True
    assert report["metrics"]["frame_count"] == 2
    assert report["metrics"]["label_count"] == 2
    assert report["metrics"]["match_count"] == 1
    assert report["metrics"]["recall"] == 0.5
    assert report["metrics"]["mask_detection_count"] == 1
    assert report["per_clip"]["clip_a"]["recall"] == 1.0
    assert report["per_clip"]["clip_b"]["recall"] == 0.0


def test_detector_mask_report_passes_when_all_labels_matched_and_masked() -> None:
    report = build_detector_mask_report(
        [
            {
                "clip_id": "clip_a",
                "frame_index": 0,
                "labels": [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}],
                "detections": [
                    {
                        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                        "score": 0.9,
                        "mask": {"present": True, "area_px": 50},
                    }
                ],
            }
        ],
        model_sources={"detector": "detector", "mask": "mask"},
        iou_threshold=0.5,
        recall_gate=0.9,
    )

    assert report["status"] == "pass"
    assert report["metrics"]["recall"] == 1.0
    assert report["metrics"]["mask_coverage_rate"] == 1.0


def test_detections_from_yolo_boxes_converts_class_zero_only() -> None:
    detections = detections_from_yolo_boxes(
        boxes_xyxy=[[1, 2, 11, 12], [50, 60, 70, 80]],
        scores=[0.8, 0.7],
        classes=[0, 1],
        class_id=0,
    )

    assert detections == [
        {
            "bbox_xyxy": [1.0, 2.0, 11.0, 12.0],
            "score": 0.8,
            "text_label": "paddle",
        }
    ]


def test_merge_tiled_detections_offsets_boxes_and_suppresses_duplicates() -> None:
    detections = merge_tiled_detections(
        [
            {
                "tile_xyxy": [0, 0, 100, 100],
                "detections": [
                    {"bbox_xyxy": [10, 20, 40, 60], "score": 0.6, "text_label": "paddle"},
                    {"bbox_xyxy": [80, 80, 95, 95], "score": 0.4, "text_label": "paddle"},
                ],
            },
            {
                "tile_xyxy": [20, 10, 120, 110],
                "detections": [
                    {"bbox_xyxy": [0, 10, 30, 50], "score": 0.9, "text_label": "paddle"},
                    {"bbox_xyxy": [40, 40, 70, 80], "score": 0.7, "text_label": "paddle"},
                ],
            },
        ],
        image_width=120,
        image_height=110,
        nms_iou_threshold=0.5,
    )

    assert detections == [
        {
            "bbox_xyxy": [20.0, 20.0, 50.0, 60.0],
            "score": 0.9,
            "text_label": "paddle",
            "tile_xyxy": [20.0, 10.0, 120.0, 110.0],
        },
        {
            "bbox_xyxy": [60.0, 50.0, 90.0, 90.0],
            "score": 0.7,
            "text_label": "paddle",
            "tile_xyxy": [20.0, 10.0, 120.0, 110.0],
        },
        {
            "bbox_xyxy": [80.0, 80.0, 95.0, 95.0],
            "score": 0.4,
            "text_label": "paddle",
            "tile_xyxy": [0.0, 0.0, 100.0, 100.0],
        },
    ]


def test_filter_detections_by_box_geometry_keeps_only_configured_paddle_sized_boxes() -> None:
    detections = [
        {"bbox_xyxy": [10, 10, 40, 40], "score": 0.9},  # 900 px2, aspect 1.0
        {"bbox_xyxy": [0, 0, 10, 10], "score": 0.8},  # too small
        {"bbox_xyxy": [0, 0, 400, 400], "score": 0.7},  # too large
        {"bbox_xyxy": [0, 0, 200, 20], "score": 0.6},  # too wide
    ]

    filtered = filter_detections_by_box_geometry(
        detections,
        min_box_area=500.0,
        max_box_area=35_000.0,
        min_box_aspect=0.4,
        max_box_aspect=3.0,
    )

    assert filtered == [{"bbox_xyxy": [10, 10, 40, 40], "score": 0.9}]


def test_predict_and_write_masks_batches_box_prompts_without_reordering(tmp_path: Path) -> None:
    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeTorch:
        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

    class FakePredictor:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []
            self.set_image_count = 0

        def set_image(self, _image_rgb: np.ndarray) -> None:
            self.set_image_count += 1

        def predict(self, *, box: np.ndarray, multimask_output: bool) -> tuple[np.ndarray, np.ndarray, None]:
            assert multimask_output is False
            self.calls.append(box.copy())
            masks = np.zeros((len(box), 4, 4), dtype=bool)
            for index in range(len(box)):
                masks[index, index % 4, index % 4] = True
            scores = np.linspace(0.1, 0.9, num=len(box), dtype=np.float32)
            return masks, scores, None

    predictor = FakePredictor()
    boxes = np.asarray(
        [
            [0, 0, 4, 4],
            [1, 1, 4, 4],
            [2, 2, 4, 4],
            [0, 0, 3, 3],
            [1, 0, 4, 3],
        ],
        dtype=np.float32,
    )

    infos = _predict_and_write_masks(
        predictor,
        np.zeros((4, 4, 3), dtype=np.uint8),
        boxes,
        mask_root=tmp_path,
        clip_id="clip",
        frame_index=12,
        torch=FakeTorch,
        box_batch_size=2,
    )

    assert predictor.set_image_count == 1
    assert [len(call) for call in predictor.calls] == [2, 2, 1]
    assert len(infos) == 5
    assert [Path(info["path"]).name for info in infos] == [
        "clip_000012_det00.png",
        "clip_000012_det01.png",
        "clip_000012_det02.png",
        "clip_000012_det03.png",
        "clip_000012_det04.png",
    ]
    assert all(info["present"] is True for info in infos)
    assert all(Path(info["path"]).is_file() for info in infos)


def test_run_racket_detector_mask_gate_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/run_racket_detector_mask_gate.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--recall-gate" in completed.stdout


def test_run_racket_detector_mask_gate_cli_fails_closed_on_missing_manifest(tmp_path: Path) -> None:
    out_path = tmp_path / "report.json"
    missing_manifest = tmp_path / "does_not_exist_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_racket_detector_mask_gate.py",
            "--manifest",
            str(missing_manifest),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not out_path.exists()
