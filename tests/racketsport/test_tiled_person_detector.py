from __future__ import annotations

import numpy as np

from threed.racketsport import tiled_person_detector
from threed.racketsport.tiled_person_detector import (
    crop_region_pixels,
    merge_tiled_detections,
    offset_crop_detections,
    parse_crop_regions,
)


def test_crop_region_pixels_clamps_normalized_regions_to_frame_bounds() -> None:
    assert crop_region_pixels(1920, 1080, (-0.1, 0.25, 1.2, 0.75)) == (0, 270, 1920, 810)


def test_parse_crop_regions_supports_named_presets_and_explicit_regions() -> None:
    assert parse_crop_regions("full") == ((0.0, 0.0, 1.0, 1.0),)
    assert len(parse_crop_regions("default4")) == 4
    assert parse_crop_regions("0,0,0.5,1;0.5,0,1,1") == ((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0))


def test_parse_adaptive_crop_regions_returns_primary_and_fallback_sets() -> None:
    primary, fallback, min_detections = tiled_person_detector.parse_adaptive_crop_regions("adaptive_full_tb3")

    assert primary == ((0.0, 0.0, 1.0, 1.0),)
    assert fallback == ((0.0, 0.0, 1.0, 0.58), (0.0, 0.42, 1.0, 1.0))
    assert min_detections == 4


def test_offset_crop_detections_restores_full_frame_coordinates() -> None:
    detections = offset_crop_detections(
        [{"bbox": [10.0, 20.0, 30.0, 40.0], "conf": 0.9, "class": "person"}],
        x0=100,
        y0=200,
    )

    assert detections == [{"bbox": [110.0, 220.0, 130.0, 240.0], "conf": 0.9, "class": "person"}]


def test_merge_tiled_detections_keeps_highest_confidence_duplicate() -> None:
    merged = merge_tiled_detections(
        [
            {"bbox": [10.0, 10.0, 50.0, 80.0], "conf": 0.5, "class": "person"},
            {"bbox": [12.0, 12.0, 52.0, 82.0], "conf": 0.9, "class": "person"},
            {"bbox": [200.0, 10.0, 240.0, 80.0], "conf": 0.7, "class": "person"},
        ],
        iou_threshold=0.5,
    )

    assert merged == [
        {"bbox": [12.0, 12.0, 52.0, 82.0], "conf": 0.9, "class": "person"},
        {"bbox": [200.0, 10.0, 240.0, 80.0], "conf": 0.7, "class": "person"},
    ]


def test_merge_tiled_detections_drops_degenerate_boxes() -> None:
    merged = merge_tiled_detections(
        [
            {"bbox": [20.0, 10.0, 20.0, 80.0], "conf": 0.99, "class": "person"},
            {"bbox": [100.0, 10.0, 140.0, 80.0], "conf": 0.7, "class": "person"},
        ],
        iou_threshold=0.5,
    )

    assert merged == [{"bbox": [100.0, 10.0, 140.0, 80.0], "conf": 0.7, "class": "person"}]


def test_batched_payload_sends_crops_to_model_in_batches_and_restores_offsets() -> None:
    model = _FakeYolo()
    frames = [np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200, 3), dtype=np.uint8)]

    payload = tiled_person_detector.yolo_tiled_detections_for_frames_batched(
        model=model,
        frames=frames,
        fps=30.0,
        crop_regions=((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)),
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device="0",
        nms_iou=0.55,
        batch_size=3,
        half=True,
    )

    assert [call["image_count"] for call in model.calls] == [3, 1]
    assert all(call["kwargs"]["batch"] == 3 for call in model.calls)
    assert all(call["kwargs"]["half"] is True for call in model.calls)
    assert all(call["kwargs"]["device"] == "0" for call in model.calls)
    assert payload["fps"] == 30.0
    assert [frame["frame"] for frame in payload["frames"]] == [0, 1]
    assert payload["frames"][0]["detections"] == [
        {"bbox": [10.0, 20.0, 30.0, 70.0], "conf": 0.9, "class": "person", "track_id": 1},
        {"bbox": [110.0, 20.0, 130.0, 70.0], "conf": 0.9, "class": "person", "track_id": 2},
    ]
    assert payload["frames"][1]["detections"] == [
        {"bbox": [10.0, 20.0, 30.0, 70.0], "conf": 0.9, "class": "person", "track_id": 1},
        {"bbox": [110.0, 20.0, 130.0, 70.0], "conf": 0.9, "class": "person", "track_id": 2},
    ]


def test_adaptive_payload_only_runs_fallback_crops_for_frames_missing_people() -> None:
    model = _AdaptiveFakeYolo()
    complete_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    incomplete_frame = np.ones((100, 200, 3), dtype=np.uint8)

    payload = tiled_person_detector.yolo_adaptive_tiled_detections_for_frames_batched(
        model=model,
        frames=[complete_frame, incomplete_frame],
        fps=30.0,
        primary_crop_regions=((0.0, 0.0, 1.0, 1.0),),
        fallback_crop_regions=((0.0, 0.0, 1.0, 0.58), (0.0, 0.42, 1.0, 1.0)),
        min_detections=4,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device="0",
        nms_iou=0.55,
        batch_size=2,
        half=True,
    )

    assert [call["image_count"] for call in model.calls] == [2, 2]
    assert all(call["kwargs"]["half"] is True for call in model.calls)
    assert payload["fallback_frame_count"] == 1
    assert payload["crop_eval_count"] == 4
    assert [len(frame["detections"]) for frame in payload["frames"]] == [4, 4]
    assert payload["frames"][1]["detections"][-1] == {
        "bbox": [150.0, 20.0, 170.0, 70.0],
        "conf": 0.86,
        "class": "person",
        "track_id": 4,
    }


class _FakeYolo:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, images: list[np.ndarray], **kwargs: object) -> list["_FakeResult"]:
        self.calls.append({"image_count": len(images), "kwargs": kwargs})
        return [_FakeResult() for _ in images]


class _FakeBox:
    xyxy = [np.array([10.0, 20.0, 30.0, 70.0])]
    conf = [0.9]


class _FakeResult:
    boxes = [_FakeBox()]


class _AdaptiveFakeYolo:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, images: list[np.ndarray], **kwargs: object) -> list["_AdaptiveFakeResult"]:
        self.calls.append({"image_count": len(images), "kwargs": kwargs})
        if len(self.calls) == 1:
            return [
                _AdaptiveFakeResult(_adaptive_boxes(4)),
                _AdaptiveFakeResult(_adaptive_boxes(3)),
            ]
        return [
            _AdaptiveFakeResult([_AdaptiveFakeBox([150.0, 20.0, 170.0, 70.0], 0.86)]),
            _AdaptiveFakeResult([]),
        ]


def _adaptive_boxes(count: int) -> list["_AdaptiveFakeBox"]:
    return [
        _AdaptiveFakeBox([10.0 + index * 35.0, 20.0, 30.0 + index * 35.0, 70.0], 0.9 - index * 0.01)
        for index in range(count)
    ]


class _AdaptiveFakeBox:
    def __init__(self, bbox: list[float], conf: float) -> None:
        self.xyxy = [np.array(bbox)]
        self.conf = [conf]


class _AdaptiveFakeResult:
    def __init__(self, boxes: list[_AdaptiveFakeBox]) -> None:
        self.boxes = boxes
