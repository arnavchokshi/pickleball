from __future__ import annotations

import pytest

from threed.racketsport.person_yolo_dataset import yolo_label_line
from threed.racketsport.schemas import PersonLabel


def test_yolo_label_line_normalizes_xywh_bbox_to_center_format() -> None:
    label = PersonLabel(track_id=7, bbox_xywh=(10.0, 20.0, 40.0, 80.0), class_name="player")

    line = yolo_label_line(label, image_width=200, image_height=400)

    assert line == "0 0.150000 0.150000 0.200000 0.200000"


def test_yolo_label_line_clips_bbox_to_image_bounds() -> None:
    label = PersonLabel(track_id=7, bbox_xywh=(-10.0, 20.0, 40.0, 80.0))

    line = yolo_label_line(label, image_width=200, image_height=400)

    assert line == "0 0.075000 0.150000 0.150000 0.200000"


def test_yolo_label_line_rejects_box_outside_image() -> None:
    label = PersonLabel(track_id=7, bbox_xywh=(300.0, 20.0, 40.0, 80.0))

    with pytest.raises(ValueError, match="outside image"):
        yolo_label_line(label, image_width=200, image_height=400)
