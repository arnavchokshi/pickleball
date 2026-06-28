from __future__ import annotations

from threed.racketsport.detection_scaling import scale_detection_payload_bboxes


def test_scale_detection_payload_bboxes_keeps_payload_shape_and_scales_xyxy_fields() -> None:
    payload = {
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [100.0, 200.0, 300.0, 400.0], "track_id": 1},
                    {"bbox_xyxy": [10.0, 20.0, 30.0, 40.0], "track_id": 2},
                ],
            }
        ],
    }

    scaled = scale_detection_payload_bboxes(payload, scale_x=0.5, scale_y=0.25)

    assert scaled["fps"] == 30.0
    assert scaled["frames"][0]["detections"][0]["bbox"] == [50.0, 50.0, 150.0, 100.0]
    assert scaled["frames"][0]["detections"][1]["bbox_xyxy"] == [5.0, 5.0, 15.0, 10.0]
    assert payload["frames"][0]["detections"][0]["bbox"] == [100.0, 200.0, 300.0, 400.0]
