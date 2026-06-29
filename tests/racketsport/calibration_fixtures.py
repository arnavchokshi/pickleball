from __future__ import annotations

MINIMAL_CALIBRATION_IMAGE_PTS = [
    [100.0, 300.0],
    [900.0, 300.0],
    [900.0, 700.0],
    [100.0, 700.0],
]
MINIMAL_CALIBRATION_WORLD_PTS = [
    [0.0, 0.0, 0.0],
    [6.1, 0.0, 0.0],
    [6.1, 13.4, 0.0],
    [0.0, 13.4, 0.0],
]


def minimal_calibration_image_pts() -> list[list[float]]:
    return [point.copy() for point in MINIMAL_CALIBRATION_IMAGE_PTS]


def minimal_calibration_world_pts() -> list[list[float]]:
    return [point.copy() for point in MINIMAL_CALIBRATION_WORLD_PTS]


def minimal_ready_court_line_evidence() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "test_hough_template",
        "line_observations": [
            _court_line_observation("near_nvz", [[100.0, 300.0], [900.0, 300.0]]),
            _court_line_observation("far_nvz", [[120.0, 180.0], [880.0, 180.0]]),
            _court_line_observation("near_centerline", [[500.0, 300.0], [500.0, 700.0]]),
            _court_line_observation("far_centerline", [[500.0, 180.0], [500.0, 40.0]]),
        ],
        "keypoint_observations": [],
        "net_observations": [
            {
                "net_id": "top_net",
                "image_points": [[100.0, 240.0], [500.0, 238.0], [900.0, 240.0]],
                "confidence": 0.88,
                "frame_indexes": [1, 2, 3],
                "residual_px": {"mean": 2.0, "p95": 3.0},
                "source": "net_top_roi",
            }
        ],
        "aggregate": {
            "accepted_line_ids": ["near_nvz", "far_nvz", "near_centerline", "far_centerline"],
            "rejected_line_ids": [],
            "missing_required_line_ids": [],
            "missing_required_net_ids": [],
            "mean_residual_px": 2.0,
            "p95_residual_px": 4.0,
            "temporal_stability_px": 3.0,
            "auto_calibration_ready": True,
            "reasons": [],
        },
    }


def _court_line_observation(line_id: str, image_segment: list[list[float]]) -> dict:
    return {
        "line_id": line_id,
        "image_segment": image_segment,
        "confidence": 0.9,
        "frame_indexes": [1, 2, 3],
        "residual_px": {"mean": 1.0, "p95": 2.0},
        "visible_fraction": 0.9,
        "source": "hough_template",
    }
