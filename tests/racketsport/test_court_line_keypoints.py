from __future__ import annotations

import math

import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS, keypoint_labels_from_court_corners
from threed.racketsport.court_line_keypoints import (
    detect_court_keypoints_from_image,
    keypoints_from_semantic_lines,
)


def test_keypoints_from_semantic_lines_recovers_taxonomy_intersections() -> None:
    labels = keypoint_labels_from_court_corners(
        {
            "near_left": [120.0, 360.0],
            "near_right": [620.0, 300.0],
            "far_right": [380.0, 90.0],
            "far_left": [80.0, 130.0],
        }
    )
    semantic_lines = _semantic_lines_from_labels(labels)

    predictions = keypoints_from_semantic_lines(semantic_lines)

    assert set(predictions) == {point.name for point in PICKLEBALL_KEYPOINTS}
    for name, expected_xy in labels.items():
        assert predictions[name]["xy"] == pytest.approx(expected_xy, abs=1e-6)
        assert predictions[name]["confidence"] == pytest.approx(1.0)


def test_detect_court_keypoints_from_image_decodes_synthetic_white_line_court() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    labels = keypoint_labels_from_court_corners(
        {
            "near_left": [120.0, 360.0],
            "near_right": [620.0, 300.0],
            "far_right": [380.0, 90.0],
            "far_left": [80.0, 130.0],
        }
    )
    image = np.zeros((420, 720, 3), dtype=np.uint8)
    image[:, :] = (36, 42, 45)
    for start, end in (
        ("near_left_corner", "near_right_corner"),
        ("far_left_corner", "far_right_corner"),
        ("near_left_corner", "far_left_corner"),
        ("near_right_corner", "far_right_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
        ("net_left_sideline", "net_right_sideline"),
        ("near_baseline_center", "far_baseline_center"),
    ):
        cv2.line(image, _point(labels[start]), _point(labels[end]), (245, 245, 245), 8, cv2.LINE_AA)

    predictions = detect_court_keypoints_from_image(image)

    errors = []
    for name, expected_xy in labels.items():
        pred_xy = predictions.keypoints[name]["xy"]
        errors.append(math.dist(pred_xy, expected_xy))
    assert _median(errors) < 2.0
    assert max(errors) < 4.0


def _semantic_lines_from_labels(labels: dict[str, list[float]]) -> dict[str, list[list[float]]]:
    return {
        "far_baseline": [labels["far_left_corner"], labels["far_right_corner"]],
        "far_nvz": [labels["far_nvz_left"], labels["far_nvz_right"]],
        "net": [labels["net_left_sideline"], labels["net_right_sideline"]],
        "near_nvz": [labels["near_nvz_left"], labels["near_nvz_right"]],
        "near_baseline": [labels["near_left_corner"], labels["near_right_corner"]],
        "left_sideline": [labels["far_left_corner"], labels["near_left_corner"]],
        "centerline": [labels["far_baseline_center"], labels["near_baseline_center"]],
        "right_sideline": [labels["far_right_corner"], labels["near_right_corner"]],
    }


def _point(xy: list[float]) -> tuple[int, int]:
    return (int(round(xy[0])), int(round(xy[1])))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
