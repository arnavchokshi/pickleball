from __future__ import annotations

import pytest

from threed.racketsport.court_keypoint_lines import (
    COURT_LINE_FAMILIES,
    fit_court_lines_from_masks,
    intersect_court_keypoints_from_lines,
    line_mask_targets_for_keypoints,
    validate_round3_input_resolution,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _rectangular_keypoints() -> dict[str, list[float]]:
    return {
        "near_left_corner": [100.0, 330.0],
        "near_baseline_center": [320.0, 330.0],
        "near_right_corner": [540.0, 330.0],
        "far_right_corner": [540.0, 30.0],
        "far_baseline_center": [320.0, 30.0],
        "far_left_corner": [100.0, 30.0],
        "near_nvz_left": [100.0, 230.0],
        "near_nvz_center": [320.0, 230.0],
        "near_nvz_right": [540.0, 230.0],
        "net_left_sideline": [100.0, 180.0],
        "net_center": [320.0, 180.0],
        "net_right_sideline": [540.0, 180.0],
        "far_nvz_left": [100.0, 130.0],
        "far_nvz_center": [320.0, 130.0],
        "far_nvz_right": [540.0, 130.0],
    }


def test_round3_input_resolution_requires_representable_5px_gate() -> None:
    assert validate_round3_input_resolution(640, 360) == {
        "mode": "full_frame",
        "image_width": 640,
        "image_height": 360,
        "patch_size": None,
        "min_width": 640,
    }

    with pytest.raises(ValueError, match="at least 640px wide"):
        validate_round3_input_resolution(320, 180)

    assert validate_round3_input_resolution(320, 180, patch_size=640)["mode"] == "patch_based"


def test_line_masks_fit_lines_and_intersect_canonical_keypoints() -> None:
    keypoints = _rectangular_keypoints()

    masks = line_mask_targets_for_keypoints(keypoints, width=640, height=360, line_width=3)
    assert set(masks) == {family.name for family in COURT_LINE_FAMILIES}
    assert masks["near_baseline"].shape == (360, 640)
    assert masks["near_baseline"].max() == pytest.approx(1.0)
    assert masks["near_baseline"][330, 320] == pytest.approx(1.0)

    lines = fit_court_lines_from_masks(masks, threshold=0.5)
    predicted = intersect_court_keypoints_from_lines(lines)

    assert set(predicted) == {point.name for point in PICKLEBALL_KEYPOINTS}
    for name, expected in keypoints.items():
        assert predicted[name] == pytest.approx(expected, abs=0.75)
