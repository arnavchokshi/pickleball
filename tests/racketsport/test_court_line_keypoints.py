from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.racketsport.train_court_keypoint_heatmap import court_corner_keypoint_labels
from threed.racketsport.court_keypoint_labels import VISIBLE, load_partial_court_keypoints
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS, keypoint_labels_from_court_corners
from threed.racketsport.court_line_keypoints import (
    _LineGroup,
    _line_from_segment,
    _select_semantic_lines,
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


def test_detect_court_keypoints_from_image_learns_nonwhite_line_color_from_court_surface() -> None:
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
    image[:, :] = (26, 35, 42)
    image[105:, :] = (62, 114, 82)
    cv2.line(image, (40, 126), (680, 126), (245, 245, 245), 7, cv2.LINE_AA)
    muted_paint = (142, 92, 48)
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
        cv2.line(image, _point(labels[start]), _point(labels[end]), muted_paint, 8, cv2.LINE_AA)

    predictions = detect_court_keypoints_from_image(image, white_threshold=245)

    errors = [
        math.dist(predictions.keypoints[name]["xy"], expected_xy)
        for name, expected_xy in labels.items()
    ]
    assert _median(errors) < 3.0
    assert max(errors) < 8.0


def test_detect_court_keypoints_from_image_prefers_plausible_semantic_lines_on_outdoor_eval_clip() -> None:
    cv2 = pytest.importorskip("cv2")
    label_path = Path("eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_corners.json")
    if not label_path.is_file():
        pytest.skip("committed outdoor court-corner label is unavailable")
    row = court_corner_keypoint_labels(json.loads(label_path.read_text(encoding="utf-8")), clip_root=label_path.parent.parent)

    capture = cv2.VideoCapture(str(label_path.parent.parent / "source.mp4"))
    try:
        assert capture.isOpened()
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ok, image = capture.read()
    finally:
        capture.release()
    assert ok

    predictions = detect_court_keypoints_from_image(image)

    errors = [
        math.dist(predictions.keypoints[name]["xy"], expected_xy)
        for name, expected_xy in row["keypoints"].items()
    ]
    assert _median(errors) <= 10.0


def test_detect_court_keypoints_from_image_uses_high_support_sides_on_indoor_eval_clip() -> None:
    cv2 = pytest.importorskip("cv2")
    label_path = Path("eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/labels/court_corners.json")
    if not label_path.is_file():
        pytest.skip("committed indoor court-corner label is unavailable")
    row = court_corner_keypoint_labels(json.loads(label_path.read_text(encoding="utf-8")), clip_root=label_path.parent.parent)

    capture = cv2.VideoCapture(str(label_path.parent.parent / "source.mp4"))
    try:
        assert capture.isOpened()
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ok, image = capture.read()
    finally:
        capture.release()
    assert ok

    predictions = detect_court_keypoints_from_image(image)

    errors = [
        math.dist(predictions.keypoints[name]["xy"], expected_xy)
        for name, expected_xy in row["keypoints"].items()
    ]
    # This is a real-video smoke against a corrected_unverified label, not a CAL acceptance gate.
    assert _median(errors) <= 6.0


def test_detect_court_keypoints_from_image_img1605_wrong_near_strip_is_low_confidence() -> None:
    cv2 = pytest.importorskip("cv2")
    image_path = Path(
        "runs/owner_data/owner_IMG_1605_8a193402780b/prelabels/review_frames/"
        "owner_IMG_1605_8a193402780b/frame_000151.jpg"
    )
    labels_path = Path("eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints_partial.json")
    if not image_path.is_file() or not labels_path.is_file():
        pytest.skip("IMG_1605 reviewed partial-label fixture is unavailable")
    image = cv2.imread(str(image_path))
    assert image is not None

    predictions = detect_court_keypoints_from_image(image)
    labels = load_partial_court_keypoints(labels_path)
    frame = labels.frames[0]
    errors = [
        math.dist(predictions.keypoints[name]["xy"], xy)
        for name, xy in frame.keypoints.items()
        if frame.visibility_by_keypoint.get(name) == VISIBLE
    ]

    assert _median(errors) > 100.0
    assert predictions.confidence < 0.5


def test_select_semantic_lines_uses_regulation_spacing_over_tennis_ordering() -> None:
    groups = [
        _line_group_from_segment(((100.0, 80.0), (100.0, 620.0)), support=620.0),
        _line_group_from_segment(((500.0, 80.0), (500.0, 620.0)), support=620.0),
        _line_group_from_segment(((900.0, 80.0), (900.0, 620.0)), support=620.0),
        _line_group_from_segment(((100.0, 100.0), (900.0, 100.0)), support=820.0),
        _line_group_from_segment(((100.0, 190.0), (900.0, 190.0)), support=820.0),  # tennis service overlay
        _line_group_from_segment(((100.0, 250.0), (900.0, 250.0)), support=820.0),
        _line_group_from_segment(((100.0, 320.0), (900.0, 320.0)), support=820.0),
        _line_group_from_segment(((100.0, 390.0), (900.0, 390.0)), support=820.0),
        _line_group_from_segment(((100.0, 540.0), (900.0, 540.0)), support=820.0),
    ]

    selected = _select_semantic_lines(groups, width=1000.0, height=700.0)

    assert _mean_y(selected["far_baseline"].segment) == pytest.approx(100.0)
    assert _mean_y(selected["far_nvz"].segment) == pytest.approx(250.0)
    assert _mean_y(selected["net"].segment) == pytest.approx(320.0)
    assert _mean_y(selected["near_nvz"].segment) == pytest.approx(390.0)
    assert _mean_y(selected["near_baseline"].segment) == pytest.approx(540.0)


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


def _line_group_from_segment(segment: tuple[tuple[float, float], tuple[float, float]], *, support: float) -> _LineGroup:
    return _LineGroup(
        line=_line_from_segment(segment),
        segment=segment,
        support_length_px=support,
        source_segment_count=1,
    )


def _mean_y(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return (segment[0][1] + segment[1][1]) / 2.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
