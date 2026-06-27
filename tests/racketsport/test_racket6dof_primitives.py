from __future__ import annotations

import pytest

from threed.racketsport.racket6dof import (
    SE3PoseConfidence,
    normalize_face_normal,
    validate_contact_point_face_cm,
    validate_paddle_dimensions,
)


def test_validate_paddle_dimensions_accepts_named_or_short_dimension_keys() -> None:
    named = validate_paddle_dimensions({"length": 16.0, "width": 8.0})
    short = validate_paddle_dimensions({"h": 15.5, "w": 7.75})

    assert named.length_in == pytest.approx(16.0)
    assert named.width_in == pytest.approx(8.0)
    assert named.length_cm == pytest.approx(40.64)
    assert named.width_cm == pytest.approx(20.32)
    assert short.length_in == pytest.approx(15.5)
    assert short.width_in == pytest.approx(7.75)


def test_validate_paddle_dimensions_rejects_missing_non_positive_or_non_finite_values() -> None:
    with pytest.raises(ValueError, match="paddle_dims_in"):
        validate_paddle_dimensions({"length": 16.0})

    with pytest.raises(ValueError, match="positive"):
        validate_paddle_dimensions({"length": 16.0, "width": 0.0})

    with pytest.raises(ValueError, match="finite"):
        validate_paddle_dimensions({"length": float("nan"), "width": 8.0})


def test_normalize_face_normal_returns_unit_vector_without_changing_direction() -> None:
    assert normalize_face_normal([0.0, 0.0, 4.0]) == pytest.approx((0.0, 0.0, 1.0))
    assert normalize_face_normal([3.0, 4.0, 0.0]) == pytest.approx((0.6, 0.8, 0.0))

    with pytest.raises(ValueError, match="face_normal"):
        normalize_face_normal([1.0, 0.0])

    with pytest.raises(ValueError, match="non-zero"):
        normalize_face_normal([0.0, 0.0, 0.0])


def test_validate_contact_point_face_cm_accepts_points_inside_paddle_face() -> None:
    dims = validate_paddle_dimensions({"length": 16.0, "width": 8.0})

    assert validate_contact_point_face_cm([0.0, 0.0], dims) == pytest.approx((0.0, 0.0))
    assert validate_contact_point_face_cm([10.16, -20.32], dims) == pytest.approx((10.16, -20.32))


def test_validate_contact_point_face_cm_rejects_points_outside_paddle_face() -> None:
    dims = validate_paddle_dimensions({"length": 16.0, "width": 8.0})

    with pytest.raises(ValueError, match="width"):
        validate_contact_point_face_cm([10.17, 0.0], dims)

    with pytest.raises(ValueError, match="length"):
        validate_contact_point_face_cm([0.0, 20.33], dims)

    with pytest.raises(ValueError, match="contact_point_face_cm"):
        validate_contact_point_face_cm([0.0], dims)


def test_se3_pose_confidence_validates_cpu_only_ukf_placeholder_fields() -> None:
    pose = SE3PoseConfidence(
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.1, 0.2, 0.3],
        confidence=0.85,
    )

    assert pose.R == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert pose.t == pytest.approx((0.1, 0.2, 0.3))
    assert pose.confidence == pytest.approx(0.85)
    assert pose.source == "ukf_placeholder"

    with pytest.raises(ValueError, match="R"):
        SE3PoseConfidence(R=[[1.0]], t=[0.0, 0.0, 0.0], confidence=0.5)

    with pytest.raises(ValueError, match="t"):
        SE3PoseConfidence(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0],
            confidence=0.5,
        )

    with pytest.raises(ValueError, match="confidence"):
        SE3PoseConfidence(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 0.0],
            confidence=1.01,
        )
