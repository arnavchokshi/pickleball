from __future__ import annotations

import math

import pytest

from threed.racketsport.camera_distortion import (
    distort_normalized,
    distort_pixel,
    distortion_coefficients,
    has_distortion,
    is_radially_invertible,
    max_normalized_radius,
    undistort_normalized,
    undistort_pixel,
)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

INTRINSICS_ZERO = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0, "dist": [0.0, 0.0, 0.0, 0.0]}
INTRINSICS_BARREL = {"fx": 1252.8, "fy": 1252.8, "cx": 960.0, "cy": 540.0, "dist": [-0.30035, 0.09861, 0.0, 0.0]}


def test_zero_distortion_is_an_exact_identity():
    for uv in ((0.0, 0.0), (960.0, 540.0), (1919.0, 1079.0), (-45.5, 1207.1)):
        assert undistort_pixel(INTRINSICS_ZERO, uv) == (uv[0], uv[1])
        assert distort_pixel(INTRINSICS_ZERO, uv) == (uv[0], uv[1])
    assert not has_distortion(INTRINSICS_ZERO)
    assert not has_distortion({"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0})
    assert has_distortion(INTRINSICS_BARREL)


def test_missing_or_malformed_dist_degrades_to_zero_distortion():
    assert distortion_coefficients(None) == (0.0, 0.0, 0.0, 0.0)
    assert distortion_coefficients({"dist": "nope"}) == (0.0, 0.0, 0.0, 0.0)
    assert distortion_coefficients({"dist": [-0.3]}) == (-0.3, 0.0, 0.0, 0.0)
    assert distortion_coefficients({"dist": [float("nan"), 0.1]}) == (0.0, 0.1, 0.0, 0.0)


@pytest.mark.parametrize(
    "dist",
    [
        [-0.30035, 0.09861, 0.0, 0.0],
        [-0.18, 0.0, 0.0, 0.0],
        [0.25, -0.1, 0.0, 0.0],
        [-0.28, 0.05, 0.002, -0.001],
    ],
)
def test_undistort_inverts_distort_over_the_whole_frame(dist):
    intrinsics = {"fx": 1252.8, "fy": 1252.8, "cx": 960.0, "cy": 540.0, "dist": dist}
    assert is_radially_invertible(dist[0], dist[1], max_normalized_radius((1920, 1080), 1252.8, 1252.8, 960.0, 540.0))
    for u in (1.0, 480.0, 960.0, 1440.0, 1919.0):
        for v in (1.0, 270.0, 540.0, 810.0, 1079.0):
            back = distort_pixel(intrinsics, undistort_pixel(intrinsics, (u, v)))
            assert back[0] == pytest.approx(u, abs=1e-4)
            assert back[1] == pytest.approx(v, abs=1e-4)


def test_non_invertible_radial_models_are_detected():
    max_radius = max_normalized_radius((1920, 1080), 1252.8, 1252.8, 960.0, 540.0)
    assert max_radius == pytest.approx(0.8786, abs=1e-3)
    # k1 = -0.6 folds at f(r) = 0.497 and k1 = -0.4276 at f(r) = 0.589: neither
    # ever reaches the 0.879 this frame needs, so neither is invertible here.
    assert not is_radially_invertible(-0.6, 0.0, max_radius)
    assert not is_radially_invertible(-0.4276, 0.0, max_radius)
    # The same -0.4276 IS invertible on a longer lens, where the frame only
    # reaches r = 0.438 -- feasibility is a property of the model AND the frame.
    assert is_radially_invertible(-0.4276, 0.0, max_normalized_radius((1920, 1080), 2514.0, 2514.0, 960.0, 540.0))
    assert is_radially_invertible(-0.30035, 0.09861, max_radius)
    assert is_radially_invertible(0.0, 0.0, max_radius)


def test_forward_model_matches_opencv_project_points():
    """Our `distort_normalized` must be OpenCV's model, not a lookalike."""

    intrinsics = INTRINSICS_BARREL
    k = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]])
    dist = np.array(intrinsics["dist"], dtype=np.float64)
    points = np.array([[x, y, 1.0] for x in (-0.4, -0.1, 0.0, 0.2, 0.5) for y in (-0.3, 0.0, 0.35)], dtype=np.float64)
    expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, dist)
    expected = expected.reshape(-1, 2)
    for point, want in zip(points, expected):
        got = distort_pixel(intrinsics, (point[0] * intrinsics["fx"] + intrinsics["cx"], point[1] * intrinsics["fy"] + intrinsics["cy"]))
        assert got[0] == pytest.approx(float(want[0]), abs=1e-9)
        assert got[1] == pytest.approx(float(want[1]), abs=1e-9)


def test_inverse_is_the_exact_inverse_of_opencvs_forward_model():
    """Scored against `cv2.projectPoints`, the authoritative forward direction.

    `cv2.undistortPoints` is itself only a 5-iteration fixed point and is ~0.4px
    off at the frame corner for these coefficients, so the honest check is: undo
    the distortion with ours, push the result back through OpenCV's own forward
    model, and land on the pixel we started from.
    """

    intrinsics = INTRINSICS_BARREL
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist = np.array(intrinsics["dist"], dtype=np.float64)
    for u in (100.0, 960.0, 1800.0):
        for v in (80.0, 540.0, 1000.0):
            ideal = undistort_pixel(intrinsics, (u, v))
            normalized = np.array([[(ideal[0] - cx) / fx, (ideal[1] - cy) / fy, 1.0]], dtype=np.float64)
            round_trip, _ = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, dist)
            got = round_trip.reshape(2)
            assert math.dist((float(got[0]), float(got[1])), (u, v)) < 1e-4


def test_normalized_helpers_round_trip_at_the_optical_centre():
    coefficients = (-0.3, 0.1, 0.0, 0.0)
    assert undistort_normalized(0.0, 0.0, coefficients) == (0.0, 0.0)
    assert distort_normalized(0.0, 0.0, coefficients) == (0.0, 0.0)
