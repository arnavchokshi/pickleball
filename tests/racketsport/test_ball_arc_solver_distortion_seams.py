"""The two monocular pinhole seams must honour `intrinsics.dist`.

Before this, `pixel_ray_world` and `_project_world_point` read fx/fy/cx/cy and
ignored `dist` entirely, so a fitted k1 changed nothing downstream while the
focal length it was fit alongside changed everything. These tests pin the two
properties that matter: zero distortion is bit-identical to the old pinhole
arithmetic, and with distortion the two seams are exact inverses of each other
and of OpenCV's model.
"""

from __future__ import annotations

import math

import pytest

from threed.racketsport.ball_arc_solver import _project_world_point, pixel_ray_world

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

# A behind-baseline camera looking down the court, world = court_netcenter_z_up_m.
R = [
    [1.0, 0.0, 0.0],
    [0.0, 0.35836795, -0.93358043],
    [0.0, 0.93358043, 0.35836795],
]
T = [0.0, 1.30978, 13.98261]


def _calibration(dist):
    return {
        "intrinsics": {"fx": 1252.8, "fy": 1252.8, "cx": 960.0, "cy": 540.0, "dist": list(dist)},
        "extrinsics": {"R": [row[:] for row in R], "t": list(T)},
        "image_size": [1920, 1080],
    }


ZERO = _calibration([0.0, 0.0, 0.0, 0.0])
BARREL = _calibration([-0.30035, 0.09861, 0.0, 0.0])

WORLD_POINTS = [
    (0.0, 0.0, 0.9),
    (-3.0, -6.0, 0.037),
    (3.0, 6.0, 1.8),
    (-2.0, 4.0, 0.5),
]


def test_zero_distortion_projection_is_the_bare_pinhole():
    for world in WORLD_POINTS:
        camera_z = sum(R[2][i] * world[i] for i in range(3)) + T[2]
        camera_x = sum(R[0][i] * world[i] for i in range(3)) + T[0]
        camera_y = sum(R[1][i] * world[i] for i in range(3)) + T[1]
        expected = (1252.8 * camera_x / camera_z + 960.0, 1252.8 * camera_y / camera_z + 540.0)
        assert _project_world_point(ZERO, world) == expected


def test_zero_distortion_ray_is_the_bare_pinhole():
    for pixel in ((100.0, 200.0), (960.0, 540.0), (1800.0, 1000.0)):
        _origin, direction = pixel_ray_world(ZERO, pixel)
        raw = (
            (pixel[0] - 960.0) / 1252.8,
            (pixel[1] - 540.0) / 1252.8,
            1.0,
        )
        world_ray = [sum(R[k][i] * raw[k] for k in range(3)) for i in range(3)]
        norm = math.sqrt(sum(v * v for v in world_ray))
        for got, want in zip(direction, world_ray):
            assert got == pytest.approx(want / norm, abs=1e-12)


def test_distorted_projection_matches_opencv():
    rvec, _ = cv2.Rodrigues(np.array(R, dtype=np.float64))
    k = np.array([[1252.8, 0.0, 960.0], [0.0, 1252.8, 540.0], [0.0, 0.0, 1.0]])
    dist = np.array(BARREL["intrinsics"]["dist"], dtype=np.float64)
    expected, _ = cv2.projectPoints(
        np.array(WORLD_POINTS, dtype=np.float64), rvec, np.array(T, dtype=np.float64), k, dist
    )
    for world, want in zip(WORLD_POINTS, expected.reshape(-1, 2)):
        got = _project_world_point(BARREL, world)
        # 1e-5 px absorbs the R -> rvec -> R round trip inside cv2.Rodrigues.
        assert math.dist(got, (float(want[0]), float(want[1]))) < 1e-5


def test_ray_and_projection_are_exact_inverses_under_distortion():
    for world in WORLD_POINTS:
        pixel = _project_world_point(BARREL, world)
        origin, direction = pixel_ray_world(BARREL, pixel)
        # The ray must pass through the point it was projected from.
        offset = [world[i] - origin[i] for i in range(3)]
        depth = sum(offset[i] * direction[i] for i in range(3))
        closest = [origin[i] + depth * direction[i] for i in range(3)]
        assert math.dist(closest, world) < 1e-7


def test_distortion_actually_moves_the_ray():
    """Guard against the change being a silent no-op."""

    pixel = (150.0, 150.0)
    _o_zero, direction_zero = pixel_ray_world(ZERO, pixel)
    _o_barrel, direction_barrel = pixel_ray_world(BARREL, pixel)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(direction_zero, direction_barrel))))))
    assert angle > 3.0
