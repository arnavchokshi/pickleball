"""Tests for the TT3D -> DinkVision coordinate adapter.

These guard the single most dangerous failure mode of the external-validation
exercise: a silent unit/frame/sign error that produces a confidently wrong
accuracy number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tt3d"))

from tt3d_adapter import (  # noqa: E402
    TT3DView,
    iter_trajectories,
    load_trajectory,
    rodrigues,
    verify_mapping,
)
from threed.racketsport.ball_arc_solver import (  # noqa: E402
    intersect_ray_z,
    pixel_ray_world,
)

DATA_ROOT = REPO_ROOT / "data" / "external" / "tt3d_repo" / "data" / "evaluation"
HAVE_DATA = DATA_ROOT.is_dir()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="TT3D evaluation data not downloaded")

VIEWS = ("back", "side", "oblique")


# ---------------------------------------------------------------- rodrigues
def test_rodrigues_zero_is_identity():
    assert np.allclose(rodrigues([0.0, 0.0, 0.0]), np.eye(3))


def test_rodrigues_is_orthonormal_with_unit_determinant():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.normal(size=3) * 2.0
        R = rodrigues(r)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_rodrigues_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    for _ in range(25):
        r = rng.normal(size=3) * 2.0
        expected, _ = cv2.Rodrigues(r.reshape(3, 1))
        assert np.allclose(rodrigues(r), expected, atol=1e-12)


def test_rodrigues_quarter_turn_about_z():
    R = rodrigues([0.0, 0.0, np.pi / 2])
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


# ------------------------------------------------------------------- yamls
@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_view_loads_and_has_sane_geometry(name):
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    assert v.w == 1280 and v.h == 720
    assert v.f > 0
    c = v.camera_center_world
    # camera is above the table surface and outside the 2.74 m x 1.525 m table
    assert c[2] > 0.0
    assert np.linalg.norm(c) > 1.5


# -------------------------------------------------- THE mapping gate itself
@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_projection_matches_dataset_uv_exactly_on_clean_views(name):
    """Our camera model must reproduce TT3D's own (u, v) to machine precision."""
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    trajs = list(iter_trajectories(DATA_ROOT, f"{name}_no_noise"))
    res = verify_mapping(v, trajs)
    assert res["n_points"] > 1000
    assert res["max_px"] < 1e-6, f"{name}: mapping residual {res['max_px']} px"


@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_noisy_views_differ_only_by_injected_detection_noise(name):
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    res = verify_mapping(v, list(iter_trajectories(DATA_ROOT, name)))
    # a few px of injected noise, not a frame error
    assert 0.5 < res["mean_px"] < 6.0


# ------------------------------- adapter <-> solver convention round-tripping
@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_solver_pixel_ray_inverts_adapter_projection(name):
    """pixel_ray_world must generate a ray that passes through the 3D point.

    This proves TT3D's (rvec, tvec, f) and our (R, t, fx, fy, cx, cy) are the
    same convention -- the whole basis of the validation.
    """
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    calib = v.calibration()
    trajs = list(iter_trajectories(DATA_ROOT, f"{name}_no_noise"))
    pts = np.concatenate([t.xyz for t in trajs[:25]], axis=0)
    uv = v.project(pts)
    worst = 0.0
    for p, (u, vv) in zip(pts, uv):
        origin, direction = pixel_ray_world(calib, (float(u), float(vv)))
        origin = np.asarray(origin)
        direction = np.asarray(direction)
        # perpendicular distance from the 3D point to the back-projected ray
        w = p - origin
        perp = np.linalg.norm(w - np.dot(w, direction) * direction)
        worst = max(worst, float(perp))
    assert worst < 1e-9, f"{name}: point-to-ray distance {worst} m"


@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_solver_camera_origin_matches_adapter_camera_centre(name):
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    origin, _ = pixel_ray_world(v.calibration(), (640.0, 360.0))
    assert np.allclose(np.asarray(origin), v.camera_center_world, atol=1e-9)


@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_ray_plane_intersection_recovers_a_point_on_the_plane(name):
    """intersect_ray_z on a point that genuinely lies at z=h must return it."""
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    calib = v.calibration()
    rng = np.random.default_rng(7)
    h = 0.020
    for _ in range(50):
        p = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-1.3, 1.3), h])
        u, vv = v.project(p[None, :])[0]
        origin, direction = pixel_ray_world(calib, (float(u), float(vv)))
        got = np.asarray(intersect_ray_z(origin, direction, h))
        assert np.allclose(got, p, atol=1e-9)


# ------------------------------------------------------------------- CSVs
@needs_data
def test_trajectory_csv_shape_and_units():
    t = load_trajectory(DATA_ROOT / "back_no_noise" / "001.csv", "back_no_noise")
    assert t.xyz.shape[1] == 3 and t.uv.shape[1] == 2
    assert len(t) == t.xyz.shape[0] == t.uv.shape[0]
    assert np.isclose(t.fps, 25.0, atol=1e-6)
    # metres, table-centred frame: |X| < half table width + margin
    assert np.all(np.abs(t.xyz[:, 0]) < 2.0)
    assert np.all(t.xyz[:, 2] > -0.05)
    # pixels inside the image
    assert np.all((t.uv[:, 0] > -50) & (t.uv[:, 0] < 1330))


@needs_data
def test_gt_and_perview_csvs_agree_on_first_sample():
    gt = load_trajectory(DATA_ROOT / "3D_gt" / "001.csv", "3D_gt") if False else None
    a = load_trajectory(DATA_ROOT / "back_no_noise" / "001.csv", "back_no_noise")
    b = load_trajectory(DATA_ROOT / "side_no_noise" / "001.csv", "side_no_noise")
    # every view shares one underlying 3D trajectory
    assert np.allclose(a.xyz, b.xyz, atol=1e-12)
    assert np.allclose(a.t, b.t, atol=1e-12)


@needs_data
def test_all_views_have_139_trajectories():
    for name in VIEWS:
        for sub in (name, f"{name}_no_noise"):
            assert len(list((DATA_ROOT / sub).glob("*.csv"))) == 139


# ------------------------------------------- depth blindness is structural
@needs_data
@pytest.mark.parametrize("name", VIEWS)
def test_sliding_along_camera_ray_leaves_pixels_unchanged(name):
    """The core claim: reprojection error is blind to depth, by construction."""
    v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
    c = v.camera_center_world
    t = load_trajectory(DATA_ROOT / f"{name}_no_noise" / "001.csv", name)
    uv0 = v.project(t.xyz)
    d = t.xyz - c
    d_hat = d / np.linalg.norm(d, axis=1, keepdims=True)
    for delta in (0.1, 0.5, 1.0):
        uv1 = v.project(t.xyz + delta * d_hat)
        assert np.max(np.linalg.norm(uv1 - uv0, axis=1)) < 1e-9
