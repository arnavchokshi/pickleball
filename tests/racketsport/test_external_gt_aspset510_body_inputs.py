"""Tests for the ASPset-510 BODY-inference input adapter.

Covers the pure-math core (projection, homography synthesis, bbox padding) with exact
geometric fixtures, plus schema-validity checks against the real
`threed.racketsport.schemas.Tracks`/`CourtCalibration` models this adapter's output must
satisfy (see `threed/racketsport/external_gt_aspset510_body_inputs.py`'s docstring for the
adapter design and the `CourtCalibration._point_lists_must_be_paired` "metric fields"
trap this module deliberately avoids).
"""

from __future__ import annotations

import numpy as np
import pytest

from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES
from threed.racketsport.external_gt_aspset510_body_inputs import (
    BodyInputAdapterError,
    build_court_calibration_payload,
    build_tracks_payload,
    load_camera_calibration_raw,
    project_world_points,
)
from threed.racketsport.schemas import CourtCalibration, Tracks

IDENTITY_K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 300.0], [0.0, 0.0, 1.0]])
IDENTITY_R = np.eye(3)
ZERO_T = np.zeros(3)


def test_load_camera_calibration_raw_converts_mm_translation_to_meters() -> None:
    # Real ASPset-510 layout, confirmed against a real downloaded `<subject>-<camera>.json`
    # (`1e28-mid.json`): row-major flattened [R | t; 0 0 0 1], i.e. translation lives in
    # the last *column* of the first 3 rows ([:3, 3]), not the last row -- verified by
    # checking the real file's R block is orthonormal (det=1, R@R.T=I) only under this
    # reshape, and its final 4 flat values are exactly [0, 0, 0, 1].
    payload = {
        "intrinsic_matrix": [4000.0, 0.0, 1900.0, 0.0, 0.0, 3950.0, 1000.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "extrinsic_matrix": [
            1.0, 0.0, 0.0, -8000.0,
            0.0, 1.0, 0.0, -10.0,
            0.0, 0.0, 1.0, 4000.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    }
    K, R, t_m = load_camera_calibration_raw(payload)
    assert K[0, 0] == 4000.0
    assert K[1, 1] == 3950.0
    np.testing.assert_allclose(R, np.eye(3))
    # extrinsic_matrix translation (mm) at [:3, 3] must be converted to meters.
    np.testing.assert_allclose(t_m, [-8.0, -0.01, 4.0])


def test_project_world_points_matches_hand_solved_pinhole_projection() -> None:
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    uv, depth = project_world_points(points, K=IDENTITY_K, R=IDENTITY_R, t_m=ZERO_T)
    np.testing.assert_allclose(depth, [2.0, 2.0])
    # point on the optical axis projects to the principal point exactly.
    np.testing.assert_allclose(uv[0], [500.0, 300.0])
    # point offset by 1m at depth 2m: u = fx * (1/2) + cx = 1000*0.5+500 = 1000.
    np.testing.assert_allclose(uv[1], [1000.0, 300.0])


def test_project_world_points_uses_world_to_camera_convention() -> None:
    """camera = R @ world + t (matches court_calibration.project_world_points)."""

    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # 90deg yaw
    t = np.array([0.0, 0.0, 5.0])
    world_point = np.array([[0.0, 0.0, 0.0]])
    uv, depth = project_world_points(world_point, K=IDENTITY_K, R=R, t_m=t)
    # camera = R @ [0,0,0] + t = [0,0,5]
    assert depth[0] == pytest.approx(5.0)
    np.testing.assert_allclose(uv[0], [500.0, 300.0])


def test_build_court_calibration_payload_is_schema_valid_and_zero_reprojection() -> None:
    K = np.array([[3970.48, 0.0, 1851.07], [0.0, 3950.72, 1012.90], [0.0, 0.0, 1.0]])
    R = np.array(
        [
            [0.800508, -0.003445, 0.599312],
            [-0.008599, 0.999815, 0.017233],
            [-0.59926, -0.018949, 0.80033],
        ]
    )
    t_m = np.array([-8.933111334, -0.011090412, 4.488441674])
    payload = build_court_calibration_payload(K=K, R=R, t_m=t_m, source_label="test:mid")
    model = CourtCalibration.model_validate(payload)
    assert model.sport == "pickleball"
    assert model.reprojection_error_px.median == 0.0
    assert model.reprojection_error_px.p95 == 0.0
    # the "metric calibration" fields must stay unset (see module docstring) so the
    # CourtCalibration validator does not require a pickleball-specific coordinate_frame.
    assert model.coordinate_frame is None
    assert model.source is None
    assert model.extrinsics.camera_height_m > 0.0


def test_build_court_calibration_payload_rejects_degenerate_plane() -> None:
    # identity extrinsic (camera IS the world origin): Z=0 plane passes through the
    # camera center, which must fail loudly rather than emit a garbage homography.
    with pytest.raises(BodyInputAdapterError):
        build_court_calibration_payload(K=IDENTITY_K, R=IDENTITY_R, t_m=ZERO_T, source_label="test:degenerate")


def test_build_court_calibration_payload_accepts_nonzero_reference_plane() -> None:
    payload = build_court_calibration_payload(
        K=IDENTITY_K,
        R=IDENTITY_R,
        t_m=ZERO_T,
        source_label="test:identity-with-depth",
        reference_plane_z_m=17.5,
        reference_plane_center_xy_m=(0.0, 0.0),
        reference_plane_half_extent_m=1.0,
    )
    model = CourtCalibration.model_validate(payload)
    assert model.reprojection_error_px.median == 0.0
    for world_pt in model.world_pts:
        assert world_pt[2] == pytest.approx(17.5)


def _fixture_frame_joints(depth: float) -> list[list[float]]:
    # a plausible standing pose: 12 shared-core joints spread across a small XY box at a
    # fixed depth, ordered to match SHARED_CORE_JOINT_NAMES.
    base_xy = {
        "left_shoulder": (-0.2, -0.5), "right_shoulder": (0.2, -0.5),
        "left_elbow": (-0.3, -0.2), "right_elbow": (0.3, -0.2),
        "left_wrist": (-0.35, 0.1), "right_wrist": (0.35, 0.1),
        "left_hip": (-0.15, 0.0), "right_hip": (0.15, 0.0),
        "left_knee": (-0.15, 0.5), "right_knee": (0.15, 0.5),
        "left_ankle": (-0.15, 1.0), "right_ankle": (0.15, 1.0),
    }
    return [[base_xy[name][0], base_xy[name][1], depth] for name in SHARED_CORE_JOINT_NAMES]


def test_build_tracks_payload_is_schema_valid_with_padded_bbox() -> None:
    frame_indices = [0, 10, 20]
    joints = [_fixture_frame_joints(5.0) for _ in frame_indices]
    payload, notes = build_tracks_payload(
        frame_indices=frame_indices,
        joints_world_m_by_frame=joints,
        K=IDENTITY_K,
        R=IDENTITY_R,
        t_m=np.array([0.0, 0.0, 0.0]),
        fps=50.0,
    )
    assert notes == []
    model = Tracks.model_validate(payload)
    assert model.fps == 50.0
    assert len(model.players) == 1
    frames = model.players[0].frames
    assert len(frames) == 3
    assert frames[1].t == pytest.approx(10.0 / 50.0)
    x1, y1, x2, y2 = frames[0].bbox
    assert x2 > x1 and y2 > y1
    # padding must strictly grow the box beyond the raw joint-pixel extent.
    raw_uv, _ = project_world_points(np.array(joints[0]), K=IDENTITY_K, R=IDENTITY_R, t_m=ZERO_T)
    raw_x1, raw_y1 = raw_uv.min(axis=0)
    raw_x2, raw_y2 = raw_uv.max(axis=0)
    assert x1 < raw_x1
    assert y1 < raw_y1
    assert x2 > raw_x2
    assert y2 > raw_y2


def test_build_tracks_payload_world_xy_is_ankle_midpoint() -> None:
    frame_indices = [0]
    joints = [_fixture_frame_joints(5.0)]
    payload, _ = build_tracks_payload(
        frame_indices=frame_indices,
        joints_world_m_by_frame=joints,
        K=IDENTITY_K,
        R=IDENTITY_R,
        t_m=ZERO_T,
    )
    world_xy = payload["players"][0]["frames"][0]["world_xy"]
    assert world_xy == pytest.approx([0.0, 1.0])


def test_build_tracks_payload_drops_frames_behind_camera() -> None:
    frame_indices = [0, 10]
    joints = [_fixture_frame_joints(5.0), _fixture_frame_joints(-3.0)]
    payload, notes = build_tracks_payload(
        frame_indices=frame_indices,
        joints_world_m_by_frame=joints,
        K=IDENTITY_K,
        R=IDENTITY_R,
        t_m=ZERO_T,
    )
    assert len(payload["players"][0]["frames"]) == 1
    assert len(notes) == 1
    assert "behind camera" in notes[0]


def test_build_tracks_payload_rejects_mismatched_lengths() -> None:
    with pytest.raises(BodyInputAdapterError):
        build_tracks_payload(
            frame_indices=[0, 10],
            joints_world_m_by_frame=[_fixture_frame_joints(5.0)],
            K=IDENTITY_K,
            R=IDENTITY_R,
            t_m=ZERO_T,
        )
