from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from threed.racketsport import ball_court_filter, ball_inout_uncertainty, ball_physics3d, placement, virtual_world
from threed.racketsport.coordinates import CoordinateSpace, project_world_points
from threed.racketsport.court_calibration import (
    project_image_points_to_world,
    project_image_points_to_world_typed,
    project_world_points as project_world_points_legacy,
)
from threed.racketsport.paddle_pose_fused import (
    _project_world_points_for_raw_detector_reference,
    build_paddle_pose_fused_from_skeleton,
)
from threed.racketsport.person_fast import person_detection_from_bbox
from threed.racketsport.schemas import CourtCalibration


_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO
    / "runs/lanes/w7_critique_20260709/wolv_world"
    / "wolverine_mixed_0200_mid_steep_corner"
)
_FIXTURE_SHA256 = {
    "court_calibration.json": "fb4e6f7f54d2c40e2c7b491e436261f747240945a6f0d154c4dd943e28edbacf",
    "ball_track.json": "fd33df9e48950c79777b80fbf6ae954e735761f2d3e4fa53d0d98fd9bc97a9c5",
    "ball_track_arc_solved.json": "4fc4ca25dde1b2bf8018a41b53926fcd6b9cfbc17e8afb76c8425b7e3fb3b0bb",
    "placement.json": "f624c2782575ff75cbfabd65345f06ed3a72cde6ffcf742fbde1ca3e105a7049",
    "skeleton3d.json": "1e22da40152d2b91eee1cb57acf90c2d9ac78416c0faf3b62f8079fa2ddb0e0f",
    "tracks.json": "0489a0697f4abf541bccf723bd704a8c9a189d9a31a603fe46547202b022a93d",
    "virtual_world.json": "9405813d8e3296c601fd6cb71e370564bad0414dc59ca2473df0e6dd5e7a69b5",
}


def _load(name: str) -> dict:
    path = _FIXTURE / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _FIXTURE_SHA256[name]
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_frozen_wolverine_court_projection_seams_are_exact() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))

    legacy_world_to_image = project_world_points_legacy(
        calibration.extrinsics, calibration.intrinsics, calibration.world_pts
    )
    typed_world_to_image = project_world_points(
        calibration.extrinsics,
        calibration.intrinsics,
        calibration.world_pts,
        input_space=CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
        output_space=CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        reference_space=CoordinateSpace.PIXELS_RAW_NATIVE,
    )
    assert typed_world_to_image == legacy_world_to_image
    assert _digest(typed_world_to_image) == "95230ba3bcea1db43e76583b0593292b8471ab35d5a68926fdd43b6a833f6a03"

    legacy_image_to_world = project_image_points_to_world(calibration.homography, calibration.image_pts)
    typed_image_to_world = project_image_points_to_world_typed(
        calibration.homography,
        calibration.image_pts,
        input_space=CoordinateSpace.PIXELS_RAW_NATIVE,
        homography_space=CoordinateSpace.PIXELS_RAW_NATIVE,
        output_space=CoordinateSpace.WORLD_XY_HOMOGRAPHY_M,
    )
    assert typed_image_to_world == legacy_image_to_world
    assert _digest(typed_image_to_world) == "56f9fd9502180a576ed129e7347b80394b1356d2fb8d42b99ce4ca4189b96473"


def test_frozen_wolverine_paddle_detector_projection_route_is_exact() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))
    legacy = project_world_points_legacy(
        calibration.extrinsics,
        calibration.intrinsics,
        calibration.world_pts,
    )
    typed = _project_world_points_for_raw_detector_reference(
        calibration.extrinsics,
        calibration.intrinsics,
        calibration.world_pts,
    )

    assert typed == legacy
    assert _digest(typed) == "95230ba3bcea1db43e76583b0593292b8471ab35d5a68926fdd43b6a833f6a03"


def test_frozen_wolverine_person_bottom_center_projection_is_exact() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))
    frame = _load("tracks.json")["players"][0]["frames"][0]
    detection = person_detection_from_bbox(
        calibration,
        bbox_xyxy=tuple(frame["bbox"]),
        confidence=frame["conf"],
        bbox_space=CoordinateSpace.PIXELS_RAW_NATIVE,
        homography_space=CoordinateSpace.PIXELS_RAW_NATIVE,
    )
    assert _digest(detection.__dict__) == "1adba0b968b723733b548cf42d19ac2b035de2901914ed4de17a01e133d230e9"


def test_frozen_wolverine_paddle_raw_numeric_payload_is_exact() -> None:
    skeleton = _load("skeleton3d.json")
    output = build_paddle_pose_fused_from_skeleton(
        skeleton,
        clip_id="wolverine_mixed_0200_mid_steep_corner",
    )
    numeric = [
        {
            "id": player["id"],
            "frames": [
                {
                    "t": frame["t"],
                    "frame": frame["frame"],
                    "pose_se3": frame["pose_se3"],
                    "conf": frame["conf"],
                }
                for frame in player["frames"]
            ],
        }
        for player in output["players"]
    ]
    assert sum(len(player["frames"]) for player in output["players"]) == 705
    assert _digest(numeric) == "b2a725a775045b465e0b4ce21e1162c169e78df95ca3e3ddf1513c70ef8a8a93"
    assert output["world_frame"] == "court_Z0"
    assert output["coordinate_frame"] == "court_netcenter_z_up_m"
    assert output["coordinate_space"] == "world_court_netcenter_z_up_m"


def test_frozen_wolverine_placement_stage_projection_is_byte_stable() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))
    tracks = _load("tracks.json")
    _load("placement.json")
    camera_matrix = np.asarray(
        [
            [calibration.intrinsics.fx, 0.0, calibration.intrinsics.cx],
            [0.0, calibration.intrinsics.fy, calibration.intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    homography = np.asarray(calibration.homography, dtype=float)
    projected = []
    for player in tracks["players"]:
        for frame in player["frames"]:
            x1, _y1, x2, y2 = frame["bbox"]
            signal = placement._signal_from_pixel(
                name="bbox",
                pixel_xy=[(x1 + x2) / 2.0, y2],
                confidence=frame["conf"],
                sigma_px=24.0,
                side=player["side"],
                homography=homography,
                camera_matrix=camera_matrix,
                dist=calibration.intrinsics.dist,
                undistort_applied=False,
                homography_pixel_space=CoordinateSpace.PIXELS_RAW_NATIVE,
                config=placement.PlacementConfig(),
            )
            projected.append([player["id"], frame.get("frame_idx"), signal["xy"], signal["used"]])

    assert len(projected) == 705
    assert _digest(projected) == "6b066685ad739b455ce968074769f384318dae68aa310d426c4f7195a13e3389"


def test_frozen_wolverine_ball_filter_and_arc_projection_are_byte_stable() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))
    ball = _load("ball_track.json")
    arc = _load("ball_track_arc_solved.json")
    polygon = ball_court_filter.build_target_court_polygon(calibration)
    visible_world = [
        ball_court_filter._target_image_xy_to_world_xy(
            frame["xy"], calibration=calibration, target_size=None
        )
        for frame in ball["frames"]
        if frame["visible"]
    ]
    world = np.asarray(
        [frame["world_xyz"] for frame in arc["frames"] if frame.get("world_xyz") is not None],
        dtype=float,
    )
    camera = {
        "intrinsics": calibration.intrinsics,
        "rotation": np.asarray(calibration.extrinsics.R, dtype=float),
        "translation": np.asarray(calibration.extrinsics.t, dtype=float),
        "reference_space": CoordinateSpace.PIXELS_RAW_NATIVE,
    }
    arc_pixels = ball_physics3d._project_world_array(world, camera=camera, np_module=np)

    assert _digest(polygon) == "528d09ba26c4637c8122390b4b5435aaffec7b13f0f14c1ee6cd0493f8e4f43e"
    assert len(visible_world) == 243
    assert _digest(visible_world) == "95dbe43a485aeec2cd32af9103a5a125e32ccf85b682631a8d8c452725f59102"
    assert len(arc_pixels) == 300
    assert _digest(arc_pixels.tolist()) == "26d2ac5e3f64889685e51df0e81952f67cbf4c422c445e425ada570528e5f71c"


def test_frozen_wolverine_inout_pose_and_world_ball_are_byte_stable() -> None:
    calibration = CourtCalibration.model_validate(_load("court_calibration.json"))
    ball = _load("ball_track.json")
    _load("virtual_world.json")
    pose = ball_inout_uncertainty.solve_manual_corner_camera_pose(
        calibration.image_pts,
        calibration.world_pts,
    )
    world_ball = [
        virtual_world._ball_world_xyz(
            frame,
            calibration=calibration,
            ball_world_policy="court_plane_approx_for_review_only",
        )
        for frame in ball["frames"]
    ]

    assert _digest(pose.__dict__) == "b00a262abe2e3d564b84e0c47fd03d7e7036c238734e76896a69941a31b1416e"
    assert len(world_ball) == 300
    assert _digest(world_ball) == "f0b1cd353832d73f7fd11d8e047b92cb33727a03185e95178e5a119bf33f7ab3"
