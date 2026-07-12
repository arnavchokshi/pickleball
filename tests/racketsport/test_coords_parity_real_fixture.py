from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    "skeleton3d.json": "1e22da40152d2b91eee1cb57acf90c2d9ac78416c0faf3b62f8079fa2ddb0e0f",
    "tracks.json": "0489a0697f4abf541bccf723bd704a8c9a189d9a31a603fe46547202b022a93d",
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
