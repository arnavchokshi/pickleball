from __future__ import annotations

import math

import pytest

from threed.racketsport.skeleton_lift_2d import Lift2DConfig, lift_skeleton_from_2d


JOINT_NAMES = ["left_ankle", "right_ankle", "pelvis", "nose"]
COCO_BODY_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "dist": []},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "homography": [[80.0, 0.0, 320.0], [0.0, 80.0, 240.0], [0.0, 0.0, 1.0]],
        "image_size": [640, 480],
    }


def _project(point: list[float]) -> list[float]:
    x, y, z = point
    depth = z + 10.0
    return [800.0 * x / depth + 320.0, 800.0 * y / depth + 240.0]


def _world_frames() -> list[dict[str, list[list[float]]]]:
    frames = []
    for frame_idx, root_x in enumerate((0.0, 0.1)):
        frames.append(
            {
                "frame_idx": frame_idx,
                "joints": [
                    [root_x - 0.15, 0.0, 0.0],
                    [root_x + 0.15, 0.0, 0.0],
                    [root_x, 0.0, 0.9],
                    [root_x, 0.0, 1.7],
                ],
            }
        )
    return frames


def _keypoints_from_world(*, ankle_conf: float = 0.95) -> dict:
    players = [{"id": "p1", "height_m": 1.7, "frames": []}]
    for frame in _world_frames():
        keypoints = []
        for joint_name, point in zip(JOINT_NAMES, frame["joints"], strict=True):
            x_px, y_px = _project(point)
            conf = ankle_conf if joint_name.endswith("ankle") else 0.99
            keypoints.append({"joint": joint_name, "x_px": x_px, "y_px": y_px, "conf": conf})
        players[0]["frames"].append(
            {
                "frame_idx": frame["frame_idx"],
                "t": frame["frame_idx"] / 30.0,
                "keypoints": keypoints,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "fps": 30.0,
        "convention": "synthetic_coco",
        "joint_names": JOINT_NAMES,
        "bone_priors": [
            {"parent": "left_ankle", "child": "pelvis", "length_m": math.hypot(0.15, 0.9)},
            {"parent": "pelvis", "child": "nose", "length_m": 0.8},
        ],
        "players": players,
    }


def _tracks() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": "p1",
                "frames": [
                    {"frame_idx": 0, "t": 0.0, "world_xy": [0.0, 0.0]},
                    {"frame_idx": 1, "t": 1.0 / 30.0, "world_xy": [0.1, 0.0]},
                ],
            }
        ],
    }


def test_lift_skeleton_from_2d_projects_back_to_authoritative_keypoints() -> None:
    skeleton, report = lift_skeleton_from_2d(
        _keypoints_from_world(),
        tracks_payload=_tracks(),
        calibration_payload=_calibration(),
        config=Lift2DConfig(root_smoothing_radius=0),
    )

    assert skeleton["artifact_type"] == "racketsport_skeleton3d_v2"
    assert skeleton["provenance"]["lane"] == "lane_b_2d_first"
    assert skeleton["joint_names"] == JOINT_NAMES
    assert report["summary"]["frame_count"] == 2
    assert report["summary"]["root_sources"]["ankle_midpoint_ray_court_plane"] == 2

    for player in skeleton["players"]:
        for frame in player["frames"]:
            source = _keypoints_from_world()["players"][0]["frames"][frame["frame_idx"]]["keypoints"]
            by_joint = {item["joint"]: item for item in source}
            assert frame["transl_world"][2] == pytest.approx(0.0)
            for joint_name, joint_world in zip(JOINT_NAMES, frame["joints_world"], strict=True):
                projected = _project(joint_world)
                detected = by_joint[joint_name]
                assert math.hypot(projected[0] - detected["x_px"], projected[1] - detected["y_px"]) < 1e-6

    pelvis_idx = JOINT_NAMES.index("pelvis")
    nose_idx = JOINT_NAMES.index("nose")
    frame0 = skeleton["players"][0]["frames"][0]
    assert math.dist(frame0["joints_world"][pelvis_idx], frame0["joints_world"][nose_idx]) == pytest.approx(0.8)


def test_lift_skeleton_from_2d_uses_track_root_when_ankles_are_not_visible() -> None:
    skeleton, report = lift_skeleton_from_2d(
        _keypoints_from_world(ankle_conf=0.01),
        tracks_payload=_tracks(),
        calibration_payload=_calibration(),
        config=Lift2DConfig(min_joint_confidence=0.2, root_smoothing_radius=0),
    )

    assert report["summary"]["root_sources"]["track_world_xy"] == 2
    assert skeleton["players"][0]["frames"][1]["transl_world"][:2] == pytest.approx([0.1, 0.0])


def _coco_body_keypoints_with_one_scrambled_frame() -> dict:
    base_points = {
        "nose": [0.0, 0.0, 1.7],
        "left_shoulder": [-0.22, 0.0, 1.42],
        "right_shoulder": [0.22, 0.0, 1.42],
        "left_elbow": [-0.45, 0.0, 1.15],
        "right_elbow": [0.45, 0.0, 1.15],
        "left_wrist": [-0.58, 0.0, 0.95],
        "right_wrist": [0.58, 0.0, 0.95],
        "left_hip": [-0.16, 0.0, 0.92],
        "right_hip": [0.16, 0.0, 0.92],
        "left_knee": [-0.17, 0.0, 0.47],
        "right_knee": [0.17, 0.0, 0.47],
        "left_ankle": [-0.18, 0.0, 0.0],
        "right_ankle": [0.18, 0.0, 0.0],
    }
    frames = []
    for frame_idx in range(7):
        points = {name: [point[0] + frame_idx * 0.02, point[1], point[2]] for name, point in base_points.items()}
        if frame_idx == 3:
            points["right_shoulder"] = [points["right_shoulder"][0] + 2.0, 0.0, 1.42]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "keypoints": [
                    {"joint": joint_name, "x_px": _project(points[joint_name])[0], "y_px": _project(points[joint_name])[1], "conf": 0.95}
                    for joint_name in COCO_BODY_NAMES
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "fps": 30.0,
        "convention": "synthetic_coco_wholebody",
        "joint_names": COCO_BODY_NAMES,
        "players": [{"id": "p1", "height_m": 1.7, "frames": frames}],
    }


def test_lift_skeleton_from_2d_flags_per_player_bone_length_outliers() -> None:
    skeleton, report = lift_skeleton_from_2d(
        _coco_body_keypoints_with_one_scrambled_frame(),
        tracks_payload=None,
        calibration_payload=_calibration(),
        config=Lift2DConfig(root_smoothing_radius=0),
    )

    frames = skeleton["players"][0]["frames"]
    flagged = [frame for frame in frames if frame["skeleton_implausible"]]

    assert [frame["frame_idx"] for frame in flagged] == [3]
    assert flagged[0]["trust_band"]["badge"] == "low_confidence"
    assert flagged[0]["trust_band"]["gate_id"] == "skeleton_lift_2d_plausibility"
    assert report["summary"]["skeleton_implausible_frame_count"] == 1
    assert report["players"]["p1"]["skeleton_implausible_frame_count"] == 1
