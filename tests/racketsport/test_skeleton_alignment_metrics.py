from __future__ import annotations

import math

import pytest

from threed.racketsport.skeleton_alignment_metrics import score_skeleton_alignment


JOINT_NAMES = ["left_ankle", "right_ankle", "pelvis", "nose"]


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


def _skeleton(*, x_offset_m: float = 0.0, nose_jitter_m: float = 0.0) -> dict:
    frames = []
    for frame_idx, root_x in enumerate((0.0, 0.1, 0.2)):
        joints = [
            [root_x - 0.15 + x_offset_m, 0.0, 0.0],
            [root_x + 0.15 + x_offset_m, 0.0, 0.0],
            [root_x + x_offset_m, 0.0, 0.9],
            [root_x + x_offset_m, nose_jitter_m if frame_idx == 1 else 0.0, 1.7],
        ]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "transl_world": [root_x + x_offset_m, 0.0, 0.0],
                "joints_world": joints,
                "joint_conf": [0.95, 0.95, 0.99, 0.99],
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "racketsport_skeleton3d_v2",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": JOINT_NAMES,
        "players": [{"id": "p1", "frames": frames}],
    }


def _keypoints() -> dict:
    player_frames = []
    for frame in _skeleton()["players"][0]["frames"]:
        keypoints = []
        for joint_name, point in zip(JOINT_NAMES, frame["joints_world"], strict=True):
            x_px, y_px = _project(point)
            keypoints.append({"joint": joint_name, "x_px": x_px, "y_px": y_px, "conf": 0.95})
        player_frames.append({"frame_idx": frame["frame_idx"], "t": frame["t"], "keypoints": keypoints})
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
        "players": [{"id": "p1", "frames": player_frames}],
    }


def test_score_skeleton_alignment_reports_zero_projection_error_for_ray_consistent_skeleton() -> None:
    report = score_skeleton_alignment(_skeleton(), _keypoints(), _calibration())

    assert report["artifact_type"] == "racketsport_skeleton_alignment_metrics"
    assert report["projection_error_px"]["overall"]["p90"] == pytest.approx(0.0, abs=1e-9)
    assert report["stature_stability"]["players"]["p1"]["stature_p90_abs_dev_m"] == pytest.approx(0.0)
    assert report["bone_length_variance"]["summary"]["max_cv"] == pytest.approx(0.0)


def test_score_skeleton_alignment_exposes_projection_offset_and_joint_group_jitter() -> None:
    report = score_skeleton_alignment(_skeleton(x_offset_m=0.2, nose_jitter_m=0.05), _keypoints(), _calibration())

    assert report["projection_error_px"]["overall"]["p50"] > 10.0
    assert report["jitter_m_per_frame"]["groups"]["head"]["p90"] > report["jitter_m_per_frame"]["groups"]["lower_body"]["p90"]
