from __future__ import annotations

import importlib
import importlib.util
from copy import deepcopy

import pytest

from threed.racketsport.joint_schema import BODY65_JOINT_NAMES
from threed.racketsport.skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP


def _splice_func():
    spec = importlib.util.find_spec("threed.racketsport.contact_splice")
    assert spec is not None, "contact_splice module is required for Lane B mesh-to-skeleton hand override"
    module = importlib.import_module("threed.racketsport.contact_splice")
    return module.splice_contact_skeleton_with_body_mesh


def _skeleton3d() -> dict:
    joints = [[float(idx), 0.0, 1.0] for idx in range(len(BODY65_JOINT_NAMES))]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": list(BODY65_JOINT_NAMES),
        "preview_only": False,
        "players": [
            {
                "id": 7,
                "frames": [
                    {"frame_idx": 12, "t": 0.4, "joints_world": deepcopy(joints), "joint_conf": [0.5] * len(joints)}
                ],
            },
            {
                "id": 8,
                "frames": [
                    {"frame_idx": 12, "t": 0.4, "joints_world": deepcopy(joints), "joint_conf": [0.7] * len(joints)}
                ],
            },
        ],
        "provenance": {"lane": "A"},
    }


def _body_compute_execution() -> dict:
    return {
        "artifact_type": "racketsport_body_compute_execution",
        "scheduled_frames": [
            {
                "frame_idx": 12,
                "target_player_ids": [7],
                "source_window_index": 0,
                "reasons": ["contact_window"],
            }
        ],
        "summary": {"scheduled_frame_count": 1, "scheduled_player_frame_count": 1},
    }


def test_contact_splice_overrides_only_scheduled_hitter_wrist_and_hand_joints() -> None:
    splice = _splice_func()
    body_mesh = {
        "artifact_type": "racketsport_body_mesh",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": ["left_wrist", "right_wrist", "left_hand_00"],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 12,
                        "t": 0.4,
                        "source_window_index": 0,
                        "joints_world": [[9.0, 0.0, 1.1], [10.0, 0.0, 1.2], [11.0, 0.0, 1.3]],
                        "joint_conf": [0.91, 0.92, 0.93],
                        "mesh_vertices_world": [[9.0, 0.0, 0.0], [10.0, 0.0, 1.8]],
                    }
                ],
            }
        ],
        "summary": {"mesh_frame_count": 1},
    }

    skeleton, report = splice(
        _skeleton3d(),
        body_mesh=body_mesh,
        body_compute_execution=_body_compute_execution(),
    )

    joint_names = skeleton["joint_names"]
    left_wrist_idx = joint_names.index("left_wrist")
    right_wrist_idx = joint_names.index("right_wrist")
    left_hand_idx = joint_names.index("left_hand_00")
    left_elbow_idx = joint_names.index("left_elbow")
    hitter_frame = skeleton["players"][0]["frames"][0]
    other_frame = skeleton["players"][1]["frames"][0]

    assert hitter_frame["joints_world"][left_wrist_idx] == pytest.approx([9.0, 0.0, 1.1])
    assert hitter_frame["joints_world"][right_wrist_idx] == pytest.approx([10.0, 0.0, 1.2])
    assert hitter_frame["joints_world"][left_hand_idx] == pytest.approx([11.0, 0.0, 1.3])
    assert hitter_frame["joint_conf"][left_wrist_idx] == pytest.approx(0.91)
    assert hitter_frame["joints_world"][left_elbow_idx] == pytest.approx([float(left_elbow_idx), 0.0, 1.0])
    assert other_frame["joints_world"][left_wrist_idx] == pytest.approx([float(left_wrist_idx), 0.0, 1.0])
    assert report["summary"] == {
        "scheduled_contact_count": 1,
        "spliced_contact_count": 1,
        "mesh_unavailable_count": 0,
        "fallback_spliced_count": 0,
        "overridden_joint_count": 3,
    }
    assert skeleton["provenance"]["contact_splice"]["mesh_source"] == "body_mesh.json"


def test_contact_splice_keeps_lane_a_skeleton_and_flags_missing_mesh() -> None:
    splice = _splice_func()
    original = _skeleton3d()

    skeleton, report = splice(
        deepcopy(original),
        body_mesh={"artifact_type": "racketsport_body_mesh", "players": [], "summary": {"mesh_frame_count": 0}},
        body_compute_execution=_body_compute_execution(),
    )

    assert skeleton["players"] == original["players"]
    assert report["summary"] == {
        "scheduled_contact_count": 1,
        "spliced_contact_count": 0,
        "mesh_unavailable_count": 1,
        "fallback_spliced_count": 0,
        "overridden_joint_count": 0,
    }
    assert report["events"][0]["status"] == "mesh_unavailable"
    assert skeleton["provenance"]["contact_splice"]["mesh_unavailable_count"] == 1


def test_contact_splice_uses_explicit_skeleton_fallback_when_mesh_is_unavailable() -> None:
    splice = _splice_func()
    fallback_joints = [[float(idx), 1.0, 1.5] for idx in range(len(BODY65_JOINT_NAMES))]
    fallback_conf = [0.77] * len(fallback_joints)
    fallback = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints_fallback",
        "joint_names": list(BODY65_JOINT_NAMES),
        "preview_only": False,
        "players": [
            {
                "id": 7,
                "frames": [
                    {"frame_idx": 12, "t": 0.4, "joints_world": fallback_joints, "joint_conf": fallback_conf}
                ],
            }
        ],
        "provenance": {"lane": "B_fallback"},
    }

    skeleton, report = splice(
        _skeleton3d(),
        body_mesh={"artifact_type": "racketsport_body_mesh", "players": [], "summary": {"mesh_frame_count": 0}},
        body_compute_execution=_body_compute_execution(),
        fallback_skeleton3d=fallback,
    )

    joint_names = skeleton["joint_names"]
    left_wrist_idx = joint_names.index("left_wrist")
    right_wrist_idx = joint_names.index("right_wrist")
    left_hand_idx = joint_names.index("left_hand_00")
    hitter_frame = skeleton["players"][0]["frames"][0]
    assert hitter_frame["joints_world"][left_wrist_idx] == pytest.approx(fallback_joints[left_wrist_idx])
    assert hitter_frame["joints_world"][right_wrist_idx] == pytest.approx(fallback_joints[right_wrist_idx])
    assert hitter_frame["joints_world"][left_hand_idx] == pytest.approx(fallback_joints[left_hand_idx])
    assert hitter_frame["joint_conf"][left_wrist_idx] == pytest.approx(0.77)
    assert report["events"][0]["status"] == "mesh_unavailable_skeleton_fallback"
    assert report["events"][0]["mesh_unavailable"] is True
    assert report["events"][0]["fallback_source"] == "fallback_skeleton3d.json"
    assert report["summary"]["mesh_unavailable_count"] == 1
    assert report["summary"]["fallback_spliced_count"] == 1
    assert skeleton["provenance"]["contact_splice"]["fallback_spliced_count"] == 1


def test_contact_splice_overrides_generic_sam3d_mhr70_wrist_indices() -> None:
    splice = _splice_func()
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    joints = [[float(idx), 0.0, 1.0] for idx in range(70)]
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": joint_names,
        "preview_only": False,
        "players": [
            {
                "id": 7,
                "frames": [
                    {"frame_idx": 12, "t": 0.4, "joints_world": deepcopy(joints), "joint_conf": [0.5] * 70}
                ],
            }
        ],
        "provenance": {"source": "sam3d_body_joints"},
    }
    mesh_joints = [[float(idx), 1.0, 2.0] for idx in range(70)]
    body_mesh = {
        "artifact_type": "racketsport_body_mesh",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": joint_names,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 12,
                        "t": 0.4,
                        "joints_world": mesh_joints,
                        "joint_conf": [0.91] * 70,
                        "mesh_vertices_world": [[9.0, 0.0, 0.0], [10.0, 0.0, 1.8]],
                    }
                ],
            }
        ],
        "summary": {"mesh_frame_count": 1},
    }

    spliced, report = splice(
        skeleton,
        body_mesh=body_mesh,
        body_compute_execution=_body_compute_execution(),
    )

    left_wrist_idx = SAM3D_BODY_MHR70_SEMANTIC_MAP.joints["left_wrist"]
    right_wrist_idx = SAM3D_BODY_MHR70_SEMANTIC_MAP.joints["right_wrist"]
    frame = spliced["players"][0]["frames"][0]
    assert frame["joints_world"][left_wrist_idx] == pytest.approx(mesh_joints[left_wrist_idx])
    assert frame["joints_world"][right_wrist_idx] == pytest.approx(mesh_joints[right_wrist_idx])
    assert frame["joint_conf"][left_wrist_idx] == pytest.approx(0.91)
    assert report["summary"]["spliced_contact_count"] == 1
    assert report["summary"]["overridden_joint_count"] == 2
    assert report["events"][0]["overridden_joint_names"] == ["left_wrist", "right_wrist"]
