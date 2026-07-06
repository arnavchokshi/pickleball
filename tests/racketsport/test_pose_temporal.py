from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch is optional in the local .venv test lane",
)
if _TORCH_AVAILABLE:
    import torch
else:
    torch = None

from threed.racketsport.joint_schema import BODY65_JOINT_NAMES
from threed.racketsport.pose_temporal import MotionBERTTemporalRuntime, refine_lane_a_skeleton3d


def _frame(frame_idx: int, joints: list[list[float]]) -> dict:
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "joints_world": joints,
        "joint_conf": [0.9 for _name in BODY65_JOINT_NAMES],
    }


def _frame_with_conf(frame_idx: int, joints: list[list[float]], conf_by_name: dict[str, float]) -> dict:
    frame = _frame(frame_idx, joints)
    conf = list(frame["joint_conf"])
    for name, value in conf_by_name.items():
        conf[_idx(name)] = value
    frame["joint_conf"] = conf
    return frame


def _base_joints() -> list[list[float]]:
    return [[0.0, 0.0, 1.0] for _name in BODY65_JOINT_NAMES]


def _skeleton(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": list(BODY65_JOINT_NAMES),
        "preview_only": False,
        "players": [{"id": 7, "frames": frames}],
        "provenance": {"lane": "A"},
    }


def _idx(name: str) -> int:
    return list(BODY65_JOINT_NAMES).index(name)


def test_refine_lane_a_skeleton3d_one_euro_smooths_feet_and_hands() -> None:
    foot_idx = _idx("left_big_toe")
    hand_idx = _idx("left_hand_00")
    frames = []
    for frame_idx, offset in enumerate([0.0, 1.0, 0.0]):
        joints = _base_joints()
        joints[foot_idx] = [offset, 0.0, 0.0]
        joints[hand_idx] = [offset, 0.0, 1.4]
        frames.append(_frame(frame_idx, joints))

    refined = refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0)

    assert refined["preview_only"] is False
    assert refined["provenance"]["temporal_refine"]["one_euro"] == {
        "mincutoff": 1.0,
        "beta": 0.3,
        "core_body_mincutoff": 0.45,
        "core_body_beta": 0.05,
        "wrist_mincutoff": 1000.0,
        "wrist_beta": 0.0,
        "foot_mincutoff": 1.0,
        "foot_beta": 0.3,
        "applied_joint_groups": ["core_body", "feet", "hands", "wrists"],
        "filtered_joint_count": 65,
    }
    assert len(refined["players"][0]["frames"]) == 3
    assert refined["players"][0]["frames"][1]["joints_world"][foot_idx][0] < 1.0
    assert refined["players"][0]["frames"][1]["joints_world"][hand_idx][0] < 1.0
    assert refined["players"][0]["frames"][1]["joint_conf"][hand_idx] == pytest.approx(0.9)


def test_refine_lane_a_skeleton3d_foot_one_euro_override_only_affects_feet() -> None:
    # foot_one_euro_mincutoff/beta default to None, which falls back to the generic
    # mincutoff/beta (identical to pre-fix behavior for every existing caller). Passing
    # explicit near-pass-through values must change ONLY the "feet" group's lag, not
    # "hands" (which still shares the generic bucket) or "core_body"/"wrists". World
    # grounding is disabled here because it re-anchors every joint's XY off the support
    # foot position (by design), which would otherwise cascade a feet-only smoothing
    # change onto every other joint including hands -- an orthogonal, expected effect
    # this test does not want to exercise.
    foot_idx = _idx("left_big_toe")
    hand_idx = _idx("left_hand_00")
    frames = []
    for frame_idx, offset in enumerate([0.0, 1.0, 0.0]):
        joints = _base_joints()
        joints[foot_idx] = [offset, 0.0, 0.0]
        joints[hand_idx] = [offset, 0.0, 1.4]
        frames.append(_frame(frame_idx, joints))

    default_fallback = refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0, apply_world_grounding=False)
    low_lag_feet = refine_lane_a_skeleton3d(
        _skeleton(frames),
        fps=30.0,
        apply_world_grounding=False,
        foot_one_euro_mincutoff=1000.0,
        foot_one_euro_beta=0.0,
    )

    one_euro = low_lag_feet["provenance"]["temporal_refine"]["one_euro"]
    assert one_euro["foot_mincutoff"] == 1000.0
    assert one_euro["foot_beta"] == 0.0
    assert default_fallback["provenance"]["temporal_refine"]["one_euro"]["foot_mincutoff"] == 1.0

    low_lag_foot_x = low_lag_feet["players"][0]["frames"][1]["joints_world"][foot_idx][0]
    default_foot_x = default_fallback["players"][0]["frames"][1]["joints_world"][foot_idx][0]
    assert low_lag_foot_x > default_foot_x

    # Hands are untouched by the foot-specific override -- still the generic bucket.
    low_lag_hand_x = low_lag_feet["players"][0]["frames"][1]["joints_world"][hand_idx][0]
    default_hand_x = default_fallback["players"][0]["frames"][1]["joints_world"][hand_idx][0]
    assert low_lag_hand_x == pytest.approx(default_hand_x)


def test_refine_lane_a_skeleton3d_rejects_non_positive_foot_mincutoff() -> None:
    frames = [_frame(0, _base_joints())]

    with pytest.raises(ValueError, match="foot_one_euro_mincutoff"):
        refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0, foot_one_euro_mincutoff=0.0)

    with pytest.raises(ValueError, match="foot_one_euro_beta"):
        refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0, foot_one_euro_beta=-0.1)


def test_refine_lane_a_skeleton3d_enforces_per_player_body_bone_lengths() -> None:
    shoulder_idx = _idx("left_shoulder")
    elbow_idx = _idx("left_elbow")
    frames = []
    for frame_idx, elbow_x in enumerate([1.0, 1.0, 2.0]):
        joints = _base_joints()
        joints[shoulder_idx] = [0.0, 0.0, 1.5]
        joints[elbow_idx] = [elbow_x, 0.0, 1.5]
        frames.append(_frame(frame_idx, joints))

    refined = refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0)

    last = refined["players"][0]["frames"][2]["joints_world"]
    length = math.dist(last[shoulder_idx], last[elbow_idx])
    assert length == pytest.approx(1.0)
    assert refined["provenance"]["temporal_refine"]["bone_length_constraint"] == "body17_median_per_player"


def test_refine_lane_a_skeleton3d_foot_locks_low_slow_support_foot_by_vertical_velocity() -> None:
    right_heel_idx = _idx("right_heel")
    left_heel_idx = _idx("left_heel")
    nose_idx = _idx("nose")
    frames = []
    for frame_idx, right_x, left_z, nose_x in [
        (0, 1.0, 0.04, 1.0),
        (1, 1.02, 0.20, 1.02),
        (2, 1.03, 0.24, 1.03),
    ]:
        joints = _base_joints()
        joints[right_heel_idx] = [right_x, 0.0, 0.0]
        joints[left_heel_idx] = [0.0, 0.0, left_z]
        joints[nose_idx] = [nose_x, 0.0, 1.6]
        frames.append(_frame_with_conf(frame_idx, joints, {"right_heel": 0.96, "left_heel": 0.90}))

    refined = refine_lane_a_skeleton3d(
        _skeleton(frames),
        fps=30.0,
        one_euro_mincutoff=1000.0,
        one_euro_beta=0.0,
    )

    output_frames = refined["players"][0]["frames"]
    assert output_frames[1]["joints_world"][right_heel_idx][:2] == pytest.approx([1.0, 0.0], abs=0.002)
    assert output_frames[2]["joints_world"][right_heel_idx][:2] == pytest.approx([1.0, 0.0], abs=0.002)
    assert output_frames[1]["joints_world"][nose_idx][0] < 1.02
    grounding = refined["provenance"]["world_grounding"]
    assert grounding["support_foot_strategy"] == "max_conf_lowest_z_lowest_vertical_velocity_5f"
    assert grounding["foot_lock"]["locked_frame_count"] >= 2


def test_refine_lane_a_skeleton3d_holds_airborne_xy_and_reanchors_on_landing() -> None:
    right_heel_idx = _idx("right_heel")
    left_heel_idx = _idx("left_heel")
    nose_idx = _idx("nose")
    frames = []
    for frame_idx, right_x, foot_z, nose_x in [
        (0, 1.0, 0.0, 1.0),
        (1, 1.8, 0.25, 1.8),
        (2, 2.0, 0.0, 2.0),
    ]:
        joints = _base_joints()
        joints[right_heel_idx] = [right_x, 0.0, foot_z]
        joints[left_heel_idx] = [right_x - 0.3, 0.0, foot_z + 0.05]
        joints[nose_idx] = [nose_x, 0.0, 1.6 + foot_z]
        frames.append(_frame_with_conf(frame_idx, joints, {"right_heel": 0.96, "left_heel": 0.90}))

    refined = refine_lane_a_skeleton3d(
        _skeleton(frames),
        fps=30.0,
        one_euro_mincutoff=1000.0,
        one_euro_beta=0.0,
    )

    output_frames = refined["players"][0]["frames"]
    assert output_frames[1]["joints_world"][right_heel_idx][0] == pytest.approx(1.0, abs=0.01)
    assert output_frames[1]["joints_world"][nose_idx][0] < 1.1
    assert output_frames[2]["joints_world"][right_heel_idx][0] == pytest.approx(2.0, abs=0.01)
    assert refined["provenance"]["world_grounding"]["airborne"]["held_frame_count"] == 1
    assert refined["provenance"]["world_grounding"]["airborne"]["reanchored_landing_count"] == 1


def test_refine_lane_a_skeleton3d_applies_motionbert_body17_in_243_frame_windows() -> None:
    nose_idx = _idx("nose")
    foot_idx = _idx("left_big_toe")
    hand_idx = _idx("left_hand_00")
    frames = []
    for frame_idx in range(244):
        joints = _base_joints()
        joints[nose_idx] = [float(frame_idx) * 0.01, 0.0, 1.6]
        joints[foot_idx] = [10.0, 0.0, 0.0]
        joints[hand_idx] = [20.0, 0.0, 1.3]
        frames.append(_frame(frame_idx, joints))

    class FakeMotionBERTRuntime:
        model_id = "motionbert_lift_smooth"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def refine_body17_window(self, *, player_id: int, frames: list[dict], joint_names: list[str]) -> list[list[list[float]]]:
            self.calls.append({"player_id": player_id, "frame_count": len(frames), "joint_names": list(joint_names)})
            refined: list[list[list[float]]] = []
            for frame in frames:
                body17 = [[float(value) for value in joint] for joint in frame["joints_world"][:17]]
                body17[nose_idx] = [body17[nose_idx][0] + 0.25, 0.0, 1.7]
                refined.append(body17)
            return refined

    runtime = FakeMotionBERTRuntime()

    refined = refine_lane_a_skeleton3d(
        _skeleton(frames),
        fps=30.0,
        one_euro_mincutoff=1000.0,
        one_euro_beta=0.0,
        core_one_euro_mincutoff=1000.0,
        core_one_euro_beta=0.0,
        motionbert_runtime=runtime,
    )

    output_frames = refined["players"][0]["frames"]
    assert [call["frame_count"] for call in runtime.calls] == [243, 1]
    assert all(call["player_id"] == 7 for call in runtime.calls)
    assert runtime.calls[0]["joint_names"] == list(BODY65_JOINT_NAMES[:17])
    assert output_frames[0]["joints_world"][nose_idx] == pytest.approx([0.25, 0.0, 1.7])
    assert output_frames[243]["joints_world"][nose_idx] == pytest.approx([2.68, 0.0, 1.7], abs=1e-4)
    assert output_frames[0]["joints_world"][foot_idx][0] == pytest.approx(10.0)
    assert output_frames[0]["joints_world"][hand_idx][0] == pytest.approx(20.0)
    temporal = refined["provenance"]["temporal_refine"]
    assert temporal["motionbert"] == "applied"
    assert temporal["motionbert_model_id"] == "motionbert_lift_smooth"
    assert temporal["motionbert_window_max_frames"] == 243
    assert temporal["motionbert_window_count"] == 2
    assert temporal["motionbert_frame_count"] == 244


def test_motionbert_runtime_fails_closed_when_manifest_checkpoint_is_not_available(tmp_path: Path) -> None:
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "motionbert_lift_smooth",
                        "stage": "lane_a_temporal_refine",
                        "use": "Temporal lift and smoothing candidate for Lane A body-17 windows",
                        "source": "https://github.com/Walter0807/MotionBERT",
                        "license": "Apache-2.0",
                        "commercial_posture": "ok",
                        "status": "pending_download",
                        "local_path": "/workspace/checkpoints/body4d/motionbert/best_epoch.bin",
                        "sha256": None,
                        "fallbacks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime = MotionBERTTemporalRuntime(manifest_path=manifest_path, config_path=tmp_path / "config.yaml")

    with pytest.raises(RuntimeError, match="model motionbert_lift_smooth is not available_on_h100: status=pending_download"):
        runtime.refine_body17_window(player_id=7, frames=[_frame(0, _base_joints())], joint_names=list(BODY65_JOINT_NAMES[:17]))


def test_motionbert_runtime_maps_body17_to_h36m_and_merges_supported_joints(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_epoch.bin"
    checkpoint_bytes = b"fake motionbert checkpoint"
    checkpoint_path.write_bytes(checkpoint_bytes)
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "motionbert_lift_smooth",
                        "stage": "lane_a_temporal_refine",
                        "use": "Temporal lift and smoothing candidate for Lane A body-17 windows",
                        "source": "https://github.com/Walter0807/MotionBERT",
                        "license": "Apache-2.0",
                        "commercial_posture": "ok",
                        "status": "available_on_h100",
                        "local_path": str(checkpoint_path),
                        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                        "fallbacks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    joints = _base_joints()
    for name, xyz in {
        "nose": [1.0, 3.0, 1.7],
        "left_eye": [0.8, 3.1, 1.75],
        "right_eye": [1.2, 3.1, 1.75],
        "left_shoulder": [0.0, 2.0, 1.4],
        "right_shoulder": [2.0, 2.0, 1.4],
        "left_elbow": [-0.4, 2.0, 1.1],
        "right_elbow": [2.4, 2.0, 1.1],
        "left_wrist": [-0.7, 2.0, 0.9],
        "right_wrist": [2.7, 2.0, 0.9],
        "left_hip": [0.0, 0.0, 1.0],
        "right_hip": [2.0, 0.0, 1.0],
        "left_knee": [0.0, -0.8, 0.5],
        "right_knee": [2.0, -0.8, 0.5],
        "left_ankle": [0.0, -1.4, 0.0],
        "right_ankle": [2.0, -1.4, 0.0],
    }.items():
        joints[_idx(name)] = xyz
    frame = _frame_with_conf(0, joints, {"left_wrist": 0.55, "right_wrist": 0.66})

    class FakeModel:
        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            self.seen = inputs.detach().cpu()
            output = torch.zeros((1, 1, 17, 3), dtype=torch.float32, device=inputs.device)
            output[:, :, 9, :] = torch.tensor([0.25, 0.0, 0.10], device=inputs.device)
            output[:, :, 11, :] = torch.tensor([-0.50, 0.25, 0.30], device=inputs.device)
            output[:, :, 16, :] = torch.tensor([0.75, -0.25, -0.20], device=inputs.device)
            return output

    fake_model = FakeModel()
    runtime = MotionBERTTemporalRuntime(manifest_path=manifest_path, config_path=tmp_path / "config.yaml", device="cpu")
    runtime._load_model = lambda: {  # type: ignore[method-assign]
        "model": fake_model,
        "args": SimpleNamespace(no_conf=False, rootrel=False, flip=False),
        "torch": torch,
        "flip_data": None,
        "device": "cpu",
    }

    refined = runtime.refine_body17_window(
        player_id=7,
        frames=[frame],
        joint_names=list(BODY65_JOINT_NAMES[:17]),
    )

    assert fake_model.seen is not None
    assert fake_model.seen.shape == (1, 1, 17, 3)
    h36m = fake_model.seen[0, 0]
    assert h36m[0, :2].tolist() == pytest.approx([0.0, 0.0])
    assert h36m[1, :2].tolist() == pytest.approx([0.5, 0.0])
    assert h36m[4, :2].tolist() == pytest.approx([-0.5, 0.0])
    assert h36m[13, 2].item() == pytest.approx(0.55)
    assert h36m[16, 2].item() == pytest.approx(0.66)
    assert refined[0][_idx("nose")] == pytest.approx([1.5, 0.0, 1.2])
    assert refined[0][_idx("left_shoulder")] == pytest.approx([0.0, 0.5, 1.6])
    assert refined[0][_idx("right_wrist")] == pytest.approx([2.5, -0.5, 0.6])
    assert refined[0][_idx("left_eye")] == pytest.approx(joints[_idx("left_eye")])
