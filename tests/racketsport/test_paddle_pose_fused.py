from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.paddle_pose_fused import (
    BAND_CONTACT_LOCKED,
    BAND_GRIP_EXTRAPOLATED,
    BAND_PALM_FITTED,
    RAW_JOINT_COUNT,
    _angle_deg,
    _detect_segments,
    _fit_grip_transform,
    _HandFrame,
    _orthonormal_frame_from_y_z,
    _RotationOneEuro,
    _smooth_poses,
    _solve_wahba,
    build_paddle_pose_fused_from_skeleton,
)
from scipy.spatial.transform import Rotation


_JOINT_INDEX = {name: i for i, name in enumerate(MHR70_JOINT_NAMES)}
assert RAW_JOINT_COUNT == 70


def _joint_array(overrides: dict[str, tuple[float, float, float]]) -> list[list[float]]:
    joints = [[0.0, 0.0, 0.0] for _ in range(RAW_JOINT_COUNT)]
    for name, point in overrides.items():
        joints[_JOINT_INDEX[name]] = list(point)
    return joints


def _frame(
    t: float,
    *,
    side: str = "right",
    wrist=(0.0, 0.0, 1.0),
    elbow=(0.0, -0.3, 1.0),
    middle=(0.0, 0.1, 1.0),
    index_pt=(0.05, 0.05, 1.0),
    pinky=(-0.05, 0.05, 1.0),
    conf: float = 0.9,
    include_fingers: bool = True,
    frame_idx: int | None = None,
    other_side_confidence: float = 0.0,
) -> dict:
    overrides = {f"{side}_wrist": wrist, f"{side}_elbow": elbow}
    if include_fingers:
        overrides[f"{side}_middle_third_joint"] = middle
        overrides[f"{side}_index_third_joint"] = index_pt
        overrides[f"{side}_pinky_third_joint"] = pinky
    # Park the opposite hand far away AND at low confidence (default 0.0) so auto-side-selection
    # is unambiguous unless the caller explicitly wants both hands populated with real signal.
    other = "left" if side == "right" else "right"
    overrides.setdefault(f"{other}_wrist", (5.0, 5.0, 5.0))
    overrides.setdefault(f"{other}_elbow", (5.0, 5.0, 4.9))
    joints = _joint_array(overrides)
    joint_conf = [conf] * RAW_JOINT_COUNT
    for name in (f"{other}_wrist", f"{other}_elbow"):
        joint_conf[_JOINT_INDEX[name]] = other_side_confidence
    return {
        "t": t,
        "frame_idx": frame_idx if frame_idx is not None else int(round(t * 30.0)),
        "joints_world": joints,
        "joint_conf": joint_conf,
    }


def _skeleton(frames: list[dict], *, player_id: int = 1, fps: float = 30.0) -> dict:
    return {"fps": fps, "players": [{"id": player_id, "frames": frames}]}


def _rotation_error(rotation: list[list[float]]) -> float:
    r = np.array(rotation)
    return float(np.max(np.abs(r.T @ r - np.eye(3))))


# --------------------------------------------------------------------------------------
# Hand-frame construction, including handedness
# --------------------------------------------------------------------------------------


def test_orthonormal_frame_from_y_z_is_proper_rotation_for_arbitrary_inputs() -> None:
    y_axis = np.array([0.0, 1.0, 0.0])
    z_candidate = np.array([0.3, 0.0, 0.95])
    z_candidate = z_candidate / np.linalg.norm(z_candidate)
    rotation = _orthonormal_frame_from_y_z(y_axis, z_candidate)
    assert rotation.shape == (3, 3)
    assert np.max(np.abs(rotation.T @ rotation - np.eye(3))) < 1e-9
    assert pytest.approx(np.linalg.det(rotation), abs=1e-9) == 1.0
    # Y column matches the requested grip axis exactly (by construction it is preserved).
    assert rotation[:, 1] == pytest.approx(y_axis)


def test_build_paddle_pose_fused_selects_correct_hand_side_via_auto_scoring() -> None:
    frames = []
    for i in range(15):
        t = i / 30.0
        frames.append(
            _frame(
                t,
                side="left",
                wrist=(-1.0 - 0.01 * i, 0.0, 1.0),
                elbow=(-1.0, -0.3, 1.0),
                middle=(-1.0 - 0.01 * i, 0.1, 1.0),
                index_pt=(-1.0 - 0.01 * i + 0.05, 0.05, 1.0),
                pinky=(-1.0 - 0.01 * i - 0.05, 0.05, 1.0),
            )
        )
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic", dominant_hand="auto")
    player = out["players"][0]
    assert player["dominant_hand"] == "left"
    assert player["hand_selection"]["side_scores"]["left"]["usable_frame_count"] == 15
    assert player["hand_selection"]["side_scores"]["right"]["usable_frame_count"] == 0


def test_build_paddle_pose_fused_respects_explicit_dominant_hand_and_player_override() -> None:
    frames = [_frame(i / 30.0, side="right") for i in range(5)]
    # Right-hand data only, but force "left": with no left data at all this must fail closed for
    # that player (no usable wrist frames), proving the override is honored rather than silently
    # falling back to the side with real data.
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(
        skeleton, clip_id="synthetic", dominant_hand_by_player={1: "left"}
    )
    player = out["players"][0]
    assert player["dominant_hand"] == "left"
    assert player["frames"] == []
    assert "no_usable_wrist_frames_for_player" in out["blockers"]


def test_hand_frame_falls_back_to_forearm_axis_when_fingers_degenerate() -> None:
    frames = [_frame(i / 30.0, include_fingers=False) for i in range(5)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic")
    player = out["players"][0]
    assert len(player["frames"]) == 5
    # Fingers missing every frame -> every frame is grip_extrapolated (fallback path), never palm_fitted.
    assert player["band_distribution"][BAND_PALM_FITTED] == 0
    assert player["band_distribution"][BAND_GRIP_EXTRAPOLATED] == 5


# --------------------------------------------------------------------------------------
# Fail-closed: no wrist -> no frame
# --------------------------------------------------------------------------------------


def test_missing_wrist_frame_is_hidden_not_emitted() -> None:
    good = _frame(0.0)
    # A frame with too few raw joints (simulating a non-70-joint payload) must be hidden.
    bad = {"t": 1.0 / 30.0, "frame_idx": 1, "joints_world": [[0.0, 0.0, 0.0]] * 5, "joint_conf": [0.9] * 5}
    skeleton = _skeleton([good, bad])
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic")
    player = out["players"][0]
    assert len(player["frames"]) == 1
    assert player["hidden_frame_count"] == 1
    hidden = out["hidden_frames"][0]
    assert hidden["reason"] == "insufficient_raw_joint_count"


def test_low_confidence_wrist_hides_frame() -> None:
    frames = [_frame(0.0, conf=0.9), _frame(1.0 / 30.0, conf=0.05, frame_idx=1)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic", min_joint_confidence=0.25)
    player = out["players"][0]
    assert len(player["frames"]) == 1
    assert player["hidden_frame_count"] == 1
    assert out["hidden_frames"][0]["reason"] == "missing_or_low_confidence_wrist"


def test_no_players_have_any_wrist_data_yields_blocked_status() -> None:
    frames = [{"t": 0.0, "frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]] * 3, "joint_conf": [0.9] * 3}]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic")
    assert out["status"] == "blocked"
    assert out["summary"]["estimate_frame_count"] == 0


def test_membership_excluded_player_gets_no_paddle_frames() -> None:
    frames = [_frame(i / 30.0) for i in range(5)]
    skeleton = _skeleton(frames, player_id=7)
    membership = {"players": {"7": {"member": False}}}
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic", membership=membership)
    player = out["players"][0]
    assert player["frames"] == []
    assert player.get("membership_excluded") is True


# --------------------------------------------------------------------------------------
# Grip-segment fitting on synthetic data with a KNOWN G
# --------------------------------------------------------------------------------------


def test_solve_wahba_recovers_known_rotation_from_exact_vector_pairs() -> None:
    known_rotation = Rotation.from_euler("xyz", [12.0, -30.0, 55.0], degrees=True).as_matrix()
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(6):
        v_local = rng.normal(size=3)
        v_local = v_local / np.linalg.norm(v_local)
        v_body = known_rotation @ v_local
        pairs.append((v_body, v_local, 1.0))
    recovered = _solve_wahba(pairs)
    assert np.max(np.abs(recovered - known_rotation)) < 1e-9
    assert pytest.approx(np.linalg.det(recovered), abs=1e-9) == 1.0


def test_fit_grip_transform_with_only_prior_recovers_identity() -> None:
    samples = [
        _HandFrame(
            t=i / 30.0,
            frame_idx=i,
            wrist=np.array([0.0, 0.0, 1.0]),
            rotation=np.eye(3),
            z_candidate_raw=np.array([0.0, 0.0, 1.0]),
            joint_confidence=0.9,
            used_finger_grip_axis=True,
            used_finger_palm_normal=True,
        )
        for i in range(10)
    ]
    r_g, t_g, fit_info = _fit_grip_transform(
        samples,
        prior_rotation=np.eye(3),
        prior_translation=np.array([0.0, 0.3, 0.0]),
        prior_rotation_weight=0.4,
        prior_translation_weight=6.0,
        reflection_pairs=[],
        use_reflection=True,
    )
    assert np.max(np.abs(r_g - np.eye(3))) < 1e-9
    assert t_g == pytest.approx([0.0, 0.3, 0.0], abs=1e-9)
    assert fit_info["reflection_observation_count"] == 0


def test_fit_grip_transform_reflection_pulls_rotation_toward_known_target() -> None:
    # Known target G rotates paddle Z (0,0,1) to hand-local (0,-1,0): a 90deg tilt.
    known_r_g = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    samples = [
        _HandFrame(
            t=0.0,
            frame_idx=0,
            wrist=np.array([0.0, 0.0, 1.0]),
            rotation=np.eye(3),
            z_candidate_raw=np.array([0.0, 0.0, 1.0]),
            joint_confidence=0.9,
            used_finger_grip_axis=True,
            used_finger_palm_normal=True,
        )
    ]
    # A reflection observation says: in hand-local coordinates, the paddle's local Z axis (0,0,1)
    # should be observed at known_r_g @ (0,0,1) -- i.e. the exact vector the known G would produce.
    v_body = known_r_g @ np.array([0.0, 0.0, 1.0])
    reflection_pairs = [(v_body, np.array([0.0, 0.0, 1.0]), 1000.0)]  # huge weight dominates the prior
    r_g, _t_g, _fit_info = _fit_grip_transform(
        samples,
        prior_rotation=np.eye(3),
        prior_translation=np.array([0.0, 0.3, 0.0]),
        prior_rotation_weight=0.4,
        prior_translation_weight=6.0,
        reflection_pairs=reflection_pairs,
        use_reflection=True,
    )
    # With overwhelming reflection weight, R_g @ (0,0,1) should match the known target closely.
    assert r_g @ np.array([0.0, 0.0, 1.0]) == pytest.approx(v_body, abs=1e-3)


def test_translation_jitter_minimization_prefers_stable_axis_direction() -> None:
    # H(t) rotation jitters around the X axis only; the "quiet" direction is the X axis itself.
    rng = np.random.default_rng(3)
    samples = []
    for i in range(40):
        wobble_deg = 20.0 * np.sin(i * 0.7)
        rotation = Rotation.from_euler("x", wobble_deg, degrees=True).as_matrix()
        samples.append(
            _HandFrame(
                t=i / 30.0,
                frame_idx=i,
                wrist=np.array([0.0, 0.0, 1.0]),
                rotation=rotation,
                z_candidate_raw=rotation[:, 2],
                joint_confidence=0.9,
                used_finger_grip_axis=True,
                used_finger_palm_normal=True,
            )
        )
    prior = np.array([0.0, 0.0, 0.3])  # prior offset along Z, orthogonal to the quiet X axis
    _r_g, t_g, _fit_info = _fit_grip_transform(
        samples,
        prior_rotation=np.eye(3),
        prior_translation=prior,
        prior_rotation_weight=0.4,
        prior_translation_weight=0.5,  # weak prior so the jitter term can pull t_g off-prior
        reflection_pairs=[],
        use_reflection=True,
    )
    # The fit should deviate from a prior that sits entirely off the quiet (X) axis, since the
    # jitter penalty is zero along X and nonzero along Y/Z components of a rotating-about-X frame.
    assert not np.allclose(t_g, prior, atol=1e-6)


# --------------------------------------------------------------------------------------
# Reflection cone factor on synthetic contacts (end-to-end through the public build API)
# --------------------------------------------------------------------------------------


def test_reflection_channel_rotates_grip_and_marks_contact_locked_band() -> None:
    frames = [_frame(i / 30.0) for i in range(30)]
    skeleton = _skeleton(frames)
    physics_estimate = {
        "estimates": [
            {
                "t": 0.5,
                "selected_wrist": {"player_id": 1, "side": "right"},
                "face_normal_world": [0.0, -1.0, 0.0],
                "uncertainty": {"normal_angle_bound_deg": 5.0},
            }
        ]
    }
    out_off = build_paddle_pose_fused_from_skeleton(
        skeleton, clip_id="synthetic", physics_estimate=physics_estimate, use_reflection=False, reflection_weight_scale=50.0
    )
    out_on = build_paddle_pose_fused_from_skeleton(
        skeleton, clip_id="synthetic", physics_estimate=physics_estimate, use_reflection=True, reflection_weight_scale=50.0
    )
    r_g_off = np.array(out_off["players"][0]["segments"][0]["grip_rotation"])
    r_g_on = np.array(out_on["players"][0]["segments"][0]["grip_rotation"])
    assert np.max(np.abs(r_g_off - np.eye(3))) < 1e-9
    assert np.max(np.abs(r_g_on - np.eye(3))) > 0.1

    frames_out = out_on["players"][0]["frames"]
    near_contact = min(frames_out, key=lambda f: abs(f["t"] - 0.5))
    assert near_contact["trust_band"]["note"] == BAND_CONTACT_LOCKED
    far_frame = frames_out[0]
    assert far_frame["trust_band"]["note"] != BAND_CONTACT_LOCKED
    assert out_on["summary"]["evidence_channels"]["reflection_contacts_available"] is True
    assert "reflection_channel_dormant_no_usable_ball_contacts" not in out_on["warnings"]


def test_reflection_channel_dormant_when_no_contacts_available() -> None:
    frames = [_frame(i / 30.0) for i in range(5)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic")
    assert out["summary"]["evidence_channels"]["reflection_contacts_available"] is False
    assert "reflection_channel_dormant_no_usable_ball_contacts" in out["warnings"]
    assert out["summary"]["band_distribution"][BAND_CONTACT_LOCKED] == 0


# --------------------------------------------------------------------------------------
# Segment detection (residual break + hysteresis)
# --------------------------------------------------------------------------------------


def test_segment_break_requires_sustained_deviation_not_a_brief_blip() -> None:
    stable = np.array([0.0, 0.0, 1.0])
    rotated = np.array([0.0, 1.0, 0.0])  # 90 degrees away

    def make(t, z):
        return _HandFrame(
            t=t,
            frame_idx=int(round(t * 30)),
            wrist=np.array([0.0, 0.0, 1.0]),
            rotation=np.eye(3),
            z_candidate_raw=z,
            joint_confidence=0.9,
            used_finger_grip_axis=True,
            used_finger_palm_normal=True,
        )

    samples = []
    t = 0.0
    for _ in range(60):
        samples.append(make(t, stable)); t += 1.0 / 30.0
    for _ in range(9):  # 0.3s blip: must NOT cut a segment
        samples.append(make(t, rotated)); t += 1.0 / 30.0
    for _ in range(30):
        samples.append(make(t, stable)); t += 1.0 / 30.0

    segments = _detect_segments(samples, min_duration_s=2.0, break_angle_deg=75.0)
    assert len(segments) == 1
    assert segments[0] == (0, len(samples) - 1)


def test_segment_break_cuts_after_sustained_hold() -> None:
    stable = np.array([0.0, 0.0, 1.0])
    rotated = np.array([0.0, 1.0, 0.0])

    def make(t, z):
        return _HandFrame(
            t=t,
            frame_idx=int(round(t * 30)),
            wrist=np.array([0.0, 0.0, 1.0]),
            rotation=np.eye(3),
            z_candidate_raw=z,
            joint_confidence=0.9,
            used_finger_grip_axis=True,
            used_finger_palm_normal=True,
        )

    samples = []
    t = 0.0
    for _ in range(60):
        samples.append(make(t, stable)); t += 1.0 / 30.0
    for _ in range(70):  # >= 2s sustained hold at the new orientation
        samples.append(make(t, rotated)); t += 1.0 / 30.0

    segments = _detect_segments(samples, min_duration_s=2.0, break_angle_deg=75.0)
    assert len(segments) == 2
    assert segments[0][0] == 0
    assert segments[1][1] == len(samples) - 1


# --------------------------------------------------------------------------------------
# Band assignment
# --------------------------------------------------------------------------------------


def test_band_assignment_mixes_palm_fitted_and_grip_extrapolated_per_frame() -> None:
    frames = []
    for i in range(6):
        t = i / 30.0
        include_fingers = i % 2 == 0
        frames.append(_frame(t, include_fingers=include_fingers))
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic")
    bands = [f["trust_band"]["note"] for f in out["players"][0]["frames"]]
    assert bands == [
        BAND_PALM_FITTED,
        BAND_GRIP_EXTRAPOLATED,
        BAND_PALM_FITTED,
        BAND_GRIP_EXTRAPOLATED,
        BAND_PALM_FITTED,
        BAND_GRIP_EXTRAPOLATED,
    ]
    assert out["players"][0]["band_distribution"] == {
        BAND_CONTACT_LOCKED: 0,
        BAND_PALM_FITTED: 3,
        BAND_GRIP_EXTRAPOLATED: 3,
    }


# --------------------------------------------------------------------------------------
# SO(3) smoothing behavior
# --------------------------------------------------------------------------------------


def test_rotation_one_euro_reduces_jitter_versus_raw_noisy_sequence() -> None:
    rng = np.random.default_rng(11)
    rotations_raw = []
    t = 0.0
    for i in range(120):
        base_deg = 2.0 * i  # a slow, smooth underlying trend
        noise_deg = rng.normal(scale=8.0)  # per-frame noise dominating a slow trend
        rotations_raw.append((t, Rotation.from_euler("z", base_deg + noise_deg, degrees=True)))
        t += 1.0 / 30.0

    filt = _RotationOneEuro(min_cutoff=1.0, beta=0.1, d_cutoff=1.0)
    smoothed = []
    prev_t = None
    for sample_t, rotation in rotations_raw:
        dt = (sample_t - prev_t) if prev_t is not None else 0.0
        prev_t = sample_t
        smoothed.append(filt.apply(rotation, dt))

    def jitter(sequence):
        angles = []
        for prev, cur in zip(sequence, sequence[1:]):
            relative = prev.inv() * cur
            angles.append(np.degrees(relative.magnitude()))
        return float(np.median(angles))

    raw_jitter = jitter([r for _t, r in rotations_raw])
    smoothed_jitter = jitter(smoothed)
    assert smoothed_jitter < raw_jitter


def test_smooth_poses_output_is_always_orthonormal_proper_rotation() -> None:
    rng = np.random.default_rng(5)
    poses = []
    t = 0.0
    for _ in range(20):
        euler = rng.normal(scale=40.0, size=3)
        rotation = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        translation = rng.normal(size=3)
        poses.append((t, rotation, translation))
        t += 1.0 / 30.0
    smoothed = _smooth_poses(poses, min_cutoff=1.2, beta=0.35, d_cutoff=1.0)
    for rotation, _translation in smoothed:
        assert np.max(np.abs(rotation.T @ rotation - np.eye(3))) < 1e-6
        assert pytest.approx(np.linalg.det(rotation), abs=1e-6) == 1.0


# --------------------------------------------------------------------------------------
# Artifact contract fields (matches paddle_proxy.py's racket_pose_estimate.json contract)
# --------------------------------------------------------------------------------------


def test_artifact_contract_top_level_fields() -> None:
    frames = [_frame(i / 30.0) for i in range(5)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic_clip")
    assert out["schema_version"] == 1
    assert out["artifact_type"] == "racketsport_racket_pose_estimate"
    assert out["clip_id"] == "synthetic_clip"
    assert out["source"] == "wrist_palm_grip_fused"
    assert out["render_only"] is True
    assert out["not_for_detection_metrics"] is True
    assert out["trusted_for_rkt_promotion"] is False
    assert out["never_canonical_racket_pose"] is True
    assert out["canonical_output_forbidden"] == "racket_pose.json"
    assert out["rkt_gate_unscoreable"] is True
    assert out["trust"] == "estimated_from_wrist"
    assert out["world_frame"] == "court_Z0"
    assert out["coordinate_frame"] == "court_netcenter_z_up_m"
    assert out["coordinate_space"] == "world_court_netcenter_z_up_m"
    assert out["input_coordinate_space"] == "world_court_netcenter_z_up_m"
    assert out["parameters"]["coordinate_contract"] == {
        "world_input": "world_court_netcenter_z_up_m",
        "pinhole_output": "pixels_undistorted_native",
        "detector_reference": "pixels_raw_native",
        "transform_applied": False,
        "parity_note": "declaration_only_no_distortion_resize_crop_or_reference_transform",
    }
    assert out["translation_unit"] == "m"
    assert "players" in out and isinstance(out["players"], list)
    assert "hidden_frames" in out
    assert "summary" in out and "band_distribution" in out["summary"]


def test_artifact_contract_per_frame_fields_match_virtual_world_consumption_contract() -> None:
    frames = [_frame(i / 30.0) for i in range(3)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic_clip")
    frame = out["players"][0]["frames"][0]
    # Required by virtual_world._paddle_frame:
    assert isinstance(frame["t"], float)
    assert frame["world_frame"] == "court_Z0"
    assert frame["coordinate_frame"] == "court_netcenter_z_up_m"
    assert frame["coordinate_space"] == "world_court_netcenter_z_up_m"
    assert frame["translation_unit"] == "m"
    assert isinstance(frame["conf"], float)
    pose = frame["pose_se3"]
    assert len(pose["R"]) == 3 and all(len(row) == 3 for row in pose["R"])
    assert len(pose["t"]) == 3
    assert _rotation_error(pose["R"]) < 1e-6
    assert pytest.approx(np.linalg.det(np.array(pose["R"])), abs=1e-6) == 1.0
    # trust_band / trust resolve to the recognized "estimated_from_wrist" status so
    # virtual_world._trust_status() picks the correct (low_confidence, estimated) branch.
    assert frame["trust"] == "estimated_from_wrist"
    assert frame["trust_band"]["status"] == "estimated_from_wrist"
    # Band info rides trust_band.note / confidence_provenance.predictor as the spec requires.
    assert frame["trust_band"]["note"] in {BAND_CONTACT_LOCKED, BAND_PALM_FITTED, BAND_GRIP_EXTRAPOLATED}
    cp = frame["confidence_provenance"]
    assert set(cp) == {"band", "display_band", "predictor", "horizon_frames", "predicted_sigma_m"}
    assert cp["band"] == "estimated_from_wrist"
    assert cp["predictor"] == frame["trust_band"]["note"]
    assert frame["render_only"] is True
    assert frame["not_for_detection_metrics"] is True
    assert frame["ambiguous"] is False
    assert frame["source"] == "wrist_palm_grip_fused"


def test_confidence_provenance_never_has_extra_keys_forbidden_by_schema() -> None:
    # ConfidenceProvenance is model_config=extra("forbid") and copied VERBATIM by virtual_world,
    # so this artifact must never add keys beyond band/display_band/predictor/horizon_frames/
    # predicted_sigma_m.
    frames = [_frame(i / 30.0) for i in range(2)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic_clip")
    for player in out["players"]:
        for frame in player["frames"]:
            cp = frame["confidence_provenance"]
            assert set(cp) <= {"band", "display_band", "predictor", "horizon_frames", "predicted_sigma_m"}
            assert isinstance(cp["horizon_frames"], int)


def test_invalid_parameters_raise_value_error() -> None:
    frames = [_frame(0.0)]
    skeleton = _skeleton(frames)
    with pytest.raises(ValueError):
        build_paddle_pose_fused_from_skeleton(skeleton, clip_id="x", dominant_hand="up")
    with pytest.raises(ValueError):
        build_paddle_pose_fused_from_skeleton(skeleton, clip_id="x", grip_offset_m=-1.0)
    with pytest.raises(ValueError):
        build_paddle_pose_fused_from_skeleton(skeleton, clip_id="x", min_joint_confidence=2.0)
    with pytest.raises(ValueError):
        build_paddle_pose_fused_from_skeleton(skeleton, clip_id="x", min_segment_duration_s=0.0)


def test_coverage_matches_input_frame_count_when_all_wrists_present() -> None:
    frames = [_frame(i / 30.0) for i in range(37)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic_clip")
    assert out["summary"]["estimate_frame_count"] == 37
    assert out["summary"]["hidden_frame_count"] == 0


# --------------------------------------------------------------------------------------
# CLI direct-reference test (also satisfies the scaffold-tool-index coverage requirement:
# scripts/racketsport/list_scaffold_tools.py's direct_cli_reference_test check looks for the
# literal command path string inside a tests/racketsport/test_*.py file).
# --------------------------------------------------------------------------------------


def test_cli_builds_fused_pose_estimate_from_skeleton3d(tmp_path: Path) -> None:
    frames = [_frame(i / 30.0) for i in range(8)]
    skeleton = _skeleton(frames)
    skeleton_path = tmp_path / "skeleton3d.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")
    out_path = tmp_path / "racket_pose_estimate.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_paddle_pose_fused.py",
            "--skeleton",
            str(skeleton_path),
            "--out",
            str(out_path),
            "--clip",
            "synthetic_cli_clip",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_racket_pose_estimate"
    assert payload["source"] == "wrist_palm_grip_fused"
    assert payload["clip_id"] == "synthetic_cli_clip"
    assert payload["summary"]["estimate_frame_count"] == 8
    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["out"] == str(out_path)


def test_cli_rejects_missing_skeleton_file(tmp_path: Path) -> None:
    out_path = tmp_path / "racket_pose_estimate.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_paddle_pose_fused.py",
            "--skeleton",
            str(tmp_path / "does_not_exist.json"),
            "--out",
            str(out_path),
            "--clip",
            "synthetic_cli_clip",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "ERROR" in completed.stderr
    assert not out_path.exists()


# --------------------------------------------------------------------------------------
# Optional detector-box factor (functional but experimental; lightly tested)
# --------------------------------------------------------------------------------------


def _synthetic_calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 300.0, "source": "synthetic"},
        "extrinsics": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_height_m": 3.0},
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
        "capture_quality": {"grade": "good"},
        "image_pts": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_pts": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    }


def test_detector_box_channel_applies_bounded_correction_when_enabled() -> None:
    frames = [_frame(i / 30.0, wrist=(0.0, 0.0, 5.0), elbow=(0.0, -0.3, 5.0), middle=(0.0, 0.1, 5.0), index_pt=(0.05, 0.05, 5.0), pinky=(-0.05, 0.05, 5.0)) for i in range(6)]
    skeleton = _skeleton(frames)
    calibration = _synthetic_calibration()
    detector_boxes = {"records": [{"frame": i, "bbox_xyxy": [520, 280, 560, 320], "conf": 0.8} for i in range(6)]}

    out_disabled = build_paddle_pose_fused_from_skeleton(
        skeleton, clip_id="synthetic", detector_boxes=detector_boxes, calibration=calibration, use_detector_boxes=False
    )
    out_enabled = build_paddle_pose_fused_from_skeleton(
        skeleton,
        clip_id="synthetic",
        detector_boxes=detector_boxes,
        calibration=calibration,
        use_detector_boxes=True,
        detector_box_wrist_gate_radius_px=200.0,
        detector_box_max_correction_m=0.05,
    )
    assert out_disabled["players"][0]["segments"][0]["detector_box"]["used"] is False
    box_info = out_enabled["players"][0]["segments"][0]["detector_box"]
    assert box_info["used"] is True
    assert box_info["matched_box_count"] == 6
    assert box_info["correction_norm_m"] <= 0.05 + 1e-9
    assert out_enabled["summary"]["evidence_channels"]["detector_boxes_enabled"] is True


def test_detector_box_channel_reports_reason_when_calibration_invalid() -> None:
    frames = [_frame(i / 30.0) for i in range(3)]
    skeleton = _skeleton(frames)
    out = build_paddle_pose_fused_from_skeleton(
        skeleton,
        clip_id="synthetic",
        detector_boxes={"records": [{"frame": 0, "bbox_xyxy": [1, 2, 3, 4], "conf": 0.5}]},
        calibration={"not": "a valid calibration"},
        use_detector_boxes=True,
    )
    box_info = out["players"][0]["segments"][0]["detector_box"]
    assert box_info["used"] is False
    assert "invalid_calibration" in box_info["reason"]


# --------------------------------------------------------------------------------------
# Hand-switch intervals (detector-box votes; >=2s hysteresis) and reprojection roll solve
# --------------------------------------------------------------------------------------


def test_hand_intervals_switch_on_sustained_majority_and_resist_blips() -> None:
    from threed.racketsport.paddle_pose_fused import _hand_intervals_from_box_votes

    votes = []
    t = 0.0
    for _ in range(60):  # 2s right votes
        votes.append((t, "right")); t += 1.0 / 30.0
    for _ in range(9):  # 0.3s left blip: must NOT switch
        votes.append((t, "left")); t += 1.0 / 30.0
    for _ in range(30):
        votes.append((t, "right")); t += 1.0 / 30.0
    switch_t = t
    for _ in range(90):  # 3s sustained left: must switch
        votes.append((t, "left")); t += 1.0 / 30.0

    intervals, evidence = _hand_intervals_from_box_votes(votes, initial_side="right", min_hold_s=2.0)
    sides = [side for _a, _b, side in intervals]
    assert sides == ["right", "left"]
    assert intervals[0][1] == pytest.approx(switch_t, abs=0.5)
    assert evidence["interval_count"] == 2
    assert len(evidence["switches"]) == 1
    assert evidence["switches"][0]["reason"] == "sustained_majority"


def test_hand_intervals_opening_window_majority_overrides_wrong_initial_side() -> None:
    from threed.racketsport.paddle_pose_fused import _hand_intervals_from_box_votes

    votes = [(i / 30.0, "left") for i in range(90)]
    intervals, evidence = _hand_intervals_from_box_votes(votes, initial_side="right", min_hold_s=2.0)
    assert [side for _a, _b, side in intervals] == ["left"]
    assert evidence["switches"][0]["reason"] == "opening_window_majority"


def test_grip_roll_solve_recovers_synthetic_roll_against_boxes() -> None:
    from threed.racketsport.court_calibration import project_world_points
    from threed.racketsport.paddle_pose_fused import (
        _gated_boxes_by_frame,
        _paddle_footprint_local,
        _project_paddle_bbox,
        _solve_grip_roll_against_boxes,
    )
    from threed.racketsport.schemas import CourtCalibration

    calibration = CourtCalibration.model_validate(_synthetic_calibration())
    dims = {"length": 16.0, "width": 8.0}
    true_roll_deg = 45.0
    true_r_g = Rotation.from_euler("y", true_roll_deg, degrees=True).as_matrix()
    t_g = np.array([0.0, 0.3, 0.0])
    footprint = _paddle_footprint_local(dims)

    samples = []
    detector_records = []
    for i in range(12):
        wrist = np.array([0.0, 0.0, 5.0])
        hand_rotation = Rotation.from_euler("z", 5.0 * i, degrees=True).as_matrix()
        sample = _HandFrame(
            t=i / 30.0,
            frame_idx=i,
            wrist=wrist,
            rotation=hand_rotation,
            z_candidate_raw=hand_rotation[:, 2],
            joint_confidence=0.9,
            used_finger_grip_axis=True,
            used_finger_palm_normal=True,
        )
        samples.append(sample)
        # Detector box = exact bbox of the TRUE-roll paddle footprint.
        true_bbox = _project_paddle_bbox(
            hand_rotation @ true_r_g, wrist + hand_rotation @ t_g, footprint, calibration, project_world_points
        )
        assert true_bbox is not None
        detector_records.append((i, tuple(true_bbox), 0.9))

    gated = _gated_boxes_by_frame(
        detector_records, samples, calibration, project_world_points, wrist_gate_radius_px=400.0
    )
    solved, info = _solve_grip_roll_against_boxes(
        samples,
        r_g=np.eye(3),
        t_g=t_g,
        gated_boxes=gated,
        dims=dims,
        calibration_model=calibration,
        project_world_points=project_world_points,
    )
    assert info["applied"] is True
    # Coordinate-descent grid resolution is 7.5 deg; recovered roll must be within one step.
    assert abs(info["roll_deg"] - true_roll_deg) <= 7.5 + 1e-9
    assert np.max(np.abs(solved.T @ solved - np.eye(3))) < 1e-9


# --------------------------------------------------------------------------------------
# I1c: deviation slew limit, position jump clamp, membership per_player verdicts,
# distinct fused trust-band wording
# --------------------------------------------------------------------------------------


def test_per_frame_deviation_is_slew_limited_to_1cm_per_frame() -> None:
    from threed.racketsport.court_calibration import project_world_points
    from threed.racketsport.paddle_pose_fused import _ComposedFrame, _per_frame_box_deviation
    from threed.racketsport.schemas import CourtCalibration

    calibration = CourtCalibration.model_validate(_synthetic_calibration())
    # A far-off detector box demands a large correction; the deviation may only add <=1cm/frame.
    composed = [
        _ComposedFrame(
            t=i / 30.0,
            frame_idx=i,
            rotation=np.eye(3),
            translation=np.array([0.0, 0.0, 5.0]),
            used_finger_palm_normal=True,
            joint_confidence=0.9,
        )
        for i in range(10)
    ]
    baseline = [frame.translation.copy() for frame in composed]
    gated = {i: [((700.0, 280.0, 740.0, 320.0), 0.9)] for i in range(10)}
    samples = []  # unused by the deviation routine
    info = _per_frame_box_deviation(
        samples,
        composed,
        gated_boxes=gated,
        calibration_model=calibration,
        project_world_points=project_world_points,
        max_deviation_m=0.25,
        decay=0.85,
        slew_m_per_frame=0.01,
    )
    assert info["slew_m_per_frame"] == 0.01
    added = [np.linalg.norm(frame.translation - base) for frame, base in zip(composed, baseline)]
    # Deviation magnitude may grow by at most slew per frame.
    assert added[0] <= 0.01 + 1e-9
    for previous, current in zip(added, added[1:]):
        assert current - previous <= 0.01 + 1e-9


def test_output_position_jump_clamped_outside_declared_switches() -> None:
    # A raw wrist teleport (2 m in one frame) must not produce a >0.30 m one-frame jump in the
    # output when it is not a declared hand switch.
    frames = []
    for i in range(20):
        wrist = (0.0, 0.0, 1.0) if i != 10 else (2.0, 0.0, 1.0)
        middle = (wrist[0], 0.1, 1.0)
        index_pt = (wrist[0] + 0.05, 0.05, 1.0)
        pinky = (wrist[0] - 0.05, 0.05, 1.0)
        elbow = (wrist[0], -0.3, 1.0)
        frames.append(_frame(i / 30.0, wrist=wrist, elbow=elbow, middle=middle, index_pt=index_pt, pinky=pinky))
    out = build_paddle_pose_fused_from_skeleton(_skeleton(frames), clip_id="synthetic")
    player = out["players"][0]
    positions = [np.array(f["pose_se3"]["t"]) for f in sorted(player["frames"], key=lambda f: f["t"])]
    deltas = [float(np.linalg.norm(b - a)) for a, b in zip(positions, positions[1:])]
    assert player["declared_hand_switch_times"] == []
    assert max(deltas) <= 0.30 + 1e-6
    assert player["position_jump_clamped_frame_count"] > 0


def test_membership_per_player_adjacent_or_spectator_is_excluded() -> None:
    frames = [_frame(i / 30.0) for i in range(3)]
    skeleton = _skeleton(frames, player_id=3)
    membership = {"per_player": {"3": {"verdict": "adjacent_or_spectator"}}}
    out = build_paddle_pose_fused_from_skeleton(skeleton, clip_id="synthetic", membership=membership)
    player = out["players"][0]
    assert player["frames"] == []
    assert player.get("membership_excluded") is True
    # "uncertain" and "on_target_court" must NOT exclude.
    for verdict in ("uncertain", "on_target_court"):
        out2 = build_paddle_pose_fused_from_skeleton(
            skeleton, clip_id="synthetic", membership={"per_player": {"3": {"verdict": verdict}}}
        )
        assert len(out2["players"][0]["frames"]) == 3


def test_fused_trust_band_wording_is_distinct_from_wrist_proxy() -> None:
    frames = [_frame(i / 30.0) for i in range(2)]
    out = build_paddle_pose_fused_from_skeleton(_skeleton(frames), clip_id="synthetic")
    top = out["trust_band"]
    assert top["gate_id"] == "wrist_palm_grip_fused_estimated_paddle"
    assert top["gate_status"] == "estimated_from_wrist"  # stays a status virtual_world recognizes
    assert "fused" in top["reason"].lower()
    assert "wrist_proxy" not in top["gate_id"]
    frame = out["players"][0]["frames"][0]
    band = frame["trust_band"]
    assert band["gate_id"] == "wrist_palm_grip_fused_estimated_paddle"
    assert band["status"] == "estimated_from_wrist"
    assert band["gate_status"] == "estimated_from_wrist"
    assert "not the legacy wrist proxy" in band["reason"].lower()
