from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.pose_temporal import (
    _apply_sam3d_skeleton_plausibility,
    _joint_smoothing_group,
    apply_sam3d_wrist_bone_lock,
    compare_wrist_peak_timing,
    refine_lane_a_skeleton3d,
    refine_sam3d_skeleton3d,
)
from threed.racketsport.schemas import Skeleton3D
from threed.racketsport.skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP


FIXTURE = Path("tests/racketsport/fixtures/sam3d_smpl_motion_237_excerpt_skeleton3d.json")


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _sam3d_idx(name: str) -> int:
    return SAM3D_BODY_MHR70_SEMANTIC_MAP.joints[name]


def _minimal_sam3d_lock_skeleton() -> dict:
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    left_elbow = _sam3d_idx("left_elbow")
    left_wrist = _sam3d_idx("left_wrist")
    right_elbow = _sam3d_idx("right_elbow")
    right_wrist = _sam3d_idx("right_wrist")
    frames = []
    left_dirs = [
        (1.0, 0.0, 0.0),
        (0.985, 0.174, 0.0),
        (0.866, 0.500, 0.0),
        (0.766, 0.643, 0.0),
    ]
    right_dirs = [
        (0.0, 1.0, 0.0),
        (0.174, 0.985, 0.0),
        (0.500, 0.866, 0.0),
        (0.643, 0.766, 0.0),
    ]
    for frame_idx, (left_dir, right_dir) in enumerate(zip(left_dirs, right_dirs)):
        joints = [[0.0, 0.0, 0.0] for _idx in range(70)]
        joints[left_elbow] = [1.0, 0.0, 1.0]
        joints[right_elbow] = [-1.0, 0.0, 1.0]
        joints[left_wrist] = [1.0 + 0.60 * left_dir[0], 0.60 * left_dir[1], 1.0 + 0.60 * left_dir[2]]
        joints[right_wrist] = [-1.0 + 0.70 * right_dir[0], 0.70 * right_dir[1], 1.0 + 0.70 * right_dir[2]]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "joints_world": joints,
                "joint_conf": [1.0] * 70,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": joint_names,
        "preview_only": False,
        "players": [{"id": 7, "frames": frames}],
        "provenance": {"source": "sam3d_body_joints"},
    }


def test_sam3d_postprocess_preserves_all_70_joints_and_checks_wrist_peak_timing() -> None:
    skeleton = _load_fixture()
    assert skeleton["provenance"]["fixture_original_player_frame_count"] == 237

    refined = refine_sam3d_skeleton3d(skeleton, fps=30.0)

    assert refined["source_model"] == "sam3d_body_joints"
    assert len(refined["joint_names"]) == 70
    assert refined["players"][0]["frames"]
    assert all(len(frame["joints_world"]) == 70 for frame in refined["players"][0]["frames"])
    assert all(len(frame["smoothing_flag"]) == 70 for frame in refined["players"][0]["frames"])
    temporal = refined["provenance"]["temporal_refine"]
    assert temporal["source"] == "sam3d_body_joints"
    assert temporal["one_euro"]["filtered_joint_count"] == 70
    assert temporal["wrist_peak_timing"]["status"] == "pass"
    assert temporal["wrist_peak_timing"]["max_abs_delta_frames"] <= 1
    assert refined["provenance"]["sam3d_skeleton_plausibility"]["checked_frame_count"] == len(
        refined["players"][0]["frames"]
    )
    assert "stature_check" in refined["provenance"]


def test_sam3d_wrist_bone_lock_projects_wrist_only_and_preserves_peak_frames() -> None:
    skeleton = _minimal_sam3d_lock_skeleton()
    left_elbow = _sam3d_idx("left_elbow")
    left_wrist = _sam3d_idx("left_wrist")
    right_elbow = _sam3d_idx("right_elbow")
    right_wrist = _sam3d_idx("right_wrist")
    canonical = {
        "artifact_type": "bone_calib_canonical_lengths",
        "players": {
            "7": {
                "bones": {
                    "left_lower_arm": {"median_m": 0.30},
                    "right_lower_arm": {"median_m": 0.35},
                }
            }
        },
    }

    locked = apply_sam3d_wrist_bone_lock(skeleton, canonical_bone_lengths=canonical)

    timing = compare_wrist_peak_timing(skeleton, locked, top_k=2, max_allowed_delta_frames=0, min_peak_speed_mps=0.0)
    assert timing["status"] == "pass"
    assert timing["max_abs_delta_frames"] == 0
    for before, after in zip(skeleton["players"][0]["frames"], locked["players"][0]["frames"]):
        assert after["joints_world"][left_elbow] == pytest.approx(before["joints_world"][left_elbow])
        assert after["joints_world"][right_elbow] == pytest.approx(before["joints_world"][right_elbow])
        assert _distance(after["joints_world"][left_elbow], after["joints_world"][left_wrist]) == pytest.approx(0.30)
        assert _distance(after["joints_world"][right_elbow], after["joints_world"][right_wrist]) == pytest.approx(0.35)
        assert _unit_vector(before["joints_world"][left_elbow], before["joints_world"][left_wrist]) == pytest.approx(
            _unit_vector(after["joints_world"][left_elbow], after["joints_world"][left_wrist])
        )
    provenance = locked["provenance"]["sam3d_wrist_bone_lock"]
    assert provenance["status"] == "applied"
    assert provenance["players"]["7"]["left_lower_arm"]["locked_frame_count"] == 4
    assert provenance["players"]["7"]["right_lower_arm"]["locked_frame_count"] == 4
    assert provenance["players"]["7"]["left_lower_arm"]["mean_abs_post_length_delta_m"] == pytest.approx(0.0)
    assert provenance["wrist_peak_timing_after_lock"]["max_abs_delta_frames"] == 0


def test_sam3d_wrist_bone_lock_skips_low_confidence_and_degenerate_frames() -> None:
    skeleton = _minimal_sam3d_lock_skeleton()
    left_elbow = _sam3d_idx("left_elbow")
    left_wrist = _sam3d_idx("left_wrist")
    frames = skeleton["players"][0]["frames"]
    frames[0]["joints_world"][left_wrist] = list(frames[0]["joints_world"][left_elbow])
    frames[1]["joint_conf"][left_wrist] = 0.01

    locked = apply_sam3d_wrist_bone_lock(
        skeleton,
        canonical_bone_lengths={"players": {"7": {"bones": {"left_lower_arm": {"median_m": 0.30}}}}},
        confidence_floor=0.25,
    )

    left = locked["provenance"]["sam3d_wrist_bone_lock"]["players"]["7"]["left_lower_arm"]
    assert left["locked_frame_count"] == 2
    assert left["degenerate_frame_count"] == 1
    assert left["low_confidence_frame_count"] == 1
    assert locked["players"][0]["frames"][0]["joints_world"][left_wrist] == pytest.approx(
        skeleton["players"][0]["frames"][0]["joints_world"][left_wrist]
    )


def test_sam3d_wrist_bone_lock_disabled_preserves_payload() -> None:
    skeleton = _minimal_sam3d_lock_skeleton()

    locked = apply_sam3d_wrist_bone_lock(skeleton, enabled=False)

    assert locked == skeleton


def test_core_body_speed_clamp_engagement_ignores_raw_fast_motion() -> None:
    joint_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    frames = []
    for frame_idx, nose_x in enumerate([0.0, 0.0, 0.4, 0.8]):
        joints = [[0.0, 0.0, 1.0] for _name in joint_names]
        joints[0] = [nose_x, 0.0, 1.0]
        frames.append({"frame_idx": frame_idx, "t": frame_idx / 30.0, "joints_world": joints, "joint_conf": [1.0] * len(joints)})
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "joint_names": joint_names,
        "players": [{"id": 3, "frames": frames}],
    }

    refined = refine_lane_a_skeleton3d(
        skeleton,
        fps=30.0,
        apply_world_grounding=False,
        smoothing_max_displacement_m=0.03,
    )

    engagement = refined["provenance"]["temporal_refine"]["physical_plausibility"][
        "core_body_speed_clamp_engagement_by_player"
    ]["3"]
    assert engagement["frame_count"] == 4
    assert engagement["clamped_frame_count"] == 0
    assert engagement["clamp_engagement_fraction"] == 0.0
    assert refined["provenance"]["temporal_refine"]["physical_plausibility"][
        "core_body_speed_clamp_engagement_flag"
    ] == "final_core_speed_clamped"


def test_refine_sam3d_skeleton3d_caps_temporal_displacement_from_raw_by_default() -> None:
    skeleton = _minimal_sam3d_lock_skeleton()
    nose_idx = MHR70_JOINT_NAMES.index("nose")
    for frame_pos, frame in enumerate(skeleton["players"][0]["frames"]):
        frame["joints_world"][nose_idx] = [float(frame_pos), 0.0, 1.6]

    refined = refine_sam3d_skeleton3d(skeleton, fps=30.0)

    temporal = refined["provenance"]["temporal_refine"]
    assert temporal["one_euro"]["smoothing_max_displacement_m"] == pytest.approx(0.03)
    assert temporal["physical_plausibility"]["smoothing_max_displacement_m"] == pytest.approx(0.03)
    for before, after in zip(skeleton["players"][0]["frames"], refined["players"][0]["frames"]):
        raw_joint = before["joints_world"][nose_idx]
        refined_joint = after["joints_world"][nose_idx]
        assert _distance(raw_joint, refined_joint) <= 0.03 + 1e-9


def test_sam3d_plausibility_gate_flags_bone_zscore_and_confidence_floor() -> None:
    skeleton = _load_fixture()
    mutated = copy.deepcopy(skeleton)
    frames = mutated["players"][0]["frames"]
    bad_frame = frames[len(frames) // 2]
    right_shoulder = _sam3d_idx("right_shoulder")
    right_elbow = _sam3d_idx("right_elbow")
    left_wrist = _sam3d_idx("left_wrist")
    bad_frame["joints_world"][right_elbow] = [
        bad_frame["joints_world"][right_shoulder][0] + 3.0,
        bad_frame["joints_world"][right_shoulder][1],
        bad_frame["joints_world"][right_shoulder][2],
    ]
    bad_frame["joint_conf"][left_wrist] = 0.05

    refined = refine_sam3d_skeleton3d(
        mutated,
        fps=30.0,
        plausibility_joint_confidence_floor=0.25,
        plausibility_max_bone_zscore=6.0,
    )

    out_frame = refined["players"][0]["frames"][len(frames) // 2]
    reasons = out_frame["skeleton_plausibility"]["reasons"]
    assert out_frame["skeleton_implausible"] is True
    assert any(reason.startswith("joint_conf_below_floor:left_wrist") for reason in reasons)
    assert any(reason.startswith("bone_length_zscore:right_shoulder-right_elbow") for reason in reasons)
    summary = refined["provenance"]["sam3d_skeleton_plausibility"]
    assert summary["implausible_frame_count"] >= 1
    assert summary["reason_counts"]["joint_conf_below_floor"] >= 1
    assert summary["reason_counts"]["bone_length_zscore"] >= 1


def test_sam3d_bone_length_outlier_smoothing_downweights_mad_outlier_joint() -> None:
    skeleton = _minimal_sam3d_lock_skeleton()
    left_shoulder = _sam3d_idx("left_shoulder")
    left_elbow = _sam3d_idx("left_elbow")
    for frame in skeleton["players"][0]["frames"]:
        frame["joints_world"][left_shoulder] = [0.0, 0.0, 1.5]
        frame["joints_world"][left_elbow] = [0.3, 0.0, 1.5]
    skeleton["players"][0]["frames"][2]["joints_world"][left_elbow] = [0.6, 0.0, 1.5]

    enabled = refine_sam3d_skeleton3d(
        skeleton,
        fps=30.0,
        sam3d_bone_length_outlier_smoothing=True,
        smoothing_max_displacement_m=None,
        plausibility_max_bone_zscore=3.0,
        plausibility_min_bone_samples=3,
    )
    disabled = refine_sam3d_skeleton3d(
        skeleton,
        fps=30.0,
        sam3d_bone_length_outlier_smoothing=False,
        smoothing_max_displacement_m=None,
        plausibility_max_bone_zscore=3.0,
        plausibility_min_bone_samples=3,
    )

    temporal = enabled["provenance"]["temporal_refine"]
    assert temporal["bone_length_outlier_smoothing"]["enabled"] is True
    assert temporal["bone_length_outlier_smoothing"]["weighted_joint_count"] >= 1
    assert temporal["smoothing_flags"]["bone_length_outlier_weighted"] >= 1
    assert "bone_length_outlier_weighted" in enabled["players"][0]["frames"][2]["smoothing_flag"][left_elbow]
    assert "bone_length_outlier_weighted" not in disabled["players"][0]["frames"][2]["smoothing_flag"][left_elbow]


def test_refine_sam3d_skeleton3d_enables_bone_outlier_smoothing_by_default() -> None:
    refined = refine_sam3d_skeleton3d(_minimal_sam3d_lock_skeleton(), fps=30.0)

    smoothing = refined["provenance"]["temporal_refine"]["bone_length_outlier_smoothing"]
    assert smoothing["enabled"] is True
    assert smoothing["source"] == "mad_bone_length_zscore"


def test_sam3d_plausibility_annotations_validate_through_skeleton3d_schema() -> None:
    skeleton = _load_fixture()
    mutated = {
        "schema_version": skeleton["schema_version"],
        "artifact_type": skeleton["artifact_type"],
        "source_model": skeleton["source_model"],
        "joint_names": skeleton["joint_names"],
        "preview_only": skeleton["preview_only"],
        "players": [
            {
                "id": skeleton["players"][0]["id"],
                "frames": [
                    {
                        "frame_idx": frame.get("frame_idx"),
                        "t": frame["t"],
                        "joints_world": copy.deepcopy(frame["joints_world"]),
                        "joint_conf": copy.deepcopy(frame["joint_conf"]),
                    }
                    for frame in skeleton["players"][0]["frames"]
                ],
            }
        ],
    }
    frame = mutated["players"][0]["frames"][len(mutated["players"][0]["frames"]) // 2]
    left_wrist = _sam3d_idx("left_wrist")
    frame["joint_conf"][left_wrist] = 0.01

    annotated, summary = _apply_sam3d_skeleton_plausibility(
        mutated,
        confidence_floor=0.25,
        max_bone_zscore=6.0,
        min_bone_samples=4,
        min_sigma_m=0.03,
    )
    annotated_frame = annotated["players"][0]["frames"][len(mutated["players"][0]["frames"]) // 2]

    parsed = Skeleton3D.model_validate(annotated)

    parsed_frame = parsed.players[0].frames[len(mutated["players"][0]["frames"]) // 2]
    assert summary["implausible_frame_count"] >= 1
    assert annotated_frame["skeleton_implausible"] is True
    assert annotated_frame["skeleton_plausibility"]["status"] == "low_confidence"
    assert annotated_frame["trust_band"]["badge"] == "low_confidence"
    assert parsed_frame.skeleton_implausible is True
    assert parsed_frame.skeleton_plausibility is not None
    assert parsed_frame.skeleton_plausibility.source == "sam3d_body_joints"
    assert parsed_frame.trust_band is not None
    assert parsed_frame.trust_band.badge == "low_confidence"


def test_sam3d_postprocess_rejects_non_sam3d_skeleton() -> None:
    skeleton = _load_fixture()
    skeleton["source_model"] = "legacy_body65_joints"

    with pytest.raises(ValueError, match="SAM-3D"):
        refine_sam3d_skeleton3d(skeleton)


def test_sam3d_heel_and_toe_tip_joints_resolve_to_feet_smoothing_group() -> None:
    # runs/sam3d_foot_wander_20260703T1024Z/REPORT.md: heel and toe-tip joints were
    # silently bucketed as "core_body" for SAM-3D's generic sam3dbody_joint_### payloads
    # because SAM3D_BODY_MHR70_SEMANTIC_MAP only covers the 12 wrist/limb/ankle gate
    # joints, and _joint_smoothing_group compared its canonicalized name against a set of
    # RAW joint_names entries that could never match. Ankle, hip, knee, shoulder, elbow,
    # and wrist were already covered by the semantic map and must stay exactly as before.
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    expected_group = {
        "left_ankle": "feet",
        "right_ankle": "feet",
        "left_big_toe_tip": "feet",
        "left_small_toe_tip": "feet",
        "left_heel": "feet",
        "right_big_toe_tip": "feet",
        "right_small_toe_tip": "feet",
        "right_heel": "feet",
        "left_wrist": "wrists",
        "right_wrist": "wrists",
        "left_hip": "core_body",
        "left_knee": "core_body",
        "left_shoulder": "core_body",
        "left_elbow": "core_body",
    }
    for canonical_name, expected in expected_group.items():
        idx = MHR70_JOINT_NAMES.index(canonical_name)
        assert _joint_smoothing_group(joint_names[idx], joint_names) == expected, canonical_name


def _sam3d_foot_lag_skeleton() -> dict:
    left_elbow = _sam3d_idx("left_elbow")
    left_wrist = _sam3d_idx("left_wrist")
    right_elbow = _sam3d_idx("right_elbow")
    right_wrist = _sam3d_idx("right_wrist")
    left_heel = MHR70_JOINT_NAMES.index("left_heel")
    frames = []
    for frame_idx, offset in enumerate([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
        joints = [[0.0, 0.0, 0.0] for _idx in range(70)]
        joints[left_elbow] = [1.0, 0.0, 1.0]
        joints[right_elbow] = [-1.0, 0.0, 1.0]
        joints[left_wrist] = [1.3, 0.0, 1.0]
        joints[right_wrist] = [-1.3, 0.0, 1.0]
        joints[left_heel] = [offset, 0.0, 0.02]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "joints_world": joints,
                "joint_conf": [1.0] * 70,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(70)],
        "preview_only": False,
        "players": [{"id": 7, "frames": frames}],
        "provenance": {"source": "sam3d_body_joints"},
    }


def test_refine_sam3d_skeleton3d_foot_low_lag_smoothing_reduces_heel_lag_vs_legacy() -> None:
    skeleton = _sam3d_foot_lag_skeleton()
    left_heel = MHR70_JOINT_NAMES.index("left_heel")

    low_lag = refine_sam3d_skeleton3d(skeleton, fps=30.0)
    legacy = refine_sam3d_skeleton3d(
        skeleton,
        fps=30.0,
        sam3d_foot_low_lag_smoothing=False,
        smoothing_max_displacement_m=None,
    )

    # Frame 2 is the first frame after the heel's step change to 1.0; near-pass-through
    # foot smoothing must track the step much more closely than the legacy generic
    # (mincutoff=1.0, beta=0.3) treatment shared with "hands".
    low_lag_x = low_lag["players"][0]["frames"][2]["joints_world"][left_heel][0]
    legacy_x = legacy["players"][0]["frames"][2]["joints_world"][left_heel][0]
    assert low_lag_x > legacy_x
    assert low_lag_x == pytest.approx(1.0, abs=0.6)
    assert legacy_x < 0.3


def test_refine_sam3d_skeleton3d_records_foot_treatment_provenance() -> None:
    skeleton = _sam3d_foot_lag_skeleton()

    enabled = refine_sam3d_skeleton3d(skeleton, fps=30.0)
    disabled = refine_sam3d_skeleton3d(skeleton, fps=30.0, sam3d_foot_low_lag_smoothing=False)

    enabled_treatment = enabled["provenance"]["sam3d_foot_treatment"]
    assert enabled_treatment["low_lag_smoothing_enabled"] is True
    assert enabled_treatment["foot_one_euro_mincutoff"] == 1000.0
    assert enabled_treatment["foot_one_euro_beta"] == 0.0
    assert enabled_treatment["heel_toe_canonical_name_resolution"] == "mhr70_positional_fallback"

    disabled_treatment = disabled["provenance"]["sam3d_foot_treatment"]
    assert disabled_treatment["low_lag_smoothing_enabled"] is False
    assert disabled_treatment["foot_one_euro_mincutoff"] == 1.0
    assert disabled_treatment["foot_one_euro_beta"] == 0.3

    enabled_one_euro = enabled["provenance"]["temporal_refine"]["one_euro"]
    assert enabled_one_euro["foot_mincutoff"] == 1000.0
    assert enabled_one_euro["foot_beta"] == 0.0


def _distance(first: list[float], second: list[float]) -> float:
    return sum((first[idx] - second[idx]) ** 2 for idx in range(3)) ** 0.5


def _unit_vector(first: list[float], second: list[float]) -> list[float]:
    length = _distance(first, second)
    return [(second[idx] - first[idx]) / length for idx in range(3)]
