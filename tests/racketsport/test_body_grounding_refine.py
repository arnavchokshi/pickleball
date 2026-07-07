from __future__ import annotations

import pytest

from threed.racketsport.body_grounding_refine import GroundingRefineConfig, refine_body_grounding


JOINT_NAMES = ("left_hip", "right_hip", "left_ankle", "right_ankle")


def _skeleton(*, offsets_by_frame: dict[int, tuple[float, float, float]], frames: range) -> dict:
    players = [{"id": "p1", "frames": []}]
    for frame_idx in frames:
        t = frame_idx / 30.0
        root_x = 1.0 + frame_idx * 0.1
        root_y = 2.0
        dx, dy, dz = offsets_by_frame.get(frame_idx, (0.0, 0.0, 0.0))
        true_joints = [
            [root_x - 0.1, root_y, 1.0],
            [root_x + 0.1, root_y, 1.0],
            [root_x - 0.12, root_y, 0.0],
            [root_x + 0.12, root_y, 0.0],
        ]
        players[0]["frames"].append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "joints_world": [[x + dx, y + dy, z + dz] for x, y, z in true_joints],
                "joint_conf": [1.0, 1.0, 1.0, 1.0],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_skeleton3d",
        "fps": 30,
        "joint_names": list(JOINT_NAMES),
        "players": players,
    }


def _tracks(frames: range) -> dict:
    return {
        "schema_version": 1,
        "fps": 30,
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "t": frame_idx / 30.0,
                        "world_xy": [1.0 + frame_idx * 0.1, 2.0],
                        "conf": 1.0,
                    }
                    for frame_idx in frames
                ],
            }
        ],
    }


def _phases(*ranges: range) -> dict:
    return {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "phase_count": len(ranges),
        "phases": [
            {
                "player_id": "p1",
                "foot": "left",
                "start_frame_index": frame_range.start,
                "end_frame_index": frame_range.stop - 1,
                "frame_indices": list(frame_range),
                "frame_count": len(frame_range),
                "anchor_position_xyz": [0.0, 0.0, 0.0],
                "min_confidence": 0.95,
                "max_height_m": 0.01,
                "max_speed_mps": 0.10,
                "source_thresholds": {"min_confidence": 0.20},
                "source_phase_foot": "left",
                "foot_assignment": "per_foot_body_contact",
                "assignment_evidence": {"body_detector_agreement": 0.95},
            }
            for frame_range in ranges
        ],
    }


def _weak_bilateral_phases(*ranges: range) -> dict:
    return {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "phase_count": len(ranges),
        "phases": [
            {
                "player_id": "p1",
                "foot": "left",
                "start_frame_index": frame_range.start,
                "end_frame_index": frame_range.stop - 1,
                "frame_indices": list(frame_range),
                "frame_count": len(frame_range),
                "source": "native_keypoint_low_speed",
                "source_phase_foot": "unknown",
                "foot_assignment": "bilateral_from_player_stance",
                "weak": True,
                "demoted": True,
                "rejection_reason": "weak_bilateral_unknown_foot",
            }
            for frame_range in ranges
        ],
    }


def test_known_rigid_offset_is_recovered_without_changing_pose_shape() -> None:
    frames = range(0, 4)
    injected = (0.30, -0.20, 0.08)
    source = _skeleton(offsets_by_frame={frame_idx: injected for frame_idx in frames}, frames=frames)

    refined, report = refine_body_grounding(
        source,
        foot_contact_phases=_phases(frames),
        tracks=_tracks(frames),
        config=GroundingRefineConfig(smoothness_weight=0.0, max_correction_warn_m=1.0),
    )

    correction = report["players"]["p1"]["frames"][0]["translation_delta_xyz"]
    assert correction == pytest.approx([-0.30, 0.20, -0.08], abs=1e-9)
    assert report["summary"]["foot_plane_residual_m"]["mean_abs_after"] < 1e-9
    assert report["summary"]["track_residual_m"]["mean_after"] < 1e-9
    before_vector = [
        source["players"][0]["frames"][0]["joints_world"][1][axis]
        - source["players"][0]["frames"][0]["joints_world"][0][axis]
        for axis in range(3)
    ]
    after_vector = [
        refined["players"][0]["frames"][0]["joints_world"][1][axis]
        - refined["players"][0]["frames"][0]["joints_world"][0][axis]
        for axis in range(3)
    ]
    assert after_vector == pytest.approx(before_vector, abs=1e-9)


def test_contact_phase_piecewise_offsets_are_constant_inside_each_phase() -> None:
    frames = range(0, 8)
    source = _skeleton(
        offsets_by_frame={
            **{frame_idx: (0.10, 0.00, 0.05) for frame_idx in range(0, 3)},
            **{frame_idx: (-0.20, 0.00, 0.10) for frame_idx in range(5, 8)},
        },
        frames=frames,
    )

    _refined, report = refine_body_grounding(
        source,
        foot_contact_phases=_phases(range(0, 3), range(5, 8)),
        tracks=_tracks(frames),
        config=GroundingRefineConfig(smoothness_weight=0.0, max_correction_warn_m=1.0),
    )

    by_frame = {frame["frame_index"]: frame["translation_delta_xyz"] for frame in report["players"]["p1"]["frames"]}
    assert by_frame[0] == pytest.approx(by_frame[1])
    assert by_frame[1] == pytest.approx(by_frame[2])
    assert by_frame[5] == pytest.approx(by_frame[6])
    assert by_frame[6] == pytest.approx(by_frame[7])
    assert by_frame[0] != pytest.approx(by_frame[5])


def test_large_corrections_use_physics_corrected_warn_band() -> None:
    source = _skeleton(offsets_by_frame={0: (0.40, 0.0, 0.0)}, frames=range(0, 1))

    refined, report = refine_body_grounding(
        source,
        foot_contact_phases=_phases(range(0, 1)),
        tracks=_tracks(range(0, 1)),
        config=GroundingRefineConfig(smoothness_weight=0.0, max_correction_warn_m=0.15),
    )

    frame_report = report["players"]["p1"]["frames"][0]
    frame_payload = refined["players"][0]["frames"][0]
    assert frame_report["confidence_provenance"]["band"] == "physics_corrected_warn"
    assert frame_payload["confidence_provenance"]["band"] == "physics_corrected_warn"
    assert frame_report["correction_magnitude_m"] > 0.15


def test_xy_disabled_mode_keeps_placement_anchor_and_only_repairs_z() -> None:
    source = _skeleton(offsets_by_frame={0: (0.40, -0.20, 0.08)}, frames=range(0, 1))
    source["players"][0]["frames"][0]["transl_world"] = [1.0, 2.0, 0.08]

    refined, report = refine_body_grounding(
        source,
        foot_contact_phases=_phases(range(0, 1)),
        tracks=_tracks(range(0, 1)),
        config=GroundingRefineConfig(smoothness_weight=0.0, max_correction_warn_m=1.0, xy_translation_enabled=False),
    )

    correction = report["players"]["p1"]["frames"][0]["translation_delta_xyz"]
    assert correction == pytest.approx([0.0, 0.0, -0.08], abs=1e-9)
    assert refined["players"][0]["frames"][0]["transl_world"] == pytest.approx([1.0, 2.0, 0.0], abs=1e-9)
    assert report["policy"]["xy_translation_enabled"] is False


def test_weak_bilateral_phases_are_filtered_and_noop_honestly() -> None:
    frames = range(0, 3)
    source = _skeleton(offsets_by_frame={frame_idx: (0.40, -0.20, 0.08) for frame_idx in frames}, frames=frames)

    refined, report = refine_body_grounding(
        source,
        foot_contact_phases=_weak_bilateral_phases(frames),
        tracks=_tracks(frames),
        config=GroundingRefineConfig(smoothness_weight=0.0, max_correction_warn_m=1.0),
    )

    assert refined["players"][0]["frames"] == source["players"][0]["frames"]
    assert report["summary"]["status"] == "no_op_no_confident_per_foot_phases"
    assert report["summary"]["kill_recommended"] is False
    assert report["source"]["foot_contact_phase_count"] == 1
    assert report["source"]["consumed_foot_contact_phase_count"] == 0
    assert report["phase_filter"]["rejected_phase_count"] == 1
    assert report["phase_filter"]["rejected_phases"][0]["reason"] == "weak_bilateral_unknown_foot"
    assert report["summary"]["foot_plane_residual_m"]["count"] == 0


def test_per_foot_phases_without_body_detector_agreement_noop_honestly() -> None:
    frames = range(0, 3)
    source = _skeleton(offsets_by_frame={frame_idx: (0.0, 0.0, 0.05) for frame_idx in frames}, frames=frames)
    phases = _phases(frames)
    phases["phases"][0].pop("assignment_evidence")

    refined, report = refine_body_grounding(
        source,
        foot_contact_phases=phases,
        config=GroundingRefineConfig(smoothness_weight=0.0),
    )

    assert refined["players"] == source["players"]
    assert report["summary"]["status"] == "no_op_no_confident_per_foot_phases"
    assert report["summary"]["kill_recommended"] is False
    assert report["source"]["consumed_foot_contact_phase_count"] == 0
    assert report["phase_filter"]["rejected_reasons"] == {"missing_body_detector_agreement": 1}
