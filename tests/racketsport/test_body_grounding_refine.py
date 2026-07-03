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
