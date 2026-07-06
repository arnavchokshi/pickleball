from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.skeleton_upright import (
    ROTATION_CONVENTION_OFFSET_ROW_TIMES_R,
    repair_skeleton_upright_payload,
    score_upright_conventions,
)


def _tilted_rotation() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def _skeleton(*, height_m: float = 1.62) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "source_model": "sam3d_body_joints",
        "world_frame": "court_Z0",
        "preview_only": False,
        "joint_names": ["nose", "left_hip", "right_hip", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe"],
        "provenance": {"lane": "A"},
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "joints_world": [
                            [0.0, height_m, 0.05],
                            [-0.12, 0.82, 0.05],
                            [0.12, 0.82, 0.05],
                            [-0.12, 0.05, 0.05],
                            [0.12, 0.05, 0.05],
                            [-0.18, 0.0, 0.05],
                            [0.18, 0.0, 0.05],
                        ],
                        "joint_conf": [0.9] * 7,
                    }
                ],
            }
        ],
    }


def test_repair_skeleton_upright_rotates_offsets_and_regrounds_feet() -> None:
    repaired, report = repair_skeleton_upright_payload(
        _skeleton(),
        calibration_rotation=_tilted_rotation(),
        calibration_path="synthetic/court_calibration.json",
    )

    frame = repaired["players"][0]["frames"][0]
    assert frame["joints_world"][0][2] == pytest.approx(1.62)
    assert min(joint[2] for joint in frame["joints_world"][5:]) == pytest.approx(0.0)
    assert frame["transl_world"][2] == pytest.approx(0.82)
    assert report["selected_convention"] == ROTATION_CONVENTION_OFFSET_ROW_TIMES_R
    assert report["metrics_after"]["mean_z_span_m"] == pytest.approx(1.62)
    assert repaired["provenance"]["skeleton_upright_repair"]["rotation_convention"] == ROTATION_CONVENTION_OFFSET_ROW_TIMES_R


def test_score_upright_conventions_selects_row_times_r_on_real_wolverine_frame_fixture() -> None:
    fixture = json.loads(Path("tests/racketsport/fixtures/skeleton_upright_wolverine_real_frame.json").read_text(encoding="utf-8"))

    report = score_upright_conventions(
        fixture["skeleton"],
        calibration_rotation=fixture["calibration"]["extrinsics"]["R"],
    )

    original = report["variants"]["original_no_rotation"]
    selected = report["variants"][report["selected_convention"]]
    assert report["selected_convention"] == ROTATION_CONVENTION_OFFSET_ROW_TIMES_R
    assert selected["mean_z_span_m"] > original["mean_z_span_m"] + 0.5
    assert selected["heads_above_ankles_rate"] == pytest.approx(1.0)


def test_stature_check_flags_uniform_scale_suspect_short_skeletons() -> None:
    repaired, report = repair_skeleton_upright_payload(
        _skeleton(height_m=0.95),
        calibration_rotation=_tilted_rotation(),
        calibration_path="synthetic/court_calibration.json",
        overlay_scale_suspect_caption="SCALE SUSPECT: stature ~0.95m — under investigation",
    )

    player_check = report["stature_check"]["players"]["1"]
    assert player_check["scale_suspect"] is True
    assert player_check["median_standing_z_span_m"] == pytest.approx(0.95)
    assert report["stature_check"]["scale_suspect"] is True
    assert repaired["provenance"]["skeleton_upright_repair"]["stature_check"]["scale_suspect"] is True
    assert repaired["provenance"]["skeleton_upright_repair"]["overlay_caption_extra"].startswith("SCALE SUSPECT")


def test_repair_clamps_severe_smoothed_foot_penetration_outliers() -> None:
    skeleton = _skeleton()
    base_frame = skeleton["players"][0]["frames"][0]
    frames = []
    for frame_idx in range(3):
        frame = json.loads(json.dumps(base_frame))
        frame["frame_idx"] = frame_idx
        frame["t"] = frame_idx / 30.0
        if frame_idx == 1:
            frame["joints_world"][6][1] = -10.0
        frames.append(frame)
    skeleton["players"][0]["frames"] = frames

    repaired, report = repair_skeleton_upright_payload(
        skeleton,
        calibration_rotation=_tilted_rotation(),
        calibration_path="synthetic/court_calibration.json",
        z_smoothing_radius=1,
    )

    outlier_frame = repaired["players"][0]["frames"][1]
    right_big_toe_index = skeleton["joint_names"].index("right_big_toe")
    assert outlier_frame["joints_world"][right_big_toe_index][2] == pytest.approx(0.0)
    clamp = report["z_grounding"]["severe_foot_penetration_outlier_clamp"]
    assert clamp["clamped_joint_count"] == 1
    assert clamp["clamped_frame_count"] == 1
