from __future__ import annotations

import pytest

from threed.racketsport.player_scale import (
    estimate_player_metric_heights,
    normalize_skeleton_scale_payload,
)


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "image_size": [1920, 1080],
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "synthetic"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "metric_confidence": "high",
    }


def _bbox_for_height(height_m: float, *, court_y_m: float, top: float = 100.0) -> list[float]:
    depth_m = 10.0 - court_y_m
    pixel_height = 1000.0 * height_m / depth_m
    return [500.0, top, 560.0, top + pixel_height]


def _tracks_for_height(height_m: float, *, frame_count: int = 80, player_id: int = 1) -> dict:
    frames = []
    for frame_idx in range(frame_count):
        court_y = -3.0 + 0.03 * frame_idx
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "bbox": _bbox_for_height(height_m, court_y_m=court_y),
                "world_xy": [0.2, court_y],
                "conf": 0.92,
            }
        )
    return {"schema_version": 1, "fps": 30.0, "players": [{"id": player_id, "frames": frames}]}


def _skeleton(height_m: float = 1.0) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "source_model": "synthetic",
        "world_frame": "court_Z0",
        "joint_names": ["nose", "pelvis", "left_ankle", "right_ankle"],
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "transl_world": [0.0, 0.0, height_m * 0.5],
                        "joints_world": [
                            [0.0, 0.0, height_m],
                            [0.0, 0.0, height_m * 0.5],
                            [-0.1, 0.0, 0.0],
                            [0.1, 0.0, 0.0],
                        ],
                        "joint_conf": [0.9, 0.9, 0.9, 0.9],
                    }
                ],
            }
        ],
    }


def test_estimate_player_metric_height_recovers_known_height_at_multiple_depths() -> None:
    report = estimate_player_metric_heights(
        _tracks_for_height(1.72),
        _calibration(),
        samples_per_window=12,
    )

    player = report["players"]["1"]
    assert player["status"] == "ok"
    assert player["height_m"] == pytest.approx(1.72, abs=0.02)
    assert player["valid_sample_count"] == 80
    assert player["window_height_spread_m"] <= 0.03
    assert player["confidence"] >= 0.8
    assert report["unstable"] is False
    assert report["protected_eval_labels_used"] is False


def test_estimate_player_metric_height_marks_window_instability() -> None:
    tracks = _tracks_for_height(1.55, frame_count=80)
    for frame_idx, frame in enumerate(tracks["players"][0]["frames"]):
        if frame_idx >= 40:
            court_y = frame["world_xy"][1]
            frame["bbox"] = _bbox_for_height(1.95, court_y_m=court_y)

    report = estimate_player_metric_heights(
        tracks,
        _calibration(),
        window_spread_max_m=0.25,
        samples_per_window=10,
    )

    player = report["players"]["1"]
    assert player["status"] == "unstable"
    assert player["window_height_spread_m"] > 0.25
    assert report["unstable"] is True


def test_normalize_skeleton_scale_payload_scales_about_root_and_clamps_target() -> None:
    estimates = estimate_player_metric_heights(
        _tracks_for_height(2.4),
        _calibration(),
        samples_per_window=12,
    )

    normalized, report = normalize_skeleton_scale_payload(
        _skeleton(height_m=1.0),
        estimates,
        estimate_path="synthetic/player_scale_estimates.json",
        pre_scale_backup_path="synthetic/skeleton3d.pre_player_scale.json",
        min_confidence=0.5,
    )

    frame = normalized["players"][0]["frames"][0]
    assert report["players"]["1"]["estimated_height_m"] == pytest.approx(2.4, abs=0.03)
    assert report["players"]["1"]["target_height_m"] == pytest.approx(2.05)
    assert report["players"]["1"]["clamped"] is True
    assert report["players"]["1"]["scale_factor"] == pytest.approx(2.05)
    assert frame["joints_world"][1] == [0.0, 0.0, 0.5]
    assert frame["joints_world"][0][2] == pytest.approx(1.525)
    assert frame["joints_world"][2][2] == pytest.approx(-0.525)
    provenance = normalized["provenance"]["player_scale_normalization"]
    assert provenance["pre_scale_backup"] == "synthetic/skeleton3d.pre_player_scale.json"
    assert provenance["players"]["1"]["scale_factor"] == pytest.approx(2.05)


def test_normalize_skeleton_scale_uses_head_to_foot_stature_not_bad_limb_outlier() -> None:
    skeleton = _skeleton(height_m=1.0)
    skeleton["joint_names"].append("left_wrist")
    frame = skeleton["players"][0]["frames"][0]
    frame["joints_world"].append([0.2, 0.0, 5.0])
    frame["joint_conf"].append(0.2)
    estimates = estimate_player_metric_heights(
        _tracks_for_height(1.7),
        _calibration(),
        samples_per_window=12,
    )

    normalized, report = normalize_skeleton_scale_payload(
        skeleton,
        estimates,
        estimate_path="synthetic/player_scale_estimates.json",
        pre_scale_backup_path="synthetic/skeleton3d.pre_player_scale.json",
        min_confidence=0.5,
    )

    assert report["players"]["1"]["pre_scale_stature_m"] == pytest.approx(1.0)
    assert report["players"]["1"]["scale_factor"] == pytest.approx(1.7, abs=0.02)
    normalized_frame = normalized["players"][0]["frames"][0]
    assert normalized_frame["joints_world"][0][2] == pytest.approx(1.35, abs=0.02)
