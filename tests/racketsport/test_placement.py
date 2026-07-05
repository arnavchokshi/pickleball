from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import threed.racketsport.placement as placement_module
from threed.racketsport.placement import (
    FootSignal,
    PlacementConfig,
    bbox_pixel_sigma,
    homography_world_covariance,
    inverse_covariance_fuse,
    kalman_rts_smooth,
    rewrite_tracks_with_placement,
    undistort_pixel,
)
from threed.racketsport.schemas import PlacementArtifact, validate_artifact_file


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibration_payload(*, dist: list[float] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 960.0, "cy": 540.0, "dist": dist or [], "source": "manual"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 0.0],
            "camera_height_m": 1.5,
        },
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[100.0, 300.0], [900.0, 300.0], [500.0, 300.0], [500.0, 180.0]],
        "world_pts": [[-3.0, -6.7, 0.0], [3.0, -6.7, 0.0], [0.0, 6.7, 0.0], [0.0, 0.0, 0.0]],
    }


def _tracks_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": frame_idx / 30.0, "frame_idx": frame_idx, "bbox": [990.0, 900.0, 1110.0, 1050.0], "world_xy": [0.0, 0.0], "conf": 0.9}
                    for frame_idx in range(12)
                ],
            }
        ],
        "rally_spans": [],
    }


def _native2d_payload() -> dict[str, object]:
    foot_joints = [
        {"name": "left_ankle", "x_px": 1048.0, "y_px": 1050.0, "conf": 8.0},
        {"name": "left_heel", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "left_big_toe", "x_px": 1052.0, "y_px": 1050.0, "conf": 7.0},
        {"name": "left_small_toe", "x_px": 1050.0, "y_px": 1052.0, "conf": 7.0},
        {"name": "right_ankle", "x_px": 1048.0, "y_px": 1048.0, "conf": 8.0},
        {"name": "right_heel", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "right_big_toe", "x_px": 1052.0, "y_px": 1050.0, "conf": 7.0},
        {"name": "right_small_toe", "x_px": 1050.0, "y_px": 1052.0, "conf": 7.0},
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "clip": "test_clip",
        "model": "rtmw-test",
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": frame_idx, "t": frame_idx / 30.0, "joints": foot_joints}
                    for frame_idx in range(12)
                ],
            }
        ],
    }


def _sam3d_sidecar_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_keypoints_2d",
        "source": "unit-test",
        "foot_keypoint_indices": {
            "left_ankle": 13,
            "right_ankle": 14,
            "left_toe": 15,
            "right_toe": 16,
            "left_heel": 17,
            "right_heel": 20,
        },
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "t": frame_idx / 30.0,
                        "keypoints": [
                            {"name": "left_ankle", "index": 13, "xy_px": [1050.0, 1050.0], "conf": 0.85},
                            {"name": "right_ankle", "index": 14, "xy_px": [1050.0, 1050.0], "conf": 0.85},
                            {"name": "left_toe", "index": 15, "xy_px": [1050.0, 1050.0], "conf": 0.85},
                            {"name": "right_toe", "index": 16, "xy_px": [1050.0, 1050.0], "conf": 0.85},
                        ],
                    }
                    for frame_idx in range(12)
                ],
            }
        ],
    }


def _moving_native2d_payload() -> dict[str, object]:
    frames = []
    offsets = [0.0, 32.0, -28.0, 24.0, -20.0, 0.0]
    for frame_idx, offset in enumerate(offsets):
        x = 1050.0 + offset
        foot_joints = [
            {"name": "left_ankle", "x_px": x, "y_px": 1050.0, "conf": 9.0},
            {"name": "left_heel", "x_px": x, "y_px": 1050.0, "conf": 9.0},
            {"name": "left_big_toe", "x_px": x, "y_px": 1050.0, "conf": 9.0},
            {"name": "right_ankle", "x_px": x, "y_px": 1050.0, "conf": 9.0},
            {"name": "right_heel", "x_px": x, "y_px": 1050.0, "conf": 9.0},
            {"name": "right_big_toe", "x_px": x, "y_px": 1050.0, "conf": 9.0},
        ]
        frames.append({"frame_idx": frame_idx, "t": frame_idx / 30.0, "joints": foot_joints})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "clip": "test_clip",
        "model": "rtmw-test",
        "players": [{"id": 1, "frames": frames}],
    }


def _six_frame_tracks_payload() -> dict[str, object]:
    payload = _tracks_payload()
    frames = [
        {"t": frame_idx / 30.0, "bbox": [990.0, 900.0, 1010.0, 1000.0], "world_xy": [0.0, 0.0], "conf": 0.9}
        for frame_idx in range(6)
    ]
    payload["players"][0]["frames"] = frames  # type: ignore[index]
    return payload


def _long_clipped_tail_tracks_payload() -> dict[str, object]:
    frames = []
    for frame_idx in range(80):
        if frame_idx < 2:
            bbox = [990.0, 900.0, 1010.0, 1000.0]
            world_xy = [0.0, 0.0]
        else:
            bbox = [990.0, 100.0, 1010.0, 250.0]
            world_xy = [0.0, -7.5]
        frames.append({"t": frame_idx / 30.0, "frame_idx": frame_idx, "bbox": bbox, "world_xy": world_xy, "conf": 0.9})
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [{"id": 1, "side": "near", "role": "left", "frames": frames}],
        "rally_spans": [],
    }


def _native2d_payload_for_frames(frame_indices: list[int]) -> dict[str, object]:
    foot_joints = [
        {"name": "left_ankle", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "left_heel", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "left_big_toe", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "right_ankle", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "right_heel", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
        {"name": "right_big_toe", "x_px": 1050.0, "y_px": 1050.0, "conf": 9.0},
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "clip": "test_clip",
        "model": "rtmw-test",
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": frame_idx, "t": frame_idx / 30.0, "joints": foot_joints}
                    for frame_idx in frame_indices
                ],
            }
        ],
    }


def _camera_motion_payload(*, compensated_frames: dict[int, list[list[float]]], uncompensated_frames: set[int] | None = None) -> dict[str, object]:
    uncompensated_frames = uncompensated_frames or set()
    frames: list[dict[str, object]] = []
    for frame_idx, matrix in sorted(compensated_frames.items()):
        frames.append({"frame_idx": frame_idx, "compensated": True, "model": "homography", "M": matrix})
    for frame_idx in sorted(uncompensated_frames):
        frames.append(
            {
                "frame_idx": frame_idx,
                "compensated": False,
                "model": "identity",
                "reason": "unit_test_uncompensated",
                "M": [[1.0, 0.0, 999.0], [0.0, 1.0, 999.0], [0.0, 0.0, 1.0]],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_camera_motion",
        "verified": False,
        "not_gate_verified": True,
        "reference_frame_idx": 0,
        "frames": frames,
        "summary": {"n_frames": len(frames), "n_compensated": len(compensated_frames)},
    }


def _fast_native2d_excursion_payload() -> tuple[dict[str, object], dict[str, object]]:
    track_frames = []
    keypoint_frames = []
    for frame_idx in range(80):
        if frame_idx < 20:
            y_world = 0.5
        elif frame_idx < 40:
            y_world = 0.5 + (frame_idx - 19) * 0.18
        elif frame_idx < 60:
            y_world = 4.1 - (frame_idx - 39) * 0.18
        else:
            y_world = 0.5
        x_px = 1050.0
        y_px = 1000.0 + y_world * 100.0
        track_frames.append(
            {
                "t": frame_idx / 30.0,
                "frame_idx": frame_idx,
                "bbox": [x_px - 10.0, y_px - 100.0, x_px + 10.0, y_px],
                "world_xy": [0.5, y_world],
                "conf": 0.9,
            }
        )
        joints = [
            {"name": name, "x_px": x_px, "y_px": y_px, "conf": 9.0}
            for name in ("left_ankle", "left_heel", "left_big_toe", "right_ankle", "right_heel", "right_big_toe")
        ]
        keypoint_frames.append({"frame_idx": frame_idx, "t": frame_idx / 30.0, "joints": joints})
    return (
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [{"id": 1, "side": "near", "role": "left", "frames": track_frames}],
            "rally_spans": [],
        },
        {
            "schema_version": 1,
            "artifact_type": "racketsport_keypoints_2d",
            "clip": "test_clip",
            "model": "rtmw-test",
            "players": [{"id": 1, "frames": keypoint_frames}],
        },
    )


def _stance_phases_payload() -> dict[str, object]:
    return {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "phase_count": 1,
        "phases": [
            {
                "player_id": 1,
                "foot": "left",
                "start_frame_index": 0,
                "end_frame_index": 5,
                "frame_indices": [0, 1, 2, 3, 4, 5],
                "frame_count": 6,
                "anchor_position_xyz": [0.0, 0.0, 0.0],
            }
        ],
    }


def test_inverse_covariance_fusion_weights_by_precision() -> None:
    fused_xy, fused_cov = inverse_covariance_fuse(
        [
            FootSignal("noisy", [0.0, 0.0], [[4.0, 0.0], [0.0, 4.0]], used=True),
            FootSignal("precise", [10.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], used=True),
        ]
    )

    assert fused_xy[0] == pytest.approx(8.0)
    assert fused_xy[1] == pytest.approx(0.0)
    assert fused_cov[0][0] == pytest.approx(0.8)
    assert fused_cov[1][1] == pytest.approx(0.8)


def test_homography_covariance_scales_farther_court_more_than_near() -> None:
    # world->image homography with perspective compression for larger y.
    homography = np.array([[100.0, 0.0, 500.0], [0.0, 100.0, 500.0], [0.0, -0.08, 1.0]])
    near_cov = homography_world_covariance(homography, [500.0, 800.0], sigma_px=4.0)
    far_cov = homography_world_covariance(homography, [500.0, 250.0], sigma_px=4.0)

    assert np.trace(far_cov) > np.trace(near_cov)


def test_kalman_rts_smoothing_coasts_occlusions_and_tightens_stance() -> None:
    measurements = {
        0: ([0.0, 0.0], [[0.04, 0.0], [0.0, 0.04]], False),
        1: ([1.0, 0.0], [[0.04, 0.0], [0.0, 0.04]], False),
        2: ([2.0, 0.0], [[0.04, 0.0], [0.0, 0.04]], False),
        4: ([4.3, 0.15], [[0.09, 0.0], [0.0, 0.09]], True),
        5: ([3.8, -0.20], [[0.09, 0.0], [0.0, 0.09]], True),
        6: ([4.1, 0.10], [[0.09, 0.0], [0.0, 0.09]], True),
    }

    smoothed = kalman_rts_smooth(measurements, frame_indices=list(range(7)), fps=30.0)

    assert 2.0 < smoothed[3].xy[0] < 4.3
    raw_stance_range = max(np.linalg.norm(np.array(measurements[i][0]) - [4.0, 0.0]) for i in (4, 5, 6))
    smooth_stance_range = max(np.linalg.norm(np.array(smoothed[i].xy) - np.array(smoothed[4].xy)) for i in (4, 5, 6))
    assert smooth_stance_range < raw_stance_range
    assert abs(smoothed[6].velocity[0]) < 1.0


def test_undistort_pixel_matches_opencv() -> None:
    cv2 = pytest.importorskip("cv2")
    k = [[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]]
    dist = [0.1, -0.02, 0.001, 0.0005]
    pixel = [1200.0, 700.0]

    actual = undistort_pixel(pixel, k, dist)
    expected = cv2.undistortPoints(
        np.array([[pixel]], dtype=np.float64),
        np.array(k, dtype=np.float64),
        np.array(dist, dtype=np.float64),
        P=np.array(k, dtype=np.float64),
    )[0, 0]

    assert actual == pytest.approx(expected.tolist())


def test_bbox_sigma_inflates_for_far_crouched_boxes() -> None:
    near_standing = bbox_pixel_sigma([100.0, 100.0, 140.0, 260.0], side="near", height_norm=1.0)
    far_crouched = bbox_pixel_sigma([100.0, 100.0, 180.0, 180.0], side="far", height_norm=0.65)

    assert far_crouched > near_standing * 2.0


def test_rewrite_tracks_writes_backup_provenance_and_keeps_coverage(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(tracks_path, _tracks_payload())
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _native2d_payload())

    result = rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=80.0, process_noise_mps2=0.1),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    backup = json.loads((tmp_path / "tracks_prewrite_backup.json").read_text(encoding="utf-8"))
    parsed = validate_artifact_file("placement", placement_path)
    assert isinstance(parsed, PlacementArtifact)
    assert result.coverage_unchanged is True
    assert len(rewritten["players"][0]["frames"]) == len(backup["players"][0]["frames"])
    assert rewritten["players"][0]["frames"][0]["world_xy"] == pytest.approx([0.5, 0.5], abs=0.08)
    assert rewritten["placement_provenance"]["tracks_backup"] == "tracks_prewrite_backup.json"
    assert backup["placement_provenance"]["backup_of"] == "tracks.json"
    assert parsed.summary.source_counts["native2d"] == 12


def test_refine_pass_is_idempotent_for_same_sam3d_sidecar(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    sam3d_path = tmp_path / "sam3d_keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(tracks_path, _tracks_payload())
    _write_json(calibration_path, _calibration_payload())
    _write_json(sam3d_path, _sam3d_sidecar_payload())

    kwargs = {
        "tracks_path": tracks_path,
        "calibration_path": calibration_path,
        "placement_path": placement_path,
        "sam3d_keypoints_path": sam3d_path,
        "refine_from_sam3d": True,
        "config": PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=80.0, process_noise_mps2=0.1),
    }
    rewrite_tracks_with_placement(**kwargs)
    first = json.loads(tracks_path.read_text(encoding="utf-8"))
    rewrite_tracks_with_placement(**kwargs)
    second = json.loads(tracks_path.read_text(encoding="utf-8"))

    assert second["players"][0]["frames"] == first["players"][0]["frames"]
    assert (tmp_path / "tracks_prewrite_backup.json").is_file()


def test_stance_phase_artifact_anchors_external_contact_windows(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    phases_path = tmp_path / "foot_contact_phases.json"
    placement_path = tmp_path / "placement.json"
    _write_json(tracks_path, _six_frame_tracks_payload())
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _moving_native2d_payload())
    _write_json(phases_path, _stance_phases_payload())

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        stance_phases_path=phases_path,
        refine_from_sam3d=True,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=200.0, process_noise_mps2=3.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    xs = [frame["world_xy"][0] for frame in rewritten["players"][0]["frames"]]
    assert max(xs) - min(xs) == pytest.approx(0.0, abs=1e-9)
    placement = validate_artifact_file("placement", placement_path)
    assert isinstance(placement, PlacementArtifact)
    assert placement.provenance["stance_phases"] == "foot_contact_phases.json"
    assert placement.summary.stance_wobble_before_after_m["1"]["phase_count"] == 1


def test_rewrite_tracks_emits_native_stance_from_low_speed_dwell_without_external_phases(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    frames = []
    world_x_by_frame = [
        0.000,
        0.003,
        0.006,
        0.009,
        0.012,
        0.015,
        0.170,
        0.330,
        0.490,
        0.650,
        0.810,
        0.970,
        1.130,
        1.290,
        1.450,
        1.453,
        1.456,
        1.459,
        1.462,
        1.465,
    ]
    for frame_idx, world_x in enumerate(world_x_by_frame):
        bottom_x = 1000.0 + world_x * 100.0
        frames.append(
            {
                "t": frame_idx / 30.0,
                "bbox": [bottom_x - 10.0, 900.0, bottom_x + 10.0, 1000.0],
                "world_xy": [world_x, 0.0],
                "conf": 0.9,
            }
        )
    _write_json(
        tracks_path,
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [{"id": 1, "side": "near", "role": "left", "frames": frames}],
            "rally_spans": [],
        },
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        config=PlacementConfig(process_noise_mps2=0.1),
    )

    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    placement_frames = placement["players"][0]["frames"]
    stance_frames = [frame for frame in placement_frames if frame["stance"]]
    coverage = len(stance_frames) / len(placement_frames)
    stance_frame_indices = {frame["frame_idx"] for frame in stance_frames}
    pair_speeds = []
    for left, right in zip(placement_frames, placement_frames[1:], strict=False):
        if left["frame_idx"] not in stance_frame_indices or right["frame_idx"] not in stance_frame_indices:
            continue
        dt = (right["frame_idx"] - left["frame_idx"]) / 30.0
        pair_speeds.append(
            np.linalg.norm(np.array(right["smoothed_world_xy"]) - np.array(left["smoothed_world_xy"])) / dt
        )

    assert placement["provenance"]["stance_phases"] is None
    assert placement["provenance"]["stance_phase_count"] > 0
    assert placement["provenance"]["stance_detection"]["players"]["1"]["native_phase_count"] == 2
    assert 0.15 <= coverage <= 0.80
    assert pair_speeds
    assert max(pair_speeds) <= PlacementConfig().stance_speed_mps


def test_rejected_out_of_bounds_gap_does_not_become_smoothing_measurement(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(tracks_path, _long_clipped_tail_tracks_payload())
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _native2d_payload_for_frames([0, 1]))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=100.0, process_noise_mps2=0.1),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    frames = rewritten["players"][0]["frames"]
    speeds = [
        np.linalg.norm(np.array(next_frame["world_xy"]) - np.array(frame["world_xy"])) * 30.0
        for frame, next_frame in zip(frames, frames[1:], strict=False)
    ]
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    no_signal_frames = [
        frame for frame in placement["players"][0]["frames"][2:] if not any(signal["used"] for signal in frame["signals"])
    ]

    assert len(no_signal_frames) == 78
    assert np.quantile(np.asarray(speeds), 0.90) < 1.0
    assert min(frame["world_xy"][1] for frame in frames[2:]) > -1.0


def test_default_temporal_model_rejects_implausibly_fast_native2d_excursion(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    tracks_payload, keypoints_payload = _fast_native2d_excursion_payload()
    _write_json(tracks_path, tracks_payload)
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, keypoints_payload)

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    frames = rewritten["players"][0]["frames"]
    speeds = [
        np.linalg.norm(np.array(next_frame["world_xy"]) - np.array(frame["world_xy"])) * 30.0
        for frame, next_frame in zip(frames, frames[1:], strict=False)
    ]

    assert np.quantile(np.asarray(speeds), 0.90) < 2.5


def test_apply_placement_cli_help_direct_reference() -> None:
    command_path = "scripts/racketsport/apply_placement.py"
    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert command_path
    assert "--tracks" in completed.stdout
    assert "--placement-out" in completed.stdout
    assert "--stance-phases" in completed.stdout


def _lane_foot_joints(x_px: float, y_px: float, *, conf: float = 9.0) -> list[dict[str, object]]:
    return [
        {"name": name, "x_px": float(x_px), "y_px": float(y_px), "conf": float(conf)}
        for name in (
            "left_ankle",
            "left_heel",
            "left_big_toe",
            "left_small_toe",
            "right_ankle",
            "right_heel",
            "right_big_toe",
            "right_small_toe",
        )
    ]


def _lane_native2d_payload(player_pixels: dict[int, dict[int, tuple[float, float]]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "clip": "lane_p_synthetic",
        "model": "rtmw-test",
        "players": [
            {
                "id": player_id,
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "t": frame_idx / 30.0,
                        "joints": _lane_foot_joints(x_px, y_px),
                    }
                    for frame_idx, (x_px, y_px) in sorted(frames.items())
                ],
            }
            for player_id, frames in sorted(player_pixels.items())
        ],
    }


def _lane_tracks_payload(players: list[dict[str, object]], *, fps: float = 30.0) -> dict[str, object]:
    return {"schema_version": 1, "fps": fps, "players": players, "rally_spans": []}


def _lane_frame(frame_idx: int, x_world: float, y_world: float, *, conf: float = 0.9) -> dict[str, object]:
    x_px = 1000.0 + 100.0 * x_world
    y_px = 1000.0 + 100.0 * y_world
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "bbox": [x_px - 20.0, y_px - 100.0, x_px + 20.0, y_px],
        "world_xy": [x_world, y_world],
        "conf": conf,
    }


def test_sidecar_identity_reassociation_blocks_outdoor_integer_id_swap(tmp_path: Path) -> None:
    frame_indices = list(range(12))
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [
                {
                    "id": 1,
                    "side": "near",
                    "role": "left",
                    "frames": [_lane_frame(idx, 0.25, -4.8) for idx in frame_indices],
                },
                {
                    "id": 4,
                    "side": "far",
                    "role": "right",
                    "frames": [_lane_frame(idx, 1.4, 5.0) for idx in frame_indices],
                },
            ]
        ),
    )
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _lane_native2d_payload({4: {idx: (1025.0, 520.0) for idx in frame_indices}}))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=250.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    far_player = next(player for player in rewritten["players"] if player["id"] == 4)
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    mapping = placement["summary"]["sidecar_identity"]["native2d"]["players"]["4"]

    assert all(frame["world_xy"][1] > 0.0 for frame in far_player["frames"])
    assert mapping["mapped_track_id"] == 1
    assert mapping["integer_match"] is False
    assert placement["summary"]["sidecar_identity"]["native2d"]["totals"]["reassigned_obs"] == 12


def test_aligned_sidecar_identity_mapping_keeps_using_keypoints(tmp_path: Path) -> None:
    frame_indices = list(range(12))
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [
                {
                    "id": 1,
                    "side": "near",
                    "role": "left",
                    "frames": [_lane_frame(idx, 0.0, -4.0) for idx in frame_indices],
                }
            ]
        ),
    )
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _lane_native2d_payload({1: {idx: (1010.0, 590.0) for idx in frame_indices}}))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=250.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    assert placement["summary"]["sidecar_identity"]["native2d"]["players"]["1"]["mapped_track_id"] == 1
    assert placement["summary"]["sidecar_identity"]["native2d"]["totals"]["used_obs"] == 12
    assert rewritten["players"][0]["frames"][0]["world_xy"] == pytest.approx([0.1, -4.1], abs=0.18)


def test_unmappable_sidecar_player_is_dropped_fail_closed(tmp_path: Path) -> None:
    frame_indices = list(range(12))
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [{"id": 1, "side": "near", "role": "left", "frames": [_lane_frame(idx, 0.0, -4.0) for idx in frame_indices]}]
        ),
    )
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _lane_native2d_payload({434: {idx: (300.0 + idx * 40.0, 300.0) for idx in frame_indices}}))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
    )

    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    player_diag = placement["summary"]["sidecar_identity"]["native2d"]["players"]["434"]
    assert player_diag["mapped_track_id"] is None
    assert player_diag["dropped"] is True
    assert placement["summary"]["sidecar_identity"]["native2d"]["totals"]["dropped_obs"] == 12
    assert placement["summary"]["source_counts"].get("native2d", 0) == 0


def test_sidecar_per_frame_validation_drops_pixel_outside_mapped_bbox(tmp_path: Path) -> None:
    frame_indices = list(range(12))
    sidecar_pixels = {idx: (1000.0, 600.0) for idx in frame_indices}
    sidecar_pixels[7] = (1000.0, 1500.0)
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [{"id": 1, "side": "near", "role": "left", "frames": [_lane_frame(idx, 0.0, -4.0) for idx in frame_indices]}]
        ),
    )
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _lane_native2d_payload({1: sidecar_pixels}))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=250.0),
    )

    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    frame7 = next(frame for frame in placement["players"][0]["frames"] if frame["frame_idx"] == 7)
    native_signal = next(signal for signal in frame7["signals"] if signal["name"] == "native2d")
    assert native_signal["used"] is False
    assert native_signal["reason"] == "identity_pixel_mismatch"
    assert native_signal["sidecar_player_id"] == 1
    assert native_signal["mapped_player_id"] == 1
    assert placement["summary"]["sidecar_identity"]["native2d"]["dropped_identity_mismatch_by_player"]["1"] == 1


def test_raw_homography_convention_does_not_undistort_distorted_intrinsics(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    undistorted_calibration_path = tmp_path / "court_calibration_undistorted.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [{"id": 1, "side": "far", "role": "right", "frames": [_lane_frame(0, 0.5, 0.5)]}]
        ),
    )
    raw_calibration = _calibration_payload(dist=[0.2, -0.05, 0.002, 0.001])
    _write_json(calibration_path, raw_calibration)

    rewrite_tracks_with_placement(tracks_path=tracks_path, calibration_path=calibration_path, placement_path=placement_path)

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    assert placement["homography_pixel_convention"] == "raw_pixels"
    assert placement["undistort_applied"] is False
    assert rewritten["players"][0]["frames"][0]["world_xy"] == pytest.approx([0.5, 0.5], abs=1e-6)

    undistorted_calibration = dict(raw_calibration)
    undistorted_calibration["homography_pixel_convention"] = "undistorted_pixels"
    _write_json(tracks_path, _lane_tracks_payload([{"id": 1, "side": "far", "role": "right", "frames": [_lane_frame(0, 0.5, 0.5)]}]))
    _write_json(undistorted_calibration_path, undistorted_calibration)
    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=undistorted_calibration_path,
        placement_path=placement_path,
    )
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    assert placement["homography_pixel_convention"] == "undistorted_pixels"
    assert placement["undistort_applied"] is True


def test_net_guard_clamps_unsupported_gap_that_free_running_rts_would_cross(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    frames = [_lane_frame(0, 0.0, -0.40), _lane_frame(1, 0.0, -0.20), _lane_frame(2, 0.0, -0.05)]
    for frame_idx in range(3, 21):
        frame = _lane_frame(frame_idx, 0.0, 9.0)
        frame["world_xy"] = [0.0, -0.05]
        frames.append(frame)
    measurements = {
        0: ([0.0, -0.40], [[0.01, 0.0], [0.0, 0.01]], False),
        1: ([0.0, -0.20], [[0.01, 0.0], [0.0, 0.01]], False),
        2: ([0.0, -0.05], [[0.01, 0.0], [0.0, 0.01]], False),
    }
    unconstrained = kalman_rts_smooth(measurements, frame_indices=list(range(21)), fps=30.0, process_noise_mps2=2.0)
    assert max(frame.xy[1] for idx, frame in unconstrained.items() if idx > 2) > 0.0
    _write_json(
        tracks_path,
        _lane_tracks_payload([{"id": 1, "side": "near", "role": "left", "frames": frames}]),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        config=PlacementConfig(process_noise_mps2=2.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    assert all(frame["world_xy"][1] <= 0.0 for frame in rewritten["players"][0]["frames"])
    assert placement["summary"]["boundary_guards"]["players"]["1"]["net_gap_clamped_frames"] > 0


def test_side_and_role_recomputed_from_post_placement_track(tmp_path: Path) -> None:
    frame_indices = list(range(12))
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [
                {
                    "id": 7,
                    "side": "far",
                    "role": "right",
                    "frames": [_lane_frame(idx, -0.75, -4.5) for idx in frame_indices],
                }
            ]
        ),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(tracks_path=tracks_path, calibration_path=calibration_path, placement_path=placement_path)

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    player = rewritten["players"][0]
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    consistency = placement["summary"]["side_quadrant_consistency"]["players"]["7"]
    assert player["side"] == "near"
    assert player["role"] == "left"
    assert player["side_original"] == "far"
    assert player["role_original"] == "right"
    assert player["side_source"] == "placement_recomputed"
    assert consistency["side_label_original"] == "far"
    assert consistency["side_recomputed"] == "near"
    assert consistency["first_label_mismatch_frame_idx"] == 0


def test_segmented_smoothing_and_crossfade_prevent_outdoor_shape_teleport(tmp_path: Path) -> None:
    fps = 60.0
    frames = []
    sidecar_pixels: dict[int, tuple[float, float]] = {}
    for frame_idx in range(90):
        if frame_idx < 20:
            y_world = -5.0 + frame_idx * 0.055
            frame = _lane_frame(frame_idx, 0.2, y_world)
            frame["t"] = frame_idx / fps
            sidecar_pixels[frame_idx] = (1020.0, 1000.0 + 100.0 * y_world)
        elif frame_idx < 72:
            frame = _lane_frame(frame_idx, 0.2, 9.0)
            frame["world_xy"] = [0.2, -3.95]
            frame["t"] = frame_idx / fps
        else:
            y_world = -4.45 + (frame_idx - 72) * 0.005
            frame = _lane_frame(frame_idx, 0.2, y_world)
            frame["t"] = frame_idx / fps
            sidecar_pixels[frame_idx] = (1020.0, 1000.0 + 100.0 * y_world)
        frames.append(frame)

    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    keypoints_path = tmp_path / "keypoints_2d.json"
    placement_path = tmp_path / "placement.json"
    _write_json(
        tracks_path,
        _lane_tracks_payload([{"id": 2, "side": "near", "role": "right", "frames": frames}], fps=fps),
    )
    _write_json(calibration_path, _calibration_payload())
    _write_json(keypoints_path, _lane_native2d_payload({2: sidecar_pixels}))

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        native2d_keypoints_path=keypoints_path,
        config=PlacementConfig(keypoint_base_sigma_px=1.0, bbox_base_sigma_px=250.0, process_noise_mps2=2.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    written = [frame["world_xy"] for frame in rewritten["players"][0]["frames"]]
    displacements = [
        float(np.linalg.norm(np.array(right) - np.array(left)))
        for left, right in zip(written, written[1:], strict=False)
    ]
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    placement_frames = {frame["frame_idx"]: frame for frame in placement["players"][0]["frames"]}
    resume_errors = [
        float(
            np.linalg.norm(
                np.array(placement_frames[idx]["smoothed_world_xy"]) - np.array(placement_frames[idx]["fused_world_xy"])
            )
        )
        for idx in range(72, 82)
    ]

    assert max(displacements) <= PlacementConfig().max_written_speed_mps / fps + 1e-6
    assert all(point[1] < 0.0 for point in written)
    assert max(resume_errors) < 0.3
    assert any(placement_frames[idx].get("gap_interpolated") is True for idx in range(20, 72))
    assert not any(placement_frames[idx].get("gap_hold") is True for idx in range(20, 72))
    assert placement["summary"]["smoothing_guards"]["players"]["2"]["fallback_transition_blends"] > 0


def test_default_process_noise_tracks_two_mps_sinusoidal_walker_at_60fps() -> None:
    fps = 60.0
    frame_indices = list(range(180))
    truth = {idx: [np.sin(2.0 * idx / fps), -3.0] for idx in frame_indices}
    measurements = {
        idx: (xy, [[0.01, 0.0], [0.0, 0.01]], False)
        for idx, xy in truth.items()
    }

    smoothed = kalman_rts_smooth(measurements, frame_indices=frame_indices, fps=fps)
    rms = float(
        np.sqrt(
            np.mean(
                [
                    np.sum((np.array(smoothed[idx].xy) - np.array(truth[idx])) ** 2)
                    for idx in frame_indices
                ]
            )
        )
    )

    assert PlacementConfig().process_noise_mps2 == pytest.approx(2.0)
    assert rms < 0.15


def test_camera_motion_transforms_bbox_and_native2d_pixels_before_homography(tmp_path: Path) -> None:
    translation_px = 20.0
    tracks_static = tmp_path / "static" / "tracks.json"
    tracks_motion = tmp_path / "motion" / "tracks.json"
    calibration_static = tmp_path / "static" / "court_calibration.json"
    calibration_motion = tmp_path / "motion" / "court_calibration.json"
    placement_static = tmp_path / "static" / "placement.json"
    placement_motion = tmp_path / "motion" / "placement.json"
    native_static = tmp_path / "static" / "keypoints_2d.json"
    native_motion = tmp_path / "motion" / "keypoints_2d.json"
    camera_motion_path = tmp_path / "camera_motion.json"
    tracks = _tracks_payload()
    tracks["players"][0]["frames"] = tracks["players"][0]["frames"][:3]  # type: ignore[index]
    native = _native2d_payload_for_frames([0, 1, 2])
    tracks_static.parent.mkdir(parents=True, exist_ok=True)
    tracks_motion.parent.mkdir(parents=True, exist_ok=True)
    for path in (tracks_static, tracks_motion):
        _write_json(path, tracks)
    for path in (calibration_static, calibration_motion):
        _write_json(path, _calibration_payload())
    for path in (native_static, native_motion):
        _write_json(path, native)
    _write_json(
        camera_motion_path,
        _camera_motion_payload(
            compensated_frames={0: [[1.0, 0.0, translation_px], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            uncompensated_frames={1},
        ),
    )

    rewrite_tracks_with_placement(
        tracks_path=tracks_static,
        calibration_path=calibration_static,
        placement_path=placement_static,
        native2d_keypoints_path=native_static,
        config=PlacementConfig(vote_min=1),
    )
    rewrite_tracks_with_placement(
        tracks_path=tracks_motion,
        calibration_path=calibration_motion,
        placement_path=placement_motion,
        native2d_keypoints_path=native_motion,
        camera_motion_path=camera_motion_path,
        config=PlacementConfig(vote_min=1),
    )

    static = json.loads(placement_static.read_text(encoding="utf-8"))
    motion = json.loads(placement_motion.read_text(encoding="utf-8"))
    static_frames = {frame["frame_idx"]: frame for frame in static["players"][0]["frames"]}
    motion_frames = {frame["frame_idx"]: frame for frame in motion["players"][0]["frames"]}
    expected_world_dx = translation_px / 100.0

    assert motion_frames[0]["fused_world_xy"][0] - static_frames[0]["fused_world_xy"][0] == pytest.approx(expected_world_dx)
    assert motion_frames[1]["fused_world_xy"] == pytest.approx(static_frames[1]["fused_world_xy"])
    assert motion_frames[2]["fused_world_xy"] == pytest.approx(static_frames[2]["fused_world_xy"])

    motion_signals = {signal["name"]: signal for signal in motion_frames[0]["signals"]}
    static_signals = {signal["name"]: signal for signal in static_frames[0]["signals"]}
    assert motion_signals["bbox"]["xy"][0] - static_signals["bbox"]["xy"][0] == pytest.approx(expected_world_dx)
    assert motion_signals["native2d"]["xy"][0] - static_signals["native2d"]["xy"][0] == pytest.approx(expected_world_dx)
    assert motion["summary"]["camera_motion_frames_used"] == 1
    assert motion["summary"]["camera_motion_frames_uncompensated"] == 2
    assert motion["provenance"]["camera_motion_frames_used"] == 1
    assert motion["provenance"]["camera_motion_frames_uncompensated"] == 2


def test_omitted_camera_motion_keeps_placement_payload_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(placement_module, "_utc_now", lambda: "2026-07-04T00:00:00+00:00")
    for dirname in ("omitted", "explicit_none"):
        run = tmp_path / dirname
        run.mkdir(parents=True, exist_ok=True)
        _write_json(run / "tracks.json", _tracks_payload())
        _write_json(run / "court_calibration.json", _calibration_payload())

    rewrite_tracks_with_placement(
        tracks_path=tmp_path / "omitted" / "tracks.json",
        calibration_path=tmp_path / "omitted" / "court_calibration.json",
        placement_path=tmp_path / "omitted" / "placement.json",
    )
    rewrite_tracks_with_placement(
        tracks_path=tmp_path / "explicit_none" / "tracks.json",
        calibration_path=tmp_path / "explicit_none" / "court_calibration.json",
        placement_path=tmp_path / "explicit_none" / "placement.json",
        camera_motion_path=None,
    )

    assert (tmp_path / "omitted" / "placement.json").read_bytes() == (tmp_path / "explicit_none" / "placement.json").read_bytes()
    assert (tmp_path / "omitted" / "tracks.json").read_bytes() == (tmp_path / "explicit_none" / "tracks.json").read_bytes()


def _max_written_speed_mps(tracks_payload: dict[str, object]) -> float:
    fps = float(tracks_payload["fps"])
    max_speed = 0.0
    for player in tracks_payload["players"]:  # type: ignore[index]
        frames = player["frames"]  # type: ignore[index]
        for left, right in zip(frames, frames[1:], strict=False):
            left_xy = np.asarray(left["world_xy"], dtype=float)
            right_xy = np.asarray(right["world_xy"], dtype=float)
            left_idx = int(left.get("frame_idx", round(float(left["t"]) * fps)))
            right_idx = int(right.get("frame_idx", round(float(right["t"]) * fps)))
            dt = max((right_idx - left_idx) / fps, 1.0 / fps)
            max_speed = max(max_speed, float(np.linalg.norm(right_xy - left_xy) / dt))
    return max_speed


def test_interior_gap_under_reacquisition_speed_is_interpolated_and_speed_capped(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    frames = [_lane_frame(0, 0.0, -6.0)]
    for frame_idx in range(1, 20):
        frame = _lane_frame(frame_idx, 0.0, 9.0)
        frame["world_xy"] = [0.0, -6.0]
        frames.append(frame)
    frames.append(_lane_frame(20, 0.0, -1.0))
    _write_json(
        tracks_path,
        _lane_tracks_payload([{"id": 1, "side": "near", "role": "left", "frames": frames}], fps=30.0),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(tracks_path=tracks_path, calibration_path=calibration_path, placement_path=placement_path)

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    placement_frames = {frame["frame_idx"]: frame for frame in placement["players"][0]["frames"]}
    interior = [placement_frames[idx] for idx in range(1, 20)]

    assert all(frame.get("gap_interpolated") is True for frame in interior)
    assert not any(frame.get("gap_hold") is True for frame in interior)
    assert _max_written_speed_mps(rewritten) <= 8.0 + 1e-6
    assert _max_written_speed_mps(rewritten) <= 10.0
    assert placement["summary"].get("gap_reacquisition_speed_violations", []) == []


def test_fast_reacquisition_gap_holds_records_violation_and_transitions_under_speed_cap(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    frames = [_lane_frame(0, 0.0, -6.0)]
    for frame_idx in range(1, 18):
        frame = _lane_frame(frame_idx, 0.0, 9.0)
        frame["world_xy"] = [0.0, -6.0]
        frames.append(frame)
    for frame_idx in range(18, 70):
        frames.append(_lane_frame(frame_idx, 0.0, 6.0))
    _write_json(
        tracks_path,
        _lane_tracks_payload([{"id": 1, "side": "near", "role": "left", "frames": frames}], fps=30.0),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(tracks_path=tracks_path, calibration_path=calibration_path, placement_path=placement_path)

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    placement_frames = {frame["frame_idx"]: frame for frame in placement["players"][0]["frames"]}
    violations = placement["summary"]["gap_reacquisition_speed_violations"]

    assert all(placement_frames[idx].get("gap_hold") is True for idx in range(1, 18))
    assert not any(placement_frames[idx].get("gap_interpolated") is True for idx in range(1, 18))
    assert len(violations) == 1
    assert violations[0]["player_id"] == 1
    assert violations[0]["gap_start_frame"] == 1
    assert violations[0]["gap_end_frame"] == 17
    assert violations[0]["displacement_m"] == pytest.approx(12.0, abs=1e-6)
    assert violations[0]["implied_speed_mps"] == pytest.approx(20.0, abs=1e-6)
    assert _max_written_speed_mps(rewritten) <= 8.0 + 1e-6


def test_global_written_speed_invariant_covers_divergence_snaps_and_gaps(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    gap_frames = [_lane_frame(0, -2.0, -6.0)]
    for frame_idx in range(1, 18):
        frame = _lane_frame(frame_idx, -2.0, 9.0)
        frame["world_xy"] = [-2.0, -6.0]
        gap_frames.append(frame)
    for frame_idx in range(18, 70):
        gap_frames.append(_lane_frame(frame_idx, 2.0, 6.0))
    snap_frames = []
    for frame_idx in range(70):
        y_world = -5.5 if frame_idx < 25 else -1.5
        snap_frames.append(_lane_frame(frame_idx, 0.5, y_world))
    _write_json(
        tracks_path,
        _lane_tracks_payload(
            [
                {"id": 1, "side": "near", "role": "left", "frames": gap_frames},
                {"id": 2, "side": "near", "role": "right", "frames": snap_frames},
            ],
            fps=30.0,
        ),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(
        tracks_path=tracks_path,
        calibration_path=calibration_path,
        placement_path=placement_path,
        config=PlacementConfig(stance_min_duration_s=99.0),
    )

    rewritten = json.loads(tracks_path.read_text(encoding="utf-8"))
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    smoothing_players = placement["summary"]["smoothing_guards"]["players"]

    assert smoothing_players["2"]["divergence_snap_frames"] > 0
    assert smoothing_players["1"]["fallback_transition_blends"] > 0
    assert placement["summary"]["gap_reacquisition_speed_violations"]
    assert _max_written_speed_mps(rewritten) <= 8.0 + 1e-6


def test_gap_interpolation_still_uses_net_crossing_guard_for_opposite_side_endpoints(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    placement_path = tmp_path / "placement.json"
    frames = [_lane_frame(0, 0.0, -0.4)]
    for frame_idx in range(1, 20):
        frame = _lane_frame(frame_idx, 0.0, 9.0)
        frame["world_xy"] = [0.0, -0.4]
        frames.append(frame)
    frames.append(_lane_frame(20, 0.0, 0.4))
    _write_json(
        tracks_path,
        _lane_tracks_payload([{"id": 1, "side": "near", "role": "left", "frames": frames}], fps=30.0),
    )
    _write_json(calibration_path, _calibration_payload())

    rewrite_tracks_with_placement(tracks_path=tracks_path, calibration_path=calibration_path, placement_path=placement_path)

    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    placement_frames = {frame["frame_idx"]: frame for frame in placement["players"][0]["frames"]}

    assert all(placement_frames[idx].get("gap_interpolated") is True for idx in range(1, 20))
    assert placement_frames[10]["smoothed_world_xy"][1] == pytest.approx(-PlacementConfig().net_clamp_epsilon_m)
    assert placement["summary"]["boundary_guards"]["players"]["1"]["net_gap_clamped_frames"] > 0
