from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
                    {"t": 0.0, "bbox": [990.0, 900.0, 1010.0, 1000.0], "world_xy": [0.0, 0.0], "conf": 0.9},
                    {"t": 1.0 / 30.0, "bbox": [990.0, 900.0, 1010.0, 1000.0], "world_xy": [0.0, 0.0], "conf": 0.9},
                    {"t": 2.0 / 30.0, "bbox": [990.0, 900.0, 1010.0, 1000.0], "world_xy": [0.0, 0.0], "conf": 0.9},
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
                    for frame_idx in range(3)
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
                    for frame_idx in range(3)
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
    assert parsed.summary.source_counts["native2d"] == 3


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
