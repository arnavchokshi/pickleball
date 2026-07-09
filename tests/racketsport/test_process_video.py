from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import process_video
from scripts.racketsport.remote_body_dispatch import RemoteBodyDispatchError, RemoteBodyDispatchResult
from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _court_corners_payload(width: int = 960, height: int = 540) -> dict[str, Any]:
    return {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [100.0, 450.0],
                        "near_right": [860.0, 450.0],
                        "far_right": [700.0, 150.0],
                        "far_left": [260.0, 150.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": [width, height],
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }


def _court_calibration_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 0.0], "camera_height_m": 1.5},
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": minimal_calibration_image_pts(),
        "world_pts": minimal_calibration_world_pts(),
    }


def _low_angle_not_fully_visible_calibration_payload() -> dict[str, Any]:
    payload = _court_calibration_payload()
    payload["image_size"] = [960, 540]
    payload["image_pts"] = [
        [120.0, 510.0],
        [840.0, 510.0],
        [720.0, 190.0],
        [-45.0, 190.0],
    ]
    payload["world_pts"] = [
        [-3.048, -6.7056, 0.0],
        [3.048, -6.7056, 0.0],
        [3.048, 6.7056, 0.0],
        [-3.048, 6.7056, 0.0],
    ]
    payload["extrinsics"] = {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [0.0, 0.0, 0.0],
        "camera_height_m": 0.75,
    }
    return payload


def _match_stats_court_zones_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "zones": {
            "court": [[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0]],
            "near_nvz": [[-3.0, -2.0], [3.0, -2.0], [3.0, 0.0], [-3.0, 0.0]],
            "far_nvz": [[-3.0, 0.0], [3.0, 0.0], [3.0, 2.0], [-3.0, 2.0]],
        },
    }


def _match_stats_placement_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": 1.0,
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": 0, "t": 0.0, "smoothed_world_xy": [-1.0, -5.0]},
                    {"frame_idx": 1, "t": 1.0, "smoothed_world_xy": [0.0, -3.0]},
                    {"frame_idx": 2, "t": 2.0, "smoothed_world_xy": [1.0, -1.0]},
                ],
            }
        ],
    }


def _external_metric_calibration_payload(*, dist: list[float] | None = None, source: str = "metric_15pt_reviewed") -> dict[str, Any]:
    """A court_calibration.json-shaped payload as ExternalCalibrationRunner would
    write it -- these process_video-level tests fake orchestrator.run_pipeline
    entirely, so this only needs to be schema-plausible, not metric-complete
    (that stricter 15-keypoint completeness rule is exercised against the real
    ExternalCalibrationRunner in test_orchestrator_spine.py)."""

    payload = _court_calibration_payload()
    payload["intrinsics"] = {"fx": 1391.18, "fy": 1391.18, "cx": 960.0, "cy": 540.0, "dist": dist or [], "source": source}
    return payload


def _tracks_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {"id": 1, "side": "near", "role": "left", "frames": [{"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [1.0, 2.0], "conf": 0.9}]},
            {"id": 2, "side": "far", "role": "right", "frames": [{"t": 0.0, "bbox": [300.0, 120.0, 380.0, 320.0], "world_xy": [-1.0, -2.0], "conf": 0.85}]},
        ],
        "rally_spans": [],
    }


def _tracks_payload_with_dead_time() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [1.0, 2.0], "conf": 0.9},
                    {"t": 5.0, "bbox": [110.0, 100.0, 210.0, 300.0], "world_xy": [1.1, 2.0], "conf": 0.9},
                ],
            }
        ],
        "rally_spans": [],
    }


def _ball_track_payload(*, frame_count: int = 1, fps: float = 30.0) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": fps,
        "source": "wasb",
        "frames": [
            {"t": index / fps, "xy": [400.0 + index, 300.0], "conf": 0.9, "visible": True, "approx": False}
            for index in range(frame_count)
        ],
        "bounces": [],
    }


def _ball_candidates_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_candidates",
        "source": "wasb",
        "source_mode": "wasb_test",
        "fps": 30.0,
        "primary_output": "ball_track.json",
        "max_candidates_per_frame": 5,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "frames": [
            {
                "frame": 0,
                "candidates": [
                    {"xy": [400.0, 300.0], "score": 0.95, "source_detector": "wasb_concomp"},
                ],
            }
        ],
    }


def _wolverine_fps_mismatch_ball_track_payload(*, frame_count: int = 300) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "wasb",
        "frames": [
            {
                "t": index / 60.0,
                "xy": [354.676941 + index * 0.1, 307.531799 + index * 0.1],
                "conf": 0.8,
                "visible": True,
                "approx": False,
            }
            for index in range(frame_count)
        ],
        "bounces": [],
    }


def _lane_a_skeleton_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "legacy_body65_joints",
        "joint_names": ["left_wrist", "right_wrist", "left_hip", "right_hip", "left_ankle", "right_ankle"],
        "preview_only": False,
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "transl_world": [0.2, -0.1, 0.0],
                        "joints_world": [
                            [-0.2, 0.0, 1.2],
                            [0.2, 0.0, 1.2],
                            [0.1, -0.1, 1.0],
                            [0.3, -0.1, 1.0],
                            [0.08, -0.1, 0.05],
                            [0.32, -0.1, 0.05],
                        ],
                        "joint_conf": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                ],
            }
        ],
        "provenance": {"lane": "A"},
    }


def _sam3d_skeleton_payload() -> dict[str, Any]:
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    joints[5] = [-0.2, 0.0, 1.4]
    joints[6] = [0.2, 0.0, 1.4]
    joints[7] = [-0.35, 0.0, 1.2]
    joints[8] = [0.35, 0.0, 1.2]
    joints[9] = [-0.15, -0.1, 1.0]
    joints[10] = [0.15, -0.1, 1.0]
    joints[11] = [-0.12, -0.1, 0.5]
    joints[12] = [0.12, -0.1, 0.5]
    joints[13] = [-0.1, -0.1, 0.05]
    joints[14] = [0.1, -0.1, 0.05]
    joints[41] = [0.55, 0.0, 1.1]
    joints[62] = [-0.55, 0.0, 1.1]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": joint_names,
        "preview_only": False,
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "transl_world": [0.2, -0.1, 0.0],
                        "joints_world": joints,
                        "joint_conf": [1.0] * 70,
                    }
                ],
            }
        ],
        "provenance": {
            "lane": "BODY_TIER2",
            "source": "sam3d_body_joints",
            "model_family": "sam3dbody_world_joints",
            "protected_eval_labels_used": False,
        },
    }


def _contact_windows_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.0,
                "frame": 0,
                "player_id": 1,
                "confidence": 0.8,
                "sources": {"wrist_vel": 0.85, "ball_inflection": 0.75},
                "window": {"t0": 0.0, "t1": 0.05, "importance": 0.8},
            }
        ],
    }


def _virtual_world_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "length_m": 13.4112,
            "width_m": 6.096,
            "line_segments": {
                "near_baseline": [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0]],
                "left_sideline": [[-3.048, -6.7056, 0.0], [-3.048, 6.7056, 0.0]],
            },
            "net": {
                "endpoints": [[-3.048, 0.0, 0.914], [3.048, 0.0, 0.914]],
                "center_height_m": 0.86,
                "post_height_m": 0.914,
            },
        },
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "representation": "joints",
                "frames": [
                    {
                        "t": 0.0,
                        "track_world_xy": [0.0, -2.0],
                        "joints_world": [[0.0, -2.0, 1.0]],
                        "joint_conf": [0.9],
                    },
                    {
                        "t": 1.0,
                        "track_world_xy": [0.1, -1.8],
                        "joints_world": [[0.1, -1.8, 1.0]],
                        "joint_conf": [0.9],
                    },
                ],
            }
        ],
        "ball": {
            "source": "physics_filled",
            "frames": [
                {"t": 0.0, "xy": [400.0, 300.0], "conf": 0.9, "visible": True, "world_xyz": [0.0, -1.0, 0.5]},
                {"t": 1.0, "xy": [420.0, 310.0], "conf": 0.2, "visible": False, "world_xyz": [0.1, -0.8, 0.4]},
            ],
        },
        "paddles": [],
        "summary": {"player_count": 1, "ball_frame_count": 2, "paddle_frame_count": 0},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_video(path: Path, *, frame_count: int = 5, fps: float = 30.0) -> None:
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    for _ in range(frame_count):
        writer.write(np.zeros((540, 960, 3), dtype="uint8"))
    writer.release()


def _base_options(tmp_path: Path, *, video: Path, court_corners: Path | None) -> process_video.PipelineOptions:
    return process_video.PipelineOptions(
        video=video,
        clip="test_clip",
        run_dir=tmp_path / "run",
        court_corners=court_corners,
        skip_ball=True,
        no_gpu=True,
        vite_allow_root=tmp_path,
    )


def test_ingest_stage_writes_frame_times_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=3, fps=30.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    calls: list[tuple[Path, Path]] = []

    def _fake_write_frame_time_table(path: Path, out_path: Path) -> dict[str, Any]:
        calls.append((path, out_path))
        payload = {
            "artifact_type": "racketsport_frame_times",
            "provenance": "ffprobe_pts",
            "frames": [{"frame_idx": 0, "pts_s": 0.0}],
        }
        _write_json(out_path, payload)
        return payload

    monkeypatch.setattr(process_video, "write_frame_time_table", _fake_write_frame_time_table)

    outcome = pipeline._stage_ingest()

    assert calls == [(options.clip_dir / "source.mp4", options.clip_dir / "frame_times.json")]
    assert "frame_times.json" in outcome.artifacts
    assert (options.clip_dir / "frame_times.json").is_file()
    assert "wrote frame_times.json from ffprobe PTS" in outcome.notes


def _fake_run_pipeline_factory(write_for_stage: dict[str, dict[str, Any]]):
    """Build a stand-in for orchestrator.run_pipeline that writes fixture
    artifacts for a given stage instead of running any real model."""

    calls: list[str] = []

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        calls.append(stage)
        payload = write_for_stage.get(stage)
        if payload is None:
            return {"status": "fail", "stages": [{"stage": stage, "status": "fail", "notes": [f"no fixture for {stage}"]}]}
        for filename, content in payload.items():
            _write_json(Path(run_dir) / filename, content)
        return {"status": "pass", "stages": [{"stage": stage, "status": "ran", "notes": []}]}

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def _fake_run_pipeline_capturing_kwargs(write_for_stage: dict[str, dict[str, Any]]):
    """Like _fake_run_pipeline_factory, but also records every kwarg passed
    (notably `runners=`) so tests can assert *which* runner process_video wired
    in for the calibration stage without needing a real video/court evidence."""

    calls: list[dict[str, Any]] = []

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        calls.append({"clip": clip, "inputs_dir": inputs_dir, "run_dir": run_dir, "stage": stage, **kwargs})
        payload = write_for_stage.get(stage)
        if payload is None:
            return {"status": "fail", "stages": [{"stage": stage, "status": "fail", "notes": [f"no fixture for {stage}"]}]}
        for filename, content in payload.items():
            _write_json(Path(run_dir) / filename, content)
        return {"status": "pass", "stages": [{"stage": stage, "status": "ran", "notes": []}]}

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


# ---------------------------------------------------------------------------
# court-corners / capture-sidecar contract
# ---------------------------------------------------------------------------


def test_read_declared_court_corners_requires_image_size(tmp_path: Path) -> None:
    payload = _court_corners_payload()
    del payload["annotation"]["items"][0]["image_size"]
    path = tmp_path / "court_corners.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="image_size"):
        process_video._read_declared_court_corners(path)


def test_read_declared_court_corners_requires_all_four_corners(tmp_path: Path) -> None:
    payload = _court_corners_payload()
    del payload["annotation"]["items"][0]["court_corners"]["far_left"]
    path = tmp_path / "court_corners.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="far_left"):
        process_video._read_declared_court_corners(path)


def test_capture_sidecar_from_court_corners_is_schema_valid(tmp_path: Path) -> None:
    from threed.racketsport.schemas import CaptureSidecar

    path = tmp_path / "court_corners.json"
    _write_json(path, _court_corners_payload(width=960, height=540))

    sidecar = process_video._capture_sidecar_from_court_corners(path, fps=30.0)
    validated = CaptureSidecar.model_validate(sidecar)
    assert validated.resolution == (960, 540)
    assert len(validated.manual_court_taps) == 4


def test_capture_sidecar_from_auto_predicted_court_corners_is_unverified_preview(tmp_path: Path) -> None:
    from threed.racketsport.schemas import CaptureSidecar

    payload = _court_corners_payload(width=960, height=540)
    item = payload["annotation"]["items"][0]
    item["source"] = "court_detector_v2:selected_hypothesis=hypothesis_0001"
    item["status"] = "auto_preview_unverified"
    item["review_status"] = "auto_predicted_unreviewed"
    item["not_cal3_verified"] = True
    path = tmp_path / "court_corners.json"
    _write_json(path, payload)

    sidecar = process_video._capture_sidecar_from_court_corners(path, fps=30.0)
    validated = CaptureSidecar.model_validate(sidecar)

    assert validated.capture_quality.grade == "poor"
    assert "process_video_auto_court_corners_preview" in validated.capture_quality.reasons
    assert "manual_taps_seeded_from_unverified_detector" in validated.capture_quality.reasons


def test_capture_sidecar_from_court_corners_feeds_real_calibration_solve(tmp_path: Path) -> None:
    from threed.racketsport.court_calibration import calibration_from_manual_taps

    path = tmp_path / "court_corners.json"
    _write_json(path, _court_corners_payload(width=960, height=540))
    sidecar = process_video._capture_sidecar_from_court_corners(path, fps=30.0)
    sidecar_path = tmp_path / "capture_sidecar.json"
    _write_json(sidecar_path, sidecar)

    calibration = calibration_from_manual_taps(sidecar_path, sport="pickleball")
    assert calibration.reprojection_error_px.median >= 0.0


def test_stage_calibration_can_seed_missing_taps_from_preview_auto_corners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    original_sidecar = {
        "schema_version": 1,
        "device_tier": "fallback",
        "device_model": "iphone-test",
        "fps": 30,
        "format": "hevc",
        "resolution": [960, 540],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {"fx": 1152.0, "fy": 1152.0, "cx": 480.0, "cy": 270.0, "dist": [], "source": "avfoundation_fov_estimate"},
        "arkit_camera_pose": None,
        "court_plane": None,
        "manual_court_taps": [],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {"grade": "warn", "reasons": ["manual_taps_missing"]},
    }
    sidecar_path = tmp_path / "capture_sidecar.json"
    _write_json(sidecar_path, original_sidecar)

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.capture_sidecar = sidecar_path
    options.allow_auto_court_corners_preview = True
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    auto_corners = _court_corners_payload(width=960, height=540)
    auto_corners["annotation"]["items"][0]["source"] = "auto_white_line_preview"
    monkeypatch.setattr(
        process_video,
        "_auto_court_corners_preview_from_video",
        lambda video_path, out_path: _write_json(out_path, auto_corners) or out_path,
        raising=False,
    )
    monkeypatch.setattr(
        process_video.orchestrator,
        "run_pipeline",
        _fake_run_pipeline_capturing_kwargs({"calibration": {"court_calibration.json": _court_calibration_payload()}}),
    )

    outcome = pipeline._stage_calibration()

    seeded_sidecar = json.loads((options.clip_dir / "capture_sidecar.json").read_text(encoding="utf-8"))
    assert len(seeded_sidecar["manual_court_taps"]) == 4
    assert seeded_sidecar["capture_quality"]["grade"] == "poor"
    assert "auto-court preview" in outcome.notes[0]


def test_stage_tracking_uses_wide_margin_for_auto_court_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(pipeline.clip_dir / "auto_court_corners_preview.json", _court_corners_payload())
    _write_json(pipeline.clip_dir / "court_calibration.json", _court_calibration_payload())

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs({"tracking": {"tracks.json": _tracks_payload()}})
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    outcome = pipeline._stage_tracking()

    assert outcome.status == "ran"
    assert fake_run_pipeline.calls[-1]["court_margin_m"] == pytest.approx(1000.0)
    assert any("auto-court preview" in note for note in outcome.notes)


def test_stage_tracking_blocks_poor_unverified_calibration_when_line_evidence_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    calibration = _court_calibration_payload()
    calibration["intrinsics"]["source"] = "estimated_from_declared_court_corners"
    calibration["capture_quality"] = {
        "grade": "poor",
        "reasons": ["process_video_manual_court_corners", "estimated_intrinsics", "corrected_unverified"],
    }
    _write_json(pipeline.clip_dir / "court_calibration.json", calibration)
    _write_json(
        pipeline.clip_dir / "court_line_evidence.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "source": "img1605_regression",
            "line_observations": [],
            "keypoint_observations": [],
            "net_observations": [],
            "aggregate": {
                "accepted_line_ids": [],
                "rejected_line_ids": [],
                "missing_required_line_ids": ["near_nvz", "far_nvz", "near_centerline", "far_centerline"],
                "missing_required_net_ids": ["top_net"],
                "mean_residual_px": 0.0,
                "p95_residual_px": 0.0,
                "temporal_stability_px": 0.0,
                "auto_calibration_ready": False,
                "reasons": ["missing_required_lines", "missing_required_net"],
            },
        },
    )

    def _should_not_run(**kwargs):  # noqa: ANN001
        raise AssertionError("tracking should be blocked before orchestrator.run_pipeline")

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _should_not_run)

    outcome = pipeline._stage_tracking()

    assert outcome.status == "blocked"
    assert "court_correction_task.json" in outcome.artifacts
    assert any("court_calibration_unverified_or_evidence_not_ready" in note for note in outcome.notes)
    correction = json.loads((pipeline.clip_dir / "court_correction_task.json").read_text(encoding="utf-8"))
    assert correction["court_status"] == "needs_user_correction"
    assert correction["reason"] == "court_calibration_unverified_or_evidence_not_ready"
    assert correction["blocked_downstream"] == [
        "tracking_court_filter",
        "body_world",
        "ball_world",
        "virtual_world_metric",
    ]
    assert correction["calibration"]["capture_quality_grade"] == "poor"
    assert correction["court_line_evidence"]["auto_calibration_ready"] is False


def test_stage_tracking_blocks_when_detector_v2_proposal_not_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(
        pipeline.clip_dir / "court_detector_v2_proposals.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_court_detector_v2_proposals",
            "clip": "clip",
            "source_frame": "frame_000001.jpg",
            "image_size": [960, 540],
            "promoted": False,
            "verified": False,
            "not_cal3_verified": True,
            "promotion_status": "needs_user_input",
            "promotion_blockers": ["self_verification_not_promotable"],
            "selected_hypothesis_id": "hypothesis_0001",
            "hypotheses": [],
            "net_evidence": {},
            "surface_evidence": {},
            "verification": {},
            "needs_user_input": ["near_left_corner"],
        },
    )

    def _should_not_run(**kwargs):  # noqa: ANN001
        raise AssertionError("tracking should be blocked before orchestrator.run_pipeline")

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _should_not_run)

    outcome = pipeline._stage_tracking()

    assert outcome.status == "blocked"
    assert "court_correction_task.json" in outcome.artifacts
    assert any("court_detector_v2_not_promoted" in note for note in outcome.notes)
    correction = json.loads((pipeline.clip_dir / "court_correction_task.json").read_text(encoding="utf-8"))
    assert correction["court_status"] == "needs_user_correction"
    assert correction["reason"] == "court_detector_v2_not_promoted"
    assert correction["blocked_downstream"] == [
        "tracking_court_filter",
        "body_world",
        "ball_world",
        "virtual_world_metric",
    ]
    assert correction["detector_v2"]["promotion_blockers"] == ["self_verification_not_promotable"]


# ---------------------------------------------------------------------------
# --court-calibration consumption (Task #33 CAL-MIGRATION)
# ---------------------------------------------------------------------------


def test_resolved_court_calibration_path_prefers_explicit_flag_over_corners_and_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    metric_path = tmp_path / "metric15.json"
    options = _base_options(tmp_path, video=video, court_corners=tmp_path / "court_corners.json")
    options.capture_sidecar = tmp_path / "capture_sidecar.json"
    options.court_calibration = metric_path
    pipeline = process_video.ProcessVideoPipeline(options)

    assert pipeline._resolved_court_calibration_path() == metric_path


def test_resolved_court_calibration_path_none_when_court_corners_given_and_no_explicit_flag(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=tmp_path / "court_corners.json")
    pipeline = process_video.ProcessVideoPipeline(options)

    # an explicit tap choice must never be silently overridden by auto-discovery.
    assert pipeline._resolved_court_calibration_path() is None


def test_resolved_court_calibration_path_none_when_capture_sidecar_given_and_no_explicit_flag(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.capture_sidecar = tmp_path / "capture_sidecar.json"
    pipeline = process_video.ProcessVideoPipeline(options)

    assert pipeline._resolved_court_calibration_path() is None


def test_resolved_court_calibration_path_auto_discovers_next_to_video_labels_dir(tmp_path: Path) -> None:
    clip_dir = tmp_path / "eval_clips" / "ball" / "some_clip"
    video = clip_dir / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake")
    metric_path = clip_dir / "labels" / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload())

    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)

    assert pipeline._resolved_court_calibration_path() == metric_path


def test_resolved_court_calibration_path_no_auto_discovery_when_labels_file_absent(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)

    assert pipeline._resolved_court_calibration_path() is None


def test_stage_calibration_consumes_explicit_court_calibration_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    metric_path = tmp_path / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0]))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.court_calibration = metric_path
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs(
        {"calibration": {"court_calibration.json": _external_metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0])}}
    )
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    outcome = pipeline._stage_calibration()

    assert outcome.status == "ran"
    assert any("consumed externally-provided" in note for note in outcome.notes)
    assert not any("intrinsics.dist is nonzero" in note for note in outcome.notes)
    assert outcome.metrics["intrinsics_source"] == "metric_15pt_reviewed"
    assert outcome.metrics["intrinsics_dist_nonzero"] is False
    assert (options.clip_dir / "court_calibration.json").is_file()
    # the tap-building path must not have run alongside the external-artifact path.
    assert not (options.clip_dir / "capture_sidecar.json").is_file()

    [call] = fake_run_pipeline.calls  # type: ignore[attr-defined]
    assert call["stage"] == "calibration"
    runner = call["runners"]["calibration"]
    assert isinstance(runner, process_video.orchestrator.ExternalCalibrationRunner)
    assert runner.source_path == metric_path


def test_stage_calibration_court_calibration_flag_takes_precedence_over_court_corners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))
    metric_path = tmp_path / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload())

    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    options.court_calibration = metric_path
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs({"calibration": {"court_calibration.json": _external_metric_calibration_payload()}})
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    outcome = pipeline._stage_calibration()

    assert outcome.status == "ran"
    assert any("consumed externally-provided" in note for note in outcome.notes)
    [call] = fake_run_pipeline.calls  # type: ignore[attr-defined]
    assert "runners" in call  # took the external path, not the --court-corners tap-building path.
    assert not (options.clip_dir / "capture_sidecar.json").is_file()


def test_stage_calibration_flags_nonzero_distortion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    metric_path = tmp_path / "court_calibration_metric15pt.json"
    dist = [-0.30035182958629364, 0.09861181595540636, 0.0, 0.0]
    _write_json(metric_path, _external_metric_calibration_payload(dist=dist))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.court_calibration = metric_path
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs({"calibration": {"court_calibration.json": _external_metric_calibration_payload(dist=dist)}})
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    outcome = pipeline._stage_calibration()

    assert outcome.metrics["intrinsics_dist_nonzero"] is True
    assert any("intrinsics.dist is nonzero" in note for note in outcome.notes)
    assert any("person_fast" in note and "virtual_world" in note for note in outcome.notes)


def test_stage_calibration_accepts_advisory_blocked_status_from_spine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task #45 S1 (process_video.py layer): run_pipeline can legitimately return an
    overall "blocked" status for a single-stage stage="calibration" call even when the
    calibration stage itself succeeded (advisory evidence for a trusted source --
    pipeline_contracts' separate readiness report always reports "not_ready" for a lone
    calibration-only call, since it also checks every later stage's artifacts). That
    must not be treated as a calibration hard-failure here."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))
    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        _write_json(Path(run_dir) / "court_calibration.json", _court_calibration_payload())
        return {
            "status": "blocked",
            "stages": [
                {
                    "stage": "calibration",
                    "status": "ran",
                    "notes": ["ADVISORY (not blocking -- trusted calibration source): automatic court evidence not ready"],
                }
            ],
            "readiness": {"status": "not_ready"},
        }

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    outcome = pipeline._stage_calibration()
    assert outcome.status == "ran"
    assert (options.clip_dir / "court_calibration.json").is_file()


def test_stage_calibration_still_hard_fails_when_spine_reports_a_real_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the fix above: a genuine calibration-stage failure (the
    runner raised / returned a failing StageRun) must still hard-fail -- calibration
    stays the one stage nothing downstream can substitute for."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))
    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        return {
            "status": "fail",
            "stages": [{"stage": "calibration", "status": "fail", "notes": ["calibration failed: no trusted no-tap calibration seed"]}],
        }

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    with pytest.raises(process_video._HardStageFailure, match="no trusted no-tap calibration seed"):
        pipeline._stage_calibration()


def test_stage_calibration_hard_fails_when_court_calibration_path_missing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.court_calibration = tmp_path / "does_not_exist.json"
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    with pytest.raises(process_video._HardStageFailure, match="not found"):
        pipeline._stage_calibration()


def test_stage_calibration_hard_fails_when_external_consumption_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    metric_path = tmp_path / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload(source="estimated_from_declared_court_corners"))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.court_calibration = metric_path
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        return {
            "status": "fail",
            "stages": [{"stage": stage, "status": "fail", "notes": ["not a trusted external calibration source"]}],
        }

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    with pytest.raises(process_video._HardStageFailure, match="not a trusted external calibration source"):
        pipeline._stage_calibration()


def test_stage_calibration_auto_discovers_metric_artifact_next_to_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_dir = tmp_path / "eval_clips" / "ball" / "some_clip"
    video = clip_dir / "source.mp4"
    clip_dir.mkdir(parents=True)
    _make_video(video)
    metric_path = clip_dir / "labels" / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload())

    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs({"calibration": {"court_calibration.json": _external_metric_calibration_payload()}})
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    outcome = pipeline._stage_calibration()

    assert outcome.status == "ran"
    assert any("auto-discovered" in note for note in outcome.notes)
    [call] = fake_run_pipeline.calls  # type: ignore[attr-defined]
    assert call["runners"]["calibration"].source_path == metric_path


# ---------------------------------------------------------------------------
# confidence gate + replay manifest tail integration
# ---------------------------------------------------------------------------


def test_confidence_gate_stage_writes_gated_world_and_band_counts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    world = _virtual_world_payload()
    _write_json(options.clip_dir / "virtual_world.json", world)
    _write_json(
        options.clip_dir / "ball_track_physics_filled.json",
        {
            "frames": [
                world["ball"]["frames"][0],
                {
                    **world["ball"]["frames"][1],
                    "source": "physics_interpolated",
                    "physics_fill": {
                        "render_only": True,
                        "not_for_detection_metrics": True,
                        "uncertainty_m": 0.15,
                        "gap_distance_frames": 1,
                    },
                },
            ]
        },
    )
    curves_path = tmp_path / "calibration_curves.json"
    _write_json(curves_path, {"ball": {"horizon_buckets": {"1-3": {"p50_m": 0.2, "p90_m": 0.6}}}})
    options.confidence_calibration_curves = curves_path
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_confidence_gate()

    assert outcome.status == "ran"
    assert outcome.artifacts == ["confidence_gated_world.json", "confidence_gate_summary.json"]
    assert outcome.metrics["counts_by_entity_band"]["ball"]["measured"] == 1
    assert outcome.metrics["counts_by_entity_band"]["ball"]["physics_predicted"] == 1
    assert outcome.metrics["calibration_curves"] == str(curves_path)
    gated = json.loads((options.clip_dir / "confidence_gated_world.json").read_text(encoding="utf-8"))
    assert gated["ball"]["frames"][1]["confidence_provenance"]["predictor"] == "BallBallisticAdapter"
    assert json.loads((options.clip_dir / "virtual_world.json").read_text(encoding="utf-8")) == world


def test_confidence_gate_stage_can_be_skipped_by_option(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.confidence_gate = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_confidence_gate()

    assert outcome.status == "skipped"
    assert "--no-confidence-gate" in " ".join(outcome.notes)
    assert not (options.clip_dir / "confidence_gated_world.json").exists()


def test_confidence_gate_counts_are_written_to_pipeline_summary(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_confidence_gate()
    pipeline.stage_outcomes.append(outcome)

    summary = pipeline._write_summary(wall_seconds=0.1)

    [stage] = summary["stages"]
    assert stage["stage"] == "confidence_gate"
    assert stage["status"] == "ran"
    assert stage["metrics"]["counts_by_entity_band"]["ball"]["measured"] == 1
    assert stage["metrics"]["counts_by_entity_band"]["player_joints"]["measured"] == 2


def _foot_contact_phases_payload(*, phases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = phases if phases is not None else [
        {
            "player_id": 1,
            "foot": "left",
            "start_frame_index": 0,
            "end_frame_index": 0,
            "frame_indices": [0],
            "frame_count": 1,
            "anchor_position_xyz": [0.0, 0.0, 0.0],
            "min_confidence": 0.95,
            "max_height_m": 0.01,
            "max_speed_mps": 0.05,
            "source": "unit_test",
            "source_phase_foot": "left",
            "foot_assignment": "per_foot_keypoint_support",
            "source_thresholds": {"min_confidence": 0.20},
            "assignment_evidence": {"body_detector_agreement": 0.95},
        }
    ]
    return {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "phase_count": len(items),
        "phases": items,
    }


def test_grounding_refine_stage_refines_skeleton_between_body_and_world(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _lane_a_skeleton_payload())
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_grounding_refine()

    assert outcome.status == "ran"
    assert "body_grounding_refinement.json" in outcome.artifacts
    assert "skeleton3d_pre_grounding_refine.json" in outcome.artifacts
    assert outcome.metrics["phase_count"] == 1
    assert outcome.metrics["correction_magnitude_m"]["warn_count"] == 1
    assert outcome.metrics["policy_note"] == "render-honest estimated grounding, not gate evidence"
    assert any("render-honest estimated grounding, not gate evidence" in note for note in outcome.notes)

    refined = json.loads((options.clip_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    frame = refined["players"][0]["frames"][0]
    assert frame["transl_world"] == pytest.approx([1.0, 2.0, -0.05])
    assert frame["confidence_provenance"]["band"] == "physics_corrected_warn"
    report = json.loads((options.clip_dir / "body_grounding_refinement.json").read_text(encoding="utf-8"))
    assert report["policy"]["accuracy_claim"] == "render-honest estimated grounding, not gate evidence"


def test_grounding_refine_stage_skips_zero_contact_phases_without_touching_payload(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    skeleton = _lane_a_skeleton_payload()
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", skeleton)
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload(phases=[]))
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_grounding_refine()

    assert outcome.status == "skipped"
    assert outcome.metrics["status"] == "skipped_no_contact_phases"
    assert outcome.metrics["phase_count"] == 0
    assert json.loads((options.clip_dir / "skeleton3d.json").read_text(encoding="utf-8")) == skeleton
    report = json.loads((options.clip_dir / "body_grounding_refinement.json").read_text(encoding="utf-8"))
    assert report["status"] == "skipped_no_contact_phases"


def test_grounding_refine_stage_backfills_missing_skeleton_transl_world_from_tracks(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    skeleton = _lane_a_skeleton_payload()
    del skeleton["players"][0]["frames"][0]["transl_world"]
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", skeleton)
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_grounding_refine()

    assert outcome.status == "ran"
    assert outcome.metrics["transl_world_backfilled_frames"]["skeleton3d.json"] == 1
    original = json.loads((options.clip_dir / "skeleton3d_pre_grounding_refine.json").read_text(encoding="utf-8"))
    assert "transl_world" not in original["players"][0]["frames"][0]
    refined = json.loads((options.clip_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    frame = refined["players"][0]["frames"][0]
    assert frame["transl_world"] == pytest.approx([1.0, 2.0, -0.05])
    assert frame["confidence_provenance"]["band"] == "physics_corrected"


def test_grounding_refine_stage_disables_xy_when_body_has_r3_grounding_provenance(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    skeleton = _lane_a_skeleton_payload()
    skeleton["provenance"]["grounding_anchor_source"] = "placement_track_world_xy"
    skeleton["players"][0]["frames"][0]["transl_world"] = [0.2, -0.1, 0.0]
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", skeleton)
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_grounding_refine()

    assert outcome.status == "ran"
    assert outcome.metrics["xy_translation_enabled"] is False
    refined = json.loads((options.clip_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    frame = refined["players"][0]["frames"][0]
    assert frame["transl_world"][:2] == pytest.approx([0.2, -0.1])
    report = json.loads((options.clip_dir / "body_grounding_refinement.json").read_text(encoding="utf-8"))
    assert report["summary"]["xy_translation_enabled"] is False


def test_grounding_refine_stage_can_be_disabled(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.grounding_refine = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _lane_a_skeleton_payload())
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_grounding_refine()

    assert outcome.status == "skipped"
    assert "--no-grounding-refine" in " ".join(outcome.notes)
    assert not (options.clip_dir / "body_grounding_refinement.json").exists()


def test_manifest_uses_gated_world_and_replay_points_from_rally_span_with_event_trust_notes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    world = _virtual_world_payload()
    _write_json(options.clip_dir / "virtual_world.json", world)
    gated_world = {**world, "confidence_gate": {"counts_by_entity_band": {"ball": {"measured": 2}}}}
    _write_json(options.clip_dir / "confidence_gated_world.json", gated_world)
    _write_json(options.clip_dir / "contact_windows.json", _contact_windows_payload())
    _write_json(
        options.clip_dir / "rally_spans.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_rally_spans",
            "clip_id": options.clip,
            "not_ground_truth": True,
            "spans": [{"t0": 0.0, "t1": 1.0, "sources": ["ball", "player_motion"]}],
        },
    )

    outcome = pipeline._stage_manifest()

    assert outcome.status == "ran"
    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["virtual_world_url"].endswith("/confidence_gated_world.json")
    assert manifest["replay_scene_url"].endswith("/replay_scene.json")
    replay_scene = json.loads((options.clip_dir / "replay_scene.json").read_text(encoding="utf-8"))
    assert replay_scene["points"] == [
        {
            "id": 1,
            "t0": 0.0,
            "t1": 0.05,
            "glb_url": "replay_review/points/point_001_review.glb",
            "size_mb": replay_scene["points"][0]["size_mb"],
        }
    ]
    assert (options.clip_dir / replay_scene["court_glb"]).is_file()
    assert (options.clip_dir / replay_scene["points"][0]["glb_url"]).is_file()
    contact_windows = json.loads((options.clip_dir / "contact_windows.json").read_text(encoding="utf-8"))
    assert contact_windows["events"][0]["trust_band_note"] == "wrist+ball cues, unverified"
    assert outcome.metrics["replay_point_count"] == 1
    assert outcome.metrics["replay_point_source"] == "contact_windows"
    assert outcome.metrics["contact_event_trust_notes"]["wrist+ball cues, unverified"] == 1


def test_manifest_prefers_tight_contact_windows_over_whole_clip_rally_span_for_replay_points(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    world = _virtual_world_payload()
    _write_json(options.clip_dir / "virtual_world.json", world)
    _write_json(options.clip_dir / "confidence_gated_world.json", world)
    _write_json(
        options.clip_dir / "contact_windows.json",
        {
            "schema_version": 1,
            "events": [
                {
                    "type": "contact",
                    "t": 0.175,
                    "frame": 5,
                    "player_id": 3,
                    "confidence": 0.68,
                    "sources": {"wrist_vel": 0.6, "ball_inflection": 0.76},
                    "window": {"t0": 0.14, "t1": 0.23, "importance": 0.68},
                },
                {
                    "type": "contact",
                    "t": 4.925,
                    "frame": 148,
                    "player_id": 2,
                    "confidence": 0.8,
                    "sources": {"wrist_vel": 0.7, "ball_inflection": 0.9},
                    "window": {"t0": 4.89, "t1": 4.98, "importance": 0.8},
                },
            ],
        },
    )
    _write_json(
        options.clip_dir / "rally_spans.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_rally_spans",
            "clip_id": options.clip,
            "not_ground_truth": True,
            "spans": [{"t0": 0.0, "t1": 10.0, "sources": ["ball", "player_motion"]}],
        },
    )

    outcome = pipeline._stage_manifest()

    replay_scene = json.loads((options.clip_dir / "replay_scene.json").read_text(encoding="utf-8"))
    spans = [(point["t0"], point["t1"]) for point in replay_scene["points"]]
    assert spans == [(0.14, 0.23), (4.89, 4.98)]
    assert all((t1 - t0) < 1.5 for t0, t1 in spans)
    assert outcome.metrics["replay_point_count"] == 2
    assert outcome.metrics["replay_point_source"] == "contact_windows"
    assert outcome.metrics["replay_point_skipped_broad_span_count"] == 1


def test_manifest_skips_whole_clip_rally_span_when_no_tight_replay_point_spans_exist(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    world = _virtual_world_payload()
    _write_json(options.clip_dir / "virtual_world.json", world)
    _write_json(options.clip_dir / "confidence_gated_world.json", world)
    _write_json(
        options.clip_dir / "rally_spans.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_rally_spans",
            "clip_id": options.clip,
            "not_ground_truth": True,
            "spans": [{"t0": 0.0, "t1": 10.0, "sources": ["ball", "player_motion"]}],
        },
    )

    outcome = pipeline._stage_manifest()

    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["replay_scene_url"] is None
    assert not (options.clip_dir / "replay_scene.json").exists()
    assert outcome.metrics["replay_point_count"] == 0
    assert outcome.metrics["replay_point_skipped_broad_span_count"] == 1


def test_manifest_no_scene_points_option_disables_replay_scene_points(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.scene_points = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    world = _virtual_world_payload()
    _write_json(options.clip_dir / "virtual_world.json", world)
    _write_json(options.clip_dir / "confidence_gated_world.json", world)
    _write_json(options.clip_dir / "contact_windows.json", _contact_windows_payload())

    outcome = pipeline._stage_manifest()

    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["replay_scene_url"] is None
    assert not (options.clip_dir / "replay_scene.json").exists()
    assert outcome.metrics["replay_point_count"] == 0
    assert outcome.metrics["replay_point_source"] == "disabled"


def test_manifest_links_body_mesh_index_and_marks_windowed_mesh_status(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    _write_json(options.clip_dir / "confidence_gated_world.json", _virtual_world_payload())
    _write_json(options.clip_dir / "body_mesh.json", {"artifact_type": "racketsport_body_mesh"})
    _write_json(options.clip_dir / "body_mesh_index.json", {"artifact_type": "racketsport_body_mesh_index"})

    outcome = pipeline._stage_manifest()

    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert outcome.status == "ran"
    assert manifest["body_mesh_url"].endswith("/body_mesh.json")
    assert manifest["body_mesh_index_url"].endswith("/body_mesh_index.json")
    assert manifest["mesh_status"] == "windowed_index"
    assert "mesh_status=windowed_index" in " ".join(outcome.notes)


def test_manifest_uses_fetched_body_mesh_index_directory_without_monolith(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    index_dir = options.clip_dir / "body_mesh_index"
    index_dir.mkdir(parents=True)
    _write_json(index_dir / "body_mesh_index.json", {"artifact_type": "racketsport_body_mesh_index"})

    outcome = pipeline._stage_manifest()

    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    joined_notes = " ".join(outcome.notes)
    assert outcome.status == "ran"
    assert manifest["body_mesh_url"] is None
    assert manifest["body_mesh_index_url"].endswith("/body_mesh_index/body_mesh_index.json")
    assert manifest["mesh_status"] == "windowed_index"
    assert "body_mesh.json not fetched (speed default)" in joined_notes
    assert "mesh_status=windowed_index" in joined_notes


def test_manifest_marks_monolithic_mesh_without_index_as_unverified(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    _write_json(options.clip_dir / "confidence_gated_world.json", _virtual_world_payload())
    _write_json(options.clip_dir / "body_mesh.json", {"artifact_type": "racketsport_body_mesh"})

    outcome = pipeline._stage_manifest()

    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert outcome.status == "ran"
    assert manifest["body_mesh_url"].endswith("/body_mesh.json")
    assert manifest["body_mesh_index_url"] is None
    assert manifest["mesh_status"] == "monolithic_unverified"
    assert "mesh_status=monolithic_unverified" in " ".join(outcome.notes)


def test_manifest_uses_raw_world_when_confidence_gate_is_disabled(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.confidence_gate = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    _write_json(options.clip_dir / "confidence_gated_world.json", _virtual_world_payload())

    outcome = pipeline._stage_manifest()

    assert outcome.status == "ran"
    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["virtual_world_url"].endswith("/virtual_world.json")


def test_ball_fill_stage_writes_render_only_physics_artifact_with_reviewed_bounces(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    frames = []
    for index in range(7):
        t = index / 30.0
        frames.append(
            {
                "t": t,
                "xy": [400.0 + index * 3.0, 300.0],
                "conf": 0.95,
                "visible": True,
                "world_xyz": [float(index) * 0.05, 0.0, 0.0],
                "approx": True,
            }
        )
    _write_json(options.clip_dir / "ball_track.json", {"schema_version": 1, "fps": 30.0, "source": "wasb", "frames": frames, "bounces": []})
    _write_json(
        options.clip_dir / "reviewed_ball_bounces.json",
        {
            "artifact_type": "racketsport_reviewed_ball_bounces",
            "status": "human_reviewed",
            "source": "human_review",
            "bounces": [{"frame": 3, "t": 3 / 30.0, "review_id": "bounce_0001"}],
        },
    )
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_ball_fill()

    assert outcome.status == "ran"
    assert "ball_track_physics_filled.json" in outcome.artifacts
    payload = json.loads((options.clip_dir / "ball_track_physics_filled.json").read_text(encoding="utf-8"))
    assert payload["physics_fill"]["bounce_boundaries"][0]["source"] == "human_reviewed"
    assert payload["physics_fill"]["render_only"] is True
    assert payload["physics_fill"]["not_for_detection_metrics"] is True


def test_manifest_passes_existing_reviewed_bounces_coaching_facts_and_rally_spans(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    _write_json(
        options.clip_dir / "reviewed_ball_bounces.json",
        {"artifact_type": "racketsport_reviewed_ball_bounces", "status": "human_reviewed", "bounces": []},
    )
    _write_json(options.clip_dir / "coaching_card_facts.json", {"artifact_type": "racketsport_coaching_card_facts"})
    _write_json(
        options.clip_dir / "rally_spans.json",
        {"artifact_type": "racketsport_rally_spans", "not_ground_truth": True, "spans": []},
    )

    outcome = pipeline._stage_manifest()

    assert outcome.status == "ran"
    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reviewed_bounces_url"].endswith("/reviewed_ball_bounces.json")
    assert manifest["coaching_card_facts_url"].endswith("/coaching_card_facts.json")
    assert manifest["rally_spans_url"].endswith("/rally_spans.json")


def test_manifest_exposes_ball_arc_artifacts_when_present(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "virtual_world.json", _virtual_world_payload())
    _write_json(
        options.clip_dir / "ball_track_arc_solved.json",
        {"artifact_type": "racketsport_ball_track_arc_solved", "status": "ran", "frames": []},
    )
    _write_json(
        options.clip_dir / "ball_bounce_candidates.json",
        {"artifact_type": "racketsport_ball_bounce_candidates", "candidates": []},
    )
    _write_json(
        options.clip_dir / "ball_flight_sanity.json",
        {"artifact_type": "racketsport_ball_flight_sanity", "summary": {"demoted_frame_count": 0}},
    )

    outcome = pipeline._stage_manifest()

    assert outcome.status == "ran"
    manifest = json.loads((options.clip_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ball_arc_solved_url"].endswith("/ball_track_arc_solved.json")
    assert manifest["auto_bounce_candidates_url"].endswith("/ball_bounce_candidates.json")
    assert manifest["ball_bounce_candidates_url"].endswith("/ball_bounce_candidates.json")
    assert manifest["ball_flight_sanity_url"].endswith("/ball_flight_sanity.json")


# ---------------------------------------------------------------------------
# stage sequencing + resume
# ---------------------------------------------------------------------------


def test_input_quality_advisory_bands_not_fully_visible_low_angle_before_heavy_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=120, fps=60.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.input_quality_mode = "advisory"  # type: ignore[attr-defined]
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _low_angle_not_fully_visible_calibration_payload())

    monkeypatch.setattr(
        process_video,
        "_probe_video_quality_samples",
        lambda *args, **kwargs: {"blur_laplacian_var": 500.0, "luminance_mean": 0.5, "sampled_frame_count": 3},
        raising=False,
    )

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_input_quality()
    payload = json.loads((options.clip_dir / "input_quality.json").read_text(encoding="utf-8"))

    assert outcome.status == "degraded"
    assert payload["band"] == "degraded_input"
    assert payload["rejection_reasons"][0] == "court_not_fully_visible_low_angle"
    assert outcome.metrics["input_quality"]["band"] == "degraded_input"


def test_input_quality_strict_fail_closes_and_stops_before_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=120, fps=60.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.input_quality_mode = "strict"  # type: ignore[attr-defined]
    options.clip_dir.mkdir(parents=True, exist_ok=True)

    def _fake_ingest() -> process_video.StageOutcome:
        return process_video.StageOutcome(stage="ingest", status="ran", wall_seconds=0.0)

    def _fake_calibration() -> process_video.StageOutcome:
        _write_json(options.clip_dir / "court_calibration.json", _low_angle_not_fully_visible_calibration_payload())
        return process_video.StageOutcome(stage="calibration", status="ran", wall_seconds=0.0)

    def _fail_if_tracking_runs() -> process_video.StageOutcome:
        raise AssertionError("tracking should not run after strict input-quality rejection")

    monkeypatch.setattr(
        process_video,
        "_probe_video_quality_samples",
        lambda *args, **kwargs: {"blur_laplacian_var": 500.0, "luminance_mean": 0.5, "sampled_frame_count": 3},
        raising=False,
    )
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_ingest", lambda self: _fake_ingest())
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_calibration", lambda self: _fake_calibration())
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_tracking", lambda self: _fail_if_tracking_runs())

    summary = process_video.ProcessVideoPipeline(options).run()

    assert summary["status"] == "failed"
    assert [stage["stage"] for stage in summary["stages"]] == ["ingest", "calibration", "input_quality"]
    input_stage = summary["stages"][-1]
    assert input_stage["status"] == "failed"
    assert input_stage["metrics"]["input_quality"]["strict"] is True


def test_body_summary_populates_postchain_bypassed_stages_from_phase_timing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        options.clip_dir / "body_stage_phase_timing.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_stage_phase_timing",
            "postchain_bypasses": {
                "status": "postchain_bypassed",
                "stages": [
                    "temporal_smoothing",
                    "foot_lock",
                    "foot_pin",
                    "contact_splice",
                    "wrist_lock",
                    "world_joint_visual_smoothing",
                ],
                "raw_grounded_joints_sidecar": "body_raw_grounded_joints.json",
            },
        },
    )
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline.stage_outcomes.append(process_video.StageOutcome(stage="body", status="ran", wall_seconds=1.0))

    summary = pipeline._write_summary(wall_seconds=1.0)

    body_stage = summary["stages"][0]
    assert body_stage["metrics"]["postchain_bypassed_stages"] == [
        "temporal_smoothing",
        "foot_lock",
        "foot_pin",
        "contact_splice",
        "wrist_lock",
        "world_joint_visual_smoothing",
    ]
    assert body_stage["metrics"]["postchain_bypass_status"] == "postchain_bypassed"


def test_match_stats_stage_emits_body_court_only_consumer_artifact(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.match_stats = True  # type: ignore[attr-defined]
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "placement.json", _match_stats_placement_payload())
    _write_json(options.clip_dir / "court_zones.json", _match_stats_court_zones_payload())
    _write_json(options.clip_dir / "trust_bands.json", {"court": {"badge": "preview"}, "body": {"badge": "low_confidence"}})

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_match_stats()
    payload = json.loads((options.clip_dir / "match_stats.json").read_text(encoding="utf-8"))

    assert outcome.status == "ran"
    assert outcome.artifacts == ["match_stats.json"]
    assert payload["policy"]["body_court_only"] is True
    assert payload["inputs"]["ball"] is None
    assert outcome.metrics["player_count"] == 1


def test_match_stats_stage_skips_loudly_when_consumer_inputs_are_absent(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.match_stats = True  # type: ignore[attr-defined]
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_zones.json", _match_stats_court_zones_payload())

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_match_stats()

    assert outcome.status == "skipped"
    assert outcome.artifacts == []
    assert outcome.metrics["reason"] == "missing_inputs"
    assert outcome.metrics["missing_inputs"] == ["placement.json"]


def test_pipeline_runs_all_stages_in_order_with_mocked_heavy_runners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))

    fake_run_pipeline = _fake_run_pipeline_factory(
        {
            "calibration": {"court_calibration.json": _court_calibration_payload(), "court_zones.json": {"schema_version": 1, "sport": "pickleball", "zones": []}, "net_plane.json": {"schema_version": 1, "sport": "pickleball", "net_height_center_m": 0.86, "net_height_post_m": 0.914, "y_m": 6.7}, "court_line_evidence.json": {"schema_version": 1, "sport": "pickleball", "source": "test", "line_observations": [], "keypoint_observations": [], "net_observations": [], "aggregate": {"accepted_line_ids": [], "rejected_line_ids": [], "missing_required_line_ids": [], "missing_required_net_ids": [], "mean_residual_px": 0.0, "p95_residual_px": 0.0, "temporal_stability_px": 0.0, "auto_calibration_ready": True, "reasons": []}}},
        }
    )
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    pipeline = process_video.ProcessVideoPipeline(options)
    summary = pipeline.run()

    stage_names = [s["stage"] for s in summary["stages"]]
    assert stage_names == [
        "ingest",
        "calibration",
        "input_quality",
        "tracking",
        "camera_motion",
        "placement",
        "ball",
        "ball_arc",
        "events",
        "ball_fill",
        "frames",
        "body",
        "placement_refine",
        "grounding_refine",
        "paddle_pose",
        "world",
        "confidence_gate",
        "manifest",
        "match_stats",
    ]
    assert summary["status"] in {"complete", "partial"}
    # tracking/frames/body all blocked/degraded because no_gpu=True and no reuse artifacts given.
    by_stage = {s["stage"]: s for s in summary["stages"]}
    assert by_stage["calibration"]["status"] == "ran"
    assert by_stage["input_quality"]["status"] in {"ran", "degraded"}
    assert by_stage["tracking"]["status"] == "blocked"
    assert by_stage["placement"]["status"] == "blocked"
    assert by_stage["frames"]["status"] == "blocked"
    assert "pose" not in by_stage
    assert by_stage["ball"]["status"] == "skipped"
    assert by_stage["ball_arc"]["status"] == "skipped"
    assert by_stage["ball_fill"]["status"] == "blocked"
    assert by_stage["body"]["status"] == "degraded"
    assert "SAM-3D BODY skipped" in " ".join(by_stage["body"]["notes"])
    assert by_stage["placement_refine"]["status"] == "skipped"
    assert by_stage["grounding_refine"]["status"] == "skipped"
    # world/manifest still assemble a partial (court-only) bundle, not a crash.
    assert by_stage["world"]["status"] == "ran"
    assert by_stage["confidence_gate"]["status"] == "ran"
    assert by_stage["manifest"]["status"] == "ran"
    assert by_stage["match_stats"]["status"] in {"ran", "skipped"}
    assert (options.clip_dir / "virtual_world.json").is_file()
    assert (options.run_dir / "PIPELINE_SUMMARY.json").is_file()


def test_placement_stage_rewrites_tracks_after_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    keypoints_path = tmp_path / "native_keypoints.json"
    _write_json(keypoints_path, {"schema_version": 1, "artifact_type": "racketsport_keypoints_2d", "players": []})
    options.placement_keypoints_2d = keypoints_path

    calls: list[dict[str, Any]] = []

    def _fake_rewrite(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        _write_json(
            kwargs["placement_path"],
            {
                "schema_version": 1,
                "artifact_type": "racketsport_placement",
                "fps": 30.0,
                "source": "test",
                "tracks_path": "tracks.json",
                "backup_tracks_path": "tracks_prewrite_backup.json",
                "refine_from_sam3d": False,
                "undistort_applied": False,
                "players": [],
                "summary": {
                    "player_count": 0,
                    "frame_count": 0,
                    "coverage_unchanged": True,
                    "source_counts": {"bbox": 0, "native2d": 0, "sam3d": 0},
                    "jitter_before_after_mps": {},
                    "court_bounds_violations": 0,
                },
                "provenance": {},
            },
        )
        _write_json(
            kwargs["foot_contact_phases_out_path"],
            {"schema_version": 1, "artifact_type": "foot_contact_phases", "phase_count": 1, "phases": []},
        )
        return type("Result", (), {"coverage_unchanged": True, "source_counts": {"bbox": 2, "native2d": 1, "sam3d": 0}})()

    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_placement()

    assert outcome.status == "ran"
    assert calls[0]["tracks_path"] == options.clip_dir / "tracks.json"
    assert calls[0]["native2d_keypoints_path"] == keypoints_path
    assert calls[0]["stance_phases_path"] is None
    assert calls[0]["foot_contact_phases_out_path"] == options.clip_dir / "foot_contact_phases.json"
    assert calls[0]["refine_from_sam3d"] is False
    assert "placement.json" in outcome.artifacts
    assert "foot_contact_phases.json" in outcome.artifacts
    assert "foot_contact_phases=foot_contact_phases.json" in outcome.notes


def test_default_stage_order_runs_camera_motion_before_placement(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)

    assert [name for name, _fn in pipeline._build_prefix_stage_fns()] == [
        "ingest",
        "calibration",
        "input_quality",
        "tracking",
        "camera_motion",
        "placement",
    ]
    assert [name for name, _fn in pipeline._middle_stage_fns()] == ["ball", "ball_arc", "events", "ball_fill"]


def test_camera_motion_auto_default_skips_static_probe_without_writing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    probe_calls: list[dict[str, Any]] = []

    def _fake_probe(
        video_path: Path,
        calibration_path: Path,
        *,
        tracks_path: Path | None,
        params: Any,
        threshold: float,
    ) -> dict[str, Any]:
        probe_calls.append(
            {
                "video_path": video_path,
                "calibration_path": calibration_path,
                "tracks_path": tracks_path,
                "params": params,
            }
        )
        return {
            "motion_score": 0.5,
            "threshold": 5.0,
            "enabled": False,
            "forced": "auto",
            "sampled_frame_count": 6,
            "wall_seconds": 0.123,
            "verified": False,
            "not_gate_verified": True,
        }

    monkeypatch.setattr(process_video, "estimate_camera_motion_probe", _fake_probe, raising=False)
    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full estimator should not run when AUTO is off")),
    )

    outcome = pipeline._stage_camera_motion()

    assert outcome.status == "skipped"
    assert len(probe_calls) == 1
    assert probe_calls[0]["video_path"] == options.clip_dir / "source.mp4"
    assert probe_calls[0]["calibration_path"] == options.clip_dir / "court_calibration.json"
    assert probe_calls[0]["tracks_path"] == options.clip_dir / "tracks.json"
    assert probe_calls[0]["params"].estimator_mode == "hardened"
    assert not (options.clip_dir / "camera_motion.json").exists()
    assert pipeline._camera_motion_auto == {
        "score": 0.5,
        "threshold": 5.0,
        "enabled": False,
        "forced": "auto",
        "probe_wall_seconds": 0.123,
        "sampled_frame_count": 6,
    }
    assert outcome.metrics["camera_motion_auto"] == pipeline._camera_motion_auto
    summary = pipeline._write_summary(wall_seconds=0.25)
    persisted_auto = json.loads((options.run_dir / "PIPELINE_SUMMARY.json").read_text(encoding="utf-8"))["camera_motion_auto"]
    assert persisted_auto == summary["camera_motion_auto"] == pipeline._camera_motion_auto
    for key in (
        "decode_orientation_mismatch",
        "decode_orientation_consequential_mismatch",
        "decode_orientation_untrusted",
        "decode_orientation_mismatch_reason",
    ):
        assert key not in persisted_auto


def test_camera_motion_auto_summary_persists_decode_orientation_probe_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    expected_base = {
        "score": 0.5,
        "threshold": 5.0,
        "enabled": False,
        "forced": "auto_decode_orientation_untrusted:rotation_meta_disagrees_with_auto",
        "probe_wall_seconds": 0.123,
        "sampled_frame_count": 6,
    }
    expected_decode_keys = {
        "decode_orientation_mismatch": True,
        "decode_orientation_consequential_mismatch": True,
        "decode_orientation_untrusted": True,
        "decode_orientation_mismatch_reason": "rotation_meta_disagrees_with_auto",
    }

    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion_probe",
        lambda *_args, **_kwargs: {
            "motion_score": expected_base["score"],
            "threshold": expected_base["threshold"],
            "enabled": expected_base["enabled"],
            "forced": expected_base["forced"],
            "sampled_frame_count": expected_base["sampled_frame_count"],
            "wall_seconds": expected_base["probe_wall_seconds"],
            **expected_decode_keys,
        },
        raising=False,
    )
    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full estimator should not run when AUTO is off")),
    )

    outcome = pipeline._stage_camera_motion()
    summary = pipeline._write_summary(wall_seconds=0.25)
    persisted_auto = json.loads((options.run_dir / "PIPELINE_SUMMARY.json").read_text(encoding="utf-8"))["camera_motion_auto"]

    assert outcome.status == "skipped"
    assert {key: pipeline._camera_motion_auto[key] for key in expected_base} == expected_base
    for key, value in expected_decode_keys.items():
        assert pipeline._camera_motion_auto[key] == value
    assert persisted_auto == summary["camera_motion_auto"] == pipeline._camera_motion_auto


def test_pipeline_summary_persists_camera_motion_auto_decision(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._camera_motion_auto = {
        "score": 0.5,
        "threshold": 5.0,
        "enabled": False,
        "forced": "auto",
        "probe_wall_seconds": 0.123,
        "sampled_frame_count": 6,
    }

    summary = pipeline._write_summary(wall_seconds=1.25)

    assert summary["camera_motion_auto"] == pipeline._camera_motion_auto
    assert json.loads((options.run_dir / "PIPELINE_SUMMARY.json").read_text(encoding="utf-8"))["camera_motion_auto"] == pipeline._camera_motion_auto
    assert json.loads((options.clip_dir / "PIPELINE_SUMMARY.json").read_text(encoding="utf-8"))["camera_motion_auto"] == pipeline._camera_motion_auto


def test_camera_motion_auto_off_prevents_stale_clip_artifact_from_reaching_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(
        options.clip_dir / "camera_motion.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion",
            "frames": [],
            "verified": False,
            "not_gate_verified": True,
        },
    )
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion_probe",
        lambda *_args, **_kwargs: {
            "motion_score": 0.5,
            "threshold": 5.0,
            "enabled": False,
            "forced": "auto",
            "sampled_frame_count": 6,
            "wall_seconds": 0.123,
        },
        raising=False,
    )

    def _fake_rewrite(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return type(
            "Result",
            (),
            {
                "coverage_unchanged": True,
                "source_counts": {"bbox": 2},
                "court_bounds_violations": 0,
                "summary": {},
            },
        )()

    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite)

    camera_outcome = pipeline._stage_camera_motion()
    placement_outcome = pipeline._stage_placement()

    assert camera_outcome.status == "skipped"
    assert placement_outcome.status == "ran"
    assert calls[0]["camera_motion_path"] is None
    assert any("camera_motion=not_used source=auto_disabled" in note for note in placement_outcome.notes)


def test_camera_motion_stage_runs_hardened_default_estimator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.enable_camera_motion = True
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    calls: list[dict[str, Any]] = []

    def _fake_estimate_camera_motion(video_path: Path, calibration_path: Path, *, tracks_path: Path, params: Any) -> dict[str, Any]:
        calls.append(
            {
                "video_path": video_path,
                "calibration_path": calibration_path,
                "tracks_path": tracks_path,
                "params": params,
            }
        )
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion",
            "video": str(video_path),
            "method": "homography",
            "reference_frame_idx": 0,
            "params": {"estimator_mode": params.estimator_mode, "flow_backend": params.flow_backend},
            "summary": {
                "n_frames": 2,
                "n_compensated": 2,
                "drift_px_p50": 0.0,
                "drift_px_p95": 0.0,
                "drift_px_max": 0.0,
                "residual_px_p50": 0.0,
                "residual_px_p95": 0.0,
                "residual_px_max": 0.0,
            },
            "frames": [],
            "verified": False,
            "not_gate_verified": True,
        }

    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("force-on should bypass AUTO probe")),
        raising=False,
    )
    monkeypatch.setattr(process_video, "estimate_camera_motion", _fake_estimate_camera_motion)

    outcome = pipeline._stage_camera_motion()

    assert outcome.status == "ran"
    assert outcome.artifacts == ["camera_motion.json"]
    assert calls[0]["video_path"] == options.clip_dir / "source.mp4"
    assert calls[0]["calibration_path"] == options.clip_dir / "court_calibration.json"
    assert calls[0]["tracks_path"] == options.clip_dir / "tracks.json"
    assert calls[0]["params"].estimator_mode == "hardened"
    assert calls[0]["params"].flow_backend == "lk"
    assert calls[0]["params"].use_person_masks is True
    assert outcome.metrics["n_compensated"] == 2
    assert pipeline._camera_motion_auto["enabled"] is True
    assert pipeline._camera_motion_auto["forced"] == "on"


def test_camera_motion_force_off_records_decision_without_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_camera_motion = True
    pipeline = process_video.ProcessVideoPipeline(options)

    monkeypatch.setattr(
        process_video,
        "estimate_camera_motion_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("force-off should bypass AUTO probe")),
        raising=False,
    )

    outcome = pipeline._stage_camera_motion()

    assert outcome.status == "skipped"
    assert pipeline._camera_motion_auto == {
        "score": None,
        "threshold": process_video.CAMERA_MOTION_AUTO_THRESHOLD,
        "enabled": False,
        "forced": "off",
        "probe_wall_seconds": 0.0,
        "sampled_frame_count": 0,
    }
    assert outcome.metrics["camera_motion_auto"] == pipeline._camera_motion_auto


def test_placement_stage_auto_discovers_camera_motion_and_surfaces_guard_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(
        options.clip_dir / "camera_motion.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion",
            "frames": [],
            "verified": False,
            "not_gate_verified": True,
        },
    )
    calls: list[dict[str, Any]] = []

    def _fake_rewrite(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        summary = {
            "camera_motion_frames_used": 3,
            "camera_motion_frames_uncompensated": 1,
            "side_quadrant_consistency": {
                "players": {
                    "1": {
                        "side_label_original": "near",
                        "side_recomputed": "far",
                        "role_original": "left",
                        "role_recomputed": "right",
                    }
                }
            },
            "boundary_guards": {"totals": {"net_gap_clamped_frames": 2, "centerline_gap_clamped_frames": 1}},
            "smoothing_guards": {"totals": {"divergence_snap_frames": 4, "fallback_transition_blends": 5}},
            "sidecar_identity": {
                "native2d": {"totals": {"reassigned_obs": 6, "dropped_obs": 7}},
                "sam3d": {"totals": {"reassigned_obs": 8, "dropped_obs": 9}},
            },
        }
        return type(
            "Result",
            (),
            {
                "coverage_unchanged": True,
                "source_counts": {"bbox": 2, "native2d": 1, "sam3d": 0},
                "court_bounds_violations": 0,
                "summary": summary,
            },
        )()

    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_placement()

    assert outcome.status == "ran"
    assert calls[0]["camera_motion_path"] == options.clip_dir / "camera_motion.json"
    joined_notes = " | ".join(outcome.notes)
    assert "camera_motion=used" in joined_notes
    assert "source=auto_discovered" in joined_notes
    assert "frames_used=3" in joined_notes
    assert "frames_uncompensated=1" in joined_notes
    assert "side_recompute(player=1: side near->far, role left->right)" in joined_notes
    assert "net_gap_clamped_frames=2" in joined_notes
    assert "centerline_gap_clamped_frames=1" in joined_notes
    assert "divergence_snap_frames=4" in joined_notes
    assert "fallback_transition_blends=5" in joined_notes
    assert "native2d_reassigned=6" in joined_notes
    assert "native2d_dropped=7" in joined_notes
    assert "sam3d_reassigned=8" in joined_notes
    assert "sam3d_dropped=9" in joined_notes


def test_placement_refine_stage_is_disabled_same_pass_even_when_sam3d_sidecar_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    keypoints_path = tmp_path / "native_keypoints.json"
    _write_json(keypoints_path, {"schema_version": 1, "artifact_type": "racketsport_keypoints_2d", "players": []})
    options.placement_keypoints_2d = keypoints_path
    pipeline = process_video.ProcessVideoPipeline(options)

    skipped = pipeline._stage_placement_refine()
    assert skipped.status == "skipped"

    _write_json(
        options.clip_dir / "sam3d_keypoints_2d.json",
        {"schema_version": 1, "artifact_type": "racketsport_sam3d_keypoints_2d", "source": "test", "players": []},
    )
    def _fail_rewrite(**_kwargs):  # noqa: ANN001
        raise AssertionError("same-pass placement_refine must not rewrite tracks after BODY")

    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fail_rewrite)

    ran = pipeline._stage_placement_refine()

    assert ran.status == "skipped"
    assert ran.metrics["same_pass_track_rewrite_disabled"] is True
    assert "second pass before a fresh BODY run" in " ".join(ran.notes)


def test_placement_refine_stage_is_disabled_same_pass_with_stance_phases_without_sam3d_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "foot_contact_phases.json", _foot_contact_phases_payload())
    def _fail_rewrite(**_kwargs):  # noqa: ANN001
        raise AssertionError("same-pass placement_refine must not rewrite tracks after BODY")

    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fail_rewrite)
    pipeline = process_video.ProcessVideoPipeline(options)

    ran = pipeline._stage_placement_refine()

    assert ran.status == "skipped"
    assert ran.metrics["same_pass_track_rewrite_disabled"] is True
    assert "second pass before a fresh BODY run" in " ".join(ran.notes)


def test_pipeline_blocks_wrist_cues_before_body_without_pose_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    options.skip_ball = True
    options.max_players = 2
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _external_metric_calibration_with_points())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())

    dispatch_calls: list[dict[str, Any]] = []

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        dispatch_calls.append(kwargs)
        clip_dir = Path(kwargs["clip_dir"])
        contact_windows = json.loads((clip_dir / "contact_windows.json").read_text(encoding="utf-8"))
        frame_plan = json.loads((clip_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
        wrist = json.loads((clip_dir / "wrist_velocity_peaks.json").read_text(encoding="utf-8"))
        assert wrist["status"] == "blocked"
        assert "missing_sam3d_skeleton3d" in wrist["blockers"]
        assert len(contact_windows["events"]) == 1
        assert frame_plan["summary"]["deep_mesh_frame_count"] > 0
        _write_json(
            clip_dir / "smpl_motion.json",
            {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []},
        )
        return RemoteBodyDispatchResult(
            status="ran",
            remote_run_dir="remote:/tmp/fake-body",
            synced_outputs=["smpl_motion.json"],
            wall_seconds=4.0,
            notes=["BODY consumed regenerated contact plan"],
        )

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    monkeypatch.setattr(process_video, "fuse_contact_windows_from_cue_payloads", lambda **kwargs: _contact_windows_payload())

    pipeline = process_video.ProcessVideoPipeline(options)
    summary = pipeline.run()

    by_stage = {stage["stage"]: stage for stage in summary["stages"]}
    assert len(dispatch_calls) == 1
    assert "pose" not in by_stage
    assert by_stage["events"]["status"] == "ran"
    assert by_stage["events"]["metrics"]["contact_event_count"] == 1
    assert by_stage["body"]["status"] == "ran"


def test_events_uses_preview_ghost_mesh_when_auto_court_has_no_contacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_audio = True
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "auto_court_corners_preview.json", _court_corners_payload())

    monkeypatch.setattr(
        process_video,
        "build_wrist_velocity_peaks_from_file",
        lambda *args, **kwargs: {"schema_version": 1, "source": "test", "summary": {"peak_count": 0}, "peaks": []},
    )
    monkeypatch.setattr(
        process_video,
        "fuse_contact_windows_from_cue_payloads",
        lambda **kwargs: {"schema_version": 1, "events": []},
    )

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_events()

    plan = json.loads((options.clip_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    selected = [frame for frame in plan["frames"] if frame["tier_rationale"]["mesh_selected"]]
    ghost_selected = [frame for frame in selected if frame["recommended_tier"] == "human_review"]

    assert outcome.status == "ran"
    assert plan["summary"]["deep_mesh_frame_count"] > 0
    assert ghost_selected
    assert {frame["target_representation"] for frame in ghost_selected} == {"manual_review_required"}
    assert {frame.get("trust_badge") for frame in ghost_selected} == {"preview"}


def _events_selected_payload() -> dict[str, Any]:
    """events_selected.json from scripts/racketsport/solve_ball_arcs.py
    (physically-validated contacts only); matches _tracks_payload's player 1
    at frame 0."""
    return {
        "artifact_type": "racketsport_ball_arc_events_selected",
        "selected": [
            {
                "anchor_id": "contact_000_p1_left",
                "kind": "contact",
                "frame": 0,
                "t": 0.0,
                "player_id": 1,
                "candidate_confidence": 0.8,
                "selected": True,
            }
        ],
        "rejected": [],
        "selected_count": 1,
    }


def _ball_track_arc_solved_payload() -> dict[str, Any]:
    """ball_track_arc_solved.json world position co-located with
    _tracks_payload's player 2 (world_xy=[-1.0, -2.0]) at frame 0, so only
    player 2 falls inside the default 1.5m ball_proximity_m band."""
    return {
        "artifact_type": "racketsport_ball_track_arc_solved",
        "frames": [{"t": 0.0, "world_xyz": [-1.0, -2.0, 0.3], "visible": True, "band": "anchored_measured"}],
    }


def test_events_ball_aware_mode_schedules_from_events_proximity_and_swing_not_raw_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_audio = True
    options.mesh_coverage_mode = "ball_aware"
    options.max_players = 2
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "events_selected.json", _events_selected_payload())
    _write_json(options.clip_dir / "ball_track_arc_solved.json", _ball_track_arc_solved_payload())

    monkeypatch.setattr(
        process_video,
        "build_wrist_velocity_peaks_from_file",
        lambda *args, **kwargs: {"schema_version": 1, "source": "test", "summary": {"peak_count": 1}, "peaks": []},
    )
    # A high-confidence (>= default 0.6 floor) fused wrist+ball cue -- under
    # ball_aware mode this must surface as high_confidence_swing, never the
    # raw contact_window reason.
    monkeypatch.setattr(process_video, "fuse_contact_windows_from_cue_payloads", lambda **kwargs: _contact_windows_payload())

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_events()

    assert outcome.status == "ran"
    assert outcome.metrics["mesh_coverage_mode"] == "ball_aware"
    trigger_counts = outcome.metrics["ball_aware_trigger_source_counts"]
    assert trigger_counts["events"] == 1
    assert trigger_counts["proximity"] == 1
    assert trigger_counts["swing"] == 1
    assert any("loaded events_selected.json" in note for note in outcome.notes)
    assert any("loaded ball_track_arc_solved.json" in note for note in outcome.notes)

    plan = json.loads((options.clip_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    assert plan["mesh_coverage_policy"]["mode"] == "ball_aware"
    frame_zero = next(frame for frame in plan["frames"] if frame["frame_idx"] == 0)
    assert "ball_aware_contact" in frame_zero["reasons"]
    assert "ball_proximity" in frame_zero["reasons"]
    assert "high_confidence_swing" in frame_zero["reasons"]
    for frame in plan["frames"]:
        assert "contact_window" not in frame["reasons"]


def test_events_ball_aware_mode_degrades_gracefully_when_solver_artifacts_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No events_selected.json/ball_track_arc_solved.json in the clip dir
    (the common case until PIPELINE-GUARDS wires scripts/racketsport/solve_ball_arcs.py
    into the pipeline) must not crash -- ball_aware mode just has zero
    events/proximity candidates and falls back to uniform fill."""
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_audio = True
    options.mesh_coverage_mode = "ball_aware"
    options.max_players = 2
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())

    monkeypatch.setattr(
        process_video,
        "build_wrist_velocity_peaks_from_file",
        lambda *args, **kwargs: {"schema_version": 1, "source": "test", "summary": {"peak_count": 0}, "peaks": []},
    )
    monkeypatch.setattr(process_video, "fuse_contact_windows_from_cue_payloads", lambda **kwargs: {"schema_version": 1, "events": []})

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_events()

    assert outcome.status == "ran"
    assert outcome.metrics["mesh_coverage_mode"] == "ball_aware"
    # zero events/proximity/swing candidates -- the sole eligible frame falls
    # back to uniform fill rather than being silently dropped.
    assert outcome.metrics["ball_aware_trigger_source_counts"] == {"events": 0, "proximity": 0, "swing": 0, "uniform_fill": 1}
    assert any("no events_selected.json found" in note for note in outcome.notes)
    assert any("no ball_track_arc_solved.json found" in note for note in outcome.notes)


def test_cli_parses_mesh_coverage_flags_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    events_selected = tmp_path / "events_selected.json"
    ball_track_arc_solved = tmp_path / "ball_track_arc_solved.json"
    _write_json(events_selected, _events_selected_payload())
    _write_json(ball_track_arc_solved, _ball_track_arc_solved_payload())
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")]))
    assert default_options.mesh_coverage_mode == "ball_aware"
    # sanctioned default change: best-stack doctrine, owner PLAYBACK RULING 2026-07-08
    assert default_options.target_mesh_frame_budget is None
    assert default_options.mesh_byte_budget_mib == 300.0
    assert default_options.events_selected is None
    assert default_options.ball_track_arc_solved is None
    assert default_options.remote_config.sam3d_body_input_size_px == 384
    assert default_options.remote_config.sam3d_crop_bucket_sizes == (8, 16)
    assert default_options.remote_config.sam3d_torch_compile is True
    assert default_options.remote_config.sam3d_compile_warmup_buckets == (8, 16)
    assert default_options.remote_config.sam3d_skip_tier2_mesh_vertices is True
    assert default_options.remote_config.fetch_body_monoliths is False
    assert default_options.remote_config.body_postchain_mode == "default"
    assert default_options.remote_config.body_temporal_smoothing is True
    assert default_options.remote_config.body_foot_lock is True
    assert default_options.remote_config.body_foot_pin is True
    assert default_options.remote_config.body_contact_splice is True
    assert default_options.remote_config.sam3d_wrist_bone_lock is True
    assert default_options.remote_config.body_world_joint_visual_smoothing is True
    assert default_options.remote_config.target_mesh_frame_budget is None
    assert default_options.remote_config.mesh_byte_budget_mib == 300.0

    ball_aware_options = process_video.build_options_from_args(
        parser.parse_args(
            [
                "--video",
                str(video),
                "--out",
                str(tmp_path / "run2"),
                "--mesh-coverage-mode",
                "ball_aware",
                "--ball-proximity-m",
                "2.0",
                "--high-confidence-swing-floor",
                "0.7",
                "--target-mesh-frame-budget",
                "250",
                "--mesh-byte-budget-mib",
                "200",
                "--sam3d-body-input-size-px",
                "512",
                "--sam3d-crop-bucket-sizes",
                "4,8",
                "--no-sam3d-torch-compile",
                "--sam3d-compile-warmup-buckets",
                "4,8",
                "--serialize-tier2-mesh-vertices",
                "--fetch-body-monoliths",
                "--body-postchain",
                "raw",
                "--no-body-contact-splice",
                "--events-selected",
                str(events_selected),
                "--ball-track-arc-solved",
                str(ball_track_arc_solved),
            ]
        )
    )
    assert ball_aware_options.mesh_coverage_mode == "ball_aware"
    assert ball_aware_options.ball_proximity_m == 2.0
    assert ball_aware_options.high_confidence_swing_floor == 0.7
    assert ball_aware_options.target_mesh_frame_budget == 250
    assert ball_aware_options.mesh_byte_budget_mib is None
    assert ball_aware_options.events_selected == events_selected.resolve()
    assert ball_aware_options.ball_track_arc_solved == ball_track_arc_solved.resolve()
    assert ball_aware_options.remote_config.sam3d_body_input_size_px == 512
    assert ball_aware_options.remote_config.sam3d_crop_bucket_sizes == (4, 8)
    assert ball_aware_options.remote_config.sam3d_torch_compile is False
    assert ball_aware_options.remote_config.sam3d_compile_warmup_buckets == (4, 8)
    assert ball_aware_options.remote_config.sam3d_skip_tier2_mesh_vertices is False
    assert ball_aware_options.remote_config.fetch_body_monoliths is True
    assert ball_aware_options.remote_config.body_postchain_mode == "raw"
    assert ball_aware_options.remote_config.body_temporal_smoothing is False
    assert ball_aware_options.remote_config.body_foot_lock is False
    assert ball_aware_options.remote_config.body_foot_pin is False
    assert ball_aware_options.remote_config.body_contact_splice is False
    assert ball_aware_options.remote_config.sam3d_wrist_bone_lock is False
    assert ball_aware_options.remote_config.body_world_joint_visual_smoothing is False
    assert ball_aware_options.remote_config.target_mesh_frame_budget == 250
    assert ball_aware_options.remote_config.mesh_byte_budget_mib is None

    no_cap_options = process_video.build_options_from_args(
        parser.parse_args(
            [
                "--video",
                str(video),
                "--out",
                str(tmp_path / "run3"),
                "--target-mesh-frame-budget",
                "0",
            ]
        )
    )
    assert no_cap_options.target_mesh_frame_budget is None  # 0 means "no cap"
    assert no_cap_options.mesh_byte_budget_mib is None


def test_process_video_cli_help_direct_reference() -> None:
    """Direct-CLI reference test for scripts/racketsport/process_video.py
    (test_scaffold_tool_index.py's coverage audit requires the literal
    command path to appear in a real test file, matched by subprocess-
    invoking it -- --help is the only invocation that needs no video/GPU/
    network). Also pins that this lane's new ball-aware mesh-scheduling
    flags are actually wired into the parser."""
    command_path = "scripts/racketsport/process_video.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--video" in completed.stdout
    assert "--mesh-coverage-mode" in completed.stdout
    assert "--target-mesh-frame-budget" in completed.stdout
    assert "--mesh-byte-budget-mib" in completed.stdout
    assert "--ball-proximity-m" in completed.stdout
    assert "--high-confidence-swing-floor" in completed.stdout
    assert "--events-selected" in completed.stdout
    assert "--ball-track-arc-solved" in completed.stdout
    assert "--enable-camera-motion" in completed.stdout
    assert "--disable-camera-motion" in completed.stdout
    assert "--sam3d-body-input-size-px" in completed.stdout
    assert "--sam3d-crop-bucket-sizes" in completed.stdout
    assert "--sam3d-compile-warmup-buckets" in completed.stdout
    assert "--serialize-tier2-mesh-vertices" in completed.stdout
    assert "--fetch-body-monoliths" in completed.stdout
    assert "--body-postchain" in completed.stdout
    assert "--no-body-temporal-smoothing" in completed.stdout
    assert "--no-body-foot-lock" in completed.stdout
    assert "--no-body-foot-pin" in completed.stdout
    assert "--no-body-contact-splice" in completed.stdout
    assert "--no-body-wrist-lock" in completed.stdout
    assert "--no-body-world-joint-visual-smoothing" in completed.stdout


def test_process_video_cli_rejects_missing_video() -> None:
    command_path = "scripts/racketsport/process_video.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--video", "/nonexistent/does-not-exist.mp4", "--court-corners", "/nonexistent/corners.json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "video not found" in completed.stderr or "video not found" in completed.stdout


def test_calibration_stage_skips_when_already_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())

    calls: list[str] = []
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", lambda **kwargs: calls.append(kwargs["stage"]) or {"status": "pass", "stages": []})

    pipeline = process_video.ProcessVideoPipeline(options)
    outcome = pipeline._stage_ingest()
    assert outcome.status == "ran"
    outcome = pipeline._stage_calibration()

    assert outcome.status == "skipped"
    assert calls == []  # never invoked the real spine


def test_calibration_stage_hard_fails_without_court_corners_or_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    with pytest.raises(process_video._HardStageFailure, match="court-corners"):
        pipeline._stage_calibration()


def test_court_proposals_preview_writes_artifact_without_bypassing_calibration(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.court_proposals_preview = True
    options.max_frames = 2
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    with pytest.raises(process_video._HardStageFailure, match="court proposals preview"):
        pipeline._stage_calibration()

    proposal_path = options.clip_dir / "court_proposals.json"
    correction_path = options.clip_dir / "court_correction_task.json"
    assert proposal_path.is_file()
    assert correction_path.is_file()
    proposal = json.loads(proposal_path.read_text())
    correction = json.loads(correction_path.read_text())
    assert proposal["verified"] is False
    assert proposal["not_cal3_verified"] is True
    assert proposal["ranking"]["abstain"] is True
    assert correction["reason"] == "court_proposals_preview_not_trusted_calibration"


def test_tracking_stage_reuses_precomputed_tracks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    reuse_tracks = tmp_path / "champion_tracks.json"
    _write_json(reuse_tracks, _tracks_payload())

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.tracks_reuse = reuse_tracks
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    calls: list[str] = []
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", lambda **kwargs: calls.append(kwargs["stage"]) or {"status": "pass", "stages": []})

    outcome = pipeline._stage_tracking()
    assert outcome.status == "reused"
    assert calls == []
    assert (options.clip_dir / "tracks.json").is_file()
    assert pipeline.trust_bands["track"]["badge"] == "low_confidence"


def test_tracking_stage_degrades_when_live_tracking_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False  # allow attempting live tracking
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    def _raise(**kwargs):  # noqa: ANN001
        raise RuntimeError("ultralytics is required for real YOLO26m BoT-SORT-ReID tracking")

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _raise)

    outcome = pipeline._stage_tracking()
    assert outcome.status == "degraded"
    assert "ultralytics" in " ".join(outcome.notes)


def test_tracking_stage_succeeds_despite_advisory_blocked_overall_spine_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task #45 S1 (generalized): a real tracking run that wrote a valid tracks.json
    must not be reported as "degraded" just because run_pipeline's aggregate status is
    "blocked" -- which, downstream of an advisory-not-ready trusted calibration (S1),
    it always will be, since pipeline_contracts' readiness report inherits calibration's
    unresolved evidence-readiness blocker into every later stage's closure. Only a
    literal failure of the "tracking" stage itself should degrade this stage."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        _write_json(Path(run_dir) / "tracks.json", _tracks_payload())
        return {
            "status": "blocked",
            "stages": [
                {"stage": "calibration", "status": "ran", "notes": ["ADVISORY: automatic court evidence not ready"]},
                {"stage": "tracking", "status": "ran", "notes": ["ran BoT-SORT"]},
            ],
            "readiness": {"status": "not_ready"},
        }

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    outcome = pipeline._stage_tracking()
    assert outcome.status == "ran"
    assert (options.clip_dir / "tracks.json").is_file()


def test_tracking_stage_uses_runtime_manifest_with_local_yolo_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    repo_root = tmp_path / "repo"
    local_yolo = repo_root / "models" / "checkpoints" / "yolo26m.pt"
    local_yolo.parent.mkdir(parents=True)
    local_yolo.write_bytes(b"fake checkpoint")
    manifest_path = repo_root / "models" / "MANIFEST.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "models": [
                {
                    "id": "yolo26m",
                    "local_path": "/home/arnavchokshi/pickleball_git/models/checkpoints/yolo26m.pt",
                    "sha256": "fake",
                }
            ],
        },
    )
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.manifest_path = manifest_path
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    captured_manifest: dict[str, Any] = {}

    def _fake(*, run_dir, stage, manifest_path, **kwargs):  # noqa: ANN001
        assert stage == "tracking"
        captured_manifest["path"] = Path(manifest_path)
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        [entry] = [model for model in payload["models"] if model["id"] == "yolo26m"]
        assert entry["local_path"] == str(local_yolo)
        _write_json(Path(run_dir) / "tracks.json", _tracks_payload())
        return {"status": "pass", "stages": [{"stage": "tracking", "status": "ran", "notes": []}]}

    monkeypatch.setattr(process_video, "ROOT", repo_root)
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    outcome = pipeline._stage_tracking()

    assert outcome.status == "ran"
    assert captured_manifest["path"] == options.clip_dir / "runtime_model_manifest.json"


def test_global_association_uses_resolved_reid_device_and_batch64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.reid_model = tmp_path / "osnet.pt"
    options.reid_model.write_bytes(b"fake checkpoint")
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(
        options.clip_dir / "tracked_detections.json",
        {
            "fps": 30.0,
            "frames": [
                {
                    "frame": 0,
                    "detections": [
                        {"bbox": [100.0, 100.0, 130.0, 180.0], "conf": 0.91, "class": "person", "track_id": 10}
                    ],
                }
            ],
        },
    )
    pipeline = process_video.ProcessVideoPipeline(options)

    captured: dict[str, Any] = {}

    def _fake_resolve(requested_device: str | None) -> str:
        captured["requested_device"] = requested_device
        return "mps"

    def _fake_authority(**kwargs):  # noqa: ANN001
        config = kwargs["config"]
        captured["reid_device"] = config.reid_device
        captured["reid_batch_size"] = config.reid_batch_size
        out_dir = Path(kwargs["out_dir"])
        refined_tracks = out_dir / "tracks.json"
        _write_json(refined_tracks, _tracks_payload())
        return {"tracks_path": str(refined_tracks)}

    monkeypatch.setattr(process_video, "resolve_reid_device", _fake_resolve, raising=False)
    monkeypatch.setattr(process_video, "run_raw_pool_authority_candidate", _fake_authority)

    notes = pipeline._attempt_global_association()

    assert captured["requested_device"] is None
    assert captured["reid_device"] == "mps"
    assert captured["reid_batch_size"] == 64
    assert any("device=mps" in note and "batch=64" in note for note in notes)


def test_global_association_default_profile_declares_wolverine_internal_val_tuning() -> None:
    profile = process_video.RAW_POOL_GLOBAL_ASSOCIATION_PROFILES[process_video.DEFAULT_GLOBAL_ASSOCIATION_PROFILE]

    assert "Wolverine internal-val" in profile.note
    config, profile_name = process_video._raw_pool_authority_config_for_profile(
        process_video.DEFAULT_GLOBAL_ASSOCIATION_PROFILE,
        expected_players=4,
        reid_device="mps",
        reid_batch_size=64,
    )
    assert profile_name == process_video.DEFAULT_GLOBAL_ASSOCIATION_PROFILE
    assert config.court_margin_m == 2.0
    assert config.reid_device == "mps"


def test_no_flag_global_association_profile_is_manifest_default_for_preregistered_clips(tmp_path: Path) -> None:
    local_wolverine_pool = tmp_path / "local_wolverine_pool"
    scaled_wolverine_pool = tmp_path / "scaled_wolverine_pool"
    _write_json(
        local_wolverine_pool / "metrics.json",
        {
            "counts": {
                "bbox_scale_status": "identity",
                "bbox_scale_x": 1.0,
                "bbox_scale_y": 1.0,
                "source_width": 1920,
                "source_height": 1080,
                "calibration_width": 1920,
                "calibration_height": 1080,
            }
        },
    )
    _write_json(
        scaled_wolverine_pool / "metrics.json",
        {
            "counts": {
                "bbox_scale_x": 0.5,
                "bbox_scale_y": 0.5,
                "source_width": 1920,
                "source_height": 1080,
                "calibration_width": 960,
                "calibration_height": 540,
            }
        },
    )

    for clip, raw_pool_dir in (
        ("wolverine_mixed_0200_mid_steep_corner", local_wolverine_pool),
        ("wolverine_mixed_0200_mid_steep_corner", scaled_wolverine_pool),
        ("wolverine_mixed_0200_mid_steep_corner", None),
        ("burlington_gold_0300_low_steep_corner", None),
        ("outdoor_webcam_iynbd_1500_long_high_baseline", None),
    ):
        assert (
            process_video._default_raw_pool_authority_profile_for_clip(clip, raw_pool_dir=raw_pool_dir)
            == process_video.DEFAULT_GLOBAL_ASSOCIATION_PROFILE
        )

    config, profile_name = process_video._raw_pool_authority_config_for_profile(
        "outdoor_preregistered_unshopped_base",
        expected_players=4,
        reid_device="mps",
        reid_batch_size=64,
    )

    assert profile_name == "outdoor_preregistered_unshopped_base"
    assert config.court_margin_m == 3.0
    assert config.min_conf == 0.0
    assert config.appearance_weight == 1.0
    assert config.max_gap_fill_frames == 24
    assert config.max_merge_cost == 3.0
    assert config.cardinality_backfill is True

    config, profile_name = process_video._raw_pool_authority_config_for_profile(
        "wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill",
        expected_players=4,
        reid_device="mps",
        reid_batch_size=64,
    )

    assert profile_name == "wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill"
    assert config.court_margin_m == 1.0
    assert config.min_conf == 0.3
    assert config.appearance_weight == 0.5
    assert config.max_gap_fill_frames == 48
    assert config.max_merge_cost == 2.0
    assert config.cardinality_backfill is True


# ---------------------------------------------------------------------------
# frames stage (Task #46: body_frames/ JPEG extraction)
# ---------------------------------------------------------------------------


def test_frames_stage_blocked_without_tracks(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_frames()
    assert outcome.status == "blocked"
    assert "requires tracks.json" in " ".join(outcome.notes)


def test_frames_stage_skips_when_no_gpu(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = True  # _base_options() default, kept explicit for clarity
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_frames()
    assert outcome.status == "skipped"
    assert "--no-gpu" in " ".join(outcome.notes)
    assert not (options.clip_dir / "body_frames").exists()


def test_frames_stage_reuses_existing_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    body_frames = options.clip_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"jpg")
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_frames()
    assert outcome.status == "skipped"
    assert "reusing" in " ".join(outcome.notes)
    assert outcome.metrics["frame_count"] == 1


def test_frames_stage_extracts_real_jpegs_from_tracked_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)  # 5 frames @ 30fps, 960x540
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())  # both players tracked at t=0.0 -> frame 0

    outcome = pipeline._stage_frames()

    assert outcome.status == "ran"
    assert outcome.metrics["frame_count"] == 1
    assert outcome.metrics["schedule_source"] == "tracks_union"
    assert outcome.metrics["capped"] is False
    assert outcome.metrics["total_mb"] >= 0.0
    assert (options.clip_dir / "body_frames" / "frame_000000.jpg").is_file()
    assert any("extracted 1 scheduled JPEG" in note for note in outcome.notes)


def test_frames_stage_passes_max_frames_and_frame_compute_plan_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.max_frames = 42
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())

    calls: list[dict[str, Any]] = []

    def _fake_materialize(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frame_000000.jpg").write_bytes(b"jpg")
        return {
            "schedule": {"capped": False, "source": "tracks_union"},
            "notes": ["fake schedule note"],
            "extraction": {"extracted_frame_count": 1},
            "out_dir": str(out_dir),
            "frame_count": 1,
            "total_bytes": 3,
        }

    monkeypatch.setattr(process_video, "materialize_process_video_frames", _fake_materialize)

    outcome = pipeline._stage_frames()

    assert outcome.status == "ran"
    [call] = calls
    assert call["max_frames"] == 42
    assert call["frame_compute_plan_path"] == options.clip_dir / "frame_compute_plan.json"
    assert call["out_dir"] == options.clip_dir / "body_frames"


def test_frames_stage_degrades_when_video_unreadable(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    # simulate a corrupt/unreadable ingested source (bypasses _stage_ingest()'s
    # own cv2 probe, which would otherwise fail ingest itself, not frames).
    (options.clip_dir / "source.mp4").write_bytes(b"this is not a real video file")
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())

    outcome = pipeline._stage_frames()

    assert outcome.status == "degraded"
    assert "body_frames/ extraction unavailable" in " ".join(outcome.notes)
    assert "degrade to court/skeleton-only" in " ".join(outcome.notes)
    assert not (options.clip_dir / "body_frames" / "frame_000000.jpg").is_file()


def test_pose_stage_is_removed_from_production_process_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())

    def _fail_if_pose_runs(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("process_video must not call the removed production pose stage")

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fail_if_pose_runs)
    monkeypatch.setattr(process_video, "dispatch_body_stage", _fail_if_pose_runs)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_pose()

    assert outcome.status == "skipped"
    assert not (options.clip_dir / "skeleton3d.json").exists()
    assert any("SAM-3D BODY" in note for note in outcome.notes)


def test_events_stage_runs_with_sam3d_skeleton_and_regenerates_frame_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.max_players = 2
    _clip_dir_with_tracks_only(options)
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    monkeypatch.setattr(
        process_video,
        "build_wrist_velocity_peaks_from_file",
        lambda *args, **kwargs: {"schema_version": 1, "source": "test", "summary": {"peak_count": 1}, "peaks": []},
    )
    monkeypatch.setattr(
        process_video,
        "build_ball_inflections_from_ball_track",
        lambda payload, **kwargs: {"schema_version": 1, "source": "test", "summary": {"candidate_count": 1}, "candidates": []},
    )
    monkeypatch.setattr(process_video, "fuse_contact_windows_from_cue_payloads", lambda **kwargs: _contact_windows_payload())

    outcome = pipeline._stage_events()

    assert outcome.status == "ran"
    assert outcome.metrics["contact_event_count"] == 1
    plan = json.loads((options.clip_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    assert plan["summary"]["deep_mesh_frame_count"] > 0
    assert any("cues used" in note for note in outcome.notes)


def test_events_stage_fails_closed_without_prebody_sam3d_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_audio = True
    options.max_players = 2
    _clip_dir_with_tracks_only(options)
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload())

    monkeypatch.setattr(
        process_video,
        "build_ball_inflections_from_ball_track",
        lambda payload, **kwargs: {"schema_version": 1, "source": "test", "summary": {"candidate_count": 0}, "candidates": []},
    )
    monkeypatch.setattr(process_video, "fuse_contact_windows_from_cue_payloads", lambda **kwargs: _contact_windows_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_events()

    assert outcome.status == "ran"
    wrist = json.loads((options.clip_dir / "wrist_velocity_peaks.json").read_text(encoding="utf-8"))
    assert wrist["status"] == "blocked"
    assert "missing_sam3d_skeleton3d" in wrist["blockers"]
    assert any("SAM-3D skeleton unavailable" in note for note in outcome.notes)


def test_local_body_stage_succeeds_despite_advisory_blocked_overall_spine_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        return {
            "status": "blocked",
            "stages": [{"stage": "calibration", "status": "ran", "notes": []}, {"stage": "body", "status": "ran", "notes": []}],
            "readiness": {"status": "not_ready"},
        }

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    outcome = pipeline._run_body_local()
    assert outcome.status == "ran"


def test_local_body_stage_uses_vm_proven_no_moge_body_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = False
    monkeypatch.delenv("FAST_SAM_PYTHON", raising=False)
    _clip_dir_with_tracks_and_sam3d_skeleton(options)

    calls: list[dict[str, Any]] = []

    def _fake_run_pipeline(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return {"stages": [{"stage": "body", "status": "ran", "notes": []}]}

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake_run_pipeline)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._run_body_local()

    assert outcome.status == "ran"
    [call] = calls
    body_runner = call["runners"]["body"]
    assert body_runner.detector_name == ""
    assert body_runner.fov_name == ""
    assert str(body_runner.fast_sam_repo) == options.remote_config.fast_sam_root
    assert os.environ["FAST_SAM_PYTHON"] == options.remote_config.fast_sam_python


def test_tracking_and_local_body_stages_thread_reuse_existing_stage_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_video.py calls orchestrator.run_pipeline() once per active model stage
    against the same clip_dir over the course of one run -- tracking/body-local must
    opt into reuse_existing_stage_artifacts=True so a dependency stage that already
    completed earlier in *this* run (e.g. calibration) is treated as authoritative
    instead of being re-derived (and potentially re-failing) inside a later stage's call."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    calls: list[dict[str, Any]] = []

    def _fake(*, clip, inputs_dir, run_dir, stage, **kwargs):  # noqa: ANN001
        calls.append({"stage": stage, **kwargs})
        return {"status": "fail", "stages": [{"stage": stage, "status": "fail", "notes": ["stub"]}]}

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake)

    # _stage_tracking() attempts live tracking (no tracks.json exists yet) -- captures
    # the "tracking" call.
    pipeline._stage_tracking()
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    pipeline._run_body_local()

    by_stage = {call["stage"]: call for call in calls}
    assert by_stage["tracking"]["reuse_existing_stage_artifacts"] is True
    assert by_stage["body"]["reuse_existing_stage_artifacts"] is True


def test_tracking_stage_runs_raw_pool_authority_over_exported_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.reid_model = tmp_path / "osnet.pt"
    options.reid_model.write_bytes(b"fake test checkpoint")
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    raw_pool_payload = {
        "fps": 30.0,
        "source_width": 1920,
        "source_height": 1080,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [200.0, 200.0, 260.0, 360.0], "conf": 0.91, "class": "person", "track_id": 10}
                ],
            }
        ],
    }
    scaled_pool_payload = {
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [100.0, 100.0, 130.0, 180.0], "conf": 0.91, "class": "person", "track_id": 10}
                ],
            }
        ],
    }
    metrics_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_tracker_candidate",
        "counts": {
            "source_width": 1920,
            "source_height": 1080,
            "calibration_width": 960,
            "calibration_height": 540,
            "bbox_scale_x": 0.5,
            "bbox_scale_y": 0.5,
        },
    }

    def _fake_run_pipeline(*, run_dir, stage, **kwargs):  # noqa: ANN001
        assert stage == "tracking"
        _write_json(Path(run_dir) / "tracks.json", _tracks_payload())
        _write_json(Path(run_dir) / "raw_tracked_detections.json", raw_pool_payload)
        _write_json(Path(run_dir) / "tracked_detections.json", scaled_pool_payload)
        _write_json(Path(run_dir) / "metrics.json", metrics_payload)
        return {"status": "pass", "stages": [{"stage": "tracking", "status": "ran", "notes": []}]}

    calls: list[dict[str, Any]] = []

    def _fake_raw_pool_authority(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        refined = _tracks_payload()
        refined["players"][0]["id"] = 77
        _write_json(out_dir / "tracks.json", refined)
        _write_json(out_dir / "raw_pool_authority_summary.json", {"status": "pass"})
        return {"tracks_path": str(out_dir / "tracks.json"), "summary_path": str(out_dir / "raw_pool_authority_summary.json")}

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(process_video, "run_raw_pool_authority_candidate", _fake_raw_pool_authority, raising=False)
    monkeypatch.setattr(process_video, "resolve_reid_device", lambda device: "mps")

    outcome = pipeline._stage_tracking()

    assert outcome.status == "ran"
    assert calls
    [call] = calls
    assert call["raw_pool_dir"] == options.clip_dir
    assert call["candidate"] == "botsort_loose_pool_raw"
    assert call["reid_model_path"] == options.reid_model
    assert call["ground_truth_path"] is None
    assert call["config"].reid_device == "mps"
    assert call["config"].reid_batch_size == 64
    assert call["config"].court_margin_m == 2.0
    assert (options.clip_dir / "raw_tracked_detections.json").is_file()
    assert (options.clip_dir / "tracked_detections.json").is_file()
    assert (options.clip_dir / "metrics.json").is_file()
    rewritten = json.loads((options.clip_dir / "tracks.json").read_text(encoding="utf-8"))
    assert rewritten["players"][0]["id"] == 77
    assert any("raw-pool global association" in note and "device=mps" in note for note in outcome.notes)


def test_pipeline_never_raises_on_unexpected_stage_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload())
    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    # Pre-seed a valid court_calibration.json so the calibration stage takes
    # the resume/skip path instead of running the real spine (which fails
    # closed on this synthetic all-black test video's auto court-line
    # evidence -- a real, correct hard-fail unrelated to what this test
    # is checking: that a later stage's *unexpected* crash doesn't abort
    # the whole run).
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    pipeline = process_video.ProcessVideoPipeline(options)

    def _boom():
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(pipeline, "_stage_ball", _boom)
    summary = pipeline.run()

    by_stage = {s["stage"]: s for s in summary["stages"]}
    assert by_stage["ball"]["status"] == "degraded"
    assert "boom" in " ".join(by_stage["ball"]["notes"])
    # the pipeline kept going past the crashing stage instead of aborting.
    assert "events" in by_stage
    assert "world" in by_stage


# ---------------------------------------------------------------------------
# ball stage
# ---------------------------------------------------------------------------


def test_ball_arc_stage_passes_event_sidecars_to_frozen_chain_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload(frame_count=9))
    _write_json(options.clip_dir / "contact_windows.json", _contact_windows_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "net_plane.json", {"artifact_type": "racketsport_net_plane", "height_m": 0.86})
    _write_json(options.clip_dir / "ball_candidates.json", _ball_candidates_payload())
    pipeline = process_video.ProcessVideoPipeline(options)
    captured: dict[str, Any] = {}

    def _fake_default_chain(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        out_dir = Path(kwargs["out_dir"])
        _write_json(
            out_dir / "ball_bounce_candidates.json",
            {"artifact_type": "racketsport_ball_bounce_candidates", "summary": {"final_candidate_count": 1}, "candidates": []},
        )
        _write_json(
            out_dir / "ball_track_arc_solved.json",
            {
                "artifact_type": "racketsport_ball_track_arc_solved",
                "status": "ran",
                "summary": {"coverage_world_xyz_count": 7, "segment_count": 1},
                "frames": [],
            },
        )
        _write_json(
            out_dir / "ball_flight_sanity.json",
            {"artifact_type": "racketsport_ball_flight_sanity", "summary": {"demoted_frame_count": 2}},
        )
        _write_json(
            out_dir / "ball_arc_render.json",
            {"artifact_type": "racketsport_ball_arc_render", "summary": {"sample_count": 11, "bridge_sample_count": 3}},
        )
        return {
            "status": "ran",
            "summary": {
                "auto_bounce_candidate_count": 1,
                "coverage_world_xyz_count": 7,
                "segment_count": 1,
                "ball_arc_render_sample_count": 11,
                "ball_arc_render_bridge_sample_count": 3,
                "flight_sanity_demoted_frame_count": 2,
            },
        }

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", _fake_default_chain, raising=False)

    outcome = pipeline._stage_ball_arc()

    assert outcome.status == "ran"
    assert captured["clip"] == options.clip
    assert captured["ball_track_path"] == options.clip_dir / "ball_track.json"
    assert captured["court_calibration_path"] == options.clip_dir / "court_calibration.json"
    assert captured["contact_windows_path"] == options.clip_dir / "contact_windows.json"
    assert captured["skeleton3d_path"] == options.clip_dir / "skeleton3d.json"
    assert captured["net_plane_path"] == options.clip_dir / "net_plane.json"
    assert captured["rally_spans_path"] is None
    assert captured["ball_candidate_paths"] == [options.clip_dir / "ball_candidates.json"]
    assert (options.clip_dir / "ball_bounce_candidates.json").is_file()
    assert (options.clip_dir / "ball_track_arc_solved.json").is_file()
    assert (options.clip_dir / "ball_arc_render.json").is_file()
    assert (options.clip_dir / "ball_flight_sanity.json").is_file()
    assert outcome.artifacts == [
        "ball_bounce_candidates.json",
        "ball_track_arc_solved.json",
        "ball_arc_render.json",
        "ball_flight_sanity.json",
        "ball_chain_manifest.json",
    ]
    assert outcome.metrics["solver_status"] == "ran"
    assert outcome.metrics["ball_arc_render_sample_count"] == 11
    assert outcome.metrics["ball_arc_render_bridge_sample_count"] == 3
    assert outcome.metrics["flight_sanity_demoted_frame_count"] == 2


def test_ball_arc_stage_accepts_explicit_ball_candidate_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.ball_candidates_reuse = (tmp_path / "wasb_candidates.json", tmp_path / "tracknet_candidates.json")
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload(frame_count=9))
    for path in options.ball_candidates_reuse:
        _write_json(path, _ball_candidates_payload())
    pipeline = process_video.ProcessVideoPipeline(options)
    captured: dict[str, Any] = {}

    def _fake_default_chain(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        out_dir = Path(kwargs["out_dir"])
        _write_json(
            out_dir / "ball_bounce_candidates.json",
            {"artifact_type": "racketsport_ball_bounce_candidates", "summary": {"final_candidate_count": 1}, "candidates": []},
        )
        _write_json(
            out_dir / "ball_track_arc_solved.json",
            {"artifact_type": "racketsport_ball_track_arc_solved", "status": "ran", "summary": {}, "frames": []},
        )
        _write_json(
            out_dir / "ball_flight_sanity.json",
            {"artifact_type": "racketsport_ball_flight_sanity", "summary": {}},
        )
        return {"status": "ran", "summary": {}}

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", _fake_default_chain, raising=False)

    outcome = pipeline._stage_ball_arc()

    assert outcome.status == "ran"
    assert captured["ball_candidate_paths"] == list(options.ball_candidates_reuse)


def test_wasb_ball_runtime_emits_candidate_sidecar_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    checkpoint = tmp_path / "wasb.pth.tar"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    wasb_repo = tmp_path / "WASB-SBDT"
    wasb_repo.mkdir()
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.wasb_checkpoint = checkpoint
    options.wasb_repo = wasb_repo
    pipeline = process_video.ProcessVideoPipeline(options)
    captured: dict[str, Any] = {}

    def _fake_run_wasb_or_convert(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        _write_json(kwargs["out"], _ball_track_payload(frame_count=5))
        if kwargs.get("emit_candidates"):
            _write_json(Path(kwargs["out"]).with_name("ball_candidates.json"), _ball_candidates_payload())
        return {"status": "tested", "out": str(kwargs["out"])}

    from threed.racketsport import wasb_adapter

    monkeypatch.setattr(wasb_adapter, "run_wasb_or_convert", _fake_run_wasb_or_convert)

    assert pipeline._run_wasb_zero_shot(options.clip_dir / "ball_track.json") is True

    assert captured["emit_candidates"] is True
    assert captured["candidate_top_k"] == 5
    assert (options.clip_dir / "ball_candidates.json").is_file()


def test_wasb_ball_runtime_can_opt_out_of_candidate_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    checkpoint = tmp_path / "wasb.pth.tar"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    wasb_repo = tmp_path / "WASB-SBDT"
    wasb_repo.mkdir()
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.wasb_checkpoint = checkpoint
    options.wasb_repo = wasb_repo
    options.emit_ball_candidates = False
    pipeline = process_video.ProcessVideoPipeline(options)
    captured: dict[str, Any] = {}

    def _fake_run_wasb_or_convert(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        _write_json(kwargs["out"], _ball_track_payload(frame_count=5))
        return {"status": "tested", "out": str(kwargs["out"])}

    from threed.racketsport import wasb_adapter

    monkeypatch.setattr(wasb_adapter, "run_wasb_or_convert", _fake_run_wasb_or_convert)

    assert pipeline._run_wasb_zero_shot(options.clip_dir / "ball_track.json") is True

    assert captured["emit_candidates"] is False
    assert not (options.clip_dir / "ball_candidates.json").exists()


def test_ball_arc_stage_opt_out_skips_default_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_ball_arc = True
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload(frame_count=9))
    pipeline = process_video.ProcessVideoPipeline(options)

    def _unexpected_default_chain(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("ball_arc helper should not run when --no-ball-arc is set")

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", _unexpected_default_chain, raising=False)

    outcome = pipeline._stage_ball_arc()

    assert outcome.status == "skipped"
    assert "--no-ball-arc set" in " ".join(outcome.notes)
    assert not (options.clip_dir / "ball_track_arc_solved.json").exists()


def test_ball_arc_stage_failure_degrades_without_crashing_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload(frame_count=9))
    pipeline = process_video.ProcessVideoPipeline(options)

    def _failing_default_chain(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", _failing_default_chain, raising=False)

    outcome = pipeline._stage_ball_arc()

    assert outcome.status == "degraded"
    assert "solver exploded" in " ".join(outcome.notes)
    assert not (options.clip_dir / "ball_track_arc_solved.json").exists()


def test_ball_stage_reuses_precomputed_ball_track_and_runs_bounce_inout(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))
    reuse_ball = tmp_path / "wasb_ball_track.json"
    _write_json(reuse_ball, _ball_track_payload(frame_count=5))

    options = _base_options(tmp_path, video=video, court_corners=corners_path)
    options.skip_ball = False
    options.ball_track_reuse = reuse_ball
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()
    assert outcome.status == "reused"
    assert (options.clip_dir / "ball_track.json").is_file()
    assert pipeline.trust_bands["ball"]["badge"] == "low_confidence"


def test_ball_stage_rescales_reused_ball_track_when_frame_count_matches_video_but_fps_differs(tmp_path: Path) -> None:
    video = tmp_path / "wolverine_300f_30fps.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    reuse_ball = tmp_path / "wolverine_wasb_fps60_ball_track.json"
    _write_json(reuse_ball, _wolverine_fps_mismatch_ball_track_payload(frame_count=300))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_ball = False
    options.ball_track_reuse = reuse_ball
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()

    assert outcome.status == "reused"
    normalized = json.loads((options.clip_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert normalized["fps"] == pytest.approx(30.0)
    assert len(normalized["frames"]) == 300
    assert normalized["frames"][148]["t"] == pytest.approx(148 / 30.0)
    assert normalized["frames"][-1]["t"] == pytest.approx(299 / 30.0)
    provenance = json.loads((options.clip_dir / "ball_track_timing_provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "rescaled_reused_ball_track_timestamps"
    assert provenance["source_fps"] == pytest.approx(60.0)
    assert provenance["video_fps"] == pytest.approx(30.0)
    assert provenance["source_frame_count"] == 300
    assert provenance["video_frame_count"] == 300
    assert outcome.metrics["ball_timeline_coverage_before"] == pytest.approx(0.5)
    assert outcome.metrics["ball_timeline_coverage_after"] == pytest.approx(1.0)
    assert any("rescaled reused ball_track timestamps" in warning for warning in outcome.metrics["warnings"])
    assert any("rescaled reused ball_track timestamps" in note for note in outcome.notes)


def test_ball_stage_normalizes_reused_ball_track_to_frame_times_table(tmp_path: Path) -> None:
    video = tmp_path / "vfr_clip.mp4"
    _make_video(video, frame_count=3, fps=30.0)
    reuse_ball = tmp_path / "constant_fps_ball_track.json"
    _write_json(reuse_ball, _ball_track_payload(frame_count=3, fps=30.0))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_ball = False
    options.ball_track_reuse = reuse_ball
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        options.clip_dir / "frame_times.json",
        {
            "artifact_type": "racketsport_frame_times",
            "provenance": "ffprobe_pts",
            "frames": [
                {"frame_idx": 0, "pts_s": 0.0},
                {"frame_idx": 1, "pts_s": 0.1},
                {"frame_idx": 2, "pts_s": 0.3},
            ],
        },
    )
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_ball()

    assert outcome.status == "reused"
    normalized = json.loads((options.clip_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert [frame["t"] for frame in normalized["frames"]] == [0.0, 0.1, 0.3]
    provenance = json.loads((options.clip_dir / "ball_track_timing_provenance.json").read_text(encoding="utf-8"))
    assert provenance["normalization"] == "t=frame_times[index]"
    assert provenance["frame_times_path"] == str(options.clip_dir / "frame_times.json")


def test_ball_stage_rescales_exact_wolverine_fps_mismatch_fixture(tmp_path: Path) -> None:
    """Regression fixture for manager diagnosis:
    300 WASB frames stamped at 60fps reused against a 300-frame 30fps Wolverine clip.
    """

    video = tmp_path / "wolverine_mixed_0200_mid_steep_corner.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    reuse_ball = tmp_path / "ball_track.json"
    _write_json(reuse_ball, _wolverine_fps_mismatch_ball_track_payload(frame_count=300))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip = "wolverine_mixed_0200_mid_steep_corner"
    options.skip_ball = False
    options.ball_track_reuse = reuse_ball
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()

    normalized = json.loads((options.clip_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert outcome.status == "reused"
    assert normalized["fps"] == pytest.approx(30.0)
    assert normalized["frames"][-1]["t"] == pytest.approx(9.9666666667)
    assert outcome.metrics["ball_timeline_coverage_before"] == pytest.approx(0.5)
    assert outcome.metrics["ball_timeline_coverage_after"] == pytest.approx(1.0)


def test_ball_stage_fails_closed_when_reused_ball_track_frame_count_mismatches_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    reuse_ball = tmp_path / "short_ball_track.json"
    _write_json(reuse_ball, _wolverine_fps_mismatch_ball_track_payload(frame_count=299))

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_ball = False
    options.ball_track_reuse = reuse_ball
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    with pytest.raises(process_video._HardStageFailure, match="reused ball_track frame count mismatch"):
        pipeline._stage_ball()


def test_ball_stage_auto_discovers_precomputed_ball_track_for_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    clip = "wolverine_mixed_0200_mid_steep_corner"
    auto_track = (
        tmp_path
        / "runs"
        / "eval0"
        / "prototype_gate_h100_v2"
        / clip
        / "tracknet_smoke_0000_0010"
        / "ball_track_fusion_temporal_vball100_localtraj.json"
    )
    _write_json(auto_track, _ball_track_payload(frame_count=5))
    monkeypatch.setattr(process_video, "DEFAULT_RUN_ROOT", tmp_path / "runs")

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip = clip
    options.skip_ball = False
    options.ball_track_auto_discovery = True
    options.wasb_checkpoint = tmp_path / "does_not_exist.pth.tar"
    options.wasb_repo = tmp_path / "does_not_exist_repo"
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()

    assert outcome.status == "reused"
    assert (options.clip_dir / "ball_track.json").is_file()
    assert json.loads((options.clip_dir / "ball_track.json").read_text(encoding="utf-8"))["frames"]
    assert any("auto-discovered" in note and str(auto_track) in note for note in outcome.notes)
    assert any("not a verified fresh BALL run" in note for note in outcome.notes)
    assert pipeline.trust_bands["ball"]["badge"] == "low_confidence"


def test_ball_stage_does_not_auto_discover_precomputed_track_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    clip = "wolverine_mixed_0200_mid_steep_corner"
    auto_track = (
        tmp_path
        / "runs"
        / "eval0"
        / "prototype_gate_h100_v2"
        / clip
        / "tracknet_smoke_0000_0010"
        / "ball_track_fusion_temporal_vball100_localtraj.json"
    )
    _write_json(auto_track, _ball_track_payload())
    monkeypatch.setattr(process_video, "DEFAULT_RUN_ROOT", tmp_path / "runs")

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.clip = clip
    options.skip_ball = False
    options.wasb_checkpoint = tmp_path / "does_not_exist.pth.tar"
    options.wasb_repo = tmp_path / "does_not_exist_repo"
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()

    assert outcome.status == "blocked"
    assert not (options.clip_dir / "ball_track.json").is_file()
    assert any("--ball-track" in note and "--allow-auto-ball-track" in note for note in outcome.notes)


def test_ball_stage_degrades_without_checkpoint_or_reuse(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.skip_ball = False
    options.wasb_checkpoint = tmp_path / "does_not_exist.pth.tar"
    options.wasb_repo = tmp_path / "does_not_exist_repo"
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()

    outcome = pipeline._stage_ball()
    assert outcome.status == "blocked"
    assert not (options.clip_dir / "ball_track.json").is_file()


# ---------------------------------------------------------------------------
# rally gating stage
# ---------------------------------------------------------------------------


def test_rally_gating_stage_filters_dead_time_tracks_and_ball_with_trust_note(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=300, fps=30.0)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.rally_gating = True
    options.rally_gating_pad_seconds = 0.0
    options.skip_ball = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload_with_dead_time())
    ball_track = _ball_track_payload(frame_count=300)
    for frame in ball_track["frames"][1:]:
        frame["visible"] = False
    _write_json(options.clip_dir / "ball_track.json", ball_track)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_rally_gating()

    assert outcome.status == "ran"
    assert (options.clip_dir / "rally_spans.json").is_file()
    assert (options.clip_dir / "tracks_pre_rally_gating.json").is_file()
    assert (options.clip_dir / "ball_track_pre_rally_gating.json").is_file()
    filtered_tracks = json.loads((options.clip_dir / "tracks.json").read_text(encoding="utf-8"))
    filtered_ball = json.loads((options.clip_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert [frame["t"] for frame in filtered_tracks["players"][0]["frames"]] == [0.0]
    assert [frame["t"] for frame in filtered_ball["frames"]] == [0.0]
    assert outcome.metrics["track_frames_skipped"] == 1
    assert outcome.metrics["ball_frames_skipped"] == 299
    assert any("rally gating active" in note for note in outcome.notes)
    assert any("trust note" in note for note in outcome.notes)


def test_cli_parses_rally_gating_flag_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    args = process_video.build_arg_parser().parse_args(["--video", str(video), "--rally-gating", "--out", str(tmp_path / "run")])
    options = process_video.build_options_from_args(args)
    assert options.rally_gating is True


def test_cli_parses_camera_motion_path_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    camera_motion = tmp_path / "camera_motion.json"
    args = process_video.build_arg_parser().parse_args(
        ["--video", str(video), "--camera-motion", str(camera_motion), "--out", str(tmp_path / "run")]
    )
    options = process_video.build_options_from_args(args)
    assert options.camera_motion_path == camera_motion.resolve()


def test_cli_parses_default_camera_motion_stage_controls(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "default")]))
    enabled_options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--out", str(tmp_path / "enabled"), "--enable-camera-motion"])
    )
    options = process_video.build_options_from_args(
        parser.parse_args(
            [
                "--video",
                str(video),
                "--out",
                str(tmp_path / "run"),
                "--disable-camera-motion",
                "--camera-motion-estimator",
                "legacy",
                "--camera-motion-flow-backend",
                "raft-small",
                "--no-camera-motion-person-mask",
            ]
        )
    )

    assert default_options.skip_camera_motion is False
    assert default_options.enable_camera_motion is False
    assert enabled_options.skip_camera_motion is False
    assert enabled_options.enable_camera_motion is True
    assert options.skip_camera_motion is True
    assert options.enable_camera_motion is False
    assert options.camera_motion_estimator == "legacy"
    assert options.camera_motion_flow_backend == "raft-small"
    assert options.camera_motion_person_masks is False


def test_cli_parses_confidence_gate_opt_out_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")]))
    disabled_options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--no-confidence-gate", "--out", str(tmp_path / "run2")])
    )

    assert default_options.confidence_gate is True
    assert disabled_options.confidence_gate is False


def test_cli_parses_ball_arc_opt_out_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")]))
    disabled_options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--no-ball-arc", "--out", str(tmp_path / "run2")])
    )

    assert default_options.no_ball_arc is False
    assert disabled_options.no_ball_arc is True


def test_cli_parses_grounding_refine_opt_out_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")]))
    disabled_options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--no-grounding-refine", "--out", str(tmp_path / "run2")])
    )

    assert default_options.grounding_refine is True
    assert disabled_options.grounding_refine is False


def test_cli_defaults_to_no_auto_ball_track_unless_explicitly_allowed(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()

    default_options = process_video.build_options_from_args(parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")]))
    allowed_options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--allow-auto-ball-track", "--out", str(tmp_path / "run2")])
    )

    assert default_options.ball_track_auto_discovery is False
    assert allowed_options.ball_track_auto_discovery is True


# ---------------------------------------------------------------------------
# body stage: remote dispatch + --no-gpu degradation
# ---------------------------------------------------------------------------


def _clip_dir_with_tracks_and_sam3d_skeleton(options: process_video.PipelineOptions) -> None:
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())


def _clip_dir_with_tracks_only(options: process_video.PipelineOptions) -> None:
    """tracks.json only -- no pre-BODY skeleton3d.json at all."""

    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())


def test_body_stage_no_gpu_degrades_to_skeleton_only(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()
    assert outcome.status == "degraded"
    assert "SAM-3D BODY skipped" in " ".join(outcome.notes)
    assert not (options.clip_dir / "smpl_motion.json").is_file()


def test_body_stage_blocked_without_tracks_or_pose(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()
    assert outcome.status == "blocked"


def test_body_stage_remote_dispatch_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=12.3, notes=["dispatched to fake host"])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()
    assert outcome.status == "ran"
    assert outcome.trust_badge == "preview"
    assert (options.clip_dir / "smpl_motion.json").is_file()


def test_body_stage_reuses_sam3d_skeleton_and_gate_without_smpl_monolith(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_only(options)
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "body_full_clip_gate.json", {"schema_version": 1, "artifact_type": "body_full_clip_gate"})

    def _fail_dispatch(**_kwargs):  # noqa: ANN001
        raise AssertionError("no-force BODY reuse must not dispatch when skeleton3d.json + body_full_clip_gate.json are present")

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fail_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "skipped"
    assert outcome.artifacts == ["skeleton3d.json", "body_full_clip_gate.json"]
    joined_notes = " ".join(outcome.notes)
    assert "reusing completed BODY evidence without smpl_motion.json" in joined_notes
    assert "skeleton3d.json + body_full_clip_gate.json" in joined_notes


def test_body_stage_remote_dispatch_passes_body_frames_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task #46: remote dispatch must point at the exact directory the new
    "frames" stage extracts body_frames/ into, so the remote A100's own
    pose/body re-derivation (BODY depends_on pose) has the JPEGs it needs."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)

    calls: list[dict[str, Any]] = []

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    pipeline._stage_body()

    [call] = calls
    assert call["body_frames_dir"] == options.clip_dir / "body_frames"


def test_body_stage_remote_dispatch_failure_degrades_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        raise RemoteBodyDispatchError("shared GPU lock busy on arnavchokshi@34.126.67.233: did not acquire scripts/gpu-eval-run.sh within 60s")

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()
    assert outcome.status == "degraded"
    assert "lock busy" in " ".join(outcome.notes)
    assert "no fallback pose skeleton" in " ".join(outcome.notes)


def test_body_stage_remote_dispatch_proceeds_without_prebody_sam3d_skeleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote BODY dispatch must not require a preexisting local skeleton3d.json."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_only(options)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(
            clip_dir / "smpl_motion.json",
            {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []},
        )
        return RemoteBodyDispatchResult(
            status="ran",
            remote_run_dir="remote:/tmp/fake",
            synced_outputs=["smpl_motion.json"],
            wall_seconds=9.1,
            notes=["dispatched to fake host"],
        )

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()
    assert outcome.status == "ran"
    assert (options.clip_dir / "smpl_motion.json").is_file()
    assert outcome.trust_badge == "preview"


def test_body_stage_local_dispatch_no_longer_requires_prebody_sam3d_skeleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = False
    _clip_dir_with_tracks_only(options)

    def _fake_run_pipeline(**kwargs):  # noqa: ANN001
        assert kwargs["stage"] == "body"
        assert "runners" in kwargs
        return {"stages": [{"stage": "body", "status": "ran", "notes": []}]}

    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", _fake_run_pipeline)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "ran"
    assert not (options.clip_dir / "skeleton3d.json").exists()


# ---------------------------------------------------------------------------
# body stage: remote calibration seed (Task #46)
# ---------------------------------------------------------------------------


def _external_metric_calibration_with_points() -> dict[str, Any]:
    """A metric15-style calibration payload with image_pts/world_pts covering the
    four court corners (world x/y extremes) in a 1920x1080 pixel space, shaped
    like the real wolverine court_calibration_metric15pt.json."""

    payload = _external_metric_calibration_payload()
    payload["image_size"] = [1920, 1080]
    payload["image_pts"] = [
        [928.7, 929.8],   # near_left  (world -3.048, -6.7056)
        [1877.5, 524.8],  # near_right (world +3.048, -6.7056)
        [69.4, 381.6],    # far_left   (world -3.048, +6.7056)
        [653.4, 327.9],   # far_right  (world +3.048, +6.7056)
        [680.3, 428.6],   # a non-corner mid point (world 0, 0)
    ]
    payload["world_pts"] = [
        [-3.048, -6.7056, 0.0],
        [3.048, -6.7056, 0.0],
        [-3.048, 6.7056, 0.0],
        [3.048, 6.7056, 0.0],
        [0.0, 0.0, 0.0],
    ]
    return payload


def test_remote_seed_sidecar_derives_corner_taps_from_external_calibration() -> None:
    from threed.racketsport.schemas import CaptureSidecar

    payload = _external_metric_calibration_with_points()
    sidecar = process_video._remote_seed_capture_sidecar_from_calibration(payload, fps=30.0)

    validated = CaptureSidecar.model_validate(sidecar)
    assert validated.resolution == (1920, 1080)
    # SIDECAR_CORNER_ORDER: near_left, near_right, far_right, far_left --
    # near = the baseline lower in the image, left = smaller image x.
    assert sidecar["manual_court_taps"] == [[928.7, 929.8], [1877.5, 524.8], [653.4, 327.9], [69.4, 381.6]]
    # intrinsics carry over from the trusted calibration, not the coarse estimate.
    assert sidecar["intrinsics"]["fx"] == pytest.approx(1391.18)
    assert "derived_from_external_metric_calibration" in sidecar["capture_quality"]["reasons"]


def test_remote_seed_sidecar_raises_without_matched_points() -> None:
    payload = _external_metric_calibration_payload()  # no image_pts/world_pts corners
    payload.pop("image_pts", None)
    payload.pop("world_pts", None)

    with pytest.raises(ValueError, match="image_pts/world_pts"):
        process_video._remote_seed_capture_sidecar_from_calibration(payload, fps=30.0)


def test_body_stage_remote_dispatch_writes_calibration_seed_when_sidecar_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task #46: the metric15 variant never writes capture_sidecar.json locally,
    but the remote A100's committed orchestrator hard-requires one for its own
    calibration re-derivation -- dispatch must seed it from the external
    calibration's own corner points before rsync-up."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)
    _write_json(options.clip_dir / "court_calibration.json", _external_metric_calibration_with_points())

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        clip_dir = Path(kwargs["clip_dir"])
        # the seed must already exist by the time dispatch (and its rsync-up) runs.
        assert (clip_dir / "capture_sidecar.json").is_file()
        _write_json(clip_dir / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "ran"
    assert any("remote calibration seed: derived capture_sidecar.json" in note for note in outcome.notes)
    seed = json.loads((options.clip_dir / "capture_sidecar.json").read_text(encoding="utf-8"))
    assert seed["device_model"].startswith("process_video_remote_body_seed")


def test_remote_seed_sidecar_is_dependency_only_and_keeps_local_metric_calibration(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    calibration = _external_metric_calibration_with_points()
    _write_json(options.clip_dir / "court_calibration.json", calibration)

    pipeline = process_video.ProcessVideoPipeline(options)
    note = pipeline._ensure_remote_calibration_seed()

    assert "remote calibration seed" in note
    assert json.loads((options.clip_dir / "court_calibration.json").read_text(encoding="utf-8")) == calibration
    seed = json.loads((options.clip_dir / "capture_sidecar.json").read_text(encoding="utf-8"))
    assert "remote_dependency_only_not_local_world_calibration" in seed["capture_quality"]["reasons"]
    assert "metric_15pt_reviewed" not in seed["intrinsics"]["source"]


def test_body_stage_remote_dispatch_skeleton_level_result_is_honest_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote run that syncs SAM-3D skeleton3d.json but no smpl_motion.json is
    a low-confidence skeleton-level success, not a fabricated mesh."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_only(options)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
        return RemoteBodyDispatchResult(
            status="ran",
            remote_run_dir="remote:/tmp/fake",
            synced_outputs=["skeleton3d.json", "frame_compute_plan.json"],
            wall_seconds=101.0,
            notes=["dispatched to fake host"],
        )

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "ran"
    assert outcome.trust_badge == "low_confidence"
    assert any("skeleton-level only" in note for note in outcome.notes)
    assert not (options.clip_dir / "smpl_motion.json").is_file()
    band = pipeline.trust_bands["body"]
    assert "low-confidence SAM-3D BODY output" in band["reason"]


def test_body_stage_remote_dispatch_degrades_when_nothing_useful_synced_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_only(options)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=[], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "degraded"
    assert any("neither smpl_motion.json nor skeleton3d.json" in note for note in outcome.notes)


def test_body_stage_remote_dispatch_keeps_existing_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)
    existing = {"schema_version": 1, "device_model": "real_arkit_capture"}
    _write_json(options.clip_dir / "capture_sidecar.json", existing)

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    outcome = pipeline._stage_body()

    assert outcome.status == "ran"
    assert any("existing capture_sidecar.json" in note for note in outcome.notes)
    # the pre-existing sidecar was NOT overwritten by a derived seed.
    assert json.loads((options.clip_dir / "capture_sidecar.json").read_text(encoding="utf-8")) == existing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reports_failed_status_and_stops_early_on_missing_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_video = tmp_path / "missing.mp4"
    exit_code = process_video.main(["--video", str(missing_video), "--court-corners", str(tmp_path / "nope.json"), "--out", str(tmp_path / "run"), "--json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    # the hard failure in "ingest" must stop the run, not cascade into every later stage.
    assert [s["stage"] for s in payload["stages"]] == ["ingest"]
    assert payload["stages"][0]["status"] == "failed"


def test_cli_end_to_end_smoke_with_mocked_heavy_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    corners_path = tmp_path / "court_corners.json"
    _write_json(corners_path, _court_corners_payload(width=960, height=540))

    fake_run_pipeline = _fake_run_pipeline_factory({"calibration": {"court_calibration.json": _court_calibration_payload(), "court_zones.json": {"schema_version": 1, "sport": "pickleball", "zones": []}, "net_plane.json": {"schema_version": 1, "sport": "pickleball", "net_height_center_m": 0.86, "net_height_post_m": 0.914, "y_m": 6.7}, "court_line_evidence.json": {"schema_version": 1, "sport": "pickleball", "source": "test", "line_observations": [], "keypoint_observations": [], "net_observations": [], "aggregate": {"accepted_line_ids": [], "rejected_line_ids": [], "missing_required_line_ids": [], "missing_required_net_ids": [], "mean_residual_px": 0.0, "p95_residual_px": 0.0, "temporal_stability_px": 0.0, "auto_calibration_ready": True, "reasons": []}}}})
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    exit_code = process_video.main(
        [
            "--video",
            str(video),
            "--court-corners",
            str(corners_path),
            "--out",
            str(tmp_path / "run"),
            "--no-gpu",
            "--skip-ball",
            "--vite-allow-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "run" / "PIPELINE_SUMMARY.json").is_file()


def test_cli_parses_court_calibration_flag_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    metric_path = tmp_path / "metric15.json"
    args = process_video.build_arg_parser().parse_args(
        ["--video", str(video), "--court-calibration", str(metric_path), "--out", str(tmp_path / "run")]
    )
    options = process_video.build_options_from_args(args)
    assert options.court_calibration == metric_path.resolve()
    assert options.court_corners is None


def test_cli_parses_court_proposals_preview_flag_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    args = process_video.build_arg_parser().parse_args(
        ["--video", str(video), "--court-proposals-preview", "--out", str(tmp_path / "run")]
    )

    options = process_video.build_options_from_args(args)

    assert options.court_proposals_preview is True
    assert options.court_corners is None


def test_cli_no_scene_points_disables_replay_scene_points(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    args = process_video.build_arg_parser().parse_args(
        ["--video", str(video), "--out", str(tmp_path / "run"), "--no-scene-points"]
    )
    options = process_video.build_options_from_args(args)
    assert options.scene_points is False


def test_cli_end_to_end_smoke_with_court_calibration_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The v1 corners-vs-metric CLI selection: --court-calibration reaches the
    ExternalCalibrationRunner seam end to end, in place of --court-corners."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    metric_path = tmp_path / "court_calibration_metric15pt.json"
    _write_json(metric_path, _external_metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0]))

    fake_run_pipeline = _fake_run_pipeline_capturing_kwargs(
        {"calibration": {"court_calibration.json": _external_metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0])}}
    )
    monkeypatch.setattr(process_video.orchestrator, "run_pipeline", fake_run_pipeline)

    exit_code = process_video.main(
        [
            "--video",
            str(video),
            "--court-calibration",
            str(metric_path),
            "--out",
            str(tmp_path / "run"),
            "--no-gpu",
            "--skip-ball",
            "--vite-allow-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert exit_code == 0
    summary = json.loads((tmp_path / "run" / "PIPELINE_SUMMARY.json").read_text(encoding="utf-8"))
    calibration_stage = next(s for s in summary["stages"] if s["stage"] == "calibration")
    assert calibration_stage["status"] == "ran"
    assert any("consumed externally-provided" in note for note in calibration_stage["notes"])
    [call] = fake_run_pipeline.calls  # type: ignore[attr-defined]
    assert call["runners"]["calibration"].source_path == metric_path.resolve()


# ---------------------------------------------------------------------------
# --body-schedule=overlap (SCHED-A) + B12 camera-motion threading into remote
# BODY dispatch
# ---------------------------------------------------------------------------


def test_cli_parses_body_schedule_flag_into_options(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    args = process_video.build_arg_parser().parse_args(["--video", str(video), "--out", str(tmp_path / "run")])
    options = process_video.build_options_from_args(args)
    assert options.body_schedule == "serial"

    args = process_video.build_arg_parser().parse_args(
        ["--video", str(video), "--out", str(tmp_path / "run"), "--body-schedule", "overlap"]
    )
    options = process_video.build_options_from_args(args)
    assert options.body_schedule == "overlap"


def test_dispatch_body_remote_threads_explicit_camera_motion_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B12 green: an explicit --camera-motion path must reach dispatch_body_stage."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)
    external_motion = tmp_path / "sidecars" / "owner_camera_motion.json"
    _write_json(external_motion, {"schema_version": 1, "artifact_type": "racketsport_camera_motion", "frames": []})
    options.camera_motion_path = external_motion

    calls: list[dict[str, Any]] = []

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(
            clip_dir / "smpl_motion.json",
            {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []},
        )
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    pipeline._stage_body()

    [call] = calls
    assert call["camera_motion_path"] == external_motion


def test_dispatch_body_remote_clip_dir_only_camera_motion_stays_unthreaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B12 "clip-dir case unchanged": with no explicit --camera-motion, an
    auto-discovered clip_dir/camera_motion.json must keep relying on
    remote_body_dispatch's own BODY_INPUT_ARTIFACTS clip-dir autosync (this
    call site must pass None here) -- threading the resolved clip-dir path
    through as "explicit" would make _rsync_up's explicit-vs-canonical
    dedupe (remote_body_dispatch.py:1033-1046) skip syncing it entirely."""

    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    _clip_dir_with_tracks_and_sam3d_skeleton(options)
    _write_json(
        options.clip_dir / "camera_motion.json",
        {"schema_version": 1, "artifact_type": "racketsport_camera_motion", "frames": []},
    )
    assert options.camera_motion_path is None

    calls: list[dict[str, Any]] = []

    def _fake_dispatch(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(
            clip_dir / "smpl_motion.json",
            {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []},
        )
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["smpl_motion.json"], wall_seconds=1.0, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)
    pipeline = process_video.ProcessVideoPipeline(options)

    pipeline._stage_body()

    [call] = calls
    assert call["camera_motion_path"] is None


class _FakePlacementResult:
    def __init__(self) -> None:
        self.coverage_unchanged = True
        self.source_counts = {"bbox": 2, "native2d": 0, "sam3d": 0}
        self.court_bounds_violations = 0
        self.summary: dict[str, Any] = {}


def _fake_rewrite_tracks_with_placement_noop(**kwargs: Any) -> _FakePlacementResult:
    """Deterministic stand-in for rewrite_tracks_with_placement so the
    --body-schedule overlap tests exercise real pipeline scheduling without
    depending on the real placement solver's behavior on synthetic tracks."""

    placement_path = kwargs["placement_path"]
    _write_json(
        placement_path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_placement",
            "fps": 30.0,
            "source": "test",
            "tracks_path": "tracks.json",
            "backup_tracks_path": "tracks_prewrite_backup.json",
            "refine_from_sam3d": False,
            "undistort_applied": False,
            "players": [],
            "summary": {
                "player_count": 0,
                "frame_count": 0,
                "coverage_unchanged": True,
                "source_counts": {"bbox": 0, "native2d": 0, "sam3d": 0},
                "jitter_before_after_mps": {},
                "court_bounds_violations": 0,
            },
            "provenance": {},
        },
    )
    return _FakePlacementResult()


def _overlap_ready_options(tmp_path: Path) -> process_video.PipelineOptions:
    """A clip_dir with tracks.json (via --tracks reuse), a valid
    court_calibration.json (no-force reuse skip), and an already-extracted
    body_frames/ JPEG (no-force reuse skip) -- exactly the BODY dispatch
    inputs design item 1 requires to be "ready" before overlap can dispatch
    BODY, with --skip-ball so ball/ball_arc/ball_fill resolve instantly and
    only real pipeline code (never a mocked orchestrator.run_pipeline) runs
    for calibration/tracking/placement/frames."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "clip.mp4"
    _make_video(video)
    reuse_tracks = tmp_path / "reuse_tracks.json"
    _write_json(reuse_tracks, _tracks_payload())

    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_remote = True
    options.tracks_reuse = reuse_tracks
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    body_frames = options.clip_dir / "body_frames"
    body_frames.mkdir(parents=True, exist_ok=True)
    (body_frames / "frame_000000.jpg").write_bytes(b"fake-jpeg-bytes")
    return options


def _run_pipeline_with_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schedule: str,
    dispatch_fn,
) -> dict[str, Any]:
    options = _overlap_ready_options(tmp_path)
    options.body_schedule = schedule
    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite_tracks_with_placement_noop)
    monkeypatch.setattr(process_video, "dispatch_body_stage", dispatch_fn)
    pipeline = process_video.ProcessVideoPipeline(options)
    return pipeline.run()


def _successful_fake_dispatch(**kwargs: Any):  # noqa: ANN001
    clip_dir = Path(kwargs["clip_dir"])
    _write_json(clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(clip_dir / "body_full_clip_gate.json", {"schema_version": 1, "artifact_type": "body_full_clip_gate"})
    return RemoteBodyDispatchResult(
        status="ran",
        remote_run_dir="remote:/tmp/fake",
        synced_outputs=["skeleton3d.json", "body_full_clip_gate.json"],
        wall_seconds=0.01,
        notes=["dispatched to fake host"],
    )


def test_overlap_mode_matches_serial_stage_shape_with_parallel_body_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SELF-VERIFICATION (a): overlap mode on the fake runtime still produces
    the same stage outcomes/artifact set as serial, though serial now runs
    events before frames for the sanctioned best-stack contact-density default.
    Notes cannot be byte-identical because overlap adds its readiness note."""

    serial_summary = _run_pipeline_with_schedule(
        tmp_path / "serial", monkeypatch, schedule="serial", dispatch_fn=_successful_fake_dispatch
    )
    overlap_summary = _run_pipeline_with_schedule(
        tmp_path / "overlap", monkeypatch, schedule="overlap", dispatch_fn=_successful_fake_dispatch
    )

    def _shape(summary: dict[str, Any]) -> list[tuple[str, str, tuple[str, ...], str | None]]:
        return [(s["stage"], s["status"], tuple(s["artifacts"]), s["trust_badge"]) for s in summary["stages"]]

    assert sorted(_shape(serial_summary)) == sorted(_shape(overlap_summary))
    assert "parallel_body" not in serial_summary
    assert "parallel_body" in overlap_summary

    parallel_body = overlap_summary["parallel_body"]
    assert parallel_body["enabled"] is True
    assert parallel_body["body_started_after"] == "frames"
    assert parallel_body["overlapped_stages"] == ["ball", "ball_arc", "events", "ball_fill"]
    assert parallel_body["input_mutation_guard"]["tripped"] is False
    assert parallel_body["input_mutation_guard"]["mutated_inputs"] == []
    assert isinstance(parallel_body["body_wall_s"], float)
    assert isinstance(parallel_body["join_wait_s"], float)
    assert isinstance(parallel_body["overlap_saved_s_estimate"], float)
    assert parallel_body["body_inputs_missing_due_to_overlap"]  # non-empty on this cold fixture

    serial_body = next(s for s in serial_summary["stages"] if s["stage"] == "body")
    overlap_body = next(s for s in overlap_summary["stages"] if s["stage"] == "body")
    assert serial_body["notes"] == [note for note in overlap_body["notes"] if "overlap readiness note" not in note]
    assert any("overlap readiness note" in note for note in overlap_body["notes"])


def test_overlap_body_thread_failure_matches_serial_failure_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SELF-VERIFICATION (b): a BODY-thread failure in overlap mode must
    surface the same status/trust_badge/artifacts as a serial BODY failure
    (HONESTY CONTRACTS d), modulo the mandatory overlap readiness note."""

    def _fail_dispatch(**_kwargs: Any):
        raise RemoteBodyDispatchError(
            "shared GPU lock busy on arnavchokshi@34.126.67.233: did not acquire scripts/gpu-eval-run.sh within 60s"
        )

    serial_summary = _run_pipeline_with_schedule(
        tmp_path / "serial", monkeypatch, schedule="serial", dispatch_fn=_fail_dispatch
    )
    overlap_summary = _run_pipeline_with_schedule(
        tmp_path / "overlap", monkeypatch, schedule="overlap", dispatch_fn=_fail_dispatch
    )

    serial_body = next(s for s in serial_summary["stages"] if s["stage"] == "body")
    overlap_body = next(s for s in overlap_summary["stages"] if s["stage"] == "body")

    assert serial_body["status"] == overlap_body["status"] == "degraded"
    assert serial_body["trust_badge"] == overlap_body["trust_badge"] is None
    assert serial_body["artifacts"] == overlap_body["artifacts"] == []
    assert serial_body["notes"] == [note for note in overlap_body["notes"] if "overlap readiness note" not in note]
    assert "no fallback pose skeleton" in " ".join(serial_body["notes"])
    assert "no fallback pose skeleton" in " ".join(overlap_body["notes"])


def test_overlap_input_mutation_guard_trips_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SELF-VERIFICATION (c) / HONESTY CONTRACTS c: an overlapped local stage
    that mutates a BODY dispatch input (tracks.json here) while BODY is in
    flight must trip the guard, fail the body stage closed, and stop the
    pipeline before any BODY-dependent stage runs."""

    options = _overlap_ready_options(tmp_path)
    options.body_schedule = "overlap"
    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite_tracks_with_placement_noop)

    mutate_ready = threading.Event()

    def _fake_dispatch(**kwargs: Any):
        # Block until the main thread has mutated a guarded BODY input, so
        # the post-join hash snapshot deterministically observes the change
        # regardless of thread scheduling.
        assert mutate_ready.wait(timeout=5.0), "overlapped local stage never signaled its mutation"
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["skeleton3d.json"], wall_seconds=0.01, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)

    original_stage_events = process_video.ProcessVideoPipeline._stage_events

    def _mutating_stage_events(self):  # noqa: ANN001
        outcome = original_stage_events(self)
        (self.clip_dir / "tracks.json").write_text(
            json.dumps({"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": [], "mutated_by_test": True}),
            encoding="utf-8",
        )
        mutate_ready.set()
        return outcome

    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_events", _mutating_stage_events)

    pipeline = process_video.ProcessVideoPipeline(options)
    summary = pipeline.run()

    body_stage = next(s for s in summary["stages"] if s["stage"] == "body")
    assert body_stage["status"] == "failed"
    assert "GUARD TRIPPED" in " ".join(body_stage["notes"])

    guard = summary["parallel_body"]["input_mutation_guard"]
    assert guard["tripped"] is True
    assert "tracks.json" in guard["mutated_inputs"]

    stage_names = [s["stage"] for s in summary["stages"]]
    assert stage_names[-1] == "body"
    assert "world" not in stage_names
    assert "manifest" not in stage_names
    assert summary["status"] == "failed"


def test_overlap_join_barrier_world_never_starts_before_body_thread_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SELF-VERIFICATION (d): world must never start before the BODY thread
    has completed, proven via event ordering rather than timing alone."""

    options = _overlap_ready_options(tmp_path)
    options.body_schedule = "overlap"
    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite_tracks_with_placement_noop)

    events: list[str] = []

    def _fake_dispatch(**kwargs: Any):
        events.append("body_start")
        time.sleep(0.05)
        clip_dir = Path(kwargs["clip_dir"])
        _write_json(clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
        events.append("body_end")
        return RemoteBodyDispatchResult(status="ran", remote_run_dir="remote:/tmp/fake", synced_outputs=["skeleton3d.json"], wall_seconds=0.05, notes=[])

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fake_dispatch)

    original_build_virtual_world_state = process_video.build_virtual_world_state

    def _tracking_build_virtual_world_state(*args: Any, **kwargs: Any):
        events.append("world_start")
        return original_build_virtual_world_state(*args, **kwargs)

    monkeypatch.setattr(process_video, "build_virtual_world_state", _tracking_build_virtual_world_state)

    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline.run()

    assert "body_start" in events
    assert "body_end" in events
    assert "world_start" in events
    assert events.index("body_end") < events.index("world_start")


def test_overlap_takes_serial_path_with_no_thread_when_body_artifacts_valid_for_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design item 1: "Reuse semantics unchanged: if BODY artifacts are valid
    for no-force reuse, take the serial path (no thread)." dispatch_body_stage
    must never be called at all in that case."""

    options = _overlap_ready_options(tmp_path)
    options.body_schedule = "overlap"
    monkeypatch.setattr(process_video, "rewrite_tracks_with_placement", _fake_rewrite_tracks_with_placement_noop)
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "body_full_clip_gate.json", {"schema_version": 1, "artifact_type": "body_full_clip_gate"})

    def _fail_if_dispatched(**_kwargs: Any):
        raise AssertionError("no-force BODY reuse must not dispatch (and must not spend a thread) in overlap mode")

    monkeypatch.setattr(process_video, "dispatch_body_stage", _fail_if_dispatched)

    pipeline = process_video.ProcessVideoPipeline(options)
    summary = pipeline.run()

    body_stage = next(s for s in summary["stages"] if s["stage"] == "body")
    assert body_stage["status"] == "skipped"
    assert summary["parallel_body"]["enabled"] is False
    assert summary["parallel_body"]["overlapped_stages"] == []
