from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.sat_hmr_body_fallback import build_sat_hmr_body_fallback
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    Skeleton3D,
    SmplMotion,
    validate_artifact_file,
)


def test_sat_hmr_body_fallback_writes_world_artifacts_and_blocks_mpjpe_gate(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    calibration = tmp_path / "court_calibration.json"
    execution = tmp_path / "body_compute_execution.json"
    predictions = tmp_path / "predictions"
    out = tmp_path / "out"
    predictions.mkdir()
    _write_json(tracks, _tracks())
    _write_json(calibration, _identity_calibration().model_dump(mode="json"))
    _write_json(execution, _body_compute_execution())
    _write_prediction(
        predictions / "frame_000012_personId_0.pkl",
        {
            "image_name": "frame_000012",
            "person_index": 0,
            "confidence": 0.88,
            "bbox_xyxy": [101.0, 98.0, 201.0, 301.0],
            "pred_j3ds": _sat_joints(),
            "pred_verts": [[0.0, 0.0, 0.0], [0.1, 0.0, 1.6]],
            "pred_poses": [0.01, 0.02, 0.03, *([0.1] * 69)],
            "pred_betas": [0.0] * 10,
            "pred_transl": [0.0, 0.0, 0.0],
        },
    )

    report = build_sat_hmr_body_fallback(
        clip="clip_001",
        predictions_dir=predictions,
        tracks_path=tracks,
        calibration_path=calibration,
        body_compute_execution_path=execution,
        out_dir=out,
    )

    assert report["artifact_type"] == "racketsport_sat_hmr_body_fallback"
    assert report["status"] == "ran_not_gate_verified"
    assert report["world_mpjpe_gate"]["status"] == "blocked_missing_body_world_gt"
    assert report["foot_slide_gate"]["value_m"] == pytest.approx(0.0)
    assert report["assignment_summary"] == {
        "assigned_prediction_count": 1,
        "scheduled_player_frame_count": 1,
        "min_assignment_iou": 0.05,
    }

    smpl = validate_artifact_file("smpl_motion", out / "smpl_motion.json")
    skeleton = validate_artifact_file("skeleton3d", out / "skeleton3d.json")
    assert isinstance(smpl, SmplMotion)
    assert isinstance(skeleton, Skeleton3D)
    assert smpl.model == "sat_hmr_world_joints"
    assert smpl.players[0].id == 7
    assert smpl.players[0].frames[0].track_world_xy == pytest.approx([2.0, 3.0])
    assert len(smpl.players[0].frames[0].mesh_vertices_world) == 2

    quality = json.loads((out / "body_joint_quality.json").read_text(encoding="utf-8"))
    assert quality["status"] == "quality_checked_needs_accuracy_gate"
    assert quality["summary"]["joint_frame_count"] == 1
    assert quality["promotion_blockers"] == ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"]


def test_sat_hmr_body_fallback_rejects_predictions_without_3d_body_output(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    calibration = tmp_path / "court_calibration.json"
    execution = tmp_path / "body_compute_execution.json"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_json(tracks, _tracks())
    _write_json(calibration, _identity_calibration().model_dump(mode="json"))
    _write_json(execution, _body_compute_execution())
    _write_prediction(
        predictions / "frame_000012_personId_0.pkl",
        {"image_name": "frame_000012", "confidence": 0.88, "bbox_xyxy": [100.0, 100.0, 200.0, 300.0]},
    )

    with pytest.raises(ValueError, match="missing SAT-HMR 3D joints/vertices"):
        build_sat_hmr_body_fallback(
            clip="clip_001",
            predictions_dir=predictions,
            tracks_path=tracks,
            calibration_path=calibration,
            body_compute_execution_path=execution,
            out_dir=tmp_path / "out",
        )


def test_sat_hmr_body_fallback_fails_closed_when_prediction_does_not_overlap_track(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    calibration = tmp_path / "court_calibration.json"
    execution = tmp_path / "body_compute_execution.json"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_json(tracks, _tracks())
    _write_json(calibration, _identity_calibration().model_dump(mode="json"))
    _write_json(execution, _body_compute_execution())
    _write_prediction(
        predictions / "frame_000012_personId_0.pkl",
        {
            "image_name": "frame_000012",
            "confidence": 0.88,
            "bbox_xyxy": [500.0, 500.0, 600.0, 700.0],
            "pred_j3ds": _sat_joints(),
        },
    )

    with pytest.raises(ValueError, match="no SAT-HMR predictions matched scheduled BODY player frames"):
        build_sat_hmr_body_fallback(
            clip="clip_001",
            predictions_dir=predictions,
            tracks_path=tracks,
            calibration_path=calibration,
            body_compute_execution_path=execution,
            out_dir=tmp_path / "out",
        )


def test_sat_hmr_body_fallback_defaults_limit_root_jumps_and_shift_mesh_with_joints(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    calibration = tmp_path / "court_calibration.json"
    execution = tmp_path / "body_compute_execution.json"
    predictions = tmp_path / "predictions"
    out = tmp_path / "out"
    predictions.mkdir()
    frame_indexes = [12, 13]
    _write_json(tracks, _tracks_with_world_points([[2.0, 3.0], [12.0, 3.0]], frame_indexes=frame_indexes))
    _write_json(calibration, _identity_calibration().model_dump(mode="json"))
    _write_json(execution, _body_compute_execution_for_frames(frame_indexes))
    for frame_idx in frame_indexes:
        _write_prediction(
            predictions / f"frame_{frame_idx:06d}_personId_0.pkl",
            {
                "image_name": f"frame_{frame_idx:06d}",
                "person_index": 0,
                "confidence": 0.88,
                "bbox_xyxy": [101.0, 98.0, 201.0, 301.0],
                "pred_j3ds": _sat_joints(),
                "pred_verts": (
                    [[0.0, 0.0, 0.0], [0.1, 0.0, 1.6]]
                    if frame_idx == 12
                    else [[0.0, 0.0, -1.0], [0.1, 0.0, 0.6]]
                ),
            },
        )

    report = build_sat_hmr_body_fallback(
        clip="clip_001",
        predictions_dir=predictions,
        tracks_path=tracks,
        calibration_path=calibration,
        body_compute_execution_path=execution,
        out_dir=out,
    )

    assert report["grounding_metrics"]["max_root_speed_mps"] == pytest.approx(8.0)
    assert report["grounding_metrics"]["max_track_anchor_smoothing_residual_m"] == pytest.approx(0.75)
    assert report["grounding_metrics"]["root_speed_limited_frames"] == 1
    assert report["grounding_metrics"]["track_anchor_residual_reset_frames"] == 1
    assert report["grounding_metrics"]["max_track_anchor_residual_m"] == pytest.approx(0.0)

    smpl = validate_artifact_file("smpl_motion", out / "smpl_motion.json")
    assert isinstance(smpl, SmplMotion)
    first, second = smpl.players[0].frames
    assert second.temporal_smoothing_reset is True
    assert second.transl_world == pytest.approx([12.0, 3.0, 0.0])
    mesh_delta = second.mesh_vertices_world[0][0] - first.mesh_vertices_world[0][0]
    joint_delta = second.joints_world[0][0] - first.joints_world[0][0]
    assert mesh_delta == pytest.approx(10.0)
    assert joint_delta == pytest.approx(10.0)

    quality = json.loads((out / "body_joint_quality.json").read_text(encoding="utf-8"))
    assert "root_motion_temporal_jump" not in quality["quality_blockers"]
    assert quality["summary"]["temporal_smoothing_reset_count"] == 1


def test_build_sat_hmr_body_fallback_cli_writes_report(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    calibration = tmp_path / "court_calibration.json"
    execution = tmp_path / "body_compute_execution.json"
    predictions = tmp_path / "predictions"
    out = tmp_path / "out"
    predictions.mkdir()
    _write_json(tracks, _tracks())
    _write_json(calibration, _identity_calibration().model_dump(mode="json"))
    _write_json(execution, _body_compute_execution())
    _write_prediction(
        predictions / "frame_000012_personId_0.pkl",
        {
            "image_name": "frame_000012",
            "confidence": 0.88,
            "bbox_xyxy": [101.0, 98.0, 201.0, 301.0],
            "pred_j3ds": _sat_joints(),
            "pred_verts": [[0.0, 0.0, 0.0], [0.1, 0.0, 1.6]],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_sat_hmr_body_fallback.py",
            "--clip",
            "clip_001",
            "--predictions-dir",
            str(predictions),
            "--tracks",
            str(tracks),
            "--court-calibration",
            str(calibration),
            "--body-compute-execution",
            str(execution),
            "--out-dir",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "ran_not_gate_verified"
    assert (out / "sat_hmr_body_fallback_report.json").is_file()


def _tracks() -> dict:
    return _tracks_with_world_points([[2.0, 3.0]], frame_indexes=[12])


def _tracks_with_world_points(world_xys: list[list[float]], *, frame_indexes: list[int]) -> dict:
    if len(world_xys) != len(frame_indexes):
        raise ValueError("world_xys and frame_indexes must have the same length")
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "unknown",
                "frames": [
                    {
                        "t": frame_idx / 30.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": world_xy,
                        "conf": 0.91,
                    }
                    for frame_idx, world_xy in zip(frame_indexes, world_xys)
                ],
            }
        ],
        "rally_spans": [],
    }


def _sat_joints() -> list[list[float]]:
    joints = [[0.02 * index, 0.0, 0.4 + 0.04 * (index % 8)] for index in range(17)]
    joints[1] = [-0.2, 0.0, 0.02]
    joints[2] = [0.2, 0.0, 0.03]
    return joints


def _body_compute_execution() -> dict:
    return _body_compute_execution_for_frames([12])


def _body_compute_execution_for_frames(frame_indexes: list[int]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "fps": 30.0,
        "mode": "adaptive_frame_compute_plan",
        "scheduled_frames": [
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "target_player_ids": [7],
                "active_player_ids": [7],
                "target_representation": "world_mesh",
                "reasons": ["test"],
            }
            for frame_idx in frame_indexes
        ],
        "skipped_frames": [],
        "summary": {"scheduled_frame_count": len(frame_indexes), "scheduled_player_frame_count": len(frame_indexes)},
    }


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 0.0],
            camera_height_m=1.5,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def _write_prediction(path: Path, payload: dict) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
