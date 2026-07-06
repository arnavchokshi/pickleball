from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError
from threed.racketsport.skeleton_video_overlay import (
    LOW_CONFIDENCE_THRESHOLD,
    caption_extra_from_skeleton,
    color_for_player,
    project_skeleton_joints,
)


cv2_available = importlib.util.find_spec("cv2") is not None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibration(path: Path) -> Path:
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        image_size=(1920, 1080),
        homography=[[20.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )
    path.write_text(calibration.model_dump_json(), encoding="utf-8")
    return path


def _skeleton(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "source_model": "sam3d_body_joints",
            "world_frame": "court_Z0",
            "fps": 30.0,
            "joint_names": ["nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
            "provenance": {"lane": "A"},
            "players": [
                {
                    "id": 1,
                    "frames": [
                        {
                            "frame_idx": 0,
                            "t": 0.0,
                            "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                            "joint_conf": [0.95, LOW_CONFIDENCE_THRESHOLD - 0.01, 0.5],
                            "smoothing_flag": ["none", "single_frame_jump_clamped", "none"],
                        },
                        {
                            "frame_idx": 1,
                            "t": 1.0 / 30.0,
                            "joints_2d": [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]],
                            "joints_world": [[99.0, 99.0, 0.0], [98.0, 98.0, 0.0], [97.0, 97.0, 0.0]],
                            "joint_conf": [0.8, 0.7, 0.6],
                        },
                    ],
                },
                {
                    "id": 2,
                    "frames": [
                        {
                            "frame_idx": 0,
                            "t": 0.0,
                            "joints_world": [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
                            "joint_conf": [0.9, 0.8, 0.7],
                        }
                    ],
                },
            ],
        },
    )
    return path


def test_project_skeleton_joints_uses_calibration_for_world_joints(tmp_path: Path) -> None:
    calibration = CourtCalibration.model_validate_json(_calibration(tmp_path / "court_calibration.json").read_text())
    frame = {
        "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        "joint_conf": [0.9, 0.4, 0.1],
        "smoothing_flag": ["none", "single_frame_jump_clamped", "none"],
    }

    projected = project_skeleton_joints(frame, calibration=calibration)

    assert [joint.xy for joint in projected] == [[960.0, 540.0], [980.0, 540.0], [960.0, 580.0]]
    assert [joint.source for joint in projected] == ["world_projection", "world_projection", "world_projection"]
    assert projected[1].low_confidence is False
    assert projected[2].low_confidence is True
    assert projected[1].smoothing_flag == "single_frame_jump_clamped"


def test_project_skeleton_joints_prefers_native_2d_when_present(tmp_path: Path) -> None:
    calibration = CourtCalibration.model_validate_json(_calibration(tmp_path / "court_calibration.json").read_text())
    frame = {
        "joints_2d": [[100.0, 200.0], [110.0, 210.0]],
        "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "joint_conf": [0.9, 0.8],
    }

    projected = project_skeleton_joints(frame, calibration=calibration)

    assert [joint.xy for joint in projected] == [[100.0, 200.0], [110.0, 210.0]]
    assert [joint.source for joint in projected] == ["native_2d", "native_2d"]


def test_project_skeleton_joints_keeps_missing_joints_as_unprojected(tmp_path: Path) -> None:
    calibration = CourtCalibration.model_validate_json(_calibration(tmp_path / "court_calibration.json").read_text())
    frame = {
        "joints_world": [[0.0, 0.0, 0.0], None, [1.0], ["nan", 0.0, 0.0]],
        "joint_conf": [0.9, 0.8, 0.7, 0.6],
    }

    projected = project_skeleton_joints(frame, calibration=calibration)

    assert [joint.xy for joint in projected] == [[960.0, 540.0], None, None, None]
    assert [joint.source for joint in projected] == ["world_projection", "missing", "missing", "missing"]


def test_color_for_player_is_stable_and_distinct() -> None:
    assert color_for_player(1) == color_for_player(1)
    assert color_for_player(1) != color_for_player(2)


def test_caption_extra_from_skeleton_surfaces_scale_suspect_provenance() -> None:
    skeleton = {
        "provenance": {
            "skeleton_upright_repair": {
                "overlay_caption_extra": "SCALE SUSPECT: stature ~0.95m — under investigation",
            }
        }
    }

    assert caption_extra_from_skeleton(skeleton) == "SCALE SUSPECT: stature ~0.95m — under investigation"


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for skeleton overlay rendering")
def test_render_skeleton_overlay_cli_writes_video_contact_sheet_and_index(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _skeleton(run_dir / "skeleton3d.json")
    _calibration(run_dir / "court_calibration.json")
    _write_json(
        run_dir / "contact_windows.json",
        {
            "schema_version": 1,
            "events": [
                {
                    "type": "contact",
                    "frame": 1,
                    "t": 1.0 / 30.0,
                    "window": {"t0": 0.0, "t1": 2.0 / 30.0, "importance": 0.9},
                }
            ],
        },
    )

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        for index in range(3):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, :] = index * 30
            writer.write(frame)
    finally:
        writer.release()

    out_dir = tmp_path / "overlay_packet"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_skeleton_overlay.py",
            "--run-dir",
            str(run_dir),
            "--video",
            str(source),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "3",
            "--contact-sheet-frame-count",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["artifact_type"] == "racketsport_skeleton_video_overlay"
    assert summary["status"] == "rendered"
    assert summary["qualitative_status"] == "review_copy_not_gate_verified"
    assert summary["frame_count"] == 3
    assert summary["skeleton_source_model"] == "sam3d_body_joints"
    assert Path(summary["overlay_path"]).is_file()
    assert Path(summary["contact_sheet_path"]).is_file()
    assert Path(summary["index_path"]).is_file()
    assert Path(summary["overlay_path"]).stat().st_size > 0
    assert Path(summary["contact_sheet_path"]).stat().st_size > 0


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for skeleton overlay rendering")
def test_render_skeleton_overlay_cli_defaults_output_under_run_dir(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _skeleton(run_dir / "skeleton3d.json")
    _calibration(run_dir / "court_calibration.json")

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_skeleton_overlay.py",
            "--run-dir",
            str(run_dir),
            "--video",
            str(source),
            "--max-frames",
            "2",
            "--contact-sheet-frame-count",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["overlay_path"] == str(run_dir / "skeleton_video_overlay" / "skeleton_overlay.mp4")
    assert summary["contact_sheet_path"] == str(run_dir / "skeleton_video_overlay" / "skeleton_overlay_contact_sheet.jpg")
    assert summary["index_path"] == str(run_dir / "skeleton_video_overlay" / "skeleton_video_overlay_index.json")
