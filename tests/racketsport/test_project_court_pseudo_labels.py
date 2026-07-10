from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.racketsport.project_court_pseudo_labels import (
    PROJECTOR_VERSION,
    project_metric_template,
    validate_manifest_row,
)
from threed.racketsport.court_calibration_metric15 import fit_single_view_metric_camera
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


IMAGE_SIZE = (640, 360)


def _look_at_pose() -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray([0.0, -13.0, 5.5], dtype=np.float64)
    target = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera
    return rotation, translation


def _calibration(reference_image_path: Path) -> dict:
    rotation, translation = _look_at_pose()
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_harvest_source_court_calibration",
        "source_id": "tinySource",
        "calibration_grade": "manual_bar",
        "reference_image_path": str(reference_image_path),
        "image_size": list(IMAGE_SIZE),
        "intrinsics": {
            "fx": 480.0,
            "fy": 480.0,
            "cx": IMAGE_SIZE[0] / 2.0,
            "cy": IMAGE_SIZE[1] / 2.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "synthetic_test",
        },
        "extrinsics": {
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "camera_height_m": 5.5,
        },
        "reprojection_error_px": {"median": 0.01, "p95": 0.05},
        "per_frame_reprojection_stats": [{"frame_name": "reference.png"}],
    }


def test_project_then_resolve_self_consistency_below_point_one_px(tmp_path: Path) -> None:
    known = _calibration(tmp_path / "unused.png")
    projected = project_metric_template(known)
    world = [PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    image = [projected[name] for name in PICKLEBALL_COURT_KEYPOINT_NAMES]

    fit = fit_single_view_metric_camera(world, image, tuple(float(v) for v in IMAGE_SIZE))
    recovered = {
        "intrinsics": {
            "fx": fit.fx,
            "fy": fit.fy,
            "cx": fit.cx,
            "cy": fit.cy,
            "dist": [fit.k1, fit.k2, 0.0, 0.0],
        },
        "extrinsics": {"R": fit.R, "t": fit.t},
    }
    reproduced = project_metric_template(recovered)
    errors = [
        float(np.linalg.norm(np.asarray(reproduced[name]) - np.asarray(projected[name])))
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES
    ]

    assert max(errors) < 0.1


def test_projector_direct_cli_reference_on_tiny_static_fixture(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260709)
    frame = rng.integers(0, 255, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    cv2.rectangle(frame, (75, 80), (565, 330), (255, 255, 255), 4)
    cv2.line(frame, (120, 210), (520, 210), (30, 255, 30), 3)
    reference = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference), frame)

    calibration_dir = tmp_path / "calibrations"
    calibration_dir.mkdir()
    (calibration_dir / "tinySource.json").write_text(
        json.dumps(_calibration(reference)),
        encoding="utf-8",
    )
    rally_dir = tmp_path / "rallies" / "tinySource"
    rally_dir.mkdir(parents=True)
    video_path = rally_dir / "tinySource_rally_0001.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        IMAGE_SIZE,
    )
    assert writer.isOpened()
    for _ in range(12):
        writer.write(frame)
    writer.release()

    manifest = tmp_path / "manifest.jsonl"
    default_view = tmp_path / "default.jsonl"
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/project_court_pseudo_labels.py",
            "--calibration-dir",
            str(calibration_dir),
            "--rallies-dir",
            str(tmp_path / "rallies"),
            "--out",
            str(manifest),
            "--default-view-out",
            str(default_view),
            "--report-json",
            str(report),
            "--qa-dir",
            str(tmp_path / "qa"),
            "--stride",
            "5",
            "--static-samples",
            "8",
            "--qa-per-source",
            "0",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(result.stdout)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert stdout["full_row_count"] == 3
    assert stdout["included_row_count"] == 3
    assert len(rows) == 3
    assert default_view.read_bytes() == manifest.read_bytes()
    assert rows[0]["projector_version"] == PROJECTOR_VERSION
    assert len(rows[0]["keypoints"]) == 15
    validate_manifest_row(rows[0])

