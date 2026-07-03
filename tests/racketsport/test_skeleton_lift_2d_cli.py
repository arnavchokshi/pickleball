from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


JOINT_NAMES = ["left_ankle", "right_ankle", "pelvis", "nose"]


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "dist": []},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "homography": [[80.0, 0.0, 320.0], [0.0, 80.0, 240.0], [0.0, 0.0, 1.0]],
        "image_size": [640, 480],
    }


def _project(point: list[float]) -> list[float]:
    x, y, z = point
    depth = z + 10.0
    return [800.0 * x / depth + 320.0, 800.0 * y / depth + 240.0]


def _inputs() -> tuple[dict, dict, dict]:
    joints = [
        [-0.15, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 1.7],
    ]
    keypoints = [
        {"joint": joint_name, "x_px": _project(point)[0], "y_px": _project(point)[1], "conf": 0.95}
        for joint_name, point in zip(JOINT_NAMES, joints, strict=True)
    ]
    keypoints_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "fps": 30.0,
        "convention": "synthetic_coco",
        "joint_names": JOINT_NAMES,
        "bone_priors": [
            {"parent": "left_ankle", "child": "pelvis", "length_m": math.hypot(0.15, 0.9)},
            {"parent": "pelvis", "child": "nose", "length_m": 0.8},
        ],
        "players": [{"id": "p1", "height_m": 1.7, "frames": [{"frame_idx": 0, "t": 0.0, "keypoints": keypoints}]}],
    }
    tracks_payload = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [{"id": "p1", "frames": [{"frame_idx": 0, "t": 0.0, "world_xy": [0.0, 0.0]}]}],
    }
    return keypoints_payload, tracks_payload, _calibration()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_build_skeleton_from_2d_cli_writes_skeleton_and_report(tmp_path: Path) -> None:
    keypoints, tracks, calibration = _inputs()
    keypoints_path = tmp_path / "keypoints_2d.json"
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    out_dir = tmp_path / "lift"
    _write_json(keypoints_path, keypoints)
    _write_json(tracks_path, tracks)
    _write_json(calibration_path, calibration)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_skeleton_from_2d.py",
            "--keypoints-2d",
            str(keypoints_path),
            "--tracks",
            str(tracks_path),
            "--court-calibration",
            str(calibration_path),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    skeleton = json.loads((out_dir / "skeleton3d_v2.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "skeleton_lift_2d_report.json").read_text(encoding="utf-8"))
    assert summary["skeleton"] == str(out_dir / "skeleton3d_v2.json")
    assert skeleton["provenance"]["lane"] == "lane_b_2d_first"
    assert report["summary"]["root_sources"]["ankle_midpoint_ray_court_plane"] == 1


def test_score_skeleton_alignment_cli_scores_built_skeleton(tmp_path: Path) -> None:
    keypoints, tracks, calibration = _inputs()
    keypoints_path = tmp_path / "keypoints_2d.json"
    tracks_path = tmp_path / "tracks.json"
    calibration_path = tmp_path / "court_calibration.json"
    out_dir = tmp_path / "lift"
    score_path = tmp_path / "skeleton_alignment_metrics.json"
    _write_json(keypoints_path, keypoints)
    _write_json(tracks_path, tracks)
    _write_json(calibration_path, calibration)
    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_skeleton_from_2d.py",
            "--keypoints-2d",
            str(keypoints_path),
            "--tracks",
            str(tracks_path),
            "--court-calibration",
            str(calibration_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_skeleton_alignment.py",
            "--skeleton",
            str(out_dir / "skeleton3d_v2.json"),
            "--keypoints-2d",
            str(keypoints_path),
            "--court-calibration",
            str(calibration_path),
            "--out",
            str(score_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    report = json.loads(score_path.read_text(encoding="utf-8"))
    assert summary["metrics"] == str(score_path)
    assert report["projection_error_px"]["overall"]["p90"] < 1e-6
