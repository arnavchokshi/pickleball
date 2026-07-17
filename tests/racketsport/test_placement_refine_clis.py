from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_CLI = "scripts/racketsport/build_refined_placement.py"
VALIDATE_CLI = "scripts/racketsport/validate_placement_slide.py"


def test_build_and_validate_direct_cli_references(tmp_path: Path) -> None:
    skeleton, tracks, phases, calibration = _fixtures()
    paths = {}
    for name, payload in {
        "skeleton": skeleton,
        "tracks": tracks,
        "phases": phases,
        "calibration": calibration,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    refined = tmp_path / "placement_trajectory_refined.json"
    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / BUILD_CLI),
            "--skeleton", str(paths["skeleton"]),
            "--tracks", str(paths["tracks"]),
            "--phases", str(paths["phases"]),
            "--out", str(refined),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert json.loads(refined.read_text())["artifact_type"] == "placement_trajectory_refined"

    metrics = tmp_path / "metrics.json"
    validate = subprocess.run(
        [
            sys.executable,
            str(ROOT / VALIDATE_CLI),
            "--skeleton", str(refined),
            "--phases", str(paths["phases"]),
            "--calibration", str(paths["calibration"]),
            "--tracks", str(paths["tracks"]),
            "--clip", "synthetic",
            "--out", str(metrics),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(metrics.read_text())["accepted_phase"]["phase_count"] >= 1


def _fixtures() -> tuple[dict, dict, dict, dict]:
    frames = []
    track_frames = []
    for index in range(3):
        root = [0.01 * index, 1.0, 1.0]
        frames.append(
            {
                "frame_idx": index,
                "t": index / 30.0,
                "transl_world": root,
                "joints_world": [
                    [root[0] - 0.1, 1.0, 0.01],
                    [root[0] + 0.1, 1.0, 0.20],
                    [root[0] - 0.12, 1.02, 0.01],
                    [root[0] + 0.12, 1.02, 0.20],
                ],
                "joint_conf": [0.98] * 4,
            }
        )
        track_frames.append(
            {
                "frame_idx": index,
                "t": index / 30.0,
                "world_xy": root[:2],
                "conf": 0.98,
                "bbox": [100.0, 100.0, 200.0, 300.0],
            }
        )
    skeleton = {
        "schema_version": 1,
        "artifact_type": "skeleton3d",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "joint_names": ["left_ankle", "right_ankle", "left_heel", "right_heel"],
        "players": [{"id": "p1", "frames": frames}],
    }
    tracks = {"schema_version": 1, "fps": 30.0, "players": [{"id": "p1", "frames": track_frames}]}
    phases = {
        "schema_version": 1,
        "artifact_type": "foot_contact_phases",
        "phases": [
            {
                "player_id": "p1",
                "foot": "left",
                "frame_indices": [0, 1, 2],
                "min_confidence": 0.96,
                "assignment_evidence": {"body_detector_agreement": 0.97},
            }
        ],
    }
    calibration = {
        "homography": [[100.0, 0.0, 150.0], [0.0, 100.0, 150.0], [0.0, 0.0, 1.0]],
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 2.0,
        },
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 150.0, "cy": 150.0, "dist": [], "source": "synthetic"},
    }
    return skeleton, tracks, phases, calibration
