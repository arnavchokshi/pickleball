from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.racket6dof import paddle_face_corners_object_cm
from threed.racketsport.schemas import RacketPose, validate_artifact_file


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _court_calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 320.0, "cy": 240.0, "dist": [], "source": "synthetic"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
            "camera_height_m": 12.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [],
        "world_pts": [],
    }


def _candidate_payload() -> dict:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    dims = {"length": 16.0, "width": 8.0}
    camera_matrix = [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]]
    object_points = np.asarray(paddle_face_corners_object_cm(dims), dtype=np.float64)
    rvec = np.asarray([[0.18], [-0.10], [0.07]], dtype=np.float64)
    tvec = np.asarray([[3.0], [-1.5], [95.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.asarray(camera_matrix), None)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 60.0,
        "players": [
            {
                "id": 7,
                "paddle_dims_in": dims,
                "frames": [
                    {
                        "t": 0.0,
                        "corners_px": projected.reshape(-1, 2).tolist(),
                        "conf": 0.92,
                        "source": "synthetic_corners",
                    }
                ],
            }
        ],
    }


def test_build_racket_pose_preview_cli_writes_schema_valid_preview(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_001"
    court = _write_json(run_dir / "court_calibration.json", _court_calibration())
    candidates = _write_json(run_dir / "racket_candidates.json", _candidate_payload())
    out = run_dir / "racket_pose_preview.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_pose_preview.py",
            "--court-calibration",
            str(court),
            "--racket-candidates",
            str(candidates),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    parsed = validate_artifact_file("racket_pose", out)
    assert isinstance(parsed, RacketPose)
    assert parsed.players[0].id == 7
    assert parsed.players[0].frames[0].source == "synthetic_corners:pnp_ippe_preview"
    assert parsed.players[0].frames[0].reprojection_error_px is not None
    summary = json.loads(completed.stdout)
    assert summary["out"] == str(out)
    assert summary["preview_frame_count"] == 1
    assert summary["not_gate_verified"] is True
