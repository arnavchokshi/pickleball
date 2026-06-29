from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.racket_stage_runner import RacketStageRunner
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
        "image_pts": minimal_calibration_image_pts(),
        "world_pts": minimal_calibration_world_pts(),
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


def test_racket_stage_runner_writes_schema_valid_pose_from_explicit_four_corner_candidates(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(inputs / "racket_candidates.json", _candidate_payload())
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    result = RacketStageRunner().run(context)

    assert result.status == "ran"
    assert result.source_mode == "explicit_four_corner_candidates_pnp_ippe"
    assert result.metrics["candidate_frame_count"] == 1
    assert result.metrics["accepted_frame_count"] == 1
    parsed = validate_artifact_file("racket_pose", run_dir / "racket_pose.json")
    assert isinstance(parsed, RacketPose)
    assert parsed.players[0].id == 7
    assert parsed.players[0].frames[0].world_frame == "camera"
    assert parsed.players[0].frames[0].translation_unit == "cm"
    assert parsed.players[0].frames[0].reprojection_error_px is not None
    assert parsed.players[0].frames[0].reprojection_error_px < 0.75


def test_racket_stage_runner_rejects_box_derived_candidates_before_promotion(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    payload = _candidate_payload()
    payload["players"][0]["frames"][0]["source"] = "label_bbox:yolo26m_teacher"
    _write_json(inputs / "racket_candidates.json", payload)
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    with pytest.raises(ValueError, match="box-derived racket candidates cannot promote"):
        RacketStageRunner().run(context)

    assert not (run_dir / "racket_pose.json").exists()
    diagnostics = json.loads((run_dir / "racket_stage_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["status"] == "failed"
    assert diagnostics["metrics"]["rejected_box_derived_source_count"] == 1
    assert diagnostics["notes"][0] == "no racket_pose.json written because candidates are box-derived preview evidence"


def test_racket_stage_runner_fails_closed_when_candidates_are_missing(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    inputs.mkdir(parents=True)
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    with pytest.raises(FileNotFoundError, match="missing racket candidate artifact"):
        RacketStageRunner().run(context)

    assert not (run_dir / "racket_pose.json").exists()


def test_racket_stage_runner_rejects_schema_invalid_candidate_artifact(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    payload = _candidate_payload()
    payload["players"][0]["frames"][0]["unregistered_field"] = "must_fail_closed"
    _write_json(inputs / "racket_candidates.json", payload)
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    with pytest.raises(ValidationError):
        RacketStageRunner().run(context)

    assert not (run_dir / "racket_pose.json").exists()


def test_racket_stage_runner_fails_closed_when_all_candidates_are_rejected(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    payload = _candidate_payload()
    payload["players"][0]["frames"][0]["corners_px"][2][0] += 120.0
    payload["players"][0]["frames"][0]["corners_px"][2][1] -= 80.0
    _write_json(inputs / "racket_candidates.json", payload)
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    with pytest.raises(ValueError, match="no accepted racket pose frames"):
        RacketStageRunner(max_reprojection_error_px=2.0).run(context)

    assert not (run_dir / "racket_pose.json").exists()
    diagnostics = json.loads((run_dir / "racket_stage_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["artifact_type"] == "racketsport_racket_stage_diagnostics"
    assert diagnostics["status"] == "failed"
    assert diagnostics["metrics"]["candidate_frame_count"] == 1
    assert diagnostics["metrics"]["rejected_high_reprojection_count"] == 1
    assert diagnostics["metrics"]["accepted_frame_count"] == 0
