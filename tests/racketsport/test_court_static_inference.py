from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from threed.racketsport.court_static_inference import (
    _appearance_static_diagnostics,
    infer_static_court_model,
)
from threed.racketsport.court_static_lock import read_court_lock
from threed.racketsport.court_structured_solver import FLOOR_KEYPOINT_NAMES, FLOOR_WORLD_XY_M


_H = np.asarray(
    [[38.0, 4.0, 160.0], [-2.0, 23.0, 180.0], [0.002, -0.004, 1.0]],
    dtype=np.float64,
)


def _project(homography: np.ndarray, xy: tuple[float, float]) -> list[float]:
    value = homography @ np.asarray([xy[0], xy[1], 1.0], dtype=np.float64)
    return [float(value[0] / value[2]), float(value[1] / value[2])]


def _result(homography: np.ndarray) -> dict:
    projected = {name: _project(homography, FLOOR_WORLD_XY_M[name]) for name in FLOOR_KEYPOINT_NAMES}
    line = np.zeros((360, 640), dtype=np.uint8)
    for left, right in (
        ("near_left_corner", "near_right_corner"),
        ("far_left_corner", "far_right_corner"),
        ("near_left_corner", "far_left_corner"),
        ("near_right_corner", "far_right_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
    ):
        cv2.line(line, tuple(np.rint(projected[left]).astype(int)), tuple(np.rint(projected[right]).astype(int)), 1, 3)
    surface = np.zeros((360, 640), dtype=np.uint8)
    corners = np.asarray(
        [projected[name] for name in ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(surface, corners, 2)
    observations = [
        {
            "keypoint_name": name,
            "primary_peak": {"source_xy": xy},
            "observation_xy": xy,
            "covariance_px2": [[1.0, 0.0], [0.0, 1.0]],
            "confidence": 0.9,
        }
        for name, xy in projected.items()
    ]
    return {
        "keypoints_xy": projected,
        "keypoints_conf": {name: 0.9 for name in projected},
        "keypoints_vis": {name: 0.95 for name in projected},
        "line_family_mask": line,
        "surface_mask": surface,
        "structured_observations": observations,
        "best_court": {
            "keypoints_xy": projected,
            "point_confidence": {name: 0.9 for name in projected},
            "court_confidence": 0.85,
            "homography_image_from_court": homography.tolist(),
            "transform_covariance": (np.eye(8) * 1.0e-10).tolist(),
            "residual_stats_px": {"median": 0.5, "p90": 0.8},
            "score_components": {"line_alignment": 0.95, "surface_overlap": 0.9},
            "supported_view_probability": 0.99,
            "source": "observation_hypothesis",
        },
    }


def _patch_model(monkeypatch, *, moving: bool) -> None:
    import threed.racketsport.court_static_inference as module

    monkeypatch.setattr(module, "load_court_model_checkpoint", lambda *_args, **_kwargs: {"sha256": "a" * 64})
    dummy = type("Dummy", (), {"_point_confidence_calibrator": None, "_court_confidence_calibrator": None})()
    monkeypatch.setattr(module, "build_court_model_from_checkpoint", lambda *_args, **_kwargs: (dummy, [], (640, 360)))

    def infer(frame, **_kwargs):
        homography = _H.copy()
        if moving:
            homography[0, 2] += float(frame[0, 0, 0]) * 8.0
        return _result(homography)

    monkeypatch.setattr(module, "_infer_court_model_with_loaded_model", infer)


def test_static_frames_produce_one_serialized_review_only_lock(monkeypatch, tmp_path: Path) -> None:
    _patch_model(monkeypatch, moving=False)
    frames = [np.full((360, 640, 3), index, dtype=np.uint8) for index in range(8)]
    destination = tmp_path / "court_lock.json"

    result = infer_static_court_model(frames, tmp_path / "unused.pt", court_lock_path=destination)

    assert result["static_motion"]["status"] == "static"
    assert result["source"] == "multi_frame_point_and_line"
    assert result["court_lock"]["measurement_valid"] is False
    assert result["court_lock"]["authority_state"] == "review_only"
    assert result["selected_frame_indices"] == list(range(8))
    assert read_court_lock(destination).to_dict() == result["court_lock"]


def test_moving_camera_returns_clearest_frame_without_static_lock(monkeypatch, tmp_path: Path) -> None:
    _patch_model(monkeypatch, moving=True)
    frames = [np.full((360, 640, 3), index, dtype=np.uint8) for index in range(5)]

    result = infer_static_court_model(frames, tmp_path / "unused.pt")

    assert result["static_motion"]["status"] == "moving"
    assert result["source"] == "clearest_frame_point_and_line"
    assert result["court_lock"] is None
    assert result["best_court"].get("measurement_valid", False) is not True


def test_background_flow_calls_fixed_camera_static_despite_moving_foreground() -> None:
    rng = np.random.default_rng(13)
    background = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)
    frames = []
    for index in range(8):
        frame = background.copy()
        cv2.rectangle(frame, (40 + index * 35, 130), (140 + index * 35, 330), (0, 0, 0), -1)
        frames.append(frame)

    diagnostics = _appearance_static_diagnostics(frames, frame_indices=list(range(8)))

    assert diagnostics["status"] == "static"
    assert diagnostics["drift_px_p95"] <= 1.5
    assert diagnostics["mean_inlier_ratio"] >= 0.45


def test_background_flow_detects_real_camera_translation() -> None:
    rng = np.random.default_rng(29)
    background = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)
    frames = [
        cv2.warpAffine(
            background,
            np.asarray([[1.0, 0.0, index * 2.0], [0.0, 1.0, index * 0.75]], dtype=np.float32),
            (640, 360),
        )
        for index in range(8)
    ]

    diagnostics = _appearance_static_diagnostics(frames, frame_indices=list(range(8)))

    assert diagnostics["status"] == "moving"
    assert diagnostics["drift_px_p95"] >= 4.0
