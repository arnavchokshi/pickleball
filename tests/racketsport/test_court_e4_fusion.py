from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame, detect_court_v2_from_frames
from threed.racketsport.court_detector_v2_hypotheses import generate_neural_seed_hypotheses


_WORLD_FT: dict[str, tuple[float, float]] = {
    "near_left_corner": (-10.0, -22.0),
    "near_baseline_center": (0.0, -22.0),
    "near_right_corner": (10.0, -22.0),
    "near_nvz_left": (-10.0, -7.0),
    "near_nvz_center": (0.0, -7.0),
    "near_nvz_right": (10.0, -7.0),
    "net_left_sideline": (-10.0, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_sideline": (10.0, 0.0),
    "far_nvz_left": (-10.0, 7.0),
    "far_nvz_center": (0.0, 7.0),
    "far_nvz_right": (10.0, 7.0),
    "far_left_corner": (-10.0, 22.0),
    "far_baseline_center": (0.0, 22.0),
    "far_right_corner": (10.0, 22.0),
}

_HOMOGRAPHY = [
    [15.0, 0.4, 320.0],
    [0.2, -7.0, 300.0],
    [0.0004, -0.001, 1.0],
]


def _projected_court(*, dx: float = 0.0, dy: float = 0.0) -> dict[str, tuple[float, float]]:
    projected = project_planar_points(_HOMOGRAPHY, [_WORLD_FT[name] for name in _WORLD_FT])
    return {
        name: (float(xy[0]) + float(dx), float(xy[1]) + float(dy))
        for name, xy in zip(_WORLD_FT, projected, strict=True)
    }


def _mock_inference(keypoints: dict[str, tuple[float, float]], *, confidence: float = 0.92) -> dict[str, Any]:
    return {
        "keypoints_xy": {name: [float(x), float(y)] for name, (x, y) in keypoints.items()},
        "keypoints_conf": {name: confidence for name in keypoints},
        "keypoints_vis": {name: confidence for name in keypoints},
        "line_family_mask": None,
        "surface_mask": None,
    }


def _hypothesis(
    hypothesis_id: str,
    keypoints: dict[str, tuple[float, float]],
    *,
    score: float,
    evidence_score: float,
    source_tag: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "template": "pickleball",
        "source": "real_line_bank_pickleball_assignment",
        "promotable_as_pickleball": True,
        "keypoints": keypoints,
        "projected_keypoints": {name: [float(x), float(y)] for name, (x, y) in keypoints.items()},
        "score": float(score),
        "evidence_score": float(evidence_score),
        "line_support": {"required_lines_present": True, "semantic_line_count": 8},
        "required_lines_present": True,
        "supported_line_count": 8,
        "score_components": {
            "evidence_score": float(evidence_score),
            "joint_template_competition": {"available": True, "winner": "pickleball", "margin": 1.0},
        },
    }
    if source_tag is not None:
        item["source_tag"] = source_tag
    return item


def _wrong_geometric_keypoints(frame_index: int) -> dict[str, tuple[float, float]]:
    keypoints = _projected_court(dx=20.0 * float(frame_index))
    for name in keypoints:
        if "corner" not in name:
            x, y = keypoints[name]
            keypoints[name] = (x + 150.0 + 110.0 * float(frame_index), y)
    return keypoints


def _median_reprojection_px(
    left: dict[str, Any],
    right: dict[str, tuple[float, float]],
) -> float:
    distances = []
    for name, xy in right.items():
        lx, ly = left[name]
        distances.append(math.hypot(float(lx) - xy[0], float(ly) - xy[1]))
    distances.sort()
    mid = len(distances) // 2
    return distances[mid]


def _patch_frame_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    import threed.racketsport.court_detector_v2 as detector

    monkeypatch.setattr(detector, "detect_court_net_evidence", lambda _frame: {"confidence": 1.0, "roi": {"y_min": 260, "y_max": 340}})
    monkeypatch.setattr(detector, "build_surface_paint_evidence", lambda _frame, roi=None: {"semantic_line_candidates": []})
    monkeypatch.setattr(detector, "generate_court_hypotheses", lambda **_kwargs: [])
    monkeypatch.setattr(detector, "compute_top_net_validation", lambda _net, _keypoints: {"passed": True})
    monkeypatch.setattr(detector, "compute_tennis_overlay_rejection", lambda _hypothesis: {"passed": True})


def test_e4_mock_neural_seed_adds_true_candidate_and_geor3_vote_selects_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import threed.racketsport.court_detector_v2 as detector

    _patch_frame_evidence(monkeypatch)
    gt = _projected_court()

    def fake_generate(frame_bgr: Any, **kwargs: Any) -> list[dict[str, Any]]:
        frame_index = int(frame_bgr[0, 0, 0])
        geometric = _hypothesis(
            f"geometric_wrong_{frame_index}",
            _wrong_geometric_keypoints(frame_index),
            score=1.0,
            evidence_score=0.88,
            source_tag="geometric" if kwargs.get("neural_inference") is not None else None,
        )
        neural = []
        if kwargs.get("neural_inference") is not None:
            neural = generate_neural_seed_hypotheses(
                kwargs["neural_inference"],
                image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
            )
            for item in neural:
                item["score"] = 2.0
                item["evidence_score"] = 0.78
                item["score_components"]["evidence_score"] = 0.78
        return [geometric, *neural]

    monkeypatch.setattr(detector, "generate_homography_hypotheses", fake_generate)

    frames = [np.full((540, 720, 3), fill_value=index, dtype=np.uint8) for index in range(3)]
    geometric_default = detect_court_v2_from_frames(frames, clip_id="synthetic_e4")
    geometric_explicit_none = detect_court_v2_from_frames(frames, clip_id="synthetic_e4", neural_infer=None)

    assert geometric_explicit_none == geometric_default
    assert geometric_default["selected_hypothesis"]["source_tag"] == "geometric"

    fused = detect_court_v2_from_frames(
        frames,
        clip_id="synthetic_e4",
        neural_infer=lambda _frame: _mock_inference(gt, confidence=0.93),
    )

    assert fused["geor3"]["triggered"] is True
    assert fused["geor3"]["selected"] is True
    assert fused["selected_hypothesis"]["source_tag"] == "neural_seeded"
    assert fused["selected_hypothesis"]["model_confidence"] == pytest.approx(0.93, abs=1e-6)
    assert _median_reprojection_px(fused["selected_hypothesis"]["keypoints"], gt) <= 3.0
    assert any(
        candidate["source_tag"] == "neural_seeded"
        for frame in fused["frames"]
        for candidate in frame["top_pickleball_hypotheses"]
    )


def test_e4_garbage_model_does_not_degrade_geometric_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import threed.racketsport.court_detector_v2 as detector

    _patch_frame_evidence(monkeypatch)
    gt = _projected_court()

    def fake_visible_error(_frame_bgr: Any, keypoints: dict[str, Any]) -> dict[str, Any]:
        median = _median_reprojection_px(keypoints, gt)
        if median <= 5.0:
            return {
                "floor_visible": {"median": 0.0, "p95": 0.0},
                "visible_corners": {"median": 0.0},
                "high_confidence_over_30px_count": 0,
            }
        return {
            "floor_visible": {"median": 150.0, "p95": 180.0},
            "visible_corners": {"median": 150.0},
            "high_confidence_over_30px_count": 0,
        }

    monkeypatch.setattr(detector, "compute_visible_error_px_against_evidence", fake_visible_error)

    def fake_generate(frame_bgr: Any, **kwargs: Any) -> list[dict[str, Any]]:
        geometric = _hypothesis(
            "geometric_true",
            gt,
            score=1.0,
            evidence_score=0.8,
            source_tag="geometric" if kwargs.get("neural_inference") is not None else None,
        )
        neural = []
        if kwargs.get("neural_inference") is not None:
            neural = generate_neural_seed_hypotheses(
                kwargs["neural_inference"],
                image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
            )
        return [geometric, *neural]

    monkeypatch.setattr(detector, "generate_homography_hypotheses", fake_generate)

    frame = np.zeros((540, 720, 3), dtype=np.uint8)
    geometric_only = detect_court_v2_from_frame(frame, clip_id="synthetic_e4")
    fused = detect_court_v2_from_frame(
        frame,
        clip_id="synthetic_e4",
        neural_infer=lambda _frame: _mock_inference(_projected_court(dx=260.0), confidence=0.99),
    )

    assert geometric_only["selected_hypothesis_id"] == "geometric_true"
    assert fused["selected_hypothesis_id"] == "geometric_true"
    assert len(fused["hypotheses"]) == len(geometric_only["hypotheses"]) + 1
    neural_candidates = [candidate for candidate in fused["hypotheses"] if candidate.get("source_tag") == "neural_seeded"]
    assert len(neural_candidates) == 1
    assert neural_candidates[0]["verification"]["promotion_allowed"] is False
