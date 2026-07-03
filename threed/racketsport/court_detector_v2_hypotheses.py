"""Multi-hypothesis court geometry generation for detector v2."""

from __future__ import annotations

from typing import Any, Mapping

from .court_templates import ft_to_m
from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


NET_TOP_KEYPOINT_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
FLOOR_KEYPOINT_NAMES = tuple(name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in NET_TOP_KEYPOINT_NAMES)

_HALF_WIDTH_M = ft_to_m(10.0)
_HALF_LENGTH_M = ft_to_m(22.0)
_NVZ_M = ft_to_m(7.0)
_FLOOR_WORLD_XY = {
    "near_left_corner": (-_HALF_WIDTH_M, -_HALF_LENGTH_M),
    "near_baseline_center": (0.0, -_HALF_LENGTH_M),
    "near_right_corner": (_HALF_WIDTH_M, -_HALF_LENGTH_M),
    "far_right_corner": (_HALF_WIDTH_M, _HALF_LENGTH_M),
    "far_baseline_center": (0.0, _HALF_LENGTH_M),
    "far_left_corner": (-_HALF_WIDTH_M, _HALF_LENGTH_M),
    "near_nvz_left": (-_HALF_WIDTH_M, -_NVZ_M),
    "near_nvz_center": (0.0, -_NVZ_M),
    "near_nvz_right": (_HALF_WIDTH_M, -_NVZ_M),
    "far_nvz_left": (-_HALF_WIDTH_M, _NVZ_M),
    "far_nvz_center": (0.0, _NVZ_M),
    "far_nvz_right": (_HALF_WIDTH_M, _NVZ_M),
}


def generate_court_hypotheses(
    *,
    image_size: tuple[int, int],
    net_evidence: Mapping[str, Any],
    surface_evidence: Mapping[str, Any],
    learned_keypoints: Mapping[str, Any] | None = None,
    partial_visible_floor_points: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")

    hypotheses: list[dict[str, Any]] = []
    if learned_keypoints:
        hypotheses.append(
            _hypothesis(
                "hypothesis_0001",
                "learned_keypoint_seed",
                width=width,
                height=height,
                correspondence_names=_floor_names_from_mapping(learned_keypoints),
                evidence_score=0.45,
            )
        )
    if partial_visible_floor_points:
        hypotheses.append(
            _hypothesis(
                f"hypothesis_{len(hypotheses) + 1:04d}",
                "visible_floor_partial_seed",
                width=width,
                height=height,
                correspondence_names=_floor_names_from_mapping(partial_visible_floor_points),
                evidence_score=0.5,
            )
        )

    line_candidates = list(surface_evidence.get("semantic_line_candidates") or [])
    if line_candidates:
        hypotheses.append(
            _hypothesis(
                f"hypothesis_{len(hypotheses) + 1:04d}",
                "kitchen_first_line_seed",
                width=width,
                height=height,
                correspondence_names=FLOOR_KEYPOINT_NAMES,
                evidence_score=min(0.75, 0.1 + 0.08 * len(line_candidates)),
                semantic_line_count=len(line_candidates),
                required_lines_present=len(line_candidates) >= 4,
            )
        )

    hypotheses.append(
        _hypothesis(
            f"hypothesis_{len(hypotheses) + 1:04d}",
            "net_roi_line_seed",
            width=width,
            height=height,
            correspondence_names=("near_nvz_left", "near_nvz_right", "far_nvz_left", "far_nvz_right"),
            evidence_score=float(net_evidence.get("confidence", 0.0)) * 0.25,
        )
    )
    hypotheses.append(
        _hypothesis(
            f"hypothesis_{len(hypotheses) + 1:04d}",
            "net_only_blocked_seed",
            width=width,
            height=height,
            correspondence_names=("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"),
            evidence_score=float(net_evidence.get("confidence", 0.0)) * 0.1,
        )
    )
    return hypotheses


def _floor_names_from_mapping(points: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(name for name in FLOOR_KEYPOINT_NAMES if name in points)


def _hypothesis(
    hypothesis_id: str,
    source: str,
    *,
    width: int,
    height: int,
    correspondence_names: tuple[str, ...],
    evidence_score: float,
    semantic_line_count: int = 0,
    required_lines_present: bool = False,
) -> dict[str, Any]:
    projected = _template_projection(width=width, height=height)
    floor_correspondence_names = [name for name in correspondence_names if name in FLOOR_KEYPOINT_NAMES]
    return {
        "hypothesis_id": hypothesis_id,
        "source": source,
        "floor_correspondence_names": floor_correspondence_names,
        "projected_keypoints": {name: projected[name] for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in projected},
        "line_support": {
            "required_lines_present": bool(required_lines_present),
            "semantic_line_count": int(semantic_line_count),
        },
        "regulation_ratio_errors": {"source": "template_projection_seed", "max_abs_error": 0.0},
        "score_components": {"evidence_score": round(max(0.0, min(1.0, evidence_score)), 4)},
        "promotion_allowed": False,
    }


def _template_projection(*, width: int, height: int) -> dict[str, list[float]]:
    x_left, x_mid, x_right = width * 0.18, width * 0.5, width * 0.82
    y_far, y_far_nvz, y_net, y_near_nvz, y_near = height * 0.35, height * 0.44, height * 0.5, height * 0.60, height * 0.82
    return {
        "near_left_corner": [x_left, y_near],
        "near_baseline_center": [x_mid, y_near],
        "near_right_corner": [x_right, y_near],
        "far_right_corner": [x_right, y_far],
        "far_baseline_center": [x_mid, y_far],
        "far_left_corner": [x_left, y_far],
        "near_nvz_left": [x_left, y_near_nvz],
        "near_nvz_center": [x_mid, y_near_nvz],
        "near_nvz_right": [x_right, y_near_nvz],
        "net_left_sideline": [x_left, y_net],
        "net_center": [x_mid, y_net],
        "net_right_sideline": [x_right, y_net],
        "far_nvz_left": [x_left, y_far_nvz],
        "far_nvz_center": [x_mid, y_far_nvz],
        "far_nvz_right": [x_right, y_far_nvz],
    }
