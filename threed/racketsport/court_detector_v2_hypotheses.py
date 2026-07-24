"""Multi-hypothesis court geometry generation for detector v2."""

from __future__ import annotations

from collections.abc import Callable
import logging
import math
from typing import Any, Mapping, Sequence

from .court_templates import ft_to_m
from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


_LOGGER = logging.getLogger(__name__)
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


#
# ---------------------------------------------------------------------------
# Real Stage-2 homography hypothesis generation (CAL-GEO 2026-07-05).
#
# The functions above are the original template-projection stub (kept for
# backward compatibility / existing callers). Everything below builds actual
# candidate homographies from RANSAC-style combinatorial search over line-bank
# cross/sideline groups, with a joint pickleball-vs-tennis template
# competition so a tennis-overlay court stops polluting the pickleball fit.
# This is adapted from the proven `court_finding_technology_benchmark`
# regulation-line solver (its best selector reaches 289.5px mean floor median
# on the 5-sample benchmark), generalized to also emit tennis-tagged
# hypotheses for the joint competition and to consume surface-polygon
# evidence.
# ---------------------------------------------------------------------------

_FLOOR_WORLD_XY_FT: dict[str, tuple[float, float]] = {
    "near_left_corner": (-10.0, -22.0),
    "near_baseline_center": (0.0, -22.0),
    "near_right_corner": (10.0, -22.0),
    "far_right_corner": (10.0, 22.0),
    "far_baseline_center": (0.0, 22.0),
    "far_left_corner": (-10.0, 22.0),
    "near_nvz_left": (-10.0, -7.0),
    "near_nvz_center": (0.0, -7.0),
    "near_nvz_right": (10.0, -7.0),
    "net_left_sideline": (-10.0, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_sideline": (10.0, 0.0),
    "far_nvz_left": (-10.0, 7.0),
    "far_nvz_center": (0.0, 7.0),
    "far_nvz_right": (10.0, 7.0),
}

_PICKLEBALL_CROSS_WORLD_Y_FT = {"far_baseline": 22.0, "far_nvz": 7.0, "net": 0.0, "near_nvz": -7.0, "near_baseline": -22.0}
_TENNIS_CROSS_WORLD_Y_FT = {"far_baseline": 39.0, "far_service": 21.0, "net": 0.0, "near_service": -21.0, "near_baseline": -39.0}
_PICKLEBALL_LONG_WORLD_X_FT = {"left_sideline": -10.0, "centerline": 0.0, "right_sideline": 10.0}
_TENNIS_LONG_WORLD_X_FT = {"doubles_left": -18.0, "center_service": 0.0, "doubles_right": 18.0}

_PICKLEBALL_CROSS_SETS: tuple[tuple[str, ...], ...] = (
    ("far_baseline", "far_nvz", "near_nvz", "near_baseline"),
    ("far_baseline", "far_nvz", "near_nvz"),
    ("far_nvz", "near_nvz", "near_baseline"),
    ("far_baseline", "far_nvz", "net", "near_nvz", "near_baseline"),
)
_TENNIS_CROSS_SETS: tuple[tuple[str, ...], ...] = (
    ("far_baseline", "far_service", "near_service", "near_baseline"),
    ("far_baseline", "far_service", "near_service"),
    ("far_service", "near_service", "near_baseline"),
    ("far_baseline", "far_service", "net", "near_service", "near_baseline"),
)
_PICKLEBALL_LONG_SETS: tuple[tuple[str, ...], ...] = (
    ("left_sideline", "centerline", "right_sideline"),
    ("left_sideline", "right_sideline"),
    ("centerline", "right_sideline"),
    ("left_sideline", "centerline"),
)
_TENNIS_LONG_SETS: tuple[tuple[str, ...], ...] = (
    ("doubles_left", "center_service", "doubles_right"),
    ("doubles_left", "doubles_right"),
    ("center_service", "doubles_right"),
    ("doubles_left", "center_service"),
)

_PICKLEBALL_FLOOR_LINE_ENDPOINTS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_baseline_center", "far_nvz_center"),
}

# Round-2 dual-line support test: ALL template lines, including the floor-
# plane net line (the net band always produces strong edge segments, so a
# correctly placed net line finds segment support; a mid-court misplaced one
# does not).
_ALL_TEMPLATE_LINE_ENDPOINTS: dict[str, tuple[str, str]] = {
    **_PICKLEBALL_FLOOR_LINE_ENDPOINTS,
    "net": ("net_left_sideline", "net_right_sideline"),
}
_CROSS_TEMPLATE_LINE_NAMES = frozenset({"near_baseline", "near_nvz", "net", "far_nvz", "far_baseline"})


def _y_on_segment_at_x(p1: Sequence[float], p2: Sequence[float], x: float) -> float:
    """y of the infinite line through p1-p2 at the given x (midpoint y if vertical)."""

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x2 - x1) < 1e-9:
        return (y1 + y2) / 2.0
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _x_on_segment_at_y(p1: Sequence[float], p2: Sequence[float], y: float) -> float:
    """x of the infinite line through p1-p2 at the given y (midpoint x if horizontal)."""

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(y2 - y1) < 1e-9:
        return (x1 + x2) / 2.0
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _net_scale_depth_prior(
    keypoints: Mapping[str, tuple[float, float]],
    *,
    net_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Round-2 fix 3: net-scale depth prior / hard sanity band.

    Two independent constraints:
    - Perspective band: the near baseline is 22ft nearer the camera than the
      net, so its projected width can exceed the net-line width only by a
      bounded factor; and the far baseline can never be meaningfully WIDER
      than the net line. Violations get a steep penalty, making near-side
      depth explosion (Wolverine failure mode) score-prohibitive with no
      external evidence needed.
    - Tape-width prior (only when net evidence is confident): posts sit
      ~22ft apart while the painted court is 20ft wide, so the projected
      20ft net line should be ~(20/22) of the observed top-tape span.
    """

    result: dict[str, Any] = {"available": False, "penalty": 0.0, "components": {}}
    net_left = keypoints.get("net_left_sideline")
    net_right = keypoints.get("net_right_sideline")
    near_left = keypoints.get("near_left_corner")
    near_right = keypoints.get("near_right_corner")
    far_left = keypoints.get("far_left_corner")
    far_right = keypoints.get("far_right_corner")
    if any(point is None for point in (net_left, net_right, near_left, near_right, far_left, far_right)):
        return result
    net_width = math.dist(net_left, net_right)
    near_width = math.dist(near_left, near_right)
    far_width = math.dist(far_left, far_right)
    if net_width <= 1e-6:
        return result
    penalty = 0.0
    components: dict[str, Any] = {}
    near_ratio = near_width / net_width
    far_ratio = far_width / net_width
    components["near_over_net_width_ratio"] = round(near_ratio, 4)
    components["far_over_net_width_ratio"] = round(far_ratio, 4)
    penalty += max(0.0, near_ratio - 2.6) * 260.0 + max(0.0, 0.85 - near_ratio) * 260.0
    penalty += max(0.0, far_ratio - 1.08) * 320.0

    tape = (net_evidence or {}).get("top_tape_line")
    confidence = float((net_evidence or {}).get("confidence") or 0.0)
    if isinstance(tape, (list, tuple)) and len(tape) == 2 and confidence >= 0.45:
        tape_width = math.dist(tape[0], tape[1])
        if tape_width > 1e-6:
            expected_net_width = tape_width * (20.0 / 22.0)
            log_ratio = abs(math.log(max(1e-6, net_width / expected_net_width)))
            components["observed_tape_width_px"] = round(tape_width, 2)
            components["expected_net_width_px"] = round(expected_net_width, 2)
            components["net_width_px"] = round(net_width, 2)
            components["net_width_log_ratio"] = round(log_ratio, 4)
            penalty += max(0.0, log_ratio - math.log(1.6)) * 420.0
            components["tape_prior_used"] = True
    result.update({"available": True, "penalty": round(float(penalty), 4), "components": components})
    return result


def _lower_center_region_overlap(
    keypoints: Mapping[str, tuple[float, float]],
    *,
    width: int,
    height: int,
) -> float:
    """Round-2 fix 5: fraction of the lower-center image region covered by the court quad.

    The camera-facing (foreground) court dominates the lower-center of the
    frame in essentially all real capture; an adjacent/background court's
    quad does not. Rasterized at 1/16 scale, cv2-only.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except Exception:  # pragma: no cover
        return 0.0
    corner_names = ("near_left_corner", "far_left_corner", "far_right_corner", "near_right_corner")
    if any(keypoints.get(name) is None for name in corner_names):
        return 0.0
    scale = 1.0 / 16.0
    grid_w = max(4, int(round(width * scale)))
    grid_h = max(4, int(round(height * scale)))
    canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)
    quad = np.array(
        [[keypoints[name][0] * scale, keypoints[name][1] * scale] for name in corner_names],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [quad], 1)
    x0, x1 = int(grid_w * 0.20), int(grid_w * 0.80)
    y0, y1 = int(grid_h * 0.45), int(grid_h * 0.98)
    region = canvas[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.sum()) / float(region.size)


def _spacing_error_scale_invariant(observed: Any, expected: Any) -> float:
    if len(observed) != len(expected) or not observed:
        return 0.0
    denom = sum(value * value for value in expected)
    if denom <= 1e-6:
        return 0.0
    scale = sum(float(obs) * float(exp) for obs, exp in zip(observed, expected)) / denom
    if scale <= 1e-6:
        return float("inf")
    return sum(
        abs(float(obs) - scale * float(exp)) / max(1.0, scale * float(exp)) for obs, exp in zip(observed, expected)
    ) / len(observed)


def _cross_ratio(a: float, b: float, c: float, d: float) -> float | None:
    """Projective cross-ratio of four collinear scalar positions."""

    bc = b - c
    bd = b - d
    ac = a - c
    ad = a - d
    if abs(bc) < 1e-9 or abs(ad) < 1e-9:
        return None
    return (ac / bc) / (ad / bd) if abs(bd) > 1e-9 else None


def _cross_ratio_error(observed: list[float], expected: list[float]) -> float:
    """Sum of |log CR ratio| residuals over consecutive 4-position windows.

    The cross-ratio of 4 collinear points is invariant under ANY homography,
    unlike gap ratios: a single-scale (affine) spacing residual actively
    EXCLUDES the correct perspective fit whose near-side gaps are 2x+ its
    far-side gaps -- the exact measured cause of the Outdoor compressed-fit
    win in round 1. Sets of 3 points impose no projective constraint at all
    and are ranked by support alone.
    """

    if len(observed) != len(expected) or len(observed) < 4:
        return 0.0
    total = 0.0
    windows = 0
    for start in range(len(observed) - 3):
        obs_cr = _cross_ratio(*observed[start : start + 4])
        exp_cr = _cross_ratio(*expected[start : start + 4])
        if obs_cr is None or exp_cr is None or obs_cr <= 0.0 or exp_cr <= 0.0:
            return float("inf")
        total += abs(math.log(obs_cr / exp_cr))
        windows += 1
    return total / windows if windows else 0.0


def _ranked_cross_assignments(pool: list[Any], labels: tuple[str, ...], world_y_ft: Mapping[str, float], *, x_ref: float, limit: int) -> list[dict[str, Any]]:
    from itertools import combinations

    from .court_line_bank import line_y_at_x, segment_midpoint

    if len(pool) < len(labels):
        return []
    expected_positions = [world_y_ft[label] for label in labels]
    ranked: list[tuple[float, dict[str, Any]]] = []
    for combo in combinations(pool, len(labels)):
        ordered = sorted(combo, key=lambda group: line_y_at_x(group.line, x_ref, fallback=segment_midpoint(group.segment)[1]))
        positions = [line_y_at_x(group.line, x_ref, fallback=segment_midpoint(group.segment)[1]) for group in ordered]
        distances = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
        if any(distance <= 10.0 for distance in distances):
            continue
        # Projective-correct ranking: cross-ratio residual for 4+ lines;
        # 3-line subsets are projectively unconstrained -> support-only.
        geometry_error = _cross_ratio_error(positions, sorted(expected_positions, reverse=True))
        if not math.isfinite(geometry_error):
            continue
        support = sum(group.support_length_px for group in ordered)
        score = geometry_error * 400.0 - min(60.0, support / 45.0)
        ranked.append((score, dict(zip(labels, ordered))))
    ranked.sort(key=lambda item: item[0])
    return [assignment for _, assignment in ranked[:limit]]


def _ranked_long_assignments(pool: list[Any], labels: tuple[str, ...], world_x_ft: Mapping[str, float], *, y_ref: float, limit: int) -> list[dict[str, Any]]:
    from itertools import combinations

    from .court_line_bank import line_x_at_y, segment_midpoint

    if len(pool) < len(labels):
        return []
    expected_positions = [world_x_ft[label] for label in labels]
    expected_distances = [abs(expected_positions[index + 1] - expected_positions[index]) for index in range(len(labels) - 1)]
    ranked: list[tuple[float, dict[str, Any]]] = []
    for combo in combinations(pool, len(labels)):
        ordered = sorted(combo, key=lambda group: line_x_at_y(group.line, y_ref, fallback=segment_midpoint(group.segment)[0]))
        positions = [line_x_at_y(group.line, y_ref, fallback=segment_midpoint(group.segment)[0]) for group in ordered]
        distances = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
        if any(distance <= 12.0 for distance in distances):
            continue
        spacing_error = 0.0 if not expected_distances else _spacing_error_scale_invariant(distances, expected_distances)
        support = sum(group.support_length_px for group in ordered)
        separation_bonus = min(50.0, sum(distances) / 18.0)
        score = spacing_error * 80.0 - min(60.0, support / 45.0) - separation_bonus
        ranked.append((score, dict(zip(labels, ordered))))
    ranked.sort(key=lambda item: item[0])
    return [assignment for _, assignment in ranked[:limit]]


def _projected_pickleball_court_is_plausible(keypoints: Mapping[str, tuple[float, float]], *, width: int, height: int) -> bool:
    from .court_line_bank import point_is_finite

    margin = max(width, height) * 0.40
    for xy in keypoints.values():
        if not point_is_finite(xy):
            return False
        if xy[0] < -margin or xy[0] > width + margin or xy[1] < -margin or xy[1] > height + margin:
            return False
    required = ("far_left_corner", "far_right_corner", "near_nvz_left", "near_nvz_right", "near_left_corner", "near_right_corner")
    if not all(name in keypoints for name in required):
        return False
    far_y = (keypoints["far_left_corner"][1] + keypoints["far_right_corner"][1]) / 2.0
    near_nvz_y = (keypoints["near_nvz_left"][1] + keypoints["near_nvz_right"][1]) / 2.0
    near_y = (keypoints["near_left_corner"][1] + keypoints["near_right_corner"][1]) / 2.0
    if not far_y < near_nvz_y < near_y:
        return False
    near_width = math.dist(keypoints["near_left_corner"], keypoints["near_right_corner"])
    far_width = math.dist(keypoints["far_left_corner"], keypoints["far_right_corner"])
    if near_width < max(40.0, width * 0.08) or far_width < max(20.0, width * 0.03):
        return False

    # NOTE: a stricter absolute-scale "floor-collapse" guard (bbox height +
    # far/near width ratio, ported from the proven benchmark's
    # `_proposal_geometry_risk_score`) was tried here and measured on the
    # real 5-sample benchmark: it left Outdoor's selected hypothesis
    # completely unchanged (proving that clip's error is not a collapsed-
    # scale problem) while making Burlington and Wolverine's mean floor
    # median measurably WORSE by eliminating previously-selected
    # hypotheses in favor of worse alternatives. Reverted rather than kept
    # for an unproven theory; see report.md HONEST ISSUES for the real,
    # still-open root cause (self-consistent-but-wrong line assignment).
    return True


def generate_neural_seed_hypotheses(
    neural_inference: Mapping[str, Any] | None,
    *,
    image_size: tuple[int, int],
    min_keypoint_confidence: float = 0.05,
    min_keypoint_visibility: float = 0.05,
    min_floor_correspondences: int = 4,
) -> list[dict[str, Any]]:
    """Convert a `court_unet_v2` inference dict into hypothesis-pool candidates.

    The network provides pixel keypoints; this function fits one regulation
    homography from confident floor points and emits it as a review-only
    pickleball hypothesis. The model confidence is exposed separately from the
    geometric score so callers can report the selection source honestly.
    """

    from .court_calibration import homography_from_planar_points, project_planar_points

    if not isinstance(neural_inference, Mapping):
        return []
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")

    best_court = neural_inference.get("best_court")
    structured_selected = isinstance(best_court, Mapping) and isinstance(
        best_court.get("keypoints_xy"), Mapping
    )
    if structured_selected:
        keypoints_xy = best_court.get("keypoints_xy") or {}
        keypoints_conf = best_court.get("point_confidence") or {}
        # A structured court projects occluded points from one template.  Its
        # per-point confidence already incorporates observation/inference state;
        # do not gate those regenerated points on raw visibility again.
        keypoints_vis = {str(name): 1.0 for name in keypoints_xy}
    else:
        keypoints_xy = neural_inference.get("keypoints_xy") or neural_inference.get("keypoints") or {}
        keypoints_conf = neural_inference.get("keypoints_conf") or {}
        keypoints_vis = neural_inference.get("keypoints_vis") or {}
    if not isinstance(keypoints_xy, Mapping):
        return []

    world_points: list[tuple[float, float]] = []
    image_points: list[tuple[float, float]] = []
    used_names: list[str] = []
    confidence_values: list[float] = []
    visibility_values: list[float] = []
    for name in FLOOR_KEYPOINT_NAMES:
        xy = keypoints_xy.get(name)
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            continue
        x = _finite_float(xy[0])
        y = _finite_float(xy[1])
        if x is None or y is None:
            continue
        confidence = _probability_from_mapping(keypoints_conf, name, default=1.0)
        visibility = _probability_from_mapping(keypoints_vis, name, default=1.0)
        if confidence < float(min_keypoint_confidence) or visibility < float(min_keypoint_visibility):
            continue
        world_points.append(_FLOOR_WORLD_XY_FT[name])
        image_points.append((float(x), float(y)))
        used_names.append(name)
        confidence_values.append(float(confidence))
        visibility_values.append(float(visibility))

    if len(image_points) < int(min_floor_correspondences):
        return []

    try:
        homography = homography_from_planar_points(world_points, image_points)
        projected_raw = project_planar_points(homography, [_FLOOR_WORLD_XY_FT[name] for name in _FLOOR_WORLD_XY_FT])
        reprojected_inputs = project_planar_points(homography, world_points)
    except Exception:
        return []

    keypoints = {name: (float(xy[0]), float(xy[1])) for name, xy in zip(_FLOOR_WORLD_XY_FT, projected_raw, strict=True)}
    if not _projected_pickleball_court_is_plausible(keypoints, width=width, height=height):
        return []

    residuals = [
        math.hypot(float(observed[0]) - float(projected[0]), float(observed[1]) - float(projected[1]))
        for observed, projected in zip(image_points, reprojected_inputs, strict=True)
    ]
    fit_median = _median_float(residuals)
    fit_p95 = _p95_float(residuals)
    model_confidence = (
        _probability_from_mapping({"court": best_court.get("court_confidence")}, "court", default=0.0)
        if structured_selected
        else _mean_float(confidence_values)
    )
    model_visibility = _mean_float(visibility_values)
    combined_confidence = _mean_float([c * v for c, v in zip(confidence_values, visibility_values, strict=True)])
    fit_quality = max(0.0, min(1.0, 1.0 - fit_p95 / 35.0))
    evidence_score = max(0.02, min(0.98, 0.18 + 0.72 * combined_confidence * fit_quality))
    score = 35.0 - 120.0 * evidence_score + fit_p95 * 1.2
    required_lines_present = len(used_names) >= 8 and fit_p95 <= 20.0 and combined_confidence >= 0.10

    return [
        {
            "score": float(score),
            "evidence_score": float(evidence_score),
            "template": "pickleball",
            "promotable_as_pickleball": False,
            "source": (
                "structured_best_court_seed"
                if structured_selected
                else "neural_seeded_court_unet_v2"
            ),
            "source_tag": "structured_best_court" if structured_selected else "neural_seeded",
            "model_confidence": round(float(model_confidence), 6),
            "model_visibility": round(float(model_visibility), 6),
            "model_combined_confidence": round(float(combined_confidence), 6),
            "keypoints": keypoints,
            "supported_line_count": min(8, len(used_names)),
            "required_lines_present": bool(required_lines_present),
            "line_assignment": {},
            "neural_seed": {
                "provider": "structured_best_court" if structured_selected else "court_unet_v2",
                "structured_solver_source": (
                    str(best_court.get("source")) if structured_selected else None
                ),
                "floor_correspondence_count": len(used_names),
                "floor_correspondence_names": list(used_names),
                "fit_residual_median_px": round(float(fit_median), 4),
                "fit_residual_p95_px": round(float(fit_p95), 4),
            },
            "score_components": {
                "evidence_score": round(float(evidence_score), 4),
                "model_confidence": round(float(model_confidence), 6),
                "model_visibility": round(float(model_visibility), 6),
                "model_combined_confidence": round(float(combined_confidence), 6),
                "neural_fit_residual_median_px": round(float(fit_median), 4),
                "neural_fit_residual_p95_px": round(float(fit_p95), 4),
                "joint_template_competition": {
                    "available": True,
                    "winner": "pickleball",
                    "margin": round(float(combined_confidence), 6),
                    "source": "neural_seed_geometry",
                },
            },
        }
    ]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _probability_from_mapping(values: Any, name: str, *, default: float) -> float:
    if not isinstance(values, Mapping):
        return float(default)
    value = _finite_float(values.get(name))
    if value is None:
        return float(default)
    return max(0.0, min(1.0, float(value)))


def _mean_float(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _p95_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _hypothesis_from_line_assignment(
    *,
    template: str,
    cross_assignment: Mapping[str, Any],
    long_assignment: Mapping[str, Any],
    world_cross_ft: Mapping[str, float],
    world_long_ft: Mapping[str, float],
    image_bgr: Any,
    image_size: tuple[int, int],
    x_ref: float,
    y_ref: float,
    vp_families: list[dict[str, Any]],
    surface_evidence: Mapping[str, Any] | None,
    net_evidence: Mapping[str, Any] | None,
    segments: Sequence[Mapping[str, Any]],
    strong_cross_lines: Sequence[Any],
    pixel_mask: Any,
    color_mask: Any,
    distance_mask: Any,
    distance_map: Any,
) -> dict[str, Any] | None:
    from .court_calibration import homography_from_planar_points, project_planar_points
    from .court_line_bank import (
        angle_diff_mod_180,
        evaluate_projected_template_lines,
        line_intersection,
        line_y_at_x,
        point_inside_loose_bounds,
        point_is_finite,
        score_line_color_consistency_for_assignment,
        score_projected_line_distance_transform_against_image,
        score_projected_line_pixels_against_image,
        segment_midpoint,
    )
    from .court_template_competition import score_joint_template_competition

    width, height = image_size
    if len({id(group) for group in cross_assignment.values()} | {id(group) for group in long_assignment.values()}) != len(cross_assignment) + len(long_assignment):
        return None

    world_points: list[list[float]] = []
    image_points: list[list[float]] = []
    for cross_name, cross_group in cross_assignment.items():
        world_y = world_cross_ft[cross_name]
        for long_name, long_group in long_assignment.items():
            world_x = world_long_ft[long_name]
            try:
                xy = line_intersection(cross_group.line, long_group.line)
            except ValueError:
                continue
            if not (point_is_finite(xy) and point_inside_loose_bounds(xy, width=width, height=height)):
                continue
            world_points.append([float(world_x), float(world_y)])
            image_points.append([float(xy[0]), float(xy[1])])
    if len(world_points) < 4:
        return None

    try:
        homography = homography_from_planar_points(world_points, image_points)
        projected = project_planar_points(homography, [_FLOOR_WORLD_XY_FT[name] for name in _FLOOR_WORLD_XY_FT])
        reprojected_inputs = project_planar_points(homography, world_points)
    except Exception:
        return None
    keypoints = {name: (float(xy[0]), float(xy[1])) for name, xy in zip(_FLOOR_WORLD_XY_FT, projected)}
    if not _projected_pickleball_court_is_plausible(keypoints, width=width, height=height):
        return None

    # Self-consistency guard: with >4 correspondence points (multiple sideline
    # x-intersections per cross-line), the DLT solve is a least-squares fit.
    # If e.g. one sideline candidate is a real regulation line but the other
    # is a spurious/adjacent-court line, the fit can end up distorted overall
    # while still "explaining" the correct sideline reasonably -- a symptom
    # observed directly on real clips (one sideline near-perfect vs reviewed
    # labels, the other and the whole court badly wrong). Reject any
    # assignment whose own input correspondences do not reproject close to
    # where they were actually observed.
    self_consistency_residuals = [
        math.hypot(observed[0] - reprojected[0], observed[1] - reprojected[1])
        for observed, reprojected in zip(image_points, reprojected_inputs)
    ]
    self_consistency_rmse = (sum(value * value for value in self_consistency_residuals) / len(self_consistency_residuals)) ** 0.5
    # Hard-reject only truly degenerate fits (an order of magnitude worse than
    # real line-detection noise); otherwise this becomes a soft ranking
    # penalty below so hypotheses are not thrown out wholesale on noisy real
    # footage, but a self-inconsistent (likely mixed-court) assignment always
    # loses to a more self-consistent one when both exist.
    if self_consistency_rmse > 220.0:
        return None

    endpoint_pairs = _PICKLEBALL_FLOOR_LINE_ENDPOINTS
    pixel_support = score_projected_line_pixels_against_image(
        image_bgr, keypoints, endpoint_pairs=endpoint_pairs, line_pixel_mask=pixel_mask
    )
    distance_support = score_projected_line_distance_transform_against_image(
        image_bgr, keypoints, endpoint_pairs=endpoint_pairs, line_pixel_mask=distance_mask, distance_map=distance_map
    )
    color_consistency = score_line_color_consistency_for_assignment(
        image_bgr, {**cross_assignment, **long_assignment}, line_pixel_mask=color_mask
    )

    pixel_mean_support = float(pixel_support.get("mean_line_pixel_support_ratio") or 0.0)
    pixel_supported = int(pixel_support.get("supported_line_pixel_count") or 0)
    distance_mean = float(distance_support.get("mean_projected_line_distance_px") or 1e9)
    distance_supported = int(distance_support.get("distance_supported_line_count") or 0)
    color_penalty = min(45.0, float(color_consistency.get("mixed_layer_penalty") or 0.0) * 0.35)

    # ROUND-2 FIX 1+2: DUAL-LINE SUPPORT TEST with off-frame visibility.
    # Every template line whose projected in-image portion is observable MUST
    # have paint/segment support from the frame's full line bank; observable-
    # but-unsupported CROSS lines carry a heavy penalty (this is what kills a
    # kitchen-as-baseline compressed fit, whose projected NVZ lands in
    # unpainted mid-court). Unobservable (off-frame) lines carry ZERO penalty
    # and are recorded as such.
    dual_line_support = evaluate_projected_template_lines(
        keypoints,
        endpoint_pairs=_ALL_TEMPLATE_LINE_ENDPOINTS,
        segments=segments,
        pixel_mask=pixel_mask,
        image_size=image_size,
    )
    dual_line_penalty = 0.0
    for line_name, record in dual_line_support["per_line"].items():
        if record.get("status") != "unsupported":
            continue
        if line_name in _CROSS_TEMPLATE_LINE_NAMES:
            dual_line_penalty += 260.0
        else:
            dual_line_penalty += 90.0

    # ROUND-2 FIX 3: NET-SCALE DEPTH PRIOR. The net tape spans ~22ft
    # post-to-post (20ft court + overhang), so the projected 20ft net line
    # width should be ~(20/22) of the observed tape width; and near-baseline
    # width can only exceed net-line width by a bounded perspective factor.
    # This makes Wolverine-style near-side depth explosion score-prohibitive.
    net_scale_prior = _net_scale_depth_prior(keypoints, net_evidence=net_evidence)
    net_scale_penalty = float(net_scale_prior["penalty"])

    # ROUND-2, contrapositive of the dual-line test (needed to make the
    # correct fit WIN on Outdoor, not merely tie): every STRONG observed
    # cross line on the NEAR side of the hypothesized court (below the
    # projected near NVZ line at image center) must be EXPLAINED by the
    # hypothesis (i.e. lie close to a projected template cross line). A
    # depth-compressed fit leaves the true kitchen and near-baseline paint --
    # the two strongest lines in the frame -- dangling unexplained below its
    # tiny court; the correct fit explains both. Lines ABOVE the near NVZ
    # (far half, net structure, background bleachers/banners) are exempt:
    # net tape/mesh edges and background clutter are real but not floor
    # template lines.
    unexplained_near_penalty = 0.0
    unexplained_near_lines: list[dict[str, Any]] = []
    near_nvz_pair = (keypoints.get("near_nvz_left"), keypoints.get("near_nvz_right"))
    if all(point is not None for point in near_nvz_pair):
        near_nvz_y_at_ref = _y_on_segment_at_x(near_nvz_pair[0], near_nvz_pair[1], x_ref)
        projected_cross_y_at_ref = []
        for cross_name in ("near_nvz", "near_baseline"):
            pair_names = _ALL_TEMPLATE_LINE_ENDPOINTS[cross_name]
            pa, pb = keypoints.get(pair_names[0]), keypoints.get(pair_names[1])
            if pa is not None and pb is not None:
                projected_cross_y_at_ref.append(_y_on_segment_at_x(pa, pb, x_ref))
        strong_support_floor = max(60.0, width * 0.08)
        for group in strong_cross_lines:
            if group.support_length_px < strong_support_floor:
                continue
            seg_p1, seg_p2 = group.segment["p1"], group.segment["p2"]
            seg_min_x, seg_max_x = sorted((float(seg_p1[0]), float(seg_p2[0])))
            # Only lines that genuinely span the image-center column: a short
            # peripheral segment (scoreboard edge etc.) extrapolated to x_ref
            # is not evidence about the court's cross-line ladder.
            if not (seg_min_x - width * 0.10 <= x_ref <= seg_max_x + width * 0.10):
                continue
            from .court_line_bank import line_y_at_x as _line_y_at_x

            group_y = _line_y_at_x(group.line, x_ref, fallback=(float(seg_p1[1]) + float(seg_p2[1])) / 2.0)
            if group_y <= near_nvz_y_at_ref + 12.0:
                continue
            if any(abs(group_y - proj_y) <= 14.0 for proj_y in projected_cross_y_at_ref):
                continue
            unexplained_near_penalty += 130.0
            unexplained_near_lines.append(
                {"y_at_x_ref": round(group_y, 2), "support_length_px": round(group.support_length_px, 1)}
            )
        unexplained_near_penalty = min(390.0, unexplained_near_penalty)

    cross_items = sorted(
        (
            (name, line_y_at_x(group.line, x_ref, fallback=segment_midpoint(group.segment)[1]))
            for name, group in cross_assignment.items()
        ),
        key=lambda item: item[1],
    )
    cross_labels_sorted = [name for name, _ in cross_items]
    cross_positions = [position for _, position in cross_items]
    from .court_line_bank import line_x_at_y

    long_x_positions = sorted(
        line_x_at_y(group.line, y_ref, fallback=segment_midpoint(group.segment)[0]) for group in long_assignment.values()
    )
    joint_competition = None
    if len(cross_positions) >= 3 and len(long_x_positions) >= 2:
        joint_competition = score_joint_template_competition(
            cross_labels=cross_labels_sorted,
            cross_y_px=cross_positions,
            left_right_top_bottom_px=(long_x_positions[0], long_x_positions[-1], cross_positions[-1], cross_positions[0]),
        )

    orientation_bonus = 0.0
    if vp_families:
        cross_angle = sum(group.angle_deg for group in cross_assignment.values()) / len(cross_assignment)
        long_angle = sum(group.angle_deg for group in long_assignment.values()) / len(long_assignment)
        best_cross_delta = min(angle_diff_mod_180(cross_angle, family["angle_deg"]) for family in vp_families)
        best_long_delta = min(angle_diff_mod_180(long_angle, family["angle_deg"]) for family in vp_families)
        orientation_bonus = max(0.0, 12.0 - best_cross_delta) * 0.4 + max(0.0, 12.0 - best_long_delta) * 0.4

    surface_bonus = 0.0
    surface_polygon = (surface_evidence or {}).get("surface_polygon", {}).get("interior_polygon") if surface_evidence else None
    if surface_polygon:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np

            polygon = np.array(surface_polygon, dtype=np.float32)
            centroid = (
                (keypoints["near_left_corner"][0] + keypoints["far_right_corner"][0]) / 2.0,
                (keypoints["near_left_corner"][1] + keypoints["far_right_corner"][1]) / 2.0,
            )
            inside = cv2.pointPolygonTest(polygon, centroid, False)
            surface_bonus = 8.0 if inside >= 0 else -6.0
        except Exception:
            surface_bonus = 0.0

    assignment_support = sum(group.support_length_px for group in cross_assignment.values()) + sum(
        group.support_length_px for group in long_assignment.values()
    )
    correspondence_bonus = len(image_points) * 2.5

    # Round-2 rework: penalties/bonuses are computed over OBSERVABLE lines
    # only (fix 2). Previously off-frame template lines were counted as
    # "unsupported", which actively punished CORRECT fits whose near baseline
    # legitimately leaves the frame (IMG_1605) -- backwards.
    observable_records = [
        record
        for record in dual_line_support["per_line"].values()
        if record.get("status") in ("supported", "unsupported")
    ]
    observable_count = len(observable_records)
    supported_count = int(dual_line_support["supported_count"])
    observable_paint_ratios = [float(record.get("paint_support_ratio") or 0.0) for record in observable_records]
    pixel_mean_support_observable = (
        sum(observable_paint_ratios) / len(observable_paint_ratios) if observable_paint_ratios else 0.0
    )
    projected_pixel_penalty = max(0.0, 0.38 - pixel_mean_support_observable) * 145.0 + max(
        0, min(5, observable_count) - supported_count
    ) * 7.0
    # Distance metric over observable lines only (off-frame lines used to
    # contribute image-diagonal-sized fake distances).
    observable_line_names = {
        name for name, record in dual_line_support["per_line"].items() if record.get("status") in ("supported", "unsupported")
    }
    observable_distances = [
        float(record.get("mean_distance_px") or 0.0)
        for name, record in (distance_support.get("per_line") or {}).items()
        if name in observable_line_names and record.get("inside_image_sample_count")
    ]
    distance_mean_observable = sum(observable_distances) / len(observable_distances) if observable_distances else distance_mean
    distance_penalty = max(0.0, distance_mean_observable - 4.0) * 5.5 + max(0, min(5, observable_count) - distance_supported) * 9.0
    pixel_support_bonus = supported_count * 7.5 + pixel_mean_support_observable * 26.0
    distance_support_bonus = distance_supported * 9.0

    lower_center_overlap = _lower_center_region_overlap(keypoints, width=width, height=height)
    lower_center_bonus = lower_center_overlap * 45.0

    tennis_conflict_penalty = 0.0
    promotable_as_pickleball = template == "pickleball"
    if joint_competition and joint_competition.get("available"):
        winner = joint_competition["winner"]
        if template == "pickleball" and winner == "tennis":
            tennis_conflict_penalty = 260.0 + abs(float(joint_competition["margin"])) * 220.0
            promotable_as_pickleball = False
        if template == "tennis" and winner != "tennis":
            # tennis-tagged hypothesis whose own evidence doesn't actually look
            # like tennis is not a useful overlay-rejection witness; drop it.
            tennis_conflict_penalty = 500.0

    # Soft self-consistency penalty (see hard-reject comment above): scales
    # smoothly so a mildly noisy but otherwise strong assignment can still
    # win when nothing better exists, while a clearly mixed-court/outlier
    # assignment is reliably outranked by any more self-consistent hypothesis.
    self_consistency_penalty = self_consistency_rmse * 3.0

    score = (
        projected_pixel_penalty
        + distance_penalty
        + color_penalty
        + dual_line_penalty
        + net_scale_penalty
        + unexplained_near_penalty
        + tennis_conflict_penalty
        + self_consistency_penalty
        - pixel_support_bonus
        - distance_support_bonus
        - correspondence_bonus
        - orientation_bonus
        - surface_bonus
        - lower_center_bonus
        - min(80.0, assignment_support / 40.0)
    )
    # A logistic squash of the full real-valued cost (lower/more-negative cost
    # is better). Clipping negative costs to 0 before this step would flatten
    # every good hypothesis to the same evidence_score=1.0, destroying the
    # ranking signal `_select_hypothesis` needs among several real candidates.
    evidence_score = 1.0 / (1.0 + math.exp(min(60.0, max(-60.0, score)) / 50.0))
    supported_line_count = supported_count

    return {
        "score": float(score),
        "evidence_score": float(evidence_score),
        "template": template,
        "promotable_as_pickleball": bool(promotable_as_pickleball),
        "keypoints": keypoints,
        "supported_line_count": supported_line_count,
        "required_lines_present": supported_line_count >= 4,
        "line_assignment": {
            name: {
                "p1": group.segment["p1"],
                "p2": group.segment["p2"],
                "angle_deg": round(float(group.angle_deg), 3),
                "support_length_px": round(float(group.support_length_px), 3),
            }
            for name, group in sorted({**cross_assignment, **long_assignment}.items())
        },
        "score_components": {
            "projected_pixel_support": pixel_support,
            "projected_distance_support": distance_support,
            "line_color_consistency": color_consistency,
            "joint_template_competition": joint_competition,
            "dual_line_support": dual_line_support,
            "dual_line_penalty": round(float(dual_line_penalty), 4),
            "unexplained_near_penalty": round(float(unexplained_near_penalty), 4),
            "unexplained_near_lines": unexplained_near_lines,
            "net_scale_prior": net_scale_prior,
            "lower_center_overlap": round(float(lower_center_overlap), 4),
            "lower_center_bonus": round(float(lower_center_bonus), 4),
            "orientation_bonus": round(float(orientation_bonus), 4),
            "surface_bonus": round(float(surface_bonus), 4),
            "tennis_conflict_penalty": round(float(tennis_conflict_penalty), 4),
            "self_consistency_rmse_px": round(float(self_consistency_rmse), 3),
            "assignment_support_px": round(float(assignment_support), 3),
        },
    }


def generate_homography_hypotheses(
    image_bgr: Any,
    *,
    net_evidence: Mapping[str, Any] | None = None,
    surface_evidence: Mapping[str, Any] | None = None,
    max_hypotheses: int = 40,
    line_bank: Mapping[str, Any] | None = None,
    neural_inference: Mapping[str, Any] | None = None,
    neural_infer_provider: Callable[[Any], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Stage 2: real RANSAC-style homography hypotheses from the line bank.

    Builds a merged multi-detector line bank, groups near-collinear segments
    into candidate cross/sideline families, and searches combinations of those
    families against BOTH the pickleball and tennis regulation templates
    (Stage 3 joint competition). Returns hypotheses sorted best-first (lowest
    cost). Never raises; returns an empty list when there is not enough line
    evidence to build a 4-point correspondence set. Callers that already
    built a merged line bank for this frame (e.g. the multi-frame proposal
    path, which also needs the bank for verify metrics) can pass it via
    `line_bank` to avoid recomputing.
    """

    from .court_line_bank import (
        angle_diff_mod_180,
        build_merged_line_bank,
        cluster_line_family_directions,
        court_line_pixel_mask,
        group_candidate_lines,
        line_pixel_distance_transform,
    )

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return []
    height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if neural_inference is None:
        provider = neural_infer_provider
        if provider is None:
            from .court_model_infer import get_current_court_model_infer_provider, make_court_model_infer_provider

            provider = get_current_court_model_infer_provider()
            if provider is None:
                provider = make_court_model_infer_provider()
        if provider is not None:
            try:
                neural_inference = provider(image_bgr)
            except Exception as exc:
                _LOGGER.warning("court_unet_v2 inference failed reason=%s; using geometric-only for frame", exc)
                neural_inference = None
    neural_hypotheses = generate_neural_seed_hypotheses(
        neural_inference,
        image_size=(width, height),
    )
    try:
        if line_bank is None:
            line_bank = build_merged_line_bank(image_bgr)
        segments = line_bank["segments"]
        groups = group_candidate_lines(segments, width=float(width), height=float(height))
    except Exception:
        return _finalize_homography_hypotheses(neural_hypotheses, fusion_enabled=bool(neural_hypotheses))

    cross_pool = [
        group for group in groups if angle_diff_mod_180(group.angle_deg, 0.0) <= 12.0 and group.support_length_px >= max(28.0, width * 0.04)
    ][:10]
    long_pool = [
        group
        for group in groups
        if angle_diff_mod_180(group.angle_deg, 0.0) >= 18.0 and group.support_length_px >= max(28.0, min(width, height) * 0.045)
    ][:10]
    if len(cross_pool) < 3 or len(long_pool) < 2:
        return _finalize_homography_hypotheses(neural_hypotheses, fusion_enabled=bool(neural_hypotheses))

    try:
        vp_families = cluster_line_family_directions(segments)
    except Exception:
        vp_families = []

    # Precompute the expensive line-pixel masks/distance-transform ONCE per
    # frame. These are pure image-space evidence independent of any one
    # hypothesis, so recomputing them per-hypothesis (as an earlier version of
    # this function did) was an O(hypothesis_count) performance bug.
    pixel_mask = court_line_pixel_mask(image_bgr, dilation_px=5)
    color_mask = court_line_pixel_mask(image_bgr, dilation_px=2)
    distance_mask = court_line_pixel_mask(image_bgr, dilation_px=3)
    distance_map = line_pixel_distance_transform(distance_mask)

    x_ref = float(width) * 0.5
    y_ref = float(height) * 0.55
    hypotheses: list[dict[str, Any]] = []
    templates = (
        ("pickleball", _PICKLEBALL_CROSS_WORLD_Y_FT, _PICKLEBALL_LONG_WORLD_X_FT, _PICKLEBALL_CROSS_SETS, _PICKLEBALL_LONG_SETS),
        ("tennis", _TENNIS_CROSS_WORLD_Y_FT, _TENNIS_LONG_WORLD_X_FT, _TENNIS_CROSS_SETS, _TENNIS_LONG_SETS),
    )
    for template, world_cross, world_long, cross_sets, long_sets in templates:
        # Rank cross/sideline assignments once per label-set (they do not
        # depend on each other), instead of re-searching the sideline
        # combinations once per cross assignment.
        cross_by_labels = {
            labels: _ranked_cross_assignments(cross_pool, labels, world_cross, x_ref=x_ref, limit=10) for labels in cross_sets
        }
        long_by_labels = {
            labels: _ranked_long_assignments(long_pool, labels, world_long, y_ref=y_ref, limit=6) for labels in long_sets
        }
        for cross_labels, cross_assignments in cross_by_labels.items():
            for cross_assignment in cross_assignments:
                for long_labels, long_assignments in long_by_labels.items():
                    for long_assignment in long_assignments:
                        try:
                            hypothesis = _hypothesis_from_line_assignment(
                                template=template,
                                cross_assignment=cross_assignment,
                                long_assignment=long_assignment,
                                world_cross_ft=world_cross,
                                world_long_ft=world_long,
                                image_bgr=image_bgr,
                                image_size=(width, height),
                                pixel_mask=pixel_mask,
                                color_mask=color_mask,
                                distance_mask=distance_mask,
                                distance_map=distance_map,
                                x_ref=x_ref,
                                y_ref=y_ref,
                                vp_families=vp_families,
                                surface_evidence=surface_evidence,
                                net_evidence=net_evidence,
                                segments=segments,
                                strong_cross_lines=cross_pool,
                            )
                        except Exception:
                            hypothesis = None
                        if hypothesis is not None:
                            hypotheses.append(hypothesis)

    fusion_enabled = bool(neural_hypotheses)
    hypotheses.sort(key=lambda item: float(item["score"]))
    if fusion_enabled:
        for item in hypotheses:
            item.setdefault("source_tag", "geometric")
            item.setdefault("model_confidence", None)
        combined = hypotheses + neural_hypotheses
        combined.sort(key=lambda item: float(item["score"]))
        shortlist = combined[:max_hypotheses]
        for neural_item in neural_hypotheses:
            if all(item is not neural_item for item in shortlist):
                shortlist.append(neural_item)
        shortlist.sort(key=lambda item: float(item["score"]))
    else:
        shortlist = hypotheses[:max_hypotheses]
    return _finalize_homography_hypotheses(shortlist, fusion_enabled=fusion_enabled)


def _finalize_homography_hypotheses(
    shortlist: list[dict[str, Any]],
    *,
    fusion_enabled: bool,
) -> list[dict[str, Any]]:
    for index, item in enumerate(shortlist):
        item["hypothesis_id"] = f"homography_hypothesis_{index:04d}"
        if item.get("source_tag") == "neural_seeded":
            item["source"] = "neural_seeded_court_unet_v2"
        else:
            item["source"] = f"real_line_bank_{item['template']}_assignment"
            if fusion_enabled:
                item.setdefault("source_tag", "geometric")
                item.setdefault("model_confidence", None)
        item["floor_correspondence_names"] = list(FLOOR_KEYPOINT_NAMES)
        item["projected_keypoints"] = {name: list(xy) for name, xy in item["keypoints"].items()}
        item["line_support"] = {
            "required_lines_present": bool(item["required_lines_present"]),
            "semantic_line_count": int(item["supported_line_count"]),
        }
        item["regulation_ratio_errors"] = {"source": "homography_projection", "max_abs_error": 0.0}
        item["score_components"]["evidence_score"] = round(float(item["evidence_score"]), 4)
        item["promotion_allowed"] = False
    return shortlist


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
