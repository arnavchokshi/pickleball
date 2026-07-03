"""Line-segmentation primitives for Round-3 court calibration.

The Round-3 representation predicts court-line masks, fits sub-pixel line equations, and
derives the 15 canonical keypoints from line intersections. The three net keypoints use the
regulation top-net convention in the taxonomy; the net mask is therefore modeled as its own
image line and is excluded from ground-plane homography assumptions elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


ROUND3_MIN_INPUT_WIDTH = 640


@dataclass(frozen=True)
class CourtLineFamily:
    name: str
    keypoint_names: tuple[str, ...]
    segment_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LineEquation:
    a: float
    b: float
    c: float
    support_px: int
    rms_px: float


COURT_LINE_FAMILIES: tuple[CourtLineFamily, ...] = (
    CourtLineFamily(
        "left_sideline",
        ("near_left_corner", "near_nvz_left", "far_nvz_left", "far_left_corner"),
        (("near_left_corner", "far_left_corner"),),
    ),
    CourtLineFamily(
        "centerline",
        ("near_baseline_center", "near_nvz_center", "far_nvz_center", "far_baseline_center"),
        (("near_baseline_center", "near_nvz_center"), ("far_nvz_center", "far_baseline_center")),
    ),
    CourtLineFamily(
        "right_sideline",
        ("near_right_corner", "near_nvz_right", "far_nvz_right", "far_right_corner"),
        (("near_right_corner", "far_right_corner"),),
    ),
    CourtLineFamily(
        "near_baseline",
        ("near_left_corner", "near_baseline_center", "near_right_corner"),
        (("near_left_corner", "near_right_corner"),),
    ),
    CourtLineFamily(
        "near_nvz",
        ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
        (("near_nvz_left", "near_nvz_right"),),
    ),
    CourtLineFamily(
        "net",
        ("net_left_sideline", "net_center", "net_right_sideline"),
        (("net_left_sideline", "net_right_sideline"),),
    ),
    CourtLineFamily(
        "far_nvz",
        ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
        (("far_nvz_left", "far_nvz_right"),),
    ),
    CourtLineFamily(
        "far_baseline",
        ("far_left_corner", "far_baseline_center", "far_right_corner"),
        (("far_left_corner", "far_right_corner"),),
    ),
)

COURT_LINE_FAMILY_BY_NAME: dict[str, CourtLineFamily] = {family.name: family for family in COURT_LINE_FAMILIES}


def validate_round3_input_resolution(
    image_width: int,
    image_height: int,
    *,
    patch_size: int | None = None,
) -> dict[str, Any]:
    if isinstance(image_width, bool) or isinstance(image_height, bool):
        raise ValueError("image dimensions must be positive integers")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive integers")
    if patch_size is not None:
        if isinstance(patch_size, bool) or patch_size < ROUND3_MIN_INPUT_WIDTH:
            raise ValueError(f"patch_size must be at least {ROUND3_MIN_INPUT_WIDTH}px wide")
        mode = "patch_based"
    else:
        if image_width < ROUND3_MIN_INPUT_WIDTH:
            raise ValueError(
                f"Round-3 line segmentation input must be at least {ROUND3_MIN_INPUT_WIDTH}px wide "
                "or use patch-based inference so a 5px source-space gate is representable"
            )
        mode = "full_frame"
    return {
        "mode": mode,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "patch_size": patch_size,
        "min_width": ROUND3_MIN_INPUT_WIDTH,
    }


def line_mask_targets_for_keypoints(
    keypoints: Mapping[str, Sequence[float]],
    *,
    width: int,
    height: int,
    line_width: int = 3,
) -> dict[str, Any]:
    import numpy as np

    if line_width <= 0:
        raise ValueError("line_width must be positive")
    yy, xx = np.mgrid[0:height, 0:width]
    masks: dict[str, Any] = {}
    for family in COURT_LINE_FAMILIES:
        mask = np.zeros((height, width), dtype=np.float32)
        for start_name, end_name in family.segment_pairs:
            start = _xy(keypoints[start_name], start_name)
            end = _xy(keypoints[end_name], end_name)
            distance = _distance_to_segment(xx.astype(np.float32), yy.astype(np.float32), start, end)
            mask = np.maximum(mask, (distance <= float(line_width)).astype(np.float32))
        masks[family.name] = mask
    return masks


def fit_court_lines_from_masks(
    masks: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    min_support_px: int = 8,
) -> dict[str, LineEquation]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    lines: dict[str, LineEquation] = {}
    for family in COURT_LINE_FAMILIES:
        if family.name not in masks:
            raise ValueError(f"missing line mask for {family.name}")
        lines[family.name] = fit_line_equation_from_mask(
            masks[family.name],
            threshold=threshold,
            min_support_px=min_support_px,
        )
    return lines


def fit_line_equation_from_mask(mask: Any, *, threshold: float = 0.5, min_support_px: int = 8) -> LineEquation:
    import numpy as np

    arr = np.asarray(mask, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("line mask must be a 2D array")
    ys, xs = np.nonzero(arr >= threshold)
    if xs.size < min_support_px:
        raise ValueError(f"line mask has too little support: {xs.size} px")
    weights = arr[ys, xs]
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = np.ones_like(weights)
        weight_sum = float(weights.sum())
    x_mean = float((weights * xs).sum() / weight_sum)
    y_mean = float((weights * ys).sum() / weight_sum)
    centered = np.stack([xs.astype(np.float64) - x_mean, ys.astype(np.float64) - y_mean], axis=1)
    cov = (centered * weights[:, None]).T @ centered / weight_sum
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    a, b = float(normal[0]), float(normal[1])
    norm = math.hypot(a, b)
    if math.isclose(norm, 0.0):
        raise ValueError("degenerate line mask")
    a, b = a / norm, b / norm
    c = -(a * x_mean + b * y_mean)
    residuals = a * xs.astype(np.float64) + b * ys.astype(np.float64) + c
    rms = float(math.sqrt(float((weights * residuals * residuals).sum() / weight_sum)))
    return LineEquation(a=a, b=b, c=c, support_px=int(xs.size), rms_px=rms)


def intersect_court_keypoints_from_lines(lines: Mapping[str, LineEquation]) -> dict[str, list[float]]:
    required = {family.name for family in COURT_LINE_FAMILIES}
    missing = sorted(required - set(lines))
    if missing:
        raise ValueError(f"missing fitted court lines: {', '.join(missing)}")
    return {
        "near_left_corner": _intersection(lines["near_baseline"], lines["left_sideline"]),
        "near_baseline_center": _intersection(lines["near_baseline"], lines["centerline"]),
        "near_right_corner": _intersection(lines["near_baseline"], lines["right_sideline"]),
        "far_right_corner": _intersection(lines["far_baseline"], lines["right_sideline"]),
        "far_baseline_center": _intersection(lines["far_baseline"], lines["centerline"]),
        "far_left_corner": _intersection(lines["far_baseline"], lines["left_sideline"]),
        "near_nvz_left": _intersection(lines["near_nvz"], lines["left_sideline"]),
        "near_nvz_center": _intersection(lines["near_nvz"], lines["centerline"]),
        "near_nvz_right": _intersection(lines["near_nvz"], lines["right_sideline"]),
        "net_left_sideline": _intersection(lines["net"], lines["left_sideline"]),
        "net_center": _intersection(lines["net"], lines["centerline"]),
        "net_right_sideline": _intersection(lines["net"], lines["right_sideline"]),
        "far_nvz_left": _intersection(lines["far_nvz"], lines["left_sideline"]),
        "far_nvz_center": _intersection(lines["far_nvz"], lines["centerline"]),
        "far_nvz_right": _intersection(lines["far_nvz"], lines["right_sideline"]),
    }


def _intersection(a: LineEquation, b: LineEquation) -> list[float]:
    det = a.a * b.b - b.a * a.b
    if math.isclose(det, 0.0, abs_tol=1e-9):
        raise ValueError("parallel court lines cannot be intersected")
    x = (a.b * b.c - b.b * a.c) / det
    y = (b.a * a.c - a.a * b.c) / det
    return [float(x), float(y)]


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be a 2D point")
    return float(value[0]), float(value[1])


def _distance_to_segment(xx: Any, yy: Any, start: tuple[float, float], end: tuple[float, float]) -> Any:
    import numpy as np

    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if math.isclose(denom, 0.0):
        return np.hypot(xx - sx, yy - sy)
    t = np.clip(((xx - sx) * dx + (yy - sy) * dy) / denom, 0.0, 1.0)
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return np.hypot(xx - closest_x, yy - closest_y)


__all__ = [
    "COURT_LINE_FAMILIES",
    "COURT_LINE_FAMILY_BY_NAME",
    "LineEquation",
    "ROUND3_MIN_INPUT_WIDTH",
    "fit_court_lines_from_masks",
    "fit_line_equation_from_mask",
    "intersect_court_keypoints_from_lines",
    "line_mask_targets_for_keypoints",
    "validate_round3_input_resolution",
]
