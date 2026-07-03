from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .court_calibration import homography_from_planar_points, project_planar_points
from .court_keypoint_labels import VISIBLE, PartialCourtKeypoints, load_partial_court_keypoints
from .net_anchor_court import WORLD_XY_BY_NAME
from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES

NET_TOP_KEYPOINT_NAMES = {"net_left_sideline", "net_center", "net_right_sideline"}
FLOOR_KEYPOINT_NAMES = tuple(
    name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in NET_TOP_KEYPOINT_NAMES
)
DEFAULT_VERIFIED_MEDIAN_RESIDUAL_PX = 8.0
DEFAULT_VERIFIED_MAX_RESIDUAL_PX = 15.0


@dataclass(frozen=True)
class VisibleFloorHomographyFit:
    homography: list[list[float]]
    used_keypoints: list[str]
    excluded_keypoints: list[str]
    native_image_size: list[float]
    residual_summary_px: dict[str, float | int | None]
    per_keypoint_residual_px: dict[str, float]
    inferred_keypoints: dict[str, dict[str, Any]]
    fit_verified: bool


def fit_visible_floor_homography(
    partial: str | Path | PartialCourtKeypoints,
    *,
    verified_median_residual_px: float = DEFAULT_VERIFIED_MEDIAN_RESIDUAL_PX,
    verified_max_residual_px: float = DEFAULT_VERIFIED_MAX_RESIDUAL_PX,
) -> VisibleFloorHomographyFit:
    """Fit a court-floor homography from visible partial labels only.

    Top-net keypoints are useful validation targets, but they are 3D points above
    the court plane. This helper intentionally excludes them from the planar fit
    and only infers missing floor keypoints.
    """

    labels = load_partial_court_keypoints(partial) if isinstance(partial, (str, Path)) else partial
    visible_native = _visible_partial_keypoints_native_px(labels)
    used = [name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in visible_native and name in FLOOR_KEYPOINT_NAMES]
    excluded = [name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in visible_native and name in NET_TOP_KEYPOINT_NAMES]
    if len(used) < 4:
        raise ValueError("visible floor homography requires at least 4 visible floor keypoints")

    world = [[WORLD_XY_BY_NAME[name][0], WORLD_XY_BY_NAME[name][1]] for name in used]
    image = [visible_native[name] for name in used]
    homography = homography_from_planar_points(world, image)
    projected_used = project_planar_points(homography, world)
    residuals = {
        name: float(math.dist(image_xy, projected_xy))
        for name, image_xy, projected_xy in zip(used, image, projected_used, strict=True)
    }

    missing_floor = [name for name in FLOOR_KEYPOINT_NAMES if name not in visible_native]
    inferred_projected = project_planar_points(
        homography,
        [[WORLD_XY_BY_NAME[name][0], WORLD_XY_BY_NAME[name][1]] for name in missing_floor],
    )
    inferred = {
        name: {
            "xy": [float(xy[0]), float(xy[1])],
            "source": "visible_floor_homography",
        }
        for name, xy in zip(missing_floor, inferred_projected, strict=True)
    }
    summary = _summarize_residuals(residuals)
    median_px = summary["median_px"]
    max_px = summary["max_px"]
    fit_verified = (
        median_px is not None
        and max_px is not None
        and float(median_px) <= verified_median_residual_px
        and float(max_px) <= verified_max_residual_px
    )
    return VisibleFloorHomographyFit(
        homography=[[float(value) for value in row] for row in homography],
        used_keypoints=used,
        excluded_keypoints=excluded,
        native_image_size=[float(labels.source_resolution[0]), float(labels.source_resolution[1])],
        residual_summary_px=summary,
        per_keypoint_residual_px=residuals,
        inferred_keypoints=inferred,
        fit_verified=fit_verified,
    )


def _visible_partial_keypoints_native_px(labels: PartialCourtKeypoints) -> dict[str, list[float]]:
    label_w, label_h = labels.label_coordinate_space
    if label_w <= 0.0 or label_h <= 0.0:
        raise ValueError("label_coordinate_space must be positive")
    scale_x = labels.source_resolution[0] / label_w
    scale_y = labels.source_resolution[1] / label_h

    points_by_name: dict[str, list[tuple[float, float]]] = {}
    for frame in labels.frames:
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
            if frame.visibility_by_keypoint.get(name) != VISIBLE:
                continue
            point = frame.keypoints.get(name)
            if point is None:
                continue
            points_by_name.setdefault(name, []).append((float(point[0]) * scale_x, float(point[1]) * scale_y))

    return {
        name: [float(statistics.median([xy[0] for xy in values])), float(statistics.median([xy[1] for xy in values]))]
        for name, values in points_by_name.items()
    }


def _summarize_residuals(residuals: dict[str, float]) -> dict[str, float | int | None]:
    values = [float(residuals[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in residuals]
    if not values:
        return {"count": 0, "median_px": None, "p95_px": None, "max_px": None, "mean_px": None}
    return {
        "count": len(values),
        "median_px": float(statistics.median(values)),
        "p95_px": float(_percentile(values, 95.0)),
        "max_px": float(max(values)),
        "mean_px": float(sum(values) / len(values)),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight
