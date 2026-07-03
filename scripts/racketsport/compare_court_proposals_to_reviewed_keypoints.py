#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration_metric15 import (  # noqa: E402
    aggregate_reviewed_keypoints_native_px,
    load_reviewed_court_keypoints_15pt,
)
from threed.racketsport.court_keypoint_labels import (  # noqa: E402
    PARTIAL_LABEL_ARTIFACT_TYPE,
    VISIBLE,
    PartialCourtKeypoints,
    load_partial_court_keypoints,
)
from threed.racketsport.court_keypoint_net import COURT_CORNER_LABEL_TO_KEYPOINT  # noqa: E402
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES  # noqa: E402

NET_KEYPOINT_NAMES = {"net_left_sideline", "net_center", "net_right_sideline"}
CORNER_KEYPOINT_NAMES = set(COURT_CORNER_LABEL_TO_KEYPOINT.values())
PLANAR_KEYPOINT_NAMES = set(PICKLEBALL_COURT_KEYPOINT_NAMES) - NET_KEYPOINT_NAMES
NVZ_KEYPOINT_NAMES = {
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
}
MAX_ACCEPTABLE_VISIBLE_MEDIAN_PX = 30.0
MAX_ACCEPTABLE_FLOOR_MEDIAN_PX = 30.0


@dataclass(frozen=True)
class ReviewedPointsNative:
    clip: str
    frame_count: int
    artifact_type: str
    points: dict[str, tuple[float, float]]
    stdev_px: dict[str, dict[str, float]]
    native_size: tuple[float, float]
    missing_reviewed_keypoints: list[str]


def compare_proposal_to_reviewed_keypoints(
    *,
    reviewed_keypoints_path: str | Path,
    proposal_path: str | Path,
) -> dict[str, Any]:
    reviewed_keypoints_path = Path(reviewed_keypoints_path)
    proposal_path = Path(proposal_path)
    reviewed = _load_reviewed_points_native_px(reviewed_keypoints_path)
    reviewed_points = reviewed.points
    reviewed_stdev_px = reviewed.stdev_px
    native_size = reviewed.native_size
    proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal_points, proposal_size = _proposal_keypoints_native_px(proposal_payload, native_size)

    per_keypoint: dict[str, dict[str, Any]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        expected = reviewed_points.get(name)
        predicted = proposal_points.get(name)
        if expected is None or predicted is None:
            continue
        err = math.dist(expected, predicted)
        per_keypoint[name] = {
            "error_px": float(err),
            "prediction_xy": [float(predicted[0]), float(predicted[1])],
            "reviewed_xy": [float(expected[0]), float(expected[1])],
            "reviewed_frame_stdev_px": reviewed_stdev_px.get(name, {}),
        }

    errors = {name: item["error_px"] for name, item in per_keypoint.items()}
    groups = {
        "all": _summarize_errors(errors, set(PICKLEBALL_COURT_KEYPOINT_NAMES)),
        "corners_4": _summarize_errors(errors, CORNER_KEYPOINT_NAMES),
        "planar_12": _summarize_errors(errors, PLANAR_KEYPOINT_NAMES),
        "net_top_3": _summarize_errors(errors, NET_KEYPOINT_NAMES),
        "all_visible": _summarize_errors(errors, set(reviewed_points)),
        "floor_visible": _summarize_errors(errors, set(reviewed_points) & PLANAR_KEYPOINT_NAMES),
        "visible_corners": _summarize_errors(errors, set(reviewed_points) & CORNER_KEYPOINT_NAMES),
        "net_top_visible": _summarize_errors(errors, set(reviewed_points) & NET_KEYPOINT_NAMES),
        "nvz_visible": _summarize_errors(errors, set(reviewed_points) & NVZ_KEYPOINT_NAMES),
    }
    rejection_reasons = _rejection_reasons(groups, proposal_payload)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_proposal_review_comparison",
        "reviewed_keypoints": str(reviewed_keypoints_path),
        "proposal": str(proposal_path),
        "clip": reviewed.clip,
        "reviewed_artifact_type": reviewed.artifact_type,
        "reviewed_frame_count": reviewed.frame_count,
        "native_image_size": [float(native_size[0]), float(native_size[1])],
        "proposal_image_size": list(proposal_size) if proposal_size is not None else None,
        "matched_keypoint_count": len(per_keypoint),
        "missing_reviewed_keypoints": reviewed.missing_reviewed_keypoints,
        "missing_proposal_keypoints": [
            name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in proposal_points
        ],
        "solver_confidence": proposal_payload.get("solver_confidence"),
        "needs_user_input": proposal_payload.get("needs_user_input"),
        "needs_user_confirmation": proposal_payload.get("needs_user_confirmation"),
        "verdict": "mandatory_user_confirmation_only" if rejection_reasons else "visible_keypoints_within_gate",
        "rejection_reasons": rejection_reasons,
        "groups": groups,
        "per_keypoint": per_keypoint,
    }


def _load_reviewed_points_native_px(path: Path) -> ReviewedPointsNative:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") == PARTIAL_LABEL_ARTIFACT_TYPE:
        partial = load_partial_court_keypoints(path)
        points, stdev_px, native_size = _aggregate_partial_keypoints_native_px(partial)
        return ReviewedPointsNative(
            clip=partial.clip,
            frame_count=len(partial.frames),
            artifact_type=PARTIAL_LABEL_ARTIFACT_TYPE,
            points=points,
            stdev_px=stdev_px,
            native_size=native_size,
            missing_reviewed_keypoints=[
                name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in points
            ],
        )

    reviewed = load_reviewed_court_keypoints_15pt(path)
    points, stdev_px, native_size = aggregate_reviewed_keypoints_native_px(reviewed)
    return ReviewedPointsNative(
        clip=reviewed.clip,
        frame_count=len(reviewed.frames),
        artifact_type=str(payload.get("artifact_type") or "racketsport_court_keypoint_labels"),
        points=points,
        stdev_px=stdev_px,
        native_size=native_size,
        missing_reviewed_keypoints=[
            name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in points
        ],
    )


def _aggregate_partial_keypoints_native_px(
    partial: PartialCourtKeypoints,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, float]], tuple[float, float]]:
    label_w, label_h = partial.label_coordinate_space
    if label_w <= 0.0 or label_h <= 0.0:
        raise ValueError("label_coordinate_space must be positive")
    native_size = tuple(float(v) for v in partial.source_resolution)
    scale_x = native_size[0] / label_w
    scale_y = native_size[1] / label_h

    aggregated: dict[str, tuple[float, float]] = {}
    stdev_by_name: dict[str, dict[str, float]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        xs: list[float] = []
        ys: list[float] = []
        for frame in partial.frames:
            if frame.visibility_by_keypoint.get(name) != VISIBLE:
                continue
            point = frame.keypoints.get(name)
            if point is None:
                continue
            xs.append(float(point[0]) * scale_x)
            ys.append(float(point[1]) * scale_y)
        if not xs:
            continue
        aggregated[name] = (_median(xs), _median(ys))
        stdev_by_name[name] = {"x_stdev_px": _stdev(xs), "y_stdev_px": _stdev(ys)}
    if not aggregated:
        raise ValueError(f"{partial.clip}: no visible partial court keypoints")
    return aggregated, stdev_by_name, native_size


def _proposal_keypoints_native_px(
    payload: dict[str, Any],
    native_size: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], tuple[float, float] | None]:
    proposal_size = _proposal_image_size(payload)
    scale_x = 1.0
    scale_y = 1.0
    if proposal_size is not None:
        width, height = proposal_size
        if width > 0 and height > 0:
            scale_x = native_size[0] / width
            scale_y = native_size[1] / height

    keypoints: dict[str, tuple[float, float]] = {}
    for name, item in (payload.get("keypoints") or {}).items():
        xy = _xy_from_payload_item(item)
        if xy is not None:
            keypoints[str(name)] = (float(xy[0]) * scale_x, float(xy[1]) * scale_y)

    for corner_label, point_name in COURT_CORNER_LABEL_TO_KEYPOINT.items():
        if point_name in keypoints:
            continue
        item = (payload.get("corners") or {}).get(corner_label)
        xy = _xy_from_payload_item(item)
        if xy is not None:
            keypoints[point_name] = (float(xy[0]) * scale_x, float(xy[1]) * scale_y)
    return keypoints, proposal_size


def _proposal_image_size(payload: dict[str, Any]) -> tuple[float, float] | None:
    source = payload.get("source")
    if isinstance(source, dict):
        image_size = source.get("image_size") or source.get("source_size")
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            return (float(image_size[0]), float(image_size[1]))
    for key in ("image_size", "source_size"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    return None


def _xy_from_payload_item(item: Any) -> tuple[float, float] | None:
    if isinstance(item, dict):
        xy = item.get("xy") or item.get("image_xy") or item.get("point")
    else:
        xy = item
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        return None
    try:
        return (float(xy[0]), float(xy[1]))
    except (TypeError, ValueError):
        return None


def _summarize_errors(errors: dict[str, float], names: set[str]) -> dict[str, Any]:
    values = [float(errors[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in names and name in errors]
    if not values:
        return {"count": 0, "median_px": None, "p95_px": None, "max_px": None, "mean_px": None}
    return {
        "count": len(values),
        "median_px": float(statistics.median(values)),
        "p95_px": float(_percentile(values, 95.0)),
        "max_px": float(max(values)),
        "mean_px": float(sum(values) / len(values)),
    }


def _rejection_reasons(groups: dict[str, dict[str, Any]], proposal_payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    all_visible = groups["all_visible"]
    floor_visible = groups["floor_visible"]
    if all_visible["count"] <= 0:
        reasons.append("no_visible_keypoint_matches")
    elif all_visible["median_px"] is not None and all_visible["median_px"] > MAX_ACCEPTABLE_VISIBLE_MEDIAN_PX:
        reasons.append("visible_keypoint_residual_too_high")
    if floor_visible["median_px"] is not None and floor_visible["median_px"] > MAX_ACCEPTABLE_FLOOR_MEDIAN_PX:
        reasons.append("floor_visible_residual_too_high")
    if proposal_payload.get("needs_user_confirmation"):
        reasons.append("proposal_already_requires_user_confirmation")
    for cap in proposal_payload.get("confidence_caps") or []:
        if isinstance(cap, dict) and cap.get("reason") == "seed_only_hypothesis_not_globally_verified":
            reasons.append("seed_only_hypothesis_not_globally_verified")
            break
    return reasons


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare automatic court proposals against reviewed 15-point court labels.")
    parser.add_argument("--reviewed-keypoints", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        summary = compare_proposal_to_reviewed_keypoints(
            reviewed_keypoints_path=args.reviewed_keypoints,
            proposal_path=args.proposal,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
