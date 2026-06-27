"""Calibration overlay artifact generation."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from .court_calibration import project_planar_points
from .court_templates import COORDINATE_FRAME, get_court_template
from .net_plane import build_net_plane, project_net_plane
from .schemas import CourtCalibration, NetPlane


OverlayPayload = dict[str, Any]


def load_calibration_artifact(path: str | Path) -> CourtCalibration:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"missing calibration artifact: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return CourtCalibration.model_validate(payload)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid calibration artifact: {artifact_path}: {exc}") from exc


def load_net_plane_artifact(path: str | Path) -> NetPlane:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"missing net plane artifact: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return NetPlane.model_validate(payload)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid net plane artifact: {artifact_path}: {exc}") from exc


def build_calibration_overlay(calibration: CourtCalibration, *, net_plane: NetPlane | None = None) -> OverlayPayload:
    """Project regulation court and net geometry through a solved calibration."""

    net = net_plane or build_net_plane(calibration.sport)
    _validate_net_plane_matches_sport(calibration, net)
    template = get_court_template(calibration.sport)

    court_lines = []
    for line_id, endpoints in template.line_segments_m.items():
        world = [[float(value) for value in point] for point in endpoints]
        image = project_planar_points(calibration.homography, world)
        court_lines.append(
            {
                "id": line_id,
                "world": _round_points(world),
                "image": _round_points(image),
            }
        )

    net_points = {key: _round_point(value) for key, value in project_net_plane(calibration, net).items()}
    all_image_points = [point for line in court_lines for point in line["image"]] + list(net_points.values())
    view_box = _view_box_for_points(all_image_points)

    return {
        "schema_version": 1,
        "artifact": "calibration_overlay",
        "sport": calibration.sport,
        "coordinate_frame": COORDINATE_FRAME,
        "view_box": view_box,
        "court_lines": court_lines,
        "net_points": net_points,
        "net_segments": [
            {"id": "net_top_left", "image": [net_points["left_post"], net_points["center"]]},
            {"id": "net_top_right", "image": [net_points["center"], net_points["right_post"]]},
        ],
        "summary": {
            "court_line_count": len(court_lines),
            "net_point_count": len(net_points),
            "reprojection_median_px": float(calibration.reprojection_error_px.median),
            "reprojection_p95_px": float(calibration.reprojection_error_px.p95),
        },
    }


def overlay_to_svg(overlay: OverlayPayload) -> str:
    view_box = overlay["view_box"]
    width = max(1.0, float(view_box["width"]))
    height = max(1.0, float(view_box["height"]))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(view_box["min_x"])} '
            f'{_fmt(view_box["min_y"])} {_fmt(width)} {_fmt(height)}" '
            f'width="{_fmt(width)}" height="{_fmt(height)}" role="img" '
            f'aria-label="{html.escape(str(overlay["sport"]))} calibration overlay">'
        ),
        '<g id="court-lines" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round">',
    ]
    for court_line in overlay["court_lines"]:
        start, end = court_line["image"]
        line_id = html.escape(str(court_line["id"]), quote=True)
        lines.append(
            f'<line data-line-id="{line_id}" x1="{_fmt(start[0])}" y1="{_fmt(start[1])}" '
            f'x2="{_fmt(end[0])}" y2="{_fmt(end[1])}" />'
        )
    lines.extend(
        [
            "</g>",
            '<g id="net" fill="none" stroke="#f97316" stroke-width="3" stroke-linecap="round">',
        ]
    )
    for segment in overlay["net_segments"]:
        start, end = segment["image"]
        segment_id = html.escape(str(segment["id"]), quote=True)
        lines.append(
            f'<line data-net-segment-id="{segment_id}" x1="{_fmt(start[0])}" y1="{_fmt(start[1])}" '
            f'x2="{_fmt(end[0])}" y2="{_fmt(end[1])}" />'
        )
    lines.extend(["</g>", '<g id="net-points" fill="#f97316" stroke="#111827" stroke-width="1">'])
    for point_id, point in overlay["net_points"].items():
        safe_id = html.escape(str(point_id), quote=True)
        lines.append(f'<circle data-net-point-id="{safe_id}" cx="{_fmt(point[0])}" cy="{_fmt(point[1])}" r="4" />')
    lines.extend(["</g>", "</svg>", ""])
    return "\n".join(lines)


def write_overlay_artifacts(
    out_svg: str | Path,
    calibration: CourtCalibration,
    *,
    net_plane: NetPlane | None = None,
    summary_out: str | Path | None = None,
) -> OverlayPayload:
    overlay = build_calibration_overlay(calibration, net_plane=net_plane)
    svg_path = Path(out_svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(overlay_to_svg(overlay), encoding="utf-8")

    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return overlay


def _validate_net_plane_matches_sport(calibration: CourtCalibration, net_plane: NetPlane) -> None:
    expected = build_net_plane(calibration.sport)
    expected_points = [*expected.endpoints, [0.0, 0.0, expected.center_height_in * 0.0254]]
    actual_points = [*net_plane.endpoints, [0.0, 0.0, net_plane.center_height_in * 0.0254]]
    for actual, expected_point in zip(actual_points, expected_points, strict=True):
        if any(not math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-9) for a, e in zip(actual, expected_point, strict=True)):
            raise ValueError("net plane endpoints do not match calibration sport")


def _view_box_for_points(points: list[list[float]], *, pad_px: float = 24.0) -> dict[str, float]:
    if not points:
        raise ValueError("overlay requires at least one projected point")
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if math.isclose(min_x, max_x):
        min_x -= 0.5
        max_x += 0.5
    if math.isclose(min_y, max_y):
        min_y -= 0.5
        max_y += 0.5
    return {
        "min_x": _round_float(min_x - pad_px),
        "min_y": _round_float(min_y - pad_px),
        "width": _round_float((max_x - min_x) + 2.0 * pad_px),
        "height": _round_float((max_y - min_y) + 2.0 * pad_px),
    }


def _round_points(points: list[list[float]]) -> list[list[float]]:
    return [_round_point(point) for point in points]


def _round_point(point: list[float]) -> list[float]:
    return [_round_float(value) for value in point]


def _round_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if math.isclose(rounded, 0.0, abs_tol=1e-12) else rounded


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")
