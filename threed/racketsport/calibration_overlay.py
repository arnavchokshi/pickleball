"""Calibration overlay artifact generation."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from .court_calibration import project_planar_points
from .court_auto_evidence import calibration_for_image_size
from .court_templates import COORDINATE_FRAME, get_court_template
from .net_plane import build_net_plane, project_net_plane
from .schemas import CourtCalibration, NetPlane


OverlayPayload = dict[str, Any]


COURT_LINE_COLOR = (34, 197, 94)
NET_LINE_COLOR = (0, 140, 255)
NET_POINT_COLOR = (0, 140, 255)
TEXT_COLOR = (255, 255, 255)
MAX_TRUSTED_NET_TOP_ANGLE_DELTA_DEG = 6.0
MAX_TRUSTED_NET_TOP_LENGTH_RATIO = 4.0
UNTRUSTED_NET_TOP_INTRINSIC_SOURCES = {"estimated_from_review_frame"}


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


def build_calibration_overlay(
    calibration: CourtCalibration,
    *,
    net_plane: NetPlane | None = None,
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
) -> OverlayPayload:
    """Project regulation court and net geometry through a solved calibration."""

    net = net_plane or build_net_plane(
        calibration.sport,
        post_height_in=net_post_height_in,
        center_height_in=net_center_height_in,
    )
    _validate_net_plane_matches_sport(
        calibration,
        net,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
    )
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

    projected_net_points = {key: _round_point(value) for key, value in project_net_plane(calibration, net).items()}
    ground_net = next((line for line in court_lines if line["id"] == "net"), None)
    net_trust = _net_top_projection_trust(
        ground_net["image"] if ground_net else [],
        projected_net_points,
        intrinsics_source=calibration.intrinsics.source,
    )
    net_points = projected_net_points if net_trust["trusted"] else {}
    all_image_points = [point for line in court_lines for point in line["image"]] + list(net_points.values())
    view_box = _view_box_for_points(all_image_points)

    net_segments = (
        [
            {"id": "net_top_left", "image": [net_points["left_post"], net_points["center"]]},
            {"id": "net_top_right", "image": [net_points["center"], net_points["right_post"]]},
        ]
        if net_points
        else []
    )

    return {
        "schema_version": 1,
        "artifact": "calibration_overlay",
        "sport": calibration.sport,
        "coordinate_frame": COORDINATE_FRAME,
        "view_box": view_box,
        "court_lines": court_lines,
        "net_points": net_points,
        "net_segments": net_segments,
        "summary": {
            "court_line_count": len(court_lines),
            "net_point_count": len(net_points),
            "reprojection_median_px": float(calibration.reprojection_error_px.median),
            "reprojection_p95_px": float(calibration.reprojection_error_px.p95),
            "net_top_projection_status": net_trust["status"],
            "net_top_angle_delta_deg": net_trust["angle_delta_deg"],
            "net_top_length_ratio": net_trust["length_ratio"],
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
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
    summary_out: str | Path | None = None,
) -> OverlayPayload:
    overlay = build_calibration_overlay(
        calibration,
        net_plane=net_plane,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
    )
    svg_path = Path(out_svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(overlay_to_svg(overlay), encoding="utf-8")

    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return overlay


def render_calibration_image_overlay(
    *,
    image_path: str | Path,
    out_path: str | Path,
    calibration: CourtCalibration,
    net_plane: NetPlane | None = None,
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Draw projected calibration geometry onto a real review frame image."""

    cv2 = cv2_module or _cv2()
    image_path = Path(image_path)
    out_path = Path(out_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot open image frame: {image_path}")

    height, width = frame.shape[:2]
    calibration = calibration_for_image_size(calibration, width=int(width), height=int(height))
    overlay = build_calibration_overlay(
        calibration,
        net_plane=net_plane,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
    )
    _draw_image_overlay(cv2, frame, overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"cannot write calibration overlay image: {out_path}")

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_calibration_frame_overlay",
        "status": "rendered",
        "image_path": str(image_path),
        "out_path": str(out_path),
        "sport": overlay["sport"],
        "court_line_count": len(overlay["court_lines"]),
        "court_line_ids": [line["id"] for line in overlay["court_lines"]],
        "net_point_count": len(overlay["net_points"]),
        "reprojection_median_px": overlay["summary"]["reprojection_median_px"],
        "reprojection_p95_px": overlay["summary"]["reprojection_p95_px"],
        "qualitative_status": "corrected_unverified_visual_review_required",
    }


def render_calibration_run_overlays(
    *,
    run_root: str | Path,
    frames_root: str | Path,
    clips: list[str] | None = None,
    max_video_frames: int | None = None,
    fps: float = 10.0,
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
    cv2_module: Any | None = None,
    write_index: bool = True,
    write_markdown: bool = True,
) -> dict[str, Any]:
    """Render review-frame and frame-pack calibration overlays for a run root."""

    cv2 = cv2_module or _cv2()
    run_root = Path(run_root)
    frames_root = Path(frames_root)
    reviewed_frames = _reviewed_frames_by_clip(run_root)
    clip_names = list(clips or reviewed_frames or _artifact_clip_names(run_root))
    clip_summaries = [
        _render_clip_calibration_overlays(
            cv2=cv2,
            run_root=run_root,
            frames_root=frames_root,
            clip=clip,
            reviewed_frame=reviewed_frames.get(clip),
            max_video_frames=max_video_frames,
            fps=fps,
            net_post_height_in=net_post_height_in,
            net_center_height_in=net_center_height_in,
        )
        for clip in clip_names
    ]
    status = "rendered" if any(item.get("status") == "rendered" for item in clip_summaries) else "no_renders"
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_calibration_overlay_run",
        "status": status,
        "run_root": str(run_root),
        "frames_root": str(frames_root),
        "clip_count": len(clip_summaries),
        "rendered_clip_count": sum(1 for item in clip_summaries if item.get("status") == "rendered"),
        "clips": clip_summaries,
        "qualitative_status": "corrected_unverified_visual_review_required",
    }
    if write_index:
        (run_root / "calibration_overlay_index.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if write_markdown:
        (run_root / "calibration_overlay_index.md").write_text(_run_markdown_index(summary), encoding="utf-8")
    return summary


def _validate_net_plane_matches_sport(
    calibration: CourtCalibration,
    net_plane: NetPlane,
    *,
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
) -> None:
    expected = build_net_plane(
        calibration.sport,
        post_height_in=net_post_height_in,
        center_height_in=net_center_height_in,
    )
    expected_points = [*expected.endpoints, [0.0, 0.0, expected.center_height_in * 0.0254]]
    actual_points = [*net_plane.endpoints, [0.0, 0.0, net_plane.center_height_in * 0.0254]]
    for actual, expected_point in zip(actual_points, expected_points, strict=True):
        point_mismatch = any(
            not math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-9)
            for a, e in zip(actual, expected_point, strict=True)
        )
        if point_mismatch:
            raise ValueError("net plane endpoints do not match calibration sport")


def _draw_image_overlay(cv2: Any, frame: Any, overlay: OverlayPayload) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    for line in overlay["court_lines"]:
        start, end = [_int_point(point) for point in line["image"]]
        line_id = str(line["id"])
        color = COURT_LINE_COLOR
        thickness = 2
        cv2.line(frame, start, end, color, thickness, line_type)
        _draw_label(cv2, frame, line_id, _midpoint(start, end), color=color)

    for segment in overlay["net_segments"]:
        start, end = [_int_point(point) for point in segment["image"]]
        cv2.line(frame, start, end, NET_LINE_COLOR, 2, line_type)

    for point_id, point in overlay["net_points"].items():
        center = _int_point(point)
        cv2.circle(frame, center, 5, NET_POINT_COLOR, -1)
        _draw_label(cv2, frame, str(point_id), (center[0] + 6, center[1] - 6), color=NET_POINT_COLOR)


def _render_clip_calibration_overlays(
    *,
    cv2: Any,
    run_root: Path,
    frames_root: Path,
    clip: str,
    reviewed_frame: str | None,
    max_video_frames: int | None,
    fps: float,
    net_post_height_in: float | None,
    net_center_height_in: float | None,
) -> dict[str, Any]:
    clip_dir = run_root / clip
    calibration_path = clip_dir / "court_calibration.json"
    net_plane_path = clip_dir / "net_plane.json"
    if not calibration_path.is_file() or not net_plane_path.is_file():
        return {
            "clip": clip,
            "status": "skipped",
            "warnings": ["missing calibration or net-plane artifact"],
            "qualitative_status": "corrected_unverified_visual_review_required",
        }

    frame_dir = frames_root / clip
    frame_paths = _frame_pack_images(frame_dir, max_video_frames=max_video_frames)
    if not frame_paths:
        return {
            "clip": clip,
            "status": "skipped",
            "warnings": [f"no review frames found under {frame_dir}"],
            "qualitative_status": "corrected_unverified_visual_review_required",
        }

    calibration = load_calibration_artifact(calibration_path)
    net_plane = load_net_plane_artifact(net_plane_path)
    compare_dir = clip_dir / "compare"
    review_frame_path = _review_frame_path(frame_dir, reviewed_frame, frame_paths)
    frame_out = compare_dir / "calibration_overlay_frame.jpg"
    frame_summary = render_calibration_image_overlay(
        image_path=review_frame_path,
        out_path=frame_out,
        calibration=calibration,
        net_plane=net_plane,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
        cv2_module=cv2,
    )
    video_out = compare_dir / "calibration_overlay.mp4"
    video_frame_count = _render_calibration_video_overlay(
        cv2=cv2,
        frame_paths=frame_paths,
        out_path=video_out,
        calibration=calibration,
        net_plane=net_plane,
        fps=fps,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_calibration_clip_overlay",
        "clip": clip,
        "status": "rendered",
        "review_frame": str(review_frame_path),
        "rendered_images": [str(frame_out)],
        "rendered_videos": [str(video_out)],
        "video_frame_count": video_frame_count,
        "court_line_ids": frame_summary["court_line_ids"],
        "court_line_count": frame_summary["court_line_count"],
        "net_point_count": frame_summary["net_point_count"],
        "warnings": [],
        "qualitative_status": "corrected_unverified_visual_review_required",
    }
    (compare_dir / "calibration_overlay_index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _render_calibration_video_overlay(
    *,
    cv2: Any,
    frame_paths: list[Path],
    out_path: Path,
    calibration: CourtCalibration,
    net_plane: NetPlane,
    fps: float,
    net_post_height_in: float | None,
    net_center_height_in: float | None,
) -> int:
    first_frame = _read_first_frame(cv2, frame_paths)
    if first_frame is None:
        return 0
    height, width = first_frame.shape[:2]
    calibration = calibration_for_image_size(calibration, width=int(width), height=int(height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open calibration overlay writer: {out_path}")
    count = 0
    overlay = build_calibration_overlay(
        calibration,
        net_plane=net_plane,
        net_post_height_in=net_post_height_in,
        net_center_height_in=net_center_height_in,
    )
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            _draw_image_overlay(cv2, frame, overlay)
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count


def _read_first_frame(cv2: Any, frame_paths: list[Path]) -> Any | None:
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            return frame
    return None


def _frame_pack_images(frame_dir: Path, *, max_video_frames: int | None) -> list[Path]:
    frames = (
        sorted(
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if frame_dir.is_dir()
        else []
    )
    return frames[:max_video_frames] if max_video_frames is not None else frames


def _review_frame_path(frame_dir: Path, reviewed_frame: str | None, frame_paths: list[Path]) -> Path:
    if reviewed_frame:
        candidate = frame_dir / reviewed_frame
        if candidate.is_file():
            return candidate
    return frame_paths[0]


def _reviewed_frames_by_clip(run_root: Path) -> dict[str, str]:
    summary_path = run_root / "court_corner_calibration_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    clips = payload.get("clips")
    if not isinstance(clips, list):
        return {}
    reviewed: dict[str, str] = {}
    for item in clips:
        if isinstance(item, dict) and isinstance(item.get("clip"), str) and isinstance(item.get("frame"), str):
            reviewed[item["clip"]] = item["frame"]
    return reviewed


def _artifact_clip_names(run_root: Path) -> list[str]:
    return (
        sorted(
            path.name
            for path in run_root.iterdir()
            if path.is_dir()
            and (path / "court_calibration.json").is_file()
            and (path / "net_plane.json").is_file()
        )
        if run_root.is_dir()
        else []
    )


def _run_markdown_index(summary: dict[str, Any]) -> str:
    rows = []
    for clip in summary["clips"]:
        images = ", ".join(Path(path).name for path in clip.get("rendered_images", [])) or "none"
        videos = ", ".join(Path(path).name for path in clip.get("rendered_videos", [])) or "none"
        warnings = "; ".join(clip.get("warnings", [])) or "none"
        rows.append(
            f"| `{clip['clip']}` | `{clip['status']}` | {images} | {videos} | {warnings} |"
        )
    table = "\n".join(rows)
    return (
        "# Calibration Overlay Index\n\n"
        f"- Status: `{summary['qualitative_status']}`\n"
        f"- Rendered clips: {summary['rendered_clip_count']} / {summary['clip_count']}\n\n"
        "| Clip | Status | Images | Videos | Warnings |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{table}\n"
    )


def _net_top_projection_trust(
    ground_net_image: list[list[float]],
    net_points: dict[str, list[float]],
    *,
    intrinsics_source: str,
) -> dict[str, Any]:
    if intrinsics_source in UNTRUSTED_NET_TOP_INTRINSIC_SOURCES:
        return {
            "trusted": False,
            "status": "untrusted_estimated_intrinsics",
            "angle_delta_deg": 180.0,
            "length_ratio": float("inf"),
        }
    if len(ground_net_image) != 2 or not {"left_post", "right_post"}.issubset(net_points):
        return {
            "trusted": False,
            "status": "untrusted_pnp_geometry",
            "angle_delta_deg": 180.0,
            "length_ratio": float("inf"),
        }

    ground_start, ground_end = ground_net_image
    top_start = net_points["left_post"]
    top_end = net_points["right_post"]
    ground_length = _point_distance(ground_start, ground_end)
    top_length = _point_distance(top_start, top_end)
    if ground_length <= 0.0 or top_length <= 0.0:
        return {
            "trusted": False,
            "status": "untrusted_pnp_geometry",
            "angle_delta_deg": 180.0,
            "length_ratio": float("inf"),
        }

    angle_delta_deg = _line_angle_delta_deg(ground_start, ground_end, top_start, top_end)
    length_ratio = max(top_length / ground_length, ground_length / top_length)
    trusted = (
        angle_delta_deg <= MAX_TRUSTED_NET_TOP_ANGLE_DELTA_DEG
        and length_ratio <= MAX_TRUSTED_NET_TOP_LENGTH_RATIO
    )
    return {
        "trusted": trusted,
        "status": "trusted_pnp_geometry" if trusted else "untrusted_pnp_geometry",
        "angle_delta_deg": _round_float(angle_delta_deg),
        "length_ratio": _round_float(length_ratio),
    }


def _line_angle_delta_deg(
    first_start: list[float],
    first_end: list[float],
    second_start: list[float],
    second_end: list[float],
) -> float:
    first_angle = math.degrees(math.atan2(first_end[1] - first_start[1], first_end[0] - first_start[0]))
    second_angle = math.degrees(math.atan2(second_end[1] - second_start[1], second_end[0] - second_start[0]))
    delta = abs(first_angle - second_angle) % 180.0
    return min(delta, 180.0 - delta)


def _point_distance(first: list[float], second: list[float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _draw_label(
    cv2: Any,
    frame: Any,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)
    shadow = (0, 0, 0)
    cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), font, 0.38, shadow, 2, line_type)
    label_color = color if color != COURT_LINE_COLOR else TEXT_COLOR
    cv2.putText(frame, text, origin, font, 0.38, label_color, 1, line_type)


def _midpoint(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    return int(round((start[0] + end[0]) / 2.0)), int(round((start[1] + end[1]) / 2.0))


def _int_point(point: list[float]) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


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


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for calibration frame overlay rendering") from exc
    return cv2
