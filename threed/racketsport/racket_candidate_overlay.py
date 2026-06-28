"""Render qualitative overlays for racket/paddle candidate corners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import RacketCandidates, validate_artifact_file


CANDIDATE_COLOR = (80, 220, 255)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


def load_racket_candidates(path: str | Path) -> RacketCandidates:
    parsed = validate_artifact_file("racket_candidates", Path(path))
    if not isinstance(parsed, RacketCandidates):
        raise ValueError("racket_candidates artifact did not parse as RacketCandidates")
    return parsed


def render_racket_candidate_overlay(
    *,
    video_path: str | Path,
    candidates: RacketCandidates,
    output_path: str | Path,
    max_frames: int | None = None,
    write_index: bool = True,
    candidate_coord_width: int | None = None,
    candidate_coord_height: int | None = None,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for racket candidate overlay rendering") from exc

    video = Path(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or candidates.fps or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    if candidate_coord_width is not None and candidate_coord_width <= 0:
        cap.release()
        raise ValueError("candidate_coord_width must be positive when provided")
    if candidate_coord_height is not None and candidate_coord_height <= 0:
        cap.release()
        raise ValueError("candidate_coord_height must be positive when provided")
    coord_scale_x = float(width) / float(candidate_coord_width) if candidate_coord_width is not None else 1.0
    coord_scale_y = float(height) / float(candidate_coord_height) if candidate_coord_height is not None else 1.0
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open racket candidate overlay writer: {output}")

    frames_by_index = _candidate_frames_by_index(candidates, coord_scale_x=coord_scale_x, coord_scale_y=coord_scale_y)
    candidate_frame_count = sum(len(player.frames) for player in candidates.players)
    candidate_frame_indices = sorted(frames_by_index)
    frame_index = 0
    rendered_candidate_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rendered_candidate_count += _draw_frame(cv2, frame, frame_index, frames_by_index)
            writer.write(frame)
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    unrendered_candidate_count = max(0, candidate_frame_count - rendered_candidate_count)
    warnings = ["candidate_frames_outside_video_window"] if unrendered_candidate_count else []

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidate_overlay",
        "status": "rendered",
        "video_path": str(video),
        "overlay_path": str(output),
        "frame_count": frame_index,
        "source_video_frame_count": source_frame_count,
        "source_video_fps": float(fps),
        "candidate_fps": float(candidates.fps),
        "candidate_coord_width": candidate_coord_width,
        "candidate_coord_height": candidate_coord_height,
        "candidate_coord_scale_x": coord_scale_x,
        "candidate_coord_scale_y": coord_scale_y,
        "candidate_player_count": len(candidates.players),
        "candidate_frame_count": candidate_frame_count,
        "candidate_frame_index_min": candidate_frame_indices[0] if candidate_frame_indices else None,
        "candidate_frame_index_max": candidate_frame_indices[-1] if candidate_frame_indices else None,
        "rendered_candidate_count": rendered_candidate_count,
        "unrendered_candidate_count": unrendered_candidate_count,
        "qualitative_status": "candidate_review_not_gate_verified",
        "available_layers": ["paddle_candidates"],
        "warnings": warnings,
    }
    if write_index:
        (output.parent / "racket_candidate_overlay_index.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _candidate_frames_by_index(
    candidates: RacketCandidates,
    *,
    coord_scale_x: float,
    coord_scale_y: float,
) -> dict[int, list[dict[str, Any]]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for player in candidates.players:
        color = _color_for_id(player.id)
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(candidates.fps)))
            by_index.setdefault(frame_index, []).append(
                {
                    "player_id": player.id,
                    "corners_px": [[float(point[0]) * coord_scale_x, float(point[1]) * coord_scale_y] for point in frame.corners_px],
                    "conf": float(frame.conf),
                    "source": frame.source,
                    "color": color,
                }
            )
    return by_index


def _draw_frame(cv2: Any, frame: Any, frame_index: int, frames_by_index: dict[int, list[dict[str, Any]]]) -> int:
    line_type = getattr(cv2, "LINE_AA", 16)
    items = frames_by_index.get(frame_index, [])
    for item in items:
        corners = [_point(point) for point in item["corners_px"]]
        color = item["color"]
        for start, end in zip(corners, [*corners[1:], corners[0]]):
            cv2.line(frame, start, end, color, 2, line_type)
        for index, corner in enumerate(corners):
            cv2.circle(frame, corner, 4, color, -1, line_type)
            _draw_text(cv2, frame, str(index + 1), (corner[0] + 4, corner[1] - 4), color=color)
        label_origin = (min(point[0] for point in corners), max(14, min(point[1] for point in corners) - 8))
        _draw_text(
            cv2,
            frame,
            f"P{item['player_id']} paddle cand {float(item['conf']):.2f}",
            label_origin,
            color=color,
        )
    return len(items)


def _draw_text(cv2: Any, frame: Any, text: str, origin: tuple[int, int], *, color: tuple[int, int, int]) -> None:
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)
    cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), font, 0.45, SHADOW_COLOR, 2, line_type)
    cv2.putText(frame, text, origin, font, 0.45, color or TEXT_COLOR, 1, line_type)


def _point(value: Any) -> tuple[int, int]:
    return int(round(float(value[0]))), int(round(float(value[1])))


def _color_for_id(player_id: int) -> tuple[int, int, int]:
    palette = [
        CANDIDATE_COLOR,
        (80, 200, 80),
        (255, 180, 80),
        (220, 120, 255),
        (255, 255, 80),
        (80, 120, 255),
    ]
    return palette[player_id % len(palette)]


__all__ = ["load_racket_candidates", "render_racket_candidate_overlay"]
