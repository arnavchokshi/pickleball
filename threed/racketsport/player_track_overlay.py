"""Render qualitative overlays for player tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Tracks, validate_artifact_file


def load_tracks(path: str | Path) -> Tracks:
    parsed = validate_artifact_file("tracks", Path(path))
    if not isinstance(parsed, Tracks):
        raise ValueError("tracks artifact did not parse as Tracks")
    return parsed


def render_player_track_overlay(
    *,
    video_path: str | Path,
    tracks: Tracks,
    output_path: str | Path,
    max_frames: int | None = None,
    write_index: bool = True,
    bbox_scale_x: float = 1.0,
    bbox_scale_y: float = 1.0,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for player track overlay rendering") from exc

    video = Path(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or tracks.fps or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open overlay writer: {output}")

    frames_by_index = _track_frames_by_index(tracks, bbox_scale_x=bbox_scale_x, bbox_scale_y=bbox_scale_y)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _draw_frame(cv2, frame, frame_index, frames_by_index)
            writer.write(frame)
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_player_track_overlay",
        "status": "rendered",
        "video_path": str(video),
        "overlay_path": str(output),
        "frame_count": frame_index,
        "player_count": len(tracks.players),
        "track_frame_count": sum(len(player.frames) for player in tracks.players),
        "bbox_scale_x": float(bbox_scale_x),
        "bbox_scale_y": float(bbox_scale_y),
        "qualitative_status": "prototype_not_gate_verified",
        "available_layers": ["players"],
    }
    if write_index:
        (output.parent / "player_track_overlay_index.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _track_frames_by_index(tracks: Tracks, *, bbox_scale_x: float, bbox_scale_y: float) -> dict[int, list[dict[str, Any]]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for player in tracks.players:
        label = f"P{player.id} {player.side}-{player.role}"
        color = _color_for_id(player.id)
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            x1, y1, x2, y2 = frame.bbox
            by_index.setdefault(frame_index, []).append(
                {
                    "id": player.id,
                    "label": label,
                    "bbox": (
                        x1 * bbox_scale_x,
                        y1 * bbox_scale_y,
                        x2 * bbox_scale_x,
                        y2 * bbox_scale_y,
                    ),
                    "world_xy": frame.world_xy,
                    "conf": frame.conf,
                    "color": color,
                }
            )
    return by_index


def _draw_frame(cv2: Any, frame: Any, frame_index: int, frames_by_index: dict[int, list[dict[str, Any]]]) -> None:
    for item in frames_by_index.get(frame_index, []):
        x1, y1, x2, y2 = [int(round(float(value))) for value in item["bbox"]]
        color = item["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        foot = ((x1 + x2) // 2, y2)
        cv2.circle(frame, foot, 5, color, -1)
        label = f"{item['label']} {float(item['conf']):.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def _color_for_id(player_id: int) -> tuple[int, int, int]:
    palette = [
        (60, 220, 255),
        (80, 200, 80),
        (255, 180, 80),
        (220, 120, 255),
        (255, 255, 80),
        (80, 120, 255),
    ]
    return palette[player_id % len(palette)]


__all__ = ["load_tracks", "render_player_track_overlay"]
