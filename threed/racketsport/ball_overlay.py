"""CPU-only review video overlays for ``ball_track.json`` artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from threed.racketsport.schemas import BallFrame, BallTrack


VISIBLE_COLOR = (0, 255, 255)
TAIL_COLOR = (0, 170, 255)
INVISIBLE_COLOR = (0, 165, 255)
MISSING_COLOR = (80, 80, 255)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


def load_ball_track(path: str | Path) -> BallTrack:
    """Load and validate a schema-valid ``ball_track.json`` artifact."""

    ball_track_path = Path(path)
    if not ball_track_path.is_file():
        raise ValueError(f"missing ball_track file: {ball_track_path}")
    try:
        payload = json.loads(ball_track_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ball_track schema: {ball_track_path}: invalid JSON: {exc}") from exc
    try:
        return BallTrack.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid ball_track schema: {ball_track_path}: {exc}") from exc


def render_ball_track_overlay(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    out_path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
    tail: int = 12,
    fps_out: float | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Render a qualitative ball-track overlay MP4 from a source video."""

    video_path = Path(video_path)
    ball_track_path = Path(ball_track_path)
    out_path = Path(out_path)
    if not video_path.is_file():
        raise ValueError(f"missing video file: {video_path}")
    _validate_positive_optional_int(max_frames, "max_frames")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if tail < 0:
        raise ValueError("tail must be >= 0")
    if fps_out is not None and (not math.isfinite(float(fps_out)) or float(fps_out) <= 0.0):
        raise ValueError("fps_out must be > 0")

    ball_track = load_ball_track(ball_track_path)
    cv2 = cv2_module or _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    source_fps = _positive_float(cap.get(cv2.CAP_PROP_FPS)) or float(ball_track.fps) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"cannot determine video frame size: {video_path}")

    output_fps = float(fps_out) if fps_out is not None else source_fps / float(stride)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open ball overlay writer: {out_path}")

    samples_by_index = _ball_samples_by_frame_index(ball_track)
    source_indices: list[int] = []
    visible_count = 0
    invisible_count = 0
    missing_count = 0
    rendered_count = 0
    source_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index % stride != 0:
                source_index += 1
                continue

            track_index = _track_index_for_source_frame(
                source_index=source_index,
                source_fps=source_fps,
                ball_fps=float(ball_track.fps),
            )
            sample = samples_by_index.get(track_index)
            status = _sample_status(sample)
            if status == "visible":
                visible_count += 1
            elif status == "invisible":
                invisible_count += 1
            else:
                missing_count += 1

            _draw_ball_overlay(
                cv2,
                frame,
                samples_by_index=samples_by_index,
                track_index=track_index,
                source_index=source_index,
                source_time_s=source_index / source_fps,
                sample=sample,
                status=status,
                tail=tail,
            )
            writer.write(frame)
            source_indices.append(source_index)
            rendered_count += 1
            source_index += 1
            if max_frames is not None and rendered_count >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    if rendered_count == 0:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(f"no frames rendered from video: {video_path}")

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_overlay",
        "status": "rendered",
        "video_path": str(video_path),
        "ball_track_path": str(ball_track_path),
        "out_path": str(out_path),
        "frame_count": rendered_count,
        "visible_frame_count": visible_count,
        "invisible_frame_count": invisible_count,
        "missing_frame_count": missing_count,
        "source_frame_indices": source_indices,
        "source_fps": source_fps,
        "ball_track_fps": float(ball_track.fps),
        "fps_out": output_fps,
        "stride": stride,
        "tail": tail,
        "qualitative_status": "review_overlay_not_gate_verified",
    }


def _ball_samples_by_frame_index(ball_track: BallTrack) -> dict[int, BallFrame]:
    samples: dict[int, BallFrame] = {}
    for sample in ball_track.frames:
        samples[int(round(float(sample.t) * float(ball_track.fps)))] = sample
    return samples


def _track_index_for_source_frame(*, source_index: int, source_fps: float, ball_fps: float) -> int:
    return int(round((float(source_index) / source_fps) * ball_fps))


def _sample_status(sample: BallFrame | None) -> str:
    if sample is None:
        return "missing"
    if sample.visible:
        return "visible"
    return "invisible"


def _draw_ball_overlay(
    cv2: Any,
    frame: Any,
    *,
    samples_by_index: dict[int, BallFrame],
    track_index: int,
    source_index: int,
    source_time_s: float,
    sample: BallFrame | None,
    status: str,
    tail: int,
) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    tail_points = _tail_points(samples_by_index, track_index=track_index, tail=tail)
    if len(tail_points) >= 2:
        for start, end in zip(tail_points, tail_points[1:]):
            cv2.line(frame, start, end, TAIL_COLOR, 2, line_type)
    for point in tail_points[:-1]:
        cv2.circle(frame, point, 3, TAIL_COLOR, -1)

    if sample is not None and sample.visible:
        center = _point(sample.xy)
        cv2.circle(frame, center, 7, SHADOW_COLOR, 3)
        cv2.circle(frame, center, 6, VISIBLE_COLOR, 2)
        cv2.circle(frame, center, 2, VISIBLE_COLOR, -1)

    label = _frame_label(source_index=source_index, source_time_s=source_time_s, sample=sample, status=status)
    _draw_status_marker(cv2, frame, status=status)
    _draw_text(cv2, frame, label, (26, 18), color=_status_color(status))


def _tail_points(samples_by_index: dict[int, BallFrame], *, track_index: int, tail: int) -> list[tuple[int, int]]:
    if tail <= 0:
        return []
    visible_samples = [
        (index, sample)
        for index, sample in samples_by_index.items()
        if index <= track_index and sample.visible
    ]
    visible_samples.sort(key=lambda item: item[0])
    return [_point(sample.xy) for _index, sample in visible_samples[-tail:]]


def _draw_status_marker(cv2: Any, frame: Any, *, status: str) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    color = _status_color(status)
    center = (12, 14)
    if status == "visible":
        cv2.circle(frame, center, 5, color, -1)
        return
    cv2.circle(frame, center, 5, color, 2)
    cv2.line(frame, (8, 10), (16, 18), color, 2, line_type)
    if status == "missing":
        cv2.line(frame, (16, 10), (8, 18), color, 2, line_type)


def _frame_label(*, source_index: int, source_time_s: float, sample: BallFrame | None, status: str) -> str:
    base = f"frame={source_index} t={source_time_s:.3f}s ball={status}"
    if sample is not None:
        return f"{base} conf={sample.conf:.2f}"
    return base


def _draw_text(
    cv2: Any,
    frame: Any,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)
    cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), font, 0.42, SHADOW_COLOR, 2, line_type)
    cv2.putText(frame, text, origin, font, 0.42, color, 1, line_type)


def _point(value: Any) -> tuple[int, int]:
    return int(round(float(value[0]))), int(round(float(value[1])))


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "visible":
        return VISIBLE_COLOR
    if status == "invisible":
        return INVISIBLE_COLOR
    return MISSING_COLOR


def _validate_positive_optional_int(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0")


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed) and parsed > 0:
        return parsed
    return None


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for ball-track overlay rendering") from exc
    return cv2


__all__ = ["load_ball_track", "render_ball_track_overlay"]
