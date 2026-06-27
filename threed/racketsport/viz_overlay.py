"""CPU-only video overlay metadata payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SUPPORTED_ELEMENT_TYPES = {
    "base_of_support_box",
    "contact_marker",
    "knee_angle_arc",
    "paddle_face_indicator",
}


def build_overlay_metadata(
    *,
    video_ref: Mapping[str, Any],
    elements: Iterable[Mapping[str, Any]],
    source_artifacts: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return deterministic overlay metadata without rendering video frames."""

    sanitized_elements = [_sanitize_element(index, element) for index, element in enumerate(elements)]
    return {
        "schema_version": 1,
        "artifact_type": "video_overlay_metadata",
        "render_status": "cpu_payload_only",
        "video_ref": _sanitize_video_ref(video_ref),
        "source_artifacts": [str(artifact) for artifact in (source_artifacts or [])],
        "element_count": len(sanitized_elements),
        "elements": sanitized_elements,
    }


def _sanitize_video_ref(video_ref: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(video_ref, Mapping):
        raise ValueError("video_ref must be an object")
    return {
        "clip_id": _non_empty_str(video_ref.get("clip_id"), "video_ref/clip_id"),
        "fps": _float_value(video_ref.get("fps"), "video_ref/fps"),
        "duration_s": _float_value(video_ref.get("duration_s"), "video_ref/duration_s"),
    }


def _sanitize_element(index: int, element: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(element, Mapping):
        raise ValueError(f"elements/{index} must be an object")
    element_type = _non_empty_str(element.get("type"), f"elements/{index}/type")
    if element_type not in SUPPORTED_ELEMENT_TYPES:
        raise ValueError(f"elements/{index}/type is unsupported")

    sanitized: dict[str, Any] = {
        "type": element_type,
        "t": _float_value(element.get("t"), f"elements/{index}/t"),
        "frame": _int_value(element.get("frame"), f"elements/{index}/frame"),
    }
    if "xy_px" in element:
        sanitized["xy_px"] = _xy(element.get("xy_px"), f"elements/{index}/xy_px")
    if "points_px" in element:
        points = element.get("points_px")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise ValueError(f"elements/{index}/points_px must be an array")
        sanitized["points_px"] = [_xy(point, f"elements/{index}/points_px/{point_index}") for point_index, point in enumerate(points)]
    if "label" in element:
        sanitized["label"] = str(element["label"])
    if "value" in element:
        sanitized["value"] = _float_value(element.get("value"), f"elements/{index}/value")
    if "units" in element:
        sanitized["units"] = str(element["units"])
    if "confidence" in element:
        sanitized["confidence"] = _float_value(element.get("confidence"), f"elements/{index}/confidence")
    return sanitized


def _xy(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a 2D coordinate")
    return [_float_value(value[0], f"{field}/0"), _float_value(value[1], f"{field}/1")]


def _float_value(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        rounded = round(float(value), 6)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return 0.0 if rounded == 0 else rounded


def _int_value(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _non_empty_str(value: Any, field: str) -> str:
    text = str(value) if value is not None else ""
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


__all__ = ["build_overlay_metadata"]
