"""CPU-only court-map visualization artifact payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from threed.racketsport.court_templates import get_court_template


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def build_courtmap_payload(
    *,
    sport: str,
    player_paths: Iterable[Mapping[str, Any]],
    heatmap_bins: Iterable[Mapping[str, Any]],
    priority_metrics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic JSON-friendly court-map payload.

    This intentionally does not render images. It packages validated court-space
    data for Phase-9 report surfaces and later renderers.
    """

    template = get_court_template(sport)  # type: ignore[arg-type]
    sanitized_paths = [_sanitize_player_path(path, sport=sport) for path in player_paths]
    sanitized_heatmap = [_sanitize_heatmap_bin(index, bin_payload, sport=sport) for index, bin_payload in enumerate(heatmap_bins)]
    sanitized_markers = [
        _sanitize_priority_metric(index, marker, sport=sport) for index, marker in enumerate(priority_metrics)
    ]

    sanitized_heatmap.sort(key=lambda item: (item["xy_m"][0], item["xy_m"][1], item.get("label", "")))

    return {
        "schema_version": 1,
        "artifact_type": "court_map_payload",
        "render_status": "cpu_payload_only",
        "world_frame": "court_Z0",
        "court": {
            "sport": template.sport,
            "width_ft": float(template.width_ft),
            "length_ft": float(template.length_ft),
            "coordinate_frame": template.coordinate_frame,
        },
        "layers": {
            "player_paths": sanitized_paths,
            "heatmap": {"bins": sanitized_heatmap},
            "priority_metric_markers": sanitized_markers,
        },
        "priority_metric_marker_summary": _priority_marker_summary(sanitized_markers),
    }


def _sanitize_player_path(path: Mapping[str, Any], *, sport: str) -> dict[str, Any]:
    player_id = _int_value(path.get("player_id"), "player_paths/player_id")
    points = path.get("points", [])
    if not isinstance(points, Iterable) or isinstance(points, (str, bytes, Mapping)):
        raise ValueError("player_paths/points must be an array")
    return {
        "player_id": player_id,
        "points": [_sanitize_path_point(index, point, sport=sport) for index, point in enumerate(points)],
    }


def _sanitize_path_point(index: int, point: Mapping[str, Any], *, sport: str) -> dict[str, Any]:
    if not isinstance(point, Mapping):
        raise ValueError(f"player_paths/points/{index} must be an object")
    xy_m = _xy(point.get("xy_m"), f"player_paths/points/{index}/xy_m")
    _validate_in_court(xy_m, sport=sport, field=f"player_paths/points/{index}/xy_m")
    return {
        "t": _float_value(point.get("t"), f"player_paths/points/{index}/t"),
        "xy_m": xy_m,
        "confidence": _float_value(point.get("confidence", 1.0), f"player_paths/points/{index}/confidence"),
    }


def _sanitize_heatmap_bin(index: int, bin_payload: Mapping[str, Any], *, sport: str) -> dict[str, Any]:
    if not isinstance(bin_payload, Mapping):
        raise ValueError(f"heatmap_bins/{index} must be an object")
    xy_m = _xy(bin_payload.get("xy_m"), f"heatmap_bins/{index}/xy_m")
    _validate_in_court(xy_m, sport=sport, field=f"heatmap_bins/{index}/xy_m")
    payload = {
        "xy_m": xy_m,
        "value": _float_value(bin_payload.get("value"), f"heatmap_bins/{index}/value"),
    }
    label = bin_payload.get("label")
    if label is not None:
        payload["label"] = str(label)
    return payload


def _sanitize_priority_metric(index: int, metric: Mapping[str, Any], *, sport: str) -> dict[str, Any]:
    if not isinstance(metric, Mapping):
        raise ValueError(f"priority_metrics/{index} must be an object")
    xy_m = _xy(metric.get("xy_m"), f"priority_metrics/{index}/xy_m")
    _validate_in_court(xy_m, sport=sport, field=f"priority_metrics/{index}/xy_m")
    return {
        "metric": _non_empty_str(metric.get("metric"), f"priority_metrics/{index}/metric"),
        "player_id": _int_value(metric.get("player_id"), f"priority_metrics/{index}/player_id"),
        "t": _float_value(metric.get("t"), f"priority_metrics/{index}/t"),
        "xy_m": xy_m,
        "value": _float_value(metric.get("value"), f"priority_metrics/{index}/value"),
        "units": _non_empty_str(metric.get("units"), f"priority_metrics/{index}/units"),
        "severity": _non_empty_str(metric.get("severity"), f"priority_metrics/{index}/severity"),
        "confidence": _float_value(metric.get("confidence"), f"priority_metrics/{index}/confidence"),
    }


def _priority_marker_summary(markers: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity: dict[str, int] = {}
    for marker in markers:
        severity = marker["severity"]
        by_severity[severity] = by_severity.get(severity, 0) + 1

    primary = None
    if markers:
        primary = sorted(
            markers,
            key=lambda marker: (
                SEVERITY_ORDER.get(marker["severity"], len(SEVERITY_ORDER)),
                -float(marker["confidence"]),
                float(marker["t"]),
                str(marker["metric"]),
            ),
        )[0]

    return {
        "count": len(markers),
        "primary": primary,
        "by_severity": dict(sorted(by_severity.items())),
    }


def _validate_in_court(xy_m: Sequence[float], *, sport: str, field: str) -> None:
    template = get_court_template(sport)  # type: ignore[arg-type]
    half_width_m = template.width_m / 2.0
    half_length_m = template.length_m / 2.0
    x_m, y_m = xy_m
    if x_m < -half_width_m or x_m > half_width_m or y_m < -half_length_m or y_m > half_length_m:
        raise ValueError(f"{field} is outside {sport} court bounds")


def _xy(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a 2D coordinate")
    return [_float_value(value[0], f"{field}/0"), _float_value(value[1], f"{field}/1")]


def _float_value(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        return round(float(value), 6)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


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


__all__ = ["build_courtmap_payload"]
