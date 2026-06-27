"""CPU-only self-vs-self ghost visualization artifact payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_ghost_payload(
    *,
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
    metric_name: str,
) -> dict[str, Any]:
    """Return a contact-aligned self-vs-self ghost payload.

    The payload is render-agnostic: it contains relative-time traces and a small
    contact-frame summary that a later renderer can consume.
    """

    baseline_trace = _sanitize_trace(baseline, trace_name="baseline")
    comparison_trace = _sanitize_trace(comparison, trace_name="comparison")
    if baseline_trace["player_id"] != comparison_trace["player_id"]:
        raise ValueError("baseline and comparison must use the same player_id")

    baseline_samples = _relative_samples(baseline_trace)
    comparison_samples = _relative_samples(comparison_trace)

    return {
        "schema_version": 1,
        "artifact_type": "self_vs_self_ghost_payload",
        "render_status": "cpu_payload_only",
        "alignment": {
            "mode": "contact_frame",
            "baseline_contact_t": baseline_trace["contact_t"],
            "comparison_contact_t": comparison_trace["contact_t"],
        },
        "player_id": baseline_trace["player_id"],
        "metric_name": _non_empty_str(metric_name, "metric_name"),
        "traces": {
            "baseline": {"label": baseline_trace["label"], "samples": baseline_samples},
            "comparison": {"label": comparison_trace["label"], "samples": comparison_samples},
        },
        "summary": _summary(baseline_samples, comparison_samples),
    }


def _sanitize_trace(trace: Mapping[str, Any], *, trace_name: str) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        raise ValueError(f"{trace_name} must be an object")
    samples = trace.get("samples", [])
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError(f"{trace_name}/samples must be an array")
    return {
        "label": _non_empty_str(trace.get("label"), f"{trace_name}/label"),
        "player_id": _int_value(trace.get("player_id"), f"{trace_name}/player_id"),
        "contact_t": _float_value(trace.get("contact_t"), f"{trace_name}/contact_t"),
        "samples": [_sanitize_sample(index, sample, trace_name=trace_name) for index, sample in enumerate(samples)],
    }


def _sanitize_sample(index: int, sample: Mapping[str, Any], *, trace_name: str) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise ValueError(f"{trace_name}/samples/{index} must be an object")
    return {
        "t": _float_value(sample.get("t"), f"{trace_name}/samples/{index}/t"),
        "xy_m": _xy(sample.get("xy_m"), f"{trace_name}/samples/{index}/xy_m"),
        "metric_value": _float_value(sample.get("metric_value"), f"{trace_name}/samples/{index}/metric_value"),
    }


def _relative_samples(trace: dict[str, Any]) -> list[dict[str, Any]]:
    contact_t = trace["contact_t"]
    samples = [
        {
            "rel_t": _clean_float(sample["t"] - contact_t),
            "xy_m": sample["xy_m"],
            "metric_value": sample["metric_value"],
        }
        for sample in trace["samples"]
    ]
    samples.sort(key=lambda sample: sample["rel_t"])
    return samples


def _summary(baseline_samples: list[dict[str, Any]], comparison_samples: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_contact = _closest_contact_sample(baseline_samples)
    comparison_contact = _closest_contact_sample(comparison_samples)
    if baseline_contact is None or comparison_contact is None:
        return {"contact_delta_xy_m": None, "contact_metric_delta": None}
    return {
        "contact_delta_xy_m": [
            _clean_float(comparison_contact["xy_m"][0] - baseline_contact["xy_m"][0]),
            _clean_float(comparison_contact["xy_m"][1] - baseline_contact["xy_m"][1]),
        ],
        "contact_metric_delta": _clean_float(comparison_contact["metric_value"] - baseline_contact["metric_value"]),
    }


def _closest_contact_sample(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample["rel_t"]))


def _xy(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a 2D coordinate")
    return [_float_value(value[0], f"{field}/0"), _float_value(value[1], f"{field}/1")]


def _float_value(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        return _clean_float(float(value))
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


def _clean_float(value: float) -> float:
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


__all__ = ["build_ghost_payload"]
