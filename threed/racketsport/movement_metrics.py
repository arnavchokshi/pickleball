"""Court-frame movement metric primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from threed.racketsport.court_templates import COURT_TEMPLATES, FT_TO_M, Sport


MetricDict = dict[str, dict[str, Any]]


def nvz_margin_metric(
    foot_world_xy: Sequence[float],
    *,
    sport: Sport = "pickleball",
    conf: float = 1.0,
) -> MetricDict:
    """Return signed feet from the non-volley-zone line.

    Positive values are outside the kitchen. Negative values are between the net
    and the NVZ line, matching the downstream ``nvz_margin_below_zero`` rule.
    """

    _, y_m = _xy(foot_world_xy, name="foot_world_xy")
    template = COURT_TEMPLATES[sport]
    if template.non_volley_zone_ft is None:
        raise ValueError(f"{sport} has no non-volley-zone template")

    y_ft = y_m / FT_TO_M
    margin_ft = abs(y_ft) - template.non_volley_zone_ft
    return {
        "nvz_margin_ft": _metric_payload(
            value=margin_ft,
            conf=conf,
            unit="ft",
            source="cpu_world_foot_point",
        )
    }


def inter_player_spacing_metric(
    player_a_world_xy: Sequence[float],
    player_b_world_xy: Sequence[float],
    *,
    conf: float = 1.0,
) -> MetricDict:
    """Return planar spacing between two world foot points in feet."""

    ax, ay = _xy(player_a_world_xy, name="player_a_world_xy")
    bx, by = _xy(player_b_world_xy, name="player_b_world_xy")
    spacing_ft = math.hypot(ax - bx, ay - by) / FT_TO_M
    return {
        "inter_player_spacing_ft": _metric_payload(
            value=spacing_ft,
            conf=conf,
            unit="ft",
            source="cpu_world_foot_points",
        )
    }


def _metric_payload(*, value: float, conf: float, unit: str, source: str) -> dict[str, Any]:
    return {
        "value": float(value),
        "conf": _confidence(conf),
        "unit": unit,
        "gated": False,
        "source": source,
    }


def _xy(value: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be world_xy with exactly 2 values")
    x = _finite_float(value[0], name=f"{name}[0]")
    y = _finite_float(value[1], name=f"{name}[1]")
    return x, y


def _confidence(value: float) -> float:
    conf = _finite_float(value, name="conf")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    return conf


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
