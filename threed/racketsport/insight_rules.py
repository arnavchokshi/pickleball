"""Rule-based insight engine."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from threed.racketsport.confidence import gate_metric_claim


RULE_CLIP_PRE_ROLL_SEC = 0.25
RULE_CLIP_POST_ROLL_SEC = 1.0
RULE_PLACEHOLDER_DRILL_MINUTES = 5.0


@dataclass(frozen=True)
class InsightRule:
    id: str
    metric: str
    predicate: Callable[[Any], bool]


INSIGHT_RULES = (
    InsightRule(id="nvz_margin_below_zero", metric="nvz_margin_ft", predicate=lambda value: _numeric(value) < 0.0),
    InsightRule(id="balance_score_low", metric="balance_score", predicate=lambda value: _numeric(value) < 0.65),
    InsightRule(id="paddle_face_large_abs", metric="paddle_face_deg", predicate=lambda value: abs(_numeric(value)) >= 15.0),
)


def select_insight_rules(metrics_payload: Mapping[str, Any], *, threshold: float = 0.8) -> list[dict[str, Any]]:
    """Select deterministic fact-only habit candidates from shot metric dictionaries."""

    capture_quality = metrics_payload.get("capture_quality", "good")
    payload_missing_upstream = _list_value(metrics_payload.get("missing_upstream_reasons"))
    habits: list[dict[str, Any]] = []

    for player in _list_value(metrics_payload.get("players")):
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id")
        for shot_index, shot in enumerate(_list_value(player.get("shots"))):
            if not isinstance(shot, Mapping):
                continue
            habits.extend(
                _select_shot_habits(
                    player_id=player_id,
                    shot_index=shot_index,
                    shot=shot,
                    capture_quality=capture_quality,
                    payload_missing_upstream=payload_missing_upstream,
                    threshold=threshold,
                )
            )

    return habits


def _select_shot_habits(
    *,
    player_id: Any,
    shot_index: int,
    shot: Mapping[str, Any],
    capture_quality: Any,
    payload_missing_upstream: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    shot_metrics = shot.get("metrics", {})
    if not isinstance(shot_metrics, Mapping):
        return []

    habits: list[dict[str, Any]] = []
    for rule in INSIGHT_RULES:
        metric_payload = shot_metrics.get(rule.metric)
        if not isinstance(metric_payload, Mapping):
            continue
        if metric_payload.get("gated") is True:
            continue

        value = metric_payload.get("value")
        if not _predicate_matches(rule, value):
            continue

        missing_upstream = payload_missing_upstream + _list_value(metric_payload.get("missing_upstream_reasons"))
        claim_gate = gate_metric_claim(
            value,
            metric_payload.get("conf"),
            threshold=threshold,
            claim_name=rule.metric,
            capture_quality=capture_quality,
            missing_upstream_reasons=missing_upstream,
        )
        if claim_gate["decision"] == "omit":
            continue

        habits.append(
            _habit_dict(
                player_id=player_id,
                shot_index=shot_index,
                shot=shot,
                rule=rule,
                metric_payload=metric_payload,
                claim_gate=claim_gate,
            )
        )
    return habits


def _habit_dict(
    *,
    player_id: Any,
    shot_index: int,
    shot: Mapping[str, Any],
    rule: InsightRule,
    metric_payload: Mapping[str, Any],
    claim_gate: Mapping[str, Any],
) -> dict[str, Any]:
    shot_time_s = float(shot.get("t", 0.0))
    shot_type = str(shot.get("type", "shot"))
    value = metric_payload.get("value")
    summary_prefix = "Measured" if claim_gate["decision"] == "allow" else "Relative signal: measured"
    source = {
        "player_id": player_id,
        "shot_index": shot_index,
        "shot_type": shot_type,
        "shot_time_s": shot_time_s,
        "metric": rule.metric,
        "value": value,
    }
    if "frames" in metric_payload:
        source["frames"] = metric_payload["frames"]

    return {
        "id": f"p{player_id}_s{shot_index:03d}_{_slug(rule.metric)}",
        "rule_id": rule.id,
        "title": f"P{player_id} {shot_type} {rule.metric}",
        "summary": f"{summary_prefix} {rule.metric}={_format_value(value)} at t={shot_time_s:.3f}s.",
        "confidence": claim_gate["confidence"],
        "clip_ref": {
            "t0_sec": max(0.0, shot_time_s - RULE_CLIP_PRE_ROLL_SEC),
            "t1_sec": shot_time_s + RULE_CLIP_POST_ROLL_SEC,
        },
        "cue": f"Fact-only metric review: {rule.metric}.",
        "drill": {"name": f"Metric review: {rule.metric}", "duration_min": RULE_PLACEHOLDER_DRILL_MINUTES},
        "source": source,
        "claim_gate": dict(claim_gate),
        "llm_used": False,
    }


def _predicate_matches(rule: InsightRule, value: Any) -> bool:
    try:
        return rule.predicate(value)
    except (TypeError, ValueError):
        return False


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean metric values are not numeric")
    return float(value)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)
