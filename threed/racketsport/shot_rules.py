"""Declarative P6-1 shot-rule table evaluator.

This module intentionally does not emit an artifact. The future integration
point is the existing shot_taxonomy ``racketsport_shots`` producer.
"""

from __future__ import annotations

import json
import math
from os import PathLike
from typing import Any, Mapping, Sequence


FEATURE_KEYS = (
    "contact_t",
    "contact_frame",
    "player_id",
    "contact_world_xy",
    "contact_zone",
    "contact_height_m",
    "apex_height_m",
    "landing_world_xy",
    "landing_zone",
    "net_crossing_height_m",
    "horizontal_speed_mps",
    "outgoing_dir",
    "rally_shot_index",
    "is_serve_candidate",
    "prev_shot_bounced",
    "trust",
)
EXPECTED_CLASSES = (
    "serve",
    "return",
    "lob",
    "smash",
    "volley",
    "dink",
    "drop",
    "drive",
    "unknown",
)
OPS = {"eq", "in", "gte", "lt", "lt_feature"}


def load_shot_rules(path: str | PathLike[str]) -> dict[str, Any]:
    """Load and validate a shot-rule table from JSON."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("shot rules table must be a JSON object")
    table = dict(payload)
    _validate_table(table)
    return table


def classify_shot(features: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one in-memory S1 feature dict with a validated rule table."""

    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    rules = table.get("rules")
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValueError("table.rules must be a list")

    modifiers = table.get("confidence_modifiers")
    if not isinstance(modifiers, Mapping):
        modifiers = {}

    for rule in sorted((dict(rule) for rule in rules if isinstance(rule, Mapping)), key=lambda item: item["priority"]):
        if _condition_matches(rule["conditions"], features):
            seed_confidence = _finite_float(rule["seed_confidence"], "seed_confidence")
            modifier = _confidence_modifier(features.get("trust"), modifiers)
            confidence = min(seed_confidence, seed_confidence * modifier)
            return {
                "shot_class": str(rule["class"]),
                "confidence": round(confidence, 6),
                "rule_fired": str(rule["class"]),
                "rationale": str(rule["rationale"]),
            }

    return {
        "shot_class": "unknown",
        "confidence": 0.0,
        "rule_fired": "no_rule_matched",
        "rationale": "No rule matched and the table did not provide an unknown fallback.",
    }


def classify_shots(features_list: Sequence[Mapping[str, Any]], table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Classify a sequence of in-memory feature dicts."""

    if isinstance(features_list, (str, bytes)) or not isinstance(features_list, Sequence):
        raise ValueError("features_list must be a sequence")
    return [classify_shot(features, table) for features in features_list]


def _validate_table(table: Mapping[str, Any]) -> None:
    if table.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if table.get("table_version") != "v0":
        raise ValueError("table_version must be v0")
    if not isinstance(table.get("reconciliation_note"), str) or not table["reconciliation_note"].strip():
        raise ValueError("reconciliation_note is required")
    if tuple(table.get("feature_keys", ())) != FEATURE_KEYS:
        raise ValueError("feature_keys must match the S1 binding exactly")

    modifiers = table.get("confidence_modifiers")
    if not isinstance(modifiers, Mapping):
        raise ValueError("confidence_modifiers must be an object")
    for trust in ("estimated", "unverified_cue"):
        if trust not in modifiers:
            raise ValueError(f"confidence_modifiers.{trust} is required")
    for key, value in modifiers.items():
        number = _finite_float(value, f"confidence_modifiers.{key}")
        if number < 0.0 or number > 1.0:
            raise ValueError(f"confidence_modifiers.{key} must be in [0, 1]")

    rules = table.get("rules")
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValueError("rules must be a list")
    if [rule.get("class") for rule in rules if isinstance(rule, Mapping)] != list(EXPECTED_CLASSES):
        raise ValueError("rules must cover the v0 classes in priority order")

    priorities: list[int] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ValueError(f"rules/{index} must be an object")
        priorities.append(_priority(rule.get("priority"), index))
        if not isinstance(rule.get("class"), str) or not rule["class"]:
            raise ValueError(f"rules/{index}.class is required")
        seed = _finite_float(rule.get("seed_confidence"), f"rules/{index}.seed_confidence")
        if seed < 0.0 or seed > 1.0:
            raise ValueError(f"rules/{index}.seed_confidence must be in [0, 1]")
        if not isinstance(rule.get("rationale"), str) or not rule["rationale"].strip():
            raise ValueError(f"rules/{index}.rationale is required")
        _validate_condition(rule.get("conditions"), path=f"rules/{index}.conditions")

    if priorities != list(range(1, len(rules) + 1)):
        raise ValueError("rule priorities must be contiguous 1-based integers in table order")
    if not _is_always(rules[-1]["conditions"]):
        raise ValueError("final rule must be the unknown always fallback")


def _validate_condition(condition: Any, *, path: str) -> None:
    if not isinstance(condition, Mapping):
        raise ValueError(f"{path} must be an object")
    if "always" in condition:
        if set(condition) != {"always"} or condition["always"] is not True:
            raise ValueError(f"{path}.always must be true and standalone")
        return
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        if set(condition) != {key}:
            raise ValueError(f"{path} cannot mix {key} with other condition keys")
        children = condition[key]
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)) or not children:
            raise ValueError(f"{path}.{key} must be a non-empty list")
        for index, child in enumerate(children):
            _validate_condition(child, path=f"{path}.{key}/{index}")
        return

    feature = condition.get("feature")
    if feature not in FEATURE_KEYS:
        raise ValueError(f"{path} references unknown feature key: {feature}")
    op = condition.get("op")
    if op not in OPS:
        raise ValueError(f"{path}.op must be one of {sorted(OPS)}")
    if op == "in":
        values = condition.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"{path}.values must be a non-empty list")
    elif op == "lt_feature":
        other_feature = condition.get("other_feature")
        if other_feature not in FEATURE_KEYS:
            raise ValueError(f"{path} references unknown feature key: {other_feature}")
    elif "value" not in condition:
        raise ValueError(f"{path}.value is required")


def _condition_matches(condition: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    if condition.get("always") is True:
        return True
    children = condition.get("all")
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
        return all(_condition_matches(child, features) for child in children if isinstance(child, Mapping))
    children = condition.get("any")
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
        return any(_condition_matches(child, features) for child in children if isinstance(child, Mapping))

    feature = str(condition.get("feature"))
    value = features.get(feature)
    if value is None:
        return False
    op = condition.get("op")
    if op == "eq":
        return value == condition.get("value")
    if op == "in":
        values = condition.get("values")
        return isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and value in values
    if op == "gte":
        left = _maybe_finite_float(value)
        right = _maybe_finite_float(condition.get("value"))
        return left is not None and right is not None and left >= right
    if op == "lt":
        left = _maybe_finite_float(value)
        right = _maybe_finite_float(condition.get("value"))
        return left is not None and right is not None and left < right
    if op == "lt_feature":
        right_value = features.get(str(condition.get("other_feature")))
        if right_value is None:
            return False
        left = _maybe_finite_float(value)
        right = _maybe_finite_float(right_value)
        return left is not None and right is not None and left < right
    return False


def _confidence_modifier(trust: Any, modifiers: Mapping[str, Any]) -> float:
    if trust is None:
        return 1.0
    value = modifiers.get(str(trust), 1.0)
    number = _maybe_finite_float(value)
    if number is None:
        return 1.0
    return min(1.0, max(0.0, number))


def _priority(value: Any, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"rules/{index}.priority must be an integer")
    return int(value)


def _is_always(condition: Any) -> bool:
    return isinstance(condition, Mapping) and condition.get("always") is True and len(condition) == 1


def _finite_float(value: Any, name: str) -> float:
    number = _maybe_finite_float(value)
    if number is None:
        raise ValueError(f"{name} must be finite")
    return number


def _maybe_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = ["FEATURE_KEYS", "classify_shot", "classify_shots", "load_shot_rules"]
