"""CPU-only faithfulness checks for generated coaching copy."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"(?<![\w])-?\d+(?:\.\d+)?(?![\w])")


@dataclass(frozen=True)
class FactRegistry:
    known_keys: frozenset[str]
    known_values: frozenset[str]
    habit_ids: frozenset[str]
    priority_habit_id: str


@dataclass(frozen=True)
class CopyFaithfulnessSummary:
    passed: bool
    llm_used: bool
    priority_habit_preserved: bool
    referenced_fact_keys: list[str]
    referenced_fact_values: list[str]
    unknown_fact_keys: list[str]
    unknown_fact_values: list[str]
    reason_counts: dict[str, int]


def summarize_copy_faithfulness(
    *,
    facts: Mapping[str, Any] | None,
    habit_report: Any,
    generated_copy: str,
    generated_priority_habit_id: str | None = None,
) -> CopyFaithfulnessSummary:
    """Validate generated copy against known facts without calling an external LLM."""

    registry = build_fact_registry(facts=facts, habit_report=habit_report)
    allowed_identifiers = registry.known_keys | registry.habit_ids
    referenced_keys = _unique_in_order(
        token for token in _identifier_tokens(generated_copy) if token in registry.known_keys
    )
    referenced_values = _unique_in_order(
        token for token in _number_tokens(generated_copy) if token in registry.known_values
    )
    unknown_keys = _unique_in_order(
        token
        for token in _identifier_tokens(generated_copy)
        if "_" in token and token not in allowed_identifiers
    )
    unknown_values = _unique_in_order(
        token for token in _number_tokens(generated_copy) if token not in registry.known_values
    )

    reason_counts: Counter[str] = Counter()
    reason_counts.update({"unknown_fact_key": len(unknown_keys)} if unknown_keys else {})
    reason_counts.update({"unknown_fact_value": len(unknown_values)} if unknown_values else {})

    priority_habit_preserved = _priority_habit_preserved(
        priority_habit_id=registry.priority_habit_id,
        generated_copy=generated_copy,
        generated_priority_habit_id=generated_priority_habit_id,
    )
    if registry.priority_habit_id and registry.priority_habit_id not in generated_copy:
        reason_counts["missing_priority_habit_reference"] += 1
    if (
        registry.priority_habit_id
        and generated_priority_habit_id is not None
        and generated_priority_habit_id != registry.priority_habit_id
    ):
        reason_counts["priority_habit_mismatch"] += 1

    return CopyFaithfulnessSummary(
        passed=not reason_counts,
        llm_used=False,
        priority_habit_preserved=priority_habit_preserved,
        referenced_fact_keys=referenced_keys,
        referenced_fact_values=referenced_values,
        unknown_fact_keys=unknown_keys,
        unknown_fact_values=unknown_values,
        reason_counts=dict(sorted(reason_counts.items())),
    )


def build_fact_registry(*, facts: Mapping[str, Any] | None, habit_report: Any) -> FactRegistry:
    known_keys: set[str] = set()
    known_values: set[str] = set()
    habit_ids: set[str] = set()

    _collect_facts(_json_like(facts or {}), known_keys=known_keys, known_values=known_values)
    habit_payload = _json_like(habit_report)
    _collect_facts(habit_payload, known_keys=known_keys, known_values=known_values)

    if isinstance(habit_payload, Mapping):
        priority_habit_id = str(habit_payload.get("priority_habit_id") or "")
        for habit in _list_value(habit_payload.get("habits")):
            if isinstance(habit, Mapping) and habit.get("id") is not None:
                habit_ids.add(str(habit["id"]))
    else:
        priority_habit_id = ""

    if priority_habit_id:
        habit_ids.add(priority_habit_id)

    return FactRegistry(
        known_keys=frozenset(known_keys),
        known_values=frozenset(known_values),
        habit_ids=frozenset(habit_ids),
        priority_habit_id=priority_habit_id,
    )


def _collect_facts(value: Any, *, known_keys: set[str], known_values: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            known_keys.add(key_text)
            _collect_facts(item, known_keys=known_keys, known_values=known_values)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _collect_facts(item, known_keys=known_keys, known_values=known_values)
        return

    for normalized_value in _normalized_values(value):
        known_values.add(normalized_value)
    if isinstance(value, str):
        known_values.update(_number_tokens(value))


def _normalized_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, bool):
        return {str(value).lower()}
    if isinstance(value, int):
        return {str(value)}
    if isinstance(value, float):
        values = {str(value)}
        if value.is_integer():
            values.add(str(int(value)))
        return values
    return {str(value)}


def _priority_habit_preserved(
    *,
    priority_habit_id: str,
    generated_copy: str,
    generated_priority_habit_id: str | None,
) -> bool:
    if not priority_habit_id:
        return generated_priority_habit_id in (None, "")
    if priority_habit_id not in generated_copy:
        return False
    if generated_priority_habit_id is None:
        return True
    return generated_priority_habit_id == priority_habit_id


def _json_like(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _identifier_tokens(value: str) -> list[str]:
    return _IDENTIFIER_RE.findall(value)


def _number_tokens(value: str) -> list[str]:
    return _NUMBER_RE.findall(value)


def _unique_in_order(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
