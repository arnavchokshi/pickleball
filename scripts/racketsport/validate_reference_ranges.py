#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_SCHEMA = Path("docs/racketsport/reference_ranges_schema.json")
DEFAULT_RANGES = Path("docs/racketsport/reference_ranges_v0.json")
SKILL_ORDER = {"3.0": 0, "3.5": 1, "4.0": 2, "4.5+": 3}
KNOWN_UNITS = {"fraction", "shots", "seconds", "meters", "body_height_fraction"}
SOURCED_TIERS = {"measured", "trade_benchmark", "coach_estimate"}


def validate_reference_ranges(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    errors = _schema_errors(payload, schema, root=schema, path="$")
    errors.extend(_semantic_errors(payload))
    ranges = payload.get("ranges", [])
    if not isinstance(ranges, list):
        ranges = []

    summary = {
        "range_count": len(ranges),
        "metric_count": len({entry.get("metric_id") for entry in ranges if isinstance(entry, dict)}),
        "metric_family_count": len({entry.get("metric_family") for entry in ranges if isinstance(entry, dict)}),
        "skill_band_count": len({entry.get("skill_band") for entry in ranges if isinstance(entry, dict)}),
        "trade_or_measured_count": sum(
            1
            for entry in ranges
            if isinstance(entry, dict)
            and isinstance(entry.get("provenance"), dict)
            and entry["provenance"].get("tier") in {"measured", "trade_benchmark"}
        ),
        "placeholder_unverified_count": sum(
            1
            for entry in ranges
            if isinstance(entry, dict)
            and isinstance(entry.get("provenance"), dict)
            and entry["provenance"].get("tier") == "placeholder_unverified"
        ),
    }
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "summary": summary,
    }


def _semantic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ranges = payload.get("ranges")
    if not isinstance(ranges, list):
        return errors

    seen: set[tuple[str, str]] = set()
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, entry in enumerate(ranges):
        if not isinstance(entry, dict):
            continue
        path = f"ranges[{index}]"
        metric_id = entry.get("metric_id")
        skill_band = entry.get("skill_band")
        if isinstance(metric_id, str) and isinstance(skill_band, str):
            key = (metric_id, skill_band)
            if key in seen:
                errors.append(f"{path}: duplicate metric_id+skill_band {metric_id}+{skill_band}")
            seen.add(key)
            by_metric[metric_id].append(entry)

        range_band = entry.get("range")
        blueprint_band = entry.get("band")
        if range_band != blueprint_band:
            errors.append(f"{path}: range and band must match exactly")
        if isinstance(range_band, dict):
            errors.extend(_band_errors(range_band, path=f"{path}.range", provenance=entry.get("provenance")))

        percentiles = entry.get("percentile_bands")
        if isinstance(percentiles, dict):
            values = [percentiles.get(key) for key in ("p10", "p50", "p90")]
            if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
                if not values[0] <= values[1] <= values[2]:
                    errors.append(f"{path}.percentile_bands: p10 <= p50 <= p90 required")

        provenance = entry.get("provenance")
        if isinstance(provenance, dict):
            errors.extend(_provenance_errors(provenance, path=f"{path}.provenance", range_band=range_band))

    for metric_id, entries in by_metric.items():
        numeric_entries = [
            entry
            for entry in entries
            if isinstance(entry.get("range"), dict)
            and isinstance(entry["range"].get("lo"), int | float)
            and isinstance(entry["range"].get("hi"), int | float)
        ]
        if len(numeric_entries) < 2:
            continue
        ordered = sorted(numeric_entries, key=lambda entry: SKILL_ORDER.get(str(entry.get("skill_band")), 999))
        direction = ordered[0]["range"].get("direction")
        if direction == "higher_better":
            previous_hi = None
            for entry in ordered:
                lo = float(entry["range"]["lo"])
                hi = float(entry["range"]["hi"])
                if previous_hi is not None and hi < previous_hi:
                    errors.append(f"ranges[{metric_id}]: higher_better bands must not decrease with skill")
                    break
                previous_hi = max(previous_hi if previous_hi is not None else lo, hi)
        elif direction == "lower_better":
            previous_hi = None
            for entry in ordered:
                hi = float(entry["range"]["hi"])
                if previous_hi is not None and hi > previous_hi:
                    errors.append(f"ranges[{metric_id}]: lower_better bands must not increase with skill")
                    break
                previous_hi = hi

    return errors


def _band_errors(band: dict[str, Any], *, path: str, provenance: Any) -> list[str]:
    errors: list[str] = []
    unit = band.get("unit")
    if unit not in KNOWN_UNITS:
        errors.append(f"{path}: unknown unit {unit!r}")
    lo = band.get("lo")
    hi = band.get("hi")
    tier = provenance.get("tier") if isinstance(provenance, dict) else None
    if lo is None or hi is None:
        if tier != "placeholder_unverified":
            errors.append(f"{path}: null bounds are allowed only for placeholder_unverified ranges")
        return errors
    if not _is_number(lo) or not _is_number(hi):
        return errors
    if lo > hi:
        errors.append(f"{path}: range.lo > range.hi ({lo} > {hi})")
    if unit == "fraction" and (lo < 0.0 or hi > 1.0):
        errors.append(f"{path}: fraction bounds must stay within [0, 1]")
    return errors


def _provenance_errors(provenance: dict[str, Any], *, path: str, range_band: Any) -> list[str]:
    errors: list[str] = []
    tier = provenance.get("tier")
    source = provenance.get("source")
    url = provenance.get("url")
    notes = provenance.get("notes")
    if tier in SOURCED_TIERS:
        if not isinstance(source, str) or not source.strip():
            errors.append(f"{path}: sourced range requires provenance.source")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            errors.append(f"{path}: sourced range requires an http(s) provenance.url")
    elif tier == "placeholder_unverified":
        if url is not None:
            errors.append(f"{path}: placeholder_unverified must not attach a source URL")
        if source is not None:
            errors.append(f"{path}: placeholder_unverified must not attach a source label")
        if not isinstance(notes, str) or not re.search(r"(no credible|unverified|needs)", notes, re.IGNORECASE):
            errors.append(f"{path}: placeholder_unverified notes must state the source gap")
        if isinstance(range_band, dict) and (range_band.get("lo") is not None or range_band.get("hi") is not None):
            errors.append(f"{path}: placeholder_unverified ranges must use null bounds in v0")
    return errors


def _schema_errors(instance: Any, schema: dict[str, Any] | bool, *, root: dict[str, Any], path: str) -> list[str]:
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: schema is false"]
    if "$ref" in schema:
        return _schema_errors(instance, _resolve_ref(root, str(schema["$ref"])), root=root, path=path)

    errors: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {instance!r}")
    schema_type = schema.get("type")
    if schema_type is not None and not _matches_type(instance, schema_type):
        errors.append(f"{path}: expected type {schema_type!r}, got {type(instance).__name__}")
        return errors

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            for key in extra:
                errors.append(f"{path}: unexpected key {key!r}")
        for key, child_schema in properties.items():
            if key in instance:
                errors.extend(_schema_errors(instance[key], child_schema, root=root, path=f"{path}.{key}"))

    if isinstance(instance, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(instance) < int(min_items):
            errors.append(f"{path}: expected at least {min_items} items, got {len(instance)}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                errors.extend(_schema_errors(item, item_schema, root=root, path=f"{path}[{index}]"))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: expected minLength {schema['minLength']}, got {len(instance)}")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            errors.append(f"{path}: value {instance!r} does not match pattern {schema['pattern']!r}")
        if "format" in schema and not _matches_format(instance, str(schema["format"])):
            errors.append(f"{path}: value {instance!r} does not match format {schema['format']!r}")

    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: expected minimum {schema['minimum']}, got {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: expected maximum {schema['maximum']}, got {instance}")
    return errors


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | bool:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported schema ref: {ref}")
    current: Any = root
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def _matches_type(instance: Any, schema_type: str | list[str]) -> bool:
    if isinstance(schema_type, list):
        return any(_matches_type(instance, child_type) for child_type in schema_type)
    if schema_type == "object":
        return isinstance(instance, dict)
    if schema_type == "array":
        return isinstance(instance, list)
    if schema_type == "string":
        return isinstance(instance, str)
    if schema_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if schema_type == "number":
        return _is_number(instance)
    if schema_type == "boolean":
        return isinstance(instance, bool)
    if schema_type == "null":
        return instance is None
    raise ValueError(f"unsupported schema type: {schema_type}")


def _matches_format(instance: str, schema_format: str) -> bool:
    if schema_format == "uri":
        return bool(urlparse(instance).scheme)
    if schema_format == "date-time":
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}T", instance))
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pickleball coaching reference-range library against its schema and semantic gates."
    )
    parser.add_argument("--ranges", type=Path, default=DEFAULT_RANGES, help="Reference range JSON to validate.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="JSON schema to validate against.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable validation report.")
    args = parser.parse_args(argv)

    try:
        payload = _load_json(args.ranges)
        schema = _load_json(args.schema)
    except (OSError, json.JSONDecodeError) as exc:
        report = {"status": "fail", "errors": [str(exc)], "summary": {}}
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate_reference_ranges(payload, schema)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "pass":
        summary = report["summary"]
        print(
            "PASS reference ranges: "
            f"{summary['range_count']} ranges, "
            f"{summary['metric_family_count']} families, "
            f"{summary['trade_or_measured_count']} trade/measured entries"
        )
    else:
        print("FAIL reference ranges:", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
