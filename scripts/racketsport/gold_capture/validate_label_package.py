#!/usr/bin/env python3
"""Validate every versioned gold-capture template against its paired JSON schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


class SchemaValidationError(ValueError):
    """Raised when a template fails the package's supported JSON Schema subset."""


def _is_type(instance: Any, expected: str) -> bool:
    return {
        "object": isinstance(instance, dict),
        "array": isinstance(instance, list),
        "string": isinstance(instance, str),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "boolean": isinstance(instance, bool),
        "null": instance is None,
    }[expected]


def validate_instance(instance: Any, schema: Mapping[str, Any], *, path: str = "$") -> None:
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_is_type(instance, item) for item in expected_types):
            raise SchemaValidationError(f"{path}: expected type {expected_types}, got {type(instance).__name__}")

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {instance!r} is not in {schema['enum']!r}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path}: unexpected keys {extras}")
        for key, value in instance.items():
            if key in properties:
                validate_instance(value, properties[key], path=f"{path}.{key}")

    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            raise SchemaValidationError(f"{path}: expected at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise SchemaValidationError(f"{path}: expected at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, value in enumerate(instance):
                validate_instance(value, item_schema, path=f"{path}[{index}]")

    if isinstance(instance, str):
        if len(instance) < int(schema.get("minLength", 0)):
            raise SchemaValidationError(f"{path}: string is shorter than {schema['minLength']}")
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), instance) is None:
            raise SchemaValidationError(f"{path}: value does not match {schema['pattern']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{path}: value is below {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"{path}: value is above {schema['maximum']}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{path}: top level must be an object")
    return payload


def _independent_gt_invariants(payload: Mapping[str, Any], *, path: Path) -> None:
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.startswith("gold_capture_"):
        return
    if artifact_type in {"gold_capture_candidate_license_card", "gold_capture_court_net_survey"}:
        return
    sample_ref = payload.get("sample_ref")
    if not isinstance(sample_ref, Mapping) or "frame_index" not in sample_ref or "pts_seconds" not in sample_ref:
        raise SchemaValidationError(f"{path}: label template lacks both frame_index and pts_seconds")
    source = payload.get("label_source")
    raw = source.get("immutable_raw_reference") if isinstance(source, Mapping) else None
    if not isinstance(raw, Mapping) or not raw.get("sha256") or raw.get("write_protected") is not True:
        raise SchemaValidationError(f"{path}: immutable raw reference is incomplete")
    independence = payload.get("independence")
    if not isinstance(independence, Mapping) or independence.get("candidate_prediction_used") is not False:
        raise SchemaValidationError(f"{path}: candidate predictions cannot be independent GT")
    uncertainty = payload.get("uncertainty")
    if not isinstance(uncertainty, Mapping) or "status" not in uncertainty:
        raise SchemaValidationError(f"{path}: uncertainty must be saved")
    review = payload.get("review")
    if not isinstance(review, Mapping) or not review.get("reviewer_id"):
        raise SchemaValidationError(f"{path}: reviewer identity is required")


def validate_package(*, schema_dir: Path, template_dir: Path) -> dict[str, Any]:
    schemas = sorted(schema_dir.glob("*.schema.json"))
    if not schemas:
        raise SchemaValidationError(f"{schema_dir}: no *.schema.json files")
    results: list[dict[str, str]] = []
    for schema_path in schemas:
        stem = schema_path.name.removesuffix(".schema.json")
        template_path = template_dir / f"{stem}.template.json"
        if not template_path.is_file():
            raise SchemaValidationError(f"{template_path}: paired template is missing")
        schema = _load_json(schema_path)
        payload = _load_json(template_path)
        validate_instance(payload, schema)
        _independent_gt_invariants(payload, path=template_path)
        results.append({"name": stem, "schema": schema_path.as_posix(), "template": template_path.as_posix(), "status": "pass"})
    extras = sorted(path.name for path in template_dir.glob("*.template.json") if not (schema_dir / path.name.replace(".template.json", ".schema.json")).is_file())
    if extras:
        raise SchemaValidationError(f"{template_dir}: unpaired templates {extras}")
    return {
        "schema_version": 1,
        "artifact_type": "gold_capture_schema_validation_report",
        "status": "pass",
        "validated_count": len(results),
        "results": results,
        "candidate_prediction_gt_kill_rule_enforced": True,
        "product_boundary": "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_package(schema_dir=args.schema_dir, template_dir=args.template_dir)
    except SchemaValidationError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

