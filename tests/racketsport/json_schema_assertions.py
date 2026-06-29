from __future__ import annotations

import re
from typing import Any


class JsonSchemaAssertionError(AssertionError):
    pass


def assert_matches_json_schema(instance: Any, schema: dict[str, Any]) -> None:
    errors = list(_validate(instance, schema, root=schema, path="$"))
    if errors:
        raise JsonSchemaAssertionError("\n".join(errors))


def _validate(instance: Any, schema: dict[str, Any] | bool, *, root: dict[str, Any], path: str) -> list[str]:
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: schema is false"]
    if "$ref" in schema:
        return _validate(instance, _resolve_ref(root, str(schema["$ref"])), root=root, path=path)

    errors: list[str] = []
    if "oneOf" in schema:
        matches = sum(1 for option in schema["oneOf"] if not _validate(instance, option, root=root, path=path))
        if matches != 1:
            errors.append(f"{path}: expected exactly one oneOf match, got {matches}")
        return errors
    if "anyOf" in schema:
        matches = sum(1 for option in schema["anyOf"] if not _validate(instance, option, root=root, path=path))
        if matches == 0:
            errors.append(f"{path}: expected at least one anyOf match")
        return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
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
                errors.extend(_validate(instance[key], child_schema, root=root, path=f"{path}.{key}"))
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key, value in instance.items():
                if key not in properties:
                    errors.extend(_validate(value, additional, root=root, path=f"{path}.{key}"))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            errors.append(f"{path}: expected at least {schema['minItems']} items, got {len(instance)}")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            errors.append(f"{path}: expected at most {schema['maxItems']} items, got {len(instance)}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                errors.extend(_validate(item, item_schema, root=root, path=f"{path}[{index}]"))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: expected minLength {schema['minLength']}, got {len(instance)}")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            errors.append(f"{path}: value {instance!r} does not match pattern {schema['pattern']!r}")

    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: expected minimum {schema['minimum']}, got {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: expected maximum {schema['maximum']}, got {instance}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: expected exclusiveMinimum {schema['exclusiveMinimum']}, got {instance}")

    return errors


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | bool:
    if not ref.startswith("#/"):
        raise JsonSchemaAssertionError(f"unsupported ref: {ref}")
    current: Any = root
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def _matches_type(instance: Any, schema_type: str | list[str]) -> bool:
    if isinstance(schema_type, list):
        return any(_matches_type(instance, item) for item in schema_type)
    if schema_type == "object":
        return isinstance(instance, dict)
    if schema_type == "array":
        return isinstance(instance, list)
    if schema_type == "string":
        return isinstance(instance, str)
    if schema_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if schema_type == "number":
        return isinstance(instance, int | float) and not isinstance(instance, bool)
    if schema_type == "boolean":
        return isinstance(instance, bool)
    if schema_type == "null":
        return instance is None
    raise JsonSchemaAssertionError(f"unsupported schema type: {schema_type}")
