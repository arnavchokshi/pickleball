from __future__ import annotations

import pytest

from tests.racketsport.json_schema_assertions import JsonSchemaAssertionError, assert_matches_json_schema


def test_json_schema_assertions_enforce_allof_and_conditional_requirements() -> None:
    schema = {
        "type": "object",
        "allOf": [{"required": ["kind"]}],
        "properties": {
            "kind": {"type": "string"},
            "pose_labels": {"type": "object"},
        },
        "if": {"properties": {"kind": {"const": "aruco_gt"}}, "required": ["kind"]},
        "then": {"required": ["pose_labels"]},
    }

    with pytest.raises(JsonSchemaAssertionError, match="pose_labels"):
        assert_matches_json_schema({"kind": "aruco_gt"}, schema)


def test_json_schema_assertions_enforce_prefix_items_and_contains() -> None:
    schema = {
        "type": "array",
        "prefixItems": [
            {"const": "top_left"},
            {"const": "top_right"},
            {"const": "bottom_right"},
            {"const": "bottom_left"},
        ],
        "items": {"type": "string"},
        "contains": {"const": "bottom_left"},
        "minContains": 1,
    }

    with pytest.raises(JsonSchemaAssertionError, match="top_right"):
        assert_matches_json_schema(["top_left", "bottom_right", "top_right", "bottom_left"], schema)

    with pytest.raises(JsonSchemaAssertionError, match="contains"):
        assert_matches_json_schema(["top_left", "top_right", "bottom_right", "handle"], schema)
