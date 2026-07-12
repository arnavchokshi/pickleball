#!/usr/bin/env python3
"""Write a normalized, exhaustive path/type inventory for PB export JSONs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    raise TypeError(type(value))


def inventory(payload: Any) -> dict[str, Any]:
    leaves: dict[str, Counter[str]] = defaultdict(Counter)
    arrays: dict[str, list[int]] = defaultdict(list)
    objects: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            objects[path or "$"][tuple(sorted(value))] += 1
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
            return
        if isinstance(value, list):
            arrays[path or "$"].append(len(value))
            for child in value:
                visit(child, f"{path}.[]" if path else "[]")
            return
        leaves[path or "$"].update([scalar_type(value)])

    visit(payload, "")
    return {
        "leaf_paths": [
            {"path": path, "types": dict(sorted(counts.items())), "count": sum(counts.values())}
            for path, counts in sorted(leaves.items())
        ],
        "array_paths": [
            {
                "path": path,
                "occurrences": len(lengths),
                "length_min": min(lengths),
                "length_max": max(lengths),
                "length_total": sum(lengths),
            }
            for path, lengths in sorted(arrays.items())
        ],
        "object_key_shapes": [
            {
                "path": path,
                "shapes": [
                    {"keys": list(keys), "occurrences": count}
                    for keys, count in sorted(shapes.items(), key=lambda item: (item[0], item[1]))
                ],
            }
            for path, shapes in sorted(objects.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "artifact_type": "pbvision_export_schema_inventory",
        "schema_version": 1,
        "files": {},
    }
    for path in args.inputs:
        payload = json.loads(path.read_text())
        result["files"][path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "top_level_type": type(payload).__name__,
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else None,
            "inventory": inventory(payload),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
