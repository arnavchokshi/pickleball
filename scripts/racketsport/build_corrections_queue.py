#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.validate_corrections import validate_manifest


def build_corrections_queue(manifests: list[str | Path]) -> dict[str, Any]:
    manifest_paths = [Path(manifest) for manifest in manifests]
    if not manifest_paths:
        raise ValueError("at least one corrections manifest is required")

    source_summaries: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for manifest_path in manifest_paths:
        summary = validate_manifest(manifest_path)
        payload = _load_json(manifest_path)
        manifest_id = summary["manifest_id"]
        source_summaries.append(summary)

        for correction in payload["corrections"]:
            correction_id = correction["id"]
            key = (manifest_id, correction_id)
            if key in seen_keys:
                raise ValueError(f"duplicate queued correction id: {manifest_id}/{correction_id}")
            seen_keys.add(key)

            target = correction["target"]
            queue.append(
                {
                    "manifest_id": manifest_id,
                    "correction_id": correction_id,
                    "operation": correction["operation"],
                    "artifact": target["artifact"],
                    "clip_id": target.get("clip_id"),
                    "frame_index": target.get("frame_index"),
                    "t_s": target.get("t_s"),
                    "path": target["path"],
                    "value": correction.get("value"),
                    "confidence": correction.get("confidence"),
                    "reason": correction["reason"],
                    "annotator": correction["annotator"],
                    "created_at": correction.get("created_at", payload["created_at"]),
                }
            )

    return {
        "schema_version": 1,
        "manifest_count": len(source_summaries),
        "correction_count": len(queue),
        "source_manifests": source_summaries,
        "summary": _summarize_queue(queue),
        "corrections": queue,
    }


def discover_correction_manifests(root: str | Path, *, pattern: str = "*.json") -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"{root_path} does not exist")
    if not root_path.is_dir():
        raise ValueError(f"{root_path} is not a directory")
    return sorted(path for path in root_path.glob(pattern) if path.is_file() and path.name != "schema.json")


def write_queue(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_queue(queue: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_operation": _counter_dict(item["operation"] for item in queue),
        "by_artifact": _counter_dict(item["artifact"] for item in queue),
        "by_clip": _counter_dict(item["clip_id"] or "unknown" for item in queue),
    }


def _counter_dict(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a flat EVAL-3 corrections queue artifact.")
    parser.add_argument("manifests", nargs="*", type=Path, help="Corrections manifest JSON files.")
    parser.add_argument("--root", type=Path, help="Directory containing corrections manifest JSON files.")
    parser.add_argument("--pattern", default="*.json", help="Manifest glob used with --root. Defaults to *.json.")
    parser.add_argument("--out", type=Path, default=Path("runs/corrections_queue/corrections_queue.json"))
    args = parser.parse_args(argv)

    if bool(args.root) == bool(args.manifests):
        parser.error("provide either manifest paths or --root, but not both")

    try:
        manifests = discover_correction_manifests(args.root, pattern=args.pattern) if args.root else args.manifests
        payload = build_corrections_queue(manifests)
        write_queue(args.out, payload)
    except ValueError as exc:
        print("ERROR: corrections queue build failed:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
