#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.validate_corrections import validate_manifest
from threed.racketsport.corrections import (
    build_training_manifest_candidates,
    enriched_queue_item,
    summarize_corrections_queue,
)


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

            queue.append(enriched_queue_item(manifest_id, correction, payload["created_at"]))

    return {
        "schema_version": 1,
        "manifest_count": len(source_summaries),
        "correction_count": len(queue),
        "source_manifests": source_summaries,
        "summary": summarize_corrections_queue(queue),
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
    write_json(path, payload)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--training-manifest-out", type=Path, help="Optional accepted-correction training candidates JSON.")
    parser.add_argument(
        "--training-candidate-root",
        default="training/corrections",
        help="Relative root recorded in candidate_path fields. Defaults to training/corrections.",
    )
    args = parser.parse_args(argv)

    if bool(args.root) == bool(args.manifests):
        parser.error("provide either manifest paths or --root, but not both")

    try:
        manifests = discover_correction_manifests(args.root, pattern=args.pattern) if args.root else args.manifests
        payload = build_corrections_queue(manifests)
        write_queue(args.out, payload)
        if args.training_manifest_out:
            training_payload = build_training_manifest_candidates(
                payload, candidate_root=args.training_candidate_root
            )
            write_json(args.training_manifest_out, training_payload)
    except ValueError as exc:
        print("ERROR: corrections queue build failed:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
