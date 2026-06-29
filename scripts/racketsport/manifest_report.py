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

from threed.racketsport.model_manifest import ModelManifest, load_model_manifest


def summarize_manifest(manifest: ModelManifest) -> dict[str, Any]:
    status_counts = Counter(entry.status for entry in manifest.models)
    posture_counts = Counter(entry.commercial_posture for entry in manifest.models)
    declared_inventory_statuses = {"available_on_h100", "available_runtime_on_h100"}
    checkpoint_files = [entry.id for entry in manifest.models if entry.status == "available_on_h100"]
    declared_inventory = [entry.id for entry in manifest.models if entry.status in declared_inventory_statuses]
    pending = [entry.id for entry in manifest.models if entry.status not in declared_inventory_statuses]
    return {
        "schema_version": 1,
        "total_models": len(manifest.models),
        "status_counts": dict(sorted(status_counts.items())),
        "commercial_posture_counts": dict(sorted(posture_counts.items())),
        "declared_h100_checkpoint_files": checkpoint_files,
        "declared_h100_inventory": declared_inventory,
        "missing_or_pending": pending,
    }


def render_markdown(manifest: ModelManifest, summary: dict[str, Any]) -> str:
    rows = [
        "# Model Manifest Report",
        "",
        f"- Total models: {summary['total_models']}",
        f"- Declared H100 inventory: {len(summary['declared_h100_inventory'])}",
        f"- Missing or pending: {len(summary['missing_or_pending'])}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in summary["status_counts"].items():
        rows.append(f"- `{status}`: {count}")

    rows.extend(["", "## Entries", "", "| ID | Status | Stage | Commercial posture |", "|---|---|---|---|"])
    for entry in manifest.models:
        rows.append(f"| `{entry.id}` | `{entry.status}` | {entry.stage} | `{entry.commercial_posture}` |")
    rows.append("")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize declared models/MANIFEST.json inventory.")
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout.")
    args = parser.parse_args()

    manifest = load_model_manifest(args.manifest)
    summary = summarize_manifest(manifest)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    markdown = render_markdown(manifest, summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
