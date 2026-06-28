#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.shot_trainable_baseline import train_shot_window_baseline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CPU-only shot-window centroid baseline.")
    parser.add_argument("--manifest", type=Path, required=True, help="DATA-5 shot_dataset_manifest.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output metrics JSON path.")
    args = parser.parse_args(argv)

    try:
        payload = train_shot_window_baseline(manifest_path=args.manifest)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: shot training baseline failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "status": payload["status"],
                "test_accuracy": payload["splits"]["test"]["accuracy"],
                "test_macro_f1": payload["splits"]["test"]["macro_f1"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
