#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.owner_capture_intake import (  # noqa: E402
    DEFAULT_OWNER_CORPUS_MANIFEST,
    DEFAULT_OWNER_DATA_MANIFEST,
    apply_reviewed_cvat_export,
    ingest_owner_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register owner capture packages and reviewed CVAT exports.")
    parser.add_argument("input", type=Path, nargs="?", help="Capture package dir (clip.mov + capture_sidecar.json) or bare video.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OWNER_DATA_MANIFEST)
    parser.add_argument("--reviewed-cvat-export", type=Path, default=None)
    parser.add_argument("--capture-id", default=None, help="Capture id to flip when --reviewed-cvat-export is used without input.")
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_OWNER_CORPUS_MANIFEST)
    args = parser.parse_args()

    try:
        result: dict[str, object] = {}
        capture_id = args.capture_id
        if args.input is not None:
            result["ingest"] = ingest_owner_capture(args.input, manifest_path=args.manifest)
            capture_id = str(result["ingest"]["capture_id"])  # type: ignore[index]
        if args.reviewed_cvat_export is not None:
            if not capture_id:
                raise ValueError("--reviewed-cvat-export requires --capture-id when no input is provided")
            result["review"] = apply_reviewed_cvat_export(
                capture_id,
                reviewed_export_path=args.reviewed_cvat_export,
                manifest_path=args.manifest,
                corpus_manifest_path=args.corpus_manifest,
            )
        if not result:
            raise ValueError("provide an input package/video and/or --reviewed-cvat-export")
    except Exception as exc:
        print(f"owner capture ingest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
