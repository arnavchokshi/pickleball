#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.shot_dataset_builder import (  # noqa: E402
    DEFAULT_MAX_CONTACT_DT_S,
    DEFAULT_WINDOW_MS,
    build_shot_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a DATA-5 shot-class dataset from reviewed labels.")
    parser.add_argument("--truth-events", type=Path, required=True, help="Human-reviewed shot event labels JSON.")
    parser.add_argument("--contact-windows", type=Path, required=True, help="Contact windows JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--max-contact-dt-s", type=float, default=DEFAULT_MAX_CONTACT_DT_S)
    args = parser.parse_args(argv)

    try:
        manifest = build_shot_dataset(
            dataset_id=args.dataset_id,
            clip_id=args.clip_id,
            truth_events_payload=_read_json_object(args.truth_events, "truth events"),
            contact_windows_payload=_read_json_object(args.contact_windows, "contact windows"),
            out_dir=args.out_dir,
            split=args.split,
            fps=args.fps,
            window_ms=args.window_ms,
            max_contact_dt_s=args.max_contact_dt_s,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: shot dataset build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
