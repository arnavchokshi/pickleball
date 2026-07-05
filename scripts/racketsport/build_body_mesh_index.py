#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_mesh_index import (  # noqa: E402
    DEFAULT_QUANTIZATION_SCALE,
    build_body_mesh_index,
    build_body_mesh_index_cli_summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a monolithic body_mesh.json into replay-viewer body_mesh_index chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("clip_dir_or_body_mesh_json", help="Clip directory containing body_mesh.json, or the body_mesh.json path itself.")
    parser.add_argument("--out-dir", required=True, help="Directory for body_mesh_index.json, body_mesh_faces.json, and body_mesh_chunks/.")
    parser.add_argument(
        "--quantization-scale",
        type=int,
        default=DEFAULT_QUANTIZATION_SCALE,
        help="Int16 quantization scale in units per meter. 1000 preserves millimeter-scale coordinates within +/-32.767m.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_body_mesh_index(
        args.clip_dir_or_body_mesh_json,
        out_dir=args.out_dir,
        quantization_scale=args.quantization_scale,
    )
    print(json.dumps(build_body_mesh_index_cli_summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
