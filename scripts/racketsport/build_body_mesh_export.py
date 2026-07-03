#!/usr/bin/env python3
"""CLI wrapper: extract a compact, scheduled-frames-only mesh export from a
full `smpl_motion.json` (which can be hundreds of MB, mostly redundant
joints_world/mesh_vertices_world for frames that were never scheduled for
world-mesh compute). Reuses `threed.racketsport.mesh_export.build_body_mesh_export`,
which already filters down to the `body_compute_execution.json` scheduled
(frame_idx, player_id) set.

Intended to run server-side (e.g. on the A100 VM where the full smpl_motion.json
lives) so only the compact output needs to be pulled to a laptop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mesh_export import build_body_mesh_export  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a compact, scheduled-frames-only body mesh export from a full smpl_motion.json."
    )
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--smpl-motion", type=Path, required=True, help="Full smpl_motion.json path.")
    parser.add_argument(
        "--body-compute-execution",
        type=Path,
        help="Optional body_compute_execution.json path (restricts output to scheduled frames).",
    )
    parser.add_argument("--faces-ref", default="mhr_faces_static", help="Mesh faces reference id.")
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=None,
        help=(
            "Optional decimal-place rounding applied to every float in the payload before writing. "
            "smpl_motion.json ships full float64 precision (~17 significant digits) as JSON text, which "
            "dominates file size (vertex position floats alone can be >100MB for a couple hundred "
            "scheduled frames of a dense mesh); rounding to a few decimals (millimeters, given this data "
            "is already 'world-scale preview' per the trust band, not metrically verified) shrinks the "
            "JSON substantially with no meaningful loss of fidelity. Omit to keep full precision."
        ),
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Write minified JSON (no indent/sort) instead of the default indent=2 formatting, to save "
        "further bytes on large exports. Has no effect on the printed summary.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output body_mesh_export.json path.")
    args = parser.parse_args(argv)

    try:
        smpl_motion = json.loads(args.smpl_motion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to read --smpl-motion: {exc}", file=sys.stderr)
        return 1

    body_compute_execution = None
    if args.body_compute_execution is not None:
        try:
            body_compute_execution = json.loads(args.body_compute_execution.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: failed to read --body-compute-execution: {exc}", file=sys.stderr)
            return 1

    payload = build_body_mesh_export(
        smpl_motion,
        clip=args.clip,
        body_compute_execution=body_compute_execution,
        faces_ref=args.faces_ref,
    )
    if args.round_decimals is not None:
        payload = _round_floats(payload, args.round_decimals)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.compact_json:
        args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    else:
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_bytes = args.out.stat().st_size
    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "out_bytes": out_bytes,
                "round_decimals": args.round_decimals,
                "compact_json": args.compact_json,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _round_floats(value: Any, decimals: int) -> Any:
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list):
        return [_round_floats(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item, decimals) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
