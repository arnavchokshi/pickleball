#!/usr/bin/env python3
"""W3-REPLAY-NATIVE phase 1: bake a `body_mesh.json` (`racketsport_body_mesh`)
into an animated `.usdz` package via `usd-core` (`pxr`). See
`threed/racketsport/replay_usdz_bake.py` for the design (time-sampled
`points` on a fixed-topology mesh + contact-window-gated `visibility`, so the
mesh only appears during the windows it was actually baked for). Optional
preview controls can cap the number of authored mesh point samples while
preserving each scheduled window's endpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.replay_usdz_bake import BodyMeshUsdzBakeError, build_animated_body_usdz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bake body_mesh.json into an animated USDZ package.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--body-mesh", type=Path, required=True, help="body_mesh.json path (racketsport_body_mesh).")
    parser.add_argument("--out", type=Path, required=True, help="Output .usdz path.")
    parser.add_argument(
        "--max-mesh-frames",
        type=int,
        default=None,
        help="Optional lossy preview budget for total mesh point samples across all players/windows.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=None,
        help="Optionally round mesh vertex coordinates before authoring USD float32 points.",
    )
    args = parser.parse_args(argv)

    try:
        body_mesh = json.loads(args.body_mesh.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to read --body-mesh: {exc}", file=sys.stderr)
        return 1

    try:
        summary = build_animated_body_usdz(
            body_mesh,
            clip=args.clip,
            out_path=args.out,
            max_mesh_frames=args.max_mesh_frames,
            round_decimals=args.round_decimals,
        )
    except BodyMeshUsdzBakeError as exc:
        print(f"ERROR: USDZ bake failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
