#!/usr/bin/env python3
"""Build a deterministic plant-aware rigid court-frame placement trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.placement_trajectory_refine import (
    PlacementTrajectoryConfig,
    PlacementTrajectoryError,
    read_json_object,
    refine_placement_trajectory,
    sha256_file,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--phases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weight-field", choices=("trk_weight", "plant_weight", "smoothness_weight"))
    parser.add_argument("--weight-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        config = PlacementTrajectoryConfig()
        if args.weight_field is not None or args.weight_scale != 1.0:
            if args.weight_field is None:
                raise PlacementTrajectoryError("--weight-scale requires --weight-field")
            config = config.scaled(args.weight_field, args.weight_scale)
        payload = refine_placement_trajectory(
            read_json_object(args.skeleton),
            tracks_payload=read_json_object(args.tracks),
            foot_contact_phases=read_json_object(args.phases),
            config=config,
        )
        payload["placement_trajectory_refinement"]["provenance"] = {
            "inputs": {
                "skeleton3d": {"path": str(args.skeleton.resolve()), "sha256": sha256_file(args.skeleton)},
                "tracks": {"path": str(args.tracks.resolve()), "sha256": sha256_file(args.tracks)},
                "foot_contact_phases": {"path": str(args.phases.resolve()), "sha256": sha256_file(args.phases)},
            },
            "config": config.to_dict(),
            "code_version": "trackI_placefuse_20260716_schema_v1",
            "coordinate_space": "court_Z0",
            "typed_coordinate_space": "world_court_netcenter_z_up_m",
            "distortion_state": "not_applicable_no_image_transform_in_refiner",
            "preview_band": True,
            "VERIFIED": 0,
        }
    except (OSError, PlacementTrajectoryError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "summary": payload["placement_trajectory_refinement"]["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
