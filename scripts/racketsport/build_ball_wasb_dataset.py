#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_wasb_dataset import (  # noqa: E402
    BLURBALL_DATASET_CONFIG_YAML,
    MANIFEST_JSON,
    MANIFEST_MD,
    build_ball_wasb_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a guarded TrackNet-layout BALL dataset into WASB-SBDT pickleball layout.",
    )
    parser.add_argument(
        "--tracknet-root",
        type=Path,
        required=True,
        help="Root produced by build_ball_tracknet_cvat_dataset.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for WASB-SBDT dataset files.")
    parser.add_argument(
        "--allow-internal-val",
        action="store_true",
        help="Permit Burlington/Wolverine only as validation-during-fitting entries; strict holdouts are still refused.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_ball_wasb_dataset(
            tracknet_root=args.tracknet_root,
            out_dir=args.out_dir,
            allow_internal_val=args.allow_internal_val,
        )
    except Exception as exc:
        print(f"BALL WASB dataset build failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_json": str(args.out_dir / MANIFEST_JSON),
                "manifest_md": str(args.out_dir / MANIFEST_MD),
                "dataset_config_path": manifest["dataset_config_path"],
                "blurball_dataset_config_path": str(args.out_dir / BLURBALL_DATASET_CONFIG_YAML),
                "label_counts": manifest["label_counts"],
                "splits": {
                    split: [row["clip"] for row in rows]
                    for split, rows in manifest["splits"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
