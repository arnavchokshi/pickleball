#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_finalize import finalize_body_world_labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize reviewed BODY world-joint labels for world-MPJPE gates.")
    parser.add_argument("--template", type=Path, required=True, help="Reviewed body_world_joints.template.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output body_world_joints.json path.")
    parser.add_argument("--report-out", type=Path, help="Optional finalization report JSON path.")
    args = parser.parse_args(argv)

    try:
        report = finalize_body_world_labels(
            template_path=args.template,
            out_path=args.out,
            report_out_path=args.report_out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label finalization failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "finalized" else 1


if __name__ == "__main__":
    raise SystemExit(main())
