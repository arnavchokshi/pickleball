#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.pipeline_contracts import PipelineContractError, build_readiness_report
from threed.racketsport.pipeline_cli import build_public_contract_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CPU-only racket-sport pipeline artifact readiness.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing per-clip pipeline JSON artifacts.")
    parser.add_argument("--stage", default="e2e", help="Requested pipeline stage readiness target. Defaults to e2e.")
    parser.add_argument("--out", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--public-contracts",
        action="store_true",
        help="Validate the public run_pipeline artifact contracts instead of the legacy internal readiness contracts.",
    )
    args = parser.parse_args()

    try:
        if args.public_contracts:
            report = build_public_contract_readiness(args.run_dir, stage=args.stage)
        else:
            report = build_readiness_report(args.run_dir, stage=args.stage)
    except (PipelineContractError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
