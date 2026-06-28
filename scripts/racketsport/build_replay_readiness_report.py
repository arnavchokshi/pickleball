#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.replay_readiness import (  # noqa: E402
    build_replay_readiness_report,
    write_replay_readiness_html,
    write_replay_readiness_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed replay/metrics readiness reports from real run artifacts.")
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing per-clip run artifact folders.")
    parser.add_argument("--clips", nargs="*", help="Optional clip ids. Defaults to run-root folders with virtual_world_paddle_preview.json.")
    parser.add_argument("--labels-root", type=Path, default=Path("data/testclips"), help="Optional DATA-1 labels root.")
    parser.add_argument("--out", type=Path, required=True, help="Output readiness JSON path.")
    parser.add_argument("--html-out", type=Path, help="Optional browser-reviewable HTML summary path.")
    args = parser.parse_args(argv)

    payload = build_replay_readiness_report(
        run_root=args.run_root,
        clips=args.clips,
        labels_root=args.labels_root,
    )
    write_replay_readiness_report(args.out, payload)
    if args.html_out is not None:
        write_replay_readiness_html(args.html_out, payload)

    print(json.dumps({"schema_version": 1, "status": payload["status"], "out": str(args.out), "html_out": str(args.html_out) if args.html_out else None}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
