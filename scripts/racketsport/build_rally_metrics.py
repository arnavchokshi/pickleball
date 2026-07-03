#!/usr/bin/env python3
"""Build per-rally per-player positional metrics for coaching-card facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.rally_metrics import build_rally_metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if not run_dir.exists():
        print(f"ERROR: run dir does not exist: {run_dir}", file=sys.stderr)
        return 1
    if not run_dir.is_dir():
        print(f"ERROR: run dir is not a directory: {run_dir}", file=sys.stderr)
        return 1

    try:
        payload = build_rally_metrics(run_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    rally_metrics_path = out_dir / "rally_metrics.json"
    facts_path = out_dir / "coaching_card_facts.json"
    _write_json(rally_metrics_path, payload)
    _write_json(facts_path, payload["coaching_card_facts"])
    print(
        json.dumps(
            {
                "rally_metrics": str(rally_metrics_path),
                "coaching_card_facts": str(facts_path),
                "rally_scope": payload["rally_scope"],
                "rally_count": len(payload["rallies"]),
                "player_count": payload["player_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Input run directory containing virtual_world.json.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for rally_metrics.json and coaching facts.")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
