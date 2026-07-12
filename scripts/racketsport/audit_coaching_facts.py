#!/usr/bin/env python3
"""Audit deterministic coaching facts for source fidelity and safe authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.coaching_fact_audit import audit_coaching_facts_file


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    facts_path = Path(args.facts)
    if not facts_path.is_file():
        print(f"ERROR: facts artifact does not exist: {facts_path}", file=sys.stderr)
        return 1
    try:
        report = audit_coaching_facts_file(facts_path, manifest_path=args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.report is not None:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "pass" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path, help="Path to coaching_card_facts.json.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional replay_viewer_manifest.json used to enforce facts-before-manifest file order.",
    )
    parser.add_argument("--report", type=Path, help="Optional path for the deterministic audit JSON report.")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
