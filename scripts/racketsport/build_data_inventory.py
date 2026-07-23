#!/usr/bin/env python3
"""Generate DATA_INVENTORY.md — the scannable, per-lane view of ALL registered data.

The authority is ``runs/manager/data_ledger.json``. This tool renders it into one
human-readable page at the repo root so any agent (or the owner) can see, at a glance,
what data exists on every lane and whether it is used. DO NOT hand-edit the output;
regenerate it instead:

    .venv/bin/python scripts/racketsport/build_data_inventory.py

``--check`` re-renders and exits non-zero if the committed DATA_INVENTORY.md is stale
(used by the doc-invariant test so the living page never drifts from the ledger).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "runs" / "manager" / "data_ledger.json"
OUTPUT = ROOT / "DATA_INVENTORY.md"

LANE_BY_TRACK = {"A": "COURT", "B": "BALL", "C": "PERSON", "D": "EVENT"}
LANE_ORDER = ["COURT", "BALL", "PERSON", "EVENT", "SHARED / OTHER"]

STATUS_USED_TRAIN = "✅ used (training)"
STATUS_USED_AUDIT = "\U0001f4ca used (audit/eval only)"
STATUS_READY = "\U0001f7e1 authorized — not yet trained"
STATUS_BLOCKED = "\U0001f534 not used — blocked"
STATUS_HELDOUT = "\U0001f512 held out (eval/protected)"
STATUS_PARKED = "⏸ parked"
STATUS_REJECTED = "❌ ruled out"

SUMMARY_COLUMNS = ["used", "authorized", "blocked", "held-out", "rejected"]


def _status(asset: dict[str, Any]) -> tuple[str, str]:
    state = asset.get("state", "")
    training = asset.get("disposition", {}).get("training_intent")
    if state == "CONSUMED":
        if training is True:
            return STATUS_USED_TRAIN, "used"
        return STATUS_USED_AUDIT, "used"
    if state == "READY":
        return STATUS_READY, "authorized"
    if state == "BLOCKED":
        return STATUS_BLOCKED, "blocked"
    if state == "QUARANTINED":
        return STATUS_HELDOUT, "held-out"
    if state == "DEFERRED_WITH_REASON":
        return STATUS_PARKED, "blocked"
    if state == "REJECTED":
        return STATUS_REJECTED, "rejected"
    return state or "—", "blocked"


def _lane(asset: dict[str, Any]) -> str:
    track = asset.get("disposition", {}).get("consumer_track")
    return LANE_BY_TRACK.get(track, "SHARED / OTHER")


def _size(asset: dict[str, Any]) -> str:
    counts = asset.get("counts", {})
    for key, unit in (
        ("label_count", "labels"),
        ("dedup_kept_count", "rows"),
        ("frame_count", "frames"),
    ):
        val = counts.get(key)
        if isinstance(val, int) and val > 0:
            return f"{val:,} {unit}"
    byte_count = counts.get("byte_count")
    if isinstance(byte_count, int) and byte_count > 0:
        return f"{byte_count / 1e6:.0f} MB"
    return "—"


def _friendly(asset_id: str) -> str:
    parts = asset_id.split("_")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        parts = parts[:-1]
    return " ".join(parts)


def _source(asset: dict[str, Any]) -> str:
    channels = asset.get("source_lineage", {}).get("channels") or []
    if channels:
        return str(channels[0])
    return _friendly(asset["asset_id"])


def _clip(text: str | None, width: int = 150) -> str:
    if not text:
        return "—"
    text = " ".join(str(text).split())
    if len(text) > width:
        text = text[: width - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def _why_next(asset: dict[str, Any]) -> str:
    disp = asset.get("disposition", {})
    nub = disp.get("not_usable_because")
    if nub:
        return _clip(nub)
    return _clip(disp.get("next_queue_action") or asset.get("state_reason"))


def render(ledger: dict[str, Any]) -> str:
    assets = ledger.get("assets", [])
    by_lane: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_ORDER}
    summary: dict[str, dict[str, int]] = {
        lane: {col: 0 for col in SUMMARY_COLUMNS} for lane in LANE_ORDER
    }
    for asset in assets:
        lane = _lane(asset)
        by_lane[lane].append(asset)
        _, bucket = _status(asset)
        summary[lane][bucket] += 1

    lines: list[str] = []
    lines.append("# DinkVision Data Inventory")
    lines.append("")
    lines.append(
        "> **GENERATED FILE — do not hand-edit.** Authority is "
        "`runs/manager/data_ledger.json`; this is its scannable human view."
    )
    lines.append(
        "> Regenerate after any ledger change: "
        "`.venv/bin/python scripts/racketsport/build_data_inventory.py`"
    )
    lines.append(
        f"> Ledger generated: `{ledger.get('generated_utc', '?')}` · "
        f"{len(assets)} registered datasets · `VERIFIED=0` binding."
    )
    lines.append("")
    lines.append(
        "The single place to see **all data we have on every lane and whether it is "
        "used**. Every dataset the project has touched is a registered asset here with a "
        "state and a reason. If a dataset is not in this table, it is not registered — add "
        "a ledger row before any training touches it (the data-safety gate enforces this)."
    )
    lines.append("")
    lines.append(
        "**Legend** — "
        f"{STATUS_USED_TRAIN} · {STATUS_USED_AUDIT} · {STATUS_READY} · "
        f"{STATUS_BLOCKED} · {STATUS_HELDOUT} · {STATUS_PARKED} · {STATUS_REJECTED}"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Lane | ✅ used | \U0001f7e1 authorized | \U0001f534 blocked | \U0001f512 held-out | ❌ rejected |")
    lines.append("|---|---|---|---|---|---|")
    totals = {col: 0 for col in SUMMARY_COLUMNS}
    for lane in LANE_ORDER:
        row = summary[lane]
        if not by_lane[lane]:
            continue
        for col in SUMMARY_COLUMNS:
            totals[col] += row[col]
        lines.append(
            f"| **{lane}** | {row['used']} | {row['authorized']} | "
            f"{row['blocked']} | {row['held-out']} | {row['rejected']} |"
        )
    lines.append(
        f"| **TOTAL** | {totals['used']} | {totals['authorized']} | "
        f"{totals['blocked']} | {totals['held-out']} | {totals['rejected']} |"
    )
    lines.append("")

    for lane in LANE_ORDER:
        lane_assets = by_lane[lane]
        if not lane_assets:
            continue
        lines.append(f"## {lane}")
        lines.append("")
        lines.append("| Dataset | What / source | Size | Status | Why not used / next step |")
        lines.append("|---|---|---|---|---|")
        order = {
            STATUS_USED_TRAIN: 0, STATUS_USED_AUDIT: 1, STATUS_READY: 2,
            STATUS_BLOCKED: 3, STATUS_HELDOUT: 4, STATUS_PARKED: 5, STATUS_REJECTED: 6,
        }
        for asset in sorted(lane_assets, key=lambda a: (order.get(_status(a)[0], 9), a["asset_id"])):
            status, _ = _status(asset)
            lines.append(
                f"| `{asset['asset_id']}` | {_clip(_source(asset), 60)} | "
                f"{_size(asset)} | {status} | {_why_next(asset)} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        "_Not sure why a dataset is blocked or held out? The full provenance, hashes, "
        "partitions, and rulings live in `runs/manager/data_ledger.json` (per-asset) and "
        "`runs/manager/DATA_LEDGER.md` (the audit view)._"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the output file is stale vs the ledger (no write).",
    )
    args = parser.parse_args(argv)

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    rendered = render(ledger)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            print(
                f"STALE: {args.output.name} is out of sync with the ledger. "
                "Regenerate: .venv/bin/python scripts/racketsport/build_data_inventory.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.output.name} is in sync with the ledger.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} ({len(ledger.get('assets', []))} assets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
