"""Invariants for the generated DATA_INVENTORY.md living page and its builder CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "racketsport" / "build_data_inventory.py"
DOC = ROOT / "DATA_INVENTORY.md"
LEDGER = ROOT / "runs" / "manager" / "data_ledger.json"


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DATA_INVENTORY.md" in result.stdout


def test_inventory_is_in_sync_with_ledger() -> None:
    # The living page must never drift from the ledger. If this fails, regenerate:
    #   .venv/bin/python scripts/racketsport/build_data_inventory.py
    result = subprocess.run(
        [sys.executable, str(CLI), "--check"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "DATA_INVENTORY.md is stale vs the ledger; regenerate with "
        "scripts/racketsport/build_data_inventory.py\n" + result.stderr
    )


def test_every_ledger_asset_appears_in_the_page() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    for asset in ledger["assets"]:
        assert asset["asset_id"] in doc, asset["asset_id"]


def test_page_is_marked_generated_and_points_at_the_authority() -> None:
    doc = DOC.read_text(encoding="utf-8")
    assert "GENERATED FILE" in doc
    assert "runs/manager/data_ledger.json" in doc
    assert "`VERIFIED=0`" in doc
