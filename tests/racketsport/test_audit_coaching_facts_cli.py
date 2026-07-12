from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.racketsport.test_coaching_facts import _write_fact_run
from threed.racketsport.rally_metrics import build_rally_metrics


def test_audit_coaching_facts_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/audit_coaching_facts.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--facts" in completed.stdout
    assert "--manifest" in completed.stdout
    assert "--report" in completed.stdout


def test_audit_coaching_facts_cli_writes_passing_report(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/audit_coaching_facts.py"
    run_dir = _write_fact_run(tmp_path / "run")
    facts_path = tmp_path / "coaching_card_facts.json"
    report_path = tmp_path / "coaching_fact_audit.json"
    payload = build_rally_metrics(run_dir)["coaching_card_facts"]
    facts_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--facts",
            str(facts_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["verdict"] == "pass"
    assert json.loads(report_path.read_text(encoding="utf-8"))["verdict"] == "pass"


def test_audit_coaching_facts_cli_rejects_mutated_number(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/audit_coaching_facts.py"
    run_dir = _write_fact_run(tmp_path / "run")
    facts_path = tmp_path / "coaching_card_facts.json"
    payload = build_rally_metrics(run_dir)["coaching_card_facts"]
    payload["audited_facts"][0]["interval"]["pts_end_s"] += 1.0
    facts_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, command_path, "--facts", str(facts_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "canonical_mismatch" in json.loads(completed.stdout)["issue_codes"]
