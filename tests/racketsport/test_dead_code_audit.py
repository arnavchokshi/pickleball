from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dead_code_audit_has_no_unknown_python_source_surfaces() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/audit_dead_code.py",
            "--root",
            ".",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["artifact_type"] == "racketsport_dead_code_reference_audit"
    assert payload["status"] == "pass"
    assert payload["summary"]["unknown_python_sources"] == 0
    assert payload["summary"]["python_sources"] == len(payload["python_sources"])
    assert payload["unknown_python_sources"] == []


def test_tech_stack_points_agents_to_dead_code_audit() -> None:
    text = (ROOT / "TECH_STACK.md").read_text(encoding="utf-8")

    assert "scripts/racketsport/audit_dead_code.py" in text
    assert "dead-code candidate" in text
