from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CLI_PATH = "scripts/racketsport/synthetic_body_decode_gate.py"


def test_synthetic_body_decode_gate_cpu_mock_writes_metric_report_and_render(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_report.json"
    render_dir = tmp_path / "renders"

    completed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--out",
            str(out),
            "--render-dir",
            str(render_dir),
            "--samples",
            "2",
            "--decoder",
            "mock",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(out) in completed.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "racketsport_synthetic_body_decode_gate"
    assert report["measurement_status"] == "measured_mock_decoder"
    assert report["sample_count"] == 2
    assert report["gate_1b_world_round_trip"]["joints_world_p95_abs_error_mm"] <= 1.0
    assert report["mesh_skeleton_divergence"]["p95_mm"] <= 5.0
    assert len(report["renders"]) == 2
    for render in report["renders"]:
        assert (render_dir / render["path"]).is_file()


def test_synthetic_body_decode_gate_cli_help_and_scaffold_reference() -> None:
    help_completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--decoder" in help_completed.stdout
    assert "--samples" in help_completed.stdout

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    tools = {tool["command_path"]: tool for tool in payload["tools"]}
    entry = tools[CLI_PATH]
    assert entry["category"] == "decode"
    assert entry["direct_cli_reference_test"] == "tests/racketsport/test_synthetic_body_decode_gate.py"
