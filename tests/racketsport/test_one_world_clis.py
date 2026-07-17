from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.racketsport.test_one_world_core import make_run


ROOT = Path(__file__).resolve().parents[2]
BUILD_CLI = "scripts/racketsport/build_one_world_v1.py"
METRICS_CLI = "scripts/racketsport/report_one_world_metrics.py"
VALIDATE_CLI = "scripts/racketsport/validate_one_world_v1.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=False)


def test_direct_clis_help_synthetic_run_and_exact_exit_codes(tmp_path: Path) -> None:
    for command in (BUILD_CLI, METRICS_CLI, VALIDATE_CLI):
        help_result = _run(command, "--help")
        assert help_result.returncode == 0, help_result.stderr
    run = make_run(tmp_path / "run")
    artifact = tmp_path / "one_world_v1.json"
    build = _run(BUILD_CLI, "--run-dir", str(run), "--out", str(artifact))
    assert build.returncode == 0, build.stderr
    validation = tmp_path / "validation.json"
    validate = _run(VALIDATE_CLI, "--artifact", str(artifact), "--run-dir", str(run), "--out", str(validation))
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validation.read_text())["valid"] is True
    metrics = tmp_path / "metrics.json"
    report = _run(METRICS_CLI, "--run-dir", str(run), "--fused", str(artifact), "--out", str(metrics))
    assert report.returncode == 0, report.stderr
    assert json.loads(metrics.read_text())["VERIFIED"] == 0
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{}")
    invalid = _run(VALIDATE_CLI, "--artifact", str(malformed), "--run-dir", str(run))
    assert invalid.returncode == 1
    missing = _run(BUILD_CLI, "--run-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "x.json"))
    assert missing.returncode == 2


def test_scaffold_index_has_related_direct_schema_and_task_entries() -> None:
    result = _run("scripts/racketsport/list_scaffold_tools.py", "--root", ".")
    assert result.returncode == 0, result.stderr
    tools = {row["stem"]: row for row in json.loads(result.stdout)["tools"]}
    expected = {
        "build_one_world_v1": ("WORLD", "NS-04.6", "one_world_v1_schema.json"),
        "report_one_world_metrics": ("EVAL", "NS-04.5", "one_world_v1_metrics_schema.json"),
        "validate_one_world_v1": ("WORLD", "NS-04.5", "one_world_v1_validation_schema.json"),
    }
    for stem, (workstream, task, schema) in expected.items():
        row = tools[stem]
        assert row["related_test"].endswith("test_one_world_clis.py")
        assert row["direct_cli_reference_test"].endswith("test_one_world_clis.py")
        assert row["matching_schema"].endswith(schema)
        assert (row["workstream"], row["task_prefix"]) == (workstream, task)
