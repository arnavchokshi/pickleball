from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.racketsport.json_schema_assertions import assert_matches_json_schema

EXPECTED_MISSING_DIRECT_CLI_REFERENCE: set[str] = set()


def _write_script(root: Path, name: str) -> None:
    path = root / "scripts" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")


def _write_top_script(root: Path, name: str) -> None:
    path = root / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")


def _write_test(root: Path, name: str, body: str = "def test_placeholder():\n    assert True\n") -> None:
    path = root / "tests" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_schema(root: Path, name: str) -> None:
    path = root / "docs" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def test_scaffold_tool_index_reports_scripts_and_coverage_gaps(tmp_path: Path) -> None:
    _write_script(tmp_path, "benchmark_decode.py")
    _write_script(tmp_path, "build_serving_manifest.py")
    _write_script(tmp_path, "gpu-eval-run.sh")
    _write_script(tmp_path, "validate_pose_dataset.py")
    _write_script(tmp_path, "zz_private_helper.py")
    _write_top_script(tmp_path, "autolabel.py")
    _write_test(tmp_path, "test_decode_benchmark_summary.py")
    _write_test(tmp_path, "test_serving_manifest.py")
    _write_test(
        tmp_path,
        "test_cli_help.py",
        "def test_cli_help():\n    assert 'scripts/autolabel.py'\n    assert 'scripts/racketsport/gpu-eval-run.sh'\n",
    )
    _write_schema(tmp_path, "serving_manifest_schema.json")
    _write_schema(tmp_path, "pose_dataset_schema.json")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == 3
    assert payload["artifact_type"] == "racketsport_scaffold_tool_index"
    assert payload["scope"] == {
        "indexed_globs": ["scripts/racketsport/*.py", "scripts/racketsport/*.sh", "scripts/*.py", "scripts/*.sh"],
        "excluded_globs": [],
        "repo_wide_hygiene_report": False,
    }
    assert payload["execution"] == {
        "cpu_only": True,
        "runs_scaffold_commands": False,
        "uses_gpu": False,
        "downloads": False,
        "mutates_repo": False,
        "claims_build_or_eval_status": False,
    }
    assert payload["summary"] == {
        "tool_count": 6,
        "with_related_tests": 4,
        "missing_related_tests": 2,
        "with_direct_cli_reference_tests": 2,
        "missing_direct_cli_reference_tests": 4,
        "with_matching_json_schema_files": 2,
        "missing_matching_json_schema_files": 4,
        "category_counts": {
            "dataset": 1,
            "decode": 1,
            "eval": 1,
            "label": 1,
            "serving": 1,
            "unknown": 1,
        },
    }
    assert [tool["command_path"] for tool in payload["tools"]] == [
        "scripts/autolabel.py",
        "scripts/racketsport/benchmark_decode.py",
        "scripts/racketsport/build_serving_manifest.py",
        "scripts/racketsport/gpu-eval-run.sh",
        "scripts/racketsport/validate_pose_dataset.py",
        "scripts/racketsport/zz_private_helper.py",
    ]

    autolabel = payload["tools"][0]
    assert autolabel["related_test"] == "tests/racketsport/test_cli_help.py"
    assert autolabel["direct_cli_reference_test"] == "tests/racketsport/test_cli_help.py"

    decode = payload["tools"][1]
    assert decode["stem"] == "benchmark_decode"
    assert decode["category"] == "decode"
    assert decode["workstream"] == "EVAL"
    assert decode["task_prefix"] == "EVAL-0"
    assert decode["related_test"] == "tests/racketsport/test_decode_benchmark_summary.py"
    assert decode["direct_cli_reference_test"] is None
    assert decode["matching_schema"] is None

    serving = payload["tools"][2]
    assert serving["category"] == "serving"
    assert serving["workstream"] == "RPT"
    assert serving["task_prefix"] == "RPT-1"
    assert serving["related_test"] == "tests/racketsport/test_serving_manifest.py"
    assert serving["direct_cli_reference_test"] is None
    assert serving["matching_schema"] == "docs/racketsport/serving_manifest_schema.json"

    shell = payload["tools"][3]
    assert shell["command_path"] == "scripts/racketsport/gpu-eval-run.sh"
    assert shell["related_test"] == "tests/racketsport/test_cli_help.py"
    assert shell["direct_cli_reference_test"] == "tests/racketsport/test_cli_help.py"

    pose = payload["tools"][4]
    assert pose["category"] == "dataset"
    assert pose["workstream"] == "DATA"
    assert pose["task_prefix"] == "DATA-2"
    assert pose["related_test"] is None
    assert pose["direct_cli_reference_test"] is None
    assert pose["matching_schema"] == "docs/racketsport/pose_dataset_schema.json"

    unknown = payload["tools"][5]
    assert unknown["category"] == "unknown"
    assert unknown["workstream"] is None
    assert unknown["task_prefix"] is None
    assert unknown["related_test"] is None
    assert unknown["direct_cli_reference_test"] is None
    assert unknown["matching_schema"] is None


def test_scaffold_tool_index_rejects_invalid_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            str(tmp_path / "missing"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "root does not exist" in completed.stderr


def test_real_scaffold_tool_index_matches_checked_in_schema() -> None:
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
    schema = json.loads(Path("docs/racketsport/scaffold_tool_index_schema.json").read_text(encoding="utf-8"))

    assert_matches_json_schema(payload, schema)
    assert payload["summary"]["tool_count"] == len(payload["tools"])
    actual_missing = {
        tool["command_path"]
        for tool in payload["tools"]
        if tool.get("direct_cli_reference_test") is None
    }
    assert actual_missing == EXPECTED_MISSING_DIRECT_CLI_REFERENCE, (
        "scaffold CLI coverage drifted: new uncovered CLIs must ship a direct "
        "reference test in the same lane; newly covered CLIs must be removed "
        f"from EXPECTED_MISSING_DIRECT_CLI_REFERENCE. delta={actual_missing ^ EXPECTED_MISSING_DIRECT_CLI_REFERENCE}"
    )
    assert payload["summary"]["missing_direct_cli_reference_tests"] == 0
    assert payload["summary"]["category_counts"].get("unknown", 0) == 0
    by_path = {tool["command_path"]: tool for tool in payload["tools"]}
    assert {
        tool["command_path"]
        for tool in payload["tools"]
        if tool["direct_cli_reference_test"] is None
    } == set()
    assert (
        by_path["scripts/racketsport/list_scaffold_tools.py"]["matching_schema"]
        == "docs/racketsport/scaffold_tool_index_schema.json"
    )
