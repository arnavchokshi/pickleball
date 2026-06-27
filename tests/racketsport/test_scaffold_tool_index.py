from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_script(root: Path, name: str) -> None:
    path = root / "scripts" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")


def _write_test(root: Path, name: str) -> None:
    path = root / "tests" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")


def _write_schema(root: Path, name: str) -> None:
    path = root / "docs" / "racketsport" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def test_scaffold_tool_index_reports_scripts_and_coverage_gaps(tmp_path: Path) -> None:
    _write_script(tmp_path, "benchmark_decode.py")
    _write_script(tmp_path, "build_serving_manifest.py")
    _write_script(tmp_path, "validate_pose_dataset.py")
    _write_script(tmp_path, "zz_private_helper.py")
    _write_test(tmp_path, "test_decode_benchmark_summary.py")
    _write_test(tmp_path, "test_serving_manifest.py")
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

    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "racketsport_scaffold_tool_index"
    assert payload["execution"] == {
        "cpu_only": True,
        "runs_scaffold_commands": False,
        "uses_gpu": False,
        "downloads": False,
        "mutates_repo": False,
        "claims_build_or_eval_status": False,
    }
    assert payload["summary"] == {
        "tool_count": 4,
        "with_matching_tests": 2,
        "missing_matching_tests": 2,
        "with_matching_schemas": 2,
        "missing_matching_schemas": 2,
        "category_counts": {
            "dataset": 1,
            "decode": 1,
            "serving": 1,
            "unknown": 1,
        },
    }
    assert [tool["command_path"] for tool in payload["tools"]] == [
        "scripts/racketsport/benchmark_decode.py",
        "scripts/racketsport/build_serving_manifest.py",
        "scripts/racketsport/validate_pose_dataset.py",
        "scripts/racketsport/zz_private_helper.py",
    ]

    decode = payload["tools"][0]
    assert decode["stem"] == "benchmark_decode"
    assert decode["category"] == "decode"
    assert decode["workstream"] == "EVAL"
    assert decode["task_prefix"] == "EVAL-0"
    assert decode["matching_test"] == "tests/racketsport/test_decode_benchmark_summary.py"
    assert decode["matching_schema"] is None

    serving = payload["tools"][1]
    assert serving["category"] == "serving"
    assert serving["workstream"] == "RPT"
    assert serving["task_prefix"] == "RPT-1"
    assert serving["matching_test"] == "tests/racketsport/test_serving_manifest.py"
    assert serving["matching_schema"] == "docs/racketsport/serving_manifest_schema.json"

    pose = payload["tools"][2]
    assert pose["category"] == "dataset"
    assert pose["workstream"] == "DATA"
    assert pose["task_prefix"] == "DATA-2"
    assert pose["matching_test"] is None
    assert pose["matching_schema"] == "docs/racketsport/pose_dataset_schema.json"

    unknown = payload["tools"][3]
    assert unknown["category"] == "unknown"
    assert unknown["workstream"] is None
    assert unknown["task_prefix"] is None
    assert unknown["matching_test"] is None
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
