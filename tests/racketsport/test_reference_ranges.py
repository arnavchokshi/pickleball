from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tests.racketsport.json_schema_assertions import assert_matches_json_schema


CLI_PATH = "scripts/racketsport/validate_reference_ranges.py"
SCHEMA_PATH = Path("docs/racketsport/reference_ranges_schema.json")
RANGES_PATH = Path("docs/racketsport/reference_ranges_v0.json")


def test_reference_ranges_v0_schema_and_acceptance_counts() -> None:
    module = _load_validator_module()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = json.loads(RANGES_PATH.read_text(encoding="utf-8"))

    assert_matches_json_schema(payload, schema)
    report = module.validate_reference_ranges(payload, schema)

    assert report["status"] == "pass"
    ranges = payload["ranges"]
    assert len(ranges) >= 12
    assert len({entry["metric_family"] for entry in ranges}) >= 4
    assert len({entry["skill_band"] for entry in ranges}) >= 3
    sourced = [entry for entry in ranges if entry["provenance"]["tier"] in {"measured", "trade_benchmark"}]
    assert len(sourced) >= 8
    for entry in sourced:
        assert entry["provenance"]["url"].startswith("https://")
        assert entry["provenance"]["source"]


def test_semantic_checks_reject_duplicates_bad_units_and_unordered_bands() -> None:
    module = _load_validator_module()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = json.loads(RANGES_PATH.read_text(encoding="utf-8"))

    duplicate = copy.deepcopy(payload)
    duplicate["ranges"].append(copy.deepcopy(duplicate["ranges"][0]))
    duplicate_report = module.validate_reference_ranges(duplicate, schema)
    assert duplicate_report["status"] == "fail"
    assert any("duplicate metric_id+skill_band" in error for error in duplicate_report["errors"])

    bad_unit = copy.deepcopy(payload)
    bad_unit["ranges"][0]["range"]["unit"] = "made_up_unit"
    bad_unit_report = module.validate_reference_ranges(bad_unit, schema)
    assert bad_unit_report["status"] == "fail"
    assert any("unknown unit" in error for error in bad_unit_report["errors"])

    unordered = copy.deepcopy(payload)
    unordered["ranges"][0]["range"]["lo"] = 0.9
    unordered["ranges"][0]["range"]["hi"] = 0.1
    unordered_report = module.validate_reference_ranges(unordered, schema)
    assert unordered_report["status"] == "fail"
    assert any("range.lo > range.hi" in error for error in unordered_report["errors"])


def test_validate_reference_ranges_cli_round_trip_and_json_failure(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--ranges",
            str(RANGES_PATH),
            "--schema",
            str(SCHEMA_PATH),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pass"
    assert payload["summary"]["range_count"] >= 12

    invalid_path = tmp_path / "duplicate_reference_ranges.json"
    invalid = json.loads(RANGES_PATH.read_text(encoding="utf-8"))
    invalid["ranges"].append(copy.deepcopy(invalid["ranges"][0]))
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

    failed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--ranges",
            str(invalid_path),
            "--schema",
            str(SCHEMA_PATH),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    failure_payload = json.loads(failed.stdout)
    assert failure_payload["status"] == "fail"
    assert any("duplicate metric_id+skill_band" in error for error in failure_payload["errors"])


def test_scaffold_index_covers_validate_reference_ranges_cli() -> None:
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
    by_path = {tool["command_path"]: tool for tool in payload["tools"]}

    assert by_path[CLI_PATH]["category"] == "report"
    assert by_path[CLI_PATH]["workstream"] == "COACH"
    assert by_path[CLI_PATH]["task_prefix"] == "P6-3"
    assert by_path[CLI_PATH]["related_test"] == "tests/racketsport/test_reference_ranges.py"
    assert by_path[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_reference_ranges.py"
    assert by_path[CLI_PATH]["matching_schema"] == "docs/racketsport/reference_ranges_schema.json"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_reference_ranges_under_test", CLI_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
