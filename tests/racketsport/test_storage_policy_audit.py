from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport import audit_storage_policy


ROOT = Path(__file__).resolve().parents[2]


def test_storage_policy_audit_reports_generated_artifacts_in_temp_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
                "__pycache__/",
                ".pytest_cache/",
                "ios/.build/",
                "web/replay/dist/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "pkg" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "module.cpython-311.pyc").write_bytes(b"cache")
    (tmp_path / "web" / "replay" / "dist").mkdir(parents=True)
    (tmp_path / "web" / "replay" / "dist" / "bundle.js").write_text("generated", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "racketsport" / "audit_storage_policy.py"),
            "--root",
            str(tmp_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert report["status"] == "fail"
    assert report["generated_artifacts"] == [
        "pkg/__pycache__",
        "web/replay/dist",
    ]


def test_storage_policy_audit_finds_pytest_trees_inside_ignored_runs(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / ".gitignore").write_text("runs/\n", encoding="utf-8")
    pytest_tree = tmp_path / "runs" / "lanes" / "experiment" / ".pytest_focused"
    pytest_tree.mkdir(parents=True)
    (pytest_tree / "checkpoint.pt").write_bytes(b"checkpoint")

    report = audit_storage_policy.build_storage_report(tmp_path)

    assert report["status"] == "fail"
    assert report["generated_artifacts"] == ["runs/lanes/experiment/.pytest_focused"]


def test_storage_policy_audit_enforces_directory_size_limits(tmp_path: Path, monkeypatch) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "payload.bin").write_bytes(b"12345")
    monkeypatch.setattr(audit_storage_policy, "DIRECTORY_SIZE_LIMIT_BYTES", {"runs": 4})

    report = audit_storage_policy.build_storage_report(tmp_path, check_generated_artifacts=False)

    assert report["status"] == "fail"
    assert report["oversized_directories"] == {"runs": {"bytes": 5, "limit_bytes": 4}}


def test_storage_policy_readme_names_generated_artifact_check() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "generated cache/build leftovers" in text
    assert "__pycache__" in text
    assert "ios/.build" in text
    assert "web/replay/dist" in text
    assert "25 GiB" in text
    assert "in-repo `--basetemp`" in text
