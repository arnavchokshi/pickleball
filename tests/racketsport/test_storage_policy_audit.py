from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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


def test_storage_policy_readme_names_generated_artifact_check() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "generated cache/build leftovers" in text
    assert "__pycache__" in text
    assert "ios/.build" in text
    assert "web/replay/dist" in text
