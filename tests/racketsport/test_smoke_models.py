from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _manifest(tmp_path: Path, sha256: str) -> Path:
    path = tmp_path / "manifest.json"
    payload = {
        "schema_version": 1,
        "models": [
            {
                "id": "fake_model",
                "stage": "unit_test",
                "use": "checksum verification",
                "source": "local",
                "license": "MIT",
                "commercial_posture": "ok",
                "status": "available_on_h100",
                "local_path": str(tmp_path / "weights.bin"),
                "sha256": sha256,
                "fallbacks": [],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_file_check_verifies_available_model_files_without_claiming_forward_smoke(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"pickleball")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path, digest)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/smoke_models.py",
            "--manifest",
            str(manifest),
            "--check-files-only",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    result = payload["models"][0]
    assert result["id"] == "fake_model"
    assert result["file_ok"] is True
    assert result["checksum_ok"] is True
    assert result["integrity_ok"] is True
    assert result["forward_smoke_status"] == "not_run"
    assert "smoke_ok" not in result


def test_smoke_models_fails_on_checksum_mismatch(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"pickleball")
    manifest = _manifest(tmp_path, "0" * 64)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/smoke_models.py",
            "--manifest",
            str(manifest),
            "--check-files-only",
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["models"][0]["file_ok"] is True
    assert payload["models"][0]["checksum_ok"] is False
    assert payload["models"][0]["integrity_ok"] is False
    assert payload["models"][0]["forward_smoke_status"] == "not_run"
    assert payload["integrity_failed"] == 1
