from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_manifest(tmp_path: Path) -> Path:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"pickleball")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "fake_model",
                        "stage": "unit_test",
                        "use": "doctor smoke",
                        "source": "local",
                        "license": "MIT",
                        "commercial_posture": "ok",
                        "status": "available_on_h100",
                        "local_path": str(weights),
                        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                        "fallbacks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_doctor_cli_help_documents_incident_backed_checks() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/doctor.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scripts/racketsport/doctor.py" in Path("tests/racketsport/test_doctor.py").read_text(encoding="utf-8")
    assert "MPLBACKEND=Agg" in completed.stdout
    assert "audit_storage_policy.py --ignore-generated-artifacts" in completed.stdout
    assert "remote_body_stdout.log" in completed.stdout
    assert "--verify-version-stamp" in completed.stdout
    assert "scripts/fleet/refresh_remote_host.sh" in completed.stdout


def test_doctor_json_smoke_uses_tmp_manifest_and_reports_no_failures(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    web_replay = tmp_path / "web" / "replay"
    (web_replay / "node_modules").mkdir(parents=True)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/doctor.py",
            "--manifest",
            str(manifest),
            "--web-replay-dir",
            str(web_replay),
            "--disk-path",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["artifact_type"] == "racketsport_doctor_report"
    assert payload["status"] == "pass"
    assert payload["summary"]["fail"] == 0
    assert payload["checks"]["model_weights"]["details"]["integrity_failed"] == 0
    assert payload["checks"]["generated_artifacts_audit_hint"]["status"] == "info"
