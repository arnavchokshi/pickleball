from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.racket_model_runtime_readiness import build_racket_model_runtime_readiness


def _write_manifest(path: Path, models: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "models": models}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _model(
    model_id: str,
    *,
    status: str = "pending_download",
    local_path: str | None = None,
    sha256: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": model_id,
        "stage": "racket_6dof",
        "use": f"{model_id} test use",
        "source": "https://example.com/model",
        "license": "Apache-2.0",
        "commercial_posture": "ok",
        "status": status,
        "fallbacks": [],
    }
    if local_path is not None:
        payload["local_path"] = local_path
    if sha256 is not None:
        payload["sha256"] = sha256
    return payload


def test_racket_model_runtime_readiness_reports_missing_manifest_entries(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "models" / "MANIFEST.json", [])

    report = build_racket_model_runtime_readiness(manifest)

    assert report["artifact_type"] == "racketsport_racket_model_runtime_readiness"
    assert report["status"] == "blocked"
    assert report["execution"] == {
        "cpu_only": True,
        "uses_gpu": False,
        "downloads_models": False,
        "imports_model_runtimes": False,
        "runs_inference": False,
        "claims_model_has_run": False,
        "mutates_model_manifest": False,
    }
    assert report["summary"]["component_count"] == 6
    assert report["summary"]["runtime_ready_count"] == 0
    assert report["summary"]["may_run_gpu_smoke"] is False
    assert report["summary"]["may_promote_rkt"] is False
    components = {component["component_id"]: component for component in report["components"]}
    assert components["sam3_concept_tracker"]["manifest_status"] == "missing"
    assert components["foundationpose_pose"]["runtime_ready"] is False
    assert "sam3_concept_tracker:missing_manifest_entry" in report["blockers"]


def test_racket_model_runtime_readiness_accepts_pending_manifest_entries_but_blocks_runtime(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "models" / "MANIFEST.json",
        [_model("sam3_concept_tracker", status="pending_auth")],
    )

    report = build_racket_model_runtime_readiness(manifest)

    component = {item["component_id"]: item for item in report["components"]}["sam3_concept_tracker"]
    assert component["manifest_status"] == "pending_auth"
    assert component["license_review_status"] == "declared_ok"
    assert component["runtime_ready"] is False
    assert component["blockers"] == ["pending_manifest_entry", "missing_runtime_probe"]
    assert "sam3_concept_tracker:pending_manifest_entry" in report["blockers"]


def test_racket_model_runtime_readiness_check_files_hashes_fake_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "sam3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"sam3 fake checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = _write_manifest(
        tmp_path / "models" / "MANIFEST.json",
        [
            _model(
                "sam3_concept_tracker",
                status="available_on_h100",
                local_path=str(checkpoint),
                sha256=digest,
            )
        ],
    )

    report = build_racket_model_runtime_readiness(
        manifest,
        check_files=True,
        allowed_checkpoint_prefixes=(str(tmp_path),),
    )

    component = {item["component_id"]: item for item in report["components"]}["sam3_concept_tracker"]
    assert component["manifest_status"] == "available_on_h100"
    assert component["checkpoint_status"] == "verified"
    assert component["runtime_ready"] is True
    assert component["blockers"] == []


def test_racket_model_runtime_readiness_rejects_unsafe_paths(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "models" / "MANIFEST.json",
        [
            _model(
                "sam3_concept_tracker",
                status="available_on_h100",
                local_path="../escaped.pt",
                sha256="0" * 64,
            )
        ],
    )

    report = build_racket_model_runtime_readiness(manifest)

    component = {item["component_id"]: item for item in report["components"]}["sam3_concept_tracker"]
    assert component["path_safety"]["safe"] is False
    assert component["checkpoint_status"] == "unsafe_path"
    assert component["runtime_ready"] is False
    assert "unsafe_local_path" in component["blockers"]


def test_racket_model_runtime_readiness_cli_writes_report_and_can_fail_on_blocked(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "models" / "MANIFEST.json", [])
    out = tmp_path / "racket_model_runtime_readiness.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_model_runtime_readiness.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert json.loads(completed.stdout)["status"] == "blocked"

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_model_runtime_readiness.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--fail-on-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
