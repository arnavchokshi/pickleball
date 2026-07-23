from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.racketsport.verify_training_inputs import (
    GateProofError,
    _proof_sha256,
    assert_gate_proof,
    verify_training_inputs,
)


CODE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    CODE_ROOT / "tests/racketsport/fixtures/training_gate"
).resolve()
LEDGER = FIXTURE_ROOT / "runs/manager/data_ledger.json"
CACHE_MANIFEST = FIXTURE_ROOT / "CACHE_MANIFEST.json"
ALLOWED_INPUT = FIXTURE_ROOT / "data/allowed_input.json"
PROTECTED_INPUT = FIXTURE_ROOT / "data/protected_input.json"
ALLOWED_ASSET_ID = "synthetic_training_allowed"
PROTECTED_ASSET_ID = "synthetic_training_protected"
CACHE_ASSET_ID = "synthetic_cache_inventory"
EXPECTED_83_SHA256 = (
    "272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383"
)
CACHED_83_SHA256 = (
    "5855cb92d4170e939bda322640cfd6a38b3b5aa4f3f7ad4abb897ab3f313ce7a"
)


def _head_trainer_imports_gate_seam() -> bool:
    completed = subprocess.run(
        [
            "git",
            "show",
            "HEAD:scripts/racketsport/finetune_event_head.py",
        ],
        cwd=CODE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return False
    try:
        module = ast.parse(completed.stdout)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "scripts.racketsport.verify_training_inputs"
        and any(alias.name == "assert_gate_proof" for alias in node.names)
        for node in ast.walk(module)
    )


def _write_inputs(path: Path, inputs: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "training_input_manifest",
                "inputs": inputs,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _reason_codes(proof: dict[str, object], index: int = 0) -> set[str]:
    verdict = proof["inputs"][index]
    assert isinstance(verdict, dict)
    reasons = verdict["reasons"]
    assert isinstance(reasons, list)
    return {
        reason["code"]
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("code"), str)
    }


def _passing_proof(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(ALLOWED_INPUT),
                "asset_id": ALLOWED_ASSET_ID,
            }
        ],
    )
    proof_path = tmp_path / "gate_proof.json"
    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=proof_path,
    )
    assert proof["status"] == "PASS", proof
    return proof_path, proof


def test_direct_cli_writes_passing_machine_readable_proof(tmp_path: Path) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(ALLOWED_INPUT),
                "asset_id": ALLOWED_ASSET_ID,
            }
        ],
    )
    proof_path = tmp_path / "gate_proof.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/verify_training_inputs.py",
            "--inputs",
            str(inputs),
            "--ledger",
            str(LEDGER),
            "--repo-root",
            str(FIXTURE_ROOT),
            "--gate-proof",
            str(proof_path),
        ],
        cwd=CODE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASS"
    assert json.loads(proof_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_synthetic_83_cache_fixture_is_quarantined_before_input_bytes_are_read(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": "/cache/media/pbvision/83gyqyc10y8f/max.mp4",
                "asset_id": CACHE_ASSET_ID,
                "source_id": "83gyqyc10y8f",
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        cache_manifest_path=CACHE_MANIFEST,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert {
        "CACHE_ENTRY_FORBIDS_TRAINING",
        "CACHE_SHA256_MISMATCH",
        "LEDGER_PROVENANCE_FORBIDS_TRAINING",
    }.issubset(_reason_codes(proof))
    verdict = proof["inputs"][0]
    assert verdict["input_bytes_read_for_integrity"] is False
    assert verdict["cache_entry_sha256s"] == {
        "expected_sha256": EXPECTED_83_SHA256,
        "sha256": CACHED_83_SHA256,
    }


def test_synthetic_83_cache_correction_preserves_hashes_and_rederivation_pointer() -> None:
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    entry = next(
        row
        for row in manifest["media"]["synthetic"]
        if row["id"] == "83gyqyc10y8f"
    )

    assert entry["status"] == "QUARANTINED_SHA_MISMATCH"
    assert entry["expected_sha256"] == EXPECTED_83_SHA256
    assert entry["sha256"] == CACHED_83_SHA256
    assert entry["re_derivation"] == {
        "source_url": (
            "https://fixtures.invalid/training-gate/83gyqyc10y8f/max.mp4"
        ),
        "expected_sha256": EXPECTED_83_SHA256,
    }
    assert "must not be consumed from cache" in entry["quarantine_rationale"]


def test_cache_input_requires_cache_manifest_even_when_bytes_are_absent(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": "/cache/media/synthetic/allowed.bin",
                "asset_id": CACHE_ASSET_ID,
                "source_id": "synthetic-cache-allowed",
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert "CACHE_MANIFEST_REQUIRED" in _reason_codes(proof)


def test_never_train_input_smuggled_by_symlink_is_refused_without_opening_target(
    tmp_path: Path,
) -> None:
    alias = tmp_path / "neutral-name.json"
    alias.symlink_to(PROTECTED_INPUT)
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(alias),
                "asset_id": PROTECTED_ASSET_ID,
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert {
        "LEDGER_STATE_FORBIDS_TRAINING",
        "LEDGER_PROVENANCE_FORBIDS_TRAINING",
    }.issubset(_reason_codes(proof))
    assert proof["inputs"][0]["resolved_path"] == str(PROTECTED_INPUT.resolve())
    assert proof["inputs"][0]["input_bytes_read_for_integrity"] is False


def test_never_train_input_smuggled_by_cache_alias_is_refused(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": "/cache/media/synthetic/protected.bin",
                "asset_id": CACHE_ASSET_ID,
                "source_id": "synthetic-cache-protected",
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        cache_manifest_path=CACHE_MANIFEST,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert "CACHE_ENTRY_FORBIDS_TRAINING" in _reason_codes(proof)
    assert "LEDGER_PROVENANCE_FORBIDS_TRAINING" in _reason_codes(proof)


def test_renamed_copy_with_forbidden_bytes_is_caught_by_content_sha256(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "renamed-training-example.bin"
    shutil.copyfile(PROTECTED_INPUT, copied)
    digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(copied),
                "asset_id": ALLOWED_ASSET_ID,
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert "CONTENT_SHA256_FORBIDS_TRAINING" in _reason_codes(proof)
    assert proof["inputs"][0]["actual_sha256"] == digest
    assert proof["inputs"][0]["input_bytes_read_for_integrity"] is True


def test_recorded_sha256_mismatch_is_typed(tmp_path: Path) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(ALLOWED_INPUT),
                "asset_id": ALLOWED_ASSET_ID,
                "sha256": "0" * 64,
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert "RECORDED_SHA256_MISMATCH" in _reason_codes(proof)


def test_usage_posture_marker_forbids_training_before_payload_read(
    tmp_path: Path,
) -> None:
    marked_root = tmp_path / "marked"
    marked_root.mkdir()
    marker = marked_root / "USAGE_POSTURE.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "true_source_id": "neutral-fixture",
                "posture": "compare-only",
                "authority_ruling_pointer": (
                    "runs/manager/data_ledger.json#asset_id="
                    f"{ALLOWED_ASSET_ID}"
                ),
                "never_train": True,
                "date": "2026-07-23",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = marked_root / "renamed.bin"
    payload.write_bytes(b"must not be opened after marker refusal\n")
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": str(payload),
                "asset_id": ALLOWED_ASSET_ID,
                "provenance_marker": str(marker),
            }
        ],
    )

    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        repo_root=FIXTURE_ROOT,
        proof_path=tmp_path / "gate_proof.json",
    )

    assert proof["status"] == "FAIL"
    assert "PROVENANCE_MARKER_FORBIDS_TRAINING" in _reason_codes(proof)
    assert proof["inputs"][0]["input_bytes_read_for_integrity"] is False
    assert proof["inputs"][0]["actual_sha256"] is None


def test_stale_and_forged_gate_proofs_have_distinct_typed_refusals(
    tmp_path: Path,
) -> None:
    proof_path, proof = _passing_proof(tmp_path)
    generated_at = datetime.fromisoformat(
        str(proof["generated_at_utc"]).replace("Z", "+00:00")
    )

    with pytest.raises(GateProofError) as stale:
        assert_gate_proof(
            proof_path,
            repo_root=FIXTURE_ROOT,
            required_input_paths=[ALLOWED_INPUT],
            max_age_seconds=60,
            now=generated_at + timedelta(seconds=61),
        )
    assert stale.value.code == "GATE_PROOF_STALE"

    forged = json.loads(proof_path.read_text(encoding="utf-8"))
    forged["status"] = "FAIL"
    proof_path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(GateProofError) as tampered:
        assert_gate_proof(
            proof_path,
            repo_root=FIXTURE_ROOT,
            required_input_paths=[ALLOWED_INPUT],
            now=datetime.now(timezone.utc),
        )
    assert tampered.value.code == "GATE_PROOF_INTEGRITY_MISMATCH"


def test_forged_proof_with_recomputed_self_hash_cannot_replace_canonical_ledger(
    tmp_path: Path,
) -> None:
    proof_path, _ = _passing_proof(tmp_path)
    copied_ledger = tmp_path / "substitute_data_ledger.json"
    shutil.copyfile(LEDGER, copied_ledger)
    forged = json.loads(proof_path.read_text(encoding="utf-8"))
    forged["source_manifests"]["data_ledger"]["path"] = str(copied_ledger)
    forged["proof_sha256"] = _proof_sha256(forged)
    proof_path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(GateProofError) as refused:
        assert_gate_proof(
            proof_path,
            repo_root=FIXTURE_ROOT,
            required_input_paths=[ALLOWED_INPUT],
        )
    assert refused.value.code == "GATE_PROOF_LEDGER_UNTRUSTED"


def test_failed_gate_proof_is_typed_for_downstream_refusal(tmp_path: Path) -> None:
    inputs = _write_inputs(
        tmp_path / "training_inputs.json",
        [
            {
                "path": "/cache/media/pbvision/83gyqyc10y8f/max.mp4",
                "asset_id": CACHE_ASSET_ID,
                "source_id": "83gyqyc10y8f",
            }
        ],
    )
    proof_path = tmp_path / "gate_proof.json"
    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=LEDGER,
        cache_manifest_path=CACHE_MANIFEST,
        repo_root=FIXTURE_ROOT,
        proof_path=proof_path,
    )
    assert proof["status"] == "FAIL"

    with pytest.raises(GateProofError) as failed:
        assert_gate_proof(proof_path, repo_root=FIXTURE_ROOT)
    assert failed.value.code == "GATE_PROOF_FAILED"


@pytest.mark.skipif(
    not _head_trainer_imports_gate_seam(),
    reason="trainer-side seam lands with Track D E-v2",
)
def test_finetune_cli_refuses_forged_proof_before_training_input_read(
    tmp_path: Path,
) -> None:
    trainer_root = tmp_path / "trainer_repo"
    trainer_root.mkdir()
    shutil.copytree(CODE_ROOT / "scripts", trainer_root / "scripts")
    shutil.copytree(CODE_ROOT / "threed", trainer_root / "threed")
    shutil.copytree(FIXTURE_ROOT / "data", trainer_root / "data")
    shutil.copytree(FIXTURE_ROOT / "runs", trainer_root / "runs")
    (trainer_root / ".git").write_text(
        f"gitdir: {(CODE_ROOT / '.git').resolve()}\n",
        encoding="utf-8",
    )
    trainer_input = trainer_root / "data/allowed_input.json"
    trainer_ledger = trainer_root / "runs/manager/data_ledger.json"
    inputs = _write_inputs(
        tmp_path / "trainer_inputs.json",
        [{"path": str(trainer_input), "asset_id": ALLOWED_ASSET_ID}],
    )
    proof_path = tmp_path / "trainer_gate_proof.json"
    proof = verify_training_inputs(
        input_manifest_path=inputs,
        ledger_path=trainer_ledger,
        repo_root=trainer_root,
        proof_path=proof_path,
    )
    assert proof["status"] == "PASS", proof
    forged = json.loads(proof_path.read_text(encoding="utf-8"))
    forged["repo_head_sha"] = "0" * 40
    proof_path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")
    missing_checkpoint = tmp_path / "must-not-be-opened.pt"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/finetune_event_head.py",
            "--gate-proof",
            str(proof_path),
            "--owner-manifest",
            str(trainer_input),
            "--init-checkpoint-model-only",
            str(missing_checkpoint),
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=trainer_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 20
    assert "GATE_PROOF_INTEGRITY_MISMATCH" in completed.stderr
    assert "owner manifest" not in completed.stderr
    assert "checkpoint" not in completed.stderr
