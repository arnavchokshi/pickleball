from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.build_corrections_queue import build_corrections_queue, discover_correction_manifests
from scripts.racketsport.validate_corrections import validate_manifest


def _valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "eval3_manual_seed",
        "created_at": "2026-06-26T21:00:00Z",
        "description": "Seed manifest for manual linting only.",
        "corrections": [
            {
                "id": "corr_001",
                "target": {
                    "artifact": "runs/eval3/clip_001/racket_pose.json",
                    "clip_id": "clip_001",
                    "frame_index": 42,
                    "path": "/players/0/frames/42/conf",
                },
                "operation": "replace",
                "value": 0.91,
                "reason": "Manual review found a missed high-confidence paddle frame.",
                "annotator": "agent-k",
                "created_at": "2026-06-26T21:05:00Z",
            }
        ],
    }


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_manifest_accepts_strict_generic_corrections(tmp_path):
    manifest_path = _write_manifest(tmp_path / "corrections.json", _valid_manifest())

    summary = validate_manifest(manifest_path)

    assert summary == {
        "schema_version": 1,
        "manifest_id": "eval3_manual_seed",
        "path": str(manifest_path),
        "correction_count": 1,
        "correction_ids": ["corr_001"],
    }


def test_validate_manifest_rejects_extra_fields_and_missing_values(tmp_path):
    payload = _valid_manifest()
    payload["unexpected"] = True
    payload["corrections"][0]["operation"] = "set"
    payload["corrections"][0].pop("value")
    manifest_path = _write_manifest(tmp_path / "corrections.json", payload)

    with pytest.raises(ValueError) as excinfo:
        validate_manifest(manifest_path)

    message = str(excinfo.value)
    assert "unexpected" in message
    assert "value" in message


def test_validate_manifest_rejects_unsafe_artifact_paths_and_duplicate_ids(tmp_path):
    payload = _valid_manifest()
    payload["corrections"].append(dict(payload["corrections"][0]))
    payload["corrections"][1]["target"] = dict(payload["corrections"][1]["target"])
    payload["corrections"][1]["target"]["artifact"] = "../outside.json"
    manifest_path = _write_manifest(tmp_path / "corrections.json", payload)

    with pytest.raises(ValueError) as excinfo:
        validate_manifest(manifest_path)

    message = str(excinfo.value)
    assert "duplicate correction id: corr_001" in message
    assert "must be relative and stay within the workspace" in message


def test_validate_manifest_rejects_unknown_correction_status(tmp_path):
    payload = _valid_manifest()
    payload["corrections"][0]["status"] = "needs_human"
    manifest_path = _write_manifest(tmp_path / "corrections.json", payload)

    with pytest.raises(ValueError) as excinfo:
        validate_manifest(manifest_path)

    assert "status: must be one of accepted, pending, rejected" in str(excinfo.value)


def test_validate_corrections_cli_emits_json_summary(tmp_path):
    manifest_path = _write_manifest(tmp_path / "corrections.json", _valid_manifest())

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_corrections.py", str(manifest_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["correction_count"] == 1
    assert completed.stderr == ""


def test_validate_corrections_cli_reports_validation_errors(tmp_path):
    payload = _valid_manifest()
    payload["corrections"][0]["target"]["path"] = "players/0"
    manifest_path = _write_manifest(tmp_path / "corrections.json", payload)

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_corrections.py", str(manifest_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "target/path" in completed.stderr
    assert "does not match" in completed.stderr


def test_build_corrections_queue_flattens_valid_manifests(tmp_path):
    first = _valid_manifest()
    second = _valid_manifest()
    second["manifest_id"] = "eval3_manual_followup"
    second["corrections"][0]["id"] = "corr_002"
    second["corrections"][0]["operation"] = "append"
    second["corrections"][0]["target"]["artifact"] = "runs/eval3/clip_002/contact_windows.json"
    second["corrections"][0]["target"]["clip_id"] = "clip_002"

    queue = build_corrections_queue(
        [
            _write_manifest(tmp_path / "first.json", first),
            _write_manifest(tmp_path / "second.json", second),
        ]
    )

    assert queue["schema_version"] == 1
    assert queue["manifest_count"] == 2
    assert queue["correction_count"] == 2
    assert queue["summary"]["by_operation"] == {"append": 1, "replace": 1}
    assert queue["summary"]["by_clip"] == {"clip_001": 1, "clip_002": 1}
    assert queue["corrections"][0]["manifest_id"] == "eval3_manual_seed"
    assert queue["corrections"][0]["correction_id"] == "corr_001"
    assert queue["corrections"][0]["value"] == 0.91


def test_build_corrections_queue_groups_by_status_phase_metric_clip(tmp_path):
    accepted = _valid_manifest()
    accepted["corrections"][0]["status"] = "accepted"
    accepted["corrections"][0]["target"]["phase"] = "phase6"
    accepted["corrections"][0]["target"]["metric"] = "racket_pose"

    pending = _valid_manifest()
    pending["manifest_id"] = "eval3_manual_pending"
    pending["corrections"][0]["id"] = "corr_002"
    pending["corrections"][0]["status"] = "pending"
    pending["corrections"][0]["target"]["artifact"] = "runs/phase5/clip_001/contact_windows.json"
    pending["corrections"][0]["target"]["phase"] = "phase5"
    pending["corrections"][0]["target"]["metric"] = "contact_timing"

    queue = build_corrections_queue(
        [
            _write_manifest(tmp_path / "accepted.json", accepted),
            _write_manifest(tmp_path / "pending.json", pending),
        ]
    )

    assert queue["summary"]["by_status"] == {"accepted": 1, "pending": 1}
    assert queue["summary"]["by_phase"] == {"phase5": 1, "phase6": 1}
    assert queue["summary"]["by_metric"] == {"contact_timing": 1, "racket_pose": 1}
    assert queue["summary"]["by_phase_metric_clip"] == {
        "phase5/contact_timing/clip_001": 1,
        "phase6/racket_pose/clip_001": 1,
    }
    assert queue["corrections"][0]["status"] == "accepted"
    assert queue["corrections"][0]["phase"] == "phase6"
    assert queue["corrections"][0]["metric"] == "racket_pose"


def test_build_training_manifest_candidates_uses_only_accepted_corrections(tmp_path):
    from threed.racketsport.corrections import build_training_manifest_candidates

    accepted = _valid_manifest()
    accepted["corrections"][0]["status"] = "accepted"
    accepted["corrections"][0]["target"]["phase"] = "phase6"
    accepted["corrections"][0]["target"]["metric"] = "racket_pose"

    rejected = _valid_manifest()
    rejected["manifest_id"] = "eval3_manual_rejected"
    rejected["corrections"][0]["id"] = "corr_002"
    rejected["corrections"][0]["status"] = "rejected"
    rejected["corrections"][0]["target"]["phase"] = "phase6"
    rejected["corrections"][0]["target"]["metric"] = "racket_pose"

    queue = build_corrections_queue(
        [
            _write_manifest(tmp_path / "accepted.json", accepted),
            _write_manifest(tmp_path / "rejected.json", rejected),
        ]
    )

    manifest = build_training_manifest_candidates(queue, candidate_root="training/corrections")

    assert manifest["schema_version"] == 1
    assert manifest["accepted_correction_count"] == 1
    assert manifest["entries"] == [
        {
            "id": "eval3_manual_seed__corr_001",
            "manifest_id": "eval3_manual_seed",
            "correction_id": "corr_001",
            "clip_id": "clip_001",
            "phase": "phase6",
            "metric": "racket_pose",
            "source_artifact": "runs/eval3/clip_001/racket_pose.json",
            "source_path": "/players/0/frames/42/conf",
            "candidate_path": "training/corrections/phase6/racket_pose/clip_001/eval3_manual_seed__corr_001.json",
            "operation": "replace",
            "value": 0.91,
        }
    ]


def test_build_training_manifest_candidates_rejects_unsafe_candidate_root(tmp_path):
    from threed.racketsport.corrections import build_training_manifest_candidates

    accepted = _valid_manifest()
    accepted["corrections"][0]["status"] = "accepted"
    accepted["corrections"][0]["target"]["phase"] = "phase6"
    accepted["corrections"][0]["target"]["metric"] = "racket_pose"
    queue = build_corrections_queue([_write_manifest(tmp_path / "accepted.json", accepted)])

    with pytest.raises(ValueError) as excinfo:
        build_training_manifest_candidates(queue, candidate_root="../outside")

    assert "candidate_root must be relative and stay within the workspace" in str(excinfo.value)


def test_build_corrections_queue_rejects_duplicate_manifest_correction_ids(tmp_path):
    first = _write_manifest(tmp_path / "first.json", _valid_manifest())
    second = _write_manifest(tmp_path / "second.json", _valid_manifest())

    with pytest.raises(ValueError) as excinfo:
        build_corrections_queue([first, second])

    assert "duplicate queued correction id: eval3_manual_seed/corr_001" in str(excinfo.value)


def test_build_corrections_queue_cli_discovers_root_and_writes_queue(tmp_path):
    root = tmp_path / "manifests"
    out = tmp_path / "queue" / "corrections_queue.json"
    root.mkdir()
    (root / "schema.json").write_text("{}", encoding="utf-8")
    _write_manifest(root / "corrections.json", _valid_manifest())

    discovered = discover_correction_manifests(root)
    assert discovered == [root / "corrections.json"]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_corrections_queue.py",
            "--root",
            str(root),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["correction_count"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["corrections"][0]["clip_id"] == "clip_001"


def test_build_corrections_queue_cli_writes_training_manifest_candidates(tmp_path):
    root = tmp_path / "manifests"
    out = tmp_path / "queue" / "corrections_queue.json"
    training_out = tmp_path / "queue" / "training_candidates.json"
    root.mkdir()
    (root / "schema.json").write_text("{}", encoding="utf-8")

    payload = _valid_manifest()
    payload["corrections"][0]["status"] = "accepted"
    payload["corrections"][0]["target"]["phase"] = "phase6"
    payload["corrections"][0]["target"]["metric"] = "racket_pose"
    _write_manifest(root / "corrections.json", payload)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_corrections_queue.py",
            "--root",
            str(root),
            "--out",
            str(out),
            "--training-manifest-out",
            str(training_out),
            "--training-candidate-root",
            "training/corrections",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(training_out.read_text(encoding="utf-8"))["accepted_correction_count"] == 1
