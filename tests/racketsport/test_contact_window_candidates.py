from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.contact_window_candidates import build_contact_window_candidates_from_label_events
from threed.racketsport.schemas import ContactWindowCandidates, validate_artifact_file


def _events_payload() -> dict:
    return {
        "clip": {
            "name": "clip_001",
            "metadata": {
                "frame_rate_fps": 60,
            },
        },
        "annotation": {
            "target_file": "events.json",
            "items": [
                {
                    "confidence": 0.2,
                    "frame": "frame_000089.jpg",
                    "label": "contact?",
                    "review_id": "event_smoke_contact",
                    "status": "uncertain",
                    "type": "contact",
                    "xy_px": [960.0, 540.0],
                },
                {
                    "confidence": 0.9,
                    "frame": "frame_000120.jpg",
                    "label": "bounce?",
                    "review_id": "event_smoke_bounce",
                    "status": "uncertain",
                    "type": "bounce",
                    "xy_px": [970.0, 550.0],
                },
                {
                    "confidence": 0.7,
                    "frame": "not_a_frame.jpg",
                    "label": "bad",
                    "review_id": "bad_frame",
                    "status": "uncertain",
                    "type": "contact",
                },
            ],
        },
        "confidence": {
            "verified": False,
            "uncertainty_flags": ["teacher_model_unavailable", "smoke_generated"],
        },
    }


def test_build_contact_window_candidates_keeps_uncertain_events_review_only(tmp_path: Path) -> None:
    events_path = tmp_path / "labels" / "events.json"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(json.dumps(_events_payload()), encoding="utf-8")

    payload = build_contact_window_candidates_from_label_events(events_path)

    artifact = ContactWindowCandidates.model_validate(payload)
    assert artifact.artifact_type == "racketsport_contact_window_candidates"
    assert artifact.clip == "clip_001"
    assert artifact.fps == 60.0
    assert artifact.not_gate_verified is True
    assert artifact.trusted_for_body is False
    assert artifact.promotion_target == "contact_windows.json"
    assert artifact.summary.candidate_count == 2
    assert artifact.summary.rejected_item_count == 1
    assert artifact.summary.by_type == {"bounce": 1, "contact": 1}
    assert artifact.summary.by_status == {"uncertain": 2}
    assert artifact.summary.uncertainty_flags == ["teacher_model_unavailable", "smoke_generated"]
    assert artifact.candidates[0].review_id == "event_smoke_contact"
    assert artifact.candidates[0].frame == 89
    assert artifact.candidates[0].t == pytest.approx(89 / 60.0)
    assert artifact.candidates[0].window.t0 == pytest.approx(max(0.0, 89 / 60.0 - 0.08))
    assert artifact.candidates[0].window.t1 == pytest.approx(89 / 60.0 + 0.08)
    assert artifact.candidates[0].source_status == "uncertain"
    assert artifact.candidates[0].source_confidence == pytest.approx(0.2)
    assert artifact.candidates[0].candidate_confidence == pytest.approx(0.2)


def test_contact_window_candidates_cli_writes_schema_valid_artifact(tmp_path: Path) -> None:
    events_path = tmp_path / "labels" / "events.json"
    out = tmp_path / "contact_window_candidates.json"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(json.dumps(_events_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_window_candidates.py",
            "--events",
            str(events_path),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["summary"]["candidate_count"] == 2
    artifact = validate_artifact_file("contact_window_candidates", out)
    assert isinstance(artifact, ContactWindowCandidates)
    assert artifact.summary.rejected_item_count == 1
