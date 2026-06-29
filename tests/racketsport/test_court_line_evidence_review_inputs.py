from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.build_court_line_evidence_from_review_inputs import build_court_line_evidence_from_review_inputs
from threed.racketsport.schemas import CourtLineEvidence, validate_artifact_file


def _point(evidence_id: str, endpoint: str, x: float, y: float) -> dict:
    return {
        "evidence_id": evidence_id,
        "endpoint": endpoint,
        "status": "clicked",
        "time_s": 0.0,
        "video_width": 960,
        "video_height": 540,
        "x": x,
        "y": y,
    }


def _review_input() -> dict:
    points = {}
    for line_id, first, second in [
        ("near_nvz", (100.0, 300.0), (860.0, 300.0)),
        ("far_nvz", (120.0, 180.0), (840.0, 180.0)),
        ("near_centerline", (480.0, 500.0), (480.0, 300.0)),
        ("far_centerline", (480.0, 180.0), (480.0, 80.0)),
        ("top_net", (110.0, 240.0), (850.0, 242.0)),
    ]:
        points[f"{line_id}:a"] = _point(line_id, "a", *first)
        points[f"{line_id}:b"] = _point(line_id, "b", *second)
    return {
        "schema_version": 1,
        "clips": {
            "clip_001": {
                "court_evidence": {
                    "near_nvz": "confirmed",
                    "far_nvz": "confirmed",
                    "near_centerline": "confirmed",
                    "far_centerline": "confirmed",
                    "top_net": "confirmed",
                    "points": points,
                    "point_statuses": {key: "clicked" for key in points},
                    "notes": "",
                }
            }
        },
    }


def _tennis_review_input() -> dict:
    points = {}
    for line_id, first, second in [
        ("near_service_line", (100.0, 330.0), (860.0, 330.0)),
        ("far_service_line", (120.0, 150.0), (840.0, 150.0)),
        ("top_net", (110.0, 240.0), (850.0, 242.0)),
    ]:
        points[f"{line_id}:a"] = _point(line_id, "a", *first)
        points[f"{line_id}:b"] = _point(line_id, "b", *second)
    return {
        "schema_version": 1,
        "clips": {
            "clip_001": {
                "court_evidence": {
                    "near_service_line": "confirmed",
                    "far_service_line": "confirmed",
                    "top_net": "confirmed",
                    "points": points,
                    "point_statuses": {key: "clicked" for key in points},
                    "notes": "",
                }
            }
        },
    }


def test_build_court_line_evidence_from_review_inputs_promotes_clicked_lines() -> None:
    evidence = build_court_line_evidence_from_review_inputs(_review_input(), clip="clip_001")

    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.source == "semantic_line_evidence"
    assert evidence.aggregate.auto_calibration_ready is True
    assert evidence.aggregate.missing_required_line_ids == []
    assert evidence.aggregate.missing_required_net_ids == []
    assert [item.line_id for item in evidence.line_observations] == [
        "near_nvz",
        "far_nvz",
        "near_centerline",
        "far_centerline",
    ]
    assert evidence.net_observations[0].net_id == "top_net"
    assert evidence.net_observations[0].image_points[1] == [480.0, 241.0]


def test_build_court_line_evidence_from_review_inputs_uses_tennis_required_lines() -> None:
    evidence = build_court_line_evidence_from_review_inputs(_tennis_review_input(), clip="clip_001", sport="tennis")

    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.sport == "tennis"
    assert [item.line_id for item in evidence.line_observations] == ["near_service_line", "far_service_line"]
    assert evidence.aggregate.auto_calibration_ready is True
    assert evidence.aggregate.missing_required_line_ids == []
    assert all("nvz" not in reason and "centerline" not in reason for reason in evidence.aggregate.reasons)


def test_build_court_line_evidence_from_review_inputs_fails_closed_when_points_missing() -> None:
    review_input = _review_input()
    del review_input["clips"]["clip_001"]["court_evidence"]["points"]["top_net:b"]

    evidence = build_court_line_evidence_from_review_inputs(review_input, clip="clip_001")

    assert evidence.aggregate.auto_calibration_ready is False
    assert evidence.aggregate.missing_required_net_ids == ["top_net"]
    assert "missing_top_net" in evidence.aggregate.reasons


def test_build_court_line_evidence_from_review_inputs_cli_writes_schema_valid_artifact(tmp_path: Path) -> None:
    review_input_path = tmp_path / "review_input.json"
    out = tmp_path / "court_line_evidence.json"
    review_input_path.write_text(json.dumps(_review_input()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_court_line_evidence_from_review_inputs.py",
            "--review-input",
            str(review_input_path),
            "--clip",
            "clip_001",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["auto_calibration_ready"] is True
    parsed = validate_artifact_file("court_line_evidence", out)
    assert isinstance(parsed, CourtLineEvidence)
