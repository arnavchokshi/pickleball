from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.pipeline_contracts import (
    PIPELINE_STAGE_ORDER,
    PipelineContractError,
    build_readiness_report,
    safe_relative_path,
)


def _touch_all(run_dir: Path, names: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (run_dir / name).write_text("{}\n", encoding="utf-8")


def _write_ready_court_line_evidence(run_dir: Path) -> None:
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_court_line_evidence",
                "aggregate": {
                    "auto_calibration_ready": True,
                    "missing_required_line_ids": [],
                    "missing_required_net_ids": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_readiness_report_marks_stage_ready_only_after_dependencies_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _touch_all(run_dir, ["tracks.json"])

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["schema_version"] == 1
    assert report["artifact_type"] == "racketsport_pipeline_artifact_readiness"
    assert report["status"] == "not_ready"
    assert report["requested_stage"] == "tracking"
    assert report["stage_order"][:3] == ["calibration", "tracking", "body"]

    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["stage"] == "calibration"
    assert calibration["status"] == "not_ready"
    assert calibration["missing_artifacts"] == [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
    ]
    assert tracking["stage"] == "tracking"
    assert tracking["present_artifacts"] == ["tracks.json"]
    assert tracking["missing_artifacts"] == []
    assert tracking["status"] == "blocked"
    assert tracking["blocked_by"] == ["calibration"]


def test_readiness_report_is_ready_when_requested_stage_and_dependencies_are_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    required = [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
        "tracks.json",
        "smpl_motion.json",
        "skeleton3d.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
    ]
    _touch_all(run_dir, required)
    _write_ready_court_line_evidence(run_dir)
    (run_dir / "contact_windows.json").write_text(
        json.dumps({"schema_version": 1, "artifact_type": "racketsport_contact_windows", "events": [{}]}) + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "ready"
    assert [stage["stage"] for stage in report["stages"]] == PIPELINE_STAGE_ORDER[:6]
    assert report["required_artifacts"] == required
    assert report["missing_artifacts"] == []
    assert all(stage["status"] == "ready" for stage in report["stages"])


def test_readiness_report_blocks_on_semantically_empty_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    required = [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
        "tracks.json",
        "smpl_motion.json",
        "skeleton3d.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
    ]
    _touch_all(run_dir, required)
    _write_ready_court_line_evidence(run_dir)
    (run_dir / "contact_windows.json").write_text(
        json.dumps({"schema_version": 1, "artifact_type": "racketsport_contact_windows", "events": []}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "body_compute_execution.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "scheduled_frames": [],
                "summary": {"scheduled_frame_count": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "body_mesh_readiness.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_mesh_readiness",
                "representation_decision": "no_world_mesh_requested",
                "trusted_for_body_promotion": False,
                "status": "mesh_available_needs_accuracy_gate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "not_ready"
    assert report["missing_artifacts"] == []
    assert report["semantic_blockers"] == [
        "body:body_compute_execution_has_no_scheduled_frames",
        "body:body_mesh_no_world_mesh_requested",
        "body:body_mesh_not_trusted_for_promotion",
        "ball_events:contact_windows_has_no_events",
    ]
    body = next(stage for stage in report["stages"] if stage["stage"] == "body")
    ball_events = next(stage for stage in report["stages"] if stage["stage"] == "ball_events")
    racket = next(stage for stage in report["stages"] if stage["stage"] == "racket")
    assert body["status"] == "blocked"
    assert body["semantic_blockers"] == [
        "body_compute_execution_has_no_scheduled_frames",
        "body_mesh_no_world_mesh_requested",
        "body_mesh_not_trusted_for_promotion",
    ]
    assert ball_events["semantic_blockers"] == ["contact_windows_has_no_events"]
    assert racket["status"] == "blocked"
    assert racket["blocked_by"] == ["physics", "ball_events"]


def test_readiness_report_blocks_when_court_line_evidence_is_not_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase2" / "clip_001"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
        ],
    )
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_court_line_evidence",
                "aggregate": {
                    "auto_calibration_ready": False,
                    "missing_required_line_ids": ["near_nvz"],
                    "missing_required_net_ids": ["top_net"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["status"] == "not_ready"
    assert report["missing_artifacts"] == []
    assert report["semantic_blockers"] == [
        "calibration:court_line_evidence_not_ready",
        "calibration:court_line_evidence_missing_required_line_near_nvz",
        "calibration:court_line_evidence_missing_required_net_top_net",
    ]
    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["status"] == "blocked"
    assert calibration["semantic_blockers"] == [
        "court_line_evidence_not_ready",
        "court_line_evidence_missing_required_line_near_nvz",
        "court_line_evidence_missing_required_net_top_net",
    ]
    assert tracking["status"] == "blocked"
    assert tracking["blocked_by"] == ["calibration"]


def test_readiness_report_skips_retired_burlington_court_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prototype_gate_h100_v2" / "burlington_gold_0300_low_steep_corner"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
        ],
    )
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_court_line_evidence",
                "aggregate": {
                    "auto_calibration_ready": False,
                    "missing_required_line_ids": ["near_nvz"],
                    "missing_required_net_ids": ["top_net"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["semantic_blockers"] == []
    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["status"] == "ready"
    assert tracking["status"] == "ready"


def test_safe_relative_path_rejects_absolute_and_parent_traversal() -> None:
    assert safe_relative_path("clip_001/court_calibration.json") == Path("clip_001/court_calibration.json")

    for value in ["", ".", "/tmp/court_calibration.json", "../court_calibration.json", "clip/../../tracks.json"]:
        with pytest.raises(PipelineContractError):
            safe_relative_path(value)


def test_unknown_stage_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PipelineContractError, match="unknown pipeline stage"):
        build_readiness_report(tmp_path, stage="eval4")


def test_validate_pipeline_artifacts_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase7" / "clip_001"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
            "smpl_motion.json",
            "skeleton3d.json",
        ],
    )
    _write_ready_court_line_evidence(run_dir)
    out = tmp_path / "readiness.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_pipeline_artifacts.py",
            "--run-dir",
            str(run_dir),
            "--stage",
            "metrics",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "not_ready" in completed.stdout

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "not_ready"
    assert payload["requested_stage"] == "metrics"
    assert payload["missing_artifacts"] == [
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
        "racket_sport_metrics.json",
        "habit_report.json",
    ]
    assert payload["stages"][-1]["stage"] == "metrics"
    assert payload["stages"][-1]["status"] == "not_ready"
