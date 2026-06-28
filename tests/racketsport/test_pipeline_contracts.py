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

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "ready"
    assert [stage["stage"] for stage in report["stages"]] == PIPELINE_STAGE_ORDER[:6]
    assert report["required_artifacts"] == required
    assert report["missing_artifacts"] == []
    assert all(stage["status"] == "ready" for stage in report["stages"])


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
