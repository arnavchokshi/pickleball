from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.body_full_clip_gate import build_body_full_clip_gate


def _tracks(*, frames: int = 4) -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": index / 30.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [float(index) * 0.1, -3.0],
                        "conf": 0.9,
                    }
                    for index in range(frames)
                ],
            }
        ],
        "rally_spans": [],
    }


def _multi_player_tracks(*, player_count: int = 2, frames: int = 40) -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": player_id,
                "frames": [
                    {
                        "t": index / 30.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [float(index) * 0.1, -3.0 + float(player_id)],
                        "conf": 0.9,
                    }
                    for index in range(frames)
                ],
            }
            for player_id in range(1, player_count + 1)
        ],
        "rally_spans": [],
    }


def _body_compute_execution(*, scheduled_player_frames: int) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "summary": {
            "scheduled_frame_count": scheduled_player_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
        },
    }


def _body_joint_quality(*, joint_frames: int, usable: bool = True, skeleton_joint_frames: int | None = None) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_joint_quality",
        "status": "quality_checked_needs_accuracy_gate" if usable else "quality_blocked",
        "usable_for_review": usable,
        "quality_blockers": [] if usable else ["scheduled_body_output_incomplete"],
        "summary": {
            "joint_frame_count": joint_frames,
            "skeleton_joint_frame_count": skeleton_joint_frames if skeleton_joint_frames is not None else joint_frames,
            "scheduled_player_frame_count": joint_frames,
            "schedule_coverage_ratio": 1.0 if joint_frames else 0.0,
        },
    }


def _contact_splice(
    *,
    scheduled: int = 2,
    spliced: int = 1,
    mesh_unavailable: int = 1,
    fallback_spliced: int = 0,
) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_splice",
        "summary": {
            "scheduled_contact_count": scheduled,
            "spliced_contact_count": spliced,
            "mesh_unavailable_count": mesh_unavailable,
            "fallback_spliced_count": fallback_spliced,
            "overridden_joint_count": spliced + fallback_spliced,
        },
    }


def test_body_full_clip_gate_fails_sparse_body_coverage() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=4),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=2),
        body_joint_quality=_body_joint_quality(joint_frames=2),
        min_coverage=0.95,
    )

    assert payload["artifact_type"] == "racketsport_body_full_clip_gate"
    assert payload["passed"] is False
    assert payload["coverage"] == pytest.approx(0.5)
    assert payload["evaluated_frame_count"] == 2
    assert payload["summary"]["tracked_player_frame_count"] == 4
    assert payload["summary"]["joint_player_frame_count"] == 2
    assert payload["blockers"] == ["full_clip_body_coverage_below_threshold"]


def test_body_full_clip_gate_defaults_to_joint_pipeline_98pct_coverage() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_multi_player_tracks(player_count=2, frames=40),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=80),
        body_joint_quality=_body_joint_quality(joint_frames=78),
    )

    assert payload["passed"] is False
    assert payload["min_coverage"] == pytest.approx(0.98)
    assert payload["coverage"] == pytest.approx(0.975)
    assert "full_clip_body_coverage_below_threshold" in payload["blockers"]


def test_body_full_clip_gate_passes_when_every_tracked_player_frame_has_quality_body_output() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=4),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=4),
        body_joint_quality=_body_joint_quality(joint_frames=4),
        min_coverage=0.95,
    )

    assert payload["passed"] is True
    assert payload["coverage"] == pytest.approx(1.0)
    assert payload["evaluated_frame_count"] == 4
    assert payload["blockers"] == []


def test_body_full_clip_gate_uses_eligible_scheduled_measured_denominator() -> None:
    execution = _body_compute_execution(scheduled_player_frames=2)
    execution["summary"].update(
        {
            "coverage_denominator_policy": "eligible_scheduled_measured_samples",
            "coverage_denominator_player_frame_count": 2,
            "interpolated_player_frame_excluded_count": 1,
        }
    )

    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=3),
        body_compute_execution=execution,
        body_joint_quality=_body_joint_quality(joint_frames=2),
        min_coverage=0.95,
    )

    assert payload["passed"] is True
    assert payload["coverage"] == pytest.approx(1.0)
    assert payload["tracked_player_frame_count"] == 3
    assert payload["coverage_denominator_policy"] == "eligible_scheduled_measured_samples"
    assert payload["coverage_denominator_player_frame_count"] == 2
    assert payload["warnings"] == []


def test_body_full_clip_gate_uses_lane_a_skeleton_coverage_when_mesh_is_contact_only() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_multi_player_tracks(player_count=4, frames=68),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=1),
        body_joint_quality=_body_joint_quality(joint_frames=1, skeleton_joint_frames=272),
        contact_splice=_contact_splice(scheduled=1, spliced=1, mesh_unavailable=0),
    )

    assert payload["passed"] is True
    assert payload["coverage"] == pytest.approx(1.0)
    assert payload["evaluated_frame_count"] == 272
    assert payload["summary"]["joint_player_frame_count"] == 272
    assert payload["summary"]["mesh_joint_player_frame_count"] == 1
    assert payload["summary"]["skeleton_joint_player_frame_count"] == 272
    assert payload["warnings"] == ["body_not_scheduled_for_all_tracked_player_frames"]


def test_body_full_clip_gate_accepts_mesh_unavailable_as_contact_outcome_and_reports_latency() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=60),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=60),
        body_joint_quality=_body_joint_quality(joint_frames=60),
        contact_splice=_contact_splice(scheduled=2, spliced=1, mesh_unavailable=1, fallback_spliced=1),
        runtime_timing={"body_wall_seconds": 6.0},
    )

    assert payload["passed"] is True
    assert payload["contact_mesh_coverage"] == pytest.approx(1.0)
    assert payload["summary"]["scheduled_contact_count"] == 2
    assert payload["summary"]["contact_mesh_frame_count"] == 1
    assert payload["summary"]["mesh_unavailable_contact_count"] == 1
    assert payload["summary"]["fallback_spliced_contact_count"] == 1
    assert payload["latency_seconds_per_video_minute"] == pytest.approx(180.0)
    assert payload["summary"]["clip_duration_s"] == pytest.approx(2.0)
    assert payload["blockers"] == []


def test_body_full_clip_gate_blocks_missing_contact_mesh_or_unavailable_outcome() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=10),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=10),
        body_joint_quality=_body_joint_quality(joint_frames=10),
        contact_splice=_contact_splice(scheduled=3, spliced=1, mesh_unavailable=1),
    )

    assert payload["passed"] is False
    assert payload["contact_mesh_coverage"] == pytest.approx(2.0 / 3.0)
    assert "contact_mesh_or_unavailable_coverage_incomplete" in payload["blockers"]


def test_body_full_clip_gate_fails_when_body_quality_is_blocked() -> None:
    payload = build_body_full_clip_gate(
        clip="clip_001",
        tracks=_tracks(frames=1),
        body_compute_execution=_body_compute_execution(scheduled_player_frames=1),
        body_joint_quality=_body_joint_quality(joint_frames=1, usable=False),
    )

    assert payload["passed"] is False
    assert "body_joint_quality_blocked" in payload["blockers"]


def test_build_body_full_clip_gate_cli_writes_artifact(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    execution = tmp_path / "body_compute_execution.json"
    quality = tmp_path / "body_joint_quality.json"
    contact_splice = tmp_path / "contact_splice.json"
    runtime_timing = tmp_path / "body_runtime_timing.json"
    out = tmp_path / "body_full_clip_gate.json"
    tracks.write_text(json.dumps(_tracks(frames=60)), encoding="utf-8")
    execution.write_text(json.dumps(_body_compute_execution(scheduled_player_frames=60)), encoding="utf-8")
    quality.write_text(json.dumps(_body_joint_quality(joint_frames=60)), encoding="utf-8")
    contact_splice.write_text(json.dumps(_contact_splice(scheduled=2, spliced=1, mesh_unavailable=1)), encoding="utf-8")
    runtime_timing.write_text(json.dumps({"body_wall_seconds": 6.0}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_full_clip_gate.py",
            "--clip",
            "clip_001",
            "--tracks",
            str(tracks),
            "--body-compute-execution",
            str(execution),
            "--body-joint-quality",
            str(quality),
            "--contact-splice",
            str(contact_splice),
            "--runtime-timing",
            str(runtime_timing),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(out.read_text(encoding="utf-8"))
    assert stdout_payload["passed"] is True
    assert stdout_payload["contact_mesh_coverage"] == pytest.approx(1.0)
    assert stdout_payload["latency_seconds_per_video_minute"] == pytest.approx(180.0)
    assert file_payload["artifact_type"] == "racketsport_body_full_clip_gate"
