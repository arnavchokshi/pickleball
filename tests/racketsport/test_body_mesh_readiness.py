from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_mesh_readiness import build_body_mesh_readiness


def _smpl_motion(*, include_mesh: bool = True) -> dict:
    frame: dict[str, object] = {
        "t": 0.0,
        "joints_world": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1]],
        "joint_conf": [0.9, 0.8],
        "transl_world": [0.0, 0.0, 1.0],
    }
    if include_mesh:
        frame["mesh_vertices_world"] = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]
    return {
        "schema_version": 1,
        "fps": 60.0,
        "world_frame": "court",
        "model": "fast_sam_3d_body",
        "players": [{"id": 7, "frames": [frame], "skate_free": False, "physics": {}}],
    }


def _skeleton3d() -> dict:
    return {
        "schema_version": 1,
        "preview_only": True,
        "joint_names": ["pelvis", "neck"],
        "players": [{"id": 7, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1]]}]}],
    }


def _frame_plan(*, deep_mesh_frames: int, player_targets: dict[str, int] | None = None) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 60.0,
        "frame_count": 20,
        "frames": [],
        "deep_mesh_windows": [],
        "summary": {
            "deep_mesh_frame_count": deep_mesh_frames,
            "deep_mesh_window_count": 1 if deep_mesh_frames else 0,
            "human_review_frame_count": 3,
            "by_player_target_representation": player_targets
            or {"manual_review_required": 3, "lane_a_skeleton": 2, "world_mesh": deep_mesh_frames},
        },
    }


def _body_execution(*, scheduled_frames: int, scheduled_player_frames: int) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "mode": "adaptive_frame_compute_plan",
        "scheduled_frames": [],
        "skipped_frames": [],
        "summary": {
            "scheduled_frame_count": scheduled_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
            "scheduled_by_target_representation": {"world_mesh": scheduled_frames} if scheduled_frames else {},
            "skipped_frame_count": 0,
        },
    }


def _full_clip_gate(*, passed: bool = True) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_full_clip_gate",
        "passed": passed,
        "coverage": 1.0 if passed else 0.5,
        "evaluated_frame_count": 1,
        "blockers": [] if passed else ["full_clip_body_coverage_below_threshold"],
    }


def test_body_mesh_readiness_reports_real_mesh_available_but_not_accuracy_verified() -> None:
    payload = build_body_mesh_readiness(
        clip="clip_001",
        smpl_motion=_smpl_motion(include_mesh=True),
        skeleton3d=_skeleton3d(),
    )

    assert payload["artifact_type"] == "racketsport_body_mesh_readiness"
    assert payload["status"] == "mesh_available_needs_accuracy_gate"
    assert payload["world_mesh_available"] is True
    assert payload["trusted_for_body_promotion"] is False
    assert payload["summary"] == {
        "player_count": 1,
        "mesh_player_count": 1,
        "mesh_frame_count": 1,
        "mesh_vertex_count_min": 3,
        "mesh_vertex_count_max": 3,
        "joints_player_count": 1,
        "joints_frame_count": 1,
    }
    assert payload["blockers"] == ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"]
    assert payload["warnings"] == ["mesh_not_accuracy_verified"]


def test_body_mesh_readiness_respects_passing_full_clip_gate() -> None:
    payload = build_body_mesh_readiness(
        clip="clip_001",
        smpl_motion=_smpl_motion(include_mesh=True),
        skeleton3d=_skeleton3d(),
        frame_compute_plan=_frame_plan(deep_mesh_frames=1),
        body_compute_execution=_body_execution(scheduled_frames=1, scheduled_player_frames=1),
        body_full_clip_gate=_full_clip_gate(passed=True),
    )

    assert payload["status"] == "mesh_available_needs_accuracy_gate"
    assert payload["blockers"] == ["missing_world_mpjpe_gate"]
    assert payload["representation_plan"]["blockers"] == ["missing_world_mpjpe_gate"]
    assert payload["body_full_clip_gate_path"] == ""


def test_body_mesh_readiness_rejects_boolean_or_nonfinite_mesh_vertices() -> None:
    boolean_mesh = _smpl_motion(include_mesh=False)
    boolean_mesh["players"][0]["frames"][0]["mesh_vertices_world"] = [[True, False, True]]
    nan_mesh = _smpl_motion(include_mesh=False)
    nan_mesh["players"][0]["frames"][0]["mesh_vertices_world"] = [[0.0, float("nan"), 1.0]]

    for smpl_motion in (boolean_mesh, nan_mesh):
        payload = build_body_mesh_readiness(
            clip="clip_001",
            smpl_motion=smpl_motion,
            skeleton3d=_skeleton3d(),
            frame_compute_plan=_frame_plan(deep_mesh_frames=1),
            body_compute_execution=_body_execution(scheduled_frames=1, scheduled_player_frames=1),
        )

        assert payload["world_mesh_available"] is False
        assert payload["summary"]["mesh_frame_count"] == 0
        assert "world_mesh_required_but_missing" in payload["blockers"]


def test_body_mesh_readiness_compares_world_mesh_demand_to_available_mesh() -> None:
    payload = build_body_mesh_readiness(
        clip="clip_001",
        smpl_motion=_smpl_motion(include_mesh=True),
        skeleton3d=_skeleton3d(),
        frame_compute_plan=_frame_plan(deep_mesh_frames=12),
        body_compute_execution=_body_execution(scheduled_frames=1, scheduled_player_frames=2),
    )

    assert payload["representation_decision"] == "world_mesh_required_available_unverified"
    assert payload["representation_plan"] == {
        "requested_world_mesh_frame_count": 12,
        "requested_world_mesh_player_target_count": 12,
        "scheduled_world_mesh_frame_count": 1,
        "scheduled_world_mesh_player_frame_count": 2,
        "available_mesh_frame_count": 1,
        "available_joint_frame_count": 1,
        "lane_a_skeleton_target_count": 2,
        "manual_review_required_target_count": 3,
        "blockers": [
            "world_mesh_demand_exceeds_available_mesh",
            "missing_world_mpjpe_gate",
            "missing_full_clip_body_gate",
        ],
        "warnings": [
            "mesh_not_accuracy_verified",
            "world_mesh_demand_exceeds_available_mesh",
        ],
    }
    assert "world_mesh_demand_exceeds_available_mesh" in payload["blockers"]
    assert "world_mesh_demand_exceeds_available_mesh" in payload["warnings"]


def test_body_mesh_readiness_reports_no_world_mesh_requested_by_frame_plan() -> None:
    payload = build_body_mesh_readiness(
        clip="clip_001",
        frame_compute_plan=_frame_plan(
            deep_mesh_frames=0,
            player_targets={"manual_review_required": 9, "lane_a_skeleton": 4, "track_only": 2},
        ),
        body_compute_execution=_body_execution(scheduled_frames=0, scheduled_player_frames=0),
    )

    assert payload["status"] == "missing_body_output"
    assert payload["representation_decision"] == "no_world_mesh_requested"
    assert payload["representation_plan"] == {
        "requested_world_mesh_frame_count": 0,
        "requested_world_mesh_player_target_count": 0,
        "scheduled_world_mesh_frame_count": 0,
        "scheduled_world_mesh_player_frame_count": 0,
        "available_mesh_frame_count": 0,
        "available_joint_frame_count": 0,
        "lane_a_skeleton_target_count": 4,
        "manual_review_required_target_count": 9,
        "blockers": [
            "no_trusted_world_mesh_triggers",
            "manual_review_required_before_mesh",
        ],
        "warnings": [
            "world_mesh_not_requested_by_current_frame_plan",
        ],
    }
    assert "no_trusted_world_mesh_triggers" in payload["blockers"]
    assert "world_mesh_not_requested_by_current_frame_plan" in payload["warnings"]


def test_body_mesh_readiness_keeps_joints_only_fail_closed() -> None:
    payload = build_body_mesh_readiness(
        clip="clip_001",
        smpl_motion=_smpl_motion(include_mesh=False),
        skeleton3d=_skeleton3d(),
    )

    assert payload["status"] == "joints_only_no_mesh"
    assert payload["world_mesh_available"] is False
    assert payload["summary"]["mesh_frame_count"] == 0
    assert payload["summary"]["joints_frame_count"] == 1
    assert payload["blockers"] == ["joints_only_no_mesh_vertices", "missing_world_mpjpe_gate"]
    assert payload["warnings"] == ["missing_mesh_vertices", "joints_preview_only"]


def test_body_mesh_readiness_reports_missing_body_output() -> None:
    payload = build_body_mesh_readiness(clip="clip_001")

    assert payload["status"] == "missing_body_output"
    assert payload["world_mesh_available"] is False
    assert payload["summary"]["player_count"] == 0
    assert payload["blockers"] == ["missing_smpl_motion_json", "missing_skeleton3d_json"]
    assert payload["warnings"] == ["missing_mesh_vertices", "missing_body_joints"]


def test_body_mesh_readiness_cli_writes_report(tmp_path: Path) -> None:
    smpl_motion = tmp_path / "smpl_motion.json"
    skeleton = tmp_path / "skeleton3d.json"
    frame_plan = tmp_path / "frame_compute_plan.json"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_mesh_readiness.json"
    smpl_motion.write_text(json.dumps(_smpl_motion(include_mesh=True)), encoding="utf-8")
    skeleton.write_text(json.dumps(_skeleton3d()), encoding="utf-8")
    frame_plan.write_text(json.dumps(_frame_plan(deep_mesh_frames=2)), encoding="utf-8")
    execution.write_text(json.dumps(_body_execution(scheduled_frames=1, scheduled_player_frames=1)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_mesh_readiness.py",
            "--clip",
            "clip_001",
            "--smpl-motion",
            str(smpl_motion),
            "--skeleton3d",
            str(skeleton),
            "--frame-compute-plan",
            str(frame_plan),
            "--body-compute-execution",
            str(execution),
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
    assert payload["status"] == "mesh_available_needs_accuracy_gate"
    assert payload["representation_decision"] == "world_mesh_required_available_unverified"
    assert json.loads(completed.stdout)["status"] == "mesh_available_needs_accuracy_gate"
