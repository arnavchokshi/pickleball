from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.eval.body_gate_report import (
    CORE_BODY_JOINT_NAMES,
    FOOT_JOINT_NAMES,
    build_body_gate_report,
    render_body_gate_markdown,
)
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES
from threed.racketsport.pose_fast import BODY_17_JOINT_NAMES, FOOT_6_JOINT_NAMES


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _smpl_motion() -> dict:
    return {
        "schema_version": 1,
        "model": "smplx",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 1,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "none",
                "frames": [
                    {
                        "t": 0.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.0, 0.0, 0.0],
                        "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "joint_conf": [0.9, 0.8],
                        "foot_contact": {"left": False, "right": False},
                        "grf": [],
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ],
    }


def _skeleton3d() -> dict:
    return {
        "schema_version": 1,
        "joint_names": ["pelvis", "neck"],
        "preview_only": True,
        "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "joint_conf": [0.9, 0.8]}]}],
    }


def _body_compute_execution(*, scheduled_frames: int = 1, scheduled_player_frames: int = 1) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "scheduled_frames": [],
        "skipped_frames": [],
        "summary": {
            "scheduled_by_reason": {"contact_window": scheduled_frames} if scheduled_frames else {},
            "scheduled_frame_count": scheduled_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
            "scheduled_by_target_representation": {"world_mesh": scheduled_frames} if scheduled_frames else {},
            "skipped_by_reason": {"missing_expected_players": 2},
            "skipped_by_target_representation": {"manual_review_required": 2},
            "skipped_by_tier": {"human_review": 2},
            "skipped_frame_count": 2,
        },
    }


def _body_mesh_readiness(*, mesh_frames: int = 1) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh_readiness",
        "status": "mesh_available_needs_accuracy_gate" if mesh_frames else "missing_body_output",
        "trusted_for_body_promotion": False,
        "summary": {
            "player_count": 1 if mesh_frames else 0,
            "mesh_player_count": 1 if mesh_frames else 0,
            "mesh_frame_count": mesh_frames,
            "mesh_vertex_count_min": 3 if mesh_frames else 0,
            "mesh_vertex_count_max": 3 if mesh_frames else 0,
            "joints_player_count": 1 if mesh_frames else 0,
            "joints_frame_count": mesh_frames,
        },
        "representation_plan": {
            "scheduled_world_mesh_frame_count": mesh_frames,
            "scheduled_world_mesh_player_frame_count": mesh_frames,
            "available_mesh_frame_count": mesh_frames,
        },
        "blockers": ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"] if mesh_frames else ["missing_smpl_motion_json"],
        "warnings": ["mesh_not_accuracy_verified"] if mesh_frames else ["missing_mesh_vertices"],
    }


def _body_grounding_quality(*, max_foot_slide_m: float = 0.0) -> dict:
    passed = max_foot_slide_m <= 0.03
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_grounding_quality",
        "status": "pass" if passed else "fail",
        "clip": "clip_001",
        "foot_slide_gate": {
            "name": "foot_slide_max_m",
            "threshold_m": 0.03,
            "value_m": max_foot_slide_m,
            "passed": passed,
        },
        "blockers": [] if passed else ["foot_slide_gate_failed"],
    }


def _tracks() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_tracks",
        "fps": 30.0,
        "players": [{"id": 1, "frames": [{"t": 0.0, "bbox": [0.0, 0.0, 1.0, 1.0], "world_xy": [0.0, 0.0]}]}],
        "rally_spans": [],
    }


def _person_track_gt_score(tracks_path: Path, *, id_switches: int = 0) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_track_gt_score",
        "clip_id": "clip_001",
        "candidate": "body_tracks",
        "tracks_path": str(tracks_path),
        "id_switches": id_switches,
        "identity_switch_event_count": id_switches,
        "identity_switch_events": [
            {
                "frame_index": 1,
                "gt_track_id": 1,
                "previous_pred_track_id": 1,
                "new_pred_track_id": 2,
            }
        ]
        if id_switches
        else [],
    }


def _person_track_gt_scoring_report(tracks_path: Path, *, id_switches: int = 0) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_track_gt_scoring_report",
        "status": "scored_existing_tracks_only",
        "sources": [
            {
                "track_source_id": "body_tracks",
                "rows": [
                    {
                        "clip_id": "clip_001",
                        "tracks_path": str(tracks_path),
                        "id_switches": id_switches,
                        "identity_switch_event_count": id_switches,
                        "identity_switch_events": [],
                    }
                ],
            }
        ],
    }


def _pipeline_run_with_body_grounding(*, max_foot_slide_m: float = 0.0) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_pipeline_run",
        "stages": [
            {
                "stage": "body",
                "status": "ran",
                "metrics": {
                    "max_foot_lock_slide_m": max_foot_slide_m,
                    "foot_lock_contact_samples": 12,
                },
            }
        ],
    }


def _body_world_label_packet() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_packet",
        "status": "needs_review",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "samples": [
            {
                "sample_id": "frame_000000_player_1",
                "frame_index": 0,
                "player_id": 1,
                "predicted_joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            }
        ],
    }


def _body_packet_quality(*, usable: bool = True, quality_blockers: list[str] | None = None) -> dict:
    blockers = quality_blockers or []
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_joint_quality",
        "status": "quality_checked_needs_accuracy_gate" if usable and not blockers else "quality_blocked",
        "usable_for_review": usable,
        "world_joints_available": True,
        "accuracy_verified": False,
        "trusted_for_body_promotion": False,
        "summary": {
            "joint_source": "body_world_label_packet",
            "joint_frame_count": 1,
            "scheduled_player_frame_count": 1,
            "schedule_coverage_ratio": 1.0,
            "joint_count_min": 2,
            "joint_count_max": 2,
            "max_track_anchor_residual_m": 0.25,
        },
        "quality_blockers": blockers,
        "promotion_blockers": ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"],
        "warnings": [],
    }


def _body_label_review_bundle(*, status: str = "ready_for_review") -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_review_bundle",
        "status": status,
        "clip": "clip_001",
        "source_packet": "runs/clip_001/body_world_label_packet.json",
        "source_video": "source.mp4",
        "suggested_label_path": "labels/body_world_joints.json",
        "queue_path": "runs/clip_001/body_world_label_review_bundle/body_world_label_review_queue.json",
        "label_template_path": "runs/clip_001/body_world_label_review_bundle/body_world_joints.template.json",
        "final_label_path": "runs/clip_001/labels/body_world_joints.json",
        "finalization_report_path": "runs/clip_001/body_world_label_review_bundle/body_world_label_finalization.json",
        "finalize_command": "python scripts/racketsport/finalize_body_world_labels.py --template body_world_joints.template.json --out labels/body_world_joints.json",
        "selected_sample_count": 20,
        "required_sample_count": 20,
        "missing_frame_count": 0,
        "missing_selected_sample_count": 0,
        "missing_selected_sample_ids": [],
        "missing_frames": [],
        "not_ground_truth": True,
    }


def _body_label_template(*, accepted: bool = False) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_joints_labels",
        "status": "draft_review_template",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "clip": "clip_001",
        "selected_sample_ids": ["frame_000000_player_1"],
        "samples": [
            {
                "sample_id": "frame_000000_player_1",
                "frame_index": 0,
                "player_id": 1,
                "accepted": accepted,
                "review_status": "reviewed" if accepted else "needs_review",
                "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] if accepted else [],
                "predicted_joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            }
        ],
    }


def _write_body_run(
    root: Path,
    clip: str,
    *,
    mesh_frames: int = 1,
    grounding_quality: bool = True,
    identity_score: bool = True,
) -> Path:
    run_dir = root / clip
    tracks_path = run_dir / "tracks.json"
    _write_json(tracks_path, _tracks())
    _write_json(run_dir / "smpl_motion.json", _smpl_motion())
    _write_json(run_dir / "skeleton3d.json", _skeleton3d())
    _write_json(
        run_dir / "body_compute_execution.json",
        _body_compute_execution(scheduled_frames=mesh_frames, scheduled_player_frames=mesh_frames),
    )
    _write_json(run_dir / "body_mesh_readiness.json", _body_mesh_readiness(mesh_frames=mesh_frames))
    if grounding_quality:
        _write_json(run_dir / "body_grounding_quality.json", _body_grounding_quality())
    if identity_score:
        _write_json(run_dir / "person_track_gt_score.json", _person_track_gt_score(tracks_path))
    (run_dir / "virtual_world_paddle_preview.html").write_text("<html></html>", encoding="utf-8")
    return run_dir


def test_body_gate_report_keeps_mesh_smoke_unverified_without_labels_or_full_clip_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["mesh_smoke"]["status"] == "pass"
    assert clip["mesh_smoke"]["scheduled_frame_count"] == 1
    assert clip["mesh_smoke"]["mesh_player_frame_count"] == 1
    assert clip["world_mpjpe"]["status"] == "not_measured"
    assert clip["world_mpjpe"]["label_path"] == ""
    assert clip["body_grounding_quality"]["status"] == "pass"
    assert clip["tracking_identity_quality"]["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "not_measured"
    assert set(clip["blockers"]) == {"missing_world_mpjpe_gate", "missing_full_clip_body_gate"}
    assert clip["inspectable_outputs"] == [
        "virtual_world_paddle_preview.html",
        "body_grounding_quality.json",
        "person_track_gt_score.json",
    ]


def test_body_gate_report_blocks_missing_grounding_quality_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001", grounding_quality=False)

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert clip["status"] == "blocked"
    assert clip["body_grounding_quality"]["status"] == "not_measured"
    assert clip["body_grounding_quality"]["blockers"] == ["missing_body_grounding_quality_gate"]
    assert "missing_body_grounding_quality_gate" in clip["blockers"]


def test_body_gate_report_blocks_missing_tracking_identity_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001", identity_score=False)

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert clip["status"] == "blocked"
    assert clip["tracking_identity_quality"]["status"] == "not_measured"
    assert clip["tracking_identity_quality"]["blockers"] == ["missing_person_track_identity_gate"]
    assert "missing_person_track_identity_gate" in clip["blockers"]


def test_body_gate_report_fails_nonzero_tracking_identity_switches(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    tracks_path = run_dir / "tracks.json"
    _write_json(run_dir / "person_track_gt_score.json", _person_track_gt_score(tracks_path, id_switches=1))

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "fail"
    assert clip["status"] == "fail"
    assert clip["tracking_identity_quality"]["status"] == "fail"
    assert clip["tracking_identity_quality"]["id_switches"] == 1
    assert clip["tracking_identity_quality"]["blockers"] == ["person_track_identity_switches_present"]


def test_body_gate_report_accepts_matching_scoring_report_identity_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001", identity_score=False)
    tracks_path = run_dir / "tracks.json"
    _write_json(run_dir / "person_track_gt_scoring_report.json", _person_track_gt_scoring_report(tracks_path))

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    identity = payload["clips"][0]["tracking_identity_quality"]
    assert identity["status"] == "pass"
    assert identity["source"] == "person_track_gt_scoring_report"
    assert identity["id_switches"] == 0


def test_body_gate_report_uses_pipeline_run_body_grounding_metrics_as_legacy_fallback(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001", grounding_quality=False)
    _write_json(run_dir / "pipeline_run.json", _pipeline_run_with_body_grounding(max_foot_slide_m=0.0))

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    grounding = payload["clips"][0]["body_grounding_quality"]
    assert grounding["status"] == "pass"
    assert grounding["source"] == "pipeline_run_body_stage_metrics"
    assert grounding["path"] == str(run_dir / "pipeline_run.json")
    assert grounding["foot_slide_gate"]["value_m"] == 0.0
    assert "missing_body_grounding_quality_gate" not in payload["clips"][0]["blockers"]


def test_body_gate_report_fails_failed_foot_slide_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(run_dir / "body_grounding_quality.json", _body_grounding_quality(max_foot_slide_m=0.04))

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "fail"
    assert clip["status"] == "fail"
    assert clip["body_grounding_quality"]["status"] == "fail"
    assert clip["body_grounding_quality"]["foot_slide_gate"]["value_m"] == 0.04
    assert "foot_slide_gate_failed" in clip["blockers"]


def test_body_gate_report_preserves_body_execution_schedule_breakdown(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    mesh_smoke = payload["clips"][0]["mesh_smoke"]
    assert mesh_smoke["scheduled_by_reason"] == {"contact_window": 1}
    assert mesh_smoke["scheduled_by_target_representation"] == {"world_mesh": 1}
    assert mesh_smoke["skipped_frame_count"] == 2
    assert mesh_smoke["skipped_by_reason"] == {"missing_expected_players": 2}
    assert mesh_smoke["skipped_by_target_representation"] == {"manual_review_required": 2}
    assert mesh_smoke["skipped_by_tier"] == {"human_review": 2}


def test_body_gate_report_computes_world_mpjpe_when_future_labels_exist(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.03, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["status"] == "pass"
    assert mpjpe["sample_count"] == 1
    assert mpjpe["joint_count"] == 2
    assert mpjpe["mean_error_m"] == 0.015
    assert mpjpe["threshold_m"] == 0.05
    assert payload["clips"][0]["blockers"] == ["missing_full_clip_body_gate"]


def test_body_gate_report_uses_objective_world_mpjpe_threshold_by_default(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.06, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert payload["world_mpjpe_threshold_m"] == 0.05
    assert mpjpe["threshold_m"] == 0.05
    assert mpjpe["mean_error_m"] == 0.03
    assert mpjpe["status"] == "pass"


def test_body_gate_report_enforces_wrist_mpjpe_threshold_separately(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "joint_names": ["pelvis", "left_wrist"],
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.04, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["status"] == "fail"
    assert mpjpe["mean_error_m"] == 0.02
    assert mpjpe["wrist_threshold_m"] == 0.03
    assert mpjpe["wrist_mean_error_m"] == 0.04
    assert mpjpe["blockers"] == ["wrist_mpjpe_gate_failed"]


def test_core_body_joint_names_matches_pose_fast_schema() -> None:
    """Drift guard: the gate's core-17 set must track pose_fast's canonical schema."""

    assert CORE_BODY_JOINT_NAMES == BODY_17_JOINT_NAMES
    assert len(CORE_BODY_JOINT_NAMES) == 17
    assert FOOT_JOINT_NAMES == FOOT_6_JOINT_NAMES


def _write_core_schema_run(
    root: Path,
    clip: str,
    *,
    joint_names: list[str],
    predicted_joints: list[list[float]],
) -> Path:
    """Write a minimal BODY run whose prediction uses the real 65/70-joint naming."""

    run_dir = root / clip
    smpl_motion = {
        "schema_version": 1,
        "model": "smplx",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": list(joint_names),
        "players": [
            {
                "id": 1,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "none",
                "frames": [
                    {
                        "t": 0.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.0, 0.0, 0.0],
                        "joints_world": predicted_joints,
                        "joint_conf": [0.9] * len(predicted_joints),
                        "foot_contact": {"left": False, "right": False},
                        "grf": [],
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ],
    }
    _write_json(run_dir / "smpl_motion.json", smpl_motion)
    _write_json(
        run_dir / "body_compute_execution.json",
        _body_compute_execution(scheduled_frames=1, scheduled_player_frames=1),
    )
    return run_dir


def _write_core_schema_labels(labels_root: Path, clip: str, *, joint_names: list[str], joints_world: list[list[float]]) -> None:
    _write_json(
        labels_root / clip / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "joint_names": joint_names,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": joints_world,
                }
            ],
        },
    )


def test_body_gate_report_core_joint_gating_ignores_hand_finger_error_for_pass_fail(tmp_path: Path) -> None:
    """Owner finger policy (2026-07-02): huge finger-joint error must not fail the gate."""

    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    joint_names = list(CORE_BODY_JOINT_NAMES) + ["left_hand_00", "right_hand_00"]
    predicted_joints = [[0.0, 0.0, 0.0] for _ in joint_names]
    _write_core_schema_run(root, "clip_001", joint_names=joint_names, predicted_joints=predicted_joints)
    label_joints = [[0.01, 0.0, 0.0] for _ in CORE_BODY_JOINT_NAMES] + [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    _write_core_schema_labels(labels_root, "clip_001", joint_names=joint_names, joints_world=label_joints)

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["core_joint_gating_enabled"] is True
    assert mpjpe["core_joint_gating_active"] is True
    assert mpjpe["joint_gating_mode"] == "core17_hand_foot_diagnostic"
    assert mpjpe["core_mean_error_m"] == 0.01
    assert mpjpe["core_joint_count"] == 17
    assert mpjpe["wrist_mean_error_m"] == 0.01
    assert mpjpe["hand_mean_error_m"] == 5.0
    assert mpjpe["hand_joint_count"] == 2
    assert mpjpe["status"] == "pass"
    assert mpjpe["blockers"] == []


def test_body_gate_report_core_joint_gating_still_fails_on_core_joint_error(tmp_path: Path) -> None:
    """The core-17 gate is real: a large core-joint error still fails, hand error or not."""

    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    joint_names = list(CORE_BODY_JOINT_NAMES) + ["left_hand_00"]
    predicted_joints = [[0.0, 0.0, 0.0] for _ in joint_names]
    _write_core_schema_run(root, "clip_001", joint_names=joint_names, predicted_joints=predicted_joints)
    label_joints = [[0.2, 0.0, 0.0] for _ in CORE_BODY_JOINT_NAMES] + [[0.0, 0.0, 0.0]]
    _write_core_schema_labels(labels_root, "clip_001", joint_names=joint_names, joints_world=label_joints)

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["core_mean_error_m"] == 0.2
    assert mpjpe["hand_mean_error_m"] == 0.0
    assert mpjpe["status"] == "fail"
    assert "world_mpjpe_gate_failed" in mpjpe["blockers"]


def test_body_gate_report_core_joint_gating_ignores_foot_error_for_pass_fail(tmp_path: Path) -> None:
    """Foot/toe joints sit outside the standard core-17 set — diagnostic-only, not gated."""

    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    joint_names = list(CORE_BODY_JOINT_NAMES) + list(FOOT_JOINT_NAMES)
    predicted_joints = [[0.0, 0.0, 0.0] for _ in joint_names]
    _write_core_schema_run(root, "clip_001", joint_names=joint_names, predicted_joints=predicted_joints)
    label_joints = [[0.01, 0.0, 0.0] for _ in CORE_BODY_JOINT_NAMES] + [[2.0, 0.0, 0.0] for _ in FOOT_JOINT_NAMES]
    _write_core_schema_labels(labels_root, "clip_001", joint_names=joint_names, joints_world=label_joints)

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["foot_mean_error_m"] == 2.0
    assert mpjpe["foot_joint_count"] == 6
    assert mpjpe["core_mean_error_m"] == 0.01
    assert mpjpe["status"] == "pass"
    assert mpjpe["blockers"] == []


def test_body_gate_report_core_joint_gating_can_be_disabled_via_flag(tmp_path: Path) -> None:
    """core_joint_gating_enabled=False reverts to the pre-2026-07-02 legacy behavior."""

    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    joint_names = list(CORE_BODY_JOINT_NAMES) + ["left_hand_00", "right_hand_00"]
    predicted_joints = [[0.0, 0.0, 0.0] for _ in joint_names]
    _write_core_schema_run(root, "clip_001", joint_names=joint_names, predicted_joints=predicted_joints)
    label_joints = [[0.01, 0.0, 0.0] for _ in CORE_BODY_JOINT_NAMES] + [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    _write_core_schema_labels(labels_root, "clip_001", joint_names=joint_names, joints_world=label_joints)

    payload = build_body_gate_report(
        root=root,
        clips=["clip_001"],
        labels_root=labels_root,
        world_mpjpe_threshold_m=0.05,
        core_joint_gating_enabled=False,
    )

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["core_joint_gating_enabled"] is False
    assert mpjpe["core_joint_gating_active"] is False
    assert mpjpe["joint_gating_mode"] == "legacy_all_joints"
    # Legacy body_feet bucket still includes the huge hand-joint error.
    assert mpjpe["body_feet_mean_error_m"] > 0.05
    assert mpjpe["status"] == "fail"
    assert mpjpe["blockers"] == ["world_mpjpe_gate_failed"]


def test_body_gate_report_world_mpjpe_matches_partial_external_gt_labels_by_name_not_index(
    tmp_path: Path,
) -> None:
    """Regression test for review finding F1 (2026-07-02, CRITICAL).

    A partial external-ground-truth label set -- e.g. ASPset-510, whose
    `SHARED_CORE_JOINT_NAMES` supplies only the 12 shared limb joints, in
    `CORE_BODY_JOINT_NAMES` order but starting mid-schema (shoulders onward, no
    face joints) -- must be scored against the matching *named* prediction joints,
    never against whatever prediction joints happen to sit at the same raw index.
    Before this fix, `_joint_errors` zipped `prediction[index]` against `label[index]`
    for `index in range(min(len(prediction), len(label)))`; with a 17-joint
    core-body prediction and this 12-joint label, that silently compared e.g.
    predicted `nose` (index 0) to labeled `left_shoulder` (index 0), corrupting the
    world-MPJPE gate score with meaningless distances.
    """

    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"

    # Prediction: the full 17-joint core-body schema, in CORE_BODY_JOINT_NAMES order.
    # Give every joint a distinct, known position so a positional-vs-named mismatch
    # produces a very different (and easily distinguished) error.
    predicted_joints = [[float(index), 0.0, 0.0] for index in range(len(CORE_BODY_JOINT_NAMES))]
    predicted_by_name = dict(zip(CORE_BODY_JOINT_NAMES, predicted_joints))
    run_dir = _write_core_schema_run(root, "clip_001", joint_names=list(CORE_BODY_JOINT_NAMES), predicted_joints=predicted_joints)
    _write_json(
        run_dir / "body_compute_execution.json",
        _body_compute_execution(scheduled_frames=1, scheduled_player_frames=1),
    )

    # Label: only the 12 joints ASPset-510 actually supplies (SHARED_CORE_JOINT_NAMES),
    # each offset from its *correctly named* prediction joint by exactly +0.01m on x.
    assert set(SHARED_CORE_JOINT_NAMES) < set(CORE_BODY_JOINT_NAMES)
    assert len(SHARED_CORE_JOINT_NAMES) == 12
    label_joints = [
        [predicted_by_name[name][0] + 0.01, 0.0, 0.0] for name in SHARED_CORE_JOINT_NAMES
    ]
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "external_ground_truth",
            "not_ground_truth": False,
            "trusted_for_world_mpjpe": True,
            "joint_names": list(SHARED_CORE_JOINT_NAMES),
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "label_source": "external_ground_truth",
                    "joint_names": list(SHARED_CORE_JOINT_NAMES),
                    "joints_world": label_joints,
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    # Correct, name-matched scoring: every one of the 12 shared joints is off by
    # exactly 0.01m, so mean/core error must be 0.01 and the gate must pass.
    assert mpjpe["joint_count"] == 12
    assert mpjpe["mean_error_m"] == pytest.approx(0.01, abs=1e-9)
    assert mpjpe["core_mean_error_m"] == pytest.approx(0.01, abs=1e-9)
    assert mpjpe["core_joint_count"] == 12
    assert mpjpe["status"] == "pass"
    assert mpjpe["blockers"] == []
    # The old raw-index bug would have zipped label[i] against prediction[i] (both
    # 0-indexed), which -- because SHARED_CORE_JOINT_NAMES starts at
    # CORE_BODY_JOINT_NAMES[5] ("left_shoulder") -- pairs every label joint with a
    # prediction joint 5 slots away, producing a ~5.01m error instead of 0.01m.
    assert mpjpe["mean_error_m"] < 1.0


def test_body_gate_report_can_score_world_mpjpe_from_compact_prediction_packet(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = root / "clip_001"
    tracks_path = run_dir / "tracks.json"
    _write_json(tracks_path, _tracks())
    _write_json(run_dir / "body_compute_execution.json", _body_compute_execution())
    _write_json(run_dir / "body_mesh_readiness.json", _body_mesh_readiness())
    _write_json(run_dir / "body_grounding_quality.json", _body_grounding_quality())
    _write_json(run_dir / "person_track_gt_score.json", _person_track_gt_score(tracks_path))
    _write_json(run_dir / "body_world_label_packet.json", _body_world_label_packet())
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 1,
        },
    )
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.04, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    clip = payload["clips"][0]
    assert clip["status"] == "pass"
    assert clip["blockers"] == []
    assert clip["mesh_smoke"]["status"] == "pass"
    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["status"] == "pass"
    assert mpjpe["prediction_source"] == "body_world_label_packet"
    assert mpjpe["sample_count"] == 1
    assert mpjpe["joint_count"] == 2
    assert mpjpe["mean_error_m"] == 0.02


def test_body_gate_report_blocks_world_mpjpe_when_reviewed_labels_are_too_sparse(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001", mesh_frames=40)
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.02, 0.0, 0.0]],
                }
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 40,
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    clip = payload["clips"][0]
    mpjpe = clip["world_mpjpe"]
    assert clip["status"] == "blocked"
    assert mpjpe["status"] == "not_measured"
    assert mpjpe["mean_error_m"] is None
    assert mpjpe["label_coverage"]["accepted_sample_count"] == 1
    assert mpjpe["label_coverage"]["expected_sample_count"] == 40
    assert mpjpe["label_coverage"]["required_sample_count"] == 20
    assert mpjpe["blockers"] == ["world_mpjpe_label_coverage_too_low"]
    assert clip["blockers"] == ["world_mpjpe_label_coverage_too_low"]


def test_body_gate_report_rejects_draft_world_joint_labels_and_lists_expected_gate_paths(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    label_path = labels_root / "clip_001" / "labels" / "body_world_joints.json"
    _write_json(
        label_path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "draft_prototype_unverified",
            "not_ground_truth": True,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    world = clip["world_mpjpe"]
    assert world["status"] == "not_measured"
    assert world["label_path"] == str(label_path)
    assert world["mean_error_m"] is None
    assert world["label_import"]["status"] == "rejected_not_ground_truth"
    assert world["label_import"]["path"] == str(label_path)
    assert world["label_import"]["payload_status"] == "draft_prototype_unverified"
    assert world["label_import"]["not_ground_truth"] is True
    assert world["label_import"]["accepted_sample_count"] == 1
    assert world["label_import"]["expected_paths"] == [
        str(labels_root / "clip_001" / "body_world_joints.json"),
        str(labels_root / "clip_001" / "body_world_mpjpe.json"),
        str(labels_root / "clip_001" / "labels" / "body_world_joints.json"),
        str(labels_root / "clip_001" / "labels" / "body_world_mpjpe.json"),
    ]
    assert world["blockers"] == ["missing_world_mpjpe_gate", "body_world_labels_not_ground_truth"]
    assert "body_world_labels_not_ground_truth" in clip["blockers"]
    assert clip["full_clip_body_gate"]["expected_paths"] == [
        str(run_dir / "body_full_clip_gate.json"),
        str(labels_root / "clip_001" / "body_full_clip_gate.json"),
        str(labels_root / "clip_001" / "labels" / "body_full_clip_gate.json"),
    ]


def test_body_gate_report_rejects_body_packet_labels_without_independent_source(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "human_reviewed",
            "not_ground_truth": False,
            "trusted_for_world_mpjpe": True,
            "source_packet": "body_world_label_packet.json",
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "label_source": "accepted_candidate_prediction",
                    "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    world = clip["world_mpjpe"]
    assert world["status"] == "not_measured"
    assert world["label_import"]["status"] == "rejected_not_independent_ground_truth"
    assert "accepted_candidate_labels_not_independent_ground_truth" in world["blockers"]
    assert "accepted_candidate_labels_not_independent_ground_truth" in clip["blockers"]


def test_body_gate_report_dedupes_full_clip_expected_paths_when_labels_default_to_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = _write_body_run(root, "clip_001")

    payload = build_body_gate_report(root=root, clips=["clip_001"])

    assert payload["clips"][0]["full_clip_body_gate"]["expected_paths"] == [
        str(run_dir / "body_full_clip_gate.json"),
        str(run_dir / "labels" / "body_full_clip_gate.json"),
    ]


def test_body_gate_report_honors_full_clip_gate_when_present(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 0.97,
            "evaluated_frame_count": 120,
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "pass"
    assert clip["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert clip["blockers"] == []


def test_body_gate_report_preserves_joint_pipeline_full_clip_gate_metrics(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 0.99,
            "evaluated_frame_count": 99,
            "min_coverage": 0.98,
            "contact_mesh_coverage": 1.0,
            "latency_seconds_per_video_minute": 42.5,
            "summary": {
                "scheduled_contact_count": 2,
                "contact_mesh_frame_count": 1,
                "mesh_unavailable_contact_count": 1,
                "fallback_spliced_contact_count": 1,
                "clip_duration_s": 60.0,
            },
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    full = payload["clips"][0]["full_clip_body_gate"]
    assert full["min_coverage"] == pytest.approx(0.98)
    assert full["contact_mesh_coverage"] == pytest.approx(1.0)
    assert full["latency_seconds_per_video_minute"] == pytest.approx(42.5)
    assert full["scheduled_contact_count"] == 2
    assert full["mesh_unavailable_contact_count"] == 1


def test_body_gate_report_blocks_promotion_when_selected_overlay_has_warnings(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 0.97,
            "evaluated_frame_count": 120,
        },
    )
    _write_json(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "ready_for_review_with_overlay_warnings",
            "rendered_count": 1,
            "sample_count": 1,
            "floor_anchor_projection_failed_count": 0,
            "floor_anchor_projection_warning_count": 0,
            "alignment_failed_count": 0,
            "alignment_warning_count": 1,
            "competing_player_warning_count": 0,
            "blockers": [],
            "overlays": [
                {
                    "sample_id": "frame_000000_player_1",
                    "frame_index": 0,
                    "player_id": 1,
                    "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000000_player_1_overlay.jpg",
                    "warnings": ["body_joint_overlay_alignment_warning"],
                    "joint_bbox_alignment": {
                        "status": "warning",
                        "center_delta_px": 40.0,
                        "center_delta_bbox_diag": 0.2,
                        "containment_ratio": 0.4,
                    },
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["world_mpjpe"]["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert clip["body_review_overlay_alignment"]["status"] == "warning"
    assert clip["blockers"] == ["body_joint_overlay_warning_review_required"]


def test_body_gate_report_allows_explicitly_reviewed_overlay_warnings(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 0.97,
            "evaluated_frame_count": 120,
        },
    )
    _write_json(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "ready_for_review_with_overlay_warnings",
            "rendered_count": 1,
            "sample_count": 1,
            "floor_anchor_projection_failed_count": 0,
            "floor_anchor_projection_warning_count": 0,
            "alignment_failed_count": 0,
            "alignment_warning_count": 1,
            "competing_player_warning_count": 0,
            "blockers": [],
            "overlays": [
                {
                    "sample_id": "frame_000000_player_1",
                    "frame_index": 0,
                    "player_id": 1,
                    "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000000_player_1_overlay.jpg",
                    "warnings": ["body_joint_overlay_alignment_warning"],
                    "warning_review_status": "accepted",
                    "warning_review_note": "Reviewer confirmed projected joints stay on the intended player.",
                    "joint_bbox_alignment": {
                        "status": "warning",
                        "center_delta_px": 40.0,
                        "center_delta_bbox_diag": 0.2,
                        "containment_ratio": 0.4,
                    },
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    overlay = clip["body_review_overlay_alignment"]
    assert payload["status"] == "pass"
    assert clip["status"] == "pass"
    assert overlay["status"] == "pass"
    assert overlay["warning_sample_count"] == 1
    assert overlay["resolved_warning_sample_count"] == 1
    assert overlay["unresolved_warning_sample_count"] == 0
    assert overlay["warning_samples"][0]["warning_review_status"] == "accepted"
    assert clip["blockers"] == []


def test_body_gate_report_blocks_failed_compact_packet_quality(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 1,
        },
    )
    _write_json(
        run_dir / "body_joint_quality_from_packet.json",
        _body_packet_quality(usable=False, quality_blockers=["joint_below_court_floor"]),
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    packet_quality = clip["body_packet_quality"]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["world_mpjpe"]["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert packet_quality["status"] == "blocked"
    assert packet_quality["payload_status"] == "quality_blocked"
    assert packet_quality["joint_source"] == "body_world_label_packet"
    assert packet_quality["joint_frame_count"] == 1
    assert packet_quality["blockers"] == ["joint_below_court_floor"]
    assert clip["blockers"] == ["joint_below_court_floor"]
    assert "body_joint_quality_from_packet.json" in clip["inspectable_outputs"]


def test_body_gate_report_surfaces_ready_body_label_review_bundle(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    bundle_dir = run_dir / "body_world_label_review_bundle"
    _write_json(bundle_dir / "body_world_label_review_bundle.json", _body_label_review_bundle())
    _write_json(bundle_dir / "body_world_joints.template.json", _body_label_template())

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    review = clip["body_label_review"]
    assert clip["status"] == "blocked"
    assert review["status"] == "ready_for_review"
    assert review["selected_sample_count"] == 20
    assert review["required_sample_count"] == 20
    assert review["missing_frame_count"] == 0
    assert review["template_status"] == "draft_review_template"
    assert review["template_selected_sample_count"] == 1
    assert review["template_accepted_sample_count"] == 0
    assert review["template_not_ground_truth"] is True
    assert review["template_trusted_for_world_mpjpe"] is False
    assert review["blockers"] == []
    assert clip["blockers"] == ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"]
    assert "body_world_label_review_bundle/body_world_label_review_bundle.json" in clip["inspectable_outputs"]
    assert "body_world_label_review_bundle/body_world_joints.template.json" in clip["inspectable_outputs"]


def test_body_gate_report_blocks_failed_body_label_finalization(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    bundle_dir = run_dir / "body_world_label_review_bundle"
    _write_json(bundle_dir / "body_world_label_review_bundle.json", _body_label_review_bundle())
    _write_json(bundle_dir / "body_world_joints.template.json", _body_label_template(accepted=True))
    _write_json(
        bundle_dir / "body_world_label_finalization.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_finalization",
            "status": "blocked",
            "selected_sample_count": 1,
            "accepted_sample_count": 1,
            "overlay_warning_selected_sample_count": 1,
            "overlay_warning_selected_sample_ids": ["frame_000001_player_7"],
            "blockers": [
                "selected_samples_have_overlay_warnings",
                "template_not_reviewed",
                "template_marked_not_ground_truth",
                "template_not_trusted_for_world_mpjpe",
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    review = clip["body_label_review"]
    assert clip["status"] == "blocked"
    assert review["status"] == "blocked_finalization"
    assert review["finalization_status"] == "blocked"
    assert review["finalization_accepted_sample_count"] == 1
    assert review["finalization_overlay_warning_selected_sample_count"] == 1
    assert review["finalization_overlay_warning_selected_sample_ids"] == ["frame_000001_player_7"]
    assert review["blockers"] == ["body_world_label_finalization_blocked"]
    assert review["finalization_blockers"] == [
        "selected_samples_have_overlay_warnings",
        "template_not_reviewed",
        "template_marked_not_ground_truth",
        "template_not_trusted_for_world_mpjpe",
    ]
    assert "body_world_label_finalization_blocked" in clip["blockers"]


def test_body_gate_report_markdown_surfaces_label_finalization_warning_ids(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    bundle_dir = run_dir / "body_world_label_review_bundle"
    _write_json(bundle_dir / "body_world_label_review_bundle.json", _body_label_review_bundle())
    _write_json(bundle_dir / "body_world_joints.template.json", _body_label_template(accepted=True))
    _write_json(
        bundle_dir / "body_world_label_finalization.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_finalization",
            "status": "blocked",
            "selected_sample_count": 1,
            "accepted_sample_count": 1,
            "overlay_warning_selected_sample_count": 1,
            "overlay_warning_selected_sample_ids": ["frame_000001_player_7"],
            "blockers": [
                "selected_samples_have_overlay_warnings",
                "template_not_reviewed",
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    markdown = render_body_gate_markdown(payload)
    assert "## BODY Label Finalization Blockers" in markdown
    assert "| clip_001 | blocked_finalization | 1 | 1 | selected_samples_have_overlay_warnings, template_not_reviewed | frame_000001_player_7 |" in markdown


def test_body_gate_report_blocks_failed_body_review_overlay_alignment(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 1,
        },
    )
    _write_json(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "rendered_alignment_failed",
            "rendered_count": 1,
            "sample_count": 1,
            "alignment_failed_count": 1,
            "alignment_warning_count": 0,
            "blockers": ["body_joint_overlay_alignment_failed"],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["world_mpjpe"]["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert clip["body_review_overlay_alignment"]["status"] == "blocked"
    assert clip["body_review_overlay_alignment"]["alignment_failed_count"] == 1
    assert clip["blockers"] == ["body_joint_overlay_alignment_failed"]
    assert "body_world_label_review_bundle/overlays/body_world_label_review_overlay_index.json" in clip["inspectable_outputs"]


def test_body_gate_report_surfaces_body_review_overlay_warning_samples(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "ready_for_review_with_alignment_warnings",
            "rendered_count": 2,
            "sample_count": 2,
            "floor_anchor_projection_failed_count": 0,
            "floor_anchor_projection_warning_count": 0,
            "alignment_failed_count": 0,
            "alignment_warning_count": 1,
            "competing_player_warning_count": 1,
            "blockers": [],
            "overlays": [
                {
                    "sample_id": "frame_000047_player_2",
                    "frame_index": 47,
                    "player_id": 2,
                    "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000047_player_2_overlay.jpg",
                    "warnings": ["body_joint_overlay_alignment_warning"],
                    "joint_bbox_alignment": {
                        "status": "warning",
                        "center_delta_px": 40.0,
                        "center_delta_bbox_diag": 0.12,
                        "containment_ratio": 0.5,
                    },
                },
                {
                    "sample_id": "frame_000060_player_1",
                    "frame_index": 60,
                    "player_id": 1,
                    "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000060_player_1_overlay.jpg",
                    "warnings": [
                        "body_joint_overlay_alignment_warning",
                        "body_joint_overlay_competing_player_warning",
                    ],
                    "joint_bbox_alignment": {
                        "status": "warning",
                        "center_delta_px": 118.068,
                        "center_delta_bbox_diag": 0.2866,
                        "containment_ratio": 0.2714,
                    },
                    "competing_player_alignment": {
                        "status": "warning",
                        "best_player_id": 2,
                        "best_player_containment_ratio": 0.9,
                        "score_margin": 0.5,
                    },
                },
                {
                    "sample_id": "frame_000000_player_1",
                    "frame_index": 0,
                    "player_id": 1,
                    "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000000_player_1_overlay.jpg",
                    "warnings": [],
                    "joint_bbox_alignment": {"status": "passed"},
                },
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    overlay = payload["clips"][0]["body_review_overlay_alignment"]
    assert overlay["status"] == "warning"
    assert overlay["competing_player_warning_count"] == 1
    assert overlay["warning_sample_count"] == 2
    assert overlay["warning_samples"] == [
        {
            "sample_id": "frame_000060_player_1",
            "frame_index": 60,
            "player_id": 1,
            "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000060_player_1_overlay.jpg",
            "warnings": [
                "body_joint_overlay_alignment_warning",
                "body_joint_overlay_competing_player_warning",
            ],
            "warning_review_status": "",
            "warning_review_note": "",
            "joint_bbox_alignment_status": "warning",
            "center_delta_px": 118.068,
            "center_delta_bbox_diag": 0.2866,
            "containment_ratio": 0.2714,
            "competing_player_alignment_status": "warning",
            "competing_player_id": 2,
            "competing_player_containment_ratio": 0.9,
            "competing_player_score_margin": 0.5,
        },
        {
            "sample_id": "frame_000047_player_2",
            "frame_index": 47,
            "player_id": 2,
            "overlay_path": "runs/clip_001/body_world_label_review_bundle/overlays/frame_000047_player_2_overlay.jpg",
            "warnings": ["body_joint_overlay_alignment_warning"],
            "warning_review_status": "",
            "warning_review_note": "",
            "joint_bbox_alignment_status": "warning",
            "center_delta_px": 40.0,
            "center_delta_bbox_diag": 0.12,
            "containment_ratio": 0.5,
        }
    ]
    markdown = render_body_gate_markdown(payload)
    assert "## BODY Overlay Warning Samples" in markdown
    assert "| clip_001 | frame_000060_player_1 | body_joint_overlay_alignment_warning, body_joint_overlay_competing_player_warning | 0.2714 | 118.068 | 2 | runs/clip_001/body_world_label_review_bundle/overlays/frame_000060_player_1_overlay.jpg |" in markdown
    assert "| clip_001 | frame_000047_player_2 | body_joint_overlay_alignment_warning | 0.5 | 40.0 | - | runs/clip_001/body_world_label_review_bundle/overlays/frame_000047_player_2_overlay.jpg |" in markdown


def test_body_gate_report_blocks_failed_body_floor_anchor_projection(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 1,
        },
    )
    _write_json(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "rendered_floor_anchor_projection_failed",
            "rendered_count": 1,
            "sample_count": 1,
            "floor_anchor_projection_failed_count": 1,
            "floor_anchor_projection_warning_count": 0,
            "alignment_failed_count": 0,
            "alignment_warning_count": 0,
            "blockers": ["body_floor_anchor_projection_failed"],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["world_mpjpe"]["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert clip["body_review_overlay_alignment"]["status"] == "blocked"
    assert clip["body_review_overlay_alignment"]["floor_anchor_projection_failed_count"] == 1
    assert clip["blockers"] == ["body_floor_anchor_projection_failed"]


def test_body_gate_report_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    out_json = tmp_path / "body_gate_report.json"
    out_md = tmp_path / "body_gate_report.md"
    _write_body_run(root, "clip_001")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_gate_report.py",
            "--root",
            str(root),
            "--clip",
            "clip_001",
            "--out",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--world-wrist-mpjpe-threshold-m",
            "0.02",
            "--write-clip-reports",
            "--allow-not-verified",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert payload["status"] == "blocked"
    assert payload["world_wrist_mpjpe_threshold_m"] == 0.02
    assert "- world_wrist_mpjpe_threshold_m: `0.02`" in markdown
    assert "| clip_001 | blocked | pass | not_measured | not_measured | pass | pass | not_measured |" in markdown
    assert "virtual_world_paddle_preview.html" in markdown
    assert (root / "clip_001" / "body_gate_report.json").is_file()
    assert (root / "clip_001" / "body_gate_report.md").is_file()


def test_body_gate_report_default_discovery_skips_non_clip_run_dirs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_body_run(root, "clip_001")
    (root / "review_bundle").mkdir(parents=True)
    (root / "ball_tracker_benchmark").mkdir(parents=True)
    (root / "__pycache__").mkdir(parents=True)

    payload = build_body_gate_report(root=root)

    assert [clip["clip"] for clip in payload["clips"]] == ["clip_001"]
