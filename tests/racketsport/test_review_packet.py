from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.review_packet import build_review_packet, write_review_packet


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_review_packet_summarizes_pipeline_runs_and_review_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate"
    pipeline = _write_json(
        run_root / "clip_001" / "pipeline_run.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_pipeline_run",
            "clip": "clip_001",
            "requested_stage": "e2e",
            "status": "fail",
            "stages": [
                {"stage": "calibration", "status": "ran", "notes": []},
                {"stage": "tracking", "status": "ran", "notes": []},
                {"stage": "body", "status": "fail", "notes": ["missing checkpoint for fast_sam_3d_body_dinov3"]},
            ],
        },
    )
    overlay = run_root / "clip_001" / "compare" / "all_labels_overlay.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"mp4")
    _write_json(
        overlay.parent / "label_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_label_overlay",
            "clip": "clip_001",
            "status": "rendered",
            "rendered_videos": [str(overlay)],
            "available_layers": ["court", "players", "ball", "racket"],
            "warnings": ["events.json not present"],
            "qualitative_status": "prototype_not_gate_verified",
        },
    )

    packet = build_review_packet(
        run_root,
        packet_id="prototype_gate_review",
        corrections_root=tmp_path / "corrections" / "inbox",
    )

    assert packet["schema_version"] == 1
    assert packet["artifact_type"] == "racketsport_human_review_packet"
    assert packet["packet_id"] == "prototype_gate_review"
    assert packet["run_root"] == str(run_root)
    assert packet["pipeline_run_count"] == 1
    assert packet["pipeline_runs"][0]["path"] == str(pipeline)
    assert packet["pipeline_runs"][0]["status"] == "fail"
    assert packet["pipeline_runs"][0]["failed_stage"] == "body"
    assert packet["pipeline_runs"][0]["notes"] == ["missing checkpoint for fast_sam_3d_body_dinov3"]
    assert packet["review_artifact_count"] == 1
    assert packet["review_artifacts"][0]["path"] == str(overlay.parent / "label_overlay_index.json")
    assert packet["review_artifacts"][0]["watch_paths"] == [str(overlay)]
    assert packet["human_next_steps"][0].startswith("For the fastest browser UI")
    assert packet["human_next_steps"][1].startswith("Open the listed review artifacts")
    assert packet["corrections_manifest_template"] == str(
        tmp_path / "corrections" / "inbox" / "prototype_gate_review_corrections.json"
    )


def test_review_packet_prefers_existing_sibling_watch_path_when_index_has_stale_root(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    compare_dir = run_root / "clip_001" / "compare"
    actual_overlay = compare_dir / "all_labels_overlay.mp4"
    actual_overlay.parent.mkdir(parents=True)
    actual_overlay.write_bytes(b"mp4")
    _write_json(
        compare_dir / "label_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_label_overlay",
            "clip": "clip_001",
            "status": "rendered",
            "rendered_videos": [
                str(tmp_path / "runs" / "eval0" / "prototype_gate" / "clip_001" / "compare" / "all_labels_overlay.mp4")
            ],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    assert packet["review_artifacts"][0]["watch_paths"] == [str(actual_overlay)]


def test_review_packet_includes_frame_compute_plan_summaries(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "frame_compute_plan.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_frame_compute_plan",
            "fps": 60.0,
            "expected_players": 4,
            "frame_count": 12,
            "frames": [],
            "deep_mesh_windows": [
                {
                    "frame_start": 20,
                    "frame_end": 31,
                    "t0": 0.3333333333333333,
                    "t1": 0.5333333333333333,
                    "frame_count": 12,
                    "target_representation": "world_mesh",
                    "fallback_representation": "skeleton_preview",
                    "target_player_ids": [1, 2],
                    "reason_counts": {"contact_window": 12},
                    "max_score": 0.85,
                }
            ],
            "summary": {
                "by_tier": {"baseline": 2, "deep_mesh": 12, "human_review": 10},
                "by_reason": {"contact_window": 12, "missing_expected_players": 10},
                "by_player_target_representation": {"manual_review_required": 2, "world_mesh": 8},
                "max_score": 0.85,
                "deep_mesh_window_count": 1,
                "deep_mesh_frame_count": 12,
                "human_review_frame_count": 10,
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")
    markdown = write_review_packet(packet, out_dir=tmp_path / "packet")["markdown_path"]

    artifact = packet["review_artifacts"][0]
    assert artifact["clip"] == "clip_001"
    assert artifact["details"] == [
        "Frames planned: 12",
        "Tiers: baseline=2, deep_mesh=12, human_review=10",
        "Reasons: contact_window=12, missing_expected_players=10",
        "Player targets: manual_review_required=2, world_mesh=8",
        "Deep mesh windows: 1 (12 frames)",
    ]
    assert "Frames planned: 12" in Path(markdown).read_text(encoding="utf-8")


def test_review_packet_includes_body_compute_execution_summaries(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "source_plan": str(run_root / "clip_001" / "frame_compute_plan.json"),
            "fps": 60.0,
            "scheduled_frames": [
                {
                    "frame_idx": 20,
                    "t": 20.0 / 60.0,
                    "target_representation": "world_mesh",
                    "target_player_ids": [1, 2],
                    "active_player_ids": [1, 2],
                    "source_window_index": 0,
                    "reason_counts": {"contact_window": 1},
                    "reasons": ["contact_window"],
                    "max_score": 0.85,
                }
            ],
            "skipped_frames": [
                {
                    "frame_idx": 19,
                    "t": 19.0 / 60.0,
                    "recommended_tier": "human_review",
                    "target_representation": "manual_review_required",
                    "skip_reason": "manual_review_required",
                    "reasons": ["missing_expected_players"],
                    "active_player_ids": [1],
                }
            ],
            "summary": {
                "scheduled_frame_count": 1,
                "scheduled_player_frame_count": 2,
                "skipped_frame_count": 1,
                "skipped_by_tier": {"human_review": 1},
                "skipped_by_target_representation": {"manual_review_required": 1},
                "skipped_by_reason": {"missing_expected_players": 1},
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")
    markdown = write_review_packet(packet, out_dir=tmp_path / "packet")["markdown_path"]

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_body_compute_execution"
    assert artifact["status"] == "scheduled"
    assert artifact["details"] == [
        "Mode: adaptive_frame_compute_plan",
        "Scheduled frames: 1",
        "Scheduled player-frames: 2",
        "Skipped frames: 1",
        "Skipped tiers: human_review=1",
        "Skipped targets: manual_review_required=1",
        "Skipped reasons: missing_expected_players=1",
    ]
    assert "Scheduled player-frames: 2" in Path(markdown).read_text(encoding="utf-8")


def test_review_packet_warns_when_body_schedule_targets_reviewed_contact_with_incomplete_coverage(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "source_plan": str(run_root / "clip_001" / "frame_compute_plan.json"),
            "fps": 60.0,
            "scheduled_frames": [
                {
                    "frame_idx": 20,
                    "t": 20.0 / 60.0,
                    "target_representation": "world_mesh",
                    "target_player_ids": [1],
                    "active_player_ids": [1],
                    "source_window_index": 0,
                    "reason_counts": {
                        "contact_window": 1,
                        "missing_expected_players": 1,
                        "reviewed_contact_targeted_body": 1,
                    },
                    "reasons": ["contact_window", "missing_expected_players", "reviewed_contact_targeted_body"],
                    "max_score": 0.55,
                }
            ],
            "skipped_frames": [],
            "summary": {
                "scheduled_frame_count": 1,
                "scheduled_player_frame_count": 1,
                "scheduled_by_target_representation": {"world_mesh": 1},
                "scheduled_by_reason": {
                    "contact_window": 1,
                    "missing_expected_players": 1,
                    "reviewed_contact_targeted_body": 1,
                },
                "scheduled_targeted_reviewed_contact_frame_count": 1,
                "scheduled_coverage_incomplete_frame_count": 1,
                "skipped_frame_count": 0,
                "skipped_by_tier": {},
                "skipped_by_target_representation": {},
                "skipped_by_reason": {},
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_body_compute_execution"
    assert artifact["warnings"] == [
        "targeted_reviewed_contact_body_schedule",
        "scheduled_with_incomplete_player_coverage",
    ]
    assert artifact["details"] == [
        "Mode: adaptive_frame_compute_plan",
        "Scheduled frames: 1",
        "Scheduled player-frames: 1",
        "Skipped frames: 0",
        "Scheduled targets: world_mesh=1",
        "Scheduled reasons: contact_window=1, missing_expected_players=1, reviewed_contact_targeted_body=1",
        "Scheduled targeted reviewed-contact frames: 1",
        "Scheduled incomplete-coverage frames: 1",
    ]


def test_review_packet_includes_body_mesh_readiness_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "body_mesh_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "clip": "clip_001",
            "status": "mesh_available_needs_accuracy_gate",
            "world_mesh_available": True,
            "trusted_for_body_promotion": False,
            "smpl_motion_path": "runs/eval0/clip_001/smpl_motion.json",
            "skeleton3d_path": "runs/eval0/clip_001/skeleton3d.json",
            "summary": {
                "player_count": 4,
                "mesh_player_count": 4,
                "mesh_frame_count": 4,
                "mesh_vertex_count_min": 18439,
                "mesh_vertex_count_max": 18439,
                "joints_player_count": 4,
                "joints_frame_count": 4,
            },
            "blockers": ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"],
            "warnings": ["mesh_not_accuracy_verified"],
            "execution": {
                "cpu_only": True,
                "uses_gpu": False,
                "runs_body_model": False,
                "creates_synthetic_mesh_from_joints": False,
                "claims_accuracy_verified": False,
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_body_mesh_readiness"
    assert artifact["status"] == "mesh_available_needs_accuracy_gate"
    assert artifact["warnings"] == [
        "mesh_not_accuracy_verified",
        "missing_world_mpjpe_gate",
        "missing_full_clip_body_gate",
    ]
    assert artifact["details"] == [
        "World mesh available: true",
        "Trusted for BODY promotion: false",
        "Players: 4",
        "Mesh players: 4",
        "Mesh frames: 4",
        "Mesh vertices/frame: 18439-18439",
        "Joints players: 4",
        "Joints frames: 4",
    ]


def test_review_packet_includes_body_representation_decision(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "body_mesh_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "clip": "clip_001",
            "status": "missing_body_output",
            "world_mesh_available": False,
            "representation_decision": "no_world_mesh_requested",
            "trusted_for_body_promotion": False,
            "summary": {
                "player_count": 0,
                "mesh_player_count": 0,
                "mesh_frame_count": 0,
                "mesh_vertex_count_min": 0,
                "mesh_vertex_count_max": 0,
                "joints_player_count": 0,
                "joints_frame_count": 0,
            },
            "representation_plan": {
                "requested_world_mesh_frame_count": 0,
                "requested_world_mesh_player_target_count": 0,
                "scheduled_world_mesh_frame_count": 0,
                "scheduled_world_mesh_player_frame_count": 0,
                "available_mesh_frame_count": 0,
                "available_joint_frame_count": 0,
                "joints_or_preview_mesh_target_count": 4,
                "manual_review_required_target_count": 9,
                "blockers": ["no_trusted_world_mesh_triggers", "manual_review_required_before_mesh"],
                "warnings": ["world_mesh_not_requested_by_current_frame_plan"],
            },
            "blockers": ["missing_smpl_motion_json", "missing_skeleton3d_json", "no_trusted_world_mesh_triggers"],
            "warnings": ["missing_mesh_vertices", "world_mesh_not_requested_by_current_frame_plan"],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_body_mesh_readiness"
    assert artifact["details"] == [
        "World mesh available: false",
        "Trusted for BODY promotion: false",
        "Players: 0",
        "Mesh players: 0",
        "Mesh frames: 0",
        "Mesh vertices/frame: 0-0",
        "Joints players: 0",
        "Joints frames: 0",
        "Representation decision: no_world_mesh_requested",
        "World mesh demand: requested=0, scheduled=0, available=0",
        "Representation targets: joints_or_preview_mesh=4, manual_review_required=9, world_mesh=0",
    ]
    assert "world_mesh_not_requested_by_current_frame_plan" in artifact["warnings"]


def test_review_packet_includes_pipeline_readiness_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "pipeline_readiness_e2e.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_pipeline_artifact_readiness",
            "run_dir": str(run_root / "clip_001"),
            "requested_stage": "e2e",
            "status": "not_ready",
            "required_artifacts": ["tracks.json", "racket_pose.json"],
            "missing_artifacts": ["racket_pose.json"],
            "semantic_blockers": ["body:body_mesh_no_world_mesh_requested"],
            "stages": [
                {"stage": "tracking", "status": "ready"},
                {
                    "stage": "body",
                    "status": "blocked",
                    "semantic_blockers": ["body_mesh_no_world_mesh_requested"],
                },
                {"stage": "racket", "status": "not_ready"},
            ],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_pipeline_artifact_readiness"
    assert artifact["status"] == "not_ready"
    assert artifact["warnings"] == [
        "pipeline_not_ready",
        "missing:racket_pose.json",
        "semantic:body:body_mesh_no_world_mesh_requested",
    ]
    assert artifact["details"] == [
        "Requested stage: e2e",
        "Missing artifacts: racket_pose.json",
        "Semantic blockers: body:body_mesh_no_world_mesh_requested",
        "Stages: ready=1, not_ready=1, blocked=1",
    ]


def test_review_packet_includes_court_line_evidence_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "court_line_evidence.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "source": "auto_hough_template_video",
            "aggregate": {
                "accepted_line_ids": ["far_baseline", "near_baseline", "net"],
                "auto_calibration_ready": False,
                "mean_residual_px": 2.7931,
                "missing_required_line_ids": ["near_nvz"],
                "missing_required_net_ids": ["top_net"],
                "p95_residual_px": 5.1816,
                "reasons": ["missing_near_nvz", "missing_top_net"],
            },
            "line_observations": [],
            "net_observations": [],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_court_line_evidence"
    assert artifact["status"] == "blocked"
    assert artifact["warnings"] == ["court_evidence_not_ready", "missing_line:near_nvz", "missing_net:top_net"]
    assert artifact["details"] == [
        "Source: auto_hough_template_video",
        "Auto calibration ready: false",
        "Accepted lines: 3 (far_baseline, near_baseline, net)",
        "Missing required lines: near_nvz",
        "Missing required net cues: top_net",
        "Residual px: mean=2.79, p95=5.18",
        "Reasons: missing_near_nvz, missing_top_net",
    ]


def test_review_packet_includes_ball_inflection_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "ball_inflections.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_inflections",
            "source": "virtual_world_court_plane_ball_path",
            "world_frame": "court_Z0",
            "not_gate_verified": True,
            "requires_additional_cues": ["audio_onsets", "wrist_velocity_peaks"],
            "summary": {
                "usable_frame_count": 288,
                "candidate_count": 13,
                "raw_candidate_count": 56,
                "min_turn_degrees": 45.0,
                "min_speed_mps": 0.75,
                "min_candidate_separation_s": 0.15,
            },
            "candidates": [],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_ball_inflections"
    assert artifact["status"] == "review_only"
    assert artifact["warnings"] == [
        "review_only_not_gate_verified",
        "missing_cue:audio_onsets",
        "missing_cue:wrist_velocity_peaks",
    ]
    assert artifact["details"] == [
        "Candidates: 13",
        "Raw candidates before suppression: 56",
        "Usable ball frames: 288",
        "Source: virtual_world_court_plane_ball_path",
        "Requires additional cues: audio_onsets, wrist_velocity_peaks",
        "Thresholds: turn>=45.0deg, speed>=0.75mps, separation>=0.15s",
    ]


def test_review_packet_includes_audio_and_wrist_cue_blockers(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "audio_onsets.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_audio_onsets",
            "status": "blocked",
            "source": "video_audio_stream",
            "source_path": str(run_root / "clip_001" / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"),
            "sample_rate_hz": None,
            "not_gate_verified": True,
            "trusted_for_contact": False,
            "blockers": ["no_audio_stream"],
            "warnings": ["audio_stream_missing"],
            "summary": {"onset_count": 0, "raw_peak_count": 0, "duration_s": 0.0},
            "onsets": [],
        },
    )
    _write_json(
        run_root / "clip_001" / "wrist_velocity_peaks.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_wrist_velocity_peaks",
            "status": "blocked",
            "source": "skeleton3d_world_joints",
            "source_path": str(run_root / "clip_001" / "skeleton3d.json"),
            "not_gate_verified": True,
            "trusted_for_contact": False,
            "joint_mapping": {},
            "blockers": ["missing_wrist_joint_mapping"],
            "warnings": ["missing_wrist_joint_mapping"],
            "summary": {
                "player_count": 1,
                "usable_sample_count": 0,
                "raw_peak_count": 0,
                "peak_count": 0,
                "min_speed_mps": 4.0,
            },
            "peaks": [],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifacts = {artifact["artifact_type"]: artifact for artifact in packet["review_artifacts"]}
    assert artifacts["racketsport_audio_onsets"]["status"] == "blocked"
    assert artifacts["racketsport_audio_onsets"]["warnings"] == [
        "audio_stream_missing",
        "cue_not_gate_verified",
        "not_trusted_for_contact",
        "no_audio_stream",
    ]
    assert artifacts["racketsport_audio_onsets"]["details"] == [
        "Onsets: 0",
        "Raw peaks before suppression: 0",
        "Source: video_audio_stream",
        "Sample rate: unavailable",
        "Blockers: no_audio_stream",
    ]
    assert artifacts["racketsport_wrist_velocity_peaks"]["status"] == "blocked"
    assert artifacts["racketsport_wrist_velocity_peaks"]["warnings"] == [
        "missing_wrist_joint_mapping",
        "cue_not_gate_verified",
        "not_trusted_for_contact",
    ]
    assert artifacts["racketsport_wrist_velocity_peaks"]["details"] == [
        "Peaks: 0",
        "Raw peaks before suppression: 0",
        "Usable wrist samples: 0",
        "Source: skeleton3d_world_joints",
        "Joint mapping: unavailable",
        "Thresholds: speed>=4.00mps",
        "Blockers: missing_wrist_joint_mapping",
    ]


def test_review_packet_includes_contact_window_candidate_summaries(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "contact_window_candidates.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_contact_window_candidates",
            "clip": "clip_001",
            "fps": 60.0,
            "source_event_path": str(run_root / "clip_001" / "labels" / "events.json"),
            "not_gate_verified": True,
            "trusted_for_body": False,
            "promotion_target": "contact_windows.json",
            "candidates": [
                {
                    "review_id": "event_smoke_contact",
                    "type": "contact",
                    "frame": 89,
                    "t": 89.0 / 60.0,
                    "xy_px": [960.0, 540.0],
                    "source_label": "contact?",
                    "source_status": "uncertain",
                    "source_confidence": 0.2,
                    "candidate_confidence": 0.2,
                    "window": {"t0": 1.4033333333333333, "t1": 1.5633333333333332, "importance": 0.2},
                }
            ],
            "summary": {
                "candidate_count": 1,
                "rejected_item_count": 0,
                "by_type": {"contact": 1},
                "by_status": {"uncertain": 1},
                "uncertainty_flags": ["teacher_model_unavailable", "smoke_generated"],
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_contact_window_candidates"
    assert artifact["status"] == "needs_review"
    assert artifact["warnings"] == ["review_only_not_gate_verified", "not_trusted_for_body"]
    assert artifact["details"] == [
        "Candidates: 1",
        "Rejected source items: 0",
        "Types: contact=1",
        "Statuses: uncertain=1",
        "Trusted for BODY: false",
        "Promotion target: contact_windows.json",
        "Uncertainty flags: teacher_model_unavailable, smoke_generated",
    ]


def test_review_packet_includes_contact_window_review_templates(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    html = run_root / "clip_001" / "contact_window_review.html"
    _write_json(
        run_root / "clip_001" / "contact_window_review.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_contact_window_review",
            "clip": "clip_001",
            "candidate_path": str(run_root / "clip_001" / "contact_window_candidates.json"),
            "promotion_target": "contact_windows.json",
            "status": "pending_review",
            "decisions": [
                {
                    "review_id": "event_smoke_contact",
                    "decision": "pending",
                    "reviewer": "",
                    "reason": "",
                    "player_id": None,
                    "t_override": None,
                    "frame_override": None,
                    "confidence_override": None,
                    "window_override": None,
                }
            ],
            "summary": {
                "candidate_count": 1,
                "pending_count": 1,
                "accepted_count": 0,
                "rejected_count": 0,
            },
        },
    )
    html.write_text("<!doctype html><title>Contact Window Review</title>", encoding="utf-8")

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_contact_window_review"
    assert artifact["status"] == "pending_review"
    assert artifact["warnings"] == ["pending_contact_review"]
    assert artifact["watch_paths"] == [str(html)]
    assert artifact["details"] == [
        "Candidates: 1",
        "Pending: 1",
        "Accepted: 0",
        "Rejected: 0",
        "Promotion target: contact_windows.json",
    ]


def test_review_packet_includes_promoted_contact_windows_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "contact_windows.json",
        {
            "schema_version": 1,
            "events": [
                {
                    "type": "contact",
                    "t": 89.0 / 60.0,
                    "frame": 89,
                    "player_id": 7,
                    "confidence": 0.95,
                    "sources": {
                        "audio": 0.0,
                        "wrist_vel": 0.0,
                        "ball_inflection": 0.0,
                        "human_review": 1.0,
                    },
                    "window": {"t0": 1.44, "t1": 1.55, "importance": 0.95},
                }
            ],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_contact_windows"
    assert artifact["status"] == "promoted"
    assert artifact["warnings"] == []
    assert artifact["details"] == [
        "Events: 1",
        "Types: contact=1",
        "Human-reviewed events: 1",
        "Frame range: 89-89",
        "Time range: 1.483-1.483s",
    ]


def test_review_packet_includes_racket_pose_readiness_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "racket_pose_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_pose_readiness",
            "clip": "clip_001",
            "status": "blocked_preview_only",
            "source_counts": {"label_bbox:manual": 4},
            "blockers": [
                "box_derived_candidate_corners",
                "missing_true_paddle_keypoints_or_cad_pose",
                "missing_promoted_racket_pose_json",
                "missing_reference_pose_gt",
                "missing_racket_pose_evaluation",
            ],
            "recommended_next_actions": ["collect true paddle corner labels or CAD/reference pose evidence"],
            "source_evidence_counts": {
                "box_derived": 4,
                "keypoint_or_mask": 0,
                "reference_gt": 0,
                "synthetic_or_cad": 0,
                "true_corners_or_pose": 0,
            },
            "summary": {
                "candidate_player_count": 1,
                "candidate_frame_count": 4,
                "box_derived_frame_count": 4,
                "true_corner_frame_count": 0,
                "reference_gt_frame_count": 0,
                "preview_pose_frame_count": 4,
                "promoted_pose_frame_count": 0,
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_racket_pose_readiness"
    assert artifact["status"] == "blocked_preview_only"
    assert artifact["warnings"] == [
        "box_derived_candidate_corners",
        "missing_true_paddle_keypoints_or_cad_pose",
        "missing_promoted_racket_pose_json",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]
    assert artifact["details"] == [
        "Candidate frames: 4",
        "Box-derived frames: 4",
        "True corner/reference frames: 0",
        "Reference/GT frames: 0",
        "Preview pose frames: 4",
        "Promoted pose frames: 0",
        "Source evidence: box_derived=4, keypoint_or_mask=0, reference_gt=0, synthetic_or_cad=0, true_corners_or_pose=0",
        "Sources: label_bbox:manual=4",
    ]


def test_review_packet_includes_racket_promotion_audit_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "racket_promotion_audit.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_promotion_audit",
            "clip": "clip_001",
            "status": "safe_preview_only",
            "canonical_racket_pose_present": False,
            "trusted_for_rkt_promotion": False,
            "not_gate_verified": True,
            "source_counts": {"label_bbox:manual": 4},
            "pose_source_counts": {},
            "unsafe_promoted_sources": {},
            "source_evidence_counts": {
                "box_derived": 4,
                "keypoint_or_mask": 0,
                "reference_gt": 0,
                "synthetic_or_cad": 0,
                "true_corners_or_pose": 0,
            },
            "summary": {
                "candidate_frame_count": 4,
                "box_derived_candidate_frame_count": 4,
                "true_corner_frame_count": 0,
                "reference_gt_frame_count": 0,
                "preview_pose_frame_count": 4,
                "promoted_pose_frame_count": 0,
                "unsafe_promoted_frame_count": 0,
            },
            "blockers": [
                "missing_true_paddle_keypoints_or_cad_pose",
                "missing_promoted_racket_pose_json",
                "missing_reference_pose_gt",
                "missing_racket_pose_evaluation",
            ],
            "warnings": ["canonical_racket_pose_missing", "preview_only_not_gate_verified"],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_racket_promotion_audit"
    assert artifact["status"] == "safe_preview_only"
    assert artifact["warnings"] == [
        "canonical_racket_pose_missing",
        "preview_only_not_gate_verified",
        "missing_true_paddle_keypoints_or_cad_pose",
        "missing_promoted_racket_pose_json",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]
    assert artifact["details"] == [
        "Canonical racket_pose.json present: false",
        "Trusted for RKT promotion: false",
        "Candidate frames: 4",
        "Box-derived candidate frames: 4",
        "True corner/reference frames: 0",
        "Reference/GT frames: 0",
        "Preview pose frames: 4",
        "Promoted pose frames: 0",
        "Unsafe promoted frames: 0",
        "Source evidence: box_derived=4, keypoint_or_mask=0, reference_gt=0, synthetic_or_cad=0, true_corners_or_pose=0",
        "Candidate sources: label_bbox:manual=4",
    ]


def test_review_packet_includes_global_racket_runtime_readiness_with_clip_filter(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "racket_model_runtime_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_model_runtime_readiness",
            "stage": "racket_6dof",
            "status": "blocked",
            "execution": {
                "cpu_only": True,
                "uses_gpu": False,
                "downloads_models": False,
                "imports_model_runtimes": False,
                "runs_inference": False,
                "claims_model_has_run": False,
                "mutates_model_manifest": False,
            },
            "summary": {
                "component_count": 6,
                "runtime_ready_count": 0,
                "asset_ready": False,
                "may_run_gpu_smoke": False,
                "may_promote_rkt": False,
            },
            "blockers": ["sam3_concept_tracker:missing_manifest_entry"],
            "components": [],
            "asset_readiness": {},
        },
    )
    _write_json(
        run_root / "clip_001" / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "summary": {"scheduled_frame_count": 0, "scheduled_player_frame_count": 0, "skipped_frame_count": 0},
            "scheduled_frames": [],
            "skipped_frames": [],
        },
    )

    packet = build_review_packet(
        run_root,
        packet_id="prototype_gate_h100_v2_review",
        include_clips=["clip_001"],
    )

    artifacts = {artifact["artifact_type"]: artifact for artifact in packet["review_artifacts"]}
    artifact = artifacts["racketsport_racket_model_runtime_readiness"]
    assert artifact["clip"] == "__global__"
    assert artifact["status"] == "blocked"
    assert artifact["warnings"] == ["sam3_concept_tracker:missing_manifest_entry"]
    assert artifact["details"] == [
        "Components: 6",
        "Runtime-ready components: 0",
        "Asset ready: false",
        "May run GPU smoke: false",
        "May promote RKT: false",
        "Execution: cpu_only=true, uses_gpu=false, runs_inference=false, claims_model_has_run=false",
    ]


def test_review_packet_html_links_artifact_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "source_plan": str(run_root / "clip_001" / "frame_compute_plan.json"),
            "fps": 60.0,
            "scheduled_frames": [],
            "skipped_frames": [],
            "summary": {
                "scheduled_frame_count": 0,
                "scheduled_player_frame_count": 0,
                "skipped_frame_count": 0,
                "skipped_by_tier": {},
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")
    html_path = Path(write_review_packet(packet, out_dir=tmp_path / "runs" / "review_packets" / "prototype_gate_h100_v2")["html_path"])

    html = html_path.read_text(encoding="utf-8")
    assert '<a href="../../eval0/prototype_gate_h100_v2/clip_001/body_compute_execution.json">' in html


def test_review_packet_indexes_player_track_overlay(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    overlay = run_root / "clip_001" / "player_tracks" / "player_track_overlay.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"mp4")
    _write_json(
        overlay.parent / "player_track_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_player_track_overlay",
            "status": "rendered",
            "overlay_path": str(overlay),
            "frame_count": 120,
            "player_count": 4,
            "track_frame_count": 360,
            "qualitative_status": "prototype_not_gate_verified",
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_player_track_overlay"
    assert artifact["clip"] == "clip_001"
    assert artifact["watch_paths"] == [str(overlay)]


def test_review_packet_indexes_racket_candidate_overlay(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    overlay_dir = run_root / "clip_001" / "racket_candidates"
    overlay = overlay_dir / "racket_candidate_overlay_h264.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"mp4")
    _write_json(
        overlay_dir / "racket_candidate_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_candidate_overlay",
            "clip": "clip_001",
            "status": "rendered",
            "overlay_path": str(overlay),
            "frame_count": 20,
            "candidate_player_count": 1,
            "candidate_frame_count": 42,
            "candidate_coord_scale_x": 2.0,
            "candidate_coord_scale_y": 2.0,
            "qualitative_status": "candidate_review_not_gate_verified",
            "available_layers": ["paddle_candidates"],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_racket_candidate_overlay"
    assert artifact["clip"] == "clip_001"
    assert artifact["watch_paths"] == [str(overlay)]
    assert artifact["details"] == [
        "Frames rendered: 20",
        "Candidate players: 1",
        "Candidate frames: 42",
        "Coordinate scale: x=2.00 y=2.00",
    ]


def test_review_packet_includes_virtual_world_summaries(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "virtual_world.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world",
            "world_frame": "court_Z0",
            "fps": 60.0,
            "court": {
                "sport": "pickleball",
                "coordinate_frame": "origin_net_center_x_width_y_length_z_up_m",
                "length_m": 13.4112,
                "width_m": 6.096,
                "line_segments": {},
                "net": {"endpoints": [[-3.3528, 0.0, 0.0], [3.3528, 0.0, 0.0]], "center_height_m": 0.8636, "post_height_m": 0.9144},
            },
            "players": [],
            "ball": {"source": None, "frames": []},
            "paddles": [],
            "summary": {
                "player_count": 0,
                "mesh_player_count": 0,
                "mesh_player_frame_count": 0,
                "joint_player_frame_count": 0,
                "track_only_player_frame_count": 0,
                "ball_frame_count": 0,
                "approx_ball_frame_count": 0,
                "paddle_player_count": 0,
                "paddle_frame_count": 0,
                "ambiguous_paddle_frame_count": 0,
                "warnings": ["missing_players", "missing_paddle_pose"],
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_virtual_world"
    assert artifact["status"] == "assembled"
    assert artifact["details"] == [
        "Players: 0",
        "Mesh players: 0",
        "Ball frames: 0",
        "Paddle frames: 0",
        "Paddle status: no racket_pose.json frames; add racket_candidates.json or run the racket stage",
        "Warnings: missing_players, missing_paddle_pose",
    ]


def test_review_packet_indexes_replay_scene_glb_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    clip_dir = run_root / "clip_001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "court_review.glb").write_bytes(b"glTF")
    (clip_dir / "points").mkdir()
    (clip_dir / "points" / "point_001_review.glb").write_bytes(b"glTF")
    _write_json(
        clip_dir / "replay_scene.json",
        {
            "schema_version": 1,
            "world_frame": "court_Z0",
            "fps": 30.0,
            "court_glb": "court_review.glb",
            "players": [1, 2],
            "points": [{"id": 1, "t0": 0.0, "t1": 1.0, "glb_url": "points/point_001_review.glb", "size_mb": 0.000004}],
        },
    )

    packet = build_review_packet(run_root, packet_id="packet")

    artifacts = {artifact["artifact_type"]: artifact for artifact in packet["review_artifacts"]}
    artifact = artifacts["racketsport_replay_scene"]
    assert artifact["clip"] == "clip_001"
    assert artifact["status"] == "review_only"
    assert artifact["watch_paths"] == [
        str(clip_dir / "court_review.glb"),
        str(clip_dir / "points" / "point_001_review.glb"),
    ]
    assert artifact["details"] == [
        "FPS: 30.0",
        "Players: 2",
        "Replay points: 1",
        "Court GLB: court_review.glb",
        "Point GLB total MB: 4e-06",
    ]
    assert artifact["warnings"] == ["review_scene_not_accuracy_gate"]


def test_review_packet_indexes_virtual_world_paddle_preview(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "virtual_world_paddle_preview.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world",
            "world_frame": "court_Z0",
            "fps": 60.0,
            "court": {
                "sport": "pickleball",
                "coordinate_frame": "origin_net_center_x_width_y_length_z_up_m",
                "length_m": 13.4112,
                "width_m": 6.096,
                "line_segments": {},
                "net": {"endpoints": [[-3.3528, 0.0, 0.0], [3.3528, 0.0, 0.0]], "center_height_m": 0.8636, "post_height_m": 0.9144},
            },
            "players": [],
            "ball": {"source": None, "frames": []},
            "paddles": [
                {
                    "player_id": 7,
                    "paddle_dims_in": {"length": 16.0, "width": 8.0},
                    "frames": [
                        {
                            "t": 0.0,
                            "pose_se3": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.1, -0.2, 0.8]},
                            "mesh_vertices_world": [],
                            "mesh_faces": [],
                            "conf": 0.3,
                            "world_frame": "court_Z0",
                            "translation_unit": "m",
                            "source": "draft_box:pnp_ippe_preview:court_Z0",
                            "reprojection_error_px": 0.2,
                            "ambiguous": True,
                        }
                    ],
                }
            ],
            "summary": {
                "player_count": 0,
                "mesh_player_count": 0,
                "mesh_player_frame_count": 0,
                "joint_player_frame_count": 0,
                "track_only_player_frame_count": 0,
                "ball_frame_count": 0,
                "approx_ball_frame_count": 0,
                "paddle_player_count": 1,
                "paddle_frame_count": 1,
                "ambiguous_paddle_frame_count": 1,
                "warnings": ["missing_players", "missing_ball_track", "ambiguous_paddle_pose"],
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["path"].endswith("virtual_world_paddle_preview.json")
    assert artifact["artifact_type"] == "racketsport_virtual_world"
    assert artifact["details"] == [
        "Players: 0",
        "Mesh players: 0",
        "Ball frames: 0",
        "Paddle players: 1",
        "Paddle frames: 1",
        "Ambiguous paddle frames: 1",
        "Warnings: missing_players, missing_ball_track, ambiguous_paddle_pose",
    ]


def test_review_packet_indexes_virtual_world_review_html(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    html = run_root / "clip_001" / "virtual_world_paddle_preview.html"
    html.parent.mkdir(parents=True)
    html.write_text("<!doctype html><title>World review</title>", encoding="utf-8")
    _write_json(
        run_root / "clip_001" / "virtual_world_review_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world_review",
            "clip": "clip_001",
            "status": "rendered",
            "source_world_path": str(run_root / "clip_001" / "virtual_world_paddle_preview.json"),
            "review_html": str(html),
            "details": ["Players: 4", "Paddle frames: 42"],
            "warnings": ["ambiguous_paddle_pose"],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_virtual_world_review"
    assert artifact["status"] == "rendered"
    assert artifact["watch_paths"] == [str(html)]
    assert artifact["details"] == ["Players: 4", "Paddle frames: 42"]
    assert artifact["warnings"] == ["ambiguous_paddle_pose"]


def test_review_packet_surfaces_racket_candidates_without_pose(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    candidates = run_root / "clip_001" / "racket_candidates.json"
    _write_json(
        candidates,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_candidates",
            "fps": 60.0,
            "players": [
                {
                    "id": 7,
                    "paddle_dims_in": {"length": 16.0, "width": 8.0},
                    "frames": [
                        {
                            "t": 0.0,
                            "corners_px": [[100.0, 100.0], [140.0, 102.0], [136.0, 180.0], [96.0, 178.0]],
                            "conf": 0.9,
                            "source": "manual_or_detector_candidate",
                        },
                        {
                            "t": 0.1,
                            "corners_px": [[101.0, 101.0], [141.0, 103.0], [137.0, 181.0], [97.0, 179.0]],
                            "conf": 0.8,
                            "source": "manual_or_detector_candidate",
                        },
                    ],
                }
            ],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_racket_candidates"
    assert artifact["clip"] == "clip_001"
    assert artifact["status"] == "candidate_only"
    assert artifact["details"] == [
        "Candidate players: 1",
        "Candidate frames: 2",
        "Candidate sources: manual_or_detector_candidate",
        f"Pose artifact: missing ({candidates.parent / 'racket_pose.json'})",
    ]
    assert artifact["warnings"] == ["missing_racket_pose"]


def test_review_packet_indexes_paddle_true_corner_review(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    clip_root = run_root / "clip_001"
    crop_sheet = clip_root / "racket_candidates" / "paddle_true_corner_crop_sheet.png"
    crop_sheet.parent.mkdir(parents=True)
    crop_sheet.write_bytes(b"png")
    review = clip_root / "paddle_true_corner_review.json"
    _write_json(
        review,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_paddle_true_corner_review",
            "clip": "clip_001",
            "status": "blocked_missing_true_corner_labels",
            "trusted_for_rkt_promotion": False,
            "candidate_frame_count": 2,
            "box_derived_candidate_count": 2,
            "true_corner_label_count": 0,
            "reference_gt_count": 0,
            "required_label_count": 2,
            "listed_required_label_count": 2,
            "source_counts": {"label_bbox:yolo26m_teacher": 2},
            "visuals": [{"type": "true_corner_label_crop_sheet", "path": str(crop_sheet)}],
            "promotion_blockers": [
                "box_candidates_are_not_true_paddle_corners",
                "missing_reviewed_true_corner_labels",
            ],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_paddle_true_corner_review"
    assert artifact["clip"] == "clip_001"
    assert artifact["status"] == "blocked_missing_true_corner_labels"
    assert artifact["watch_paths"] == [str(crop_sheet)]
    assert artifact["details"] == [
        "Trusted for RKT promotion: false",
        "Candidate frames: 2",
        "Box-derived candidate frames: 2",
        "True corner labels: 0",
        "Reference/GT labels: 0",
        "Required labels: 2",
        "Listed required labels: 2",
        "Candidate sources: label_bbox:yolo26m_teacher=2",
    ]
    assert artifact["warnings"] == [
        "box_candidates_are_not_true_paddle_corners",
        "missing_reviewed_true_corner_labels",
    ]


def test_review_packet_includes_racket_stage_diagnostics(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    _write_json(
        run_root / "clip_001" / "racket_stage_diagnostics.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_stage_diagnostics",
            "stage": "racket",
            "status": "failed",
            "source_mode": "explicit_four_corner_candidates_pnp_ippe",
            "produced_artifacts": [],
            "metrics": {
                "candidate_frame_count": 42,
                "accepted_frame_count": 0,
                "rejected_high_reprojection_count": 0,
                "rejected_ambiguous_count": 42,
                "invalid_candidate_count": 0,
            },
            "notes": ["no racket_pose.json written because all candidates failed fail-closed checks"],
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")

    artifact = packet["review_artifacts"][0]
    assert artifact["artifact_type"] == "racketsport_racket_stage_diagnostics"
    assert artifact["status"] == "failed"
    assert artifact["details"] == [
        "Candidate frames: 42",
        "Accepted pose frames: 0",
        "Rejected ambiguous: 42",
        "Rejected high reprojection: 0",
        "Invalid candidates: 0",
    ]


def test_review_packet_filters_clips_and_discovers_ball_review_surfaces(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    accepted = "clip_accepted"
    rejected = "clip_rejected_side"
    for clip in (accepted, rejected):
        _write_json(
            run_root / clip / "compare" / "label_overlay_index.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_label_overlay",
                "clip": clip,
                "status": "rendered",
                "rendered_videos": [],
            },
        )
    ball_base = run_root / accepted / "tracknet_smoke_0000_0010"
    ball_overlay = ball_base / "ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"
    ball_track = ball_base / "ball_track_fusion_temporal_vball100_localtraj.json"
    ball_overlay.parent.mkdir(parents=True)
    ball_overlay.write_bytes(b"mp4")
    _write_json(ball_track, {"schema_version": 1, "artifact_type": "racketsport_ball_track", "frames": []})
    ball_review = run_root / "ball_click_review_30" / accepted / "review.html"
    ball_review.parent.mkdir(parents=True)
    ball_review.write_text("<html></html>", encoding="utf-8")

    packet = build_review_packet(run_root, packet_id="review", include_clips=[accepted])

    clips = {artifact["clip"] for artifact in packet["review_artifacts"]}
    artifact_types = {artifact["artifact_type"] for artifact in packet["review_artifacts"]}
    assert clips == {accepted}
    assert "racketsport_ball_track_overlay" in artifact_types
    assert "racketsport_ball_click_review_html" in artifact_types
    overlay_artifact = next(
        artifact for artifact in packet["review_artifacts"] if artifact["artifact_type"] == "racketsport_ball_track_overlay"
    )
    assert overlay_artifact["watch_paths"] == [str(ball_overlay)]
    assert overlay_artifact["details"] == [f"Track JSON: {ball_track}"]
    review_artifact = next(
        artifact
        for artifact in packet["review_artifacts"]
        if artifact["artifact_type"] == "racketsport_ball_click_review_html"
    )
    assert review_artifact["watch_paths"] == [str(ball_review)]


def test_write_review_packet_writes_json_markdown_and_correction_template(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "phase11"
    _write_json(
        run_root / "clip_001" / "pipeline_run.json",
        {
            "schema_version": 1,
            "clip": "clip_001",
            "requested_stage": "tracking",
            "status": "pass",
            "stages": [{"stage": "tracking", "status": "ran", "notes": []}],
        },
    )

    packet = build_review_packet(run_root, packet_id="phase11_review", corrections_root=tmp_path / "corrections")
    summary = write_review_packet(packet, out_dir=tmp_path / "packet", write_corrections_template=True)

    json_path = Path(summary["json_path"])
    markdown_path = Path(summary["markdown_path"])
    html_path = Path(summary["html_path"])
    corrections_template = Path(summary["corrections_template_path"])
    assert json.loads(json_path.read_text(encoding="utf-8"))["packet_id"] == "phase11_review"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Pickleball Human Review Packet" in markdown
    assert "python scripts/racketsport/validate_corrections.py" in markdown
    html = html_path.read_text(encoding="utf-8")
    assert "<title>Pickleball Human Review Packet - phase11_review</title>" in html
    assert "Open overlays first, then edit corrections" in html
    assert corrections_template.is_file()
    template = json.loads(corrections_template.read_text(encoding="utf-8"))
    assert template["manifest_id"] == "phase11_review"
    assert template["corrections"] == []


def test_review_packet_html_embeds_watch_paths_and_virtual_world_warnings(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    compare_dir = run_root / "clip_001" / "compare"
    overlay = compare_dir / "all_labels_overlay.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"mp4")
    _write_json(
        compare_dir / "label_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_label_overlay",
            "clip": "clip_001",
            "status": "rendered",
            "rendered_videos": [str(overlay)],
            "warnings": ["rendered from label frame pack"],
        },
    )
    _write_json(
        run_root / "clip_001" / "virtual_world.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world",
            "world_frame": "court_Z0",
            "fps": 60.0,
            "court": {
                "sport": "pickleball",
                "coordinate_frame": "origin_net_center_x_width_y_length_z_up_m",
                "length_m": 13.4112,
                "width_m": 6.096,
                "line_segments": {},
                "net": {
                    "endpoints": [[-3.3528, 0.0, 0.0], [3.3528, 0.0, 0.0]],
                    "center_height_m": 0.8636,
                    "post_height_m": 0.9144,
                },
            },
            "players": [],
            "ball": {"source": None, "frames": []},
            "paddles": [],
            "summary": {
                "player_count": 0,
                "mesh_player_count": 0,
                "mesh_player_frame_count": 0,
                "joint_player_frame_count": 0,
                "track_only_player_frame_count": 0,
                "ball_frame_count": 0,
                "approx_ball_frame_count": 0,
                "paddle_player_count": 0,
                "paddle_frame_count": 0,
                "ambiguous_paddle_frame_count": 0,
                "warnings": ["missing_mesh_vertices", "missing_paddle_pose"],
            },
        },
    )

    packet = build_review_packet(run_root, packet_id="prototype_gate_h100_v2_review")
    summary = write_review_packet(packet, out_dir=tmp_path / "packet")
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert '<video controls preload="metadata" src="' in html
    assert "all_labels_overlay.mp4" in html
    assert "racketsport_virtual_world" in html
    assert "missing_mesh_vertices" in html
    assert "missing_paddle_pose" in html
    assert "Players: 0" in html


def test_review_packet_cli_writes_packet(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "phase11"
    _write_json(
        run_root / "clip_001" / "pipeline_run.json",
        {
            "schema_version": 1,
            "clip": "clip_001",
            "requested_stage": "tracking",
            "status": "pass",
            "stages": [{"stage": "tracking", "status": "ran", "notes": []}],
        },
    )
    out_dir = tmp_path / "packet"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_review_packet.py",
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--packet-id",
            "phase11_review",
            "--write-corrections-template",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert Path(summary["json_path"]).is_file()
    assert Path(summary["markdown_path"]).is_file()
    assert Path(summary["html_path"]).is_file()
    assert Path(summary["corrections_template_path"]).is_file()


def test_review_packet_cli_accepts_clip_filters(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "phase11"
    for clip in ("clip_001", "clip_002"):
        _write_json(
            run_root / clip / "pipeline_run.json",
            {
                "schema_version": 1,
                "clip": clip,
                "requested_stage": "tracking",
                "status": "pass",
                "stages": [{"stage": "tracking", "status": "ran", "notes": []}],
            },
        )
    out_dir = tmp_path / "packet"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_review_packet.py",
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--packet-id",
            "phase11_review",
            "--clip",
            "clip_001",
            "--exclude-clip",
            "clip_002",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    packet = json.loads((out_dir / "phase11_review.json").read_text(encoding="utf-8"))
    assert [run["clip"] for run in packet["pipeline_runs"]] == ["clip_001"]
