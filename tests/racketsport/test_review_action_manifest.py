from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.review_action_manifest import (
    build_review_action_manifest,
    review_action_manifest_html,
    write_review_action_manifest,
)


def _packet() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifact_count": 8,
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_ball_inflections",
                "status": "review_only",
                "path": "runs/eval0/clip_001/ball_inflections.json",
                "watch_paths": [],
                "details": [
                    "Candidates: 13",
                    "Requires additional cues: audio_onsets, wrist_velocity_peaks",
                ],
                "warnings": [
                    "review_only_not_gate_verified",
                    "missing_cue:audio_onsets",
                    "missing_cue:wrist_velocity_peaks",
                ],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_court_line_evidence",
                "status": "blocked",
                "path": "runs/eval0/clip_001/court_line_evidence.json",
                "watch_paths": [],
                "details": [
                    "Source: auto_hough_template_video",
                    "Auto calibration ready: false",
                    "Missing required lines: near_nvz",
                    "Missing required net cues: top_net",
                ],
                "warnings": ["court_evidence_not_ready", "missing_line:near_nvz", "missing_net:top_net"],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_contact_window_review",
                "status": "pending_review",
                "path": "runs/eval0/clip_001/contact_window_review.json",
                "watch_paths": ["runs/eval0/clip_001/contact_window_review.html"],
                "details": ["Candidates: 1", "Pending: 1", "Accepted: 0", "Rejected: 0"],
                "warnings": ["pending_contact_review"],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_racket_pose_readiness",
                "status": "blocked_preview_only",
                "path": "runs/eval0/clip_001/racket_pose_readiness.json",
                "watch_paths": [],
                "details": [
                    "Candidate frames: 42",
                    "Box-derived frames: 42",
                    "Preview pose frames: 42",
                    "Promoted pose frames: 0",
                ],
                "warnings": [
                    "box_derived_candidate_corners",
                    "missing_true_paddle_keypoints_or_cad_pose",
                    "missing_promoted_racket_pose_json",
                    "missing_reference_pose_gt",
                    "missing_racket_pose_evaluation",
                ],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_body_compute_execution",
                "status": "scheduled",
                "path": "runs/eval0/clip_001/body_compute_execution.json",
                "watch_paths": [],
                "details": [
                    "Scheduled frames: 0",
                    "Scheduled player-frames: 0",
                    "Skipped frames: 600",
                    "Skipped tiers: human_review=589, skeleton_preview=9",
                ],
                "warnings": [],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_virtual_world",
                "status": "assembled",
                "path": "runs/eval0/clip_001/virtual_world_paddle_preview.json",
                "watch_paths": [],
                "details": ["Players: 4", "Mesh players: 0", "Warnings: missing_mesh_vertices, ambiguous_paddle_pose"],
                "warnings": [],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_virtual_world_review",
                "status": "ready",
                "path": "runs/eval0/clip_001/virtual_world_review_index.json",
                "watch_paths": ["runs/eval0/clip_001/virtual_world_paddle_preview.html"],
                "details": ["Three.js review page"],
                "warnings": [],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_pipeline_artifact_readiness",
                "status": "not_ready",
                "path": "runs/eval0/clip_001/pipeline_readiness_e2e.json",
                "watch_paths": [],
                "details": [
                    "Requested stage: e2e",
                    "Missing artifacts: racket_pose.json, replay_scene.json",
                    "Stages: ready=5, not_ready=6",
                ],
                "warnings": ["pipeline_not_ready", "missing:racket_pose.json", "missing:replay_scene.json"],
            },
        ],
    }


def test_build_review_action_manifest_groups_human_review_work() -> None:
    manifest = build_review_action_manifest(_packet(), packet_path="packet.json")

    assert manifest["artifact_type"] == "racketsport_review_action_manifest"
    assert manifest["packet_id"] == "prototype_gate_h100_v2_review"
    assert manifest["summary"] == {
        "action_count": 7,
        "by_category": {
            "body_schedule": 1,
            "contact_cues": 1,
            "contact_review": 1,
            "court_evidence": 1,
            "paddle_pose": 1,
            "pipeline_readiness": 1,
            "world_review": 1,
        },
        "by_priority": {"high": 2, "medium": 5},
        "clips": ["clip_001"],
    }
    actions = {action["category"]: action for action in manifest["actions"]}
    assert actions["contact_cues"]["title"] == "Ball inflections need cue pairing"
    assert actions["contact_cues"]["blockers"] == [
        "review_only_not_gate_verified",
        "missing_cue:audio_onsets",
        "missing_cue:wrist_velocity_peaks",
    ]
    assert actions["contact_cues"]["next_commands"] == [
        "python scripts/racketsport/build_ball_inflections.py --virtual-world runs/eval0/clip_001/virtual_world.json --out runs/eval0/clip_001/ball_inflections.json",
        "python scripts/racketsport/build_contact_windows_from_cues.py --audio-onsets runs/eval0/clip_001/audio_onsets.json --wrist-velocity-peaks runs/eval0/clip_001/wrist_velocity_peaks.json --ball-inflections runs/eval0/clip_001/ball_inflections.json --tracks runs/eval0/clip_001/tracks.json --out runs/eval0/clip_001/contact_windows.json",
    ]
    assert actions["court_evidence"]["title"] == "Court-line evidence is not auto-calibration-ready"
    assert actions["court_evidence"]["blockers"] == [
        "court_evidence_not_ready",
        "missing_line:near_nvz",
        "missing_net:top_net",
    ]
    assert actions["court_evidence"]["next_commands"] == [
        "python scripts/racketsport/build_court_line_evidence.py --calibration runs/eval0/clip_001/court_calibration.json --net-plane runs/eval0/clip_001/net_plane.json --video runs/eval0/clip_001/tracknet_smoke_0000_0010/input_0000_0010.mp4 --out runs/eval0/clip_001/court_line_evidence.json"
    ]
    assert actions["contact_review"]["priority"] == "high"
    assert actions["contact_review"]["watch_paths"] == ["runs/eval0/clip_001/contact_window_review.html"]
    assert actions["contact_review"]["editable_paths"] == ["runs/eval0/clip_001/contact_window_review.json"]
    assert actions["contact_review"]["next_commands"] == [
        "python scripts/racketsport/apply_review_inputs_to_contact_review.py --candidates runs/eval0/clip_001/contact_window_candidates.json --review runs/eval0/clip_001/contact_window_review.json --review-input runs/review_inputs/pickleball_cv_review_latest.json --clip clip_001 --out-review runs/eval0/clip_001/contact_window_review.json",
        "python scripts/racketsport/promote_contact_windows.py --candidates runs/eval0/clip_001/contact_window_candidates.json --review runs/eval0/clip_001/contact_window_review.json --out-contact-windows runs/eval0/clip_001/contact_windows.json",
        "python scripts/racketsport/render_contact_window_review.py --candidates runs/eval0/clip_001/contact_window_candidates.json --review runs/eval0/clip_001/contact_window_review.json --out-html runs/eval0/clip_001/contact_window_review.html",
    ]
    assert actions["contact_review"]["suggested_action"] == "Open the contact review page, edit pending decisions, then promote accepted contacts."
    assert actions["paddle_pose"]["priority"] == "high"
    assert actions["paddle_pose"]["editable_paths"] == ["runs/eval0/clip_001/racket_candidates.json"]
    assert actions["paddle_pose"]["next_commands"] == [
        "python scripts/racketsport/build_racket_pose_readiness.py --clip clip_001 --racket-candidates runs/eval0/clip_001/racket_candidates.json --racket-pose-preview runs/eval0/clip_001/racket_pose_preview.json --out runs/eval0/clip_001/racket_pose_readiness.json"
    ]
    assert actions["paddle_pose"]["blockers"] == [
        "box_derived_candidate_corners",
        "missing_true_paddle_keypoints_or_cad_pose",
        "missing_promoted_racket_pose_json",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]
    assert actions["body_schedule"]["title"] == "BODY has zero scheduled deep-mesh frames"
    assert actions["body_schedule"]["blockers"] == [
        "missing_promoted_contact_windows",
        "no_world_mesh_player_targets",
        "player_coverage_human_review",
    ]
    assert actions["body_schedule"]["next_commands"] == [
        "python scripts/racketsport/build_frame_compute_plan.py --tracks runs/eval0/clip_001/tracks.json --ball-track runs/eval0/clip_001/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json --contact-windows runs/eval0/clip_001/contact_windows.json --expected-players 4 --out runs/eval0/clip_001/frame_compute_plan.json",
        "python scripts/racketsport/build_body_compute_execution.py --tracks runs/eval0/clip_001/tracks.json --frame-compute-plan runs/eval0/clip_001/frame_compute_plan.json --out runs/eval0/clip_001/body_compute_execution.json",
    ]
    assert actions["pipeline_readiness"]["blockers"] == [
        "pipeline_not_ready",
        "missing:racket_pose.json",
        "missing:replay_scene.json",
    ]
    assert actions["pipeline_readiness"]["next_commands"] == [
        "python scripts/racketsport/validate_pipeline_artifacts.py --run-dir runs/eval0/clip_001 --stage e2e --out runs/eval0/clip_001/pipeline_readiness_e2e.json || true"
    ]
    assert actions["world_review"]["watch_paths"] == ["runs/eval0/clip_001/virtual_world_paddle_preview.html"]
    assert actions["world_review"]["next_commands"] == [
        "python scripts/racketsport/build_virtual_world_review.py --virtual-world runs/eval0/clip_001/virtual_world_paddle_preview.json --out-html runs/eval0/clip_001/virtual_world_paddle_preview.html --index-out runs/eval0/clip_001/virtual_world_review_index.json --title 'clip_001 Paddle Preview World'"
    ]


def test_build_review_action_manifest_skips_retired_burlington_court_evidence() -> None:
    packet = _packet()
    packet["review_artifacts"] = [
        {
            "clip": "burlington_gold_0300_low_steep_corner",
            "artifact_type": "racketsport_court_line_evidence",
            "status": "blocked",
            "path": (
                "runs/eval0/prototype_gate_h100_v2/"
                "burlington_gold_0300_low_steep_corner/court_line_evidence.json"
            ),
            "watch_paths": [],
            "details": [
                "Auto calibration ready: false",
                "Missing required lines: near_nvz",
            ],
            "warnings": ["court_evidence_not_ready", "missing_line:near_nvz"],
        }
    ]

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"]["action_count"] == 0
    assert manifest["actions"] == []


def test_build_review_action_manifest_skips_pending_contact_review_when_promoted_windows_exist() -> None:
    packet = _packet()
    packet["review_artifacts"] = [
        {
            "clip": "clip_001",
            "artifact_type": "racketsport_contact_window_review",
            "status": "pending_review",
            "path": "runs/eval0/clip_001/contact_window_review.json",
            "watch_paths": ["runs/eval0/clip_001/contact_window_review.html"],
            "details": ["Candidates: 1", "Pending: 1", "Accepted: 0", "Rejected: 0"],
            "warnings": ["pending_contact_review"],
        },
        {
            "clip": "clip_001",
            "artifact_type": "racketsport_contact_windows",
            "status": "promoted",
            "path": "runs/eval0/clip_001/contact_windows.json",
            "watch_paths": [],
            "details": ["Events: 3", "Human-reviewed events: 3"],
            "warnings": [],
        },
    ]

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"]["action_count"] == 0
    assert manifest["actions"] == []


def test_empty_contact_windows_remain_body_and_ball_blockers(tmp_path: Path) -> None:
    clip_dir = tmp_path / "runs" / "clip_001"
    clip_dir.mkdir(parents=True)
    contact_windows = clip_dir / "contact_windows.json"
    contact_windows.write_text(json.dumps({"schema_version": 1, "events": []}), encoding="utf-8")
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "packet",
        "run_root": str(tmp_path / "runs"),
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_contact_windows",
                "status": "empty",
                "path": str(contact_windows),
                "watch_paths": [],
                "details": ["Events: 0"],
                "warnings": ["empty_contact_windows_no_deep_mesh"],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_body_compute_execution",
                "status": "scheduled",
                "path": str(clip_dir / "body_compute_execution.json"),
                "watch_paths": [],
                "details": [
                    "Scheduled frames: 0",
                    "Scheduled player-frames: 0",
                    "Skipped frames: 60",
                    "Skipped tiers: skeleton_preview=60",
                ],
                "warnings": [],
            },
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path=tmp_path / "packet.json")

    actions = {action["category"]: action for action in manifest["actions"]}
    assert actions["contact_cues"]["blockers"] == ["empty_contact_windows_no_deep_mesh"]
    assert actions["contact_cues"]["suggested_action"] == (
        "Provide machine cue artifacts or promote reviewed contact decisions before BODY scheduling."
    )
    assert actions["contact_cues"]["next_commands"] == [
        f"python scripts/racketsport/build_contact_windows_from_cues.py --audio-onsets {clip_dir / 'audio_onsets.json'} --wrist-velocity-peaks {clip_dir / 'wrist_velocity_peaks.json'} --ball-inflections {clip_dir / 'ball_inflections.json'} --tracks {clip_dir / 'tracks.json'} --out {clip_dir / 'contact_windows.json'}",
        f"python scripts/racketsport/apply_review_inputs_to_contact_review.py --candidates {clip_dir / 'contact_window_candidates.json'} --review {clip_dir / 'contact_window_review.json'} --review-input runs/review_inputs/pickleball_cv_review_latest.json --clip clip_001 --out-review {clip_dir / 'contact_window_review.json'}",
        f"python scripts/racketsport/promote_contact_windows.py --candidates {clip_dir / 'contact_window_candidates.json'} --review {clip_dir / 'contact_window_review.json'} --out-contact-windows {clip_dir / 'contact_windows.json'}",
        f"python scripts/racketsport/render_contact_window_review.py --candidates {clip_dir / 'contact_window_candidates.json'} --review {clip_dir / 'contact_window_review.json'} --out-html {clip_dir / 'contact_window_review.html'}",
    ]
    assert actions["body_schedule"]["blockers"] == [
        "empty_contact_windows_no_deep_mesh",
        "no_world_mesh_player_targets",
    ]


def test_build_review_action_manifest_surfaces_targeted_body_schedule_warning() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_body_compute_execution",
                "status": "scheduled",
                "path": "runs/eval0/clip_001/body_compute_execution.json",
                "watch_paths": [],
                "details": [
                    "Scheduled frames: 1",
                    "Scheduled targeted reviewed-contact frames: 1",
                    "Scheduled incomplete-coverage frames: 1",
                ],
                "warnings": [
                    "targeted_reviewed_contact_body_schedule",
                    "scheduled_with_incomplete_player_coverage",
                ],
            }
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"] == {
        "action_count": 1,
        "by_category": {"body_schedule": 1},
        "by_priority": {"medium": 1},
        "clips": ["clip_001"],
    }
    action = manifest["actions"][0]
    assert action["title"] == "BODY schedule uses targeted reviewed contacts"
    assert action["blockers"] == [
        "targeted_reviewed_contact_body_schedule",
        "scheduled_with_incomplete_player_coverage",
    ]
    assert action["next_commands"] == [
        "python scripts/racketsport/build_frame_compute_plan.py --tracks runs/eval0/clip_001/tracks.json --ball-track runs/eval0/clip_001/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json --contact-windows runs/eval0/clip_001/contact_windows.json --expected-players 4 --out runs/eval0/clip_001/frame_compute_plan.json",
        "python scripts/racketsport/build_body_compute_execution.py --tracks runs/eval0/clip_001/tracks.json --frame-compute-plan runs/eval0/clip_001/frame_compute_plan.json --out runs/eval0/clip_001/body_compute_execution.json",
    ]


def test_build_review_action_manifest_surfaces_body_mesh_readiness_warning() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_body_mesh_readiness",
                "status": "mesh_available_needs_accuracy_gate",
                "path": "runs/eval0/clip_001/body_mesh_readiness.json",
                "watch_paths": [],
                "details": [
                    "World mesh available: true",
                    "Trusted for BODY promotion: false",
                    "Mesh frames: 4",
                ],
                "warnings": [
                    "mesh_not_accuracy_verified",
                    "missing_world_mpjpe_gate",
                    "missing_full_clip_body_gate",
                ],
            }
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"] == {
        "action_count": 1,
        "by_category": {"body_mesh": 1},
        "by_priority": {"medium": 1},
        "clips": ["clip_001"],
    }
    action = manifest["actions"][0]
    assert action["title"] == "BODY mesh is not accuracy-verified"
    assert action["blockers"] == [
        "mesh_not_accuracy_verified",
        "missing_world_mpjpe_gate",
        "missing_full_clip_body_gate",
    ]
    assert action["next_commands"] == [
        "python scripts/racketsport/build_body_mesh_readiness.py --clip clip_001 --smpl-motion runs/eval0/clip_001/smpl_motion.json --skeleton3d runs/eval0/clip_001/skeleton3d.json --frame-compute-plan runs/eval0/clip_001/frame_compute_plan.json --body-compute-execution runs/eval0/clip_001/body_compute_execution.json --out runs/eval0/clip_001/body_mesh_readiness.json"
    ]


def test_build_review_action_manifest_surfaces_no_world_mesh_requested_decision() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_body_mesh_readiness",
                "status": "missing_body_output",
                "path": "runs/eval0/clip_001/body_mesh_readiness.json",
                "watch_paths": [],
                "details": [
                    "World mesh available: false",
                    "Representation decision: no_world_mesh_requested",
                    "World mesh demand: requested=0, scheduled=0, available=0",
                    "Representation targets: joints_or_preview_mesh=4, manual_review_required=9, world_mesh=0",
                ],
                "warnings": [
                    "missing_mesh_vertices",
                    "world_mesh_not_requested_by_current_frame_plan",
                    "no_trusted_world_mesh_triggers",
                ],
            }
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    action = manifest["actions"][0]
    assert action["title"] == "BODY frame plan has no world-mesh requests"
    assert action["suggested_action"] == (
        "Resolve contact-window and player-coverage blockers before running more BODY mesh."
    )


def test_build_review_action_manifest_surfaces_audio_and_wrist_cue_blockers() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_audio_onsets",
                "status": "blocked",
                "path": "runs/eval0/clip_001/audio_onsets.json",
                "watch_paths": [],
                "details": ["Onsets: 0", "Blockers: no_audio_stream"],
                "warnings": ["audio_stream_missing", "cue_not_gate_verified", "not_trusted_for_contact", "no_audio_stream"],
            },
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_wrist_velocity_peaks",
                "status": "blocked",
                "path": "runs/eval0/clip_001/wrist_velocity_peaks.json",
                "watch_paths": [],
                "details": ["Peaks: 0", "Blockers: missing_wrist_joint_mapping"],
                "warnings": ["missing_wrist_joint_mapping", "cue_not_gate_verified", "not_trusted_for_contact"],
            },
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"] == {
        "action_count": 2,
        "by_category": {"contact_cues": 2},
        "by_priority": {"medium": 2},
        "clips": ["clip_001"],
    }
    actions = {action["artifact_path"]: action for action in manifest["actions"]}
    assert actions["runs/eval0/clip_001/audio_onsets.json"]["title"] == "Audio onset cues are unavailable"
    assert actions["runs/eval0/clip_001/audio_onsets.json"]["next_commands"] == [
        "python scripts/racketsport/build_audio_onsets.py --input runs/eval0/clip_001/tracknet_smoke_0000_0010/input_0000_0010.mp4 --out runs/eval0/clip_001/audio_onsets.json --clip clip_001 --start-s 0 --duration-s 10 --analysis-sample-rate-hz 16000",
        "python scripts/racketsport/build_contact_windows_from_cues.py --audio-onsets runs/eval0/clip_001/audio_onsets.json --wrist-velocity-peaks runs/eval0/clip_001/wrist_velocity_peaks.json --ball-inflections runs/eval0/clip_001/ball_inflections.json --tracks runs/eval0/clip_001/tracks.json --out runs/eval0/clip_001/contact_windows.json",
    ]
    assert actions["runs/eval0/clip_001/wrist_velocity_peaks.json"]["title"] == "Wrist-velocity cues are unavailable"
    assert actions["runs/eval0/clip_001/wrist_velocity_peaks.json"]["next_commands"] == [
        "python scripts/racketsport/build_wrist_velocity_peaks.py --skeleton3d runs/eval0/clip_001/skeleton3d.json --out runs/eval0/clip_001/wrist_velocity_peaks.json --allow-missing",
        "python scripts/racketsport/build_contact_windows_from_cues.py --audio-onsets runs/eval0/clip_001/audio_onsets.json --wrist-velocity-peaks runs/eval0/clip_001/wrist_velocity_peaks.json --ball-inflections runs/eval0/clip_001/ball_inflections.json --tracks runs/eval0/clip_001/tracks.json --out runs/eval0/clip_001/contact_windows.json",
    ]


def test_build_review_action_manifest_adds_audio_frame_rate_when_tracks_exist(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    clip_dir = run_root / "clip_001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "tracks.json").write_text(json.dumps({"schema_version": 1, "fps": 60.0, "players": []}), encoding="utf-8")
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": str(run_root),
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_audio_onsets",
                "status": "blocked",
                "path": str(clip_dir / "audio_onsets.json"),
                "watch_paths": [],
                "details": ["Onsets: 0", "Blockers: no_audio_stream"],
                "warnings": ["audio_stream_missing"],
            },
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    command = manifest["actions"][0]["next_commands"][0]
    assert "--clip clip_001" in command
    assert "--analysis-sample-rate-hz 16000" in command
    assert "--frame-rate 60" in command


def test_build_review_action_manifest_surfaces_racket_promotion_audit() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "clip_001",
                "artifact_type": "racketsport_racket_promotion_audit",
                "status": "unsafe_box_derived_promoted",
                "path": "runs/eval0/clip_001/racket_promotion_audit.json",
                "watch_paths": [],
                "details": [
                    "Canonical racket_pose.json present: true",
                    "Unsafe promoted frames: 1",
                    "Unsafe promoted sources: label_bbox:manual:pnp_ippe=1",
                ],
                "warnings": [
                    "box_derived_racket_pose_promoted",
                    "missing_reference_pose_gt",
                    "missing_racket_pose_evaluation",
                ],
            }
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"] == {
        "action_count": 1,
        "by_category": {"paddle_pose": 1},
        "by_priority": {"high": 1},
        "clips": ["clip_001"],
    }
    action = manifest["actions"][0]
    assert action["title"] == "Racket pose promotion is unsafe"
    assert action["editable_paths"] == ["runs/eval0/clip_001/racket_candidates.json"]
    assert action["blockers"] == [
        "box_derived_racket_pose_promoted",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]
    assert action["next_commands"] == [
        "python scripts/racketsport/build_racket_promotion_audit.py --clip clip_001 --racket-candidates runs/eval0/clip_001/racket_candidates.json --racket-pose-preview runs/eval0/clip_001/racket_pose_preview.json --racket-pose runs/eval0/clip_001/racket_pose.json --out runs/eval0/clip_001/racket_promotion_audit.json"
    ]


def test_build_review_action_manifest_surfaces_global_runtime_readiness() -> None:
    packet = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_review_packet",
        "packet_id": "prototype_gate_h100_v2_review",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "review_artifacts": [
            {
                "clip": "__global__",
                "artifact_type": "racketsport_racket_model_runtime_readiness",
                "status": "blocked",
                "path": "runs/eval0/prototype_gate_h100_v2/racket_model_runtime_readiness.json",
                "watch_paths": [],
                "details": [
                    "Components: 6",
                    "Runtime-ready components: 0",
                    "Asset ready: false",
                    "May run GPU smoke: false",
                    "May promote RKT: false",
                ],
                "warnings": ["sam3_concept_tracker:missing_manifest_entry"],
            }
        ],
    }

    manifest = build_review_action_manifest(packet, packet_path="packet.json")

    assert manifest["summary"] == {
        "action_count": 1,
        "by_category": {"paddle_runtime": 1},
        "by_priority": {"medium": 1},
        "clips": ["__global__"],
    }
    action = manifest["actions"][0]
    assert action["title"] == "Paddle model/runtime readiness is blocked"
    assert action["next_commands"] == [
        "python scripts/racketsport/build_racket_model_runtime_readiness.py --manifest models/MANIFEST.json --out runs/eval0/prototype_gate_h100_v2/racket_model_runtime_readiness.json"
    ]


def test_review_action_manifest_html_lists_actions_and_links() -> None:
    manifest = build_review_action_manifest(_packet(), packet_path="packet.json")

    html = review_action_manifest_html(manifest, base_dir=Path("runs/review_packets/prototype_gate_h100_v2"))

    assert "<title>Review Actions - prototype_gate_h100_v2_review</title>" in html
    assert "Contact windows need human decisions" in html
    assert "Ball inflections need cue pairing" in html
    assert "Court-line evidence is not auto-calibration-ready" in html
    assert "Paddle pose is preview-only" in html
    assert "E2E artifact readiness is incomplete" in html
    assert "contact_window_review.html" in html
    assert "build_court_line_evidence.py" in html
    assert "build_ball_inflections.py" in html
    assert "promote_contact_windows.py" in html
    assert "validate_pipeline_artifacts.py" in html
    assert "Editable files" in html
    assert "virtual_world_paddle_preview.html" in html


def test_review_action_manifest_cli_writes_json_and_html(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    out_json = tmp_path / "review_actions.json"
    out_html = tmp_path / "review_actions.html"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_review_action_manifest.py",
            "--packet",
            str(packet_path),
            "--out-json",
            str(out_json),
            "--out-html",
            str(out_html),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["action_count"] == 7
    assert "Paddle pose is preview-only" in out_html.read_text(encoding="utf-8")
    assert json.loads(completed.stdout)["action_count"] == 7
    write_review_action_manifest(out_json, payload)
