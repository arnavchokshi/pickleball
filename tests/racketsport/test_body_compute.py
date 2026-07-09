from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_compute import build_body_compute_execution
from threed.racketsport.schemas import Tracks


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [-1.0, -2.9], "conf": 0.91},
                ],
            }
        ],
        "rally_spans": [],
    }


def _teleporting_tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [-0.95, -3.0], "conf": 0.91},
                    {"t": 2.0 / 30.0, "bbox": [420.0, 100.0, 520.0, 300.0], "world_xy": [3.0, -3.0], "conf": 0.89},
                    {"t": 3.0 / 30.0, "bbox": [422.0, 100.0, 522.0, 300.0], "world_xy": [3.05, -3.0], "conf": 0.88},
                ],
            }
        ],
        "rally_spans": [],
    }


def _world_jitter_bbox_continuous_tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [0.0, 0.0], "conf": 0.92},
                    {
                        "t": 1.0 / 30.0,
                        "bbox": [104.0, 100.0, 204.0, 300.0],
                        "world_xy": [2.0, 0.0],
                        "conf": 0.91,
                    },
                    {
                        "t": 2.0 / 30.0,
                        "bbox": [108.0, 100.0, 208.0, 300.0],
                        "world_xy": [2.04, 0.0],
                        "conf": 0.90,
                    },
                ],
            }
        ],
        "rally_spans": [],
    }


def _large_world_step_bbox_continuous_tracks_payload() -> dict:
    payload = _world_jitter_bbox_continuous_tracks_payload()
    payload["players"][0]["frames"][1]["world_xy"] = [4.0, 0.0]
    payload["players"][0]["frames"][2]["world_xy"] = [4.04, 0.0]
    return payload


def _frame_plan_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 2,
        "frames": [
            {
                "frame_idx": 0,
                "t": 0.0,
                "score": 0.8,
                "recommended_tier": "human_review",
                "target_representation": "manual_review_required",
                "reasons": ["missing_expected_players"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.92,
                "ball_conf": None,
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.92,
                        "score": 0.8,
                        "recommended_tier": "human_review",
                        "target_representation": "manual_review_required",
                        "reasons": ["missing_expected_players"],
                    }
                ],
            },
            {
                "frame_idx": 1,
                "t": 1.0 / 30.0,
                "score": 0.75,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["contact_window"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.91,
                "ball_conf": 0.6,
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.91,
                        "score": 0.75,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["contact_window"],
                    }
                ],
            },
        ],
        "deep_mesh_windows": [
            {
                "frame_start": 1,
                "frame_end": 1,
                "t0": 1.0 / 30.0,
                "t1": 2.0 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "skeleton_preview",
                "target_player_ids": [7],
                "reason_counts": {"contact_window": 1},
                "max_score": 0.75,
            }
        ],
        "summary": {
            "by_tier": {"deep_mesh": 1, "human_review": 1},
            "by_reason": {"contact_window": 1, "missing_expected_players": 1},
            "max_score": 0.8,
            "deep_mesh_window_count": 1,
            "deep_mesh_frame_count": 1,
            "human_review_frame_count": 1,
        },
    }


def _targeted_reviewed_contact_frame_plan_payload() -> dict:
    payload = _frame_plan_payload()
    payload["expected_players"] = 4
    payload["frames"] = [
        {
            "frame_idx": 1,
            "t": 1.0 / 30.0,
            "score": 0.75,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "missing_expected_players", "reviewed_contact_targeted_body"],
            "active_players": 1,
            "active_player_ids": [7],
            "missing_players": 3,
            "min_track_conf": 0.91,
            "ball_conf": None,
            "player_targets": [
                {
                    "player_id": 7,
                    "track_conf": 0.91,
                    "score": 0.55,
                    "recommended_tier": "deep_mesh",
                    "target_representation": "world_mesh",
                    "reasons": ["contact_window", "reviewed_contact_targeted_body"],
                }
            ],
        }
    ]
    payload["deep_mesh_windows"] = [
        {
            "frame_start": 1,
            "frame_end": 1,
            "t0": 1.0 / 30.0,
            "t1": 2.0 / 30.0,
            "frame_count": 1,
            "target_representation": "world_mesh",
            "fallback_representation": "skeleton_preview",
            "target_player_ids": [7],
            "reason_counts": {
                "contact_window": 1,
                "missing_expected_players": 1,
                "reviewed_contact_targeted_body": 1,
            },
            "max_score": 0.75,
        }
    ]
    payload["summary"] = {
        "by_tier": {"deep_mesh": 1},
        "by_reason": {
            "contact_window": 1,
            "missing_expected_players": 1,
            "reviewed_contact_targeted_body": 1,
        },
        "by_player_target_representation": {"world_mesh": 1},
        "max_score": 0.75,
        "deep_mesh_window_count": 1,
        "deep_mesh_frame_count": 1,
        "human_review_frame_count": 0,
        "targeted_reviewed_contact_frame_count": 1,
        "coverage_incomplete_deep_mesh_frame_count": 1,
    }
    return payload


def _contact_dense_ball_aware_frame_plan_payload() -> dict:
    frames = []
    for frame_idx in range(41):
        t = frame_idx / 30.0
        player_targets = [
            {
                "player_id": 7,
                "track_conf": 0.9,
                "score": 0.0,
                "recommended_tier": "baseline",
                "target_representation": "track_only",
                "reasons": [],
            },
            {
                "player_id": 8,
                "track_conf": 0.9,
                "score": 0.0,
                "recommended_tier": "baseline",
                "target_representation": "track_only",
                "reasons": [],
            },
        ]
        reasons = []
        recommended_tier = "baseline"
        target_representation = "track_only"
        score = 0.0
        if frame_idx == 15:
            reasons = ["ball_aware_contact"]
            recommended_tier = "deep_mesh"
            target_representation = "world_mesh"
            score = 0.9
            player_targets[0] = {
                "player_id": 7,
                "track_conf": 0.9,
                "score": 0.9,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["ball_aware_contact"],
            }
        elif frame_idx in {0, 40}:
            reasons = ["uniform_mesh_coverage"]
            recommended_tier = "deep_mesh"
            target_representation = "world_mesh"
            score = 1.0
            player_targets = [
                {
                    "player_id": player_id,
                    "track_conf": 0.9,
                    "score": 1.0,
                    "recommended_tier": "deep_mesh",
                    "target_representation": "world_mesh",
                    "reasons": ["uniform_mesh_coverage"],
                }
                for player_id in (7, 8)
            ]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "score": score,
                "recommended_tier": recommended_tier,
                "target_representation": target_representation,
                "reasons": reasons,
                "active_players": 2,
                "active_player_ids": [7, 8],
                "missing_players": 0,
                "min_track_conf": 0.9,
                "ball_conf": None,
                "player_targets": player_targets,
                "tier_rationale": {
                    "base_recommended_tier": recommended_tier,
                    "base_target_representation": target_representation,
                    "coverage_policy_mode": "ball_aware",
                    "mesh_selected": recommended_tier == "deep_mesh",
                    "selection_reasons": ["uniform_mesh_coverage"] if frame_idx in {0, 40} else [],
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 2,
        "frame_count": len(frames),
        "mesh_coverage_policy": {
            "mode": "ball_aware",
            "target_mesh_frame_budget": 100,
            "selected_mesh_frame_count": 3,
            "ball_aware_trigger_source_counts": {"events": 1, "proximity": 0, "swing": 0, "uniform_fill": 2},
        },
        "frames": frames,
        "deep_mesh_windows": [
            {
                "frame_start": 0,
                "frame_end": 0,
                "t0": 0.0,
                "t1": 1 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "lane_a_skeleton",
                "target_player_ids": [7, 8],
                "reason_counts": {"uniform_mesh_coverage": 1},
                "max_score": 1.0,
            },
            {
                "frame_start": 15,
                "frame_end": 15,
                "t0": 15 / 30.0,
                "t1": 16 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "lane_a_skeleton",
                "target_player_ids": [7],
                "reason_counts": {"ball_aware_contact": 1},
                "max_score": 0.9,
            },
            {
                "frame_start": 40,
                "frame_end": 40,
                "t0": 40 / 30.0,
                "t1": 41 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "lane_a_skeleton",
                "target_player_ids": [7, 8],
                "reason_counts": {"uniform_mesh_coverage": 1},
                "max_score": 1.0,
            },
        ],
        "summary": {
            "by_tier": {"baseline": 38, "deep_mesh": 3},
            "by_reason": {"ball_aware_contact": 1, "uniform_mesh_coverage": 2},
            "by_player_target_representation": {"track_only": 76, "world_mesh": 5},
            "max_score": 1.0,
            "deep_mesh_window_count": 3,
            "deep_mesh_frame_count": 3,
            "world_mesh_frame_count": 3,
            "mesh_coverage_mode": "ball_aware",
        },
    }


def _two_player_dense_tracks_payload() -> dict:
    frames = [
        {"t": frame_idx / 30.0, "bbox": [100 + frame_idx, 100.0, 200 + frame_idx, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.9}
        for frame_idx in range(41)
    ]
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {"id": 7, "side": "near", "role": "left", "frames": frames},
            {
                "id": 8,
                "side": "far",
                "role": "right",
                "frames": [
                    {
                        "t": frame["t"],
                        "bbox": [value + 220 if isinstance(value, (int, float)) else value for value in frame["bbox"]],
                        "world_xy": [1.0, -3.0],
                        "conf": 0.9,
                    }
                    for frame in frames
                ],
            },
        ],
        "rally_spans": [],
    }


def _ball_aware_uniform_only_frame_plan_payload() -> dict:
    payload = _contact_dense_ball_aware_frame_plan_payload()
    payload["frames"] = [frame for frame in payload["frames"] if frame["frame_idx"] in {0, 1, 2}]
    for frame in payload["frames"]:
        frame["reasons"] = []
        frame["recommended_tier"] = "baseline"
        frame["target_representation"] = "track_only"
        for target in frame["player_targets"]:
            target["reasons"] = []
            target["recommended_tier"] = "baseline"
            target["target_representation"] = "track_only"
    payload["frames"][1]["reasons"] = ["uniform_mesh_coverage"]
    payload["frames"][1]["recommended_tier"] = "deep_mesh"
    payload["frames"][1]["target_representation"] = "world_mesh"
    payload["frames"][1]["player_targets"] = [
        {
            "player_id": player_id,
            "track_conf": 0.9,
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["uniform_mesh_coverage"],
        }
        for player_id in (7, 8)
    ]
    payload["frame_count"] = 3
    payload["deep_mesh_windows"] = [
        {
            "frame_start": 1,
            "frame_end": 1,
            "t0": 1 / 30.0,
            "t1": 2 / 30.0,
            "frame_count": 1,
            "target_representation": "world_mesh",
            "fallback_representation": "lane_a_skeleton",
            "target_player_ids": [7, 8],
            "reason_counts": {"uniform_mesh_coverage": 1},
            "max_score": 1.0,
        }
    ]
    payload["mesh_coverage_policy"]["ball_aware_trigger_source_counts"] = {"events": 0, "proximity": 0, "swing": 0, "uniform_fill": 1}
    payload["summary"] = {
        "by_tier": {"baseline": 2, "deep_mesh": 1},
        "by_reason": {"uniform_mesh_coverage": 1},
        "by_player_target_representation": {"track_only": 4, "world_mesh": 2},
        "max_score": 1.0,
        "deep_mesh_window_count": 1,
        "deep_mesh_frame_count": 1,
        "world_mesh_frame_count": 1,
        "mesh_coverage_mode": "ball_aware",
    }
    return payload


def _baseline_frame_plan_payload(frame_count: int) -> dict:
    frames = [
        {
            "frame_idx": frame_idx,
            "t": frame_idx / 30.0,
            "score": 0.0,
            "recommended_tier": "baseline",
            "target_representation": "track_only",
            "reasons": [],
            "active_players": 1,
            "active_player_ids": [7],
            "missing_players": 0,
            "min_track_conf": 0.9,
            "ball_conf": None,
            "player_targets": [
                {
                    "player_id": 7,
                    "track_conf": 0.9,
                    "score": 0.0,
                    "recommended_tier": "baseline",
                    "target_representation": "track_only",
                    "reasons": [],
                }
            ],
        }
        for frame_idx in range(frame_count)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": frame_count,
        "mesh_coverage_policy": {"mode": "uniform"},
        "frames": frames,
        "deep_mesh_windows": [],
        "summary": {"by_tier": {"baseline": frame_count}},
    }


def _single_player_tracks_payload(frame_count: int, *, fps: float = 30.0) -> dict:
    return {
        "schema_version": 1,
        "fps": fps,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "t": frame_idx / fps,
                        "bbox": [100.0 + frame_idx, 100.0, 200.0 + frame_idx, 300.0],
                        "world_xy": [-1.0, -3.0 + frame_idx * 0.01],
                        "conf": 0.92,
                    }
                    for frame_idx in range(frame_count)
                ],
            }
        ],
        "rally_spans": [],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_body_compute_without_frame_plan_does_not_schedule_all_tracks_as_mesh() -> None:
    execution = build_body_compute_execution(Tracks.model_validate(_tracks_payload()))

    assert execution["mode"] == "lane_b_requires_frame_compute_plan"
    assert execution["scheduled_frames"] == []
    assert execution["summary"]["scheduled_frame_count"] == 0
    assert execution["summary"]["scheduled_player_frame_count"] == 0
    assert execution["summary"]["skipped_by_reason"] == {"missing_frame_compute_plan": 2}


def test_body_compute_default_skeleton_stride_schedules_every_other_base_frame(tmp_path: Path) -> None:
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _baseline_frame_plan_payload(5))

    execution = build_body_compute_execution(
        Tracks.model_validate(_single_player_tracks_payload(5)),
        frame_plan_path=frame_plan,
        include_tier2_body_joints=True,
    )

    assert [frame["frame_idx"] for frame in execution["scheduled_frames"]] == [0, 2, 4]
    assert execution["summary"]["base_skeleton_stride"] == 2
    assert execution["summary"]["effective_stride"] == 1.667
    assert execution["summary"]["total_track_frame_count"] == 5
    assert execution["summary"]["base_skeleton_frame_count"] == 3
    assert execution["summary"]["scheduled_vs_total_frame_count"] == {"scheduled": 3, "total": 5}
    assert execution["summary"]["skipped_by_reason"]["body_skeleton_stride_skip"] == 2


def test_body_compute_can_schedule_sam3d_body_joints_for_all_safe_tracked_frames(tmp_path: Path) -> None:
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _frame_plan_payload())

    execution = build_body_compute_execution(
        Tracks.model_validate(_tracks_payload()),
        frame_plan_path=frame_plan,
        include_tier2_body_joints=True,
    )

    scheduled = {(frame["frame_idx"], frame["target_representation"]): frame for frame in execution["scheduled_frames"]}
    assert set(scheduled) == {(0, "body_joints"), (1, "world_mesh")}
    assert scheduled[(0, "body_joints")]["target_player_ids"] == [7]
    assert scheduled[(0, "body_joints")]["recommended_tier"] == "tier2_body_joints"
    assert scheduled[(0, "body_joints")]["source"] == "sam3d_body_joints"
    assert scheduled[(0, "body_joints")]["reasons"] == ["sam3d_body_joints_all_tracked"]
    assert scheduled[(1, "world_mesh")]["recommended_tier"] == "deep_mesh"
    assert execution["summary"]["scheduled_by_target_representation"] == {
        "body_joints": 1,
        "world_mesh": 1,
    }
    assert execution["summary"]["scheduled_player_frame_count"] == 2
    assert execution["summary"]["tier2_body_joint_player_frame_count"] == 1
    assert execution["summary"]["tier1_mesh_player_frame_count"] == 1


def test_body_compute_execution_cli_writes_adaptive_manifest(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _frame_plan_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--frame-compute-plan",
            str(frame_plan),
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
    assert summary["summary"]["scheduled_frame_count"] == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_body_compute_execution"
    assert payload["mode"] == "adaptive_frame_compute_plan"
    assert payload["scheduled_frames"][0]["frame_idx"] == 1
    assert payload["scheduled_frames"][0]["window_frame_start"] == 1
    assert payload["scheduled_frames"][0]["window_frame_end"] == 1
    assert payload["scheduled_frames"][0]["window_frame_count"] == 1
    assert payload["scheduled_frames"][0]["fallback_representation"] == "skeleton_preview"
    assert payload["scheduled_frames"][0]["player_targets"] == [
        {
            "player_id": 7,
            "track_conf": 0.91,
            "score": 0.75,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window"],
        }
    ]
    assert payload["skipped_frames"][0]["player_targets"] == [
        {
            "player_id": 7,
            "track_conf": 0.92,
            "score": 0.8,
            "recommended_tier": "human_review",
            "target_representation": "manual_review_required",
            "reasons": ["missing_expected_players"],
        }
    ]
    assert payload["summary"]["skipped_by_tier"] == {"human_review": 1}
    assert payload["summary"]["skipped_by_target_representation"] == {"manual_review_required": 1}
    assert payload["summary"]["skipped_by_reason"] == {"missing_expected_players": 1}


def test_body_compute_execution_reports_targeted_reviewed_contact_schedule(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _targeted_reviewed_contact_frame_plan_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--frame-compute-plan",
            str(frame_plan),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scheduled_frames"][0]["target_player_ids"] == [7]
    assert payload["scheduled_frames"][0]["active_player_ids"] == [7]
    assert payload["scheduled_frames"][0]["reasons"] == [
        "contact_window",
        "missing_expected_players",
        "reviewed_contact_targeted_body",
    ]
    assert payload["summary"]["scheduled_by_reason"] == {
        "contact_window": 1,
        "missing_expected_players": 1,
        "reviewed_contact_targeted_body": 1,
    }
    assert payload["summary"]["scheduled_by_target_representation"] == {"world_mesh": 1}
    assert payload["summary"]["scheduled_coverage_incomplete_frame_count"] == 1
    assert payload["summary"]["scheduled_targeted_reviewed_contact_frame_count"] == 1


def test_body_compute_contact_dense_ball_aware_schedules_hitter_every_frame_with_sparse_uniform_floor(tmp_path: Path) -> None:
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _contact_dense_ball_aware_frame_plan_payload())

    execution = build_body_compute_execution(
        Tracks.model_validate(_two_player_dense_tracks_payload()),
        frame_plan_path=frame_plan,
    )

    scheduled_by_frame = {int(frame["frame_idx"]): frame for frame in execution["scheduled_frames"]}
    assert sorted(scheduled_by_frame) == list(range(31)) + [40]
    assert all(7 in frame["target_player_ids"] for frame_idx, frame in scheduled_by_frame.items() if frame_idx <= 30)
    assert [frame_idx for frame_idx, frame in scheduled_by_frame.items() if 8 in frame["target_player_ids"]] == [0, 40]
    assert execution["mesh_density_profile"]["mode"] == "contact_dense"
    assert execution["mesh_density_profile"]["status"] == "applied"
    assert execution["mesh_density_profile"]["contact_dense_pad_s"] == 0.5
    assert execution["mesh_density_profile"]["contact_dense_player_frame_count"] == 31
    assert execution["mesh_density_profile"]["uniform_floor_player_frame_count"] == 4
    assert execution["summary"]["tier1_mesh_player_frame_count"] == 34
    assert execution["summary"]["scheduled_by_reason"]["contact_dense_hitter_window"] == 31
    assert execution["summary"]["scheduled_by_reason"]["uniform_mesh_coverage"] == 2


def test_body_compute_stride_keeps_contact_dense_extra_frames_when_tier2_enabled(tmp_path: Path) -> None:
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _contact_dense_ball_aware_frame_plan_payload())

    execution = build_body_compute_execution(
        Tracks.model_validate(_two_player_dense_tracks_payload()),
        frame_plan_path=frame_plan,
        include_tier2_body_joints=True,
    )

    mesh_frame_indexes = [
        int(frame["frame_idx"])
        for frame in execution["scheduled_frames"]
        if frame["target_representation"] == "world_mesh"
    ]
    tier2_frame_indexes = [
        int(frame["frame_idx"])
        for frame in execution["scheduled_frames"]
        if frame["target_representation"] == "body_joints"
    ]

    assert mesh_frame_indexes == list(range(31)) + [40]
    assert tier2_frame_indexes == list(range(2, 31, 2)) + [32, 34, 36, 38]
    assert [
        frame["target_player_ids"]
        for frame in execution["scheduled_frames"]
        if frame["target_representation"] == "body_joints" and int(frame["frame_idx"]) <= 30
    ] == [[8]] * 15
    assert execution["summary"]["base_skeleton_stride"] == 2
    assert execution["summary"]["total_track_frame_count"] == 41
    assert execution["summary"]["base_skeleton_frame_count"] == 21
    assert execution["summary"]["event_extra_frame_count"] == 15
    assert execution["summary"]["scheduled_vs_total_frame_count"] == {"scheduled": 36, "total": 41}
    assert execution["mesh_density_profile"]["status"] == "applied"


def test_body_compute_contact_dense_ball_aware_falls_back_to_existing_uniform_when_contacts_missing(tmp_path: Path) -> None:
    frame_plan = _write_json(tmp_path / "frame_compute_plan.json", _ball_aware_uniform_only_frame_plan_payload())

    execution = build_body_compute_execution(
        Tracks.model_validate(_two_player_dense_tracks_payload()),
        frame_plan_path=frame_plan,
    )

    assert [frame["frame_idx"] for frame in execution["scheduled_frames"]] == [1]
    assert execution["scheduled_frames"][0]["target_player_ids"] == [7, 8]
    assert execution["mesh_density_profile"]["mode"] == "contact_dense"
    assert execution["mesh_density_profile"]["status"] == "uniform_fallback_missing_contact_evidence"
    assert any("falling back to existing ball_aware/uniform mesh windows" in note for note in execution["mesh_density_profile"]["notes"])
    assert execution["summary"]["tier1_mesh_player_frame_count"] == 2


def test_body_compute_execution_skips_track_player_frames_after_impossible_body_motion(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _teleporting_tracks_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "lane_b_requires_frame_compute_plan"
    assert payload["scheduled_frames"] == []
    assert [frame["frame_idx"] for frame in payload["skipped_frames"]] == [0, 1, 2, 3]
    assert [frame["skip_reason"] for frame in payload["skipped_frames"]] == [
        "missing_frame_compute_plan",
        "missing_frame_compute_plan",
        "unsafe_track_continuity",
        "unsafe_track_continuity",
    ]
    assert payload["skipped_frames"][2]["target_player_ids"] == [7]
    assert payload["summary"]["scheduled_player_frame_count"] == 0
    assert payload["summary"]["skipped_by_reason"] == {
        "missing_frame_compute_plan": 2,
        "unsafe_track_continuity": 2,
    }
    assert payload["summary"]["track_continuity_skipped_player_frame_count"] == 2
    assert payload["summary"]["track_continuity_temporal_jump_count"] == 2
    assert payload["summary"]["max_track_speed_for_body_mps"] == 10.0
    assert payload["track_continuity"]["status"] == "blocked"
    assert payload["track_continuity"]["temporal_jumps"][0]["player_id"] == 7
    assert payload["track_continuity"]["temporal_jumps"][0]["prev_frame_idx"] == 1
    assert payload["track_continuity"]["temporal_jumps"][0]["frame_idx"] == 2


def test_body_compute_execution_tolerates_world_anchor_jitter_when_bbox_is_continuous(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _world_jitter_bbox_continuous_tracks_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "lane_b_requires_frame_compute_plan"
    assert payload["scheduled_frames"] == []
    assert [frame["frame_idx"] for frame in payload["skipped_frames"]] == [0, 1, 2]
    assert {frame["skip_reason"] for frame in payload["skipped_frames"]} == {"missing_frame_compute_plan"}
    assert payload["summary"]["scheduled_player_frame_count"] == 0
    assert payload["summary"]["skipped_by_reason"] == {"missing_frame_compute_plan": 3}
    assert payload["summary"]["track_continuity_skipped_player_frame_count"] == 0
    assert payload["summary"]["track_continuity_world_anchor_jitter_player_frame_count"] == 1
    assert payload["track_continuity"]["status"] == "warning"
    assert payload["track_continuity"]["world_anchor_jitter_count"] == 1
    tolerated = payload["track_continuity"]["world_anchor_jitter"][0]
    assert tolerated["player_id"] == 7
    assert tolerated["frame_idx"] == 1
    assert tolerated["world_speed_mps"] > payload["summary"]["max_track_speed_for_body_mps"]
    assert tolerated["bbox_center_speed_diag_s"] <= payload["summary"]["max_bbox_center_speed_for_body_diag_s"]


def test_body_compute_execution_skips_bbox_continuous_jitter_when_world_step_is_too_large(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _large_world_step_bbox_continuous_tracks_payload())
    out = tmp_path / "body_compute_execution.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_compute_execution.py",
            "--tracks",
            str(tracks),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "lane_b_requires_frame_compute_plan"
    assert payload["scheduled_frames"] == []
    assert [frame["frame_idx"] for frame in payload["skipped_frames"]] == [0, 1, 2]
    assert [frame["skip_reason"] for frame in payload["skipped_frames"]] == [
        "missing_frame_compute_plan",
        "unsafe_track_continuity",
        "unsafe_track_continuity",
    ]
    assert payload["summary"]["scheduled_player_frame_count"] == 0
    assert payload["summary"]["skipped_by_reason"] == {
        "missing_frame_compute_plan": 1,
        "unsafe_track_continuity": 2,
    }
    assert payload["summary"]["track_continuity_skipped_player_frame_count"] == 2
    assert payload["summary"]["track_continuity_world_anchor_jitter_player_frame_count"] == 0
    assert payload["summary"]["max_track_world_step_for_bbox_jitter_m"] == 3.5
    first_jump = payload["track_continuity"]["temporal_jumps"][0]
    assert first_jump["bbox_center_speed_diag_s"] <= payload["summary"]["max_bbox_center_speed_for_body_diag_s"]
    assert first_jump["step_m"] > payload["summary"]["max_track_world_step_for_bbox_jitter_m"]
