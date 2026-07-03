from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.frame_rating import build_frame_compute_plan


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [-1.0, -2.9], "conf": 0.41},
                ],
            },
            {
                "id": 2,
                "side": "near",
                "role": "right",
                "frames": [
                    {"t": 0.0, "bbox": [500.0, 100.0, 600.0, 300.0], "world_xy": [1.0, -3.0], "conf": 0.9},
                ],
            },
        ],
        "rally_spans": [],
    }


def _dense_tracks_payload(*, frame_count: int = 10, fps: float = 10.0) -> dict:
    return {
        "schema_version": 1,
        "fps": fps,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "t": frame_idx / fps,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [-1.0, -3.0 + 0.01 * frame_idx],
                        "conf": 0.92,
                    }
                    for frame_idx in range(frame_count)
                ],
            },
            {
                "id": 2,
                "side": "near",
                "role": "right",
                "frames": [
                    {
                        "t": frame_idx / fps,
                        "bbox": [500.0, 100.0, 600.0, 300.0],
                        "world_xy": [1.0, -3.0 + 0.01 * frame_idx],
                        "conf": 0.9,
                    }
                    for frame_idx in range(frame_count)
                ],
            },
        ],
        "rally_spans": [{"t0": 0.0, "t1": (frame_count - 1) / fps}],
    }


def _ball_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [300.0, 200.0], "conf": 0.94, "visible": True},
            {"t": 1.0 / 30.0, "xy": [310.0, 210.0], "conf": 0.22, "visible": False},
        ],
        "bounces": [],
    }


def _sparse_ball_payload() -> dict:
    payload = _ball_payload()
    payload["frames"] = payload["frames"][:1]
    return payload


def _contact_payload() -> dict:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 1.0 / 30.0,
                "frame": 1,
                "player_id": 1,
                "confidence": 0.88,
                "sources": {"audio": 0.9, "wrist_vel": 0.7, "ball_inflection": 0.65},
                "window": {"t0": 0.02, "t1": 0.08, "importance": 0.9},
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frame_compute_plan_prioritizes_contact_low_confidence_and_ball_uncertainty() -> None:
    plan = build_frame_compute_plan(
        _tracks_payload(),
        ball_track=_ball_payload(),
        contact_windows=_contact_payload(),
        expected_players=2,
        mesh_coverage_mode="contact_only",
    )

    assert plan["artifact_type"] == "racketsport_frame_compute_plan"
    assert plan["fps"] == 30.0
    assert [frame["frame_idx"] for frame in plan["frames"]] == [0, 1]
    assert plan["frames"][1]["score"] > plan["frames"][0]["score"]
    assert plan["frames"][1]["recommended_tier"] == "deep_mesh"
    assert plan["frames"][1]["reasons"] == [
        "contact_window",
        "low_track_confidence",
        "ball_uncertain",
    ]
    assert plan["frames"][1]["target_representation"] == "world_mesh"
    assert plan["frames"][0]["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.92,
            "score": 0.55,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window"],
        },
        {
            "player_id": 2,
            "track_conf": 0.9,
            "score": 0.0,
            "recommended_tier": "baseline",
            "target_representation": "track_only",
            "reasons": [],
        },
    ]
    assert plan["frames"][1]["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.41,
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "low_track_confidence", "ball_uncertain"],
        }
    ]
    assert plan["deep_mesh_windows"] == [
        {
            "frame_start": 0,
            "frame_end": 1,
            "t0": 0.0,
            "t1": 2.0 / 30.0,
            "frame_count": 2,
            "target_representation": "world_mesh",
            "fallback_representation": "lane_a_skeleton",
            "target_player_ids": [1],
            "reason_counts": {
                "ball_uncertain": 1,
                "contact_window": 2,
                "low_track_confidence": 1,
            },
            "max_score": 1.0,
        }
    ]
    assert plan["summary"]["deep_mesh_window_count"] == 1
    assert plan["summary"]["by_player_target_representation"] == {"track_only": 1, "world_mesh": 2}


def test_frame_compute_plan_routes_skeleton_preview_to_lane_a_skeleton() -> None:
    plan = build_frame_compute_plan(_tracks_payload(), expected_players=2, mesh_coverage_mode="contact_only")

    frame_one = next(frame for frame in plan["frames"] if frame["frame_idx"] == 1)
    assert frame_one["recommended_tier"] == "skeleton_preview"
    assert frame_one["target_representation"] == "lane_a_skeleton"
    assert frame_one["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.41,
            "score": 0.25,
            "recommended_tier": "skeleton_preview",
            "target_representation": "lane_a_skeleton",
            "reasons": ["low_track_confidence"],
        }
    ]
    assert plan["summary"]["by_player_target_representation"] == {"lane_a_skeleton": 1, "track_only": 2}


def test_frame_compute_plan_marks_missing_expected_players_for_human_review() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]

    plan = build_frame_compute_plan(tracks, expected_players=4, mesh_coverage_mode="contact_only")

    assert plan["frames"][0]["active_players"] == 1
    assert "missing_expected_players" in plan["frames"][0]["reasons"]
    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert plan["frames"][0]["player_targets"][0]["target_representation"] == "manual_review_required"
    assert plan["frames"][0]["player_targets"][0]["reasons"] == ["missing_expected_players"]
    assert plan["summary"]["by_player_target_representation"] == {"manual_review_required": 2}


def test_frame_compute_plan_schedules_reviewed_assigned_contact_despite_incomplete_coverage() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]
    contacts = _contact_payload()
    contacts["events"][0]["sources"] = {
        "audio": 0.0,
        "wrist_vel": 0.0,
        "ball_inflection": 0.0,
        "human_review": 1.0,
    }

    plan = build_frame_compute_plan(tracks, contact_windows=contacts, expected_players=4, mesh_coverage_mode="contact_only")

    frame_zero = plan["frames"][0]
    assert frame_zero["recommended_tier"] == "deep_mesh"
    assert frame_zero["target_representation"] == "world_mesh"
    assert frame_zero["reasons"] == [
        "contact_window",
        "missing_expected_players",
        "reviewed_contact_targeted_body",
    ]
    assert frame_zero["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.92,
            "score": 0.55,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window", "reviewed_contact_targeted_body"],
        }
    ]
    assert plan["deep_mesh_windows"] == [
        {
            "frame_start": 0,
            "frame_end": 1,
            "t0": 0.0,
            "t1": 2.0 / 30.0,
            "frame_count": 2,
            "target_representation": "world_mesh",
            "fallback_representation": "lane_a_skeleton",
            "target_player_ids": [1],
            "reason_counts": {
                "contact_window": 2,
                "low_track_confidence": 1,
                "missing_expected_players": 2,
                "reviewed_contact_targeted_body": 2,
            },
            "max_score": 0.8,
        }
    ]
    assert plan["summary"]["targeted_reviewed_contact_frame_count"] == 2
    assert plan["summary"]["coverage_incomplete_deep_mesh_frame_count"] == 2
    assert plan["summary"]["human_review_frame_count"] == 0


def test_frame_compute_plan_keeps_unassigned_reviewed_contact_fail_closed_when_coverage_incomplete() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]
    contacts = _contact_payload()
    contacts["events"][0]["player_id"] = None
    contacts["events"][0]["sources"] = {
        "audio": 0.0,
        "wrist_vel": 0.0,
        "ball_inflection": 0.0,
        "human_review": 1.0,
    }

    plan = build_frame_compute_plan(tracks, contact_windows=contacts, expected_players=4, mesh_coverage_mode="contact_only")

    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert "reviewed_contact_targeted_body" not in plan["frames"][0]["reasons"]
    assert plan["deep_mesh_windows"] == []
    assert plan["summary"]["targeted_reviewed_contact_frame_count"] == 0


def test_frame_compute_plan_keeps_machine_contact_fail_closed_when_coverage_incomplete() -> None:
    tracks = _tracks_payload()
    tracks["players"] = tracks["players"][:1]

    plan = build_frame_compute_plan(
        tracks,
        contact_windows=_contact_payload(),
        expected_players=4,
        mesh_coverage_mode="contact_only",
    )

    assert plan["frames"][0]["recommended_tier"] == "human_review"
    assert plan["frames"][0]["target_representation"] == "manual_review_required"
    assert "reviewed_contact_targeted_body" not in plan["frames"][0]["reasons"]
    assert plan["deep_mesh_windows"] == []


def test_frame_compute_plan_marks_omitted_ball_frames_as_uncertain() -> None:
    plan = build_frame_compute_plan(
        _tracks_payload(),
        ball_track=_sparse_ball_payload(),
        expected_players=2,
        mesh_coverage_mode="contact_only",
    )

    frame_one = next(frame for frame in plan["frames"] if frame["frame_idx"] == 1)
    assert frame_one["ball_conf"] is None
    assert "ball_missing" in frame_one["reasons"]
    assert frame_one["recommended_tier"] == "skeleton_preview"
    assert plan["summary"]["by_reason"]["ball_missing"] == 1


def test_frame_compute_plan_uniform_policy_distributes_budget_across_rally_span() -> None:
    plan = build_frame_compute_plan(
        _dense_tracks_payload(),
        expected_players=2,
        mesh_coverage_mode="uniform",
        target_mesh_frame_budget=4,
    )

    deep_indexes = [frame["frame_idx"] for frame in plan["frames"] if frame["recommended_tier"] == "deep_mesh"]
    assert deep_indexes == [0, 3, 6, 9]
    assert plan["mesh_coverage_policy"] == {
        "mode": "uniform",
        "target_mesh_frame_budget": 4,
        "eligible_mesh_frame_count": 10,
        "selected_mesh_frame_count": 4,
        "contact_candidate_frame_count": 0,
        "uniform_selected_frame_count": 4,
        "contact_selected_frame_count": 0,
        "rally_span_count": 1,
        "uniform_stride_frames": 3,
        "budget_limited": True,
        "ball_aware_candidate_frame_count": None,
        "ball_aware_trigger_source_counts": None,
    }
    assert plan["summary"]["deep_mesh_frame_count"] == 4
    assert plan["summary"]["mesh_coverage_mode"] == "uniform"
    assert plan["summary"]["mesh_coverage_fraction"] == 0.4
    assert plan["frames"][3]["tier_rationale"] == {
        "base_recommended_tier": "baseline",
        "base_target_representation": "track_only",
        "coverage_policy_mode": "uniform",
        "mesh_selected": True,
        "selection_reasons": ["uniform_mesh_coverage"],
    }
    assert plan["frames"][3]["player_targets"] == [
        {
            "player_id": 1,
            "track_conf": 0.92,
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["uniform_mesh_coverage"],
        },
        {
            "player_id": 2,
            "track_conf": 0.9,
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["uniform_mesh_coverage"],
        },
    ]


def test_frame_compute_plan_hybrid_policy_keeps_contacts_and_fills_remaining_budget() -> None:
    contacts = {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.5,
                "frame": 5,
                "player_id": 1,
                "confidence": 0.9,
                "sources": {"audio": 0.0, "wrist_vel": 0.8, "ball_inflection": 0.7},
                "window": {"t0": 0.5, "t1": 0.5, "importance": 0.9},
            }
        ],
    }

    plan = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=contacts,
        expected_players=2,
        mesh_coverage_mode="hybrid",
        target_mesh_frame_budget=4,
        contact_padding_s=0.0,
    )

    deep_indexes = [frame["frame_idx"] for frame in plan["frames"] if frame["recommended_tier"] == "deep_mesh"]
    assert len(deep_indexes) == 4
    assert 5 in deep_indexes
    assert plan["mesh_coverage_policy"]["mode"] == "hybrid"
    assert plan["mesh_coverage_policy"]["contact_selected_frame_count"] == 1
    assert plan["mesh_coverage_policy"]["uniform_selected_frame_count"] == 3
    assert plan["frames"][5]["tier_rationale"]["selection_reasons"] == ["contact_boost"]
    assert any(
        frame["tier_rationale"]["selection_reasons"] == ["uniform_mesh_coverage"]
        for frame in plan["frames"]
        if frame["recommended_tier"] == "deep_mesh" and frame["frame_idx"] != 5
    )


def test_uniform_policy_budget_demotes_unselected_contact_frames() -> None:
    contacts = {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.5,
                "frame": 5,
                "player_id": 1,
                "confidence": 0.9,
                "sources": {"audio": 0.0, "wrist_vel": 0.8, "ball_inflection": 0.7},
                "window": {"t0": 0.5, "t1": 0.5, "importance": 0.9},
            }
        ],
    }

    plan = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=contacts,
        expected_players=2,
        mesh_coverage_mode="uniform",
        target_mesh_frame_budget=4,
        contact_padding_s=0.0,
    )

    selected_indexes = [
        frame["frame_idx"]
        for frame in plan["frames"]
        if frame["tier_rationale"]["mesh_selected"]
    ]
    deep_indexes = [frame["frame_idx"] for frame in plan["frames"] if frame["recommended_tier"] == "deep_mesh"]
    assert selected_indexes == [0, 3, 6, 9]
    assert deep_indexes == selected_indexes
    assert plan["frames"][5]["tier_rationale"] == {
        "base_recommended_tier": "deep_mesh",
        "base_target_representation": "world_mesh",
        "coverage_policy_mode": "uniform",
        "mesh_selected": False,
        "selection_reasons": [],
    }
    assert plan["frames"][5]["recommended_tier"] == "skeleton_preview"
    assert plan["frames"][5]["target_representation"] == "lane_a_skeleton"


def test_frame_compute_plan_default_policy_is_hybrid() -> None:
    plan = build_frame_compute_plan(
        _dense_tracks_payload(frame_count=6),
        expected_players=2,
        target_mesh_frame_budget=3,
    )

    assert plan["mesh_coverage_policy"]["mode"] == "hybrid"
    assert plan["summary"]["mesh_coverage_mode"] == "hybrid"
    assert [frame["frame_idx"] for frame in plan["frames"] if frame["recommended_tier"] == "deep_mesh"] == [0, 2, 5]


def test_frame_compute_plan_cli_writes_json(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _tracks_payload())
    ball = _write_json(tmp_path / "ball_track.json", _ball_payload())
    contacts = _write_json(tmp_path / "contact_windows.json", _contact_payload())
    out = tmp_path / "frame_compute_plan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_frame_compute_plan.py",
            "--tracks",
            str(tracks),
            "--ball-track",
            str(ball),
            "--contact-windows",
            str(contacts),
            "--expected-players",
            "2",
            "--mesh-coverage-mode",
            "contact_only",
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
    assert summary["frame_count"] == 2
    assert summary["summary"]["deep_mesh_window_count"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["frames"][1]["recommended_tier"] == "deep_mesh"


def test_frame_compute_plan_cli_accepts_hybrid_budget_flags(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _dense_tracks_payload())
    out = tmp_path / "frame_compute_plan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_frame_compute_plan.py",
            "--tracks",
            str(tracks),
            "--expected-players",
            "2",
            "--mesh-coverage-mode",
            "hybrid",
            "--target-mesh-frame-budget",
            "4",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["mesh_coverage_policy"]["mode"] == "hybrid"
    assert summary["summary"]["deep_mesh_frame_count"] == 4
    assert json.loads(out.read_text(encoding="utf-8"))["mesh_coverage_policy"]["target_mesh_frame_budget"] == 4


# ---------------------------------------------------------------------
# ball_aware mesh coverage mode (2026-07-03 owner directive: tier-1 SAM-3D
# mesh moments must be ball-aware, not raw wrist-cue windows)
# ---------------------------------------------------------------------


def _events_selected_payload() -> dict:
    """Shaped like scripts/racketsport/solve_ball_arcs.py's events_selected.json.

    Only ``selected`` entries with ``kind == "contact"`` should ever drive
    ball_aware_contact; the bounce entry here is a negative control.
    """
    return {
        "artifact_type": "racketsport_ball_arc_events_selected",
        "selected": [
            {
                "anchor_id": "contact_000_p2_left",
                "kind": "contact",
                "frame": 2,
                "player_id": 2,
                "t": 0.2,
                "candidate_confidence": 0.7,
                "selected": True,
            },
            {
                "anchor_id": "bounce_0000",
                "kind": "bounce",
                "frame": 4,
                "t": 0.4,
                "candidate_confidence": 0.65,
                "selected": True,
            },
        ],
        "rejected": [],
        "selected_count": 2,
    }


def _ball_track_arc_solved_payload(*, fps: float = 10.0, frame_count: int = 10) -> dict:
    """Shaped like ball_track_arc_solved.json; only frame 5 has a trusted
    world_xyz (co-located with player 1's world_xy at that frame -- player 2
    is always 2m away in X, outside the 1.5m default proximity threshold)."""
    frames = []
    for frame_idx in range(frame_count):
        t = frame_idx / fps
        if frame_idx == 5:
            frames.append(
                {
                    "t": t,
                    "world_xyz": [-1.0, -3.0 + 0.01 * frame_idx, 0.3],
                    "visible": True,
                    "band": "anchored_measured",
                }
            )
        else:
            # "hidden" band: solver has no trusted position this frame.
            frames.append({"t": t, "world_xyz": None, "visible": True, "band": "hidden"})
    return {"artifact_type": "racketsport_ball_track_arc_solved", "frames": frames}


def _mixed_confidence_contact_payload() -> dict:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.7,
                "frame": 7,
                "player_id": None,
                "confidence": 0.75,
                "sources": {"audio": 0.0, "wrist_vel": 0.8, "ball_inflection": 0.7},
                "window": {"t0": 0.7, "t1": 0.7, "importance": 0.75},
            },
            {
                "type": "contact",
                "t": 0.8,
                "frame": 8,
                "player_id": 1,
                "confidence": 0.5,
                "sources": {"audio": 0.0, "wrist_vel": 0.5, "ball_inflection": 0.4},
                "window": {"t0": 0.8, "t1": 0.8, "importance": 0.5},
            },
        ],
    }


def test_frame_compute_plan_ball_aware_triggers_events_proximity_and_swing_not_raw_windows() -> None:
    plan = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=_mixed_confidence_contact_payload(),
        ball_aware_events=_events_selected_payload(),
        ball_track_arc_solved=_ball_track_arc_solved_payload(),
        expected_players=2,
        mesh_coverage_mode="ball_aware",
        ball_aware_padding_s=0.0,
        target_mesh_frame_budget=3,
    )

    assert plan["mesh_coverage_policy"]["mode"] == "ball_aware"
    by_idx = {frame["frame_idx"]: frame for frame in plan["frames"]}

    # (a) physically-validated contact event (events_selected.json, kind="contact")
    assert "ball_aware_contact" in by_idx[2]["reasons"]
    assert by_idx[2]["recommended_tier"] == "deep_mesh"
    ba_targets = {t["player_id"]: t for t in by_idx[2]["player_targets"]}
    assert "ball_aware_contact" in ba_targets[2]["reasons"]
    assert "ball_aware_contact" not in ba_targets[1]["reasons"]
    # a bounce-kind selected event must never trigger ball_aware_contact
    assert "ball_aware_contact" not in by_idx[4]["reasons"]

    # (b) ball-proximity: only the player within ball_proximity_m of the
    # arc-solved ball world position gets tagged, using horizontal (XY) distance.
    assert "ball_proximity" in by_idx[5]["reasons"]
    assert by_idx[5]["recommended_tier"] == "deep_mesh"
    prox_targets = {t["player_id"]: t for t in by_idx[5]["player_targets"]}
    assert "ball_proximity" in prox_targets[1]["reasons"]
    assert "ball_proximity" not in prox_targets[2]["reasons"]

    # (c) high-confidence swing cue (>= high_confidence_swing_floor) triggers;
    # the low-confidence event one frame later does not.
    assert "high_confidence_swing" in by_idx[7]["reasons"]
    assert by_idx[7]["recommended_tier"] == "deep_mesh"
    assert "high_confidence_swing" not in by_idx[8]["reasons"]

    # NOT raw wrist-cue windows: the legacy contact_window reason (all fused
    # wrist+ball cues regardless of confidence) never appears under
    # ball_aware mode, even though contact_windows was supplied.
    for frame in plan["frames"]:
        assert "contact_window" not in frame["reasons"]

    counts = plan["mesh_coverage_policy"]["ball_aware_trigger_source_counts"]
    assert counts["events"] == 1
    assert counts["proximity"] == 1
    assert counts["swing"] == 1
    assert counts["uniform_fill"] == 0
    assert plan["mesh_coverage_policy"]["ball_aware_candidate_frame_count"] == 3


def test_frame_compute_plan_ball_aware_mode_off_ignores_ball_aware_inputs() -> None:
    """Passing ball-aware artifacts without opting into mesh_coverage_mode='ball_aware'
    must not change scoring/selection at all (backward compatible default)."""
    plan_without = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=_mixed_confidence_contact_payload(),
        expected_players=2,
        mesh_coverage_mode="hybrid",
        target_mesh_frame_budget=3,
    )
    plan_with_unused_inputs = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=_mixed_confidence_contact_payload(),
        ball_aware_events=_events_selected_payload(),
        ball_track_arc_solved=_ball_track_arc_solved_payload(),
        expected_players=2,
        mesh_coverage_mode="hybrid",
        target_mesh_frame_budget=3,
    )
    assert plan_without["frames"] == plan_with_unused_inputs["frames"]
    assert plan_without["mesh_coverage_policy"] == plan_with_unused_inputs["mesh_coverage_policy"]


def test_frame_compute_plan_ball_aware_budget_prioritizes_events_over_proximity_over_swing() -> None:
    plan = build_frame_compute_plan(
        _dense_tracks_payload(),
        contact_windows=_mixed_confidence_contact_payload(),
        ball_aware_events=_events_selected_payload(),
        ball_track_arc_solved=_ball_track_arc_solved_payload(),
        expected_players=2,
        mesh_coverage_mode="ball_aware",
        ball_aware_padding_s=0.0,
        target_mesh_frame_budget=2,
    )

    counts = plan["mesh_coverage_policy"]["ball_aware_trigger_source_counts"]
    # budget=2 covers only the two highest-scoring ball-aware candidates
    # (events > proximity > swing); the swing-only frame loses out to
    # uniform fill instead of being kept.
    assert counts["events"] == 1
    assert counts["proximity"] == 1
    assert counts["swing"] == 0
    assert plan["mesh_coverage_policy"]["contact_selected_frame_count"] == 2

    by_idx = {frame["frame_idx"]: frame for frame in plan["frames"]}
    assert by_idx[2]["tier_rationale"]["selection_reasons"] == ["ball_aware_boost"]
    assert by_idx[5]["tier_rationale"]["selection_reasons"] == ["ball_aware_boost"]
    assert by_idx[7]["tier_rationale"]["mesh_selected"] is False


def test_frame_compute_plan_ball_aware_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="ball_proximity_m"):
        build_frame_compute_plan(_dense_tracks_payload(), expected_players=2, ball_proximity_m=0.0)
    with pytest.raises(ValueError, match="high_confidence_swing_floor"):
        build_frame_compute_plan(_dense_tracks_payload(), expected_players=2, high_confidence_swing_floor=1.5)
    with pytest.raises(ValueError, match="ball_aware_padding_s"):
        build_frame_compute_plan(_dense_tracks_payload(), expected_players=2, ball_aware_padding_s=-0.1)


def test_frame_compute_plan_cli_ball_aware_mode(tmp_path: Path) -> None:
    tracks = _write_json(tmp_path / "tracks.json", _dense_tracks_payload())
    contacts = _write_json(tmp_path / "contact_windows.json", _mixed_confidence_contact_payload())
    events_selected = _write_json(tmp_path / "events_selected.json", _events_selected_payload())
    ball_arc_solved = _write_json(tmp_path / "ball_track_arc_solved.json", _ball_track_arc_solved_payload())
    out = tmp_path / "frame_compute_plan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_frame_compute_plan.py",
            "--tracks",
            str(tracks),
            "--contact-windows",
            str(contacts),
            "--expected-players",
            "2",
            "--mesh-coverage-mode",
            "ball_aware",
            "--target-mesh-frame-budget",
            "3",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # The standalone CLI (scripts/racketsport/build_frame_compute_plan.py,
    # owned by a different lane) does not yet expose --events-selected /
    # --ball-track-arc-solved flags -- confirm the ball_aware mode is at
    # least reachable and gracefully falls back to zero ball-aware
    # candidates (no raw contact_window leakage) rather than erroring.
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["mesh_coverage_policy"]["mode"] == "ball_aware"
    written = json.loads(out.read_text(encoding="utf-8"))
    for frame in written["frames"]:
        assert "contact_window" not in frame["reasons"]
