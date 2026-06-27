from __future__ import annotations

import pytest

from threed.racketsport.physics_refine import (
    ContactWindow,
    FootContactSample,
    MotionRefinementRequest,
    PlayerRootSample,
    choose_execution_plan,
    package_refinement_artifact,
    select_refinement_windows,
    summarize_constraints,
)


def test_motion_refinement_request_validates_core_inputs() -> None:
    with pytest.raises(ValueError, match="total_frames must be positive"):
        MotionRefinementRequest(
            clip_id="clip-a",
            frame_rate_hz=60.0,
            total_frames=120.5,  # type: ignore[arg-type]
            player_ids=("p1",),
        )

    with pytest.raises(ValueError, match="frame_rate_hz must be positive"):
        MotionRefinementRequest(
            clip_id="clip-a",
            frame_rate_hz=0.0,
            total_frames=120,
            player_ids=("p1",),
        )

    with pytest.raises(ValueError, match="player_ids must be unique"):
        MotionRefinementRequest(
            clip_id="clip-a",
            frame_rate_hz=60.0,
            total_frames=120,
            player_ids=("p1", "p1"),
        )

    with pytest.raises(ValueError, match="unsupported requested_mode"):
        MotionRefinementRequest(
            clip_id="clip-a",
            frame_rate_hz=60.0,
            total_frames=120,
            player_ids=("p1",),
            requested_mode="phc",
        )


def test_physics_samples_reject_nonfinite_or_nonbool_values() -> None:
    with pytest.raises(ValueError, match="frame_index must be a non-negative integer"):
        FootContactSample(
            frame_index=True,  # type: ignore[arg-type]
            player_id="p1",
            foot="left",
            position_xyz=(0.0, 0.0, 0.0),
            contact=True,
        )

    with pytest.raises(ValueError, match="position_xyz/2 must be finite"):
        FootContactSample(
            frame_index=0,
            player_id="p1",
            foot="left",
            position_xyz=(0.0, 0.0, float("nan")),
            contact=True,
        )

    with pytest.raises(ValueError, match="contact must be a bool"):
        FootContactSample(
            frame_index=0,
            player_id="p1",
            foot="left",
            position_xyz=(0.0, 0.0, 0.0),
            contact=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="radius_m must be finite"):
        PlayerRootSample(
            frame_index=0,
            player_id="p1",
            center_xyz=(0.0, 0.0, 0.0),
            radius_m=float("nan"),
        )


def test_select_refinement_windows_pads_clamps_and_merges_contact_spans() -> None:
    windows = select_refinement_windows(
        (
            ContactWindow(start_frame=10, end_frame=15, player_id="p1", reason="left_foot_contact"),
            ContactWindow(start_frame=18, end_frame=22, player_id="p1", reason="right_foot_contact"),
            ContactWindow(start_frame=80, end_frame=90, player_id="p2", reason="doubles_overlap"),
        ),
        total_frames=100,
        pad_frames=5,
    )

    assert windows == (
        ContactWindow(start_frame=5, end_frame=27, player_id="p1", reason="left_foot_contact+right_foot_contact"),
        ContactWindow(start_frame=75, end_frame=95, player_id="p2", reason="doubles_overlap"),
    )


def test_summarize_constraints_reports_floor_slide_and_player_penetration() -> None:
    foot_samples = (
        FootContactSample(frame_index=0, player_id="p1", foot="left", position_xyz=(0.0, 0.0, 0.01), contact=True),
        FootContactSample(frame_index=1, player_id="p1", foot="left", position_xyz=(0.004, 0.0, -0.002), contact=True),
        FootContactSample(frame_index=2, player_id="p1", foot="left", position_xyz=(0.020, 0.0, -0.004), contact=False),
        FootContactSample(frame_index=0, player_id="p2", foot="right", position_xyz=(1.0, 0.0, 0.02), contact=True),
    )
    roots = (
        PlayerRootSample(frame_index=0, player_id="p1", center_xyz=(0.0, 0.0, 0.9), radius_m=0.35),
        PlayerRootSample(frame_index=0, player_id="p2", center_xyz=(0.60, 0.0, 0.9), radius_m=0.35),
        PlayerRootSample(frame_index=1, player_id="p1", center_xyz=(0.0, 0.0, 0.9), radius_m=0.35),
        PlayerRootSample(frame_index=1, player_id="p2", center_xyz=(1.20, 0.0, 0.9), radius_m=0.35),
    )

    summary = summarize_constraints(foot_samples, roots, floor_z_m=0.0)

    assert summary.contact_frames == 3
    assert summary.max_contact_slide_m == pytest.approx(0.004)
    assert summary.max_floor_penetration_m == pytest.approx(0.004)
    assert summary.inter_player_penetration_frames == 1
    assert summary.max_inter_player_penetration_m == pytest.approx(0.10)
    assert summary.scaffold == "cpu_physics_refinement_scaffold_no_sim"


def test_choose_execution_plan_never_runs_mjx_and_blocks_required_mode_without_runtime() -> None:
    request = MotionRefinementRequest(
        clip_id="clip-a",
        frame_rate_hz=60.0,
        total_frames=120,
        player_ids=("p1", "p2"),
        requested_mode="auto",
    )

    auto_plan = choose_execution_plan(request, mjx_available=False)

    assert auto_plan.mode == "cpu_fallback"
    assert auto_plan.requires_mjx is False
    assert auto_plan.will_run_mjx is False
    assert "scaffold" in auto_plan.reason

    required = MotionRefinementRequest(
        clip_id="clip-a",
        frame_rate_hz=60.0,
        total_frames=120,
        player_ids=("p1",),
        requested_mode="mujoco_mjx_required",
    )

    blocked_plan = choose_execution_plan(required, mjx_available=False)

    assert blocked_plan.mode == "blocked_mjx_required"
    assert blocked_plan.requires_mjx is True
    assert blocked_plan.will_run_mjx is False
    assert "MJX runtime is required" in blocked_plan.reason


def test_package_refinement_artifact_is_schema_friendly_and_scaffold_only() -> None:
    request = MotionRefinementRequest(
        clip_id="clip-a",
        frame_rate_hz=60.0,
        total_frames=120,
        player_ids=("p1",),
        requested_mode="cpu_fallback",
    )
    windows = (ContactWindow(start_frame=5, end_frame=15, player_id="p1", reason="foot_contact"),)
    summary = summarize_constraints(
        (
            FootContactSample(
                frame_index=5,
                player_id="p1",
                foot="left",
                position_xyz=(0.0, 0.0, 0.0),
                contact=True,
            ),
        )
    )
    plan = choose_execution_plan(request, mjx_available=False)

    artifact = package_refinement_artifact(request, windows, summary, plan)

    assert artifact["clip_id"] == "clip-a"
    assert artifact["skate_free"] is False
    assert artifact["physics"] == "cpu_fallback_scaffold"
    assert artifact["foot2_done"] is False
    assert artifact["must_not_mark_done_verified"] is True
    assert artifact["refinement_windows"] == [
        {"start_frame": 5, "end_frame": 15, "player_id": "p1", "reason": "foot_contact"}
    ]
    assert artifact["constraint_summary"]["contact_frames"] == 1
    assert artifact["execution_plan"]["will_run_mjx"] is False
