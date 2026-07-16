from __future__ import annotations

import base64
import json
from pathlib import Path
import time
import zlib

import pytest

from threed.racketsport import ball_arc_solver
from threed.racketsport.ball_arc_solver import (
    BallArcSolverConfig,
    PhysicsParameters,
    build_bounce_anchor,
    solve_ball_arc_track,
)


FIXTURE = Path(__file__).parent / "fixtures/ball_arc_segment7_real_slice.json.zlib.b85"


def _real_segment7_slice() -> dict:
    """Load the checked-in, losslessly compressed R&D-reference slice.

    Trim rule: find the maximum-density 156-frame (5.166666 s) candidate
    window in the salvaged 20,922-frame input, retain every fourth real frame
    plus the final endpoint, rebase only frame indices/timestamps, and preserve
    every candidate on retained frames.  This keeps 121 real candidates from
    the original 484-candidate window while making the regression CI-sized.
    The fixture is explicitly not ground truth and is never a training input.
    """

    encoded = FIXTURE.read_text(encoding="ascii").strip().encode("ascii")
    return json.loads(zlib.decompress(base64.b85decode(encoded)))


def test_salvaged_real_segment7_slice_times_out_loudly_without_silent_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Warm process-level numeric dependencies outside the segment budget.
    from scipy.optimize import least_squares as _least_squares

    assert _least_squares is not None
    fixture = _real_segment7_slice()
    source = fixture["source"]
    assert source["not_ground_truth"] is True
    assert source["original_frame_start"] == 7940
    assert source["original_frame_end"] == 8095
    assert source["original_duration_s"] == pytest.approx(5.166666)
    assert source["original_candidate_count"] == 484
    assert source["retained_candidate_count"] == 121

    frames = fixture["frames"]
    calibration = fixture["calibration"]
    physics = PhysicsParameters()
    start = build_bounce_anchor(
        {"frame": 0, "t": frames[0]["t"]},
        calibration,
        ball_radius_m=physics.radius_m,
        ball_xy=frames[0]["xy"],
        status="r_and_d_reference",
        source="salvaged_pbvision_comparison_input_not_ground_truth",
        details={"not_ground_truth": True, "training_input": False},
    )
    end = build_bounce_anchor(
        {"frame": len(frames) - 1, "t": frames[-1]["t"]},
        calibration,
        ball_radius_m=physics.radius_m,
        ball_xy=frames[-1]["xy"],
        status="r_and_d_reference",
        source="salvaged_pbvision_comparison_input_not_ground_truth",
        details={"not_ground_truth": True, "training_input": False},
    )
    candidate_sidecar = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_candidates",
        "fps": fixture["fps"],
        "source": "salvaged_real_segment7_slice",
        "source_mode": "r_and_d_reference",
        "primary_output": "trimmed_ball_track.json",
        "max_candidates_per_frame": 12,
        "nms_radius_px": 0.0,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "provenance": source,
        "frames": fixture["candidate_frames"],
    }
    monkeypatch.setattr(ball_arc_solver, "SEGMENT_WALL_CLOCK_BUDGET_S", 0.02)

    started = time.monotonic()
    artifact = solve_ball_arc_track(
        ball_track={
            "schema_version": 1,
            "fps": fixture["fps"],
            "source": "salvaged_pbvision_comparison_input_not_ground_truth",
            "frames": frames,
            "bounces": [],
        },
        calibration=calibration,
        ball_candidate_sidecars=[candidate_sidecar],
        extra_anchors=[start, end],
        physics=physics,
        config=BallArcSolverConfig(
            enable_event_subset_selection=False,
            enable_event_discovery=False,
            enable_weak_segments=False,
            candidate_selection_max_iterations=5,
        ),
        clip_id="salvaged_segment7_r_and_d_reference",
    )
    elapsed_s = time.monotonic() - started

    # Whole-artifact validation continues after the segment abstains; the
    # bounded quantity is the segment's recorded numerical wall time.
    assert elapsed_s < 0.50
    assert artifact["status"] == "degraded"
    assert artifact["summary"]["segment_count"] == 1
    assert artifact["summary"]["fit_segment_count"] == 0
    assert artifact["summary"]["degraded_segment_count"] == 1
    assert artifact["summary"]["missing_segment_count"] == 1
    assert artifact["summary"]["segment_budget_exceeded_count"] == 1
    assert artifact["summary"]["segment_budget_exceeded_ids"] == [0]
    assert artifact["summary"]["missing_segment_reasons"] == {"segment_budget_exceeded": 1}
    assert artifact["degraded_reasons"] == [
        {
            "reason": "segment_budget_exceeded",
            "evidence_provenance": "missing",
            "segment_ids": [0],
        }
    ]
    assert len(artifact["segments"]) == 1
    segment = artifact["segments"][0]
    assert segment["status"] == "blocked:segment_budget_exceeded"
    assert segment["degradation"]["outcome_type"] == "segment_budget_exceeded"
    assert segment["degradation"]["reason"] == "segment_budget_exceeded"
    assert segment["degradation"]["evidence_provenance"] == "missing"
    assert segment["degradation"]["authority"] == "degraded"
    assert segment["degradation"]["elapsed_s"] < 0.05
    assert segment["degradation"]["candidate_count"] >= 121
    assert segment["degradation"]["duration_s"] == pytest.approx(5.166666)
