from __future__ import annotations

import pytest

from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.drift_guard import build_drift_log_artifact, evaluate_drift_sequence, should_check_frame, verify, verify_drift


def _square_world_points() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]]


def _square_image_points() -> list[list[float]]:
    return [[10.0, 20.0], [30.0, 20.0], [30.0, 25.0], [10.0, 25.0]]


def test_verify_drift_has_no_false_trigger_on_static_points():
    world_pts = _square_world_points()
    homography = homography_from_planar_points(world_pts, _square_image_points())

    result = verify_drift(
        homography=homography,
        world_pts=world_pts,
        observed_image_pts=project_planar_points(homography, world_pts),
        frame_index=120,
    )

    assert result.recalibration_required is False
    assert result.reprojection_error_px.p95 == pytest.approx(0.0)
    assert result.reasons == []


def test_verify_drift_flags_injected_twenty_pixel_bump():
    world_pts = _square_world_points()
    homography = homography_from_planar_points(world_pts, _square_image_points())
    bumped = [[point[0] + 20.0, point[1]] for point in project_planar_points(homography, world_pts)]

    result = verify_drift(
        homography=homography,
        world_pts=world_pts,
        observed_image_pts=bumped,
        frame_index=121,
    )

    assert result.recalibration_required is True
    assert result.reprojection_error_px.p95 == pytest.approx(20.0)
    assert result.reasons == ["reprojection_drift"]


def test_verify_is_doc_compatible_alias_for_verify_drift():
    world_pts = _square_world_points()
    homography = homography_from_planar_points(world_pts, _square_image_points())
    observed = project_planar_points(homography, world_pts)

    result = verify(
        homography=homography,
        world_pts=world_pts,
        observed_image_pts=observed,
        frame_index=150,
    )

    assert result == verify_drift(
        homography=homography,
        world_pts=world_pts,
        observed_image_pts=observed,
        frame_index=150,
    )


def test_should_check_frame_runs_first_frame_and_periodic_frames():
    assert should_check_frame(0, every_n_frames=30)
    assert should_check_frame(30, every_n_frames=30)
    assert not should_check_frame(31, every_n_frames=30)

    with pytest.raises(ValueError, match="positive"):
        should_check_frame(0, every_n_frames=0)


def test_evaluate_drift_sequence_requires_three_consecutive_failed_checks():
    interrupted = evaluate_drift_sequence(
        [(15, 8.5), (30, 8.4), (45, 7.9), (60, 8.6), (75, 8.7)],
        p95_gate_px=8.0,
        consecutive_failures=3,
    )
    tripped = evaluate_drift_sequence(
        [(15, 8.5), (30, 8.4), (45, 8.6)],
        p95_gate_px=8.0,
        consecutive_failures=3,
    )

    assert interrupted.recalibration_required is False
    assert interrupted.recalibration_from_frame is None
    assert tripped.recalibration_required is True
    assert tripped.recalibration_from_frame == 15
    assert tripped.reasons == ["reprojection_drift_3_consecutive"]
    assert tripped.checks[-1]["tripped"] is True


def test_build_drift_log_artifact_records_recalibration_span_from_sequence_status():
    status = evaluate_drift_sequence(
        [(15, 8.5), (30, 8.4), (45, 8.6)],
        p95_gate_px=8.0,
        consecutive_failures=3,
    )

    artifact = build_drift_log_artifact(status)

    assert artifact["artifact_type"] == "racketsport_drift_log"
    assert artifact["checks"][-1] == {"frame": 45, "p95_px": 8.6, "tripped": True}
    assert artifact["recalibrations"] == [
        {
            "from_frame": 15,
            "to_frame": 45,
            "reason": "reprojection_drift_3_consecutive",
        }
    ]
