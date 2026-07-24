from __future__ import annotations

import math

import pytest

from threed.racketsport.ball_metric3d_contract import (
    GroundTruthObservation,
    GroundTruthObservationSet,
)
from threed.racketsport.eval.ball_metric3d_eval import (
    MVP_ACCEPTED_MEDIAN_3D_M,
    MVP_BOUNCE_MEDIAN_M,
    MVP_THRESHOLD_STATUS,
    VARIANT_ORACLE_ANCHORS,
    VARIANT_ORACLE_EVENTS,
    VARIANT_PREDICTED,
    CandidateRun,
    CandidateSample,
    EventEstimate,
    Metric3DEvalError,
    dumps_report_json,
    evaluate_candidate,
    evaluate_variants,
    write_report_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(positions: list[tuple[float, float, float]]) -> GroundTruthObservationSet:
    return GroundTruthObservationSet(
        clip="synthetic_eval_clip",
        observations=tuple(
            GroundTruthObservation(
                timestamp_s=index / 30.0,
                xyz_world_m=xyz,
                sigma_xyz_m=(0.01, 0.02, 0.01),
                cameras_used=("dev_side", "dev_corner"),
                triangulation_residual_px=0.4,
                quality_flags=("gold",),
            )
            for index, xyz in enumerate(positions)
        ),
    )


def _offset_run(
    ground_truth: GroundTruthObservationSet,
    offset: tuple[float, float, float],
    *,
    accepted: list[bool] | None = None,
) -> CandidateRun:
    samples = tuple(
        CandidateSample(
            timestamp_s=obs.timestamp_s,
            xyz_world_m=(
                obs.xyz_world_m[0] + offset[0],
                obs.xyz_world_m[1] + offset[1],
                obs.xyz_world_m[2] + offset[2],
            ),
        )
        for obs in ground_truth.observations
    )
    mask = tuple(accepted) if accepted is not None else tuple(True for _ in samples)
    return CandidateRun(samples=samples, accepted=mask)


FOUR_POSITIONS = [
    (0.0, -3.0, 0.5),
    (0.2, -1.0, 1.1),
    (0.4, 1.0, 1.4),
    (0.6, 3.0, 0.8),
]


# ---------------------------------------------------------------------------
# Known-answer accuracy metrics
# ---------------------------------------------------------------------------


def test_constant_offset_gives_exact_mae_and_rmse():
    ground_truth = _gt(FOUR_POSITIONS)
    report = evaluate_candidate(ground_truth, _offset_run(ground_truth, (0.1, -0.2, 0.3)))
    stats = report["overall"]["accepted_error"]
    assert stats["count"] == 4
    assert stats["mae_x_m"] == pytest.approx(0.1)
    assert stats["mae_y_m"] == pytest.approx(0.2)
    assert stats["mae_z_m"] == pytest.approx(0.3)
    expected_3d = math.sqrt(0.1**2 + 0.2**2 + 0.3**2)
    assert stats["rmse_3d_m"] == pytest.approx(expected_3d, abs=1e-6)
    # Constant error: every percentile equals the constant.
    assert stats["err_3d_median_m"] == pytest.approx(expected_3d, abs=1e-6)
    assert stats["err_3d_p90_m"] == pytest.approx(expected_3d, abs=1e-6)
    assert stats["err_3d_p95_m"] == pytest.approx(expected_3d, abs=1e-6)
    assert stats["err_3d_max_m"] == pytest.approx(expected_3d, abs=1e-6)
    assert report["overall"]["acceptance_rate"] == pytest.approx(1.0)


def test_per_axis_errors_are_independent():
    ground_truth = _gt(FOUR_POSITIONS)
    report = evaluate_candidate(ground_truth, _offset_run(ground_truth, (0.0, 0.5, 0.0)))
    stats = report["overall"]["accepted_error"]
    assert stats["mae_x_m"] == 0.0
    assert stats["mae_y_m"] == pytest.approx(0.5)
    assert stats["mae_z_m"] == 0.0
    assert stats["rmse_3d_m"] == pytest.approx(0.5)


def test_acceptance_rate_math_and_rejected_error_not_hidden():
    ground_truth = _gt(FOUR_POSITIONS)
    samples = (
        CandidateSample(0.0 / 30.0, (0.1, -3.0, 0.5)),  # accepted, err 0.1
        CandidateSample(1.0 / 30.0, (0.2, -1.0, 1.2)),  # accepted, err 0.1
        CandidateSample(2.0 / 30.0, (0.4, 2.0, 1.4)),  # rejected but solved, err 1.0
        CandidateSample(3.0 / 30.0, None),  # no output
    )
    run = CandidateRun(samples=samples, accepted=(True, True, False, False))
    report = evaluate_candidate(ground_truth, run)
    overall = report["overall"]
    assert overall["frame_count"] == 4
    assert overall["frames_with_candidate_xyz"] == 3
    assert overall["accepted_frame_count"] == 2
    # Denominator is ALL GT frames: rejection and no-output both count against.
    assert overall["acceptance_rate"] == pytest.approx(0.5)
    assert overall["accepted_error"]["count"] == 2
    assert overall["accepted_error"]["err_3d_median_m"] == pytest.approx(0.1, abs=1e-9)
    # The rejected-but-solved frame's error is reported, never hidden.
    assert overall["rejected_with_xyz_error"]["count"] == 1
    assert overall["rejected_with_xyz_error"]["err_3d_median_m"] == pytest.approx(1.0)


def test_zero_output_run_scores_zero_acceptance_not_na():
    ground_truth = _gt(FOUR_POSITIONS)
    samples = tuple(CandidateSample(obs.timestamp_s, None) for obs in ground_truth.observations)
    run = CandidateRun(samples=samples, accepted=(False, False, False, False))
    report = evaluate_candidate(ground_truth, run)
    assert report["overall"]["acceptance_rate"] == 0.0
    assert report["overall"]["accepted_error"] is None


# ---------------------------------------------------------------------------
# Input validation (fail closed)
# ---------------------------------------------------------------------------


def test_accepted_frame_without_xyz_rejected():
    ground_truth = _gt(FOUR_POSITIONS)
    samples = tuple(CandidateSample(obs.timestamp_s, None) for obs in ground_truth.observations)
    run = CandidateRun(samples=samples, accepted=(True, False, False, False))
    with pytest.raises(Metric3DEvalError, match="accepted"):
        evaluate_candidate(ground_truth, run)


def test_timestamp_mismatch_rejected():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.0, 0.0, 0.0))
    shifted = CandidateRun(
        samples=(CandidateSample(0.5, run.samples[0].xyz_world_m),) + run.samples[1:],
        accepted=run.accepted,
    )
    with pytest.raises(Metric3DEvalError, match="timestamp"):
        evaluate_candidate(ground_truth, shifted)


def test_length_mismatch_rejected():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.0, 0.0, 0.0))
    truncated = CandidateRun(samples=run.samples[:-1], accepted=run.accepted[:-1])
    with pytest.raises(Metric3DEvalError, match="sample count"):
        evaluate_candidate(ground_truth, truncated)


def test_invalid_near_half_rejected():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.0, 0.0, 0.0))
    with pytest.raises(Metric3DEvalError, match="near_half"):
        evaluate_candidate(ground_truth, run, near_half="left")


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


def test_slice_partitions_sum_to_total():
    ground_truth = _gt(FOUR_POSITIONS)  # 2 frames y<0, 2 frames y>=0
    run = _offset_run(ground_truth, (0.1, 0.0, 0.0), accepted=[True, True, False, True])
    report = evaluate_candidate(
        ground_truth,
        run,
        observed_mask=[True, False, True, True],
        gt_events=[EventEstimate("bounce", 1.5 / 30.0, (0.3, 0.0, 0.037))],
    )
    total = report["frame_count"]
    for name in ("court_half", "detection", "bounce_phase", "acceptance"):
        partitions = report["slices"][name]["partitions"]
        assert partitions is not None, name
        assert sum(block["frame_count"] for block in partitions.values()) == total, name
        accepted_sum = sum(block["accepted_frame_count"] for block in partitions.values())
        assert accepted_sum == report["overall"]["accepted_frame_count"], name


def test_court_half_slice_counts_and_near_far_annotation():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.1, 0.0, 0.0))
    report = evaluate_candidate(ground_truth, run, near_half="y_negative")
    court = report["slices"]["court_half"]
    assert court["near_half"] == "y_negative"
    assert court["far_half"] == "y_positive"
    assert court["partitions"]["y_negative"]["frame_count"] == 2
    assert court["partitions"]["y_positive"]["frame_count"] == 2
    # Without the parameter, near/far is never guessed.
    unannotated = evaluate_candidate(ground_truth, run)
    assert unannotated["slices"]["court_half"]["near_half"] is None
    assert unannotated["slices"]["court_half"]["far_half"] is None


def test_detection_slice_not_measured_without_mask():
    ground_truth = _gt(FOUR_POSITIONS)
    report = evaluate_candidate(ground_truth, _offset_run(ground_truth, (0.0, 0.0, 0.0)))
    detection = report["slices"]["detection"]
    assert detection["status"] == "not_measured"
    assert detection["reason"] == "missing_observed_mask"
    assert detection["partitions"] is None


def test_bounce_phase_slice_split_at_first_gt_bounce():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.0, 0.0, 0.0))
    report = evaluate_candidate(
        ground_truth,
        run,
        gt_events=[EventEstimate("bounce", 1.5 / 30.0, (0.3, 0.0, 0.037))],
    )
    bounce_phase = report["slices"]["bounce_phase"]
    assert bounce_phase["partitions"]["pre_bounce"]["frame_count"] == 2
    assert bounce_phase["partitions"]["post_bounce"]["frame_count"] == 2
    no_events = evaluate_candidate(ground_truth, run)
    assert no_events["slices"]["bounce_phase"]["status"] == "not_measured"


def test_acceptance_slice_partitions_by_mask():
    ground_truth = _gt(FOUR_POSITIONS)
    run = _offset_run(ground_truth, (0.1, 0.0, 0.0), accepted=[True, False, False, True])
    report = evaluate_candidate(ground_truth, run)
    acceptance = report["slices"]["acceptance"]
    assert acceptance["partitions"]["accepted"]["frame_count"] == 2
    assert acceptance["partitions"]["rejected"]["frame_count"] == 2
    # Rejected frames still had solutions; their error is visible here.
    assert acceptance["partitions"]["rejected"]["rejected_with_xyz_error"]["count"] == 2


# ---------------------------------------------------------------------------
# Event metrics scaffold
# ---------------------------------------------------------------------------


def test_bounce_event_time_and_position_error_known_answer():
    ground_truth = _gt(FOUR_POSITIONS)
    run = CandidateRun(
        samples=tuple(
            CandidateSample(obs.timestamp_s, obs.xyz_world_m) for obs in ground_truth.observations
        ),
        accepted=(True, True, True, True),
        events=(
            EventEstimate("bounce", 1.05, (0.3, 1.4, 0.037)),
            EventEstimate("bounce", 9.0, (5.0, 5.0, 0.037)),  # unmatched extra
        ),
    )
    report = evaluate_candidate(
        ground_truth,
        run,
        gt_events=[EventEstimate("bounce", 1.0, (0.0, 1.0, 0.037))],
    )
    bounce = report["events"]["bounce"]
    assert bounce["gt_count"] == 1
    assert bounce["candidate_count"] == 2
    assert bounce["matched_count"] == 1
    assert bounce["unmatched_candidate_count"] == 1
    assert bounce["time_error_s"]["median"] == pytest.approx(0.05, abs=1e-9)
    # dx=0.3, dy=0.4, dz=0 -> 3-4-5 triangle -> 0.5 m.
    assert bounce["position_error_m"]["median"] == pytest.approx(0.5, abs=1e-9)


def test_apex_height_error_known_answer():
    ground_truth = _gt(FOUR_POSITIONS)
    run = CandidateRun(
        samples=tuple(
            CandidateSample(obs.timestamp_s, obs.xyz_world_m) for obs in ground_truth.observations
        ),
        accepted=(True, True, True, True),
        events=(EventEstimate("apex", 0.06, height_m=2.25),),
    )
    report = evaluate_candidate(
        ground_truth,
        run,
        gt_events=[EventEstimate("apex", 0.05, (0.2, -0.5, 2.0))],
    )
    apex = report["events"]["apex"]
    assert apex["matched_count"] == 1
    assert apex["height_error_m"]["median"] == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Frozen-judge protocol surface
# ---------------------------------------------------------------------------


def test_mvp_thresholds_echoed_and_marked_provisional():
    ground_truth = _gt(FOUR_POSITIONS)
    report = evaluate_candidate(ground_truth, _offset_run(ground_truth, (0.1, -0.2, 0.3)))
    thresholds = report["frozen_judge"]["mvp_thresholds"]
    assert thresholds["accepted_median_3d_m"] == MVP_ACCEPTED_MEDIAN_3D_M == 0.25
    assert thresholds["bounce_median_m"] == MVP_BOUNCE_MEDIAN_M == 0.20
    assert thresholds["status"] == MVP_THRESHOLD_STATUS
    assert "not_a_promotion_gate" in MVP_THRESHOLD_STATUS
    check = report["mvp_threshold_check"]
    # sqrt(0.14) ~= 0.374 m > 0.25 m -> under_threshold False, still reported.
    assert check["accepted_median_3d_m"]["under_threshold"] is False
    assert check["bounce_median_m"]["measured_m"] is None
    assert check["bounce_median_m"]["under_threshold"] is None


def test_acceptance_rate_always_next_to_accuracy():
    ground_truth = _gt(FOUR_POSITIONS)
    report = evaluate_candidate(ground_truth, _offset_run(ground_truth, (0.01, 0.0, 0.0)))
    assert report["frozen_judge"]["acceptance_rate_always_reported"] is True
    # Every metrics block carries acceptance_rate beside accepted_error.
    assert "acceptance_rate" in report["overall"]
    for name in ("court_half", "acceptance"):
        for block in report["slices"][name]["partitions"].values():
            assert "acceptance_rate" in block
            assert "accepted_error" in block


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_report_bytes_are_deterministic(tmp_path):
    ground_truth = _gt(FOUR_POSITIONS)
    kwargs = dict(
        observed_mask=[True, True, False, True],
        near_half="y_negative",
        gt_events=[EventEstimate("bounce", 1.5 / 30.0, (0.3, 0.0, 0.037))],
    )
    run = _offset_run(ground_truth, (0.1, -0.2, 0.3), accepted=[True, True, False, True])
    first = dumps_report_json(evaluate_candidate(ground_truth, run, **kwargs))
    second = dumps_report_json(evaluate_candidate(ground_truth, run, **kwargs))
    assert first == second
    path = write_report_json(tmp_path / "report.json", evaluate_candidate(ground_truth, run, **kwargs))
    assert path.read_text(encoding="utf-8") == first


# ---------------------------------------------------------------------------
# Oracle-variant hooks
# ---------------------------------------------------------------------------


def test_oracle_variants_scored_by_same_judge():
    ground_truth = _gt(FOUR_POSITIONS)
    predicted = _offset_run(ground_truth, (0.1, 0.4, 0.1))
    oracle_events = _offset_run(ground_truth, (0.1, 0.2, 0.1))
    oracle_anchors = _offset_run(ground_truth, (0.05, 0.1, 0.05))
    report = evaluate_variants(
        ground_truth,
        {
            VARIANT_PREDICTED: predicted,
            VARIANT_ORACLE_EVENTS: oracle_events,
            VARIANT_ORACLE_ANCHORS: oracle_anchors,
        },
    )
    assert report["variant_names"] == sorted(
        [VARIANT_PREDICTED, VARIANT_ORACLE_EVENTS, VARIANT_ORACLE_ANCHORS]
    )
    systems = report["systems"]
    assert set(systems) == {VARIANT_PREDICTED, VARIANT_ORACLE_EVENTS, VARIANT_ORACLE_ANCHORS}
    predicted_mae = systems[VARIANT_PREDICTED]["overall"]["accepted_error"]["mae_y_m"]
    oracle_mae = systems[VARIANT_ORACLE_EVENTS]["overall"]["accepted_error"]["mae_y_m"]
    assert predicted_mae == pytest.approx(0.4)
    assert oracle_mae == pytest.approx(0.2)
    # Insertion order must not matter (deterministic output).
    reordered = evaluate_variants(
        ground_truth,
        {
            VARIANT_ORACLE_ANCHORS: oracle_anchors,
            VARIANT_ORACLE_EVENTS: oracle_events,
            VARIANT_PREDICTED: predicted,
        },
    )
    assert dumps_report_json(report) == dumps_report_json(reordered)


def test_evaluate_variants_requires_at_least_one_run():
    ground_truth = _gt(FOUR_POSITIONS)
    with pytest.raises(Metric3DEvalError, match="at least one"):
        evaluate_variants(ground_truth, {})
