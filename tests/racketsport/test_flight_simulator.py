from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.ball_arc_solver import BallArcSolverConfig, PhysicsParameters, _rk4_step
from threed.racketsport.flight_simulator import (
    BounceParameters,
    DetectorNoiseProfile,
    FlightSimulationConfig,
    apply_detector_noise,
    build_flight_sanity_artifact,
    detector_noise_stats,
    evaluate_simulated_flight_sanity,
    generate_corpus,
    generate_trajectory_pair,
    load_court_calibration,
    round_trip_fit_report,
    sample_shot_family,
    simulate_flight,
)


EVAL_CALIBRATION = Path(
    "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json"
)


def test_zero_spin_step_reuses_solver_rk4_core() -> None:
    physics = PhysicsParameters.for_ball_type("outdoor")
    config = FlightSimulationConfig(dt_s=1.0 / 120.0, max_time_s=0.25, spin_scalar=0.0)
    shot = sample_shot_family("drive", np.random.default_rng(4), start_side="near")
    shot = shot.with_velocity((5.2, 8.0, 1.7)).with_position((0.0, -4.0, 1.0))

    trajectory = simulate_flight(shot, physics=physics, config=config)

    expected = _rk4_step((*shot.position_m, *shot.velocity_mps), config.dt_s, physics)
    second = trajectory.samples[1]
    assert (second.position_m + second.velocity_mps) == pytest.approx(expected, abs=1e-12)
    assert trajectory.metadata["physics_core"] == "ball_arc_solver._rk4_step"


def test_magnus_extension_is_simulator_local_and_uses_steyn_cl_formula() -> None:
    physics = PhysicsParameters.for_ball_type("outdoor")
    shot = sample_shot_family("lob", np.random.default_rng(9), start_side="near")
    shot = shot.with_velocity((0.0, 9.0, 3.0)).with_position((0.0, -4.5, 1.0))
    no_spin = simulate_flight(shot, physics=physics, config=FlightSimulationConfig(spin_scalar=0.0, max_time_s=0.5))
    lift_spin = simulate_flight(shot, physics=physics, config=FlightSimulationConfig(spin_scalar=0.8, max_time_s=0.5))

    assert lift_spin.samples[20].position_m[2] > no_spin.samples[20].position_m[2]
    assert lift_spin.metadata["magnus"]["cl"] == pytest.approx(0.195 * 0.8)
    assert lift_spin.metadata["magnus"]["source"] == "Steyn arXiv:2501.00163 Cl=0.195*S"


def test_bounce_model_uses_unmeasured_defaults_pending_h13() -> None:
    physics = PhysicsParameters.no_drag()
    shot = sample_shot_family("dink", np.random.default_rng(2), start_side="near")
    shot = shot.with_position((0.0, -1.0, 0.45)).with_velocity((0.0, 2.5, -1.8))
    bounce = BounceParameters(restitution=0.55, friction=0.20)

    trajectory = simulate_flight(
        shot,
        physics=physics,
        config=FlightSimulationConfig(max_time_s=0.9, bounce=bounce, max_bounces=1),
    )

    assert trajectory.bounces, "fixture should hit the court"
    event = trajectory.bounces[0]
    assert event["model_status"] == "unmeasured_default_pending_H13"
    assert event["pre_velocity_mps"][2] < 0.0
    assert event["post_velocity_mps"][2] == pytest.approx(-bounce.restitution * event["pre_velocity_mps"][2], rel=0.02)
    assert abs(event["post_velocity_mps"][1]) < abs(event["pre_velocity_mps"][1])


def test_shot_family_samplers_are_seed_deterministic_and_range_labeled() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    families = ["serve", "drive", "drop", "lob", "dink"]

    samples_a = [sample_shot_family(family, rng_a, start_side="near") for family in families]
    samples_b = [sample_shot_family(family, rng_b, start_side="near") for family in families]

    assert [sample.to_json() for sample in samples_a] == [sample.to_json() for sample in samples_b]
    for family, sample in zip(families, samples_a, strict=True):
        assert sample.family == family
        assert sample.assumptions["status"] == "plausible_unmeasured_prior"
        assert sample.speed_mps_min <= sample.speed_mps <= sample.speed_mps_max
        assert sample.launch_angle_deg_min <= sample.launch_angle_deg <= sample.launch_angle_deg_max


def test_projection_uses_real_eval_calibration_fixture() -> None:
    calibration = load_court_calibration(EVAL_CALIBRATION)
    rng = np.random.default_rng(7)

    record = generate_trajectory_pair(
        trajectory_id="projection-fixture",
        rng=rng,
        calibration=calibration,
        family="drive",
        clean_only=True,
    )

    assert record["projection"]["calibration_path"].endswith("court_calibration_metric15pt.json")
    assert record["projection"]["schema"] == "CourtCalibration"
    visible = [frame for frame in record["clean_2d_track"] if frame["visible"]]
    assert len(visible) >= 10
    assert all(np.isfinite(frame["xy_px"]).all() for frame in visible[:10])


def test_detector_noise_matches_measured_profile_within_twenty_percent() -> None:
    profile = DetectorNoiseProfile(p95_jitter_px=34.0, recall=0.578, hidden_fp_rate=0.021)
    clean = [
        {"frame": idx, "t": idx / 60.0, "xy_px": [600.0 + idx * 0.1, 300.0], "visible": True}
        for idx in range(6000)
    ]

    noisy = apply_detector_noise(clean, rng=np.random.default_rng(99), image_size=(1920, 1080), profile=profile)
    stats = detector_noise_stats(clean, noisy)

    assert stats["jitter_p95_px"] == pytest.approx(profile.p95_jitter_px, rel=0.20)
    assert stats["recall"] == pytest.approx(profile.recall, rel=0.20)
    assert stats["hidden_fp_rate"] == pytest.approx(profile.hidden_fp_rate, rel=0.20)
    assert stats["target_profile"] == profile.to_json()


def test_simulated_clean_track_passes_ball_flight_sanity() -> None:
    calibration = load_court_calibration(EVAL_CALIBRATION)
    record = generate_trajectory_pair(
        trajectory_id="sanity-fixture",
        rng=np.random.default_rng(11),
        calibration=calibration,
        family="drop",
        clean_only=True,
    )

    artifact = build_flight_sanity_artifact(record)
    report = evaluate_simulated_flight_sanity(record)

    assert artifact["artifact_type"] == "racketsport_ball_track_arc_solved"
    assert report["summary"]["failed_segment_count"] == 0
    assert report["summary"]["demoted_frame_count"] == 0


def test_round_trip_reports_solver_position_error_on_clean_track() -> None:
    calibration = load_court_calibration(EVAL_CALIBRATION)
    shot = sample_shot_family("drive", np.random.default_rng(5), start_side="near")
    shot = shot.with_position((0.0, -4.0, 1.1)).with_velocity((0.8, 8.5, 2.2))
    record = generate_trajectory_pair(
        trajectory_id="roundtrip-fixture",
        rng=np.random.default_rng(5),
        calibration=calibration,
        shot=shot,
        config=FlightSimulationConfig(spin_scalar=0.0, max_time_s=0.65),
        clean_only=True,
    )

    report = round_trip_fit_report(
        record,
        calibration,
        physics=PhysicsParameters.for_ball_type("outdoor"),
        solver_config=BallArcSolverConfig(max_reprojection_inlier_px=8.0, robust_pixel_sigma=2.0),
    )

    assert report["status"] == "fit"
    assert report["sample_count"] >= 10
    assert report["position_error_m"]["p95"] < 0.20
    assert report["fit"]["status"].startswith("fit")


def test_generate_corpus_is_deterministic_and_fast_for_small_cpu_sample() -> None:
    calibration = load_court_calibration(EVAL_CALIBRATION)

    started = time.perf_counter()
    first_records, first_report = generate_corpus(count=25, seed=1234, calibration=calibration, roundtrip_samples=3)
    elapsed = time.perf_counter() - started
    second_records, second_report = generate_corpus(count=25, seed=1234, calibration=calibration, roundtrip_samples=3)

    assert elapsed < 5.0
    assert first_records == second_records
    assert first_report["deterministic_seed"] == second_report["deterministic_seed"] == 1234
    assert first_report["acceptance"]["noise_profile"]["within_20_percent"] is True
    assert first_report["acceptance"]["flight_sanity"]["failed_segments"] == 0
    assert first_report["performance"]["trajectory_count"] == 25


def test_generate_flight_corpus_cli_emits_jsonl_and_report(tmp_path: Path) -> None:
    out_jsonl = tmp_path / "corpus.jsonl"
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/generate_flight_corpus.py",
            "--count",
            "5",
            "--seed",
            "42",
            "--calibration",
            str(EVAL_CALIBRATION),
            "--out",
            str(out_jsonl),
            "--report",
            str(report_path),
            "--roundtrip-samples",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(completed.stdout)
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert stdout["artifact_type"] == "racketsport_flight_corpus_generation"
    assert stdout["count"] == 5
    assert len(rows) == 5
    assert rows[0]["truth_3d"]["samples"]
    assert rows[0]["noisy_2d_detections"]["detections"]
    assert report["acceptance"]["noise_profile"]["within_20_percent"] is True
    assert report["round_trip"]["samples_evaluated"] == 2
