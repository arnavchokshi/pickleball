from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from threed.racketsport.ball_global_track import (
    BALL_RADIUS_M,
    GlobalTrackConfig,
    build_global_ball_track,
)


FPS = 30.0
CALIBRATION = {
    "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
    "extrinsics": {
        # world (x, y, z-up) -> camera (x, -z, y-depth), with positive offset
        "R": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        "t": [0.0, 0.0, 12.0],
    },
}
BASE_EVENTS = [
    {"frame": 0, "t": 0.0, "kind": "serve", "world_xyz": [-1.0, 0.0, 1.0]},
    {"frame": 15, "t": 0.5, "kind": "bounce", "world_xyz": [0.0, 4.0, BALL_RADIUS_M]},
    {"frame": 30, "t": 1.0, "kind": "contact", "world_xyz": [1.0, 7.0, 1.0]},
]


def _point(events: Sequence[Mapping[str, Any]], frame: int, gravity: float = 9.80665) -> np.ndarray:
    ordered = sorted(events, key=lambda item: int(item["frame"]))
    segment = 0
    for index in range(len(ordered) - 1):
        if int(ordered[index]["frame"]) <= frame <= int(ordered[index + 1]["frame"]):
            segment = index
            break
    left, right = ordered[segment], ordered[segment + 1]
    alpha = (frame - int(left["frame"])) / (int(right["frame"]) - int(left["frame"]))
    duration = float(right["t"]) - float(left["t"])
    dt = alpha * duration
    p0 = np.asarray(left["world_xyz"], dtype=float)
    p1 = np.asarray(right["world_xyz"], dtype=float)
    acceleration = np.asarray([0.0, 0.0, -gravity])
    return (1.0 - alpha) * p0 + alpha * p1 + 0.5 * acceleration * (
        dt * dt - alpha * duration * duration
    )


def _project(point: Sequence[float]) -> tuple[float, float, float]:
    rotation = np.asarray(CALIBRATION["extrinsics"]["R"], dtype=float)
    translation = np.asarray(CALIBRATION["extrinsics"]["t"], dtype=float)
    camera = rotation @ np.asarray(point, dtype=float) + translation
    return (
        800.0 * float(camera[0]) / float(camera[2]) + 640.0,
        800.0 * float(camera[1]) / float(camera[2]) + 360.0,
        float(camera[2]),
    )


def _candidate_payload(
    events: Sequence[Mapping[str, Any]] = BASE_EVENTS,
    *,
    missing: set[int] | None = None,
    true_score: float = 0.05,
    last_frame: int = 30,
) -> dict[str, Any]:
    missing = missing or set()
    frames = []
    for frame in range(last_frame + 1):
        candidates = []
        if frame not in missing:
            x, y, _ = _project(_point(events, frame))
            candidates = [
                {"xy": [x + 70.0, y - 40.0], "score": 0.99, "source_detector": "distractor"},
                {"xy": [x, y], "score": true_score, "source_detector": "low_score_true"},
            ]
        frames.append({"frame": frame, "candidates": candidates})
    return {"schema_version": 1, "fps": FPS, "source": "synthetic", "frames": frames}


def _build(
    *,
    payload: Mapping[str, Any] | None = None,
    events: Sequence[Mapping[str, Any]] = BASE_EVENTS,
    size: Mapping[str, Any] | None = None,
    config: GlobalTrackConfig | None = None,
) -> dict[str, Any]:
    return build_global_ball_track(
        payload or _candidate_payload(events),
        calibration=CALIBRATION,
        event_boundaries={"selected": list(events)},
        ball_size_observations=size,
        config=config or GlobalTrackConfig(min_members_per_segment=3),
        clip_id="synthetic_rally",
    )


def test_selects_low_score_candidate_by_track_consistency_not_score() -> None:
    result = _build()

    assert result["status"] == "candidate_track"
    measured = [item for item in result["frames"] if item["band"] == "measured-candidate"]
    assert measured
    assert {item["candidate_source"] for item in measured} == {"low_score_true"}
    assert max(item["candidate_score"] for item in measured) == pytest.approx(0.05)
    assert result["diagnostics"]["candidate_member_below_0_5_count"] == len(measured)
    assert result["policy"]["candidate_score_floor"] is None
    assert result["inputs"]["all_supplied_scores_eligible"] is True


def test_every_sample_has_honest_provenance_posterior_and_covariance() -> None:
    result = _build(payload=_candidate_payload(missing={7, 8, 9}))

    assert result["status"] == "candidate_track"
    assert result["summary"]["physics_predicted_count"] == 3
    assert result["summary"]["posterior_mislabeled_measured_count"] == 0
    for sample in result["frames"]:
        assert sample["band"] in {"measured-candidate", "physics_predicted"}
        assert sample["provenance_band"] == sample["band"]
        assert sample["measured"] is False
        assert sample["measurement_authority"] is False
        assert 0.0 < sample["posterior"] <= 1.0
        covariance = np.asarray(sample["covariance_position_m2"])
        assert covariance.shape == (3, 3)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-9


def test_six_frame_hole_is_physics_predicted_and_continuous() -> None:
    result = _build(payload=_candidate_payload(missing=set(range(6, 12))))

    assert result["status"] == "candidate_track"
    predicted_frames = [item["frame"] for item in result["frames"] if item["band"] == "physics_predicted"]
    assert predicted_frames == list(range(6, 12))
    assert result["summary"]["max_interior_gap_frames"] == 6
    assert result["summary"]["serve_to_terminal_continuous"] is True


def test_seven_frame_hole_refuses_entire_rally() -> None:
    result = _build(payload=_candidate_payload(missing=set(range(6, 13))))

    assert result["status"] == "refused"
    assert result["refusal"]["type"] == "rally_track_refusal"
    assert result["refusal"]["code"] == "interior_hole_above_ceiling"
    assert result["refusal"]["fail_closed"] is True
    assert result["frames"] == []
    assert result["summary"]["emitted_sample_count"] == 0


def test_no_candidates_returns_typed_rally_refusal() -> None:
    result = _build(payload={"fps": FPS, "frames": [{"frame": frame, "candidates": []} for frame in range(31)]})

    assert result["status"] == "refused"
    assert result["refusal"]["code"] == "no_2d_candidates"
    assert result["refusal"]["samples_suppressed"] is True


def test_segments_share_exact_event_boundary_and_bounce_is_at_radius() -> None:
    result = _build()

    assert result["status"] == "candidate_track"
    assert result["segments"][0]["frame_end"] == result["segments"][1]["frame_start"] == 15
    boundary = next(item for item in result["boundaries"] if item["kind"] == "bounce")
    assert boundary["shared_by_adjacent_segments"] is True
    assert boundary["fitted_world_xyz"][2] == pytest.approx(BALL_RADIUS_M, abs=1.0e-12)
    bounce = result["diagnostics"]["bounce_at_radius"]
    assert bounce["consistent_count"] == bounce["bounce_count"] == 1


def test_out_of_sequence_event_is_quarantined_not_used_as_boundary() -> None:
    event = {
        "frame": 10,
        "t": 1.0 / 3.0,
        "kind": "bounce",
        "world_xyz": [8.0, 8.0, 3.0],
        "out_of_sequence": True,
        "source": "exception_candidate",
    }
    result = _build(
        events=[BASE_EVENTS[0], event, *BASE_EVENTS[1:]],
        payload=_candidate_payload(BASE_EVENTS),
    )

    assert result["status"] == "candidate_track"
    assert [item["frame"] for item in result["boundaries"]] == [0, 15, 30]
    assert result["quarantined_events"] == [
        {
            "type": "out_of_sequence_event",
            "frame": 10,
            "kind": "bounce",
            "source": "exception_candidate",
            "used_as_boundary": False,
        }
    ]


def test_serve_initialization_discards_pre_contact_window_endpoint() -> None:
    pre = {
        "frame": -3,
        "t": -0.1,
        "kind": "rally_endpoint",
        "world_xyz": [-1.5, -1.0, BALL_RADIUS_M],
    }
    events = [pre, *BASE_EVENTS]
    result = _build(events=events, payload=_candidate_payload(BASE_EVENTS))

    assert result["status"] == "candidate_track"
    assert result["serve_initialization"]["type"] == "serve_from_first_contact"
    assert result["serve_initialization"]["frame"] == 0
    assert result["boundaries"][0]["frame"] == 0
    assert result["window"]["pre_pad_s"] == pytest.approx(0.9)
    assert result["window"]["post_pad_s"] == pytest.approx(1.1)


def test_missing_contact_uses_typed_weak_endpoint_preview_policy() -> None:
    events = [
        {**BASE_EVENTS[0], "kind": "rally_endpoint"},
        BASE_EVENTS[1],
        {**BASE_EVENTS[2], "kind": "rally_endpoint"},
    ]
    result = _build(events=events, payload=_candidate_payload(events))

    assert result["status"] == "candidate_track"
    assert result["serve_initialization"]["type"] == "weak_endpoint_preview_no_contact_available"
    assert result["serve_initialization"]["fallback_used"] is True
    assert result["diagnostics"]["serve_to_terminal_emission_policy"] == "track_extent_no_contact_available"


def test_speed_above_35_mps_refuses_without_emission() -> None:
    events = [
        {"frame": 0, "t": 0.0, "kind": "serve", "world_xyz": [-1.0, 0.0, 1.0]},
        {"frame": 15, "t": 0.5, "kind": "contact", "world_xyz": [24.0, 0.0, 1.0]},
    ]
    result = _build(events=events, payload=_candidate_payload(events, last_frame=15))

    assert result["status"] == "refused"
    assert result["refusal"]["code"] == "speed_ceiling_exceeded"
    assert result["frames"] == []
    assert result["diagnostics"]["speed_ceiling_violations"]


def test_radius_residual_is_confidence_gated_and_per_rally_calibrated() -> None:
    frames = []
    for frame in range(31):
        point = _point(BASE_EVENTS, frame)
        x, y, depth = _project(point)
        radius = 100.0 / depth
        frames.append(
            {
                "frame": frame,
                "blobs": [
                    {
                        "center_xy_px": [x, y],
                        "radius_proxy_px": radius,
                        "heatmap_peak": 0.9,
                    }
                ],
            }
        )
    config = GlobalTrackConfig(
        min_members_per_segment=3,
        radius_min_observations=8,
        radius_min_r2=0.2,
    )
    result = _build(size={"frames": frames}, config=config)

    assert result["status"] == "candidate_track"
    radius = result["diagnostics"]["radius_residual"]
    assert radius["status"] == "active"
    assert radius["per_rally_calibration"] is True
    assert radius["universal_linear_law_assumed"] is False
    assert radius["slope"] == pytest.approx(-1.0, abs=0.05)


def test_radius_residual_abstains_below_confidence_gate() -> None:
    frames = []
    for frame in range(31):
        x, y, depth = _project(_point(BASE_EVENTS, frame))
        frames.append(
            {
                "frame": frame,
                "blobs": [
                    {
                        "center_xy_px": [x, y],
                        "radius_proxy_px": 100.0 / depth,
                        "heatmap_peak": 0.69,
                    }
                ],
            }
        )
    result = _build(size={"frames": frames})

    assert result["status"] == "candidate_track"
    radius = result["diagnostics"]["radius_residual"]
    assert radius["status"] == "abstained"
    assert radius["reason"] == "insufficient_confident_observations"


def test_physics_reintegration_is_exact_for_every_emitted_segment() -> None:
    result = _build(payload=_candidate_payload(missing={7, 8, 9}))

    assert result["status"] == "candidate_track"
    physics = result["diagnostics"]["physics_reintegration"]
    assert physics["pass_rate"] == 1.0
    assert physics["pass_count"] == physics["segment_count"] == 2
    assert result["diagnostics"]["step_speed"]["over_35_mps_count"] == 0


def test_primary_ball_track_is_eligible_without_becoming_measured_authority() -> None:
    payload = _candidate_payload(missing={4})
    x, y, _ = _project(_point(BASE_EVENTS, 4))
    ball_track = {
        "fps": FPS,
        "source": "wasb",
        "frames": [
            {"visible": frame == 4, "xy": [x, y] if frame == 4 else [0.0, 0.0], "conf": 0.2}
            for frame in range(31)
        ],
    }
    result = build_global_ball_track(
        payload,
        calibration=CALIBRATION,
        event_boundaries={"selected": BASE_EVENTS},
        ball_track=ball_track,
        config=GlobalTrackConfig(min_members_per_segment=3),
    )

    assert result["status"] == "candidate_track"
    frame = next(item for item in result["frames"] if item["frame"] == 4)
    assert frame["candidate_source"] == "primary:wasb"
    assert frame["band"] == "measured-candidate"
    assert frame["measurement_authority"] is False


def test_configuration_cannot_relax_hole_ceiling_above_six() -> None:
    with pytest.raises(ValueError, match="max_hole_frames"):
        GlobalTrackConfig(max_hole_frames=7)


def test_invalid_calibration_refuses_instead_of_emitting() -> None:
    result = build_global_ball_track(
        _candidate_payload(),
        calibration={},
        event_boundaries={"selected": BASE_EVENTS},
        config=GlobalTrackConfig(min_members_per_segment=3),
    )

    assert result["status"] == "refused"
    assert result["refusal"]["code"] == "invalid_geometry_inputs"
    assert result["frames"] == []


def test_covariance_grows_for_physics_predicted_hole_samples() -> None:
    result = _build(payload=_candidate_payload(missing={7, 8, 9}))

    assert result["status"] == "candidate_track"
    covariance_by_frame = {
        item["frame"]: np.trace(np.asarray(item["covariance_position_m2"])) for item in result["frames"]
    }
    assert covariance_by_frame[8] > covariance_by_frame[7]
    assert covariance_by_frame[8] > covariance_by_frame[6]
    assert math.isfinite(covariance_by_frame[8])
