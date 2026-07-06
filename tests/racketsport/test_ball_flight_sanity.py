from __future__ import annotations

import math
from typing import Any, Callable

from threed.racketsport.ball_flight_sanity import apply_flight_sanity_demotions, evaluate_ball_flight_sanity


def test_flight_sanity_leaves_clean_parabola_untouched() -> None:
    artifact = _artifact_from_motion(lambda t: (6.0 * t, 0.25 * t, 0.6 + 4.2 * t - 0.5 * 9.80665 * t * t))

    report = evaluate_ball_flight_sanity(artifact)

    assert report["summary"]["failed_segment_count"] == 0
    assert report["summary"]["demoted_frame_count"] == 0
    assert all(item["demote"] is False for item in report["frames"])


def test_flight_sanity_demotes_segment_with_multiple_vertical_reversals() -> None:
    artifact = _artifact_from_motion(lambda t: (5.0 * t, 0.0, 1.5 + 0.3 * math.sin(4.0 * math.pi * t)))

    report = evaluate_ball_flight_sanity(artifact)

    segment = report["segments"][0]
    assert segment["verdict"] == "fail"
    assert "vertical_multi_apex" in segment["reasons"]
    assert report["summary"]["failed_segment_count"] == 1
    assert report["summary"]["demoted_frame_count"] > 0
    assert all(frame["demote"] is True for frame in report["frames"] if frame["segment_id"] == segment["segment_id"])


def test_flight_sanity_demotes_segment_with_interior_horizontal_reversal() -> None:
    def motion(t: float) -> tuple[float, float, float]:
        if t < 0.45:
            x = 5.0 * t
        elif t < 0.75:
            x = 4.5 - 5.0 * t
        else:
            x = 5.0 * t - 3.0
        return (x, 0.0, 0.6 + 2.5 * t - 0.5 * 9.80665 * t * t)

    report = evaluate_ball_flight_sanity(_artifact_from_motion(motion))

    assert report["segments"][0]["verdict"] == "fail"
    assert "horizontal_direction_reversal" in report["segments"][0]["reasons"]
    assert report["summary"]["demoted_frame_count"] > 0


def test_flight_sanity_flags_only_outside_court_volume_frames_from_solver_config() -> None:
    artifact = _artifact_from_motion(lambda t: (0.0, 0.0, 0.6 + 2.0 * t - 0.5 * 9.80665 * t * t))
    artifact["config"] = {"court_sport": "pickleball", "court_margin_m": 4.0, "court_z_min_m": -0.15}
    artifact["frames"][10]["world_xyz"] = [7.049, 0.0, 0.9]

    report = evaluate_ball_flight_sanity(artifact)

    assert report["schema_version"] == 2
    assert report["policy"]["suppresses_world_xyz_on_court_volume_failure"] is True
    assert report["policy"]["world_xyz_replacement_source"] == "bvp_anchor_fallback_or_null"
    assert "does_not_create_or_adjust_world_xyz" not in report["policy"]
    assert report["frames"][10]["demote"] is True
    assert report["frames"][10]["reasons"] == ["outside_court_volume"]
    assert report["frames"][9]["demote"] is False
    assert "outside_court_volume" in report["segments"][0]["reasons"]
    assert report["config"]["court_sport"] == "pickleball"
    assert report["config"]["court_margin_m"] == 4.0
    assert report["config"]["court_z_min_m"] == -0.15


def test_flight_sanity_suppression_scopes_to_court_volume_and_bvp_fallback() -> None:
    artifact = _artifact_from_motion(lambda t: (0.5 * t, 0.0, 0.8))
    artifact["segments"] = [
        {"segment_id": 0, "status": "fit", "frame_start": 0, "frame_end": 10},
        {"segment_id": 1, "status": "fit_bvp_fallback", "frame_start": 11, "frame_end": 30},
    ]
    artifact["frames"][5]["arc_solver"] = {"segment_id": 0}
    artifact["frames"][5]["world_xyz"] = [0.1, 0.0, 0.8]
    artifact["frames"][10]["arc_solver"] = {"segment_id": 0}
    artifact["frames"][10]["world_xyz"] = [8.0, 0.0, 0.8]
    artifact["frames"][20]["arc_solver"] = {"segment_id": 1}
    artifact["frames"][20]["world_xyz"] = [0.2, 0.0, 0.8]
    report = {
        "schema_version": 2,
        "frames": [
            {"frame": 5, "demote": True, "reasons": ["vertical_multi_apex"]},
            {"frame": 10, "demote": True, "reasons": ["outside_court_volume"]},
        ],
        "summary": {"demoted_frame_count": 2, "failed_segment_count": 1},
    }

    gated = apply_flight_sanity_demotions(artifact, report)

    assert gated["frames"][5]["world_xyz"] == [0.1, 0.0, 0.8]
    assert gated["frames"][5]["band"] == "arc_weak"
    assert gated["frames"][10]["world_xyz"] is None
    assert gated["frames"][10]["band"] == "hidden"
    assert gated["frames"][10]["flight_sanity_original"]["world_xyz"] == [8.0, 0.0, 0.8]
    assert gated["frames"][20]["world_xyz"] == [0.2, 0.0, 0.8]
    assert gated["frames"][20]["band"] == "arc_weak"
    assert gated["frames"][20]["flight_sanity_reasons"] == ["fit_bvp_fallback"]


def _artifact_from_motion(motion: Callable[[float], tuple[float, float, float]]) -> dict[str, Any]:
    fps = 30.0
    frame_count = 31
    frames = []
    for frame in range(frame_count):
        t = frame / fps
        world = motion(t)
        frames.append(
            {
                "t": t,
                "visible": True,
                "world_xyz": [round(world[0], 6), round(world[1], 6), round(world[2], 6)],
                "band": "anchored_measured",
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "synthetic",
        "fps": fps,
        "status": "ran",
        "anchors": [
            {
                "anchor_id": "start",
                "kind": "bounce",
                "t": frames[0]["t"],
                "frame": 0,
                "world_xyz": frames[0]["world_xyz"],
                "sigma_m": 0.05,
                "status": "human_reviewed",
            },
            {
                "anchor_id": "end",
                "kind": "bounce",
                "t": frames[-1]["t"],
                "frame": frame_count - 1,
                "world_xyz": frames[-1]["world_xyz"],
                "sigma_m": 0.05,
                "status": "human_reviewed",
            },
        ],
        "frames": frames,
        "summary": {
            "anchored_measured_count": frame_count,
            "arc_interpolated_count": 0,
            "arc_extrapolated_count": 0,
            "arc_weak_count": 0,
            "hidden_count": 0,
        },
    }
