from __future__ import annotations

import math
from typing import Any, Callable

from threed.racketsport.ball_flight_sanity import evaluate_ball_flight_sanity


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
