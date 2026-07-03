from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack
from scripts.racketsport.convert_vn_trajectories import convert_vn_trajectories_to_ball_track


def _harness_json(*, confidence_a: float = 0.9, confidence_b: float = 0.5) -> dict:
    """A minimal but schema-faithful VNDetectTrajectorySpike raw harness output.

    Models the real observed VNDetectTrajectoriesRequest quirk this converter
    exists to handle: the same `observation_uuid` ("traj-a") is emitted twice
    (frames 2 and 3) with byte-identical detected_points/confidence -- only
    the first (frame 2) emission should be used. A second, lower-confidence,
    overlapping candidate ("traj-b") covers frame 1 only, to exercise the
    highest-confidence-wins tie-break.
    """

    return {
        "schema_version": 1,
        "artifact_type": "vn_trajectories_spike_raw",
        "status": "TESTED-ON-REAL-DATA",
        "source_video": "input_0000_0010.mp4",
        "video": {"width": 100, "height": 50, "fps": 30.0, "frame_count": 4, "duration_s": 0.1333},
        "request_config": {
            "frame_analysis_spacing": "zero",
            "trajectory_length": 3,
            "object_minimum_normalized_radius": 0.002,
            "object_maximum_normalized_radius": 0.02,
            "vision_request_revision": 1,
        },
        "run": {
            "frames_fed": 4,
            "perform_call_count": 4,
            "emission_count": 3,
            "wall_clock_seconds": 0.01,
            "frames_per_second_processed": 400.0,
        },
        "frame_pts_s": [0.0, 0.0333, 0.0666, 0.1],
        "trajectories": [
            {
                "observation_uuid": "traj-b",
                "emitted_at_frame_index": 1,
                "confidence": confidence_b,
                "moving_average_radius_normalized": 0.01,
                "equation_coefficients": [0.0, 0.0, 0.0],
                "time_range_start_s": 0.0333,
                "time_range_duration_s": 0.0,
                "detected_points": [
                    {"frame_index": 1, "t_s": 0.0333, "x_norm": 0.90, "y_norm": 0.80},
                ],
                "projected_points": [
                    {"frame_index": 1, "t_s": 0.0333, "x_norm": 0.90, "y_norm": 0.80},
                ],
            },
            {
                "observation_uuid": "traj-a",
                "emitted_at_frame_index": 2,
                "confidence": confidence_a,
                "moving_average_radius_normalized": 0.01,
                "equation_coefficients": [1.0, 2.0, 3.0],
                "time_range_start_s": 0.0,
                "time_range_duration_s": 0.0666,
                "detected_points": [
                    {"frame_index": 0, "t_s": 0.0, "x_norm": 0.20, "y_norm": 0.60},
                    {"frame_index": 1, "t_s": 0.0333, "x_norm": 0.50, "y_norm": 0.60},
                    {"frame_index": 2, "t_s": 0.0666, "x_norm": 0.80, "y_norm": 0.60},
                ],
                "projected_points": [
                    {"frame_index": 0, "t_s": 0.0, "x_norm": 0.20, "y_norm": 0.60},
                    {"frame_index": 1, "t_s": 0.0333, "x_norm": 0.50, "y_norm": 0.60},
                    {"frame_index": 2, "t_s": 0.0666, "x_norm": 0.80, "y_norm": 0.60},
                ],
            },
            {
                # Stale re-emission of "traj-a" one frame later: Vision's real
                # behavior is to keep already-found trajectories in
                # `request.results` for several subsequent perform() calls.
                # This must NOT be double-counted or treated as a fresh window.
                "observation_uuid": "traj-a",
                "emitted_at_frame_index": 3,
                "confidence": confidence_a,
                "moving_average_radius_normalized": 0.01,
                "equation_coefficients": [1.0, 2.0, 3.0],
                "time_range_start_s": 0.0,
                "time_range_duration_s": 0.0666,
                "detected_points": [
                    {"frame_index": 1, "t_s": 0.0333, "x_norm": 0.20, "y_norm": 0.60},
                    {"frame_index": 2, "t_s": 0.0666, "x_norm": 0.50, "y_norm": 0.60},
                    {"frame_index": 3, "t_s": 0.1, "x_norm": 0.80, "y_norm": 0.60},
                ],
                "projected_points": [],
            },
        ],
        "notes": [],
    }


def test_dedupes_stale_reemitted_uuid_and_keeps_first_emission() -> None:
    ball_track_payload, metadata = convert_vn_trajectories_to_ball_track(
        harness_json=_harness_json(),
        confidence_threshold=0.0,
    )
    BallTrack.model_validate(ball_track_payload)

    assert metadata["raw_emission_count"] == 3
    assert metadata["deduped_trajectory_count"] == 2  # traj-a's frame-3 stale repeat is dropped

    frames = ball_track_payload["frames"]
    assert len(frames) == 4
    # Frame 0: only traj-a covers it -> x_norm 0.20 -> pixel x = 0.20 * 100 = 20
    assert frames[0]["visible"] is True
    assert frames[0]["xy"][0] == pytest.approx(20.0)
    # y_norm 0.60, bottom-left origin -> top-left pixel y = (1 - 0.60) * 50 = 20
    assert frames[0]["xy"][1] == pytest.approx(20.0)

    # Frame 1: contested between traj-a (conf 0.9) and traj-b (conf 0.5) -> traj-a wins
    assert frames[1]["visible"] is True
    assert frames[1]["conf"] == pytest.approx(0.9)
    assert frames[1]["xy"][0] == pytest.approx(50.0)  # traj-a's x_norm 0.50 -> 50px, not traj-b's 90px

    # Frame 2: only traj-a's first emission covers it (frame_index 2, x_norm 0.80)
    assert frames[2]["visible"] is True
    assert frames[2]["xy"][0] == pytest.approx(80.0)

    # Frame 3: only covered by the stale re-emission's detected_points, which
    # was dropped entirely by dedup -> no candidate -> invisible.
    assert frames[3]["visible"] is False
    assert frames[3]["conf"] == pytest.approx(0.0)


def test_confidence_threshold_gates_visible_but_keeps_conf() -> None:
    ball_track_payload, _ = convert_vn_trajectories_to_ball_track(
        harness_json=_harness_json(confidence_a=0.4, confidence_b=0.9),
        confidence_threshold=0.5,
    )
    frames = ball_track_payload["frames"]
    # Frame 1 is contested; traj-b (conf 0.9) now wins the max-confidence tie-break.
    assert frames[1]["visible"] is True
    assert frames[1]["conf"] == pytest.approx(0.9)
    # Frame 0 is only covered by traj-a (conf 0.4), below the 0.5 threshold -> invisible,
    # but the raw confidence is still preserved on the frame.
    assert frames[0]["visible"] is False
    assert frames[0]["conf"] == pytest.approx(0.4)


def test_rejects_blocker_report() -> None:
    blocker = {
        "schema_version": 1,
        "artifact_type": "vn_trajectories_spike_blocker",
        "status": "BLOCKED",
        "blocked_reason": "os_version_unsupported",
        "detail": "needs macOS 11+",
    }
    with pytest.raises(ValueError, match="blocker report"):
        convert_vn_trajectories_to_ball_track(harness_json=blocker)


def test_cli_end_to_end(tmp_path: Path) -> None:
    harness_path = tmp_path / "raw_harness.json"
    harness_path.write_text(json.dumps(_harness_json()), encoding="utf-8")
    out_ball_track = tmp_path / "ball_track.json"
    out_metadata = tmp_path / "conversion_metadata.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/convert_vn_trajectories.py",
            "--harness-json",
            str(harness_path),
            "--out-ball-track",
            str(out_ball_track),
            "--out-metadata",
            str(out_metadata),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(completed.stdout)
    assert metadata["deduped_trajectory_count"] == 2
    assert out_ball_track.is_file()

    ball_track_payload = json.loads(out_ball_track.read_text(encoding="utf-8"))
    BallTrack.model_validate(ball_track_payload)
    assert ball_track_payload["source"] == "vn_trajectories"
