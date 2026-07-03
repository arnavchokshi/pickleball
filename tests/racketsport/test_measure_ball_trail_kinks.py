from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.measure_ball_trail_kinks import (
    extract_ball_frames,
    horizontal_velocity_sign_changes,
    kink_report,
    segment_time_bounds,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_horizontal_velocity_sign_changes_is_zero_for_a_monotonic_flight() -> None:
    points = [[0.1 * index, -0.8 + 0.05 * index, 0.9 - 0.02 * index] for index in range(5)]
    assert horizontal_velocity_sign_changes(points) == 0


def test_horizontal_velocity_sign_changes_counts_reversals_per_axis() -> None:
    # x goes +, -, + (2 changes); y goes +, +, + (0 changes).
    points = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 3.0, 0.0]]
    assert horizontal_velocity_sign_changes(points) == 2


def test_horizontal_velocity_sign_changes_ignores_near_zero_noise() -> None:
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0000001, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert horizontal_velocity_sign_changes(points) == 0


def test_extract_ball_frames_prefers_world_shaped_payload() -> None:
    payload = {"ball": {"frames": [{"t": 0.0}]}, "frames": [{"t": 99.0}]}
    assert extract_ball_frames(payload) == [{"t": 0.0}]


def test_extract_ball_frames_falls_back_to_top_level_frames() -> None:
    payload = {"frames": [{"t": 0.0}, {"t": 1.0}]}
    assert extract_ball_frames(payload) == [{"t": 0.0}, {"t": 1.0}]


def test_segment_time_bounds_pairs_consecutive_selected_events_sorted_by_time() -> None:
    events_selected = {
        "selected": [
            {"t": 1.0, "frame": 30},
            {"t": 0.0, "frame": 0},
            {"t": 2.5, "frame": 75},
        ]
    }
    assert segment_time_bounds(events_selected) == [(0.0, 1.0), (1.0, 2.5)]


def test_kink_report_counts_kinks_only_within_segments_not_across_hidden_gaps() -> None:
    # Segment 0: [t=0, t=1] clean monotonic x. Segment 1: [t=1, t=3] has a
    # deviating mid-flight point at t=1.6 that reverses x once within the
    # segment (1.3 -> -5.0 -> 2.0). The shared anchor at t=1.0 is a
    # legitimate event boundary (assigned to exactly one segment, matching
    # `ball_arc_solver._segment_for_time`'s nearest-midpoint rule) so it must
    # never be double-counted as a fabricated within-segment reversal. A
    # frame at t=3.5 (outside every selected segment, e.g. an honestly hidden
    # tail) must not be counted even though it has a real world position.
    frames = [
        {"t": 0.0, "world_xyz": [0.0, 0.0, 0.0]},
        {"t": 0.5, "world_xyz": [0.5, 0.0, 0.0]},
        {"t": 1.0, "world_xyz": [1.0, 0.0, 0.0]},
        {"t": 1.3, "world_xyz": [1.3, 0.0, 0.0]},
        {"t": 1.6, "world_xyz": [-5.0, 0.0, 0.0]},
        {"t": 2.4, "world_xyz": [2.0, 0.0, 0.0]},
        {"t": 3.5, "world_xyz": [999.0, 999.0, 999.0]},
    ]
    events_selected = {"selected": [{"t": 0.0}, {"t": 1.0}, {"t": 3.0}]}

    report = kink_report(frames, segment_time_bounds(events_selected))

    assert report["segment_count"] == 2
    assert report["segments"][0]["kink_count"] == 0
    assert report["segments"][1]["kink_count"] == 1
    assert report["total_kink_count"] == 1
    # Segment 0 gets [0.0, 0.5, 1.0] (3 points; the shared t=1.0 anchor is
    # nearer segment 0's midpoint). Segment 1 gets [1.3, 1.6, 2.4] (3
    # points). The out-of-range t=3.5 frame is excluded entirely.
    assert report["total_point_count"] == 6


def test_kink_report_does_not_flag_a_legitimate_direction_change_at_a_bounce_event() -> None:
    """A bounce is a selected event: the ball is allowed to reverse there.

    Segment 0 approaches the bounce moving in +x; segment 1 departs the same
    bounce moving in -x. Naively slicing frames with an inclusive [t0, t1] on
    both sides of the shared boundary time would double-count that single
    anchor frame and manufacture a false kink out of a legitimate,
    owner-sanctioned direction change. It must not.
    """

    frames = [
        {"t": 0.0, "world_xyz": [0.0, 0.0, 0.0]},
        {"t": 0.3, "world_xyz": [0.3, 0.0, 0.0]},
        {"t": 0.6, "world_xyz": [0.6, 0.0, 0.0]},  # bounce anchor: approach ends moving +x
        {"t": 0.9, "world_xyz": [0.3, 0.0, 0.0]},  # departs the bounce moving -x
        {"t": 1.2, "world_xyz": [0.0, 0.0, 0.0]},
    ]
    events_selected = {"selected": [{"t": 0.0}, {"t": 0.6}, {"t": 1.2}]}

    report = kink_report(frames, segment_time_bounds(events_selected))

    assert report["total_kink_count"] == 0


def test_measure_ball_trail_kinks_cli_writes_report(tmp_path: Path) -> None:
    ball_stream = _write_json(
        tmp_path / "ball_track_arc_solved.json",
        {
            "frames": [
                {"t": 0.0, "world_xyz": [0.0, 0.0, 0.0]},
                {"t": 1.0 / 30.0, "world_xyz": [0.1, 0.0, 0.0]},
                {"t": 2.0 / 30.0, "world_xyz": [0.2, 0.0, 0.0]},
            ]
        },
    )
    events_selected = _write_json(
        tmp_path / "events_selected.json",
        {"selected": [{"t": 0.0}, {"t": 2.0 / 30.0}]},
    )
    out = tmp_path / "kink_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/measure_ball_trail_kinks.py",
            "--ball-stream",
            str(ball_stream),
            "--events-selected",
            str(events_selected),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    emitted = json.loads(out.read_text(encoding="utf-8"))
    assert emitted["total_kink_count"] == 0
    assert emitted["segment_count"] == 1
    assert json.loads(completed.stdout) == emitted


def test_measure_ball_trail_kinks_cli_reports_error_for_missing_frames(tmp_path: Path) -> None:
    ball_stream = _write_json(tmp_path / "bad.json", {"not_frames": []})
    events_selected = _write_json(tmp_path / "events_selected.json", {"selected": []})

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/measure_ball_trail_kinks.py",
            "--ball-stream",
            str(ball_stream),
            "--events-selected",
            str(events_selected),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ERROR" in completed.stderr
