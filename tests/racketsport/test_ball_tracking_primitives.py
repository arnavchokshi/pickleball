from __future__ import annotations

import pytest

from threed.racketsport import ball_tap_track, ball_tracknet


def _helpers():
    ball_frame = getattr(ball_tracknet, "ball_frame", None)
    validate_ball_frame = getattr(ball_tracknet, "validate_ball_frame", None)
    interpolate_short_gaps = getattr(ball_tracknet, "interpolate_short_gaps", None)
    apply_tap_track_corrections = getattr(ball_tap_track, "apply_tap_track_corrections", None)
    assert callable(ball_frame)
    assert callable(validate_ball_frame)
    assert callable(interpolate_short_gaps)
    assert callable(apply_tap_track_corrections)
    return ball_frame, validate_ball_frame, interpolate_short_gaps, apply_tap_track_corrections


def test_ball_frame_returns_schema_friendly_dictionary_with_validated_visibility() -> None:
    ball_frame, validate_ball_frame, _, _ = _helpers()

    frame = ball_frame(t=0.25, xy=[320, 240], conf=0.84, visible=True, world_xyz=[1, 2, 3])

    assert frame == {
        "t": 0.25,
        "xy": [320.0, 240.0],
        "conf": 0.84,
        "visible": True,
        "world_xyz": [1.0, 2.0, 3.0],
        "approx": False,
    }
    assert validate_ball_frame(frame, min_visible_conf=0.5) == frame

    with pytest.raises(ValueError, match="visible frames require conf"):
        ball_frame(t=0.0, xy=[1.0, 2.0], conf=0.49, visible=True, min_visible_conf=0.5)

    with pytest.raises(ValueError, match="xy"):
        ball_frame(t=0.0, xy=[1.0], conf=0.9, visible=True)


def test_interpolate_short_gaps_fills_only_short_invisible_runs_between_visible_samples() -> None:
    ball_frame, _, interpolate_short_gaps, _ = _helpers()

    frames = [
        ball_frame(t=0.0, xy=[0.0, 0.0], conf=0.9, visible=True),
        ball_frame(t=0.1, xy=[0.0, 0.0], conf=0.0, visible=False),
        ball_frame(t=0.2, xy=[10.0, 20.0], conf=0.8, visible=True),
        ball_frame(t=0.3, xy=[0.0, 0.0], conf=0.0, visible=False),
        ball_frame(t=0.4, xy=[0.0, 0.0], conf=0.0, visible=False),
        ball_frame(t=0.5, xy=[0.0, 0.0], conf=0.0, visible=False),
        ball_frame(t=0.6, xy=[40.0, 80.0], conf=0.7, visible=True),
    ]

    filled = interpolate_short_gaps(frames, max_gap_frames=1)

    assert filled[1]["visible"] is True
    assert filled[1]["approx"] is True
    assert filled[1]["xy"] == pytest.approx([5.0, 10.0])
    assert filled[1]["conf"] == pytest.approx(0.8)
    assert filled[3]["visible"] is False
    assert filled[4]["visible"] is False
    assert filled[5]["visible"] is False
    assert frames[1]["visible"] is False


def test_interpolate_short_gaps_interpolates_world_xyz_when_both_anchors_have_world_positions() -> None:
    ball_frame, _, interpolate_short_gaps, _ = _helpers()

    frames = [
        ball_frame(t=0.0, xy=[0.0, 0.0], conf=0.9, visible=True, world_xyz=[0.0, 0.0, 1.0]),
        ball_frame(t=0.25, xy=[0.0, 0.0], conf=0.0, visible=False),
        ball_frame(t=0.5, xy=[10.0, 20.0], conf=0.8, visible=True, world_xyz=[2.0, 4.0, 3.0]),
    ]

    filled = interpolate_short_gaps(frames, max_gap_frames=1)

    assert filled[1]["xy"] == pytest.approx([5.0, 10.0])
    assert filled[1]["world_xyz"] == pytest.approx([1.0, 2.0, 2.0])


def test_apply_tap_track_corrections_matches_by_frame_index_or_safe_time_tolerance() -> None:
    ball_frame, _, _, apply_tap_track_corrections = _helpers()

    frames = [
        ball_frame(t=0.00, xy=[0.0, 0.0], conf=0.7, visible=True),
        ball_frame(t=0.04, xy=[4.0, 0.0], conf=0.7, visible=True),
        ball_frame(t=0.08, xy=[8.0, 0.0], conf=0.7, visible=True),
    ]

    corrected = apply_tap_track_corrections(
        frames,
        [
            {"frame_index": 1, "t": 0.041, "xy": [40.0, 10.0], "conf": 1.0},
            {"t": 0.079, "xy": [80.0, 20.0], "visible": True},
        ],
        time_tolerance_s=0.003,
    )

    assert corrected[0] == frames[0]
    assert corrected[1]["xy"] == pytest.approx([40.0, 10.0])
    assert corrected[1]["conf"] == pytest.approx(1.0)
    assert corrected[1]["visible"] is True
    assert corrected[1]["approx"] is False
    assert corrected[2]["xy"] == pytest.approx([80.0, 20.0])
    assert corrected[2]["conf"] == pytest.approx(1.0)
    assert frames[1]["xy"] == [4.0, 0.0]


def test_apply_tap_track_corrections_rejects_unsafe_time_or_index_matches() -> None:
    ball_frame, _, _, apply_tap_track_corrections = _helpers()

    frames = [
        ball_frame(t=0.00, xy=[0.0, 0.0], conf=0.7, visible=True),
        ball_frame(t=0.04, xy=[4.0, 0.0], conf=0.7, visible=True),
    ]

    with pytest.raises(ValueError, match="frame_index/t mismatch"):
        apply_tap_track_corrections(
            frames,
            [{"frame_index": 1, "t": 0.00, "xy": [9.0, 9.0]}],
            time_tolerance_s=0.005,
        )

    with pytest.raises(ValueError, match="no frame within"):
        apply_tap_track_corrections(
            frames,
            [{"t": 0.02, "xy": [9.0, 9.0]}],
            time_tolerance_s=0.005,
        )
