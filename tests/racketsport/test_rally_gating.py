from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.rally_gating import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    audio_onset_intervals,
    ball_activity_intervals,
    build_rally_spans_artifact,
    build_rally_spans_artifact_from_paths,
    dead_time_fraction,
    derive_rally_spans,
    frame_schedule,
    in_rally_span,
    merge_intervals,
    missed_events,
    player_motion_intervals,
)


def _ball_frame(t: float, *, visible: bool) -> dict:
    return {"t": t, "xy": [100.0, 100.0], "conf": 0.9 if visible else 0.0, "visible": visible}


def _track_frame(t: float, *, world_xy: tuple[float, float] | None = None, bbox: tuple[float, float, float, float] | None = None) -> dict:
    frame: dict = {"t": t, "conf": 0.9}
    if world_xy is not None:
        frame["world_xy"] = list(world_xy)
    if bbox is not None:
        frame["bbox"] = list(bbox)
    return frame


# --- ball_activity_intervals -----------------------------------------------------


def test_ball_activity_intervals_merges_within_gap():
    frames = [_ball_frame(0.0, visible=True), _ball_frame(0.5, visible=True), _ball_frame(1.0, visible=True)]
    intervals = ball_activity_intervals(frames, gap_seconds=0.8)
    assert len(intervals) == 1
    assert intervals[0].t0 == pytest.approx(0.0)
    assert intervals[0].t1 == pytest.approx(1.0)


def test_ball_activity_intervals_splits_beyond_gap():
    frames = [_ball_frame(0.0, visible=True), _ball_frame(5.0, visible=True)]
    intervals = ball_activity_intervals(frames, gap_seconds=0.8)
    assert len(intervals) == 2


def test_ball_activity_intervals_ignores_invisible_frames():
    frames = [_ball_frame(0.0, visible=False), _ball_frame(1.0, visible=False)]
    assert ball_activity_intervals(frames) == []


# --- player_motion_intervals -----------------------------------------------------


def test_player_motion_intervals_world_xy_detects_fast_move():
    # 2 meters in 0.1s = 20 m/s, well above the default 0.35 m/s idle threshold.
    players = [
        {
            "id": 1,
            "frames": [
                _track_frame(0.0, world_xy=(0.0, 0.0)),
                _track_frame(0.1, world_xy=(2.0, 0.0)),
            ],
        }
    ]
    intervals = player_motion_intervals(players)
    assert len(intervals) == 1


def test_player_motion_intervals_idle_shuffle_stays_below_threshold():
    # 1 cm in 1s = 0.01 m/s, below the 0.35 m/s idle threshold.
    players = [
        {
            "id": 1,
            "frames": [
                _track_frame(0.0, world_xy=(0.0, 0.0)),
                _track_frame(1.0, world_xy=(0.01, 0.0)),
            ],
        }
    ]
    assert player_motion_intervals(players) == []


def test_player_motion_intervals_bbox_fallback_when_no_world_xy():
    players = [
        {
            "id": 1,
            "frames": [
                _track_frame(0.0, bbox=(0.0, 0.0, 100.0, 100.0)),
                _track_frame(0.1, bbox=(500.0, 0.0, 600.0, 100.0)),
            ],
        }
    ]
    intervals = player_motion_intervals(players, speed_threshold_px_s=40.0)
    assert len(intervals) == 1


def test_player_motion_intervals_multiple_players_union():
    players = [
        {"id": 1, "frames": [_track_frame(0.0, world_xy=(0.0, 0.0)), _track_frame(0.1, world_xy=(0.0, 0.0))]},
        {"id": 2, "frames": [_track_frame(5.0, world_xy=(0.0, 0.0)), _track_frame(5.1, world_xy=(2.0, 0.0))]},
    ]
    intervals = player_motion_intervals(players)
    assert len(intervals) == 1
    assert intervals[0].t0 == pytest.approx(5.0)


# --- audio_onset_intervals --------------------------------------------------------


def test_audio_onset_intervals_windows_around_each_onset():
    intervals = audio_onset_intervals([2.0], window_seconds=0.25)
    assert len(intervals) == 1
    assert intervals[0].t0 == pytest.approx(1.75)
    assert intervals[0].t1 == pytest.approx(2.25)


def test_audio_onset_intervals_merges_nearby_onsets():
    intervals = audio_onset_intervals([2.0, 2.3], window_seconds=0.25, merge_gap_seconds=0.4)
    assert len(intervals) == 1


def test_audio_onset_intervals_empty_input():
    assert audio_onset_intervals([]) == []


def test_audio_onset_intervals_clamps_negative_start():
    intervals = audio_onset_intervals([0.05], window_seconds=0.25)
    assert intervals[0].t0 == pytest.approx(0.0)


# --- merge_intervals ---------------------------------------------------------------


def test_merge_intervals_tracks_sources():
    from threed.racketsport.rally_gating import _Interval  # type: ignore[attr-defined]

    merged = merge_intervals(
        [_Interval(t0=0.0, t1=1.0, source="ball"), _Interval(t0=0.5, t1=1.5, source="player_motion")],
        gap_seconds=0.0,
    )
    assert len(merged) == 1
    assert merged[0]["sources"] == ["ball", "player_motion"]
    assert merged[0]["t1"] == pytest.approx(1.5)


# --- derive_rally_spans: fusion, padding, OR semantics ------------------------------


def test_derive_rally_spans_empty_signals_returns_empty():
    assert derive_rally_spans(duration_s=10.0) == []


def test_derive_rally_spans_pads_each_side():
    frames = [_ball_frame(4.0, visible=True), _ball_frame(4.1, visible=True)]
    spans = derive_rally_spans(ball_frames=frames, duration_s=10.0, pad_seconds=0.5, ball_gap_seconds=0.8)
    assert len(spans) == 1
    assert spans[0]["t0"] == pytest.approx(3.5)
    assert spans[0]["t1"] == pytest.approx(4.6)


def test_derive_rally_spans_clamps_to_clip_bounds():
    frames = [_ball_frame(0.0, visible=True)]
    spans = derive_rally_spans(ball_frames=frames, duration_s=10.0, pad_seconds=0.5)
    assert spans[0]["t0"] == pytest.approx(0.0)


def test_derive_rally_spans_or_semantics_ball_signal_alone_creates_span():
    """A ball-only signal (no player/audio data) must still produce a span --
    OR fusion, not AND, is required so any single cheap signal is enough."""

    frames = [_ball_frame(t, visible=True) for t in (2.0, 2.1, 2.2)]
    spans = derive_rally_spans(ball_frames=frames, duration_s=10.0)
    assert len(spans) == 1


def test_derive_rally_spans_or_semantics_player_signal_alone_creates_span():
    players = [
        {
            "id": 1,
            "frames": [_track_frame(2.0, world_xy=(0.0, 0.0)), _track_frame(2.1, world_xy=(2.0, 0.0))],
        }
    ]
    spans = derive_rally_spans(players=players, duration_s=10.0)
    assert len(spans) == 1


def test_derive_rally_spans_player_signal_recovers_ball_occlusion_gap():
    """If the ball track drops out mid-rally but players keep moving, the union
    of both signals must still bridge the gap into one continuous span."""

    ball_frames = [_ball_frame(0.0, visible=True), _ball_frame(9.5, visible=True)]
    players = [
        {
            "id": 1,
            "frames": [
                _track_frame(t, world_xy=(t * 0.5, 0.0)) for t in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            ],
        }
    ]
    spans = derive_rally_spans(
        ball_frames=ball_frames,
        players=players,
        duration_s=10.0,
        ball_gap_seconds=0.8,
        player_gap_seconds=1.5,
        merge_gap_seconds=1.0,
    )
    assert len(spans) == 1
    assert spans[0]["t0"] == pytest.approx(0.0)
    assert spans[0]["t1"] == pytest.approx(10.0)


def test_derive_rally_spans_raises_on_nonpositive_duration():
    with pytest.raises(ValueError):
        derive_rally_spans(duration_s=0.0)


def test_derive_rally_spans_two_separate_rallies_stay_separate():
    frames = [_ball_frame(1.0, visible=True), _ball_frame(1.1, visible=True), _ball_frame(8.0, visible=True), _ball_frame(8.1, visible=True)]
    spans = derive_rally_spans(ball_frames=frames, duration_s=10.0, pad_seconds=0.3, ball_gap_seconds=0.5, merge_gap_seconds=0.5)
    assert len(spans) == 2


# --- dead_time_fraction / in_rally_span / missed_events ------------------------------


def test_dead_time_fraction_no_spans_is_all_dead():
    assert dead_time_fraction([], 10.0) == 1.0


def test_dead_time_fraction_full_coverage_is_zero():
    spans = [{"t0": 0.0, "t1": 10.0}]
    assert dead_time_fraction(spans, 10.0) == pytest.approx(0.0)


def test_dead_time_fraction_partial_coverage():
    spans = [{"t0": 0.0, "t1": 2.0}, {"t0": 8.0, "t1": 10.0}]
    assert dead_time_fraction(spans, 10.0) == pytest.approx(0.6)


def test_in_rally_span_inclusive_bounds():
    spans = [{"t0": 1.0, "t1": 2.0}]
    assert in_rally_span(1.0, spans)
    assert in_rally_span(2.0, spans)
    assert in_rally_span(1.5, spans)
    assert not in_rally_span(2.1, spans)


def test_missed_events_flags_events_outside_spans():
    spans = [{"t0": 1.0, "t1": 2.0}]
    assert missed_events([1.5, 5.0], spans) == [5.0]
    assert missed_events([1.5, 1.9], spans) == []


# --- frame_schedule ------------------------------------------------------------------


def test_frame_schedule_selects_only_in_span_frames():
    spans = [{"t0": 0.0, "t1": 0.05}]
    scheduled = frame_schedule(spans, fps=10.0, frame_count=10)
    # frame times: 0.0, 0.1, 0.2, ... only frame 0 (t=0.0) is inside [0, 0.05]
    assert scheduled == [0]


def test_frame_schedule_uses_pts_frame_times_for_vfr_inputs():
    spans = [{"t0": 0.23, "t1": 0.25}]
    frame_times = {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_times",
        "provenance": "ffprobe_pts",
        "frames": [
            {"frame": 0, "pts_s": 0.00},
            {"frame": 1, "pts_s": 0.04},
            {"frame": 2, "pts_s": 0.24},
            {"frame": 3, "pts_s": 0.28},
        ],
    }

    assert frame_schedule(spans, fps=30.0, frame_count=4, frame_times=frame_times) == [2]
    assert frame_schedule(spans, fps=30.0, frame_count=4) == []


def test_frame_schedule_empty_spans_selects_nothing():
    assert frame_schedule([], fps=30.0, frame_count=100) == []


def test_frame_schedule_rejects_nonpositive_fps():
    with pytest.raises(ValueError):
        frame_schedule([{"t0": 0.0, "t1": 1.0}], fps=0.0, frame_count=10)


# --- artifact builders -----------------------------------------------------------------


def test_build_rally_spans_artifact_shape_and_provenance():
    ball_track = {"fps": 60.0, "frames": [_ball_frame(1.0, visible=True), _ball_frame(1.1, visible=True)]}
    artifact = build_rally_spans_artifact(clip_id="clip_a", duration_s=10.0, ball_track=ball_track)
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert artifact["clip_id"] == "clip_a"
    assert artifact["signals_used"] == ["ball_track"]
    assert artifact["not_ground_truth"] is True
    assert artifact["span_count"] == len(artifact["spans"])
    assert 0.0 <= artifact["dead_time_fraction"] <= 1.0


def test_build_rally_spans_artifact_no_signals_yields_empty_spans_full_dead_time():
    artifact = build_rally_spans_artifact(clip_id="clip_b", duration_s=10.0)
    assert artifact["spans"] == []
    assert artifact["signals_used"] == []
    assert artifact["dead_time_fraction"] == pytest.approx(1.0)


def test_build_rally_spans_artifact_from_paths_round_trips_through_disk(tmp_path: Path):
    ball_track_path = tmp_path / "ball_track.json"
    tracks_path = tmp_path / "tracks.json"
    audio_path = tmp_path / "audio_onsets.json"

    ball_track_path.write_text(json.dumps({"fps": 60.0, "frames": [_ball_frame(1.0, visible=True), _ball_frame(1.1, visible=True)]}))
    tracks_path.write_text(json.dumps({"fps": 60.0, "players": [{"id": 1, "frames": [_track_frame(6.0, world_xy=(0.0, 0.0)), _track_frame(6.1, world_xy=(1.0, 0.0))]}]}))
    audio_path.write_text(json.dumps({"onsets": [{"time_s": 8.0}]}))

    artifact = build_rally_spans_artifact_from_paths(
        clip_id="clip_c",
        duration_s=10.0,
        ball_track_path=ball_track_path,
        tracks_path=tracks_path,
        audio_onsets_path=audio_path,
    )
    assert set(artifact["signals_used"]) == {"ball_track", "player_motion", "audio_onsets"}
    assert artifact["signal_sources"]["ball_track_path"] == str(ball_track_path)
    # t~1 (ball) stays isolated; t~6 (player) and t~8 (audio) pad-overlap within the
    # default 1.0s merge gap and combine into one multi-source span.
    assert len(artifact["spans"]) == 2
    sources_by_span = [set(span["sources"]) for span in artifact["spans"]]
    assert {"ball"} in sources_by_span
    assert {"audio", "player_motion"} in sources_by_span


def test_build_rally_spans_artifact_from_paths_skips_missing_signals():
    artifact = build_rally_spans_artifact_from_paths(clip_id="clip_d", duration_s=5.0)
    assert artifact["spans"] == []
    assert artifact["signal_sources"] == {
        "ball_track_path": None,
        "tracks_path": None,
        "audio_onsets_path": None,
    }
