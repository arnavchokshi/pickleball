from __future__ import annotations

import pytest

from threed.racketsport import audio_pop


def test_sound_travel_delay_uses_meters_and_seconds_with_default_speed():
    assert audio_pop.sound_travel_delay_seconds(34.3) == pytest.approx(0.1)
    assert audio_pop.sound_travel_delay_seconds(10.0, speed_of_sound_mps=400.0) == pytest.approx(0.025)


def test_sound_travel_delay_rejects_invalid_distances_and_speeds():
    with pytest.raises(ValueError, match="distance_m"):
        audio_pop.sound_travel_delay_seconds(-0.1)

    with pytest.raises(ValueError, match="speed_of_sound_mps"):
        audio_pop.sound_travel_delay_seconds(1.0, speed_of_sound_mps=0.0)


def test_correct_audio_onset_shifts_observed_audio_back_to_court_time():
    assert audio_pop.correct_audio_onset_to_court_time(12.5, distance_m=17.15) == pytest.approx(12.45)

    with pytest.raises(ValueError, match="observed_time_s"):
        audio_pop.correct_audio_onset_to_court_time(-0.01, distance_m=1.0)


def test_onset_candidate_validates_cpu_only_candidate_fields():
    candidate = audio_pop.OnsetCandidate(time_s=3.25, score=0.82, source="energy_peak")

    assert candidate.time_s == pytest.approx(3.25)
    assert candidate.score == pytest.approx(0.82)
    assert candidate.source == "energy_peak"

    with pytest.raises(ValueError, match="score"):
        audio_pop.OnsetCandidate(time_s=1.0, score=1.01)

    with pytest.raises(ValueError, match="time_s"):
        audio_pop.OnsetCandidate(time_s=float("nan"), score=0.5)


def test_mel_window_bounds_returns_clamped_half_open_frame_range():
    bounds = audio_pop.mel_window_bounds(
        contact_time_s=1.0,
        sample_rate_hz=16_000,
        hop_length=160,
        pre_s=0.03,
        post_s=0.05,
        total_frames=120,
    )

    assert bounds == (97, 106)
    assert audio_pop.mel_window_bounds(0.01, sample_rate_hz=16_000, hop_length=160, pre_s=0.05, post_s=0.02) == (0, 4)


def test_fuse_audio_onsets_to_court_time_preserves_order_and_scores():
    raw = [
        audio_pop.OnsetCandidate(time_s=2.0, score=0.9, source="raw_pop"),
        audio_pop.OnsetCandidate(time_s=2.2, score=0.7, source="raw_pop"),
    ]

    fused = audio_pop.fuse_audio_onsets_to_court_time(raw, player_camera_distance_m=34.3)

    assert [candidate.time_s for candidate in fused] == pytest.approx([1.9, 2.1])
    assert [candidate.score for candidate in fused] == pytest.approx([0.9, 0.7])
    assert [candidate.source for candidate in fused] == ["raw_pop:court_time", "raw_pop:court_time"]
    assert [candidate.raw_time_s for candidate in fused] == pytest.approx([2.0, 2.2])
