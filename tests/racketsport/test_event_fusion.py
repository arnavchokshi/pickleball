from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport import event_fusion
from threed.racketsport.schemas import ContactWindows


def _fusion_api():
    fuse_contact_windows = getattr(event_fusion, "fuse_contact_windows", None)
    wrist_velocity_peak = getattr(event_fusion, "WristVelocityPeak", None)
    ball_inflection_candidate = getattr(event_fusion, "BallInflectionCandidate", None)
    assert callable(fuse_contact_windows)
    assert callable(wrist_velocity_peak)
    assert callable(ball_inflection_candidate)
    return fuse_contact_windows, wrist_velocity_peak, ball_inflection_candidate


def test_fuse_contact_windows_requires_audio_wrist_and_ball_sources_for_schema_event() -> None:
    fuse_contact_windows, WristVelocityPeak, BallInflectionCandidate = _fusion_api()

    fused = fuse_contact_windows(
        fps=120.0,
        audio_onsets=[{"time_s": 1.000, "score": 0.90}],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=1.015,
                player_id=7,
                wrist_world_xyz=(1.05, 0.05, 0.90),
                speed_mps=7.5,
                confidence=0.80,
            )
        ],
        ball_inflections=[
            BallInflectionCandidate(
                time_s=0.990,
                ball_world_xyz=(1.00, 0.00, 0.88),
                confidence=0.70,
            )
        ],
        pre_s=0.040,
        post_s=0.060,
    )

    assert fused == {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": pytest.approx(1.0025),
                "frame": 120,
                "player_id": 7,
                "confidence": pytest.approx(0.80),
                "sources": {"audio": 0.90, "wrist_vel": 0.80, "ball_inflection": 0.70},
                "window": {"t0": pytest.approx(0.9625), "t1": pytest.approx(1.0625), "importance": 0.80},
            }
        ],
    }
    ContactWindows.model_validate(fused)


def test_fuse_contact_windows_attributes_player_by_nearest_wrist_to_ball_position() -> None:
    fuse_contact_windows, WristVelocityPeak, BallInflectionCandidate = _fusion_api()

    fused = fuse_contact_windows(
        fps=60.0,
        audio_onsets=[{"time_s": 2.000, "score": 0.85}],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=2.010,
                player_id=1,
                wrist_world_xyz=(0.0, 0.0, 1.0),
                speed_mps=5.0,
                confidence=0.90,
            ),
            WristVelocityPeak(
                time_s=1.990,
                player_id=2,
                wrist_world_xyz=(3.02, 0.02, 1.1),
                speed_mps=6.0,
                confidence=0.70,
            ),
        ],
        ball_inflections=[
            BallInflectionCandidate(
                time_s=2.005,
                ball_world_xyz=(3.0, 0.0, 1.08),
                confidence=0.95,
            )
        ],
        max_time_delta_s=0.030,
    )

    event = fused["events"][0]

    assert event["player_id"] == 2
    assert event["sources"] == {"audio": 0.85, "wrist_vel": 0.70, "ball_inflection": 0.95}


def test_fuse_contact_windows_omits_events_when_required_sources_are_missing() -> None:
    fuse_contact_windows, WristVelocityPeak, BallInflectionCandidate = _fusion_api()

    assert fuse_contact_windows(
        fps=120.0,
        audio_onsets=[{"time_s": 1.0, "score": 0.9}],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=1.0,
                player_id=1,
                wrist_world_xyz=(0.0, 0.0, 1.0),
                speed_mps=5.0,
                confidence=0.8,
            )
        ],
        ball_inflections=[],
    ) == {"schema_version": 1, "events": []}

    assert fuse_contact_windows(
        fps=120.0,
        audio_onsets=[],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=1.0,
                player_id=1,
                wrist_world_xyz=(0.0, 0.0, 1.0),
                speed_mps=5.0,
                confidence=0.8,
            )
        ],
        ball_inflections=[
            BallInflectionCandidate(
                time_s=1.0,
                ball_world_xyz=(0.0, 0.0, 1.0),
                confidence=0.7,
            )
        ],
    ) == {"schema_version": 1, "events": []}


def test_fuse_contact_windows_can_opt_into_wrist_ball_fusion_without_audio() -> None:
    fuse_contact_windows, WristVelocityPeak, BallInflectionCandidate = _fusion_api()

    fused = fuse_contact_windows(
        fps=120.0,
        audio_onsets=[],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=1.010,
                player_id=7,
                wrist_world_xyz=(1.02, 0.02, 0.82),
                speed_mps=8.5,
                confidence=0.82,
            )
        ],
        ball_inflections=[
            BallInflectionCandidate(
                time_s=0.990,
                ball_world_xyz=(1.00, 0.00, 0.80),
                confidence=0.74,
            )
        ],
        require_audio=False,
        max_time_delta_s=0.030,
        pre_s=0.040,
        post_s=0.060,
    )

    assert fused == {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": pytest.approx(1.0),
                "frame": 120,
                "player_id": 7,
                "confidence": pytest.approx(0.78),
                "sources": {"wrist_vel": 0.82, "ball_inflection": 0.74},
                "window": {"t0": pytest.approx(0.96), "t1": pytest.approx(1.06), "importance": pytest.approx(0.78)},
            }
        ],
    }
    ContactWindows.model_validate(fused)


def test_fuse_contact_windows_marks_low_trust_ball_inflection_cues() -> None:
    fuse_contact_windows, WristVelocityPeak, BallInflectionCandidate = _fusion_api()

    fused = fuse_contact_windows(
        fps=120.0,
        audio_onsets=[],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=1.010,
                player_id=7,
                wrist_world_xyz=(1.02, 0.02, 0.82),
                speed_mps=8.5,
                confidence=0.95,
            )
        ],
        ball_inflections=[
            BallInflectionCandidate(
                time_s=0.990,
                ball_world_xyz=(1.00, 0.00, 0.80),
                confidence=0.05,
            )
        ],
        require_audio=False,
        max_time_delta_s=0.030,
    )

    event = fused["events"][0]
    assert event["confidence"] == pytest.approx(0.5)
    assert event["trust_band_note"] == "low-trust ball-inflection cue, unverified"
    ContactWindows.model_validate(fused)


def test_fuse_contact_windows_accepts_image_only_ball_inflections_as_low_trust() -> None:
    payload = event_fusion.fuse_contact_windows_from_cue_payloads(
        fps=120.0,
        audio_onsets_payload=[],
        wrist_velocity_peaks_payload={
            "peaks": [
                {
                    "time_s": 1.010,
                    "player_id": 7,
                    "wrist_world_xyz": [1.02, 0.02, 0.82],
                    "speed_mps": 8.5,
                    "confidence": 0.95,
                }
            ]
        },
        ball_inflections_payload={
            "candidates": [
                {
                    "time_s": 0.990,
                    "ball_world_xyz": None,
                    "confidence": 0.80,
                }
            ]
        },
        require_audio=False,
        max_time_delta_s=0.030,
    )

    event = payload["events"][0]
    assert event["sources"] == {"wrist_vel": 0.95, "ball_inflection": 0.8}
    assert event["trust_band_note"] == "image-space ball-inflection cue, unverified"
    ContactWindows.model_validate(payload)


def test_fuse_contact_windows_requires_flag_for_wrist_only_hints() -> None:
    fuse_contact_windows, WristVelocityPeak, _BallInflectionCandidate = _fusion_api()

    strict = fuse_contact_windows(
        fps=30.0,
        audio_onsets=[],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=2.0,
                player_id=7,
                wrist_world_xyz=(1.0, 0.0, 0.9),
                speed_mps=8.5,
                confidence=0.82,
            )
        ],
        ball_inflections=[],
    )

    assert strict == {"schema_version": 1, "events": []}

    hinted = fuse_contact_windows(
        fps=30.0,
        audio_onsets=[],
        wrist_velocity_peaks=[
            WristVelocityPeak(
                time_s=2.0,
                player_id=7,
                wrist_world_xyz=(1.0, 0.0, 0.9),
                speed_mps=8.5,
                confidence=0.82,
            )
        ],
        ball_inflections=[],
        allow_wrist_only_contact_hints=True,
    )

    assert hinted == {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 2.0,
                "frame": 60,
                "player_id": 7,
                "confidence": pytest.approx(0.35),
                "sources": {"wrist_vel": 0.82, "ball_inflection": 0.0},
                "window": {"t0": pytest.approx(1.88), "t1": pytest.approx(2.18), "importance": pytest.approx(0.35)},
                "trust_band_note": "wrist-cue-only, unverified",
            }
        ],
    }
    ContactWindows.model_validate(hinted)


def test_build_contact_windows_from_cues_cli_writes_schema_contact_windows(tmp_path: Path) -> None:
    audio = tmp_path / "audio_onsets.json"
    wrist = tmp_path / "wrist_velocity_peaks.json"
    ball = tmp_path / "ball_inflections.json"
    tracks = tmp_path / "tracks.json"
    out = tmp_path / "contact_windows.json"

    audio.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_audio_onsets",
                "status": "review_only",
                "onsets": [{"time_s": 1.000, "score": 0.90}],
            }
        ),
        encoding="utf-8",
    )
    wrist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_wrist_velocity_peaks",
                "status": "review_only",
                "peaks": [
                    {
                        "time_s": 1.010,
                        "player_id": 4,
                        "wrist_world_xyz": [1.05, 0.0, 0.75],
                        "speed_mps": 8.0,
                        "confidence": 0.80,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ball.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_inflections",
                "candidates": [
                    {
                        "time_s": 0.995,
                        "ball_world_xyz": [1.0, 0.0, 0.78],
                        "confidence": 0.70,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tracks.write_text(json.dumps({"schema_version": 1, "fps": 120.0, "players": []}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_windows_from_cues.py",
            "--audio-onsets",
            str(audio),
            "--wrist-velocity-peaks",
            str(wrist),
            "--ball-inflections",
            str(ball),
            "--tracks",
            str(tracks),
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    ContactWindows.model_validate(payload)
    assert payload["events"][0]["player_id"] == 4
    assert payload["events"][0]["frame"] == 120
    assert json.loads(completed.stdout)["event_count"] == 1


def test_build_contact_windows_from_cues_cli_supports_wrist_ball_mode_without_audio(tmp_path: Path) -> None:
    wrist = tmp_path / "wrist_velocity_peaks.json"
    ball = tmp_path / "ball_inflections.json"
    out = tmp_path / "contact_windows.json"

    wrist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_wrist_velocity_peaks",
                "status": "review_only",
                "peaks": [
                    {
                        "time_s": 1.010,
                        "player_id": 4,
                        "wrist_world_xyz": [1.05, 0.0, 0.75],
                        "speed_mps": 8.0,
                        "confidence": 0.80,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ball.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_inflections",
                "candidates": [
                    {
                        "time_s": 0.995,
                        "ball_world_xyz": [1.0, 0.0, 0.78],
                        "confidence": 0.70,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_windows_from_cues.py",
            "--contact-fusion-mode",
            "wrist_ball",
            "--wrist-velocity-peaks",
            str(wrist),
            "--ball-inflections",
            str(ball),
            "--fps",
            "120",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    ContactWindows.model_validate(payload)
    assert payload["events"][0]["player_id"] == 4
    assert payload["events"][0]["sources"] == {"wrist_vel": 0.8, "ball_inflection": 0.7}


def test_build_contact_windows_from_cues_cli_defaults_to_strict_audio_wrist_ball(tmp_path: Path) -> None:
    wrist = tmp_path / "wrist_velocity_peaks.json"
    ball = tmp_path / "ball_inflections.json"
    out = tmp_path / "contact_windows.json"

    wrist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_wrist_velocity_peaks",
                "status": "review_only",
                "peaks": [
                    {
                        "time_s": 1.010,
                        "player_id": 4,
                        "wrist_world_xyz": [1.05, 0.0, 0.75],
                        "speed_mps": 8.0,
                        "confidence": 0.80,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ball.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_inflections",
                "candidates": [
                    {
                        "time_s": 0.995,
                        "ball_world_xyz": [1.0, 0.0, 0.78],
                        "confidence": 0.70,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_windows_from_cues.py",
            "--wrist-velocity-peaks",
            str(wrist),
            "--ball-inflections",
            str(ball),
            "--fps",
            "120",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    ContactWindows.model_validate(payload)
    assert payload["events"] == []


def test_build_contact_windows_from_cues_cli_writes_wrist_only_hints_with_explicit_flag(tmp_path: Path) -> None:
    wrist = tmp_path / "wrist_velocity_peaks.json"
    out = tmp_path / "contact_windows.json"

    wrist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_wrist_velocity_peaks",
                "status": "review_only",
                "peaks": [
                    {
                        "time_s": 1.010,
                        "player_id": 4,
                        "wrist_world_xyz": [1.05, 0.0, 0.75],
                        "speed_mps": 8.0,
                        "confidence": 0.80,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_windows_from_cues.py",
            "--allow-wrist-only-contact-hints",
            "--wrist-velocity-peaks",
            str(wrist),
            "--fps",
            "120",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    ContactWindows.model_validate(payload)
    assert payload["events"][0]["sources"] == {"wrist_vel": 0.8, "ball_inflection": 0.0}
    assert payload["events"][0]["confidence"] <= 0.35
    assert payload["events"][0]["trust_band_note"] == "wrist-cue-only, unverified"
