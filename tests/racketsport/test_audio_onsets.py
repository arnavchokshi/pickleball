from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from threed.racketsport.audio_onsets import (
    build_audio_onsets_from_samples,
    build_audio_onsets_from_video,
    build_audio_onsets_from_wav,
    finalize_audio_onset_timing,
)


def _write_pcm16_wav(path: Path, samples: list[float], *, sample_rate_hz: int = 1_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(
            b"".join(struct.pack("<h", max(-32767, min(32767, int(sample * 32767)))) for sample in samples)
        )


def test_audio_onset_builder_detects_impulse_from_wav(tmp_path: Path) -> None:
    samples = [0.0] * 2_000
    for index in range(1_000, 1_025):
        samples[index] = 0.92
    wav_path = tmp_path / "contact.wav"
    _write_pcm16_wav(wav_path, samples)

    payload = build_audio_onsets_from_wav(wav_path, threshold_score=0.45, min_separation_s=0.2)

    assert payload["artifact_type"] == "racketsport_audio_onsets"
    assert payload["status"] == "review_only"
    assert payload["not_gate_verified"] is True
    assert payload["trusted_for_contact"] is False
    assert payload["summary"]["onset_count"] >= 1
    assert payload["onsets"][0]["time_s"] == 1.0
    assert payload["onsets"][0]["score"] > 0.45
    assert payload["onsets"][0]["source"] == "audio_energy_onset"


def test_audio_onset_video_builder_writes_no_audio_stream_blocker(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "silent.mp4"
    video_path.write_bytes(b"not a real mp4 because ffprobe is monkeypatched")

    from threed.racketsport import audio_onsets

    monkeypatch.setattr(audio_onsets, "_ffprobe_audio_stream", lambda path: None)

    payload = build_audio_onsets_from_video(video_path)

    assert payload["status"] == "blocked"
    assert payload["summary"]["onset_count"] == 0
    assert payload["blockers"] == ["no_audio_stream"]
    assert payload["warnings"] == ["audio_stream_missing"]
    assert payload["onsets"] == []


def test_build_audio_onsets_cli_writes_artifact_from_wav(tmp_path: Path) -> None:
    samples = [0.0] * 1_500
    for index in range(500, 530):
        samples[index] = 0.85
    wav_path = tmp_path / "contact.wav"
    out_path = tmp_path / "audio_onsets.json"
    _write_pcm16_wav(wav_path, samples)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_audio_onsets.py",
            "--input",
            str(wav_path),
            "--out",
            str(out_path),
            "--threshold-score",
            "0.45",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "wrote" in completed.stdout
    assert payload["artifact_type"] == "racketsport_audio_onsets"
    assert payload["summary"]["onset_count"] >= 1


def test_build_audio_onsets_cli_accepts_review_metadata_options(tmp_path: Path) -> None:
    samples = [0.0] * 1_500
    for index in range(500, 530):
        samples[index] = 0.85
    wav_path = tmp_path / "contact.wav"
    out_path = tmp_path / "audio_onsets.json"
    _write_pcm16_wav(wav_path, samples)

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_audio_onsets.py",
            "--input",
            str(wav_path),
            "--out",
            str(out_path),
            "--clip",
            "clip_001",
            "--frame-rate",
            "30",
            "--analysis-sample-rate-hz",
            "16000",
            "--threshold-score",
            "0.45",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["clip"] == "clip_001"
    assert payload["frame_rate"] == 30.0
    assert payload["summary"]["analysis_sample_rate_hz"] == 16000
    assert payload["onsets"][0]["nearest_frame"] == 15


def test_audio_timing_preserves_raw_chronology_and_reorders_by_corrected_time() -> None:
    ordered, timing = finalize_audio_onset_timing(
        [
            {
                "raw_time_s": 1.000,
                "score": 0.8,
                "class_label": "contact",
                "source_to_microphone_distance_m": 0.0,
            },
            {
                "raw_time_s": 1.010,
                "score": 0.9,
                "class_label": "bounce",
                "source_to_microphone_distance_m": 34.3,
            },
        ],
        speed_of_sound_mps=343.0,
        distance_uncertainty_m=0.343,
    )

    assert [item["class_label"] for item in ordered] == ["bounce", "contact"]
    assert [item["raw_order"] for item in ordered] == [1, 0]
    assert [item["corrected_order"] for item in ordered] == [0, 1]
    assert [item["onset_order"] for item in ordered] == [0, 1]
    assert ordered[0]["raw_time_s"] == pytest.approx(1.010)
    assert ordered[0]["corrected_time_s"] == pytest.approx(0.910)
    assert ordered[0]["time_s"] == pytest.approx(0.910)
    assert ordered[0]["timing_provenance"]["method"] == "acoustic_distance_over_declared_speed"
    assert ordered[0]["timing_provenance"]["uncertainty_s"] == pytest.approx(0.001)
    assert timing["correction_model"] == "AcousticPropagationModel"
    assert timing["corrected_order_differs_from_raw_order_count"] == 2


def test_audio_timing_uses_explicit_identity_when_distance_is_unavailable() -> None:
    payload = build_audio_onsets_from_samples(
        [0.0] * 100 + [0.9] * 20 + [0.0] * 300,
        sample_rate_hz=1_000,
        source="synthetic",
        source_path="synthetic.wav",
        threshold_score=0.1,
        frame_size_s=0.02,
        hop_s=0.005,
    )

    assert payload["onsets"]
    assert payload["timing"]["distance_policy"] == "identity_when_source_distance_unavailable"
    assert payload["timing"]["propagation_corrected_onset_count"] == 0
    assert all(item["raw_time_s"] == item["corrected_time_s"] == item["time_s"] for item in payload["onsets"])
    assert all(item["timing_provenance"]["applied"] is False for item in payload["onsets"])
