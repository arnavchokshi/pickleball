from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

from threed.racketsport.audio_onsets import build_audio_onsets_from_video, build_audio_onsets_from_wav


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
