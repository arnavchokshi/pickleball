from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from threed.racketsport.audio_onsets_v2 import build_audio_onsets_v2_from_samples


def _synthetic_pop(
    *,
    sample_rate_hz: int,
    duration_s: float,
    onset_s: float,
    frequency_hz: float = 3_200.0,
) -> list[float]:
    rng = np.random.default_rng(1234)
    samples = rng.normal(0.0, 0.003, int(round(duration_s * sample_rate_hz)))
    start = int(round(onset_s * sample_rate_hz))
    pop_length = int(round(0.006 * sample_rate_hz))
    for offset in range(pop_length):
        index = start + offset
        if index >= len(samples):
            break
        envelope = math.exp(-offset / (0.0018 * sample_rate_hz))
        samples[index] += 0.42 * envelope * math.sin(2.0 * math.pi * frequency_hz * offset / sample_rate_hz)
    return samples.tolist()


def test_audio_onsets_v2_detects_high_frequency_pop_in_noise() -> None:
    sample_rate_hz = 24_000
    onset_s = 0.713
    samples = _synthetic_pop(sample_rate_hz=sample_rate_hz, duration_s=1.4, onset_s=onset_s)

    payload = build_audio_onsets_v2_from_samples(
        samples,
        sample_rate_hz=sample_rate_hz,
        source="synthetic_pop",
        source_path=Path("synthetic.wav"),
        threshold_mad=4.0,
        min_separation_s=0.12,
    )

    assert payload["artifact_type"] == "racketsport_audio_onsets"
    assert payload["status"] == "review_only"
    assert payload["detector_version"] == "audio_onset_pop_v2"
    assert payload["not_gate_verified"] is True
    assert payload["trusted_for_contact"] is False
    assert payload["summary"]["onset_count"] >= 1

    nearest = min(payload["onsets"], key=lambda item: abs(float(item["time_s"]) - onset_s))
    assert abs(float(nearest["time_s"]) - onset_s) <= 0.006
    assert nearest["source"] == "audio_pop_v2"
    assert nearest["score"] > 0.0
    assert nearest["features"]["spectral_flux"] > 0.0
    assert nearest["features"]["high_frequency_content"] > 0.0


def test_audio_onsets_v2_prefers_pop_band_over_low_frequency_thud() -> None:
    sample_rate_hz = 24_000
    low_thud_s = 0.35
    pop_s = 0.85
    samples = np.asarray(
        _synthetic_pop(sample_rate_hz=sample_rate_hz, duration_s=1.2, onset_s=pop_s),
        dtype=float,
    )
    start = int(round(low_thud_s * sample_rate_hz))
    thud_length = int(round(0.035 * sample_rate_hz))
    for offset in range(thud_length):
        index = start + offset
        if index >= len(samples):
            break
        envelope = math.exp(-offset / (0.014 * sample_rate_hz))
        samples[index] += 0.55 * envelope * math.sin(2.0 * math.pi * 220.0 * offset / sample_rate_hz)

    payload = build_audio_onsets_v2_from_samples(
        samples.tolist(),
        sample_rate_hz=sample_rate_hz,
        source="synthetic_pop_with_thud",
        source_path=Path("synthetic.wav"),
        threshold_mad=4.0,
        min_separation_s=0.12,
    )

    assert payload["summary"]["onset_count"] >= 1
    nearest_pop = min(payload["onsets"], key=lambda item: abs(float(item["time_s"]) - pop_s))
    assert abs(float(nearest_pop["time_s"]) - pop_s) <= 0.006
    assert all(abs(float(item["time_s"]) - low_thud_s) > 0.040 for item in payload["onsets"])


def test_run_build_audio_onsets_v2_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_audio_onsets_v2.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--input" in completed.stdout


def test_run_build_audio_onsets_v2_cli_fails_closed_on_missing_input(tmp_path: Path) -> None:
    out_path = tmp_path / "audio_onsets_v2.json"
    missing_input = tmp_path / "does_not_exist.wav"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_audio_onsets_v2.py",
            "--input",
            str(missing_input),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not out_path.exists()
