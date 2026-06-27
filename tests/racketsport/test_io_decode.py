from __future__ import annotations

import json
import subprocess

import pytest

from threed.racketsport.io_decode import FrameSource, probe_clip


def _make_tiny_clip(path):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=5:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:sample_rate=44100:duration=1",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg is not installed")


def test_probe_clip_returns_phase0_metadata(tmp_path):
    clip = tmp_path / "sample.mp4"
    _make_tiny_clip(clip)

    source = probe_clip(clip)

    assert isinstance(source, FrameSource)
    assert source.path == clip
    assert source.width == 64
    assert source.height == 48
    assert source.fps == pytest.approx(5.0)
    assert source.duration_s == pytest.approx(1.0, rel=0.2)
    assert source.frame_count >= 5
    assert source.audio_sample_rate == 44100
    json.dumps(source.to_frames_meta())


def test_probe_clip_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_clip(tmp_path / "missing.mp4")
