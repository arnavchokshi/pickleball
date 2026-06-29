from __future__ import annotations

import json
import subprocess

import pytest

from scripts.racketsport.ingest_testclips import ingest_testclips
from threed.racketsport.io_decode import (
    ClipQualityProbe,
    FrameSource,
    _laplacian_variance,
    analyze_clip_qc,
    decode_clip,
    measure_decode_throughput,
    probe_clip,
)


def _make_tiny_clip(path, *, rate: int = 5, duration_s: float = 1.0):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate={rate}:duration={duration_s}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:sample_rate=44100:duration={duration_s}",
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


def test_decode_clip_alias_matches_probe_clip_metadata(tmp_path):
    clip = tmp_path / "sample.mp4"
    _make_tiny_clip(clip)

    assert decode_clip(clip, fps_out=30.0) == probe_clip(clip, fps_out=30.0)


def test_probe_clip_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_clip(tmp_path / "missing.mp4")


def test_laplacian_variance_flags_spatial_detail():
    flat = bytes([128] * 25)
    edge = bytes(
        [
            0,
            0,
            0,
            255,
            255,
            0,
            0,
            0,
            255,
            255,
            0,
            0,
            0,
            255,
            255,
            0,
            0,
            0,
            255,
            255,
            0,
            0,
            0,
            255,
            255,
        ]
    )

    assert _laplacian_variance(flat, 5, 5) == 0.0
    assert _laplacian_variance(edge, 5, 5) > 0.0


def test_analyze_clip_qc_returns_capture_quality(tmp_path):
    clip = tmp_path / "sample.mp4"
    _make_tiny_clip(clip, rate=60)

    qc = analyze_clip_qc(clip, sample_fps=2.0, max_frames=2, max_width=32)

    assert isinstance(qc, ClipQualityProbe)
    assert 1 <= qc.sampled_frames <= 2
    assert qc.sample_width == 32
    assert qc.sample_height == 24
    assert qc.qc_decode_fps > 0.0
    assert qc.blur_laplacian_var >= 0.0
    assert 0.0 <= qc.luminance_mean <= 1.0
    assert qc.luminance_std_fraction >= 0.0
    assert qc.capture_quality.grade in {"good", "warn", "poor"}
    json.dumps(qc.to_dict())


def test_ingest_testclips_writes_qc_and_capture_quality(tmp_path):
    root = tmp_path / "clips"
    out = tmp_path / "runs"
    clip = root / "pickleball" / "sample.mp4"
    clip.parent.mkdir(parents=True)
    _make_tiny_clip(clip, rate=60)

    written = ingest_testclips(root, out, qc_sample_fps=1.0, qc_max_frames=1, qc_max_width=32)

    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["source_relpath"] == "pickleball/sample.mp4"
    assert payload["capture_quality"]["grade"] in {"good", "warn", "poor"}
    assert payload["clip_qc"]["sample_resolution"] == [32, 24]
    assert payload["clip_qc"]["sampled_frames"] == 1


def test_extract_label_frames_cli_writes_manual_review_pack(tmp_path):
    root = tmp_path / "testclips"
    out = tmp_path / "label_frames"
    for clip_name in ("candidate", "second_candidate"):
        clip_dir = root / clip_name
        clip_dir.mkdir(parents=True)
        _make_tiny_clip(clip_dir / "source.mp4", rate=10, duration_s=1.0)

    completed = subprocess.run(
        [
            "python",
            "scripts/racketsport/extract_label_frames.py",
            "--root",
            str(root),
            "--out",
            str(out),
            "--every-frames",
            "3",
            "--max-width",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest = json.loads((out / "candidate" / "label_frame_manifest.json").read_text(encoding="utf-8"))
    frames = sorted((out / "candidate").glob("frame_*.jpg"))
    second_manifest = json.loads((out / "second_candidate" / "label_frame_manifest.json").read_text(encoding="utf-8"))
    second_frames = sorted((out / "second_candidate").glob("frame_*.jpg"))

    assert payload["clip_count"] == 2
    assert payload["total_frames"] == len(frames) + len(second_frames)
    assert payload["clips"][0]["clip"] == "candidate"
    assert payload["clips"][1]["clip"] == "second_candidate"
    assert manifest["clip"] == "candidate"
    assert manifest["sample_every_frames"] == 3
    assert second_manifest["clip"] == "second_candidate"
    assert second_manifest["sample_every_frames"] == 3
    assert len(frames) >= 3
    assert len(second_frames) >= 3
    assert all(frame.stat().st_size > 0 for frame in frames)
    assert all(frame.stat().st_size > 0 for frame in second_frames)


def test_extract_label_frames_cli_filters_to_requested_clip(tmp_path):
    root = tmp_path / "testclips"
    out = tmp_path / "label_frames"
    for clip_name in ("candidate", "second_candidate"):
        clip_dir = root / clip_name
        clip_dir.mkdir(parents=True)
        _make_tiny_clip(clip_dir / "source.mp4", rate=10, duration_s=1.0)

    completed = subprocess.run(
        [
            "python",
            "scripts/racketsport/extract_label_frames.py",
            "--root",
            str(root),
            "--out",
            str(out),
            "--every-frames",
            "3",
            "--max-width",
            "64",
            "--clip",
            "second_candidate",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["clip_count"] == 1
    assert payload["clips"][0]["clip"] == "second_candidate"
    assert (out / "second_candidate" / "label_frame_manifest.json").exists()
    assert not (out / "candidate").exists()


def test_extract_label_frames_cli_reports_missing_requested_clip(tmp_path):
    root = tmp_path / "testclips"
    out = tmp_path / "label_frames"
    clip_dir = root / "candidate"
    clip_dir.mkdir(parents=True)
    _make_tiny_clip(clip_dir / "source.mp4", rate=10, duration_s=1.0)

    completed = subprocess.run(
        [
            "python",
            "scripts/racketsport/extract_label_frames.py",
            "--root",
            str(root),
            "--out",
            str(out),
            "--clip",
            "missing",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requested clip not found: missing" in completed.stderr
    assert "available clips: candidate" in completed.stderr


def test_measure_decode_throughput_returns_realtime_factor(tmp_path):
    clip = tmp_path / "sample.mp4"
    _make_tiny_clip(clip, rate=10, duration_s=1.0)

    benchmark = measure_decode_throughput(clip, backend="cpu")

    assert benchmark.backend == "cpu"
    assert benchmark.frame_count >= 10
    assert benchmark.elapsed_s > 0.0
    assert benchmark.decode_fps > 0.0
    assert benchmark.realtime_factor > 0.0
    json.dumps(benchmark.to_dict())


def test_measure_decode_throughput_rejects_unknown_backend(tmp_path):
    clip = tmp_path / "sample.mp4"
    _make_tiny_clip(clip)

    with pytest.raises(ValueError, match="Unsupported decode backend"):
        measure_decode_throughput(clip, backend="bad")
