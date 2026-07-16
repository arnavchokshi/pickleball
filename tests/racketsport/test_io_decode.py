from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.ingest_testclips import ingest_testclips
from tests.racketsport.json_schema_assertions import assert_matches_json_schema
from threed.racketsport.io_decode import (
    ClipQualityProbe,
    FrameSource,
    _laplacian_variance,
    analyze_clip_qc,
    build_frame_time_table,
    build_timebase_artifacts,
    decode_clip,
    load_frame_time_table,
    measure_decode_throughput,
    probe_clip,
    write_frame_time_table,
    time_for_frame,
)
from threed.racketsport.timebase import FrameAvailabilityStatus, TimebaseContract


ROOT = Path(__file__).resolve().parents[2]


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


def _make_vfr_clip(path, *, durations_s: list[float]):
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir()
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "black"]
    frame_paths = []
    for index, _duration in enumerate(durations_s):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colors[index % len(colors)]}:s=64x48:d=0.01",
            "-frames:v",
            "1",
            str(frame_path),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError:
            pytest.skip("ffmpeg is not installed")
        frame_paths.append(frame_path)

    concat = path.parent / f"{path.stem}_concat.txt"
    lines = []
    for frame_path, duration_s in zip(frame_paths, durations_s, strict=True):
        lines.append(f"file '{frame_path}'")
        lines.append(f"duration {duration_s:.9f}")
    lines.append(f"file '{frame_paths[-1]}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-fps_mode",
        "vfr",
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


def test_write_frame_time_table_emits_pts_sidecar_for_real_clip(tmp_path):
    clip = tmp_path / "sample.mp4"
    out = tmp_path / "frame_times.json"
    _make_tiny_clip(clip, rate=10, duration_s=0.6)

    table = write_frame_time_table(clip, out)

    assert out.exists()
    reloaded = load_frame_time_table(out)
    assert table == reloaded
    assert reloaded["artifact_type"] == "racketsport_frame_times"
    assert reloaded["schema_version"] == 1
    assert reloaded["clip_path"] == str(clip)
    assert reloaded["provenance"] == "ffprobe_pts"
    assert reloaded["frame_count"] >= 5
    assert reloaded["frames"][0] == {"frame": 0, "pts_s": pytest.approx(0.0)}
    assert all("pts_s" in frame for frame in reloaded["frames"])


def test_build_frame_time_table_preserves_synthetic_vfr_pts(tmp_path):
    clip = tmp_path / "vfr.mp4"
    _make_vfr_clip(clip, durations_s=[0.04, 0.20, 0.04, 0.16, 0.04])

    table = build_frame_time_table(clip)

    assert table["artifact_type"] == "racketsport_frame_times"
    assert table["provenance"] == "ffprobe_pts"
    times = [frame["pts_s"] for frame in table["frames"][:5]]
    assert times[0] == pytest.approx(0.0, abs=0.02)
    deltas = [round(right - left, 3) for left, right in zip(times, times[1:], strict=False)]
    assert len(set(deltas[:4])) > 1
    assert times[2] == pytest.approx(0.24, abs=0.03)


def test_synthetic_vfr_contract_accounts_for_non_monotonic_pts_and_gap(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    clip = tmp_path / "synthetic-vfr.mp4"
    clip.write_bytes(b"fixture")
    stream_payload = {
        "streams": [{
            "codec_type": "video",
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "time_base": "1/1000",
            "duration": "0.240",
            "nb_frames": "6",
        }],
        "format": {"duration": "0.240"},
    }
    frame_payload = {
        "frames": [
            {"best_effort_timestamp": 0, "best_effort_timestamp_time": "0.000", "duration": 40},
            {"best_effort_timestamp": 40, "best_effort_timestamp_time": "0.040", "duration": 40},
            {"best_effort_timestamp": 105, "best_effort_timestamp_time": "0.105", "duration": 65},
            {"best_effort_timestamp": 95, "best_effort_timestamp_time": "0.095", "duration": 40},
            {},
            {"best_effort_timestamp": 210, "best_effort_timestamp_time": "0.210", "duration": 50},
        ]
    }
    monkeypatch.setattr("threed.racketsport.io_decode._run_ffprobe", lambda _path: stream_payload)
    monkeypatch.setattr("threed.racketsport.io_decode._run_ffprobe_frames", lambda _path: frame_payload)

    build = build_timebase_artifacts(clip, capture_id="synthetic-vfr-gap")

    assert build.contract is not None
    assert len(build.contract.frames) == 6
    assert build.contract.frames[3].availability.status is FrameAvailabilityStatus.DROPPED
    assert build.contract.frames[4].availability.status is FrameAvailabilityStatus.MISSING
    raw = [frame.raw_encoded_pts for frame in build.contract.frames if frame.raw_encoded_pts is not None]
    assert all(
        current.pts_ticks * previous.timescale > previous.pts_ticks * current.timescale
        for previous, current in zip(raw, raw[1:], strict=False)
    )
    assert build.evidence["count_consistency"]["all_source_frames_accounted"] is True
    assert build.evidence["count_consistency"]["non_monotonic_frame_indices"] == [3]
    assert build.evidence["count_consistency"]["unavailable_frame_indices"] == [4]
    encoded = build.contract.to_json_bytes()
    assert TimebaseContract.from_json_bytes(encoded).to_json_bytes() == encoded


def test_wolverine_real_clip_exact_pts_count_and_legacy_byte_parity():
    clip = ROOT / "eval_clips" / "ball" / "wolverine_mixed_0200_mid_steep_corner" / "source.mp4"
    if not clip.is_file():
        pytest.skip("Wolverine internal-validation clip is not present")

    build = build_timebase_artifacts(clip, capture_id="wolverine_mixed_0200_mid_steep_corner")
    legacy_frames_bytes = json.dumps(
        build.legacy_frame_times["frames"],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    assert hashlib.sha256(legacy_frames_bytes).hexdigest() == "44e416b50db01bb6bbc38d583d8134ee4235421c3e13b1bb7f1d2d76ba8fcbb5"
    assert build.legacy_frame_times["provenance"] == "ffprobe_pts"
    assert build.contract is not None
    assert len(build.contract.frames) == 300
    schema = json.loads((ROOT / "docs" / "racketsport" / "timebase_schema.json").read_text(encoding="utf-8"))
    assert_matches_json_schema(build.contract.to_dict(), schema)
    assert build.evidence["count_consistency"]["stream_frame_count"] == 300
    assert build.evidence["count_consistency"]["ffprobe_reported_frame_count"] == 300
    assert build.evidence["count_consistency"]["canonical_accounted_frame_count"] == 300
    assert build.evidence["count_consistency"]["all_source_frames_accounted"] is True
    raw = [frame.raw_encoded_pts for frame in build.contract.frames if frame.raw_encoded_pts is not None]
    assert len(raw) == 300
    assert all(item.timescale == 15_360 for item in raw)
    assert all(right.pts_ticks > left.pts_ticks for left, right in zip(raw, raw[1:], strict=False))
    assert build.evidence["raw_pts_observations"][1]["source_timestamp_decimal"] == "0.033333"
    assert build.evidence["raw_pts_observations"][1]["conversion_method"] == "ffprobe_integer_timestamp_times_stream_time_base"
    derived = [frame.corrected_pts.corrected_time_s for frame in build.contract.frames[:299]]
    assert derived == [frame["pts_s"] for frame in build.legacy_frame_times["frames"]]


def test_time_for_frame_refuses_missing_table_entry_and_declares_explicit_cfr():
    frame_times = {
        "artifact_type": "racketsport_frame_times",
        "frames": [{"frame": 0, "pts_s": 0.0}],
    }
    with pytest.raises(ValueError, match="refusing silent CFR fallback"):
        time_for_frame(1, frame_times=frame_times, fps=30.0)

    resolved = time_for_frame(1, frame_times=None, fps=30.0, return_provenance=True)
    assert resolved.time_s == pytest.approx(1 / 30.0)
    assert resolved.time_basis == "constant_fps_assumed"
    assert resolved.provenance == "explicit_fps_argument"
    assert resolved.fallback_used is True


def test_cfr_fallback_never_emits_contract_claiming_raw_pts_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    clip = tmp_path / "no-pts.mp4"
    clip.write_bytes(b"fixture")
    monkeypatch.setattr(
        "threed.racketsport.io_decode._run_ffprobe",
        lambda _path: {
            "streams": [{
                "codec_type": "video",
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "time_base": "1/90000",
                "duration": "0.100",
                "nb_frames": "3",
            }],
            "format": {"duration": "0.100"},
        },
    )
    monkeypatch.setattr(
        "threed.racketsport.io_decode._run_ffprobe_frames",
        lambda _path: {"frames": [{}, {}, {}]},
    )

    build = build_timebase_artifacts(clip, capture_id="no-pts")

    assert build.legacy_frame_times["provenance"] == "constant_fps_assumed"
    assert build.contract is None
    assert build.evidence["timing_declaration"] == "constant_fps_assumed"
    assert build.evidence["raw_pts_authority"] is False
    assert build.evidence["timebase_contract_emitted"] is False


def test_decimal_string_pts_fallback_is_exact_and_declares_conversion(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    clip = tmp_path / "decimal-only-pts.mp4"
    clip.write_bytes(b"fixture")
    monkeypatch.setattr(
        "threed.racketsport.io_decode._run_ffprobe",
        lambda _path: {
            "streams": [{
                "codec_type": "video",
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "duration": "0.033333333",
                "nb_frames": "1",
            }],
            "format": {"duration": "0.033333333"},
        },
    )
    monkeypatch.setattr(
        "threed.racketsport.io_decode._run_ffprobe_frames",
        lambda _path: {"frames": [{"best_effort_timestamp_time": "0.123456789"}]},
    )

    build = build_timebase_artifacts(clip, capture_id="decimal-only")

    assert build.contract is not None
    raw = build.contract.frames[0].raw_encoded_pts
    assert raw is not None
    assert raw.pts_ticks == 123_456_789
    assert raw.timescale == 1_000_000_000
    observation = build.evidence["raw_pts_observations"][0]
    assert observation["source_timestamp_decimal"] == "0.123456789"
    assert observation["conversion_method"] == "exact_decimal_string_to_integer_ticks"


def test_probe_clip_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_clip(tmp_path / "missing.mp4")


def test_ingest_testclips_cli_reports_missing_root_without_traceback(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/ingest_testclips.py",
            "--root",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "runs"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "root does not exist" in completed.stderr
    assert "Traceback" not in completed.stderr


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
