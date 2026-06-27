from __future__ import annotations

import json
import subprocess
import sys


def _write_benchmark(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _benchmark(*, clip, backend, decode_fps, realtime_factor, frame_count=120):
    return {
        "clip_path": clip,
        "backend": backend,
        "elapsed_s": 1.0,
        "duration_s": 4.0,
        "frame_count": frame_count,
        "decode_fps": decode_fps,
        "realtime_factor": realtime_factor,
        "resolution": [1920, 1080],
        "fps": 30.0,
    }


def test_decode_benchmark_summary_reports_fastest_backend_and_aggregates(tmp_path):
    artifacts = [
        _write_benchmark(
            tmp_path / "court_a_cpu.json",
            _benchmark(clip="court_a.mp4", backend="cpu", decode_fps=60.0, realtime_factor=2.0),
        ),
        _write_benchmark(
            tmp_path / "court_a_cuda.json",
            _benchmark(clip="court_a.mp4", backend="cuda", decode_fps=180.0, realtime_factor=6.0),
        ),
        _write_benchmark(
            tmp_path / "court_b_cpu.json",
            _benchmark(clip="court_b.mp4", backend="cpu", decode_fps=90.0, realtime_factor=3.0),
        ),
        _write_benchmark(
            tmp_path / "court_b_cuda.json",
            _benchmark(clip="court_b.mp4", backend="cuda", decode_fps=75.0, realtime_factor=2.5),
        ),
    ]

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/summarize_decode_benchmarks.py", *map(str, artifacts)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == 1
    assert payload["benchmark_count"] == 4
    assert payload["clip_count"] == 2
    assert "empirical per real clip set" in payload["backend_choice_note"]
    assert payload["fastest_backend_by_clip"] == [
        {
            "clip": "court_a.mp4",
            "backend": "cuda",
            "decode_fps": 180.0,
            "realtime_factor": 6.0,
            "source": str(artifacts[1]),
        },
        {
            "clip": "court_b.mp4",
            "backend": "cpu",
            "decode_fps": 90.0,
            "realtime_factor": 3.0,
            "source": str(artifacts[2]),
        },
    ]
    assert payload["aggregate_by_backend"] == {
        "cpu": {
            "runs": 2,
            "clips": 2,
            "mean_decode_fps": 75.0,
            "mean_realtime_factor": 2.5,
            "total_duration_s": 8.0,
            "total_elapsed_s": 2.0,
            "total_frames": 240,
        },
        "cuda": {
            "runs": 2,
            "clips": 2,
            "mean_decode_fps": 127.5,
            "mean_realtime_factor": 4.25,
            "total_duration_s": 8.0,
            "total_elapsed_s": 2.0,
            "total_frames": 240,
        },
    }


def test_decode_benchmark_summary_fails_on_invalid_payload(tmp_path):
    artifact = _write_benchmark(
        tmp_path / "bad.json",
        {
            "clip_path": "court_a.mp4",
            "backend": "cpu",
            "elapsed_s": 1.0,
            "duration_s": 4.0,
            "frame_count": 120,
            "decode_fps": "fast",
            "realtime_factor": 2.0,
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/summarize_decode_benchmarks.py", str(artifact)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "bad.json: decode_fps must be numeric" in completed.stderr


def test_decode_benchmark_summary_writes_markdown(tmp_path):
    artifacts = [
        _write_benchmark(
            tmp_path / "court_a_cpu.json",
            _benchmark(clip="court_a.mp4", backend="cpu", decode_fps=60.0, realtime_factor=2.0),
        ),
        _write_benchmark(
            tmp_path / "court_a_cuda.json",
            _benchmark(clip="court_a.mp4", backend="cuda", decode_fps=180.0, realtime_factor=6.0),
        ),
    ]
    out = tmp_path / "decode_summary.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/summarize_decode_benchmarks.py",
            *map(str, artifacts),
            "--markdown-out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    body = out.read_text(encoding="utf-8")

    assert payload["fastest_backend_by_clip"][0]["backend"] == "cuda"
    assert "# Decode Benchmark Summary" in body
    assert "Backend choice is empirical per real clip set" in body
    assert "| `court_a.mp4` | `cuda` | 180.000 | 6.000 |" in body
    assert "| `cpu` | 1 | 1 | 60.000 | 2.000 | 120 |" in body
