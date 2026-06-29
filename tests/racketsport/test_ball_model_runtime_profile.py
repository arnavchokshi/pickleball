from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_model_runtime_profile import (
    BallModelRuntimeProfile,
    RuntimeProbe,
    build_runtime_profile,
)


def _runner_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_profile_wrapper_records_cpu_smoke_without_gpu_claim(tmp_path: Path) -> None:
    runner = _runner_script(tmp_path / "runner.py", "print('runner ok')\n")

    payload = build_runtime_profile(
        candidate="tracknetv3",
        model_id="tracknetv3",
        clip_id="clip_001",
        video="input.mp4",
        source_fps=30.0,
        batch_size=16,
        command=[sys.executable, str(runner)],
        runtime_probe=RuntimeProbe(cuda_available=False),
    )

    profile = BallModelRuntimeProfile.model_validate(payload)
    assert profile.returncode == 0
    assert profile.gpu_verified is False
    assert profile.verified is False
    assert profile.claim_scope == "cpu_profiler_smoke"
    assert profile.not_accuracy_verified is True
    assert profile.stdout_tail == "runner ok\n"


def test_profile_wrapper_require_cuda_fails_closed_without_cuda(tmp_path: Path) -> None:
    runner = _runner_script(tmp_path / "runner.py", "print('should not run')\n")

    payload = build_runtime_profile(
        candidate="tracknetv3",
        model_id="tracknetv3",
        clip_id="clip_001",
        video="input.mp4",
        source_fps=30.0,
        batch_size=16,
        command=[sys.executable, str(runner)],
        require_cuda=True,
        runtime_probe=RuntimeProbe(cuda_available=False),
    )

    profile = BallModelRuntimeProfile.model_validate(payload)
    assert profile.returncode is None
    assert profile.status == "blocked_missing_cuda"
    assert profile.gpu_verified is False
    assert profile.claim_scope == "blocked_missing_cuda"
    assert "CUDA required but unavailable" in profile.notes


def test_profile_wrapper_normalizes_runner_metadata(tmp_path: Path) -> None:
    runner = _runner_script(tmp_path / "runner.py", "print('ok')\n")
    metadata = tmp_path / "ball_track_run.json"
    metadata.write_text(
        json.dumps(
            {
                "runtime": {
                    "processed_frame_count": 60,
                    "wall_seconds": 2.0,
                    "video_seconds_processed": 2.0,
                    "timing_breakdown": {"inference_seconds": 0.8},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_runtime_profile(
        candidate="tracknetv3",
        model_id="tracknetv3",
        clip_id="clip_001",
        video="input.mp4",
        source_fps=30.0,
        batch_size=16,
        command=[sys.executable, str(runner)],
        runner_metadata=metadata,
        runtime_probe=RuntimeProbe(cuda_available=False),
    )

    profile = BallModelRuntimeProfile.model_validate(payload)
    assert profile.processed_frame_count == 60
    assert profile.video_seconds_processed == pytest.approx(2.0)
    assert profile.effective_fps == pytest.approx(30.0)
    assert profile.realtime_factor == pytest.approx(1.0)
    assert profile.timing_breakdown == {"inference_seconds": 0.8}


def test_profile_schema_rejects_gpu_verified_without_cuda_evidence() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_model_runtime_profile",
        "candidate": "tracknetv3",
        "model_id": "tracknetv3",
        "clip_id": "clip_001",
        "video": "input.mp4",
        "source_fps": 30.0,
        "batch_size": 16,
        "command": ["echo", "ok"],
        "returncode": 0,
        "status": "ran",
        "wall_seconds": 1.0,
        "runtime_env": {"cuda_available": False, "cuda_device_name": None},
        "gpu_verified": True,
        "claim_scope": "h100_runtime_profile_not_accuracy_gate",
        "verified": False,
        "not_ground_truth": True,
        "not_accuracy_verified": True,
    }

    with pytest.raises(ValueError, match="gpu_verified requires CUDA evidence"):
        BallModelRuntimeProfile.model_validate(payload)


def test_profile_command_tail_is_captured_on_failure(tmp_path: Path) -> None:
    runner = _runner_script(
        tmp_path / "runner.py",
        "import sys\nprint('bad stdout')\nprint('bad stderr', file=sys.stderr)\nsys.exit(2)\n",
    )

    payload = build_runtime_profile(
        candidate="tracknetv3",
        model_id="tracknetv3",
        clip_id="clip_001",
        video="input.mp4",
        source_fps=30.0,
        batch_size=16,
        command=[sys.executable, str(runner)],
        runtime_probe=RuntimeProbe(cuda_available=False),
    )

    profile = BallModelRuntimeProfile.model_validate(payload)
    assert profile.returncode == 2
    assert profile.status == "failed"
    assert profile.verified is False
    assert "bad stderr" in profile.stderr_tail


def test_profile_cli_writes_json(tmp_path: Path) -> None:
    runner = _runner_script(tmp_path / "runner.py", "print('ok')\n")
    out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/profile_ball_model_runtime.py",
            "--candidate",
            "totnet",
            "--model-id",
            "totnet_badminton",
            "--clip-id",
            "clip_001",
            "--video",
            "input.mp4",
            "--source-fps",
            "30",
            "--batch-size",
            "8",
            "--out-json",
            str(out),
            "--",
            sys.executable,
            str(runner),
        ],
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = BallModelRuntimeProfile.model_validate(payload)
    assert profile.candidate == "totnet"
    assert profile.status == "ran"
    assert profile.gpu_verified is False
