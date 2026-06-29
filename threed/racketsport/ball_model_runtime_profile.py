"""Runtime profiling wrapper for ball-tracking model commands."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import BallModelRuntimeProfile

TAIL_LIMIT = 4000


@dataclass(frozen=True)
class RuntimeProbe:
    cuda_available: bool
    cuda_device_name: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_visible_devices: str | None = None


def current_runtime_probe() -> RuntimeProbe:
    torch_version = None
    torch_cuda_version = None
    cuda_available = False
    cuda_device_name = None
    try:
        import torch  # type: ignore[import-not-found]

        torch_version = str(torch.__version__)
        torch_cuda_version = str(torch.version.cuda)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return RuntimeProbe(
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def build_runtime_profile(
    *,
    candidate: str,
    model_id: str,
    clip_id: str,
    video: str,
    source_fps: float,
    batch_size: int,
    command: list[str],
    runner_metadata: str | Path | None = None,
    require_cuda: bool = False,
    expected_gpu_name: str | None = None,
    runtime_probe: RuntimeProbe | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command must not be empty")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    probe = runtime_probe or current_runtime_probe()
    runtime_env = _runtime_env(probe)
    if require_cuda and not probe.cuda_available:
        payload = _base_payload(
            candidate=candidate,
            model_id=model_id,
            clip_id=clip_id,
            video=video,
            source_fps=source_fps,
            batch_size=batch_size,
            command=command,
            status="blocked_missing_cuda",
            returncode=None,
            wall_seconds=None,
            runtime_env=runtime_env,
            gpu_verified=False,
            claim_scope="blocked_missing_cuda",
            notes=["CUDA required but unavailable"],
        )
        BallModelRuntimeProfile.model_validate(payload)
        return payload

    start = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - start
    metadata = _read_runner_metadata(runner_metadata)
    runner_runtime = _runner_runtime(metadata)
    processed = _processed_frame_count(runner_runtime)
    video_seconds = _video_seconds_processed(runner_runtime, processed, source_fps)
    runtime_wall = _runtime_wall_seconds(runner_runtime) or wall_seconds
    effective_fps = _effective_fps(runner_runtime, processed, runtime_wall)
    realtime_factor = _realtime_factor(runner_runtime, effective_fps, source_fps)
    gpu_verified = _gpu_verified(probe, require_cuda=require_cuda, expected_gpu_name=expected_gpu_name)
    claim_scope = "h100_runtime_profile_not_accuracy_gate" if gpu_verified else "cpu_profiler_smoke"
    payload = _base_payload(
        candidate=candidate,
        model_id=model_id,
        clip_id=clip_id,
        video=video,
        source_fps=source_fps,
        batch_size=batch_size,
        command=command,
        status="ran" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        wall_seconds=runtime_wall,
        runtime_env=runtime_env,
        gpu_verified=gpu_verified,
        claim_scope=claim_scope,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        processed_frame_count=processed,
        video_seconds_processed=video_seconds,
        effective_fps=effective_fps,
        realtime_factor=realtime_factor,
        timing_breakdown=_timing_breakdown(runner_runtime),
    )
    BallModelRuntimeProfile.model_validate(payload)
    return payload


def write_runtime_profile(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_payload(
    *,
    candidate: str,
    model_id: str,
    clip_id: str,
    video: str,
    source_fps: float,
    batch_size: int,
    command: list[str],
    status: str,
    returncode: int | None,
    wall_seconds: float | None,
    runtime_env: dict[str, Any],
    gpu_verified: bool,
    claim_scope: str,
    stdout_tail: str = "",
    stderr_tail: str = "",
    notes: list[str] | None = None,
    processed_frame_count: int | None = None,
    video_seconds_processed: float | None = None,
    effective_fps: float | None = None,
    realtime_factor: float | None = None,
    timing_breakdown: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_model_runtime_profile",
        "candidate": candidate,
        "model_id": model_id,
        "clip_id": clip_id,
        "video": video,
        "source_fps": float(source_fps),
        "batch_size": int(batch_size),
        "command": list(command),
        "returncode": returncode,
        "status": status,
        "wall_seconds": wall_seconds,
        "processed_frame_count": processed_frame_count,
        "video_seconds_processed": video_seconds_processed,
        "effective_fps": effective_fps,
        "realtime_factor": realtime_factor,
        "timing_breakdown": dict(timing_breakdown or {}),
        "runtime_env": runtime_env,
        "gpu_verified": bool(gpu_verified),
        "claim_scope": claim_scope,
        "verified": False,
        "not_ground_truth": True,
        "not_accuracy_verified": True,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "notes": list(notes or []),
    }


def _runtime_env(probe: RuntimeProbe) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": probe.torch_version,
        "torch_cuda_version": probe.torch_cuda_version,
        "cuda_available": probe.cuda_available,
        "cuda_device_name": probe.cuda_device_name,
        "cuda_visible_devices": probe.cuda_visible_devices,
    }


def _read_runner_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing runner metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"runner metadata must be an object: {metadata_path}")
    return payload


def _runner_runtime(metadata: dict[str, Any]) -> dict[str, Any]:
    runtime = metadata.get("runtime")
    if isinstance(runtime, dict):
        return runtime
    return metadata


def _processed_frame_count(runtime: dict[str, Any]) -> int | None:
    for key in ("processed_frame_count", "decoded_frame_count", "frame_count"):
        value = runtime.get(key)
        if value is not None:
            return int(value)
    return None


def _video_seconds_processed(runtime: dict[str, Any], processed: int | None, source_fps: float) -> float | None:
    value = runtime.get("video_seconds_processed")
    if value is not None:
        return float(value)
    if processed is not None:
        return float(processed) / float(source_fps)
    return None


def _runtime_wall_seconds(runtime: dict[str, Any]) -> float | None:
    for key in ("wall_seconds", "seconds"):
        value = runtime.get(key)
        if value is not None:
            return float(value)
    return None


def _effective_fps(runtime: dict[str, Any], processed: int | None, wall_seconds: float | None) -> float | None:
    value = runtime.get("effective_fps")
    if value is not None:
        return float(value)
    if processed is not None and wall_seconds is not None and wall_seconds > 0.0:
        return float(processed) / float(wall_seconds)
    return None


def _realtime_factor(runtime: dict[str, Any], effective_fps: float | None, source_fps: float) -> float | None:
    value = runtime.get("realtime_factor")
    if value is not None:
        return float(value)
    if effective_fps is not None:
        return float(effective_fps) / float(source_fps)
    return None


def _timing_breakdown(runtime: dict[str, Any]) -> dict[str, float]:
    breakdown = runtime.get("timing_breakdown")
    if not isinstance(breakdown, dict):
        return {}
    return {str(key): float(value) for key, value in breakdown.items() if isinstance(value, (int, float))}


def _gpu_verified(
    probe: RuntimeProbe,
    *,
    require_cuda: bool,
    expected_gpu_name: str | None,
) -> bool:
    if not require_cuda or not probe.cuda_available or not probe.cuda_device_name:
        return False
    if expected_gpu_name and expected_gpu_name not in probe.cuda_device_name:
        return False
    return True


def _tail(value: str) -> str:
    if len(value) <= TAIL_LIMIT:
        return value
    return value[-TAIL_LIMIT:]


__all__ = [
    "BallModelRuntimeProfile",
    "RuntimeProbe",
    "build_runtime_profile",
    "current_runtime_probe",
    "write_runtime_profile",
]
