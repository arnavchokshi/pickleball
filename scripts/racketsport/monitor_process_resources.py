#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_TYPE = "racketsport_resource_usage"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_cpu_totals() -> tuple[int, int] | None:
    stat_path = Path("/proc/stat")
    if not stat_path.is_file():
        return None
    try:
        fields = stat_path.read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [int(value) for value in fields[1:]]
    except ValueError:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return round(max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0)), 2)


def read_system_memory() -> dict[str, float] | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return None
    values: dict[str, int] = {}
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1])
    except (OSError, ValueError):
        return None
    total_kb = values.get("MemTotal")
    available_kb = values.get("MemAvailable")
    if total_kb is None or available_kb is None:
        return None
    used_mb = (total_kb - available_kb) / 1024.0
    total_mb = total_kb / 1024.0
    return {
        "system_memory_used_mb": round(used_mb, 1),
        "system_memory_total_mb": round(total_mb, 1),
    }


def read_gpu_sample() -> dict[str, float | int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = (
        "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 6:
        return None
    try:
        gpu_index = int(float(parts[0]))
        gpu_util = float(parts[1])
        gpu_mem_util = float(parts[2])
        gpu_mem_used = float(parts[3])
        gpu_mem_total = float(parts[4])
        gpu_power = float(parts[5])
    except ValueError:
        return None
    return {
        "gpu_index": gpu_index,
        "gpu_utilization_pct": round(gpu_util, 2),
        "gpu_memory_utilization_pct": round(gpu_mem_util, 2),
        "gpu_memory_used_mb": round(gpu_mem_used, 1),
        "gpu_memory_total_mb": round(gpu_mem_total, 1),
        "gpu_power_w": round(gpu_power, 2),
    }


def collect_sample(start_monotonic: float, previous_cpu: tuple[int, int] | None) -> tuple[dict[str, Any], tuple[int, int] | None]:
    current_cpu = read_cpu_totals()
    sample: dict[str, Any] = {
        "t_s": round(time.monotonic() - start_monotonic, 3),
    }
    gpu_sample = read_gpu_sample()
    if gpu_sample:
        sample.update(gpu_sample)
    memory_sample = read_system_memory()
    if memory_sample:
        sample.update(memory_sample)
    sample["cpu_utilization_pct"] = cpu_percent(previous_cpu, current_cpu)
    return sample, current_cpu


def summarize(samples: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "duration_s": round(duration_s, 3),
    }
    _add_avg_max(summary, samples, "gpu_utilization_pct", "gpu_utilization_avg_pct", "gpu_utilization_max_pct")
    _add_avg_max(summary, samples, "gpu_memory_utilization_pct", "gpu_memory_utilization_avg_pct", "gpu_memory_utilization_max_pct")
    _add_avg_max(summary, samples, "gpu_power_w", "gpu_power_avg_w", "gpu_power_max_w")
    _add_avg_max(summary, samples, "cpu_utilization_pct", "cpu_utilization_avg_pct", "cpu_utilization_max_pct")
    _add_max(summary, samples, "gpu_memory_used_mb", "gpu_memory_used_max_mb")
    _add_max(summary, samples, "gpu_memory_total_mb", "gpu_memory_total_mb")
    _add_max(summary, samples, "system_memory_used_mb", "system_memory_used_max_mb")
    _add_max(summary, samples, "system_memory_total_mb", "system_memory_total_mb")
    return summary


def _numeric_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _add_avg_max(summary: dict[str, Any], samples: list[dict[str, Any]], sample_key: str, avg_key: str, max_key: str) -> None:
    values = _numeric_values(samples, sample_key)
    if not values:
        return
    summary[avg_key] = round(sum(values) / len(values), 2)
    summary[max_key] = round(max(values), 2)


def _add_max(summary: dict[str, Any], samples: list[dict[str, Any]], sample_key: str, output_key: str) -> None:
    values = _numeric_values(samples, sample_key)
    if values:
        summary[output_key] = round(max(values), 2)


def build_artifact(
    *,
    command: list[str],
    sample_interval_s: float,
    started_at: str,
    completed_at: str,
    exit_code: int,
    samples: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "sample_interval_s": sample_interval_s,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": exit_code,
        "samples": samples,
        "summary": summarize(samples, duration_s),
    }


def run_and_monitor(command: list[str], *, out_path: Path, sample_interval_s: float) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    stop_event = threading.Event()
    started_at = utc_now()
    start_monotonic = time.monotonic()
    previous_cpu: tuple[int, int] | None = None

    def sampler() -> None:
        nonlocal previous_cpu
        while not stop_event.is_set():
            sample, previous_cpu = collect_sample(start_monotonic, previous_cpu)
            samples.append(sample)
            stop_event.wait(sample_interval_s)

    process = subprocess.Popen(command)
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    exit_code = process.wait()
    stop_event.set()
    sampler_thread.join(timeout=max(1.0, sample_interval_s))
    duration_s = time.monotonic() - start_monotonic
    completed_at = utc_now()
    artifact = build_artifact(
        command=command,
        sample_interval_s=sample_interval_s,
        started_at=started_at,
        completed_at=completed_at,
        exit_code=exit_code,
        samples=samples,
        duration_s=duration_s,
    )
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command while recording coarse CPU/GPU resource telemetry.")
    parser.add_argument("--out", required=True, type=Path, help="Path to write gpu_resource_usage.json")
    parser.add_argument("--sample-interval", type=float, default=5.0, help="Seconds between resource samples")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run_and_monitor(args.command, out_path=args.out, sample_interval_s=args.sample_interval)


if __name__ == "__main__":
    raise SystemExit(main())
