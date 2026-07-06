import subprocess
import sys

from scripts.racketsport.monitor_process_resources import build_artifact, summarize


def test_monitor_process_resources_cli_help_direct_reference() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/monitor_process_resources.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--out" in completed.stdout
    assert "--sample-interval" in completed.stdout


def test_resource_usage_summary_aggregates_gpu_vram_cpu_and_duration() -> None:
    samples = [
        {
            "gpu_utilization_pct": 10.0,
            "gpu_memory_used_mb": 1000.0,
            "gpu_memory_total_mb": 24000.0,
            "cpu_utilization_pct": 20.0,
        },
        {
            "gpu_utilization_pct": 70.0,
            "gpu_memory_used_mb": 12000.0,
            "gpu_memory_total_mb": 24000.0,
            "cpu_utilization_pct": 60.0,
        },
    ]

    summary = summarize(samples, duration_s=12.3456)

    assert summary["sample_count"] == 2
    assert summary["duration_s"] == 12.346
    assert summary["gpu_utilization_avg_pct"] == 40.0
    assert summary["gpu_utilization_max_pct"] == 70.0
    assert summary["gpu_memory_used_max_mb"] == 12000.0
    assert summary["gpu_memory_total_mb"] == 24000.0
    assert summary["cpu_utilization_avg_pct"] == 40.0


def test_resource_usage_artifact_preserves_command_and_exit_code() -> None:
    artifact = build_artifact(
        command=["python", "scripts/racketsport/process_video.py"],
        sample_interval_s=5.0,
        started_at="2026-07-04T12:00:00Z",
        completed_at="2026-07-04T12:01:00Z",
        exit_code=1,
        samples=[],
        duration_s=60.0,
    )

    assert artifact["artifact_type"] == "racketsport_resource_usage"
    assert artifact["command"] == ["python", "scripts/racketsport/process_video.py"]
    assert artifact["exit_code"] == 1
    assert artifact["summary"]["sample_count"] == 0
