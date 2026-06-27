from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_benchmark_module():
    path = Path("scripts/racketsport/benchmark_sam3dbody.py")
    spec = importlib.util.spec_from_file_location("benchmark_sam3dbody", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE_LOG = """
============================================================
SAM 3D Body GPU Profiler
============================================================
Image: /workspace/pickleball/runs/phase0/real_smoke/frame_001.jpg
Model: facebook/sam-3d-body-dinov3
Detector: yolo (yolo11n.pt)
Hand Box Source: body_decoder
Output: runs/phase0/fast_sam_real_smoke_profile
Detected 4 persons

Timing Statistics (n=3):
  Average: 342.35 ms
  Std Dev: 2.50 ms
  Min: 340.38 ms
  Max: 345.87 ms

============================================================
PROFILING SUMMARY
============================================================
Average inference time: 342.35 ms
Peak GPU memory: 4720.29 MB
Results saved to: runs/phase0/fast_sam_real_smoke_profile
============================================================
"""


def test_parse_profile_log_extracts_required_fast_sam_metrics():
    benchmark = _load_benchmark_module()

    payload = benchmark.parse_profile_log(PROFILE_LOG)

    assert payload["status"] == "measured"
    assert payload["image_path"] == "/workspace/pickleball/runs/phase0/real_smoke/frame_001.jpg"
    assert payload["model"] == "facebook/sam-3d-body-dinov3"
    assert payload["detector"] == "yolo (yolo11n.pt)"
    assert payload["hand_box_source"] == "body_decoder"
    assert payload["output_dir"] == "runs/phase0/fast_sam_real_smoke_profile"
    assert payload["metrics"] == {
        "average_ms": 342.35,
        "std_ms": 2.5,
        "min_ms": 340.38,
        "max_ms": 345.87,
        "benchmark_runs": 3,
        "peak_vram_mb": 4720.29,
        "detected_people": 4,
    }


def test_benchmark_sam3dbody_cli_writes_json_from_profile_log(tmp_path):
    profile_log = tmp_path / "profile_stdout.log"
    out = tmp_path / "sam3dbody_benchmark.json"
    profile_log.write_text(PROFILE_LOG, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/benchmark_sam3dbody.py",
            "--profile-log",
            str(profile_log),
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["profile_log"] == str(profile_log)
    assert payload["metrics"]["average_ms"] == 342.35
    assert payload["metrics"]["peak_vram_mb"] == 4720.29
