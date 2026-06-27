#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_profile_log(text: str) -> dict:
    timing = re.search(
        r"Timing Statistics \(n=(?P<runs>\d+)\):"
        r".*?Average:\s*(?P<average>[\d.]+)\s*ms"
        r".*?Std Dev:\s*(?P<std>[\d.]+)\s*ms"
        r".*?Min:\s*(?P<min>[\d.]+)\s*ms"
        r".*?Max:\s*(?P<max>[\d.]+)\s*ms",
        text,
        flags=re.DOTALL,
    )
    if timing is None:
        raise ValueError("profile log is missing Timing Statistics block")

    peak_vram_match = re.search(r"Peak GPU memory:\s*(?P<peak>[\d.]+)\s*MB", text)
    if peak_vram_match is None:
        raise ValueError("profile log is missing Peak GPU memory summary")

    detected = re.search(r"Detected\s+(?P<count>\d+)\s+(?:people|persons|humans)", text, flags=re.IGNORECASE)
    output_dir = _line_value(text, "Results saved to") or _line_value(text, "Output")
    return {
        "schema_version": 1,
        "status": "measured",
        "source": "fast_sam_3d_body_profile_nsight",
        "image_path": _line_value(text, "Image"),
        "model": _line_value(text, "Model"),
        "detector": _line_value(text, "Detector"),
        "hand_box_source": _line_value(text, "Hand Box Source"),
        "output_dir": output_dir,
        "metrics": {
            "average_ms": float(timing.group("average")),
            "std_ms": float(timing.group("std")),
            "min_ms": float(timing.group("min")),
            "max_ms": float(timing.group("max")),
            "benchmark_runs": int(timing.group("runs")),
            "peak_vram_mb": float(peak_vram_match.group("peak")),
            "detected_people": int(detected.group("count")) if detected is not None else None,
        },
        "notes": [],
    }


def _line_value(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}:\s*(?P<value>.+)$", text, flags=re.MULTILINE)
    return match.group("value").strip() if match is not None else None


def _query_gpu_info() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"
    if not Path(nvidia_smi).exists() and not shutil.which("nvidia-smi"):
        return None
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the required SAM-3D-Body H100 benchmark gate.")
    parser.add_argument("--out", type=Path, default=Path("runs/phase0/sam3dbody_benchmark.json"))
    parser.add_argument("--profile-log", type=Path, help="Parse stdout captured from Fast-SAM profile_nsight.py.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write environment readiness only; use until checkpoints are installed.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gpu_info = _query_gpu_info()
    if args.profile_log is not None:
        result = parse_profile_log(args.profile_log.read_text(encoding="utf-8"))
        result["profile_log"] = str(args.profile_log)
        result["gpu_info"] = gpu_info
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
        return 0

    result = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "blocked_missing_model_runner",
        "gpu_info": gpu_info,
        "metrics": {
            "b1_fps_per_player": None,
            "batched_crop_fps": None,
            "peak_vram_mb": None,
        },
        "notes": [
            "Phase 0 requires replacing this dry-run with real Fast SAM-3D-Body inference once checkpoints resolve."
        ],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0 if args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
