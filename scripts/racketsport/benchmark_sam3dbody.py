#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the required SAM-3D-Body H100 benchmark gate.")
    parser.add_argument("--out", type=Path, default=Path("runs/phase0/sam3dbody_benchmark.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write environment readiness only; use until checkpoints are installed.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    nvidia_smi = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"
    gpu_info = None
    if Path(nvidia_smi).exists() or shutil.which("nvidia-smi"):
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
        gpu_info = completed.stdout.strip() or completed.stderr.strip()

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
