#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_model_runtime_profile import build_runtime_profile, write_runtime_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a ball-model runner command without claiming accuracy.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--runner-metadata", type=Path, default=None)
    parser.add_argument("--expected-gpu-name", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("missing runner command after --", file=sys.stderr)
        return 2

    try:
        payload = build_runtime_profile(
            candidate=args.candidate,
            model_id=args.model_id,
            clip_id=args.clip_id,
            video=args.video,
            source_fps=args.source_fps,
            batch_size=args.batch_size,
            command=command,
            runner_metadata=args.runner_metadata,
            require_cuda=args.require_cuda,
            expected_gpu_name=args.expected_gpu_name,
        )
        write_runtime_profile(args.out_json, payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
