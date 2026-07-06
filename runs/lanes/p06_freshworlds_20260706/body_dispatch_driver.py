#!/usr/bin/env python3
"""P0-6 lane driver: BODY dispatch with transport-retried rsync (no source edits).

WHY THIS EXISTS: this Mac's OpenSSH client intermittently kills sustained bulk
uploads to fleet VM 34.143.175.207 (`client_loop: ssh_packet_write_poll: ...
Result too large`; 5 utun VPN interfaces are in the path, worst MTU 1000).
Diagnosis 2026-07-06: GNU rsync 3.4.4 vs openrsync makes no difference; IPQoS
none, cipher change, and --bwlimit do not fix it. BUT every failed rsync still
lands a partial batch and rsync is incremental, so retrying the SAME command
makes strict forward progress and completes within a few attempts.

This driver calls the committed scripts/racketsport/remote_body_dispatch.py
dispatch_body_stage() through its own public `run: RunFn` injection parameter,
wrapping ONLY `rsync` commands that fail with transport exit codes
(10/12/30/35/255) in a bounded retry loop. Every other command (ssh mkdir,
layout checks, and especially the GPU-locked remote BODY command) runs exactly
once, exactly as the pipeline would run it. Same code path, resilient transport.

Usage: body_dispatch_driver.py <clip> <clip_dir> <video_suffix> <command_timeout_s>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/arnavchokshi/Desktop/pickleball")
sys.path.insert(0, str(ROOT))

from scripts.racketsport.remote_body_dispatch import (  # noqa: E402
    RemoteConfig,
    dispatch_body_stage,
    _run,
)

RETRYABLE_RSYNC_EXIT_CODES = {10, 12, 30, 35, 255}
MAX_RSYNC_ATTEMPTS = 12
RETRY_SLEEP_S = 4


def run_with_rsync_retry(cmd, timeout):
    is_rsync = bool(cmd) and Path(cmd[0]).name == "rsync"
    attempts = MAX_RSYNC_ATTEMPTS if is_rsync else 1
    last = None
    for attempt in range(1, attempts + 1):
        last = _run(cmd, timeout)
        if last.returncode == 0:
            if attempt > 1:
                print(f"[driver] rsync succeeded on attempt {attempt}", flush=True)
            return last
        if not is_rsync or last.returncode not in RETRYABLE_RSYNC_EXIT_CODES:
            return last
        print(
            f"[driver] rsync attempt {attempt}/{attempts} rc={last.returncode}: "
            f"{(last.stderr or '').strip()[:200]}",
            flush=True,
        )
        time.sleep(RETRY_SLEEP_S)
    return last


def main() -> int:
    clip, clip_dir, video_suffix, timeout_s = (
        sys.argv[1],
        Path(sys.argv[2]),
        sys.argv[3],
        int(sys.argv[4]),
    )
    config = RemoteConfig(
        host="arnavchokshi@34.143.175.207",
        repo="/home/arnavchokshi/coldstart_20260706/repo",
        python="/home/arnavchokshi/coldstart_20260706/body_runtime/body_venv/bin/python",
        fast_sam_python="/home/arnavchokshi/coldstart_20260706/body_runtime/body_venv/bin/python",
        fast_sam_root="/home/arnavchokshi/coldstart_20260706/body_runtime/Fast-SAM-3D-Body",
        command_timeout_s=timeout_s,
    )
    started = time.monotonic()
    result = dispatch_body_stage(
        clip=clip,
        clip_dir=clip_dir,
        video_path=clip_dir / f"source{video_suffix}",
        body_frames_dir=clip_dir / "body_frames",
        config=config,
        max_players=4,
        run=run_with_rsync_retry,
    )
    print(
        json.dumps(
            {
                "driver_status": result.status,
                "remote_run_dir": result.remote_run_dir,
                "dispatch_wall_seconds": round(result.wall_seconds, 1),
                "driver_total_wall_seconds": round(time.monotonic() - started, 1),
                "synced_inputs": result.synced_inputs,
                "synced_outputs": result.synced_outputs,
                "notes": result.notes,
                "timing": result.timing,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
