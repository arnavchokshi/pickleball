#!/usr/bin/env python3
"""Byte/hash-compare BODY output artifacts between two dispatch runs.

Used for the warm_body_worker_20260728 A/B: proves a warm-worker-routed job
produces the same content-bearing artifacts as a cold job for the same
inputs. Timing/provenance-only artifacts are expected to differ (they
legitimately carry wall-clock numbers, timestamps, or the warm/cold routing
note itself) and are reported separately, not folded into the pass/fail verdict.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Files that legitimately differ run-to-run (wall-clock timing, timestamps,
# or this lane's own routing note) -- reported but not part of the
# content-identity verdict.
TIMING_OR_PROVENANCE_ONLY = {
    "body_stage_phase_timing.json",
    "body_serialization_timing.json",
    "remote_body_dispatch_timing.json",
    "remote_body_stdout.log",
    "remote_body_runner.py",
    "version_stamp.json",
    "remote_version_verification.json",
    "warm_worker_dispatch.json",
    "pipeline_run.json",
    ".run_identity",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: byte_compare.py <run_a_dir> <run_b_dir>", file=sys.stderr)
        return 2
    dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])

    names_a = {p.name for p in dir_a.glob("*.json")}
    names_b = {p.name for p in dir_b.glob("*.json")}
    common = sorted(names_a & names_b)

    identical: list[str] = []
    differing: list[str] = []
    skipped: list[str] = []

    for name in common:
        if name in TIMING_OR_PROVENANCE_ONLY:
            skipped.append(name)
            continue
        sha_a = sha256_of(dir_a / name)
        sha_b = sha256_of(dir_b / name)
        if sha_a == sha_b:
            identical.append(name)
        else:
            differing.append(name)

    result = {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "only_in_a": sorted(names_a - names_b),
        "only_in_b": sorted(names_b - names_a),
        "content_identical": identical,
        "content_differing": differing,
        "timing_or_provenance_only_skipped": skipped,
        "verdict": "byte_identical_content" if not differing else "CONTENT_DIFFERS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not differing else 1


if __name__ == "__main__":
    raise SystemExit(main())
