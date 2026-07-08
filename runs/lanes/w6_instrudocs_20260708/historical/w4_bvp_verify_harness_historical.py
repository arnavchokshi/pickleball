#!/usr/bin/env python3
"""Historical pointer for the wave-4 BVP verifier.

The original file is intentionally preserved at
`runs/lanes/w4_bvp_verify_20260707/harness/bvp_verify_harness.py`. It encodes
the pre-v2 contact-519 split-junction assertion that the manager ruled stale for
BVP span protection v2 on 2026-07-08. Keep this wrapper small so wave-4 evidence
continues to cite the original immutable instrument.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HISTORICAL_HARNESS = ROOT / "runs/lanes/w4_bvp_verify_20260707/harness/bvp_verify_harness.py"


if __name__ == "__main__":
    print(HISTORICAL_HARNESS)
