#!/usr/bin/env python3
"""Compatibility wrapper for the DATA-2 prototype autolabel bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.run_prototype_autolabel import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
