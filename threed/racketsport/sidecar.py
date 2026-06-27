"""Compatibility helpers for per-clip capture sidecars."""

from __future__ import annotations

from pathlib import Path

from .court_calibration import load_capture_sidecar as _load_capture_sidecar_file
from .schemas import CaptureSidecar

SIDECAR_FILENAME = "capture_sidecar.json"


def load_capture_sidecar(clip_or_path: str | Path) -> CaptureSidecar:
    """Load a C1 capture sidecar from a clip directory or sidecar JSON path."""
    sidecar_path = _resolve_sidecar_path(clip_or_path)
    return _load_capture_sidecar_file(sidecar_path)


def _resolve_sidecar_path(clip_or_path: str | Path) -> Path:
    path = Path(clip_or_path)
    if path.is_dir():
        return path / SIDECAR_FILENAME
    return path
