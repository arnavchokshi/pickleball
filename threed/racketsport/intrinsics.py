"""Camera intrinsic lookup helpers for racket-sport clips."""

from __future__ import annotations

from pathlib import Path

from .schemas import CameraIntrinsics
from .sidecar import load_capture_sidecar


def get_intrinsics(clip_or_path: str | Path) -> CameraIntrinsics:
    """Return measured camera intrinsics from the clip capture sidecar."""
    return load_capture_sidecar(clip_or_path).intrinsics
