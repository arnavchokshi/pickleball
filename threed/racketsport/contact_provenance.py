"""Versioned provenance helpers for coarse and post-BODY contact artifacts.

The canonical ``ContactWindows`` schema is intentionally kept unchanged.  Its
content hashes and timing lineage live in explicit companion artifacts so raw
contact proposals remain byte-immutable and schema-valid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


PROVENANCE_SCHEMA_VERSION = 1
DEPENDENCY_HASH_MISMATCH = "contact_dependency_hash_mismatch"


def file_dependency(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "sha256": None, "status": "missing"}
    if not path.is_file():
        return {"path": str(path), "sha256": None, "status": "missing"}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "present",
    }


def dependency_snapshot(paths: Mapping[str, Path | None]) -> dict[str, dict[str, Any]]:
    return {name: file_dependency(path) for name, path in paths.items()}


def dependency_hashes_match(
    recorded: Any,
    current: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not isinstance(recorded, Mapping):
        return False
    if set(recorded) != set(current):
        return False
    for name, value in current.items():
        prior = recorded.get(name)
        if not isinstance(prior, Mapping):
            return False
        if prior.get("sha256") != value.get("sha256") or prior.get("status") != value.get("status"):
            return False
    return True


def audio_provenance(payload: Any, *, source_path: Path | None) -> dict[str, Any]:
    mapping = payload if isinstance(payload, Mapping) else {}
    raw_items = mapping.get("onsets")
    items = raw_items if isinstance(raw_items, list) else []
    timing: list[dict[str, float]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        corrected = item.get("corrected_time_s", item.get("time_s"))
        raw = item.get("raw_time_s", item.get("time_s"))
        if raw is None or corrected is None:
            continue
        timing.append({"raw_time_s": float(raw), "corrected_time_s": float(corrected)})

    blockers = [str(value) for value in mapping.get("blockers", [])] if isinstance(mapping.get("blockers"), list) else []
    if blockers:
        status = "absent"
        reason = blockers[0]
    elif source_path is None or not source_path.is_file():
        status = "absent"
        reason = "no_audio_stream"
    elif timing:
        status = "present"
        reason = None
    else:
        status = "absent"
        reason = "no_audio_onset_events"
    return {
        "source": str(source_path) if source_path is not None else None,
        "status": status,
        "reason": reason,
        "timing": timing,
        "timing_policy": "raw_and_corrected_preserved_no_destructive_correction",
    }


__all__ = [
    "DEPENDENCY_HASH_MISMATCH",
    "PROVENANCE_SCHEMA_VERSION",
    "audio_provenance",
    "dependency_hashes_match",
    "dependency_snapshot",
    "file_dependency",
]
