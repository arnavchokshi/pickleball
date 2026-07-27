"""Promotion pointer for per-clip court calibration artifacts.

Background
----------
Court calibration artifacts are resolved by *filename convention*: every consumer
looks for ``<clip>/labels/court_calibration_metric15pt.json`` (see
``scripts/racketsport/process_video._auto_discover_court_calibration`` and
``threed.racketsport.virtual_world.resolve_best_court_calibration_path``). There
was no way to say "a better fit of the same owner-reviewed correspondences is now
the input" without overwriting the raw solve, and raw solves are immutable.

This module adds the missing indirection. A clip's label directory may carry a
``court_calibration_selected.json`` pointer declaring that some *other* artifact
is the selected input. The raw solve stays on disk, byte-for-byte, and the
pointer records both sha256 digests so the promotion is a checksummed, auditable
decision rather than a silently-dropped file.

What a pointer is NOT
---------------------
* It is **not** an authority escalation. ``source`` / ``intrinsics.source`` are
  unchanged by promotion, so ``orchestrator.ExternalCalibrationRunner``'s
  ``TRUSTED_INTRINSICS_SOURCES`` gate sees exactly what it saw before. A better
  fit of the same reviewed correspondences is not a new authority class.
* It is **not** a verification. Promotion is an engineering improvement to a fit;
  ``verified`` stays 0 and no capability gate is claimed.

Failure mode
------------
Deliberately fail-loud. A pointer whose target is missing, whose digest does not
match, or which does not supersede the artifact it sits beside raises rather than
silently falling back to the superseded raw solve: a broken promotion must not
degrade into a quietly different calibration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SELECTION_POINTER_FILENAME",
    "SELECTION_ARTIFACT_TYPE",
    "CourtCalibrationSelectionError",
    "file_sha256",
    "read_selection_pointer",
    "resolve_selected_calibration_path",
    "selection_pointer_for",
]

#: Sits beside the raw solve it supersedes, in the same directory.
SELECTION_POINTER_FILENAME = "court_calibration_selected.json"

SELECTION_ARTIFACT_TYPE = "racketsport_court_calibration_selection"

SELECTION_SCHEMA_VERSION = 1

_REQUIRED_TOP_LEVEL = ("schema_version", "artifact_type", "selected", "supersedes", "authority")
_REQUIRED_REF_FIELDS = ("path", "sha256")


class CourtCalibrationSelectionError(ValueError):
    """A selection pointer exists but cannot be trusted."""


def selection_pointer_for(calibration_path: str | Path) -> Path:
    """Where the pointer superseding ``calibration_path`` would live."""

    return Path(calibration_path).parent / SELECTION_POINTER_FILENAME


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_reference(reference: Mapping[str, Any], *, pointer_path: Path, field: str) -> Path:
    """Resolve a pointer reference, which is always relative to the pointer itself.

    Paths are deliberately confined to the pointer's own directory: a promotion is
    a local, self-contained decision about one clip's labels, and a pointer that
    could reach anywhere in the tree would be a much larger authority than the
    thing it replaces.
    """

    for required in _REQUIRED_REF_FIELDS:
        if required not in reference:
            raise CourtCalibrationSelectionError(
                f"{pointer_path}: {field}.{required} is required"
            )
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise CourtCalibrationSelectionError(f"{pointer_path}: {field}.path must be a non-empty string")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: {field}.path must be relative to the pointer, got absolute {candidate}"
        )
    if ".." in candidate.parts:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: {field}.path must stay inside the pointer's own directory: {raw_path}"
        )
    resolved = pointer_path.parent / candidate
    if not resolved.is_file():
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: {field}.path does not exist: {raw_path}"
        )
    return resolved


def read_selection_pointer(pointer_path: str | Path) -> dict[str, Any]:
    """Parse and structurally validate a selection pointer.

    Does not check digests -- see :func:`resolve_selected_calibration_path`.
    """

    pointer_path = Path(pointer_path)
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CourtCalibrationSelectionError(f"{pointer_path}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CourtCalibrationSelectionError(f"{pointer_path}: expected a JSON object")
    for field in _REQUIRED_TOP_LEVEL:
        if field not in payload:
            raise CourtCalibrationSelectionError(f"{pointer_path}: {field} is required")
    if payload["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: unsupported schema_version {payload['schema_version']!r}"
        )
    if payload["artifact_type"] != SELECTION_ARTIFACT_TYPE:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: artifact_type must be {SELECTION_ARTIFACT_TYPE!r}"
        )
    for field in ("selected", "supersedes"):
        if not isinstance(payload[field], Mapping):
            raise CourtCalibrationSelectionError(f"{pointer_path}: {field} must be an object")
    authority = payload["authority"]
    if not isinstance(authority, Mapping) or "class_unchanged" not in authority:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: authority.class_unchanged is required -- promotion never "
            "escalates the authority class"
        )
    if authority["class_unchanged"] is not True:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: authority.class_unchanged must be true; a promotion that "
            "changes the authority class is a source-class escalation, not a promotion"
        )
    return payload


def resolve_selected_calibration_path(calibration_path: str | Path) -> Path:
    """Return the artifact that should actually be read for ``calibration_path``.

    ``calibration_path`` is the conventional artifact a consumer already found
    (typically ``<clip>/labels/court_calibration_metric15pt.json``). If a
    validated pointer sits beside it, the pointer's target is returned; otherwise
    ``calibration_path`` is returned unchanged.

    Passing the pointer itself is also supported, so an explicit
    ``--court-calibration <pointer>`` resolves rather than failing downstream.
    """

    calibration_path = Path(calibration_path)
    if calibration_path.name == SELECTION_POINTER_FILENAME:
        pointer_path = calibration_path
    else:
        pointer_path = selection_pointer_for(calibration_path)
        if not pointer_path.is_file():
            return calibration_path

    payload = read_selection_pointer(pointer_path)
    selected = _resolve_reference(payload["selected"], pointer_path=pointer_path, field="selected")
    superseded = _resolve_reference(
        payload["supersedes"], pointer_path=pointer_path, field="supersedes"
    )

    expected_selected = str(payload["selected"]["sha256"])
    actual_selected = file_sha256(selected)
    if actual_selected != expected_selected:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: selected artifact digest mismatch for {selected}: "
            f"expected {expected_selected}, found {actual_selected}"
        )

    expected_superseded = str(payload["supersedes"]["sha256"])
    actual_superseded = file_sha256(superseded)
    if actual_superseded != expected_superseded:
        raise CourtCalibrationSelectionError(
            f"{pointer_path}: superseded raw solve digest mismatch for {superseded}: "
            f"expected {expected_superseded}, found {actual_superseded}. The raw solve is "
            "immutable; a changed digest means it was edited."
        )

    if calibration_path.name != SELECTION_POINTER_FILENAME:
        if superseded.resolve() != calibration_path.resolve():
            raise CourtCalibrationSelectionError(
                f"{pointer_path}: supersedes {superseded}, but sits beside {calibration_path}. "
                "A pointer may only supersede the artifact in its own directory."
            )

    return selected
