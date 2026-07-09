"""Best-stack manifest loader and typed default accessors.

The manifest records default selections. It does not promote or verify any
capability; gates remain authoritative elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEST_STACK_PATH = REPO_ROOT / "configs" / "racketsport" / "best_stack.json"
VALID_STATUSES = {"WIRED_DEFAULT", "PENDING", "DORMANT", "FENCED"}
REQUIRED_ENTRY_FIELDS = {
    "stage",
    "selection",
    "value",
    "status",
    "gate",
    "provenance",
    "proven_against",
    "notes",
}


class BestStackManifestError(ValueError):
    """Raised when best_stack.json is missing, malformed, or stale."""


@dataclass(frozen=True)
class BestStackEntry:
    key: str
    stage: str
    selection: str
    value: Any
    status: str
    gate: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    proven_against: Mapping[str, Any] | None
    notes: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class BestStackManifest:
    path: Path
    raw: Mapping[str, Any]
    entries: Mapping[str, BestStackEntry]

    @property
    def schema_version(self) -> int:
        return int(self.raw["schema_version"])

    @property
    def revision(self) -> int:
        return int(self.raw["revision"])

    @property
    def updated(self) -> str:
        return str(self.raw["updated"])

    @property
    def invariants(self) -> list[str]:
        return [str(item) for item in self.raw.get("invariants", [])]

    def entry(self, key: str) -> BestStackEntry:
        try:
            return self.entries[key]
        except KeyError as exc:
            raise BestStackManifestError(f"best_stack entry {key!r} missing") from exc

    def value(self, key: str) -> Any:
        return self.entry(key).value

    def string_value(self, key: str) -> str:
        value = self.value(key)
        if not isinstance(value, str):
            raise BestStackManifestError(f"best_stack entry {key!r} is not a string")
        return value

    def number_value(self, key: str) -> float:
        value = self.value(key)
        if not isinstance(value, int | float):
            raise BestStackManifestError(f"best_stack entry {key!r} is not numeric")
        return float(value)

    def path_value(self, key: str, *, must_exist: bool = True) -> Path:
        value = self.value(key)
        if isinstance(value, str):
            path_text = value
            sha256 = None
            required = True
        elif isinstance(value, Mapping):
            path_raw = value.get("path")
            if not isinstance(path_raw, str) or not path_raw:
                raise BestStackManifestError(f"best_stack entry {key!r} has no value.path")
            path_text = path_raw
            sha256 = value.get("sha256")
            required = bool(value.get("required", True))
        else:
            raise BestStackManifestError(f"best_stack entry {key!r} is not path-like")

        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if must_exist and required and not path.exists():
            raise BestStackManifestError(f"best_stack entry {key!r} points at missing path {path}")
        if must_exist and required and sha256 is not None:
            if not path.is_file():
                raise BestStackManifestError(f"best_stack entry {key!r} sha256 path is not a file: {path}")
            actual = _sha256(path)
            if actual != str(sha256):
                raise BestStackManifestError(
                    f"best_stack entry {key!r} sha256 mismatch for {path}: expected {sha256}, got {actual}"
                )
        return path

    def server_override_value(self, key: str) -> Any:
        overrides = self.raw.get("server_overrides")
        if not isinstance(overrides, Mapping):
            raise BestStackManifestError("best_stack server_overrides must be an object")
        override = overrides.get(key)
        if not isinstance(override, Mapping) or "value" not in override:
            raise BestStackManifestError(f"best_stack server override {key!r} missing")
        return override["value"]

    def body_detector_fov_defaults(self) -> tuple[str, str]:
        value = self.value("body.detector_fov")
        if not isinstance(value, Mapping):
            raise BestStackManifestError("body.detector_fov value must be an object")
        detector = value.get("detector_name")
        fov = value.get("fov_name")
        if not isinstance(detector, str) or not isinstance(fov, str):
            raise BestStackManifestError("body.detector_fov detector_name/fov_name must be strings")
        return detector, fov


_DEFAULT_MANIFEST: BestStackManifest | None = None


def load_best_stack_manifest(path: str | Path = DEFAULT_BEST_STACK_PATH) -> BestStackManifest:
    manifest_path = Path(path)
    global _DEFAULT_MANIFEST
    if manifest_path == DEFAULT_BEST_STACK_PATH and _DEFAULT_MANIFEST is not None:
        return _DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise BestStackManifestError(f"best_stack manifest missing: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BestStackManifestError(f"best_stack manifest is not valid JSON: {manifest_path}") from exc
    manifest = _build_manifest(manifest_path, raw)
    if manifest_path == DEFAULT_BEST_STACK_PATH:
        _DEFAULT_MANIFEST = manifest
    return manifest


def body_detector_fov_defaults() -> tuple[str, str]:
    return load_best_stack_manifest().body_detector_fov_defaults()


def server_override_value(key: str) -> Any:
    return load_best_stack_manifest().server_override_value(key)


def _build_manifest(path: Path, raw: Any) -> BestStackManifest:
    if not isinstance(raw, Mapping):
        raise BestStackManifestError("best_stack manifest must be an object")
    for field in ("schema_version", "revision", "updated", "invariants", "server_overrides", "entries"):
        if field not in raw:
            raise BestStackManifestError(f"best_stack manifest missing top-level field {field!r}")
    if raw["schema_version"] != 1:
        raise BestStackManifestError("best_stack schema_version must equal 1")
    if not isinstance(raw["revision"], int) or raw["revision"] <= 0:
        raise BestStackManifestError("best_stack revision must be a positive integer")
    if "A manifest entry is a DEFAULT selection, NEVER a VERIFIED claim" not in raw.get("invariants", []):
        raise BestStackManifestError("best_stack invariants must include the default-not-verified sentence")

    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, Mapping) or not raw_entries:
        raise BestStackManifestError("best_stack entries must be a non-empty object")
    entries: dict[str, BestStackEntry] = {}
    for key, item in raw_entries.items():
        if not isinstance(key, str) or not key:
            raise BestStackManifestError("best_stack entry keys must be non-empty strings")
        if not isinstance(item, Mapping):
            raise BestStackManifestError(f"best_stack entry {key!r} must be an object")
        missing = REQUIRED_ENTRY_FIELDS - set(item)
        if missing:
            raise BestStackManifestError(f"best_stack entry {key!r} missing fields: {sorted(missing)}")
        status = item["status"]
        if status not in VALID_STATUSES:
            raise BestStackManifestError(f"best_stack entry {key!r} has invalid status {status!r}")
        gate = item["gate"]
        if status == "PENDING" and gate is None:
            raise BestStackManifestError(f"best_stack entry {key!r} is PENDING without a gate")
        if status == "DORMANT" and "ruling" not in str(item["notes"]).lower():
            raise BestStackManifestError(f"best_stack entry {key!r} is DORMANT without a ruling note")
        provenance = item["provenance"]
        if not isinstance(provenance, Mapping):
            raise BestStackManifestError(f"best_stack entry {key!r} provenance must be an object")
        for field in ("lane", "commit", "date", "evidence_paths"):
            if field not in provenance:
                raise BestStackManifestError(f"best_stack entry {key!r} provenance missing {field!r}")
        entries[key] = BestStackEntry(
            key=key,
            stage=str(item["stage"]),
            selection=str(item["selection"]),
            value=item["value"],
            status=str(status),
            gate=gate if isinstance(gate, Mapping) else None,
            provenance=provenance,
            proven_against=item["proven_against"] if isinstance(item["proven_against"], Mapping) else None,
            notes=str(item["notes"]),
            raw=item,
        )

    manifest = BestStackManifest(path=path, raw=raw, entries=entries)
    for key in entries:
        _validate_path_value(manifest, key)
    _validate_server_overrides(manifest)
    return manifest


def _validate_path_value(manifest: BestStackManifest, key: str) -> None:
    value = manifest.value(key)
    if not isinstance(value, Mapping):
        return
    kind = value.get("kind")
    if kind not in {"local_path", "repo_path", "path"}:
        return
    if value.get("required", True) is False:
        return
    manifest.path_value(key)


def _validate_server_overrides(manifest: BestStackManifest) -> None:
    overrides = manifest.raw.get("server_overrides")
    if not isinstance(overrides, Mapping):
        raise BestStackManifestError("best_stack server_overrides must be an object")
    for key, override in overrides.items():
        if not isinstance(override, Mapping) or "value" not in override:
            raise BestStackManifestError(f"best_stack server override {key!r} must contain value")
        entry_key = override.get("entry")
        if isinstance(entry_key, str) and entry_key:
            manifest.entry(entry_key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
