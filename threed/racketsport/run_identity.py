"""Content-addressed stage identity and transactional stage generations.

The store deliberately separates immutable stage generations from the pipeline's
legacy flat artifact layout.  A generation is prepared under ``transactions/``
and becomes visible only after an atomic rename plus an atomic ``current`` pointer
update.  Flat artifacts may be registered as external artifacts; their hashes are
rechecked before reuse, so a partial or out-of-band mutation is never reusable.

Migration policy is fail-stale: an existing run without an identity manifest is
valid legacy data, but it is not eligible for automatic reuse.  The next successful
stage execution writes the first manifest without requiring manual cleanup.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_stage_identity"
STORE_DIR_NAME = ".run_identity"
_CHUNK_SIZE = 1024 * 1024


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_stage_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid stage name: {value!r}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    """Return a content identity for a file, symlink target, directory, or miss."""

    candidate = Path(path)
    if candidate.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(candidate),
            "size": candidate.stat().st_size,
        }
    if candidate.is_dir():
        rows: list[dict[str, Any]] = []
        total_size = 0
        for child in sorted(item for item in candidate.rglob("*") if item.is_file()):
            identity = fingerprint_path(child)
            total_size += int(identity["size"])
            rows.append({"path": child.relative_to(candidate).as_posix(), **identity})
        return {
            "kind": "directory",
            "sha256": _sha256_bytes(_canonical_json(rows)),
            "size": total_size,
            "file_count": len(rows),
        }
    return {"kind": "missing", "sha256": None, "size": 0}


@dataclass(frozen=True)
class SourceIdentity:
    sha256: str
    size: int
    timing: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path, *, timing: Mapping[str, Any] | None = None) -> "SourceIdentity":
        identity = fingerprint_path(path)
        if identity["kind"] != "file":
            raise FileNotFoundError(f"source video is not a file: {path}")
        return cls(
            sha256=str(identity["sha256"]),
            size=int(identity["size"]),
            timing=dict(timing or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size, "timing": dict(self.timing)}


@dataclass(frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...] = ()
    code: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    models: Mapping[str, Path] = field(default_factory=dict)
    explicit_inputs: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _safe_stage_name(self.name)
        if self.name in self.dependencies:
            raise ValueError(f"stage {self.name!r} cannot depend on itself")
        for dependency in self.dependencies:
            _safe_stage_name(dependency)


@dataclass(frozen=True)
class ReuseDecision:
    reusable: bool
    reason: str
    fingerprint: str
    identity: Mapping[str, Any]
    manifest: Mapping[str, Any] | None = None


class StageTransaction(AbstractContextManager[Path]):
    """Prepare one immutable stage generation and publish it on clean exit."""

    def __init__(
        self,
        store: "RunIdentityStore",
        spec: StageSpec,
        source: SourceIdentity,
        *,
        external_artifacts: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.spec = spec
        self.source = source
        self.external_artifacts = tuple(external_artifacts)
        self.metadata = dict(metadata or {})
        self.identity, self.fingerprint = store.stage_identity(spec, source)
        store.transactions_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix=f"{spec.name}-", dir=store.transactions_dir)
        )
        self._entered = False

    def __enter__(self) -> Path:
        self._entered = True
        return self.temp_dir

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            return False
        try:
            self._publish()
        except Exception:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise
        return False

    def _publish(self) -> None:
        if not self._entered:
            raise RuntimeError("stage transaction was not entered")
        managed = self._managed_artifact_rows()
        external = [self.store.external_artifact_row(path) for path in self.external_artifacts]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "stage": self.spec.name,
            "fingerprint": self.fingerprint,
            "identity": self.identity,
            "managed_artifacts": managed,
            "external_artifacts": external,
            "metadata": self.metadata,
        }
        self.store._write_json_file(self.temp_dir / "manifest.json", manifest)

        stage_root = self.store.stages_dir / self.spec.name
        stage_root.mkdir(parents=True, exist_ok=True)
        generation_name = f"{self.fingerprint}-{uuid.uuid4().hex}"
        generation_dir = stage_root / generation_name
        os.replace(self.temp_dir, generation_dir)

        pointer = {
            "schema_version": SCHEMA_VERSION,
            "stage": self.spec.name,
            "fingerprint": self.fingerprint,
            "generation": generation_dir.relative_to(self.store.store_dir).as_posix(),
        }
        current_path = self.store.current_dir / f"{self.spec.name}.json"
        self.store._atomic_write_json(current_path, pointer)

    def _managed_artifact_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(item for item in self.temp_dir.rglob("*") if item.is_file()):
            if path.name == "manifest.json":
                continue
            rows.append({
                "path": path.relative_to(self.temp_dir).as_posix(),
                "identity": fingerprint_path(path),
            })
        return rows


class RunIdentityStore:
    """Compute per-stage fingerprints and own their transactional manifests."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.store_dir = self.run_dir / STORE_DIR_NAME
        self.stages_dir = self.store_dir / "stages"
        self.current_dir = self.store_dir / "current"
        self.transactions_dir = self.store_dir / "transactions"

    def stage_identity(self, spec: StageSpec, source: SourceIdentity) -> tuple[dict[str, Any], str]:
        upstream: dict[str, Any] = {}
        for dependency in spec.dependencies:
            manifest = self.current_manifest(dependency)
            if manifest is None:
                upstream[dependency] = {"status": "missing"}
                continue
            upstream[dependency] = {
                "fingerprint": manifest.get("fingerprint"),
                "artifact_digest": self._artifact_digest(manifest),
            }
        identity = {
            "schema_version": SCHEMA_VERSION,
            "stage": spec.name,
            "source": source.as_dict(),
            "code": dict(spec.code),
            "config": dict(spec.config),
            "models": {name: fingerprint_path(path) for name, path in sorted(spec.models.items())},
            "explicit_inputs": {
                name: fingerprint_path(path) for name, path in sorted(spec.explicit_inputs.items())
            },
            "upstream": upstream,
        }
        return identity, _sha256_bytes(_canonical_json(identity))

    def decision(self, spec: StageSpec, source: SourceIdentity, *, force: bool = False) -> ReuseDecision:
        identity, fingerprint = self.stage_identity(spec, source)
        if force:
            return ReuseDecision(False, "force_requested", fingerprint, identity)
        manifest = self.current_manifest(spec.name)
        if manifest is None:
            return ReuseDecision(False, "unfingerprinted_stale", fingerprint, identity)
        if manifest.get("fingerprint") != fingerprint:
            return ReuseDecision(False, "fingerprint_mismatch", fingerprint, identity, manifest)
        if not self.manifest_artifacts_valid(manifest):
            return ReuseDecision(False, "artifact_hash_mismatch", fingerprint, identity, manifest)
        return ReuseDecision(True, "identical_fingerprint", fingerprint, identity, manifest)

    def transaction(
        self,
        spec: StageSpec,
        source: SourceIdentity,
        *,
        external_artifacts: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> StageTransaction:
        return StageTransaction(
            self,
            spec,
            source,
            external_artifacts=external_artifacts,
            metadata=metadata,
        )

    def current_manifest(self, stage: str) -> dict[str, Any] | None:
        stage_name = _safe_stage_name(stage)
        pointer_path = self.current_dir / f"{stage_name}.json"
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            generation = pointer.get("generation")
            if not isinstance(generation, str):
                return None
            generation_dir = self.store_dir / generation
            manifest = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if manifest.get("artifact_type") != ARTIFACT_TYPE or manifest.get("stage") != stage_name:
            return None
        manifest["_generation_dir"] = str(generation_dir)
        return manifest

    def current_stage_reusable(self, stage: str) -> bool:
        manifest = self.current_manifest(stage)
        return manifest is not None and self.manifest_artifacts_valid(manifest)

    def manifest_artifacts_valid(self, manifest: Mapping[str, Any]) -> bool:
        generation_value = manifest.get("_generation_dir")
        if not isinstance(generation_value, str):
            return False
        generation_dir = Path(generation_value)
        for row in manifest.get("managed_artifacts", []):
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                return False
            if fingerprint_path(generation_dir / str(row["path"])) != row.get("identity"):
                return False
        for row in manifest.get("external_artifacts", []):
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                return False
            path = self._external_path(str(row["path"]), bool(row.get("absolute")))
            if fingerprint_path(path) != row.get("identity"):
                return False
        return True

    def external_artifact_row(self, path: str | Path) -> dict[str, Any]:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.run_dir / candidate
        try:
            stored_path = candidate.relative_to(self.run_dir).as_posix()
            absolute = False
        except ValueError:
            stored_path = str(candidate)
            absolute = True
        return {
            "path": stored_path,
            "absolute": absolute,
            "identity": fingerprint_path(candidate),
        }

    def _external_path(self, value: str, absolute: bool) -> Path:
        return Path(value) if absolute else self.run_dir / value

    @staticmethod
    def _artifact_digest(manifest: Mapping[str, Any]) -> str:
        payload = {
            "managed": manifest.get("managed_artifacts", []),
            "external": manifest.get("external_artifacts", []),
        }
        return _sha256_bytes(_canonical_json(payload))

    @staticmethod
    def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _atomic_write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._write_json_file(temp_path, value)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


__all__ = [
    "ARTIFACT_TYPE",
    "RunIdentityStore",
    "ReuseDecision",
    "SCHEMA_VERSION",
    "STORE_DIR_NAME",
    "SourceIdentity",
    "StageSpec",
    "fingerprint_path",
    "sha256_file",
]
