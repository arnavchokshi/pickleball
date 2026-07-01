from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ModelStatus = Literal[
    "available_on_h100",
    "available_runtime_on_h100",
    "downloadable_local_checkpoint",
    "pending_download",
    "pending_auth",
    "pending_benchmark",
    "fallback_only",
]

CommercialPosture = Literal[
    "ok",
    "research_ok_verify_commercial",
    "research_ok_verify_checkpoint_terms",
    "agpl_caveat",
    "unknown",
    "avoid",
]


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stage: str
    use: str
    source: str
    license: str
    commercial_posture: CommercialPosture
    status: ModelStatus
    local_path: str | None = None
    sha256: str | None = None
    repo_commit: str | None = None
    fallbacks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _available_entries_are_falsifiable(self) -> "ModelEntry":
        if self.status == "available_on_h100":
            missing = []
            if not self.local_path:
                missing.append("local_path")
            if not self.sha256:
                missing.append("sha256")
            if missing:
                raise ValueError(f"available_on_h100 entries require {', '.join(missing)}")
        return self


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    models: list[ModelEntry]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "ModelManifest":
        ids = [entry.id for entry in self.models]
        duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate model ids: {', '.join(duplicates)}")
        return self


def load_model_manifest(path: str | Path) -> ModelManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ModelManifest.model_validate(json.load(handle))


def get_model_entry(manifest: ModelManifest, model_id: str) -> ModelEntry:
    for entry in manifest.models:
        if entry.id == model_id:
            return entry
    raise KeyError(f"model id not found in manifest: {model_id}")


def verify_model_checkpoint(path: str | Path, model_id: str) -> ModelEntry:
    manifest = load_model_manifest(path)
    entry = get_model_entry(manifest, model_id)
    if entry.status != "available_on_h100":
        raise RuntimeError(f"model {model_id} is not available_on_h100: status={entry.status}")
    if not entry.local_path:
        raise RuntimeError(f"model {model_id} has no local_path in manifest")
    checkpoint = Path(entry.local_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint for {model_id}: {checkpoint}")
    if not entry.sha256:
        raise RuntimeError(f"model {model_id} has no sha256 in manifest")

    digest = _sha256_file(checkpoint)
    if digest.lower() != entry.sha256.lower():
        raise ValueError(f"sha256 mismatch for {model_id}: expected {entry.sha256}, got {digest}")
    return entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
