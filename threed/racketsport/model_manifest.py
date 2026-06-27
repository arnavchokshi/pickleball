from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ModelStatus = Literal[
    "available_on_h100",
    "pending_download",
    "pending_auth",
    "pending_benchmark",
    "fallback_only",
]

CommercialPosture = Literal["ok", "research_ok_verify_commercial", "agpl_caveat", "unknown", "avoid"]


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
