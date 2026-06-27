from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


REQUIRED_LABEL_FILES: tuple[str, ...] = (
    "court_corners.json",
    "players.json",
    "feet_nvz.json",
    "ball.json",
    "events.json",
    "racket_pose.json",
    "foot_contact.json",
    "coach_habits.json",
    "manual_metrics.json",
)


class TestClipManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    path: Path
    labels_dir: Path
    required_label_files: tuple[str, ...] = Field(default=REQUIRED_LABEL_FILES)
    present_label_files: list[str] = Field(default_factory=list)
    missing_label_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _label_lists_match_required_files(self) -> "TestClipManifest":
        required = set(self.required_label_files)
        present = set(self.present_label_files)
        missing = set(self.missing_label_files)
        unexpected = sorted((present | missing) - required)
        if unexpected:
            raise ValueError(f"label status contains unexpected files: {', '.join(unexpected)}")
        if present & missing:
            overlap = ", ".join(sorted(present & missing))
            raise ValueError(f"label files cannot be both present and missing: {overlap}")
        if present | missing != required:
            omitted = ", ".join(sorted(required - present - missing))
            raise ValueError(f"label status omitted required files: {omitted}")
        return self

    @computed_field
    @property
    def is_ready(self) -> bool:
        return not self.missing_label_files


class TestClipDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    root: Path
    root_exists: bool
    required_label_files: tuple[str, ...] = Field(default=REQUIRED_LABEL_FILES)
    label_file_counts: dict[str, int]
    clips: list[TestClipManifest] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_cover_required_labels(self) -> "TestClipDatasetManifest":
        required = set(self.required_label_files)
        unexpected = sorted(set(self.label_file_counts) - required)
        if unexpected:
            raise ValueError(f"label counts contain unexpected files: {', '.join(unexpected)}")
        missing_counts = sorted(required - set(self.label_file_counts))
        if missing_counts:
            raise ValueError(f"label counts omitted required files: {', '.join(missing_counts)}")
        return self

    @computed_field
    @property
    def total_clips(self) -> int:
        return len(self.clips)

    @computed_field
    @property
    def ready_clips(self) -> int:
        return sum(1 for clip in self.clips if clip.is_ready)

    @computed_field
    @property
    def not_ready_clips(self) -> int:
        return self.total_clips - self.ready_clips

    @computed_field
    @property
    def is_ready(self) -> bool:
        return self.root_exists and self.total_clips > 0 and self.not_ready_clips == 0


def build_clip_manifest(clip_dir: str | Path) -> TestClipManifest:
    clip_path = Path(clip_dir)
    labels_dir = clip_path / "labels"
    present = [label for label in REQUIRED_LABEL_FILES if (labels_dir / label).is_file()]
    missing = [label for label in REQUIRED_LABEL_FILES if label not in present]
    return TestClipManifest(
        name=clip_path.name,
        path=clip_path,
        labels_dir=labels_dir,
        present_label_files=present,
        missing_label_files=missing,
    )


def build_testclip_manifest(root: str | Path) -> TestClipDatasetManifest:
    root_path = Path(root)
    clips: list[TestClipManifest] = []
    if root_path.exists():
        discovered = [
            build_clip_manifest(path)
            for path in sorted(root_path.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
        clips = sorted(discovered, key=lambda clip: (not clip.is_ready, clip.name))

    label_counts = {
        label: sum(1 for clip in clips if label in clip.present_label_files)
        for label in REQUIRED_LABEL_FILES
    }
    return TestClipDatasetManifest(
        root=root_path,
        root_exists=root_path.exists(),
        label_file_counts=label_counts,
        clips=clips,
    )
