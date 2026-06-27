from __future__ import annotations

import json
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


class TestClipMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    camera_height: Literal["low", "mid", "high"]
    camera_angle: Literal["shallow_baseline", "steep_corner", "side_fence", "near_overhead"]
    play_type: Literal["doubles", "singles_drill", "messy_real_world"]
    environment: Literal["indoor", "outdoor"]
    frame_rate_fps: int
    duration_s: float
    racket_gt: bool = False


class TestClipManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    path: Path
    labels_dir: Path
    metadata_path: Path
    metadata: TestClipMetadata | None = None
    metadata_errors: list[str] = Field(default_factory=list)
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

    @computed_field
    @property
    def metadata_present(self) -> bool:
        return self.metadata is not None


class TestClipDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    root: Path
    root_exists: bool
    required_label_files: tuple[str, ...] = Field(default=REQUIRED_LABEL_FILES)
    label_file_counts: dict[str, int]
    coverage_counts: dict[str, dict[str, int]]
    coverage_gaps: list[str]
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

    @computed_field
    @property
    def metadata_ready_clips(self) -> int:
        return sum(1 for clip in self.clips if clip.metadata is not None and not clip.metadata_errors)

    @computed_field
    @property
    def meets_dataset_matrix(self) -> bool:
        return not self.coverage_gaps

    @computed_field
    @property
    def dataset_ready(self) -> bool:
        return self.is_ready and self.meets_dataset_matrix


def build_clip_manifest(clip_dir: str | Path) -> TestClipManifest:
    clip_path = Path(clip_dir)
    labels_dir = clip_path / "labels"
    metadata_path = clip_path / "clip_metadata.json"
    metadata: TestClipMetadata | None = None
    metadata_errors: list[str] = []
    if metadata_path.is_file():
        try:
            metadata = TestClipMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
        except Exception as exc:  # pydantic exposes structured errors; manifest keeps them human-readable.
            metadata_errors.append(str(exc))

    present = [label for label in REQUIRED_LABEL_FILES if (labels_dir / label).is_file()]
    missing = [label for label in REQUIRED_LABEL_FILES if label not in present]
    return TestClipManifest(
        name=clip_path.name,
        path=clip_path,
        labels_dir=labels_dir,
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_errors=metadata_errors,
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
    coverage_counts = _coverage_counts(clips)
    coverage_gaps = _coverage_gaps(clips, coverage_counts)
    return TestClipDatasetManifest(
        root=root_path,
        root_exists=root_path.exists(),
        label_file_counts=label_counts,
        coverage_counts=coverage_counts,
        coverage_gaps=coverage_gaps,
        clips=clips,
    )


def _coverage_counts(clips: list[TestClipManifest]) -> dict[str, dict[str, int]]:
    counts = {
        "camera_height": {"low": 0, "mid": 0, "high": 0},
        "camera_angle": {"shallow_baseline": 0, "steep_corner": 0, "side_fence": 0, "near_overhead": 0},
        "play_type": {"doubles": 0, "singles_drill": 0, "messy_real_world": 0},
        "environment": {"indoor": 0, "outdoor": 0},
        "frame_rate": {"fps_120_or_higher": 0, "fps_240_or_higher": 0},
        "length": {"short_60_120_s": 0, "long_15_20_min": 0},
        "racket_gt": {"true": 0},
    }
    for clip in clips:
        metadata = clip.metadata
        if metadata is None or clip.metadata_errors:
            continue
        counts["camera_height"][metadata.camera_height] += 1
        counts["camera_angle"][metadata.camera_angle] += 1
        counts["play_type"][metadata.play_type] += 1
        counts["environment"][metadata.environment] += 1
        if metadata.frame_rate_fps >= 120:
            counts["frame_rate"]["fps_120_or_higher"] += 1
        if metadata.frame_rate_fps >= 240:
            counts["frame_rate"]["fps_240_or_higher"] += 1
        if 60.0 <= metadata.duration_s <= 120.0:
            counts["length"]["short_60_120_s"] += 1
        if 900.0 <= metadata.duration_s <= 1200.0:
            counts["length"]["long_15_20_min"] += 1
        if metadata.racket_gt:
            counts["racket_gt"]["true"] += 1
    return counts


def _coverage_gaps(clips: list[TestClipManifest], counts: dict[str, dict[str, int]]) -> list[str]:
    valid_metadata_count = sum(1 for clip in clips if clip.metadata is not None and not clip.metadata_errors)
    gaps: list[str] = []
    if valid_metadata_count < 24:
        gaps.append("need at least 24 clips with valid metadata")

    for value, count in counts["camera_height"].items():
        if count < 1:
            gaps.append(f"need camera_height={value}")
    for value, count in counts["camera_angle"].items():
        if count < 1:
            gaps.append(f"need camera_angle={value}")

    minimums = {
        ("play_type", "doubles"): 10,
        ("play_type", "singles_drill"): 6,
        ("play_type", "messy_real_world"): 4,
        ("environment", "indoor"): 8,
        ("environment", "outdoor"): 8,
        ("frame_rate", "fps_120_or_higher"): 6,
        ("frame_rate", "fps_240_or_higher"): 2,
        ("length", "short_60_120_s"): 12,
        ("length", "long_15_20_min"): 4,
        ("racket_gt", "true"): 3,
    }
    for (category, value), minimum in minimums.items():
        actual = counts[category][value]
        if actual < minimum:
            gaps.append(f"need {category}.{value} >= {minimum} (found {actual})")
    return gaps
