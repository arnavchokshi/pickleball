"""CVAT MOT-format person annotation import for mobile tracking evaluation."""

from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import PersonGroundTruth, PersonGroundTruthFrame, PersonGroundTruthSummary, PersonLabel

PERSON_CLASS_NAMES = {"person", "player"}


def import_mot_zip(path: str | Path, *, clip_id: str | None = None, fps: float | None = None) -> PersonGroundTruth:
    """Import a CVAT MOT 1.1 ZIP into the normalized person ground-truth artifact."""

    zip_path = Path(path)
    if not zip_path.is_file():
        raise ValueError(f"missing MOT zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        labels = _read_labels(archive)
        rows = _read_gt_rows(archive)

    by_frame: dict[int, list[PersonLabel]] = defaultdict(list)
    valid_count = 0
    ignored_count = 0
    track_ids: set[int] = set()

    for row in rows:
        source_frame_id = _positive_int(row[0], field="frame")
        track_id = _positive_int(row[1], field="track_id")
        x, y, width, height = [float(value) for value in row[2:6]]
        mot_mark = float(row[6]) if len(row) > 6 and row[6] != "" else 1.0
        class_id = _positive_int(row[7], field="class_id") if len(row) > 7 and row[7] != "" else None
        visibility = float(row[8]) if len(row) > 8 and row[8] != "" else None
        class_name = labels.get(class_id) if class_id is not None else None
        person_class = _is_person_class(class_name)
        ignored = mot_mark <= 0.0 or not person_class
        label = PersonLabel(
            track_id=track_id,
            bbox_xywh=(x, y, width, height),
            ignored=ignored,
            visibility=visibility,
            confidence=max(0.0, min(1.0, mot_mark)),
            class_id=class_id,
            class_name=class_name,
            person_class=person_class,
        )
        by_frame[source_frame_id].append(label)
        if ignored:
            ignored_count += 1
        else:
            valid_count += 1
            track_ids.add(track_id)

    frames = [
        PersonGroundTruthFrame(
            frame_index=source_frame_id - 1,
            source_frame_id=source_frame_id,
            labels=labels_for_frame,
        )
        for source_frame_id, labels_for_frame in sorted(by_frame.items())
    ]
    max_players = max((_valid_label_count(frame.labels) for frame in frames), default=0)
    summary = PersonGroundTruthSummary(
        frame_count=len(frames),
        valid_label_count=valid_count,
        ignored_label_count=ignored_count,
        track_ids=sorted(track_ids),
        max_valid_players_per_frame=max_players,
    )
    return PersonGroundTruth(
        schema_version=1,
        artifact_type="racketsport_person_ground_truth",
        clip_id=clip_id or _clip_id_from_zip(zip_path),
        source_format="cvat_mot_1_1",
        source_path=str(zip_path),
        fps=fps,
        frames=frames,
        summary=summary,
    )


def write_person_ground_truth(path: str | Path, ground_truth: PersonGroundTruth) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = ground_truth.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_labels(archive: zipfile.ZipFile) -> dict[int, str]:
    try:
        raw = archive.read("gt/labels.txt").decode("utf-8")
    except KeyError:
        return {}
    labels: dict[int, str] = {}
    for index, line in enumerate(raw.splitlines(), start=1):
        label = line.strip()
        if label:
            labels[index] = label
    return labels


def _read_gt_rows(archive: zipfile.ZipFile) -> list[list[str]]:
    try:
        raw = archive.read("gt/gt.txt").decode("utf-8")
    except KeyError as exc:
        raise ValueError("CVAT MOT zip is missing gt/gt.txt") from exc
    rows: list[list[str]] = []
    for row in csv.reader(raw.splitlines()):
        if not row:
            continue
        if len(row) < 6:
            raise ValueError(f"MOT row must contain at least 6 columns: {row!r}")
        rows.append([value.strip() for value in row])
    return rows


def _is_person_class(class_name: str | None) -> bool:
    return class_name is None or class_name.strip().lower() in PERSON_CLASS_NAMES


def _positive_int(value: str, *, field: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    parsed = int(number)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _valid_label_count(labels: list[PersonLabel]) -> int:
    return sum(1 for label in labels if not label.ignored)


def _clip_id_from_zip(path: Path) -> str:
    stem = path.stem
    return stem.replace(" ", "_")


__all__ = ["PERSON_CLASS_NAMES", "import_mot_zip", "write_person_ground_truth"]
