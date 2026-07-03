from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES

PARTIAL_LABEL_ARTIFACT_TYPE = "racketsport_court_keypoint_partial_labels"
VISIBLE = "visible"
MISSING_OCCLUDED_OR_OFF_FRAME = "missing_occluded_or_off_frame"
PARTIAL_FRAME_STATUS = "reviewed_partial_visible"


@dataclass(frozen=True)
class PartialCourtKeypointFrame:
    frame: str
    status: str
    keypoints: dict[str, tuple[float, float]]
    visibility_by_keypoint: dict[str, str]


@dataclass(frozen=True)
class PartialCourtKeypoints:
    clip: str
    label_coordinate_space: tuple[float, float]
    source_resolution: tuple[float, float]
    frames: list[PartialCourtKeypointFrame]


def build_partial_court_keypoint_label_payload(
    *,
    clip: str,
    reviewer: str,
    items: Sequence[Mapping[str, Any]],
    source_resolution: Sequence[int | float],
    label_coordinate_space: Sequence[int | float],
    available_review_frame_count: int,
    sample_every_frames: int | None,
    frame_dir: str,
    reviewed_at_utc: str,
) -> dict[str, Any]:
    """Build a visible-point court-keypoint label artifact.

    This artifact is intentionally distinct from the full metric-15 reviewed
    labels. It can carry frames where one or more keypoints are invisible or
    off-frame, so consumers can score visible points without pretending the clip
    is ready for full 15-point metric calibration.
    """

    label_w, label_h = _size_pair(label_coordinate_space, "label_coordinate_space")
    source_w, source_h = _size_pair(source_resolution, "source_resolution")
    exported_items: list[dict[str, Any]] = []
    for raw_item in items:
        raw_keypoints = raw_item.get("keypoints", {})
        if not isinstance(raw_keypoints, Mapping) or not raw_keypoints:
            continue
        keypoints: dict[str, list[float]] = {}
        for name, value in raw_keypoints.items():
            if name not in PICKLEBALL_COURT_KEYPOINT_NAMES:
                raise ValueError(f"unexpected court keypoint: {name}")
            xy = _point(value, key=f"keypoints.{name}", max_x=label_w, max_y=label_h)
            keypoints[str(name)] = [xy[0], xy[1]]
        visibility = {
            name: VISIBLE if name in keypoints else MISSING_OCCLUDED_OR_OFF_FRAME
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        }
        exported_items.append(
            {
                "frame": str(raw_item.get("frame", "")),
                "review_id": str(raw_item.get("review_id", "")),
                "status": PARTIAL_FRAME_STATUS,
                "keypoints": keypoints,
                "visibility_by_keypoint": visibility,
            }
        )

    if not exported_items:
        raise ValueError("partial court-keypoint labels require at least one visible keypoint")

    return {
        "schema_version": 1,
        "artifact_type": PARTIAL_LABEL_ARTIFACT_TYPE,
        "clip": str(clip),
        "review": {
            "status": "reviewed_partial",
            "reviewer": str(reviewer or "local_court_keypoint_review"),
            "reviewed_at_utc": str(reviewed_at_utc),
            "not_full_metric15_calibration": True,
            "visible_keypoint_frame_count": len(exported_items),
        },
        "frames": {
            "frame_dir": str(frame_dir),
            "frame_count": len(exported_items),
            "available_review_frame_count": int(available_review_frame_count),
            "source_resolution": [source_w, source_h],
            "label_coordinate_space": [label_w, label_h],
            "sample_every_frames": sample_every_frames,
        },
        "annotation": {"items": exported_items},
    }


def load_partial_court_keypoints(path: str | Path) -> PartialCourtKeypoints:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != PARTIAL_LABEL_ARTIFACT_TYPE:
        raise ValueError(f"{path}: expected {PARTIAL_LABEL_ARTIFACT_TYPE}")
    frames_meta = payload.get("frames")
    if not isinstance(frames_meta, Mapping):
        raise ValueError(f"{path}: missing frames metadata")
    label_space = frames_meta.get("label_coordinate_space")
    source_res = frames_meta.get("source_resolution")
    label_w, label_h = _size_pair(label_space, "frames.label_coordinate_space")
    source_w, source_h = _size_pair(source_res, "frames.source_resolution")

    frames: list[PartialCourtKeypointFrame] = []
    for raw_item in payload.get("annotation", {}).get("items", []):
        if not isinstance(raw_item, Mapping):
            continue
        raw_keypoints = raw_item.get("keypoints", {})
        if not isinstance(raw_keypoints, Mapping) or not raw_keypoints:
            continue
        keypoints = {
            str(name): tuple(_point(value, key=f"annotation.items.keypoints.{name}", max_x=label_w, max_y=label_h))
            for name, value in raw_keypoints.items()
            if name in PICKLEBALL_COURT_KEYPOINT_NAMES
        }
        raw_visibility = raw_item.get("visibility_by_keypoint", {})
        if not isinstance(raw_visibility, Mapping):
            raw_visibility = {}
        visibility = {
            name: str(raw_visibility.get(name) or (VISIBLE if name in keypoints else MISSING_OCCLUDED_OR_OFF_FRAME))
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        }
        frames.append(
            PartialCourtKeypointFrame(
                frame=str(raw_item.get("frame", "")),
                status=str(raw_item.get("status", "")),
                keypoints=keypoints,
                visibility_by_keypoint=visibility,
            )
        )
    if not frames:
        raise ValueError(f"{path}: no partial court-keypoint frames with visible keypoints")
    return PartialCourtKeypoints(
        clip=str(payload.get("clip", "")),
        label_coordinate_space=(label_w, label_h),
        source_resolution=(source_w, source_h),
        frames=frames,
    )


def _size_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item size")
    width = _positive_float(value[0], f"{name}[0]")
    height = _positive_float(value[1], f"{name}[1]")
    return width, height


def _point(value: Any, *, key: str, max_x: float, max_y: float) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{key} must be a two-item point")
    x = _finite_nonnegative(value[0], f"{key}[0]")
    y = _finite_nonnegative(value[1], f"{key}[1]")
    if x > max_x or y > max_y:
        raise ValueError(f"{key} is outside label coordinate space")
    return x, y


def _positive_float(value: Any, name: str) -> float:
    number = _finite_nonnegative(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if number < 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite and nonnegative")
    return number
