"""CVAT for video 1.1 annotation import for reviewed pickleball labels."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .schemas import (
    CvatVideoAnnotationSummary,
    CvatVideoAnnotations,
    CvatVideoBox,
    CvatVideoFrame,
    CvatVideoTask,
    CvatVideoTrackSummary,
    PersonGroundTruth,
    PersonGroundTruthFrame,
    PersonGroundTruthSummary,
    PersonLabel,
)

PLAYER_LABEL_NAMES = {"player"}


def import_cvat_video_zip(
    path: str | Path,
    *,
    clip_id: str | None = None,
    fps: float | None = None,
    max_frame_index: int | None = None,
) -> tuple[CvatVideoAnnotations, PersonGroundTruth]:
    """Import a CVAT for video 1.1 ZIP into reviewed-box and person GT artifacts."""

    zip_path = Path(path)
    if not zip_path.is_file():
        raise ValueError(f"missing CVAT video zip: {zip_path}")
    if max_frame_index is not None and max_frame_index < 0:
        raise ValueError("max_frame_index must be nonnegative")

    with zipfile.ZipFile(zip_path) as archive:
        try:
            raw_xml = archive.read("annotations.xml")
        except KeyError as exc:
            raise ValueError("CVAT video zip is missing annotations.xml") from exc

    root = ElementTree.fromstring(raw_xml)
    if _text(root.find("version")) != "1.1":
        raise ValueError("CVAT video annotations.xml must be version 1.1")

    task = _parse_task(root)
    stable_clip_id = clip_id or _clip_id_from_zip(zip_path)
    boxes_by_frame: dict[int, list[CvatVideoBox]] = defaultdict(list)
    labels_in_order = _task_labels(root)
    track_summaries: list[CvatVideoTrackSummary] = []
    track_count_by_label: Counter[str] = Counter()
    visible_box_count_by_label: Counter[str] = Counter()
    outside_box_count = 0

    for track in root.findall("track"):
        raw_track_id = _int_attr(track, "id", field="track.id", minimum=0)
        label = _required_attr(track, "label")
        source = track.attrib.get("source")
        track_count_by_label[label] += 1

        visible_frames: list[int] = []
        track_outside_count = 0
        track_keyframe_count = 0
        visible_count = 0

        for shape_node in _track_shapes(track):
            frame_index = _int_attr(shape_node, "frame", field=f"{shape_node.tag}.frame", minimum=0)
            outside = _bool_attr(shape_node, "outside")
            keyframe = _bool_attr(shape_node, "keyframe")
            occluded = _bool_attr(shape_node, "occluded")
            if max_frame_index is not None and frame_index > max_frame_index:
                continue
            if keyframe:
                track_keyframe_count += 1
            if outside:
                track_outside_count += 1
                outside_box_count += 1
                continue

            x1, y1, x2, y2 = _shape_bbox_xyxy(shape_node)
            blur_fields = _shape_ball_blur_fields(shape_node) if label.strip().lower() == "ball" else {}
            visible_box = CvatVideoBox(
                track_id=raw_track_id,
                label=label,
                frame_index=frame_index,
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                keyframe=keyframe,
                occluded=occluded,
                source=source,
                **blur_fields,
            )
            boxes_by_frame[frame_index].append(visible_box)
            visible_frames.append(frame_index)
            visible_count += 1
            visible_box_count_by_label[label] += 1

        track_summaries.append(
            CvatVideoTrackSummary(
                track_id=raw_track_id,
                label=label,
                visible_box_count=visible_count,
                outside_box_count=track_outside_count,
                keyframe_count=track_keyframe_count,
                first_visible_frame=min(visible_frames) if visible_frames else None,
                last_visible_frame=max(visible_frames) if visible_frames else None,
            )
        )

    if max_frame_index is None:
        frame_count = task.size if task.size > 0 else (_max_frame_index(boxes_by_frame) + 1)
    else:
        frame_count = min(task.size, max_frame_index + 1) if task.size > 0 else max_frame_index + 1
        task = task.model_copy(update={"size": frame_count, "stop_frame": frame_count - 1 if frame_count else 0})
    frames = [
        CvatVideoFrame(frame_index=frame_index, boxes=sorted(boxes_by_frame.get(frame_index, []), key=lambda box: box.track_id))
        for frame_index in range(frame_count)
    ]
    all_labels = _ordered_labels(labels_in_order, track_count_by_label)
    annotations = CvatVideoAnnotations(
        schema_version=1,
        artifact_type="racketsport_cvat_video_annotations",
        clip_id=stable_clip_id,
        source_format="cvat_video_1_1",
        source_path=str(zip_path),
        task=task,
        frames=frames,
        tracks=sorted(track_summaries, key=lambda item: (item.label, item.track_id)),
        summary=CvatVideoAnnotationSummary(
            frame_count=frame_count,
            visible_box_count=sum(visible_box_count_by_label.values()),
            outside_box_count=outside_box_count,
            labels=all_labels,
            track_count_by_label={label: track_count_by_label[label] for label in all_labels if track_count_by_label[label]},
            visible_box_count_by_label={label: visible_box_count_by_label[label] for label in all_labels if visible_box_count_by_label[label]},
        ),
    )
    return annotations, _person_ground_truth_from_annotations(annotations, fps=fps)


def write_cvat_video_annotations(path: str | Path, annotations: CvatVideoAnnotations) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = annotations.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_person_ground_truth_from_cvat_video(path: str | Path, ground_truth: PersonGroundTruth) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = ground_truth.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _person_ground_truth_from_annotations(annotations: CvatVideoAnnotations, *, fps: float | None) -> PersonGroundTruth:
    track_ids: set[int] = set()
    valid_count = 0
    frames: list[PersonGroundTruthFrame] = []
    for frame in annotations.frames:
        labels: list[PersonLabel] = []
        for box in frame.boxes:
            if box.label.strip().lower() not in PLAYER_LABEL_NAMES:
                continue
            track_id = box.track_id + 1
            track_ids.add(track_id)
            valid_count += 1
            labels.append(
                PersonLabel(
                    track_id=track_id,
                    bbox_xywh=box.bbox_xywh,
                    ignored=False,
                    visibility=0.5 if box.occluded else 1.0,
                    confidence=1.0,
                    class_id=None,
                    class_name=box.label,
                    person_class=True,
                )
            )
        frames.append(PersonGroundTruthFrame(frame_index=frame.frame_index, source_frame_id=frame.frame_index + 1, labels=labels))

    return PersonGroundTruth(
        schema_version=1,
        artifact_type="racketsport_person_ground_truth",
        clip_id=annotations.clip_id,
        source_format="cvat_video_1_1",
        source_path=annotations.source_path,
        fps=fps,
        frames=frames,
        summary=PersonGroundTruthSummary(
            frame_count=len(frames),
            valid_label_count=valid_count,
            ignored_label_count=0,
            track_ids=sorted(track_ids),
            max_valid_players_per_frame=max((len(frame.labels) for frame in frames), default=0),
        ),
    )


def _parse_task(root: ElementTree.Element) -> CvatVideoTask:
    task = root.find("./meta/task")
    if task is None:
        raise ValueError("annotations.xml missing meta/task")
    original_size = task.find("original_size")
    if original_size is None:
        raise ValueError("annotations.xml missing original_size")
    return CvatVideoTask(
        task_id=_optional_int(_text(task.find("id"))),
        name=_text(task.find("name")) or None,
        size=_optional_int(_text(task.find("size"))) or 0,
        mode=_text(task.find("mode")) or None,
        start_frame=_optional_int(_text(task.find("start_frame"))) or 0,
        stop_frame=_optional_int(_text(task.find("stop_frame"))) or 0,
        original_size=(
            _required_positive_int(_text(original_size.find("width")), field="original_size.width"),
            _required_positive_int(_text(original_size.find("height")), field="original_size.height"),
        ),
        source=_text(task.find("source")) or None,
        dumped=_text(root.find("./meta/dumped")) or None,
    )


def _task_labels(root: ElementTree.Element) -> list[str]:
    labels: list[str] = []
    for label in root.findall("./meta/task/labels/label/name"):
        name = _text(label)
        if name:
            labels.append(name)
    return labels


def _ordered_labels(labels_in_order: list[str], track_counts: Counter[str]) -> list[str]:
    labels = list(dict.fromkeys(labels_in_order))
    for label in sorted(track_counts):
        if label not in labels:
            labels.append(label)
    return labels


def _max_frame_index(boxes_by_frame: dict[int, list[CvatVideoBox]]) -> int:
    return max(boxes_by_frame, default=-1)


def _track_shapes(track: ElementTree.Element) -> list[ElementTree.Element]:
    return [node for node in track if node.tag in {"box", "ellipse"}]


def _shape_bbox_xyxy(node: ElementTree.Element) -> tuple[float, float, float, float]:
    if node.tag == "box":
        return (
            _float_attr(node, "xtl"),
            _float_attr(node, "ytl"),
            _float_attr(node, "xbr"),
            _float_attr(node, "ybr"),
        )
    if node.tag == "ellipse":
        cx = _float_attr(node, "cx")
        cy = _float_attr(node, "cy")
        rx = _float_attr(node, "rx")
        ry = _float_attr(node, "ry")
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    raise ValueError(f"unsupported CVAT track shape: {node.tag}")


def _shape_ball_blur_fields(node: ElementTree.Element) -> dict[str, Any]:
    attributes = _shape_attributes(node)
    fields: dict[str, Any] = {}
    for name in ("center_convention", "blur_label_quality"):
        value = attributes.get(name)
        if value:
            fields[name] = value
    for name in ("blur_angle_deg", "blur_length_px", "blur_width_px"):
        value = attributes.get(name)
        if value:
            fields[name] = float(value)
    return fields


def _shape_attributes(node: ElementTree.Element) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for attribute in node.findall("attribute"):
        name = attribute.attrib.get("name", "").strip()
        value = _text(attribute)
        if name and value:
            parsed[name] = value
    return parsed


def _text(node: ElementTree.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _required_attr(node: ElementTree.Element, name: str) -> str:
    value = node.attrib.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required attribute: {name}")
    return value


def _int_attr(node: ElementTree.Element, name: str, *, field: str, minimum: int) -> int:
    value = _required_attr(node, name)
    number = _optional_int(value)
    if number is None or number < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return number


def _float_attr(node: ElementTree.Element, name: str) -> float:
    value = _required_attr(node, name)
    return float(value)


def _bool_attr(node: ElementTree.Element, name: str) -> bool:
    return _required_attr(node, name) == "1"


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"expected integer text: {value}")
    return int(number)


def _required_positive_int(value: str, *, field: str) -> int:
    number = _optional_int(value)
    if number is None or number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _clip_id_from_zip(path: Path) -> str:
    return path.stem.replace(" ", "_")


__all__ = [
    "PLAYER_LABEL_NAMES",
    "import_cvat_video_zip",
    "write_cvat_video_annotations",
    "write_person_ground_truth_from_cvat_video",
]
