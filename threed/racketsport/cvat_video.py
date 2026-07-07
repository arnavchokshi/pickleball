"""CVAT for video 1.1 annotation import for reviewed pickleball labels."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .schemas import (
    BALL_VISIBILITY_LEVELS,
    BallVisibilityLevel,
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
    reviewed_frame_indices, reviewed_frame_indices_source = _derive_reviewed_frame_indices(task)
    stable_clip_id = clip_id or _clip_id_from_zip(zip_path)
    boxes_by_frame: dict[int, list[CvatVideoBox]] = defaultdict(list)
    visibility_levels_by_frame: dict[int, dict[str, BallVisibilityLevel]] = defaultdict(dict)
    labels_in_order = _task_labels(root)
    track_stats: dict[tuple[str, int], dict[str, Any]] = {}
    track_count_by_label: Counter[str] = Counter()
    visible_box_count_by_label: Counter[str] = Counter()
    outside_box_count = 0
    max_seen_frame_index = -1

    for track in root.findall("track"):
        raw_track_id = _int_attr(track, "id", field="track.id", minimum=0)
        label = _required_attr(track, "label")
        source = track.attrib.get("source")
        track_count_by_label[label] += 1

        stats_key = (label, raw_track_id)
        stats = track_stats.setdefault(
            stats_key,
            {"label": label, "track_id": raw_track_id, "outside_box_count": 0, "keyframe_count": 0},
        )

        for shape_node in _track_shapes(track):
            frame_index = _int_attr(shape_node, "frame", field=f"{shape_node.tag}.frame", minimum=0)
            outside = _bool_attr(shape_node, "outside")
            keyframe = _bool_attr(shape_node, "keyframe")
            occluded = _bool_attr(shape_node, "occluded")
            if max_frame_index is not None and frame_index > max_frame_index:
                continue
            max_seen_frame_index = max(max_seen_frame_index, frame_index)
            if keyframe:
                stats["keyframe_count"] += 1
            is_ball = label.strip().lower() == "ball"
            visibility_fields = _shape_ball_visibility_fields(shape_node) if is_ball else {}
            visibility_level = visibility_fields.get("visibility_level")
            if outside:
                stats["outside_box_count"] += 1
                outside_box_count += 1
                if visibility_level is not None:
                    if visibility_level in {"clear", "partial"}:
                        visibility_level = None
                    if visibility_level is not None:
                        _set_frame_visibility_level(
                            visibility_levels_by_frame,
                            frame_index=frame_index,
                            label=label,
                            visibility_level=visibility_level,
                        )
                continue

            if is_ball and visibility_level == "out_of_frame":
                stats["outside_box_count"] += 1
                outside_box_count += 1
                _set_frame_visibility_level(
                    visibility_levels_by_frame,
                    frame_index=frame_index,
                    label=label,
                    visibility_level=visibility_level,
                )
                continue

            x1, y1, x2, y2 = _shape_bbox_xyxy(shape_node)
            blur_fields = _shape_ball_blur_fields(shape_node) if is_ball else {}
            visible_box = CvatVideoBox(
                track_id=raw_track_id,
                label=label,
                frame_index=frame_index,
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                keyframe=keyframe,
                occluded=occluded,
                source=source,
                **visibility_fields,
                **blur_fields,
            )
            boxes_by_frame[frame_index].append(visible_box)
            if visibility_level is not None:
                _set_frame_visibility_level(
                    visibility_levels_by_frame,
                    frame_index=frame_index,
                    label=label,
                    visibility_level=visibility_level,
                )
            visible_box_count_by_label[label] += 1

    dedupe_count = _dedupe_single_ball_boxes_by_frame(boxes_by_frame)
    if dedupe_count:
        visible_box_count_by_label["ball"] -= dedupe_count

    track_summaries = _track_summaries_from_boxes(boxes_by_frame, track_stats)

    inferred_frame_count = max(task.size, task.stop_frame + 1, _max_frame_index(boxes_by_frame) + 1, max_seen_frame_index + 1)
    if max_frame_index is None:
        frame_count = inferred_frame_count
    else:
        frame_count = min(inferred_frame_count, max_frame_index + 1)
    task = task.model_copy(update={"size": frame_count, "stop_frame": frame_count - 1 if frame_count else 0})
    if reviewed_frame_indices is not None:
        reviewed_frame_indices = [index for index in reviewed_frame_indices if index < frame_count]
    frames = [
        CvatVideoFrame(
            frame_index=frame_index,
            boxes=sorted(boxes_by_frame.get(frame_index, []), key=lambda box: box.track_id),
            visibility_levels_by_label=dict(sorted(visibility_levels_by_frame.get(frame_index, {}).items())),
        )
        for frame_index in range(frame_count)
    ]
    all_labels = _ordered_labels(labels_in_order, track_count_by_label)
    annotations = CvatVideoAnnotations(
        schema_version=1,
        artifact_type="racketsport_cvat_video_annotations",
        clip_id=stable_clip_id,
        source_format="cvat_video_1_1",
        source_path=str(zip_path),
        reviewed_frame_indices=reviewed_frame_indices,
        reviewed_frame_indices_source=reviewed_frame_indices_source,
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
        task = root.find("./meta/project/tasks/task")
    if task is None:
        raise ValueError("annotations.xml missing meta/task or meta/project/tasks/task")
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
        frame_filter=_text(task.find("frame_filter")) or None,
        original_size=(
            _required_positive_int(_text(original_size.find("width")), field="original_size.width"),
            _required_positive_int(_text(original_size.find("height")), field="original_size.height"),
        ),
        source=_text(task.find("source")) or None,
        dumped=_text(root.find("./meta/dumped")) or None,
    )


def _task_labels(root: ElementTree.Element) -> list[str]:
    labels: list[str] = []
    for label in [
        *root.findall("./meta/task/labels/label/name"),
        *root.findall("./meta/project/labels/label/name"),
        *root.findall("./meta/project/tasks/task/labels/label/name"),
    ]:
        name = _text(label)
        if name:
            labels.append(name)
    return labels


def _derive_reviewed_frame_indices(task: CvatVideoTask) -> tuple[list[int] | None, str | None]:
    frame_filter = (task.frame_filter or "").strip()
    if not frame_filter:
        return None, None
    step_match = re.search(r"(?:^|[;&\s])step\s*=\s*(\d+)(?:$|[;&\s])", frame_filter)
    if step_match is None:
        return None, None
    step = int(step_match.group(1))
    if step <= 0:
        raise ValueError(f"CVAT frame_filter step must be positive: {frame_filter}")
    reviewed = list(range(task.start_frame, task.stop_frame + 1, step))
    if task.size > 0 and len(reviewed) > task.size:
        reviewed = reviewed[: task.size]
    return reviewed, "cvat_frame_filter"


def _ordered_labels(labels_in_order: list[str], track_counts: Counter[str]) -> list[str]:
    labels = list(dict.fromkeys(labels_in_order))
    for label in sorted(track_counts):
        if label not in labels:
            labels.append(label)
    return labels


def _max_frame_index(boxes_by_frame: dict[int, list[CvatVideoBox]]) -> int:
    return max(boxes_by_frame, default=-1)


def _dedupe_single_ball_boxes_by_frame(boxes_by_frame: dict[int, list[CvatVideoBox]]) -> int:
    dropped = 0
    for frame_index, boxes in list(boxes_by_frame.items()):
        ball_boxes = [box for box in boxes if box.label.strip().lower() == "ball"]
        if len(ball_boxes) <= 1:
            continue
        keep = max(ball_boxes, key=lambda box: (_box_area(box), -box.track_id))
        boxes_by_frame[frame_index] = [
            box for box in boxes if box.label.strip().lower() != "ball" or (box.track_id == keep.track_id and box.bbox_xyxy == keep.bbox_xyxy)
        ]
        dropped += len(ball_boxes) - 1
    return dropped


def _track_summaries_from_boxes(
    boxes_by_frame: dict[int, list[CvatVideoBox]],
    track_stats: dict[tuple[str, int], dict[str, Any]],
) -> list[CvatVideoTrackSummary]:
    visible_frames_by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for frame_index, boxes in boxes_by_frame.items():
        for box in boxes:
            visible_frames_by_track[(box.label, box.track_id)].append(frame_index)
    summaries: list[CvatVideoTrackSummary] = []
    for (label, track_id), stats in track_stats.items():
        visible_frames = visible_frames_by_track.get((label, track_id), [])
        summaries.append(
            CvatVideoTrackSummary(
                track_id=track_id,
                label=label,
                visible_box_count=len(visible_frames),
                outside_box_count=int(stats["outside_box_count"]),
                keyframe_count=int(stats["keyframe_count"]),
                first_visible_frame=min(visible_frames) if visible_frames else None,
                last_visible_frame=max(visible_frames) if visible_frames else None,
            )
        )
    return summaries


def _box_area(box: CvatVideoBox) -> float:
    return float(box.bbox_xywh[2]) * float(box.bbox_xywh[3])


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


def _shape_ball_visibility_fields(node: ElementTree.Element) -> dict[str, BallVisibilityLevel]:
    attributes = _shape_attributes(node)
    value = attributes.get("visibility_level")
    if value is None:
        legacy_value = attributes.get("visibility")
        if legacy_value is not None and _normalizes_to_ball_visibility_level(legacy_value):
            value = legacy_value
    if value is None:
        return {}
    return {"visibility_level": _normalize_ball_visibility_level(value)}


def _normalizes_to_ball_visibility_level(value: str) -> bool:
    try:
        _normalize_ball_visibility_level(value)
    except ValueError:
        return False
    return True


def _normalize_ball_visibility_level(value: str) -> BallVisibilityLevel:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in BALL_VISIBILITY_LEVELS:
        raise ValueError(f"visibility_level must be one of {', '.join(BALL_VISIBILITY_LEVELS)}: {value}")
    return normalized  # type: ignore[return-value]


def _set_frame_visibility_level(
    target: dict[int, dict[str, BallVisibilityLevel]],
    *,
    frame_index: int,
    label: str,
    visibility_level: BallVisibilityLevel,
) -> None:
    label_key = label.strip().lower()
    existing = target[frame_index].get(label_key)
    visible_levels = {"clear", "partial"}
    hidden_levels = {"full", "out_of_frame"}
    if existing in hidden_levels and visibility_level in visible_levels:
        target[frame_index][label_key] = visibility_level
        return
    if existing in visible_levels and visibility_level in hidden_levels:
        return
    if existing is not None and existing != visibility_level:
        raise ValueError(
            f"conflicting visibility_level values for {label_key} frame {frame_index}: {existing} vs {visibility_level}"
        )
    target[frame_index][label_key] = visibility_level


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
