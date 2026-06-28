"""Qualitative MP4 overlays for prototype draft labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from threed.racketsport.autolabel import PROTOTYPE_GATE_CLIPS

LAYER_FILES = (
    ("court", "court_corners.json"),
    ("players", "players.json"),
    ("ball", "ball.json"),
    ("events", "events.json"),
    ("racket", "racket_pose.json"),
    ("foot_contact", "foot_contact.json"),
)


@dataclass(frozen=True)
class _FramePackEntry:
    path: Path
    sampled_index: int


def render_label_overlays(
    *,
    video_path: Path,
    draft_label_dir: Path,
    output_root: Path,
    clip_name: str | None = None,
    write_index: bool = True,
    write_markdown: bool = False,
    max_frames: int | None = None,
    frame_pack_only: bool = False,
) -> dict[str, Any]:
    """Render an all-label qualitative overlay MP4 for one clip."""

    cv2 = _cv2()
    video_path = Path(video_path)
    draft_label_dir = Path(draft_label_dir)
    clip_name = clip_name or video_path.stem
    compare_dir = Path(output_root) / clip_name / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = compare_dir / "all_labels_overlay.mp4"

    label_data: dict[str, list[dict[str, Any]]]
    available_layers: list[str]
    warnings: list[str]
    frame_index = 0
    if frame_pack_only:
        frame_pack_size = _frame_pack_size(cv2, draft_label_dir, max_frames=max_frames)
        label_data, available_layers, warnings = _load_layers(
            draft_label_dir,
            render_mode="frame_pack",
            output_size=frame_pack_size,
            frame_pack_size=frame_pack_size,
        )
        fallback_count = _render_from_frame_pack(
            cv2,
            draft_label_dir=draft_label_dir,
            overlay_path=overlay_path,
            label_data=label_data,
            max_frames=max_frames,
        )
        if fallback_count > 0:
            frame_index = fallback_count
            warnings.append("rendered from label frame pack without source video decode")
        else:
            warnings.append("frame-pack-only render requested but no label frame pack fallback was available")
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        frame_pack_size = _frame_pack_size(cv2, draft_label_dir)
        label_data, available_layers, warnings = _load_layers(
            draft_label_dir,
            render_mode="source_video",
            output_size=(width, height),
            frame_pack_size=frame_pack_size,
        )
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open overlay writer: {overlay_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw_layers(cv2, frame, frame_index, label_data)
                writer.write(frame)
                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()
        if frame_index == 0:
            frame_pack_size = _frame_pack_size(cv2, draft_label_dir, max_frames=max_frames)
            label_data, available_layers, warnings = _load_layers(
                draft_label_dir,
                render_mode="frame_pack",
                output_size=frame_pack_size,
                frame_pack_size=frame_pack_size,
            )
            fallback_count = _render_from_frame_pack(
                cv2,
                draft_label_dir=draft_label_dir,
                overlay_path=overlay_path,
                label_data=label_data,
                max_frames=max_frames,
            )
            if fallback_count > 0:
                frame_index = fallback_count
                warnings.append("video decode produced 0 frames; rendered from label frame pack")
            else:
                warnings.append("video decode produced 0 frames and no label frame pack fallback was available")

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_label_overlay",
        "status": "rendered",
        "clip": clip_name,
        "video_path": str(video_path),
        "draft_label_dir": str(draft_label_dir),
        "compare_dir": str(compare_dir),
        "rendered_videos": [str(overlay_path)],
        "available_layers": available_layers,
        "warnings": warnings,
        "frame_count": frame_index,
        "qualitative_status": "prototype_not_gate_verified",
    }
    if write_index:
        (compare_dir / "label_overlay_index.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if write_markdown:
        (compare_dir / "label_overlay_index.md").write_text(_markdown_index(summary), encoding="utf-8")
    return summary


def render_prototype_gate(
    *,
    root: Path,
    clips: Sequence[str] | None = None,
    write_markdown: bool = False,
    max_frames: int | None = None,
    frame_pack_only: bool = False,
) -> dict[str, Any]:
    """Render overlays for the prototype gate under a repo/workspace root."""

    root = Path(root)
    clip_names = list(clips or PROTOTYPE_GATE_CLIPS)
    summaries: list[dict[str, Any]] = []
    for clip in clip_names:
        video_path = _prototype_video_path(root, clip)
        preferred = root / "runs" / "eval0" / "prototype_gate" / clip / "labels"
        fallback = root / "runs" / "label_drafts" / "prototype_gate" / clip / "labels"
        labels = preferred if preferred.exists() else fallback
        if not video_path.exists() or not labels.exists():
            summaries.append(
                {
                    "clip": clip,
                    "status": "skipped",
                    "video_path": str(video_path),
                    "draft_label_dir": str(labels),
                    "warnings": ["source video or draft labels missing"],
                    "qualitative_status": "prototype_not_gate_verified",
                }
            )
            continue
        summaries.append(
            render_label_overlays(
                video_path=video_path,
                draft_label_dir=labels,
                output_root=root / "runs" / "eval0" / "prototype_gate",
                clip_name=clip,
                write_index=True,
                write_markdown=write_markdown,
                max_frames=max_frames,
                frame_pack_only=frame_pack_only,
            )
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_prototype_overlay_run",
        "status": "rendered" if any(item.get("status") == "rendered" for item in summaries) else "no_renders",
        "root": str(root),
        "clip_count": len(summaries),
        "clips": summaries,
    }


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for overlay rendering") from exc
    return cv2


def _load_layers(
    labels_dir: Path,
    *,
    render_mode: Literal["source_video", "frame_pack"],
    output_size: tuple[int, int] | None = None,
    frame_pack_size: tuple[int, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
    label_data: dict[str, list[dict[str, Any]]] = {}
    available_layers: list[str] = []
    warnings: list[str] = []
    for layer, filename in LAYER_FILES:
        path = labels_dir / filename
        if not path.is_file():
            warnings.append(f"{filename} not present")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"{filename} invalid JSON: {exc}")
            continue
        items = _payload_items_for_render(
            payload,
            render_mode=render_mode,
            output_size=output_size,
            frame_pack_size=frame_pack_size,
        )
        if items:
            label_data[layer] = items
            available_layers.append(layer)
        else:
            warnings.append(f"{filename} has no drawable items")
    return label_data, available_layers, warnings


def _render_from_frame_pack(
    cv2: Any,
    *,
    draft_label_dir: Path,
    overlay_path: Path,
    label_data: dict[str, list[dict[str, Any]]],
    max_frames: int | None,
) -> int:
    entries, metadata = _frame_pack_entries_from_labels(draft_label_dir)
    if max_frames is not None:
        entries = entries[:max_frames]
    first_frame = None
    first_entry: _FramePackEntry | None = None
    usable_entries: list[_FramePackEntry] = []
    for entry in entries:
        frame = cv2.imread(str(entry.path))
        if frame is not None:
            first_frame = frame
            first_entry = entry
            usable_entries.append(entry)
            break
    if first_frame is None:
        return 0
    usable_entries.extend(entry for entry in entries if entry != first_entry)
    height, width = first_frame.shape[:2]
    fps = _frame_pack_fps(metadata, frame_count=len(entries))
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return 0
    count = 0
    try:
        for entry in usable_entries:
            frame = first_frame if count == 0 else cv2.imread(str(entry.path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            _draw_layers(cv2, frame, entry.sampled_index, label_data)
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count


def _frame_paths_from_labels(labels_dir: Path) -> list[Path]:
    frame_paths, _metadata = _frame_pack_from_labels(labels_dir)
    return frame_paths


def _frame_pack_from_labels(labels_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    entries, metadata = _frame_pack_entries_from_labels(labels_dir)
    return [entry.path for entry in entries], metadata


def _frame_pack_entries_from_labels(labels_dir: Path) -> tuple[list[_FramePackEntry], dict[str, Any]]:
    for _layer, filename in LAYER_FILES:
        path = labels_dir / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(frames, dict):
            continue
        metadata, base_dir = _frame_pack_metadata(frames, labels_dir)
        raw_frames = frames.get("frames")
        if not isinstance(raw_frames, list):
            raw_frames = metadata.get("frames")
        if not isinstance(raw_frames, list):
            continue
        entries: list[_FramePackEntry] = []
        for index, frame in enumerate(raw_frames):
            value = _frame_path_value(frame)
            if value:
                entries.append(
                    _FramePackEntry(
                        path=_resolve_frame_path(value, base_dir, labels_dir=labels_dir),
                        sampled_index=_frame_pack_sample_index(frame, fallback_index=index),
                    )
                )
        existing = [entry for entry in entries if entry.path.is_file()]
        if existing:
            return existing, metadata
    return [], {}


def _prototype_video_path(root: Path, clip: str) -> Path:
    primary = root / "data" / "testclips" / clip / "source.mp4"
    if primary.exists():
        return primary
    source_clip = root / "data" / "source_clips" / f"{clip}.mp4"
    if source_clip.exists():
        return source_clip
    return primary


def _frame_pack_metadata(frames: dict[str, Any], labels_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_payload: dict[str, Any] = {}
    base_dir = labels_dir
    manifest_path_value = frames.get("manifest_path")
    if manifest_path_value:
        manifest_path = Path(str(manifest_path_value))
        if not manifest_path.is_absolute():
            manifest_path = manifest_path if manifest_path.is_file() else labels_dir / manifest_path
        if manifest_path.is_file():
            base_dir = manifest_path.parent
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                manifest_payload = loaded
    return {**manifest_payload, **frames}, base_dir


def _frame_path_value(frame: Any) -> Any:
    if isinstance(frame, dict):
        return frame.get("path") or frame.get("name") or frame.get("frame")
    return frame


def _frame_pack_sample_index(frame: Any, *, fallback_index: int) -> int:
    value: Any
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame") or frame.get("path")
    else:
        value = frame
    if value is not None:
        digits = "".join(ch for ch in Path(str(value)).stem if ch.isdigit())
        if digits:
            return max(0, int(digits) - 1)
    return fallback_index


def _resolve_frame_path(value: Any, base_dir: Path, *, labels_dir: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [path, base_dir / path]
    if labels_dir is not None:
        candidates.extend(_prototype_gate_frame_pack_candidates(path, labels_dir))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if path.is_absolute() else base_dir / path


def _prototype_gate_frame_pack_candidates(path: Path, labels_dir: Path) -> list[Path]:
    if labels_dir.name != "labels":
        return []
    clip_dir = labels_dir.parent
    prototype_root = clip_dir.parent
    if not prototype_root.name.startswith("prototype_gate"):
        return []
    frame_name = path.name
    if not frame_name:
        return []
    clip_name = clip_dir.name
    return [
        prototype_root / "review_bundle" / "images" / clip_name / frame_name,
        prototype_root / "cvat_tasks" / clip_name / "images" / frame_name,
    ]


def _frame_pack_fps(metadata: dict[str, Any], *, frame_count: int) -> float:
    source_fps = _positive_float(metadata.get("source_fps"))
    sample_every = _positive_float(metadata.get("sample_every_frames"))
    if source_fps is not None and sample_every is not None:
        return source_fps / sample_every

    duration_s = _positive_float(metadata.get("source_duration_s"))
    manifest_frame_count = _positive_float(metadata.get("frame_count")) or float(frame_count)
    if duration_s is not None and manifest_frame_count > 0:
        return manifest_frame_count / duration_s

    return 10.0


def _frame_pack_size(cv2: Any, labels_dir: Path, *, max_frames: int | None = None) -> tuple[int, int] | None:
    entries, _metadata = _frame_pack_entries_from_labels(labels_dir)
    if max_frames is not None:
        entries = entries[:max_frames]
    for entry in entries:
        frame = cv2.imread(str(entry.path))
        if frame is not None:
            height, width = frame.shape[:2]
            return int(width), int(height)
    return None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        annotation = payload.get("annotation")
        if isinstance(annotation, dict) and isinstance(annotation.get("items"), list):
            return [item for item in annotation["items"] if isinstance(item, dict)]
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
    return []


def _payload_items_for_render(
    payload: Any,
    *,
    render_mode: Literal["source_video", "frame_pack"],
    output_size: tuple[int, int] | None = None,
    frame_pack_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    items = _payload_items(payload)
    if not isinstance(payload, dict):
        return items

    frames = payload.get("frames")
    frame_metadata = frames if isinstance(frames, dict) else {}
    sample_every = _positive_float(frame_metadata.get("sample_every_frames")) or 1.0
    source_size = _size_from_resolution(frame_metadata.get("source_resolution"))

    normalized: list[dict[str, Any]] = []
    for item in items:
        item_size = _item_coordinate_size(item, source_size=source_size, frame_pack_size=frame_pack_size)
        scale_x, scale_y = _coordinate_scale(item_size, output_size)
        normalized_item = _scale_item_coordinates(item, scale_x=scale_x, scale_y=scale_y)
        render_index = _sampled_item_frame_index(normalized_item, sample_every=sample_every, render_mode=render_mode)
        if render_index is not None:
            normalized_item["_render_frame_index"] = render_index
        normalized.append(normalized_item)
    return normalized


def _draw_layers(cv2: Any, frame: Any, frame_index: int, data: dict[str, list[dict[str, Any]]]) -> None:
    if "court" in data:
        _draw_court(cv2, frame, data["court"])
    if "ball" in data:
        _draw_ball(cv2, frame, frame_index, data["ball"])
    if "players" in data:
        _draw_players(cv2, frame, frame_index, data["players"])
    if "events" in data:
        _draw_events(cv2, frame, frame_index, data["events"])
    if "racket" in data:
        _draw_racket(cv2, frame, frame_index, data["racket"])
    if "foot_contact" in data:
        _draw_foot_contact(cv2, frame, frame_index, data["foot_contact"])


def _draw_court(cv2: Any, frame: Any, items: Iterable[dict[str, Any]]) -> None:
    corners: dict[str, tuple[int, int]] = {}
    for item in items:
        if "xy_px" in item:
            corners[str(item.get("id", len(corners)))] = _point(item["xy_px"])
        for key, value in (item.get("court_corners") or {}).items():
            corners[str(key)] = _point(value)
        if "points_px" in item:
            points = [_point(point) for point in item["points_px"]]
            if len(points) >= 2:
                cv2.line(frame, points[0], points[1], (0, 220, 255), 2)
    ordered = [corners.get(name) for name in ("far_left", "far_right", "near_right", "near_left")]
    if all(point is not None for point in ordered):
        pts = [point for point in ordered if point is not None]
        for left, right in zip(pts, pts[1:] + pts[:1]):
            cv2.line(frame, left, right, (0, 220, 255), 2)
    for name, point in corners.items():
        cv2.circle(frame, point, 4, (0, 255, 255), -1)
        cv2.putText(frame, name, (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def _draw_players(cv2: Any, frame: Any, frame_index: int, items: Iterable[dict[str, Any]]) -> None:
    for item in _items_for_frame(items, frame_index):
        if "bbox" in item:
            x, y, w, h = [int(round(float(v))) for v in item["bbox"][:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)
            cv2.putText(frame, str(item.get("id", "player")), (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        _draw_points(cv2, frame, item.get("keypoints_px"), color=(0, 255, 0))


def _draw_ball(cv2: Any, frame: Any, frame_index: int, items: Sequence[dict[str, Any]]) -> None:
    trail = [item for item in items if _item_frame_index(item) is not None and _item_frame_index(item) <= frame_index]
    for item in trail[-12:]:
        if "xy_px" in item:
            cv2.circle(frame, _point(item["xy_px"]), 3, (0, 140, 255), -1)
    for item in _items_for_frame(items, frame_index):
        if "xy_px" in item:
            cv2.circle(frame, _point(item["xy_px"]), 6, (0, 255, 255), 2)


def _draw_events(cv2: Any, frame: Any, frame_index: int, items: Iterable[dict[str, Any]]) -> None:
    for item in _items_for_frame(items, frame_index):
        point = _point(item.get("xy_px", [20, 20]))
        cv2.drawMarker(frame, point, (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        cv2.putText(frame, str(item.get("label", item.get("type", "event"))), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)


def _draw_racket(cv2: Any, frame: Any, frame_index: int, items: Iterable[dict[str, Any]]) -> None:
    for item in _items_for_frame(items, frame_index):
        bbox = _bbox_xyxy(item)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            label = _racket_label(item)
            if label:
                line_type = getattr(cv2, "LINE_AA", 16)
                origin = (x1, max(12, y1 - 5))
                cv2.putText(frame, label, (origin[0] + 1, origin[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, line_type)
                cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, line_type)
        points = [_point(point) for point in item.get("keypoints_px", [])]
        if len(points) >= 2:
            cv2.line(frame, points[0], points[1], (255, 180, 0), 3)
        _draw_points(cv2, frame, points, color=(255, 180, 0))


def _draw_foot_contact(cv2: Any, frame: Any, frame_index: int, items: Iterable[dict[str, Any]]) -> None:
    for item in _items_for_frame(items, frame_index):
        point = _point(item.get("xy_px", [20, 20]))
        cv2.circle(frame, point, 5, (255, 255, 0), 2)
        cv2.putText(frame, str(item.get("foot", "foot")), (point[0] + 5, point[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)


def _draw_points(cv2: Any, frame: Any, points: Any, *, color: tuple[int, int, int]) -> None:
    if not points:
        return
    for point in points:
        cv2.circle(frame, _point(point), 3, color, -1)


def _items_for_frame(items: Iterable[dict[str, Any]], frame_index: int) -> list[dict[str, Any]]:
    matched = [item for item in items if _item_frame_index(item) == frame_index]
    return matched


def _item_frame_index(item: dict[str, Any]) -> int | None:
    render_value = item.get("_render_frame_index")
    if isinstance(render_value, int):
        return render_value
    value = item.get("frame")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            return max(0, int(digits) - 1)
    return None


def _sampled_item_frame_index(
    item: dict[str, Any],
    *,
    sample_every: float,
    render_mode: Literal["source_video", "frame_pack"],
) -> int | None:
    value = item.get("frame")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        sampled_index = max(0, int(digits) - 1)
        if render_mode == "source_video":
            return int(round(sampled_index * sample_every))
        return sampled_index
    return None


def _point(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return 0, 0
    return int(round(float(value[0]))), int(round(float(value[1])))


def _coordinate_scale(source_resolution: Any, output_size: tuple[int, int] | None) -> tuple[float, float]:
    if output_size is None:
        return 1.0, 1.0
    if not isinstance(source_resolution, (list, tuple)) or len(source_resolution) < 2:
        return 1.0, 1.0
    source_width = _positive_float(source_resolution[0])
    source_height = _positive_float(source_resolution[1])
    if source_width is None or source_height is None:
        return 1.0, 1.0
    return float(output_size[0]) / source_width, float(output_size[1]) / source_height


def _size_from_resolution(source_resolution: Any) -> tuple[float, float] | None:
    if not isinstance(source_resolution, (list, tuple)) or len(source_resolution) < 2:
        return None
    source_width = _positive_float(source_resolution[0])
    source_height = _positive_float(source_resolution[1])
    if source_width is None or source_height is None:
        return None
    return source_width, source_height


def _item_coordinate_size(
    item: dict[str, Any],
    *,
    source_size: tuple[float, float] | None,
    frame_pack_size: tuple[int, int] | None,
) -> tuple[float, float] | None:
    if _item_uses_frame_pack_coordinates(item):
        if frame_pack_size is not None:
            return float(frame_pack_size[0]), float(frame_pack_size[1])
    return source_size


def _item_uses_frame_pack_coordinates(item: dict[str, Any]) -> bool:
    coordinate_space = item.get("coordinate_space") or item.get("coord_space")
    if isinstance(coordinate_space, str):
        normalized = coordinate_space.lower()
        if normalized in {"frame_pack", "review_frame", "label_frame", "image"}:
            return True
        if normalized in {"source", "source_video", "video"}:
            return False
    source = item.get("source")
    if isinstance(source, str):
        normalized_source = source.lower()
        if normalized_source in {"yolo26m_teacher", "human_review", "teacher_artifact"}:
            return True
        if normalized_source.endswith("_teacher"):
            return True
        if normalized_source in {"manual_4_corner_required", "smoke_generated"}:
            return False
    if "teacher_model" in item:
        return True
    return False


def _scale_item_coordinates(item: dict[str, Any], *, scale_x: float, scale_y: float) -> dict[str, Any]:
    if scale_x == 1.0 and scale_y == 1.0:
        return dict(item)

    scaled = dict(item)
    if "bbox" in scaled:
        scaled["bbox"] = _scale_bbox(scaled["bbox"], scale_x=scale_x, scale_y=scale_y)
    if "bbox_xyxy" in scaled:
        scaled["bbox_xyxy"] = _scale_xyxy(scaled["bbox_xyxy"], scale_x=scale_x, scale_y=scale_y)
    if "xy_px" in scaled:
        scaled["xy_px"] = _scale_point(scaled["xy_px"], scale_x=scale_x, scale_y=scale_y)
    if "keypoints_px" in scaled:
        scaled["keypoints_px"] = [_scale_point(point, scale_x=scale_x, scale_y=scale_y) for point in scaled["keypoints_px"]]
    if "points_px" in scaled:
        scaled["points_px"] = [_scale_point(point, scale_x=scale_x, scale_y=scale_y) for point in scaled["points_px"]]
    court_corners = scaled.get("court_corners")
    if isinstance(court_corners, dict):
        scaled["court_corners"] = {
            key: _scale_point(value, scale_x=scale_x, scale_y=scale_y)
            for key, value in court_corners.items()
        }
    return scaled


def _scale_point(value: Any, *, scale_x: float, scale_y: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return [0.0, 0.0]
    return [float(value[0]) * scale_x, float(value[1]) * scale_y]


def _scale_bbox(value: Any, *, scale_x: float, scale_y: float) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return value
    return [float(value[0]) * scale_x, float(value[1]) * scale_y, float(value[2]) * scale_x, float(value[3]) * scale_y]


def _scale_xyxy(value: Any, *, scale_x: float, scale_y: float) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return value
    return [float(value[0]) * scale_x, float(value[1]) * scale_y, float(value[2]) * scale_x, float(value[3]) * scale_y]


def _bbox_xyxy(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = item.get("bbox_xyxy")
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(int(round(float(coord))) for coord in value[:4])  # type: ignore[return-value]
    value = item.get("bbox")
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x, y, w, h = (float(coord) for coord in value[:4])
        return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    return None


def _racket_label(item: dict[str, Any]) -> str:
    confidence = _positive_float(item.get("confidence"))
    if confidence is None:
        return "racket"
    return f"racket {confidence:.2f}"


def _markdown_index(summary: dict[str, Any]) -> str:
    layers = ", ".join(summary["available_layers"]) if summary["available_layers"] else "none"
    videos = "\n".join(f"- `{Path(path).name}`" for path in summary["rendered_videos"])
    warnings = "\n".join(f"- {warning}" for warning in summary["warnings"]) or "- none"
    return (
        "# Label Overlay Index\n\n"
        f"- Clip: `{summary['clip']}`\n"
        f"- Status: `{summary['qualitative_status']}`\n"
        f"- Layers: {layers}\n"
        f"- Frames rendered: {summary['frame_count']}\n\n"
        "## Videos\n"
        f"{videos}\n\n"
        "## Warnings\n"
        f"{warnings}\n"
    )


__all__ = ["PROTOTYPE_GATE_CLIPS", "render_label_overlays", "render_prototype_gate"]
