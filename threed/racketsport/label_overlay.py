"""Qualitative MP4 overlays for prototype draft labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOTYPE_GATE_CLIPS = (
    "ppa_austin_md_qf_1200_high_baseline",
    "ppa_singles_0500_high_baseline",
    "gear360_0200_high_near_overhead",
    "burlington_gold_0300_low_steep_corner",
    "side_view_game5_0100_high_side_fence",
)

LAYER_FILES = (
    ("court", "court_corners.json"),
    ("players", "players.json"),
    ("ball", "ball.json"),
    ("events", "events.json"),
    ("racket", "racket_pose.json"),
    ("foot_contact", "foot_contact.json"),
)


def render_label_overlays(
    *,
    video_path: Path,
    draft_label_dir: Path,
    output_root: Path,
    clip_name: str | None = None,
    write_index: bool = True,
    write_markdown: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Render an all-label qualitative overlay MP4 for one clip."""

    cv2 = _cv2()
    video_path = Path(video_path)
    draft_label_dir = Path(draft_label_dir)
    clip_name = clip_name or video_path.stem
    compare_dir = Path(output_root) / clip_name / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = compare_dir / "all_labels_overlay.mp4"

    label_data, available_layers, warnings = _load_layers(draft_label_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open overlay writer: {overlay_path}")

    frame_index = 0
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
) -> dict[str, Any]:
    """Render overlays for the prototype gate under a repo/workspace root."""

    root = Path(root)
    clip_names = list(clips or PROTOTYPE_GATE_CLIPS)
    summaries: list[dict[str, Any]] = []
    for clip in clip_names:
        video_path = root / "data" / "testclips" / clip / "source.mp4"
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


def _load_layers(labels_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
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
        items = _payload_items(payload)
        if items:
            label_data[layer] = items
            available_layers.append(layer)
        else:
            warnings.append(f"{filename} has no drawable items")
    return label_data, available_layers, warnings


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        annotation = payload.get("annotation")
        if isinstance(annotation, dict) and isinstance(annotation.get("items"), list):
            return [item for item in annotation["items"] if isinstance(item, dict)]
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
    return []


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


def _point(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return 0, 0
    return int(round(float(value[0]))), int(round(float(value[1])))


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
