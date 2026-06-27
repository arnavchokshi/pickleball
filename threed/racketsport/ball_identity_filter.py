"""Sparse-click identity filtering for target-court ball tracks."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ball_overlay import load_ball_track
from .schemas import BallTrack


@dataclass(frozen=True)
class BallClickItem:
    frame_index: int
    t: float
    xy: tuple[float, float] | None
    visible: bool | None
    visibility: str | None
    review_id: str


@dataclass(frozen=True)
class BallClickReview:
    clip: str
    path: Path
    items: tuple[BallClickItem, ...]

    @property
    def visible_items(self) -> tuple[BallClickItem, ...]:
        return tuple(item for item in self.items if item.visible is True and item.xy is not None)

    @property
    def hidden_items(self) -> tuple[BallClickItem, ...]:
        return tuple(item for item in self.items if item.visible is False)

    @property
    def pending_items(self) -> tuple[BallClickItem, ...]:
        return tuple(item for item in self.items if item.visible is None)


def load_ball_click_review(path: str | Path) -> BallClickReview:
    click_path = Path(path)
    if not click_path.is_file():
        raise ValueError(f"missing ball click file: {click_path}")
    try:
        payload = json.loads(click_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ball click JSON: {click_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"ball click payload must be an object: {click_path}")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"ball click payload missing items list: {click_path}")

    items = tuple(_parse_click_item(item, index) for index, item in enumerate(raw_items))
    return BallClickReview(clip=str(payload.get("clip") or click_path.parent.name), path=click_path, items=items)


def filter_ball_track_with_click_anchors(
    *,
    ball_track_path: str | Path,
    clicks_path: str | Path,
    max_identity_error_px: float = 80.0,
    interpolate_max_gap_frames: int = 45,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_identity_error_px < 0.0:
        raise ValueError("max_identity_error_px must be >= 0")
    if interpolate_max_gap_frames < 0:
        raise ValueError("interpolate_max_gap_frames must be >= 0")

    track = load_ball_track(ball_track_path)
    review = load_ball_click_review(clicks_path)
    original_payload = track.model_dump(mode="json")
    payload = deepcopy(original_payload)
    original_samples_by_index = _payload_samples_by_frame_index(original_payload, fps=float(track.fps))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_clicks = sorted(review.visible_items, key=lambda item: item.frame_index)
    visible_clicks_by_index = {item.frame_index: item for item in visible_clicks}
    hidden_clicks_by_index = {item.frame_index: item for item in review.hidden_items}

    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    identity_rejected_count = 0
    interpolated_count = 0
    clicked_overwrite_count = 0
    clicked_hidden_suppressed_count = 0

    for frame_index, frame in samples_by_index.items():
        if frame_index in hidden_clicks_by_index:
            if frame["visible"]:
                clicked_hidden_suppressed_count += 1
            _hide_frame(frame)
            frame["approx"] = False
            continue

        clicked_visible = visible_clicks_by_index.get(frame_index)
        if clicked_visible is not None and clicked_visible.xy is not None:
            if frame.get("xy") != list(clicked_visible.xy) or not frame["visible"]:
                clicked_overwrite_count += 1
            _set_frame_xy(frame, clicked_visible.xy, conf=1.0, approx=False)
            continue

        expected = _interpolated_anchor_xy(visible_clicks, frame_index)
        if expected is None:
            continue
        anchor_gap = _anchor_gap_frames(visible_clicks, frame_index)
        if anchor_gap is None or anchor_gap > interpolate_max_gap_frames:
            continue

        if frame["visible"] and _distance(frame["xy"], expected) <= max_identity_error_px:
            continue
        if frame["visible"]:
            identity_rejected_count += 1
            _hide_frame(frame)
            frame["approx"] = False
            continue
        _set_frame_xy(frame, expected, conf=0.5, approx=True)
        interpolated_count += 1

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    metrics = {
        "source": _click_metrics(original_samples_by_index, review),
        "output": _click_metrics(_payload_samples_by_frame_index(payload, fps=float(track.fps)), review),
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_identity_filter",
        "status": "filtered_with_sparse_click_anchors_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "clicks_path": str(clicks_path),
        "clip": review.clip,
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "clicked_visible_count": len(review.visible_items),
        "clicked_hidden_count": len(review.hidden_items),
        "click_pending_count": len(review.pending_items),
        "clicked_overwrite_count": clicked_overwrite_count,
        "clicked_hidden_suppressed_count": clicked_hidden_suppressed_count,
        "identity_rejected_count": identity_rejected_count,
        "interpolated_count": interpolated_count,
        "max_identity_error_px": float(max_identity_error_px),
        "interpolate_max_gap_frames": int(interpolate_max_gap_frames),
        "metrics": metrics,
        "not_ground_truth": True,
    }
    return payload, summary


def write_identity_filtered_ball_track(
    *,
    ball_track_path: str | Path,
    clicks_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    max_identity_error_px: float = 80.0,
    interpolate_max_gap_frames: int = 45,
) -> dict[str, Any]:
    payload, summary = filter_ball_track_with_click_anchors(
        ball_track_path=ball_track_path,
        clicks_path=clicks_path,
        max_identity_error_px=max_identity_error_px,
        interpolate_max_gap_frames=interpolate_max_gap_frames,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_click_item(raw: Any, fallback_index: int) -> BallClickItem:
    if not isinstance(raw, dict):
        raise ValueError(f"ball click item must be an object at index {fallback_index}")
    frame_index = int(raw.get("frame_index", fallback_index))
    visible = raw.get("visible")
    if visible is not None and not isinstance(visible, bool):
        raise ValueError(f"ball click visible must be true, false, or null at frame {frame_index}")
    xy_raw = raw.get("ball_xy", raw.get("xy_px"))
    xy: tuple[float, float] | None = None
    if visible is True:
        if not isinstance(xy_raw, list | tuple) or len(xy_raw) != 2:
            raise ValueError(f"visible ball click missing xy at frame {frame_index}")
        xy = (float(xy_raw[0]), float(xy_raw[1]))
    return BallClickItem(
        frame_index=frame_index,
        t=float(raw.get("t", 0.0)),
        xy=xy,
        visible=visible,
        visibility=str(raw["visibility"]) if raw.get("visibility") is not None else None,
        review_id=str(raw.get("review_id", f"ball_frame_{frame_index:06d}")),
    )


def _payload_samples_by_frame_index(payload: dict[str, Any], *, fps: float) -> dict[int, dict[str, Any]]:
    return {int(round(float(frame["t"]) * fps)): frame for frame in payload["frames"]}


def _interpolated_anchor_xy(anchors: list[BallClickItem], frame_index: int) -> tuple[float, float] | None:
    if len(anchors) < 2:
        return None
    previous: BallClickItem | None = None
    next_item: BallClickItem | None = None
    for anchor in anchors:
        if anchor.frame_index <= frame_index:
            previous = anchor
        if anchor.frame_index >= frame_index:
            next_item = anchor
            break
    if previous is None or next_item is None or previous.xy is None or next_item.xy is None:
        return None
    if previous.frame_index == next_item.frame_index:
        return previous.xy
    span = next_item.frame_index - previous.frame_index
    alpha = (frame_index - previous.frame_index) / float(span)
    return (
        previous.xy[0] + (next_item.xy[0] - previous.xy[0]) * alpha,
        previous.xy[1] + (next_item.xy[1] - previous.xy[1]) * alpha,
    )


def _anchor_gap_frames(anchors: list[BallClickItem], frame_index: int) -> int | None:
    previous: BallClickItem | None = None
    next_item: BallClickItem | None = None
    for anchor in anchors:
        if anchor.frame_index <= frame_index:
            previous = anchor
        if anchor.frame_index >= frame_index:
            next_item = anchor
            break
    if previous is None or next_item is None:
        return None
    return next_item.frame_index - previous.frame_index


def _click_metrics(samples_by_index: dict[int, dict[str, Any]], review: BallClickReview) -> dict[str, Any]:
    visible_items = review.visible_items
    hidden_items = review.hidden_items
    distances: list[float] = []
    visible_hits = 0
    for item in visible_items:
        frame = samples_by_index.get(item.frame_index)
        if frame is None or not bool(frame["visible"]) or item.xy is None:
            continue
        visible_hits += 1
        distances.append(_distance(frame["xy"], item.xy))

    hidden_false_positives = 0
    for item in hidden_items:
        frame = samples_by_index.get(item.frame_index)
        if frame is not None and bool(frame["visible"]):
            hidden_false_positives += 1

    return {
        "visible_click_count": len(visible_items),
        "visible_hit_count": visible_hits,
        "visible_recall": visible_hits / len(visible_items) if visible_items else None,
        "median_error_px": _percentile(distances, 50) if distances else None,
        "p90_error_px": _percentile(distances, 90) if distances else None,
        "hidden_click_count": len(hidden_items),
        "hidden_false_positive_count": hidden_false_positives,
        "negative_false_positive_rate": hidden_false_positives / len(hidden_items) if hidden_items else None,
    }


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0


def _set_frame_xy(frame: dict[str, Any], xy: tuple[float, float], *, conf: float, approx: bool) -> None:
    frame["xy"] = [float(xy[0]), float(xy[1])]
    frame["conf"] = float(conf)
    frame["visible"] = True
    frame["approx"] = bool(approx)


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


__all__ = [
    "BallClickItem",
    "BallClickReview",
    "filter_ball_track_with_click_anchors",
    "load_ball_click_review",
    "write_identity_filtered_ball_track",
]
