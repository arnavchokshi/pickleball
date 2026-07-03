"""Derive reviewed BALL bounce/in-out labels from reviewed CVAT ball boxes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_line_calls import classify_bounce
from .ball_manual_court_inout import manual_court_projection_from_corners
from .court_calibration import project_image_points_to_world
from .court_templates import Sport
from .schemas import CvatVideoAnnotations, validate_artifact_file


REVIEWED_BOUNCES_ARTIFACT_TYPE = "racketsport_reviewed_ball_bounces"
REVIEWED_INOUT_ARTIFACT_TYPE = "racketsport_reviewed_ball_inout"
STATUS_DERIVED = "derived_from_human_reviewed_cvat_boxes"


def build_cvat_reviewed_bounce_inout_labels(
    cvat_labels: Mapping[str, Any] | str | Path,
    court_corners: Mapping[str, Any] | str | Path,
    *,
    fps: float,
    sport: Sport = "pickleball",
    min_vertical_delta_px: float = 2.0,
    min_separation_s: float = 0.10,
    max_frame_gap: int = 2,
    uncertainty_m: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build M4/M5 reviewed-label artifacts from human-reviewed CVAT ball boxes.

    The bounce timing is derived from the reviewed box trajectory, not manually
    clicked as a bounce. Keep the resulting artifact fail-closed and marked
    not-ground-truth until the downstream gate passes.
    """

    labels = _load_cvat_annotations(cvat_labels)
    fps_f = _positive_finite(fps, "fps")
    min_delta = _nonnegative_finite(min_vertical_delta_px, "min_vertical_delta_px")
    min_separation = _nonnegative_finite(min_separation_s, "min_separation_s")
    uncertainty = _nonnegative_finite(uncertainty_m, "uncertainty_m")
    if max_frame_gap < 1:
        raise ValueError("max_frame_gap must be >= 1")

    # The CVAT task already declares the pixel space its ball boxes (and
    # therefore contact_xy_img samples below) live in -- reuse it as the
    # court-corner rescale target instead of assuming corners already match.
    target_image_size = tuple(labels.task.original_size)
    projection = manual_court_projection_from_corners(court_corners, sport=sport, target_image_size=target_image_size)
    samples = _ball_center_samples(labels)
    bounces: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    last_accepted_frame: int | None = None

    for prev, cur, nxt in zip(samples, samples[1:], samples[2:], strict=False):
        if (cur["frame"] - prev["frame"]) > max_frame_gap or (nxt["frame"] - cur["frame"]) > max_frame_gap:
            continue
        dy_prev = float(cur["xy"][1]) - float(prev["xy"][1])
        dy_next = float(nxt["xy"][1]) - float(cur["xy"][1])
        is_ground_contact = dy_prev >= min_delta and dy_next <= -min_delta
        candidates.append(
            {
                "frame": int(cur["frame"]),
                "t": int(cur["frame"]) / fps_f,
                "contact_xy_img": list(cur["xy"]),
                "dy_prev_px": dy_prev,
                "dy_next_px": dy_next,
                "is_ground_contact": is_ground_contact,
            }
        )
        if not is_ground_contact:
            continue
        if last_accepted_frame is not None and ((int(cur["frame"]) - last_accepted_frame) / fps_f) < min_separation:
            continue
        world_xy = project_image_points_to_world(projection["homography"], [cur["xy"]])[0]
        line_call = classify_bounce(
            int(cur["frame"]) / fps_f,
            world_xy,
            sport=sport,
            uncertainty_radius_m=uncertainty,
        )
        court_call = "too_close_to_call" if line_call["court_call"] == "unknown" else str(line_call["court_call"])
        review_id = f"cvat_bounce_{len(bounces):04d}"
        bounces.append({"frame": int(cur["frame"]), "t": int(cur["frame"]) / fps_f, "review_id": review_id})
        if court_call in {"in", "out"}:
            calls.append({"frame": int(cur["frame"]), "t": int(cur["frame"]) / fps_f, "call": court_call, "review_id": review_id})
        last_accepted_frame = int(cur["frame"])

    common = {
        "schema_version": 1,
        "clip": labels.clip_id,
        "fps": fps_f,
        "status": STATUS_DERIVED,
        "source": "cvat_reviewed_ball_boxes_2d_inflection",
        "source_cvat_labels": str(cvat_labels) if isinstance(cvat_labels, str | Path) else None,
        "source_court_corners": str(court_corners) if isinstance(court_corners, str | Path) else None,
        "reviewed_item_count": len(bounces),
        "candidate_count": len(candidates),
        "pending_review_count": 0,
        "rejected_review_count": 0,
        "derivation": {
            "algorithm": "cvat_box_center_image_y_ground_contact_v1",
            "min_vertical_delta_px": min_delta,
            "min_separation_s": min_separation,
            "max_frame_gap": int(max_frame_gap),
            "uncertainty_m": uncertainty,
            "projection": projection,
        },
        "not_ground_truth": True,
    }
    return (
        {
            **common,
            "artifact_type": REVIEWED_BOUNCES_ARTIFACT_TYPE,
            "bounces": bounces,
        },
        {
            **common,
            "artifact_type": REVIEWED_INOUT_ARTIFACT_TYPE,
            "calls": calls,
        },
    )


def write_cvat_reviewed_bounce_inout_labels(
    *,
    cvat_labels_path: str | Path,
    court_corners_path: str | Path,
    fps: float,
    out_bounces: str | Path,
    out_inout: str | Path,
    sport: Sport = "pickleball",
    min_vertical_delta_px: float = 2.0,
    min_separation_s: float = 0.10,
    max_frame_gap: int = 2,
    uncertainty_m: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bounces, inout = build_cvat_reviewed_bounce_inout_labels(
        cvat_labels_path,
        court_corners_path,
        fps=fps,
        sport=sport,
        min_vertical_delta_px=min_vertical_delta_px,
        min_separation_s=min_separation_s,
        max_frame_gap=max_frame_gap,
        uncertainty_m=uncertainty_m,
    )
    out_bounces_path = Path(out_bounces)
    out_inout_path = Path(out_inout)
    out_bounces_path.parent.mkdir(parents=True, exist_ok=True)
    out_inout_path.parent.mkdir(parents=True, exist_ok=True)
    out_bounces_path.write_text(json.dumps(bounces, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_inout_path.write_text(json.dumps(inout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bounces, inout


def _load_cvat_annotations(value: Mapping[str, Any] | str | Path) -> CvatVideoAnnotations:
    if isinstance(value, Mapping):
        try:
            return CvatVideoAnnotations.model_validate(value)
        except Exception as exc:
            raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {exc}") from exc
    parsed = validate_artifact_file("cvat_video_annotations", value)
    if not isinstance(parsed, CvatVideoAnnotations):
        raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {value}")
    return parsed


def _ball_center_samples(labels: CvatVideoAnnotations) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for frame in labels.frames:
        ball_boxes = [box for box in frame.boxes if box.label.strip().lower() == "ball"]
        if len(ball_boxes) > 1:
            raise ValueError(f"multiple ball boxes in {labels.clip_id} frame {frame.frame_index}")
        if not ball_boxes:
            continue
        x1, y1, x2, y2 = ball_boxes[0].bbox_xyxy
        samples.append({"frame": int(frame.frame_index), "xy": [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]})
    return sorted(samples, key=lambda item: int(item["frame"]))


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


__all__ = [
    "STATUS_DERIVED",
    "build_cvat_reviewed_bounce_inout_labels",
    "write_cvat_reviewed_bounce_inout_labels",
]
