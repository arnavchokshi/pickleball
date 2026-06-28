"""Reviewed paddle true-corner labels and RKT promotion review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import RacketCandidates


ARTIFACT_TYPE = "racketsport_paddle_true_corner_review"
LABEL_ARTIFACT_TYPE = "racketsport_paddle_true_corner_labels"
SCHEMA_VERSION = 1
REFERENCE_EVIDENCE_TYPES = {"reference_gt", "aruco_gt", "april_tag_gt", "cad_gt"}
TRUE_CORNER_EVIDENCE_TYPES = {"true_corners", "mask_corners", "keypoint_corners"} | REFERENCE_EVIDENCE_TYPES
BOX_DERIVED_BLOCKER = "box_candidates_are_not_true_paddle_corners"


def true_corner_labels_to_candidates(labels_payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert reviewed paddle-face corner labels into strict racket candidates.

    This intentionally accepts only label payloads whose source is not a draft
    box path. The output still uses ``racket_candidates.json`` because the PnP
    stage consumes true four-corner image points, but the frame sources are
    explicitly non-box evidence.
    """

    if labels_payload.get("artifact_type") != LABEL_ARTIFACT_TYPE:
        raise ValueError(f"artifact_type must be {LABEL_ARTIFACT_TYPE}")
    fps = _positive_float(labels_payload.get("fps"), "fps")
    label_source = _required_text(labels_payload.get("label_source"), "label_source")
    if is_box_derived_source(label_source):
        raise ValueError("label_source must not be box-derived")

    players_payload = labels_payload.get("players")
    if not isinstance(players_payload, list):
        raise ValueError("players must be a list")

    players: list[dict[str, Any]] = []
    accepted = 0
    skipped_invalid = 0
    output_sources: set[str] = set()
    for player_payload in players_payload:
        if not isinstance(player_payload, Mapping):
            skipped_invalid += 1
            continue
        try:
            player_id = _integer(player_payload.get("id"), "player id")
            paddle_dims = _paddle_dims(player_payload.get("paddle_dims_in"))
            frames_payload = player_payload.get("frames")
            if not isinstance(frames_payload, list):
                raise ValueError("player frames must be a list")
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue

        frames: list[dict[str, Any]] = []
        for frame_payload in frames_payload:
            try:
                frame = _true_corner_frame(frame_payload, fps=fps, label_source=label_source)
            except (TypeError, ValueError):
                skipped_invalid += 1
                continue
            frames.append(frame)
            output_sources.add(frame["source"])
            accepted += 1
        if frames:
            players.append({"id": player_id, "paddle_dims_in": paddle_dims, "frames": sorted(frames, key=lambda item: item["t"])})

    if accepted == 0:
        raise ValueError(f"no reviewed true-corner labels accepted; skipped_invalid={skipped_invalid}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_racket_candidates",
        "fps": fps,
        "players": players,
    }
    parsed = RacketCandidates.model_validate(payload).model_dump(mode="json")
    source = next(iter(output_sources)) if len(output_sources) == 1 else "mixed_true_corner_sources"
    return parsed, {"accepted": accepted, "skipped_invalid": skipped_invalid, "source": source}


def build_paddle_true_corner_review(
    *,
    clip: str,
    racket_candidates: RacketCandidates | Mapping[str, Any],
    true_corner_candidates: RacketCandidates | Mapping[str, Any] | None = None,
    crop_sheet_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
    max_required_labels: int | None = None,
) -> dict[str, Any]:
    """Build a fail-closed review artifact for missing true paddle corners."""

    candidates = _candidates(racket_candidates)
    true_candidates = _candidates(true_corner_candidates) if true_corner_candidates is not None else None
    source_counts = _source_counts(candidates)
    source_evidence_counts = _source_evidence_counts(source_counts)
    true_source_counts = _source_counts(true_candidates) if true_candidates is not None else {}
    true_source_evidence_counts = _source_evidence_counts(true_source_counts)
    all_required_labels = _required_labels(candidates, max_items=None)
    required_labels = all_required_labels[:max_required_labels] if max_required_labels is not None else all_required_labels

    true_corner_label_count = true_source_evidence_counts["true_corners_or_pose"]
    reference_gt_count = true_source_evidence_counts["reference_gt"] + source_evidence_counts["reference_gt"]
    blockers: list[str] = []
    if source_evidence_counts["box_derived"]:
        blockers.append(BOX_DERIVED_BLOCKER)
    if true_corner_label_count == 0:
        blockers.append("missing_reviewed_true_corner_labels")
    if reference_gt_count == 0:
        blockers.append("missing_reference_or_cad_gt")

    status = "blocked_missing_true_corner_labels"
    if true_corner_label_count > 0 and reference_gt_count == 0:
        status = "blocked_missing_reference_or_cad_gt"
    if true_corner_label_count > 0 and reference_gt_count > 0:
        status = "true_corner_labels_present_needs_pose_eval"

    visuals = []
    if overlay_path is not None:
        visuals.append({"type": "candidate_overlay_video", "path": str(overlay_path)})
    if crop_sheet_path is not None:
        visuals.append({"type": "true_corner_label_crop_sheet", "path": str(crop_sheet_path)})

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "trusted_for_rkt_promotion": False,
        "candidate_frame_count": sum(source_counts.values()),
        "box_derived_candidate_count": source_evidence_counts["box_derived"],
        "true_corner_label_count": true_corner_label_count,
        "reference_gt_count": reference_gt_count,
        "required_label_count": len(all_required_labels),
        "listed_required_label_count": len(required_labels),
        "required_labels_truncated": len(all_required_labels) > len(required_labels),
        "source_counts": source_counts,
        "source_evidence_counts": source_evidence_counts,
        "true_corner_source_counts": true_source_counts,
        "true_corner_source_evidence_counts": true_source_evidence_counts,
        "required_labels": required_labels,
        "visuals": visuals,
        "promotion_blockers": blockers,
        "label_instructions": [
            "For each listed frame, label the actual four paddle-face corners in top-left, top-right, bottom-right, bottom-left order.",
            "Do not copy the draft box corners into the true-corner label file.",
            "Use source evidence_type=true_corners for human-reviewed image corners or reference_gt/cad_gt for measured reference evidence.",
        ],
        "expected_label_artifact_type": LABEL_ARTIFACT_TYPE,
        "promotion_target": "racket_candidates.json",
    }


def render_paddle_true_corner_crop_sheet(
    *,
    video_path: str | Path,
    racket_candidates: RacketCandidates | Mapping[str, Any],
    output_path: str | Path,
    max_items: int = 48,
    crop_padding_px: int = 24,
    tile_size: int = 180,
) -> dict[str, Any]:
    """Render candidate crops for human true-corner labeling."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for paddle crop sheet rendering") from exc

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if crop_padding_px < 0:
        raise ValueError("crop_padding_px must be non-negative")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    candidates = _candidates(racket_candidates)
    labels = _required_labels(candidates, max_items=max_items)
    video = Path(video_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    tiles: list[Any] = []
    rendered = 0
    try:
        for label in labels:
            frame_index = int(label["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = _crop_xyxy(label["candidate_corners_px"], width, height, crop_padding_px)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            _draw_crop_guides(cv2, tile, label, (x1, y1), (x2 - x1, y2 - y1), tile_size)
            tiles.append(tile)
            rendered += 1
    finally:
        cap.release()

    if not tiles:
        raise ValueError("no paddle candidate crops rendered")

    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = tile
    cv2.imwrite(str(out), sheet)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_paddle_true_corner_crop_sheet",
        "status": "rendered",
        "video_path": str(video),
        "crop_sheet_path": str(out),
        "candidate_frame_count": sum(len(player.frames) for player in candidates.players),
        "requested_crop_count": len(labels),
        "rendered_crop_count": rendered,
        "max_items": max_items,
    }


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_box_derived_source(source: str) -> bool:
    normalized = source.lower()
    return normalized.startswith("label_bbox:") or ":label_bbox:" in normalized or "box_corner" in normalized or "bbox" in normalized


def is_true_corner_source(source: str) -> bool:
    normalized = source.lower()
    if is_box_derived_source(normalized):
        return False
    return any(token in normalized for token in ("true_corner", "mask_corner", "keypoint", "aruco", "april", "tag", "gt", "ground_truth", "reference", "cad", "synthetic"))


def _true_corner_frame(frame_payload: Any, *, fps: float, label_source: str) -> dict[str, Any]:
    if not isinstance(frame_payload, Mapping):
        raise ValueError("frame must be an object")
    _required_text(frame_payload.get("reviewer"), "reviewer")
    evidence_type = _required_text(frame_payload.get("evidence_type"), "evidence_type")
    if evidence_type not in TRUE_CORNER_EVIDENCE_TYPES:
        raise ValueError(f"unsupported evidence_type: {evidence_type}")
    corners = _corners(frame_payload.get("corners_px"))
    conf = _unit_float(frame_payload.get("conf", 1.0), "conf")
    if "frame_index" in frame_payload:
        t = _integer(frame_payload.get("frame_index"), "frame_index") / fps
    else:
        t = _non_negative_float(frame_payload.get("t"), "t")
    source = f"{_source_prefix(evidence_type)}:{label_source}"
    if is_box_derived_source(source):
        raise ValueError("true-corner frame source must not be box-derived")
    return {
        "t": t,
        "corners_px": corners,
        "conf": conf,
        "source": source,
    }


def _source_prefix(evidence_type: str) -> str:
    if evidence_type in REFERENCE_EVIDENCE_TYPES:
        return "reference_gt"
    if evidence_type == "cad_gt":
        return "cad_gt"
    return "true_corners"


def _required_labels(candidates: RacketCandidates, *, max_items: int | None) -> list[dict[str, Any]]:
    required: list[dict[str, Any]] = []
    for player in candidates.players:
        for frame in player.frames:
            if not is_box_derived_source(frame.source):
                continue
            item = {
                "review_id": f"{player.id}_{int(round(frame.t * candidates.fps)):06d}",
                "player_id": player.id,
                "frame_index": int(round(frame.t * candidates.fps)),
                "t": frame.t,
                "candidate_conf": frame.conf,
                "candidate_source": frame.source,
                "candidate_corners_px": frame.corners_px,
                "crop_xyxy": _padded_bbox(frame.corners_px, padding=24),
                "required_output": {
                    "corners_px_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                    "evidence_type": "true_corners",
                    "reviewer": "required",
                },
            }
            required.append(item)
            if max_items is not None and len(required) >= max_items:
                return required
    return required


def _source_counts(candidates: RacketCandidates | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if candidates is None:
        return counts
    for player in candidates.players:
        for frame in player.frames:
            counts[frame.source] = counts.get(frame.source, 0) + 1
    return dict(sorted(counts.items()))


def _source_evidence_counts(source_counts: Mapping[str, int]) -> dict[str, int]:
    evidence = {
        "box_derived": 0,
        "keypoint_or_mask": 0,
        "reference_gt": 0,
        "synthetic_or_cad": 0,
        "true_corners_or_pose": 0,
    }
    for source, count in source_counts.items():
        normalized = source.lower()
        if is_box_derived_source(normalized):
            evidence["box_derived"] += count
        elif any(token in normalized for token in ("aruco", "april", "tag", "gt", "ground_truth", "reference")):
            evidence["reference_gt"] += count
            evidence["true_corners_or_pose"] += count
        elif any(token in normalized for token in ("synthetic", "blenderproc", "cad")):
            evidence["synthetic_or_cad"] += count
            evidence["true_corners_or_pose"] += count
        else:
            evidence["keypoint_or_mask"] += count
            evidence["true_corners_or_pose"] += count
    return evidence


def _candidates(value: RacketCandidates | Mapping[str, Any]) -> RacketCandidates:
    if isinstance(value, RacketCandidates):
        return value
    return RacketCandidates.model_validate(value)


def _corners(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("corners_px must contain four points")
    corners = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("corners_px points must be 2D")
        corners.append([float(point[0]), float(point[1])])
    if abs(_polygon_area(corners)) < 1.0:
        raise ValueError("corners_px polygon area is too small")
    return corners


def _polygon_area(corners: list[list[float]]) -> float:
    area = 0.0
    for point, next_point in zip(corners, [*corners[1:], corners[0]]):
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def _paddle_dims(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("paddle_dims_in must be an object")
    dims = {str(key): float(val) for key, val in value.items()}
    if not ({"length", "width"}.issubset(dims) or {"h", "w"}.issubset(dims)):
        raise ValueError("paddle_dims_in must include length/width or h/w")
    if any(dim <= 0.0 for dim in dims.values()):
        raise ValueError("paddle_dims_in values must be positive")
    return dims


def _padded_bbox(corners: list[list[float]], *, padding: int) -> list[int]:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [
        int(max(0, min(xs) - padding)),
        int(max(0, min(ys) - padding)),
        int(max(xs) + padding),
        int(max(ys) + padding),
    ]


def _crop_xyxy(corners: list[list[float]], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _padded_bbox(corners, padding=padding)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _draw_crop_guides(cv2: Any, tile: Any, label: Mapping[str, Any], origin: tuple[int, int], crop_size: tuple[int, int], tile_size: int) -> None:
    scale_x = tile_size / float(max(1, crop_size[0]))
    scale_y = tile_size / float(max(1, crop_size[1]))
    corners = [
        (int(round((point[0] - origin[0]) * scale_x)), int(round((point[1] - origin[1]) * scale_y)))
        for point in label["candidate_corners_px"]
    ]
    color = (80, 220, 255)
    for start, end in zip(corners, [*corners[1:], corners[0]]):
        cv2.line(tile, start, end, color, 1, getattr(cv2, "LINE_AA", 16))
    for index, corner in enumerate(corners):
        cv2.circle(tile, corner, 3, color, -1, getattr(cv2, "LINE_AA", 16))
        cv2.putText(tile, str(index + 1), (corner[0] + 3, corner[1] - 3), getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0), 0.35, (255, 255, 255), 1)
    cv2.putText(tile, f"f{label['frame_index']} p{label['player_id']}", (6, 16), getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0), 0.45, (255, 255, 255), 1)


def _positive_float(value: Any, field: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _non_negative_float(value: Any, field: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _unit_float(value: Any, field: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


__all__ = [
    "ARTIFACT_TYPE",
    "BOX_DERIVED_BLOCKER",
    "LABEL_ARTIFACT_TYPE",
    "build_paddle_true_corner_review",
    "is_box_derived_source",
    "is_true_corner_source",
    "load_json_object",
    "render_paddle_true_corner_crop_sheet",
    "true_corner_labels_to_candidates",
    "write_json_artifact",
]
