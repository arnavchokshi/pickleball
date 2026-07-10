"""Train and evaluate the legacy court-keypoint heatmap model.

Court-keypoint label items use one additive partial-label schema. ``keypoints`` must still
contain exactly the 15 canonical names. A labeled point keeps the existing two-number
``[x, y]`` coordinate value; an unlabeled point is JSON ``null``. Missing names,
``{"labeled": false}`` objects, and every other marker fail loudly. ``null`` means there is no
supervision for that channel; it does not mean occluded. An occluded-but-known point remains a
labeled ``[x, y]`` coordinate. Loaded rows expose only labeled coordinates in ``row["keypoints"]``
so target construction produces a zero per-pixel mask for every unlabeled channel, and metrics
aggregate only labeled points. Existing full-15 rows therefore follow the original path.

``label_status == "reviewed_external_dataset"`` denotes a human-annotated third-party dataset.
It is accepted for training and counted only as ``labels_external_dataset_frame_count``. It is
never included in ``labels_independent_human_frames`` or the owner gate's independent buckets.
The existing ``reviewed`` status remains the only independent owner-human status.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_keypoint_geometric_loss import court_geometric_consistency_loss
from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINT_BY_NAME,
    PICKLEBALL_KEYPOINTS,
    court_keypoint_heatmap_loss,
    court_keypoint_probabilities,
    decode_subpixel_heatmap,
    keypoint_labels_from_court_corners,
    make_court_keypoint_heatmap_model,
    refine_keypoint_xy_with_planar_homography,
)
from threed.racketsport.court_keypoint_lines import (
    COURT_LINE_FAMILIES,
    fit_court_lines_from_masks,
    intersect_court_keypoints_from_lines,
    line_mask_targets_for_keypoints,
    validate_round3_input_resolution,
)
from threed.racketsport.eval_guard import assert_not_training_on_eval_clip

# Court-keypoint review item status enum, mirrored from
# scripts/racketsport/court_keypoint_review_server.py. "reviewed" is an independent human
# review of that exact frame; "reviewed_static_camera_copy" is an owner-approved copy of an
# independent review of another frame from the same static-camera clip. "synthetic" is a
# procedurally-rendered domain-randomized court render (scripts/racketsport/
# generate_synthetic_court_keypoints.py, runs/training_corpora_20260701/court_synthetic/) with
# geometrically-exact but never human-verified labels. "reviewed_external_dataset" is a
# third-party human annotation whose provenance is not owner-independent gate truth. All four
# are accepted as usable training rows, but only "reviewed" counts toward the independent
# human-verified frame count reported in the training/gate summary.
#
# CAL-R2 provenance fix (2026-07-02): "synthetic" used to be smuggled in under
# "reviewed_static_camera_copy" (an enum workaround -- see the CAL-R2 REPORT.md in this run's
# directory), which meant loading the 2,000-image synthetic corpus silently inflated
# `labels_static_camera_copy_frame_count` -- a field whose whole purpose is to separate
# owner-approved REAL human-review copies from anything else -- with thousands of synthetic
# rows. "synthetic" is now its own status so gates can never count synthetic frames as human
# (real or copied) verification. See `labels_synthetic_frame_count` below.
INDEPENDENT_REVIEWED_STATUS = "reviewed"
STATIC_CAMERA_COPY_STATUS = "reviewed_static_camera_copy"
SYNTHETIC_STATUS = "synthetic"
EXTERNAL_DATASET_STATUS = "reviewed_external_dataset"
ACCEPTED_ITEM_STATUSES = {
    INDEPENDENT_REVIEWED_STATUS,
    STATIC_CAMERA_COPY_STATUS,
    SYNTHETIC_STATUS,
    EXTERNAL_DATASET_STATUS,
}


def court_corner_keypoint_labels(payload: dict[str, Any], *, clip_root: Path | None = None) -> dict[str, Any]:
    items = _items(payload)
    item = items[0]
    if not isinstance(item, dict):
        raise ValueError("court corner item must be an object")

    frame_name = item.get("frame")
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError("court corner item requires frame")

    corners = item.get("court_corners")
    if not isinstance(corners, dict):
        raise ValueError("court corner item requires court_corners")

    frame_dir = _frame_dir(payload)
    source_size = _source_resolution(payload)
    label_size = _infer_label_coordinate_space(corners, source_size=source_size)
    scaled_corners = _scale_corner_labels(corners, label_size=label_size, source_size=source_size)
    image_path = frame_dir / frame_name
    video_path = clip_root / "source.mp4" if clip_root is not None else _payload_source_video(payload)
    return {
        "image_path": str(image_path) if image_path.is_file() else None,
        "video_path": str(video_path) if video_path is not None else None,
        "frame_index": _frame_index_from_name(frame_name),
        "label_coordinate_space": list(label_size) if label_size is not None else None,
        "source_video_size": list(source_size) if source_size is not None else None,
        "keypoints": keypoint_labels_from_court_corners(scaled_corners),
    }


def _items(payload: dict[str, Any]) -> list[Any]:
    annotation = payload.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("court corner item annotation missing")
    items = annotation.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("court corner item missing")
    return items


def _frame_dir(payload: dict[str, Any]) -> Path:
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("frames block missing")
    frame_dir = frames.get("frame_dir")
    if not isinstance(frame_dir, str) or not frame_dir:
        raise ValueError("frames.frame_dir missing")
    return Path(frame_dir)


def _source_resolution(payload: dict[str, Any]) -> tuple[int, int] | None:
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        return None
    source_resolution = frames.get("source_resolution")
    if not isinstance(source_resolution, list) or len(source_resolution) != 2:
        return None
    width, height = source_resolution
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _infer_label_coordinate_space(
    corners: dict[str, Any],
    *,
    source_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if source_size is None:
        return None
    source_width, source_height = source_size
    numeric = [_corner_xy(corners, name) for name in ("near_left", "near_right", "far_right", "far_left")]
    max_x = max(point[0] for point in numeric)
    max_y = max(point[1] for point in numeric)
    half_width = source_width / 2.0
    half_height = source_height / 2.0
    if max_x <= half_width + 1.0 and max_y <= half_height + 1.0:
        return (int(round(half_width)), int(round(half_height)))
    return source_size


def _scale_corner_labels(
    corners: dict[str, Any],
    *,
    label_size: tuple[int, int] | None,
    source_size: tuple[int, int] | None,
) -> dict[str, list[float]]:
    if label_size is None or source_size is None:
        return {name: _corner_xy(corners, name) for name in ("near_left", "near_right", "far_right", "far_left")}
    scale_x = source_size[0] / float(label_size[0])
    scale_y = source_size[1] / float(label_size[1])
    return {
        name: [xy[0] * scale_x, xy[1] * scale_y]
        for name in ("near_left", "near_right", "far_right", "far_left")
        for xy in [_corner_xy(corners, name)]
    }


def _corner_xy(corners: dict[str, Any], key: str) -> list[float]:
    value = corners.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"court corner item missing {key}")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"court corner item has non-numeric {key}")
    return [float(x), float(y)]


def _payload_source_video(payload: dict[str, Any]) -> Path | None:
    clip = payload.get("clip")
    if not isinstance(clip, dict):
        return None
    source_video = clip.get("source_video")
    if not isinstance(source_video, str) or not source_video:
        return None
    return Path(source_video)


def _frame_index_from_name(frame_name: str) -> int:
    stem = Path(frame_name).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse frame index from {frame_name}") from exc


def load_real_corner_labels(root: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/labels/court_corners.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = court_corner_keypoint_labels(payload, clip_root=path.parent.parent)
        row["clip"] = path.parent.parent.name
        labels.append(row)
    return labels


def _label_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Split label rows into owner-independent, copied, synthetic, and external counts."""
    independent = sum(1 for row in rows if row.get("label_status") == INDEPENDENT_REVIEWED_STATUS)
    copied = sum(1 for row in rows if row.get("label_status") == STATIC_CAMERA_COPY_STATUS)
    synthetic = sum(1 for row in rows if row.get("label_status") == SYNTHETIC_STATUS)
    external = sum(1 for row in rows if row.get("label_status") == EXTERNAL_DATASET_STATUS)
    return {
        "labels_independent_human_frames": independent,
        "labels_static_camera_copy_frame_count": copied,
        "labels_synthetic_frame_count": synthetic,
        "labels_external_dataset_frame_count": external,
    }


def load_real_court_keypoint_labels(root: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/labels/court_keypoints.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in court_keypoint_label_rows(payload, clip_root=path.parent.parent):
            row["clip"] = path.parent.parent.name
            labels.append(row)
    if not labels:
        raise ValueError(f"no reviewed 15-keypoint court labels found under {root}")
    return labels


def _load_real_court_keypoint_labels_from_roots(real_root: Any) -> list[dict[str, Any]]:
    """Normalize ``args.real_root`` (None, one Path, or a list of Paths) and merge every root's
    labels into a single row list.

    Supporting multiple roots lets a single training run combine several external-corpus tiers
    (e.g. ``--real-root .../tier1_direct --real-root .../tier2_homography_derived``) without
    needing a merged/symlinked directory on disk -- useful both locally and on a fresh VM clone
    where symlinks may not have been synced.
    """

    if real_root is None:
        return []
    roots = [real_root] if isinstance(real_root, (str, Path)) else list(real_root)
    labels: list[dict[str, Any]] = []
    for root in roots:
        labels.extend(load_real_court_keypoint_labels(Path(root)))
    return labels


def court_keypoint_label_rows(payload: dict[str, Any], *, clip_root: Path | None = None) -> list[dict[str, Any]]:
    items = _items(payload)
    _require_reviewed(payload, items=items)
    return [_court_keypoint_label_row_from_item(payload, item, clip_root=clip_root) for item in items]


def court_keypoint_label_row(payload: dict[str, Any], *, clip_root: Path | None = None) -> dict[str, Any]:
    items = _items(payload)
    _require_reviewed(payload, items=items)
    return _court_keypoint_label_row_from_item(payload, items[0], clip_root=clip_root)


def _court_keypoint_label_row_from_item(
    payload: dict[str, Any],
    item: Any,
    *,
    clip_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("court keypoint item must be an object")

    frame_name = item.get("frame")
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError("court keypoint item requires frame")

    keypoints = item.get("keypoints")
    if not isinstance(keypoints, dict):
        raise ValueError("court keypoint item requires keypoints")
    expected_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    if set(keypoints) != set(expected_names):
        missing = sorted(set(expected_names) - set(keypoints))
        extra = sorted(set(keypoints) - set(expected_names))
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError("court keypoint labels must contain exactly the 15 canonical keypoints: " + "; ".join(details))

    frame_dir = _frame_dir(payload)
    source_size = _source_resolution(payload)
    label_size = _label_coordinate_space(payload)
    raw_keypoints: dict[str, list[float]] = {}
    for name in expected_names:
        value = keypoints[name]
        if value is None:
            continue
        raw_keypoints[name] = list(_xy_field(value, f"court_keypoints.{name}"))
    if not raw_keypoints:
        raise ValueError("court keypoint labels must label at least one canonical keypoint")
    source_keypoints = _scale_keypoint_labels_to_source(raw_keypoints, label_size=label_size, source_size=source_size)
    image_path = frame_dir / frame_name
    video_path = clip_root / "source.mp4" if clip_root is not None else _payload_source_video(payload)
    return {
        "image_path": str(image_path) if image_path.is_file() else None,
        "video_path": str(video_path) if video_path is not None else None,
        "frame_index": _frame_index_from_name(frame_name),
        "label_coordinate_space": list(label_size) if label_size is not None else None,
        "source_video_size": list(source_size) if source_size is not None else None,
        "keypoints": source_keypoints,
        "label_source": (
            "reviewed_15_keypoint_court_labels"
            if len(source_keypoints) == len(expected_names)
            else "reviewed_partial_court_keypoint_labels"
        ),
        # Provenance of this specific frame's label: an independent human review, an
        # owner-approved copy of another frame's independent review on the same static
        # camera, a synthetic domain-randomized render, or a human-annotated external dataset.
        # Training/gate summaries count all four separately so only owner "reviewed" rows can
        # enter the independent-human bucket.
        "label_status": item.get("status", INDEPENDENT_REVIEWED_STATUS),
    }


def _require_reviewed(payload: dict[str, Any], *, items: list[Any] | None = None) -> None:
    review = payload.get("review")
    if not isinstance(review, dict) or review.get("status") != "reviewed":
        raise ValueError("court_keypoints labels must have review.status == reviewed")
    for item in items if items is not None else _items(payload):
        if not isinstance(item, dict):
            continue
        status = item.get("status", INDEPENDENT_REVIEWED_STATUS)
        if status not in ACCEPTED_ITEM_STATUSES:
            raise ValueError(
                "court_keypoints item status must be one of "
                f"{sorted(ACCEPTED_ITEM_STATUSES)}; got {status!r}"
            )


def _label_coordinate_space(payload: dict[str, Any]) -> tuple[int, int] | None:
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        return None
    raw = frames.get("label_coordinate_space")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    width, height = raw
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _scale_keypoint_labels_to_source(
    keypoints: dict[str, list[float]],
    *,
    label_size: tuple[int, int] | None,
    source_size: tuple[int, int] | None,
) -> dict[str, list[float]]:
    if label_size is None or source_size is None:
        return {name: list(xy) for name, xy in keypoints.items()}
    scale_x = source_size[0] / float(label_size[0])
    scale_y = source_size[1] / float(label_size[1])
    return {name: [xy[0] * scale_x, xy[1] * scale_y] for name, xy in keypoints.items()}


def _xy_field(raw_value: Any, name: str) -> tuple[float, float]:
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(f"{name} must be a two-item image coordinate")
    return (_finite_float(raw_value[0], name), _finite_float(raw_value[1], name))


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def heatmaps_for_points(
    points: dict[str, list[float] | tuple[float, float]],
    keypoint_names: list[str],
    width: int,
    height: int,
    *,
    sigma: float,
) -> tuple[Any, Any]:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    heatmaps = np.zeros((len(keypoint_names), height, width), dtype=np.float32)
    masks = np.zeros_like(heatmaps, dtype=np.float32)
    for idx, name in enumerate(keypoint_names):
        if name not in points:
            continue
        x, y = float(points[name][0]), float(points[name][1])
        heatmaps[idx] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        masks[idx] = 1.0
    return heatmaps, masks


def line_masks_for_points(
    points: dict[str, list[float] | tuple[float, float]],
    width: int,
    height: int,
    *,
    line_width: int,
) -> tuple[Any, Any]:
    import numpy as np

    line_masks = line_mask_targets_for_keypoints(points, width=width, height=height, line_width=line_width)
    ordered = np.stack([line_masks[family.name] for family in COURT_LINE_FAMILIES], axis=0).astype(np.float32)
    return ordered, np.ones_like(ordered, dtype=np.float32)


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _error_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "median": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "max": float(ordered[-1]),
    }


def _pck_at_threshold(values: list[float], threshold_px: float) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if float(value) <= threshold_px) / len(values))


def _pck_error_summary(values: list[float], threshold_px: float) -> dict[str, float | int | None]:
    summary = _error_summary(values)
    summary["keypoint_count"] = summary["count"]
    summary["pck_at_5px"] = _pck_at_threshold(values, threshold_px)
    summary["pck_threshold_px"] = threshold_px
    return summary


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    rank = (len(ordered_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered_values[low])
    weight = rank - low
    return float(ordered_values[low] * (1.0 - weight) + ordered_values[high] * weight)


def _world_bounds() -> tuple[float, float, float, float]:
    xs = [point.world_xyz_m[0] for point in PICKLEBALL_KEYPOINTS]
    ys = [point.world_xyz_m[1] for point in PICKLEBALL_KEYPOINTS]
    return min(xs), max(xs), min(ys), max(ys)


def _bilinear(point: tuple[float, float], quad: list[tuple[float, float]]) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = _world_bounds()
    u = (point[0] - min_x) / (max_x - min_x)
    v = (point[1] - min_y) / (max_y - min_y)
    near_left, near_right, far_right, far_left = quad
    x = (1 - u) * (1 - v) * near_left[0] + u * (1 - v) * near_right[0] + u * v * far_right[0] + (1 - u) * v * far_left[0]
    y = (1 - u) * (1 - v) * near_left[1] + u * (1 - v) * near_right[1] + u * v * far_right[1] + (1 - u) * v * far_left[1]
    return x, y


def _random_quad(width: int, height: int, rng: random.Random) -> list[tuple[float, float]]:
    margin_x = width * rng.uniform(0.04, 0.18)
    near_y = height * rng.uniform(0.70, 0.93)
    far_y = height * rng.uniform(0.12, 0.45)
    near_left = (margin_x + rng.uniform(-6, 6), near_y + rng.uniform(-6, 6))
    near_right = (width - margin_x + rng.uniform(-6, 6), near_y + rng.uniform(-6, 6))
    far_width = width * rng.uniform(0.25, 0.70)
    far_center = width * rng.uniform(0.40, 0.60)
    far_left = (far_center - far_width / 2 + rng.uniform(-5, 5), far_y + rng.uniform(-5, 5))
    far_right = (far_center + far_width / 2 + rng.uniform(-5, 5), far_y + rng.uniform(-5, 5))
    return [near_left, near_right, far_right, far_left]


def predict_source_keypoints(
    model: Any,
    row: dict[str, Any],
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    width: int,
    height: int,
    keypoint_names: list[str],
    use_homography_refinement: bool = False,
) -> dict[str, list[float]]:
    """Run the model on one labeled row, returning predicted keypoints in source-video pixel
    space. This is the single-frame prediction primitive shared by training-time evaluation
    (`run_training`) and the standalone post-hoc checkpoint gate evaluator
    (`evaluate_checkpoint_against_real_labels`) -- both must decode predictions identically so a
    frozen checkpoint's held-out gate number matches what training-time logging showed.
    """
    image = load_label_image(row, cv2=cv2, image_module=image_module)
    label_w, label_h = _label_coordinate_size(row, fallback_size=image.size)
    resized = image.resize((width, height))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = court_keypoint_probabilities(_keypoint_heatmap_logits(model(tensor))).detach().cpu()[0]
    sx, sy = width / label_w, height / label_h
    predicted_source_points: dict[str, list[float]] = {}
    for name in row["keypoints"]:
        idx = keypoint_names.index(name)
        flat = int(pred[idx].argmax())
        py, px = divmod(flat, width)
        predicted_source_points[name] = [px / sx, py / sy]
    if use_homography_refinement:
        predicted_source_points = refine_keypoint_xy_with_planar_homography(predicted_source_points)
    return predicted_source_points


def _keypoint_heatmap_logits(model_output: Any) -> Any:
    if isinstance(model_output, dict):
        if "keypoint_heatmaps" not in model_output:
            raise ValueError("court keypoint model output dict is missing keypoint_heatmaps")
        return model_output["keypoint_heatmaps"]
    return model_output


def predict_source_keypoints_from_line_model(
    model: Any,
    row: dict[str, Any],
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    width: int,
    height: int,
    line_names: list[str],
) -> dict[str, Any]:
    image = load_label_image(row, cv2=cv2, image_module=image_module)
    label_w, label_h = _label_coordinate_size(row, fallback_size=image.size)
    resized = image.resize((width, height))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).sigmoid().detach().cpu()[0].numpy()
    masks = {
        name: _adaptive_line_probability_mask(probabilities[index], np=np)
        for index, name in enumerate(line_names)
    }
    try:
        lines = fit_court_lines_from_masks(masks, threshold=0.5)
        model_points = intersect_court_keypoints_from_lines(lines)
    except ValueError as exc:
        return {"keypoints": {}, "line_fit_rms_px": None, "line_fit_error": str(exc)}

    scale_x = label_w / float(width)
    scale_y = label_h / float(height)
    return {
        "keypoints": {
            name: [xy[0] * scale_x, xy[1] * scale_y]
            for name, xy in model_points.items()
        },
        "line_fit_rms_px": _error_summary([line.rms_px for line in lines.values()]),
        "line_fit_error": None,
    }


def _adaptive_line_probability_mask(channel: Any, *, np: Any, min_support_px: int = 64) -> Any:
    arr = np.asarray(channel, dtype=np.float32)
    for percentile in (99.7, 99.3, 99.0, 98.0, 96.0, 92.0):
        threshold = float(np.percentile(arr, percentile))
        mask = (arr >= threshold).astype(np.float32)
        if int(mask.sum()) >= min_support_px:
            return mask
    return (arr >= float(arr.mean())).astype(np.float32)


def aggregate_static_camera_predictions(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    width: int,
    height: int,
    keypoint_names: list[str],
    use_homography_refinement: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Per-clip median of per-frame model predictions -- the CAL static-camera aggregation
    policy (`NORTH_STAR_ROADMAP.md`, "Gate ladder" section).

    EVAL INTEGRITY (binding): ``rows`` must be the exact set of rows being scored (e.g. the
    held-out rows themselves, or the owner-clip gate rows), never a separate training set.
    Aggregating a clip's *own* held-out frame predictions together is a legitimate noise-
    reduction technique for a camera that does not move -- true court geometry is constant
    across those frames, so per-frame heatmap decode noise partially cancels under a median.
    Aggregating over frames the model was *trained* on instead would leak train-time
    memorization into a number reported as a held-out gate -- exactly the mistake flagged by
    the "CAL static-camera aggregation policy" note (the prior uncommitted one-off PCK@5 1.0
    check used train-side aggregation against the same checkpoint it was trained with). Every
    caller in this module passes only held-out/gate rows here; do not reintroduce a train-row
    caller.
    """
    by_clip: dict[str, dict[str, list[list[float]]]] = {}
    for row in rows:
        clip = str(row.get("clip") or "unknown")
        predicted = predict_source_keypoints(
            model,
            row,
            cv2=cv2,
            np=np,
            torch=torch,
            image_module=image_module,
            device=device,
            width=width,
            height=height,
            keypoint_names=keypoint_names,
            use_homography_refinement=use_homography_refinement,
        )
        clip_predictions = by_clip.setdefault(clip, {name: [] for name in keypoint_names})
        for name, xy in predicted.items():
            clip_predictions.setdefault(name, []).append(xy)
    return {
        clip: {
            name: [
                float(statistics.median([xy[0] for xy in values])),
                float(statistics.median([xy[1] for xy in values])),
            ]
            for name, values in keypoints.items()
            if values
        }
        for clip, keypoints in by_clip.items()
    }


def load_court_keypoint_checkpoint(checkpoint_path: Path, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    map_location = device if device == "cuda" else "cpu"
    # This repo-owned checkpoint stores training metadata (e.g. argparse Namespace/pathlib
    # values), so it cannot be loaded with weights_only=True.
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"court-keypoint checkpoint must contain a model state dict: {checkpoint_path}")
    return payload


def build_model_from_checkpoint(payload: dict[str, Any], *, device: str = "cpu") -> tuple[Any, list[str], int, int]:
    keypoint_names = [
        str(name) for name in payload.get("keypoint_names", [point.name for point in PICKLEBALL_KEYPOINTS])
    ]
    image_size = payload.get("image_size", [160, 90])
    model_width, model_height = int(image_size[0]), int(image_size[1])
    logical_architecture = str(payload.get("model_architecture", "keypoint_heatmap_v1"))
    architecture = str(payload.get("network_architecture", payload.get("model_architecture", "encoder_decoder_v1")))
    if logical_architecture == "line_segmentation_intersection_v1":
        output_count = len(payload.get("line_names") or [family.name for family in COURT_LINE_FAMILIES])
    else:
        output_count = len(keypoint_names)
    model = make_court_keypoint_heatmap_model(output_count, architecture=architecture)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, keypoint_names, model_width, model_height


def _homography_self_consistency_px(predicted: dict[str, list[float]]) -> dict[str, float | int | None] | None:
    """Fit ONE planar homography from ``predicted``'s points to their known regulation-court
    world XY, then measure how well that single homography explains the same points
    (reprojection residual, px).

    This is a cheap downstream calibration-quality proxy: it reuses the same planar-homography
    machinery `threed/racketsport/court_calibration.py` uses for real solvePnP-ready
    calibration, but -- unlike solvePnP -- needs no camera intrinsics. A full solvePnP
    reprojection check was intentionally skipped for the owner-clip gate: no per-clip camera
    intrinsics (focal length, principal point) exist for `eval_clips/ball`'s four clips, and
    assuming arbitrary intrinsics would produce a number that looks precise but is not
    grounded in anything real. Returns ``None`` when fewer than 4 points are available (a
    homography is underdetermined below 4 correspondences).
    """
    names = [name for name in predicted if name in PICKLEBALL_KEYPOINT_BY_NAME]
    if len(names) < 4:
        return None
    world_pts = [PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[:2] for name in names]
    image_pts = [predicted[name] for name in names]
    try:
        homography = homography_from_planar_points(world_pts, image_pts)
        projected = project_planar_points(homography, world_pts)
    except ValueError:
        return None
    residuals = [math.hypot(p[0] - i[0], p[1] - i[1]) for p, i in zip(projected, image_pts, strict=True)]
    return _error_summary(residuals)


def evaluate_checkpoint_against_real_labels(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    *,
    device: str = "cpu",
    use_homography_refinement: bool = False,
    pck_threshold_px: float = 5.0,
) -> dict[str, Any]:
    """Score a frozen checkpoint against real labeled rows, honestly reporting the raw
    per-frame metric side by side with the static-camera per-clip median-aggregation metric,
    each split into "independent human frames only" (primary) and "all rows" (secondary).

    Partial rows are scored only on coordinates retained in ``row["keypoints"]``; each mode's
    ``per_row`` entries expose the resulting labeled-only ``keypoint_count``. External-dataset
    rows can enter the all-rows modes, but only status ``reviewed`` enters independent modes.

    This is the single reusable implementation behind the CAL owner-clip gate evaluation
    (`scripts/racketsport/evaluate_court_keypoint_owner_gate.py`). It is read-only inference +
    scoring: it never fits or mutates ``checkpoint_path``, and the caller is responsible for
    ensuring ``rows`` were never used to train it (e.g. the 32 `eval_clips/ball/*/labels/
    court_keypoints.json` rows, which this module's training path never reads when pointed at
    an external corpus root). Aggregation, when scored, is computed over ``rows`` themselves --
    see `aggregate_static_camera_predictions`'s eval-integrity note.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    payload = load_court_keypoint_checkpoint(checkpoint_path, device=device)
    model, keypoint_names, model_width, model_height = build_model_from_checkpoint(payload, device=device)

    raw_predictions: list[dict[str, list[float]]] = [
        predict_source_keypoints(
            model,
            row,
            cv2=cv2,
            np=np,
            torch=torch,
            image_module=Image,
            device=device,
            width=model_width,
            height=model_height,
            keypoint_names=keypoint_names,
            use_homography_refinement=use_homography_refinement,
        )
        for row in rows
    ]
    aggregated_by_clip = aggregate_static_camera_predictions(
        model,
        rows,
        cv2=cv2,
        np=np,
        torch=torch,
        image_module=Image,
        device=device,
        width=model_width,
        height=model_height,
        keypoint_names=keypoint_names,
        use_homography_refinement=use_homography_refinement,
    )

    def _row_errors(row: dict[str, Any], predicted: dict[str, list[float]]) -> list[float]:
        errors: list[float] = []
        for name, xy in row["keypoints"].items():
            if name not in predicted:
                continue
            px, py = predicted[name]
            errors.append(math.hypot(px - xy[0], py - xy[1]))
        return errors

    def _summarize(selected_indices: list[int], *, aggregated: bool) -> dict[str, Any]:
        errors: list[float] = []
        by_clip: dict[str, list[float]] = {}
        per_row: list[dict[str, Any]] = []
        for index in selected_indices:
            row = rows[index]
            clip = str(row.get("clip") or "unknown")
            predicted = aggregated_by_clip.get(clip, {}) if aggregated else raw_predictions[index]
            row_errors = _row_errors(row, predicted) if predicted else []
            errors.extend(row_errors)
            by_clip.setdefault(clip, []).extend(row_errors)
            per_row.append(
                {
                    "row_index": index,
                    "clip": clip,
                    "frame_index": row.get("frame_index"),
                    "keypoint_count": len(row_errors),
                    "keypoint_error_summary": _error_summary(row_errors),
                    "pck_at_5px": _pck_at_threshold(row_errors, pck_threshold_px),
                }
            )
        return {
            "mode": "aggregated_static_camera_median" if aggregated else "raw_per_frame",
            "frame_count": len(selected_indices),
            "keypoint_error_summary": _error_summary(errors),
            "pck_at_5px": _pck_at_threshold(errors, pck_threshold_px),
            "per_row": per_row,
            "per_clip": {
                clip: _pck_error_summary(clip_errors, pck_threshold_px) for clip, clip_errors in sorted(by_clip.items())
            },
        }

    independent_indices = [
        index for index, row in enumerate(rows) if row.get("label_status") == INDEPENDENT_REVIEWED_STATUS
    ]
    all_indices = list(range(len(rows)))

    # Downstream reprojection-quality proxy (see `_homography_self_consistency_px`): one raw
    # representative prediction per clip (the independent frame's own prediction when present)
    # and the aggregated per-clip median prediction, each checked for planar self-consistency.
    raw_representative_by_clip: dict[str, dict[str, list[float]]] = {}
    for index, row in enumerate(rows):
        clip = str(row.get("clip") or "unknown")
        is_independent = row.get("label_status") == INDEPENDENT_REVIEWED_STATUS
        if clip not in raw_representative_by_clip or is_independent:
            raw_representative_by_clip[clip] = raw_predictions[index]
    homography_self_consistency = {
        "raw_representative_per_clip": {
            clip: _homography_self_consistency_px(predicted) for clip, predicted in sorted(raw_representative_by_clip.items())
        },
        "aggregated_per_clip": {
            clip: _homography_self_consistency_px(predicted) for clip, predicted in sorted(aggregated_by_clip.items())
        },
        "note": (
            "Planar-homography self-consistency reprojection error (px), not a full solvePnP "
            "check -- see _homography_self_consistency_px() docstring for why solvePnP was "
            "skipped (no per-clip camera intrinsics exist for these owner clips)."
        ),
    }

    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_owner_gate_report",
        "checkpoint": str(checkpoint_path),
        "pck_threshold_px": pck_threshold_px,
        "independent_frame_count": len(independent_indices),
        "all_frame_count": len(all_indices),
        "raw_independent": _summarize(independent_indices, aggregated=False),
        "raw_all": _summarize(all_indices, aggregated=False),
        "aggregated_independent": _summarize(independent_indices, aggregated=True),
        "aggregated_all": _summarize(all_indices, aggregated=True),
        "homography_self_consistency": homography_self_consistency,
        "notes": [
            "raw_* scores each row from its own individual prediction.",
            "aggregated_* scores every row in a clip against that clip's per-clip median "
            "prediction, computed only from the rows passed in here (never training rows).",
            "*_independent uses only status=='reviewed' rows (independent human labels); "
            "*_all additionally includes copied, synthetic, and external-dataset rows when "
            "the caller supplies them; external-dataset rows never enter *_independent.",
            "per_row keypoint_count and every aggregate metric include labeled coordinates "
            "only; null/unlabeled channels are skipped.",
        ],
    }


def curriculum_synthetic_fraction(
    epoch: int,
    total_epochs: int,
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    """CAL-R2 synthetic/real curriculum: linear ramp of the synthetic-corpus fraction of each
    real-finetune mini-batch, from ``start_fraction`` at epoch 0 to ``end_fraction`` at the
    final epoch. Default direction (see the CLI flags below) is synthetic-heavy early -- the net
    sees clean, geometrically-exact labels while it is still learning coarse localization,
    before real-camera appearance noise/occlusion/domain gap are introduced -- ramping to
    real-heavy late, so the final weights are dominated by real-camera statistics rather than
    the synthetic renderer's domain gap."""

    if total_epochs <= 1:
        return end_fraction
    t = max(0.0, min(1.0, epoch / (total_epochs - 1)))
    return start_fraction + (end_fraction - start_fraction) * t


def sample_curriculum_real_batch(
    train_real: list[dict[str, Any]],
    *,
    epoch: int,
    total_epochs: int,
    real_batch_size: int | None,
    synthetic_curriculum_start_fraction: float,
    synthetic_curriculum_end_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample this epoch's real-finetune mini-batch, optionally following the CAL-R2
    synthetic/real curriculum (`curriculum_synthetic_fraction`).

    Rows are split into two pools by ``label_status``: ``SYNTHETIC_STATUS`` (the CAL-R2
    domain-randomized corpus) vs everything else (independent-human / owner-approved-copy real
    rows). When both pools are non-empty and a nonzero curriculum fraction range is configured,
    each epoch's mini-batch mixes the two pools according to the ramped fraction; otherwise
    (no synthetic-status rows present in ``train_real``, or both curriculum fractions are 0)
    this reduces exactly to the original uniform ``rng.sample(train_real, real_batch_size)``
    behavior, so existing non-curriculum runs are unaffected.
    """

    if real_batch_size is None or len(train_real) <= real_batch_size:
        return train_real

    synthetic_pool = [row for row in train_real if row.get("label_status") == SYNTHETIC_STATUS]
    human_pool = [row for row in train_real if row.get("label_status") != SYNTHETIC_STATUS]
    if not synthetic_pool or not human_pool or (synthetic_curriculum_start_fraction <= 0 and synthetic_curriculum_end_fraction <= 0):
        return rng.sample(train_real, real_batch_size)

    fraction = curriculum_synthetic_fraction(
        epoch,
        total_epochs,
        start_fraction=synthetic_curriculum_start_fraction,
        end_fraction=synthetic_curriculum_end_fraction,
    )
    synthetic_n = min(len(synthetic_pool), round(real_batch_size * fraction))
    human_n = min(len(human_pool), real_batch_size - synthetic_n)
    # Backfill from whichever pool has slack if the other was too small to hit real_batch_size.
    remaining = real_batch_size - synthetic_n - human_n
    if remaining > 0 and len(synthetic_pool) > synthetic_n:
        extra = min(remaining, len(synthetic_pool) - synthetic_n)
        synthetic_n += extra
        remaining -= extra
    if remaining > 0 and len(human_pool) > human_n:
        human_n += min(remaining, len(human_pool) - human_n)

    sample = (rng.sample(synthetic_pool, synthetic_n) if synthetic_n else []) + (
        rng.sample(human_pool, human_n) if human_n else []
    )
    rng.shuffle(sample)
    return sample


def choose_torch_device_name(requested: str, torch_module: Any) -> str:
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "mps":
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import torch

    keypoint_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    width, height = args.image_width, args.image_height
    model_architecture = str(getattr(args, "model_architecture", "keypoint_heatmap_v1"))
    if model_architecture not in {"keypoint_heatmap_v1", "line_segmentation_intersection_v1"}:
        raise ValueError("model_architecture must be keypoint_heatmap_v1 or line_segmentation_intersection_v1")
    use_line_segmentation = model_architecture == "line_segmentation_intersection_v1"
    round3_input_resolution = (
        validate_round3_input_resolution(width, height, patch_size=getattr(args, "patch_size", None))
        if use_line_segmentation
        else None
    )
    line_names = [family.name for family in COURT_LINE_FAMILIES]
    line_width = int(getattr(args, "line_width", 3) or 3)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(choose_torch_device_name(args.device, torch))
    use_homography_refinement = bool(getattr(args, "enable_homography_refinement", False)) and not bool(
        getattr(args, "disable_homography_refinement", False)
    )
    use_static_camera_aggregation = bool(getattr(args, "static_camera_aggregate", False))

    real_labels = _load_real_court_keypoint_labels_from_roots(args.real_root)
    label_status_counts = _label_status_counts(real_labels)
    holdout_frame_stride = int(getattr(args, "holdout_frame_stride", 0) or 0)
    if holdout_frame_stride > 0:
        train_real = [
            row
            for row in real_labels
            if int(row.get("frame_index", -1)) % holdout_frame_stride != 0
        ]
        holdout_real = [
            row
            for row in real_labels
            if int(row.get("frame_index", -1)) % holdout_frame_stride == 0
        ]
        holdout_strategy = {"type": "frame_stride", "stride": holdout_frame_stride}
    else:
        holdout = set(args.holdout_clip)
        train_real = [row for row in real_labels if row.get("clip") not in holdout]
        holdout_real = [row for row in real_labels if row.get("clip") in holdout] or real_labels[-1:]
        holdout_strategy = {"type": "clip", "clips": sorted(holdout)}
    if real_labels and not train_real:
        train_real = [row for row in real_labels if row not in holdout_real] or real_labels
    if real_labels and not holdout_real:
        holdout_real = real_labels[-1:]

    # Eval-clip integrity gate (fail closed): rows that actually feed gradient
    # updates must never come from a protected eval clip, with no override.
    # Rows used only as a held-out validation-during-fitting signal may come
    # from Burlington/Wolverine (allow_internal_val=True), but Outdoor/Indoor
    # are refused even there -- see threed/racketsport/eval_guard.py.
    eval_guard_summary = {
        "train": assert_not_training_on_eval_clip(
            (row.get("clip") for row in train_real), allow_internal_val=False
        ),
        "holdout": assert_not_training_on_eval_clip(
            (row.get("clip") for row in holdout_real), allow_internal_val=True
        ),
    }

    def synthetic_batch(batch_size: int) -> tuple[Any, Any, Any]:
        images, targets, masks = [], [], []
        for _ in range(batch_size):
            quad = _random_quad(width, height, rng)
            image = Image.new("RGB", (width, height), tuple(rng.randint(35, 90) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            points = {
                point.name: _bilinear((point.world_xyz_m[0], point.world_xyz_m[1]), quad)
                for point in PICKLEBALL_KEYPOINTS
            }
            line_color = tuple(rng.randint(170, 255) for _ in range(3))
            for a, b in (
                ("near_left_corner", "near_right_corner"),
                ("near_right_corner", "far_right_corner"),
                ("far_right_corner", "far_left_corner"),
                ("far_left_corner", "near_left_corner"),
                ("near_nvz_left", "near_nvz_right"),
                ("far_nvz_left", "far_nvz_right"),
                ("net_left_sideline", "net_right_sideline"),
                ("near_baseline_center", "far_baseline_center"),
            ):
                draw.line([points[a], points[b]], fill=line_color, width=rng.randint(1, 3))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            if use_line_segmentation:
                target, mask = line_masks_for_points(points, width, height, line_width=line_width)
            else:
                target, mask = heatmaps_for_points(points, keypoint_names, width, height, sigma=args.sigma)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
            masks.append(torch.from_numpy(mask))
        return torch.stack(images), torch.stack(targets), torch.stack(masks)

    def real_batch(rows: list[dict[str, Any]]) -> tuple[Any, Any, Any] | None:
        if not rows:
            return None
        images, targets, masks = [], [], []
        for row in rows:
            image = load_label_image(row, cv2=cv2, image_module=Image)
            label_w, label_h = _label_coordinate_size(row, fallback_size=image.size)
            image = image.resize((width, height))
            scaled = {
                name: [xy[0] * width / label_w, xy[1] * height / label_h]
                for name, xy in row["keypoints"].items()
            }
            arr = np.asarray(image, dtype=np.float32) / 255.0
            if use_line_segmentation:
                target, mask = line_masks_for_points(scaled, width, height, line_width=line_width)
            else:
                target, mask = heatmaps_for_points(scaled, keypoint_names, width, height, sigma=args.sigma)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
            masks.append(torch.from_numpy(mask))
        return torch.stack(images), torch.stack(targets), torch.stack(masks)

    def predict_row_source_points(row: dict[str, Any]) -> dict[str, list[float]]:
        if use_line_segmentation:
            return predict_source_keypoints_from_line_model(
                model,
                row,
                cv2=cv2,
                np=np,
                torch=torch,
                image_module=Image,
                device=device,
                width=width,
                height=height,
                line_names=line_names,
            )["keypoints"]
        return predict_source_keypoints(
            model,
            row,
            cv2=cv2,
            np=np,
            torch=torch,
            image_module=Image,
            device=device,
            width=width,
            height=height,
            keypoint_names=keypoint_names,
            use_homography_refinement=use_homography_refinement,
        )

    def aggregate_static_camera_points(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
        # See `aggregate_static_camera_predictions`'s eval-integrity note: callers below only
        # ever pass held-out rows (never `train_real`), so this never aggregates over frames the
        # model was fit on.
        return aggregate_static_camera_predictions(
            model,
            rows,
            cv2=cv2,
            np=np,
            torch=torch,
            image_module=Image,
            device=device,
            width=width,
            height=height,
            keypoint_names=keypoint_names,
            use_homography_refinement=use_homography_refinement,
        )

    def evaluate(
        model: nn.Module,
        rows: list[dict[str, Any]],
        synthetic_batches: int = 4,
        *,
        aggregation_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        model.eval()
        real_model_input_errors: list[float] = []
        real_source_errors: list[float] = []
        real_source_errors_by_clip: dict[str, list[float]] = {}
        synthetic_errors: list[float] = []
        line_fit_rms_values: list[float] = []
        homography_self_consistency_medians: list[float] = []
        static_aggregate = aggregate_static_camera_points(aggregation_rows) if aggregation_rows and not use_line_segmentation else None
        with torch.no_grad():
            for row in rows:
                image = load_label_image(row, cv2=cv2, image_module=Image)
                label_w, label_h = _label_coordinate_size(row, fallback_size=image.size)
                sx, sy = width / label_w, height / label_h
                predicted_source_points = (
                    static_aggregate.get(str(row.get("clip") or "unknown"), {}) if static_aggregate is not None else {}
                )
                if not predicted_source_points:
                    if use_line_segmentation:
                        prediction = predict_source_keypoints_from_line_model(
                            model,
                            row,
                            cv2=cv2,
                            np=np,
                            torch=torch,
                            image_module=Image,
                            device=device,
                            width=width,
                            height=height,
                            line_names=line_names,
                        )
                        predicted_source_points = prediction["keypoints"]
                        fit_summary = prediction.get("line_fit_rms_px")
                        if isinstance(fit_summary, dict) and isinstance(fit_summary.get("mean"), (int, float)):
                            line_fit_rms_values.append(float(fit_summary["mean"]))
                    else:
                        predicted_source_points = predict_row_source_points(row)
                if predicted_source_points:
                    homography_summary = _homography_self_consistency_px(predicted_source_points)
                    if isinstance(homography_summary, dict) and isinstance(homography_summary.get("median"), (int, float)):
                        homography_self_consistency_medians.append(float(homography_summary["median"]))
                for name, xy in row["keypoints"].items():
                    if name not in predicted_source_points:
                        continue
                    source_x, source_y = predicted_source_points[name]
                    real_model_input_errors.append(math.hypot(source_x * sx - xy[0] * sx, source_y * sy - xy[1] * sy))
                    source_error = math.hypot(source_x - xy[0], source_y - xy[1])
                    real_source_errors.append(source_error)
                    real_source_errors_by_clip.setdefault(str(row.get("clip") or "unknown"), []).append(source_error)
            for _ in range(0 if use_line_segmentation else synthetic_batches):
                x, target, _ = synthetic_batch(args.batch_size)
                pred = court_keypoint_probabilities(_keypoint_heatmap_logits(model(x.to(device)))).detach().cpu()
                for batch_i in range(pred.shape[0]):
                    for idx in range(pred.shape[1]):
                        pred_flat = int(pred[batch_i, idx].argmax())
                        target_flat = int(target[batch_i, idx].argmax())
                        py, px = divmod(pred_flat, width)
                        ty, tx = divmod(target_flat, width)
                        synthetic_errors.append(math.hypot(px - tx, py - ty))
        real_model_input_summary = _error_summary(real_model_input_errors)
        real_source_summary = _error_summary(real_source_errors)
        synthetic_summary = _error_summary(synthetic_errors)
        real_source_pck_at_5 = _pck_at_threshold(real_source_errors, 5.0)
        real_source_pck_per_clip = {
            clip: _pck_error_summary(errors, 5.0)
            for clip, errors in sorted(real_source_errors_by_clip.items())
        }
        return {
            "real_metric_coordinate_space": "source_video_pixels",
            "real_corner_mean_px": real_source_summary["mean"],
            "real_corner_median_px": real_source_summary["median"],
            "real_corner_p95_px": real_source_summary["p95"],
            "real_corner_max_px": real_source_summary["max"],
            "real_corner_count": real_source_summary["count"],
            "real_corner_pck_at_5px": real_source_pck_at_5,
            "real_corner_pck_per_clip": real_source_pck_per_clip,
            "real_corner_mean_source_px": real_source_summary["mean"],
            "real_corner_median_source_px": real_source_summary["median"],
            "real_corner_p95_source_px": real_source_summary["p95"],
            "real_corner_max_source_px": real_source_summary["max"],
            "real_corner_mean_model_input_px": real_model_input_summary["mean"],
            "real_corner_median_model_input_px": real_model_input_summary["median"],
            "real_corner_p95_model_input_px": real_model_input_summary["p95"],
            "real_corner_max_model_input_px": real_model_input_summary["max"],
            "real_keypoint_mean_px": real_source_summary["mean"],
            "real_keypoint_median_px": real_source_summary["median"],
            "real_keypoint_p95_px": real_source_summary["p95"],
            "real_keypoint_max_px": real_source_summary["max"],
            "real_keypoint_count": real_source_summary["count"],
            "real_keypoint_pck_at_5px": real_source_pck_at_5,
            "real_keypoint_pck_per_clip": real_source_pck_per_clip,
            "real_keypoint_mean_source_px": real_source_summary["mean"],
            "real_keypoint_median_source_px": real_source_summary["median"],
            "real_keypoint_p95_source_px": real_source_summary["p95"],
            "real_keypoint_max_source_px": real_source_summary["max"],
            "real_keypoint_mean_model_input_px": real_model_input_summary["mean"],
            "real_keypoint_median_model_input_px": real_model_input_summary["median"],
            "real_keypoint_p95_model_input_px": real_model_input_summary["p95"],
            "real_keypoint_max_model_input_px": real_model_input_summary["max"],
            "synthetic_mean_px": synthetic_summary["mean"],
            "synthetic_median_px": synthetic_summary["median"],
            "synthetic_p95_px": synthetic_summary["p95"],
            "synthetic_count": synthetic_summary["count"],
            "prediction_mode": "line_segmentation_intersection" if use_line_segmentation else "keypoint_heatmap_argmax",
            "line_fit_rms_px": _error_summary(line_fit_rms_values) if use_line_segmentation else None,
            "homography_self_consistency_px": {
                "row_median_residual_summary": _error_summary(homography_self_consistency_medians),
                "note": (
                    "Median over rows of each prediction's planar-homography residual summary; "
                    "same diagnostic family as the CAL-R1/R2 internal self-consistency comparison."
                ),
            },
        }

    output_count = len(line_names) if use_line_segmentation else len(keypoint_names)
    net_architecture = "local_conv_v1" if use_line_segmentation else "encoder_decoder_v1"
    model = make_court_keypoint_heatmap_model(output_count, architecture=net_architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # EVAL INTEGRITY: aggregation_rows must be the held-out rows themselves, never train_real.
    # See `aggregate_static_camera_predictions`'s docstring -- aggregating over training rows
    # would score a checkpoint against its own (partially memorized) training predictions and
    # report the result as a held-out gate number, which is exactly the bug this fixes.
    aggregation_rows = holdout_real if use_static_camera_aggregation and not use_line_segmentation else None
    # Real-data mini-batch size: without this, every real-finetune epoch processes the ENTIRE
    # train_real set as one full-batch gradient step, so per-epoch cost scales with corpus size
    # (fine for a handful of owner-clip frames, but a real bottleneck once train_real is a
    # ~750-image external-corpus tier -- full-batch on that scale turns a 300-epoch fine-tune
    # into hours on CPU). `None` (the CLI default) preserves the original full-batch behavior
    # for backward compatibility; passing --real-batch-size caps each epoch's real-data cost to
    # a random sample of that size (standard mini-batch SGD), independent of corpus size.
    real_batch_size = getattr(args, "real_batch_size", None)
    # CAL-R2 synthetic/real curriculum fractions -- both default to 0.0 (no curriculum, uniform
    # sampling over train_real) so runs that never pass these flags are unaffected. See
    # `sample_curriculum_real_batch`'s docstring for the fallback behavior.
    synthetic_curriculum_start_fraction = float(getattr(args, "synthetic_curriculum_start_fraction", 0.0) or 0.0)
    synthetic_curriculum_end_fraction = float(getattr(args, "synthetic_curriculum_end_fraction", 0.0) or 0.0)
    # CAL-R2 point+line geometric-consistency loss weights. `geometric_loss_weight` defaults to
    # 0.0 (fully backward compatible: skips the extra soft-argmax/DLT work entirely, and the
    # trained model is bit-for-bit the same round-1 point-only-supervised model). See
    # `threed.racketsport.court_keypoint_geometric_loss` for the loss itself.
    geometric_loss_weight = float(getattr(args, "geometric_loss_weight", 0.0) or 0.0)
    geometric_colinearity_weight = float(getattr(args, "geometric_colinearity_weight", 1.0))
    geometric_homography_weight = float(getattr(args, "geometric_homography_weight", 1.0))
    before = evaluate(model, holdout_real, aggregation_rows=aggregation_rows)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        x, y, mask = synthetic_batch(args.batch_size)
        if train_real and epoch >= args.real_finetune_start_epoch:
            sampled_train_real = train_real
            if real_batch_size is not None and len(train_real) > real_batch_size:
                sampled_train_real = sample_curriculum_real_batch(
                    train_real,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    real_batch_size=real_batch_size,
                    synthetic_curriculum_start_fraction=synthetic_curriculum_start_fraction,
                    synthetic_curriculum_end_fraction=synthetic_curriculum_end_fraction,
                    rng=rng,
                )
            real = real_batch(sampled_train_real)
            if real is not None:
                rx, ry, rm = real
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)
                mask = torch.cat([mask, rm], dim=0)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(x)
        if use_line_segmentation:
            import torch.nn.functional as F

            heatmap_loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            heatmap_loss = court_keypoint_heatmap_loss(logits, y, mask)
        loss = heatmap_loss
        geometric_components: dict[str, float] | None = None
        if geometric_loss_weight > 0.0 and use_line_segmentation:
            raise ValueError("--geometric-loss-weight is only valid for keypoint_heatmap_v1")
        if geometric_loss_weight > 0.0:
            geometric = court_geometric_consistency_loss(
                logits,
                keypoint_names=keypoint_names,
                image_width=float(width),
                image_height=float(height),
                colinearity_weight=geometric_colinearity_weight,
                homography_weight=geometric_homography_weight,
            )
            loss = heatmap_loss + geometric_loss_weight * geometric["loss"]
            geometric_components = {
                "geometric_loss": float(geometric["loss"].detach().cpu()),
                "geometric_colinearity": float(geometric["colinearity"].detach().cpu()),
                "geometric_homography": float(geometric["homography"].detach().cpu()),
                "geometric_spread_guard": float(geometric["spread_guard"].detach().cpu()),
            }
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            row = evaluate(model, holdout_real, aggregation_rows=aggregation_rows)
            row.update({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), "heatmap_loss": float(heatmap_loss.detach().cpu())})
            if geometric_components is not None:
                row.update(geometric_components)
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after = evaluate(model, holdout_real, synthetic_batches=8, aggregation_rows=aggregation_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "court_keypoint_heatmap.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": [width, height],
            "model_architecture": model_architecture,
            "network_architecture": net_architecture,
            "keypoint_names": keypoint_names,
            "line_names": line_names if use_line_segmentation else [],
            "heatmap_activation": "sigmoid" if use_line_segmentation else "spatial_softmax",
            "loss": "binary_cross_entropy_line_masks" if use_line_segmentation else "spatial_softmax_cross_entropy",
            "args": vars(args),
        },
        checkpoint,
    )
    gate_value = after.get("real_keypoint_pck_at_5px")
    independent_reviewed_frame_count = label_status_counts["labels_independent_human_frames"]
    copied_frame_count = label_status_counts["labels_static_camera_copy_frame_count"]
    synthetic_frame_count = label_status_counts["labels_synthetic_frame_count"]
    external_dataset_frame_count = label_status_counts["labels_external_dataset_frame_count"]
    human_verification_note = (
        f"Independent human-verified frames = {independent_reviewed_frame_count}; "
        f"{copied_frame_count} additional frame(s) are owner-approved reviewed_static_camera_copy "
        "duplicates of an independent review on the same static camera and are NOT independent "
        f"human labels; {synthetic_frame_count} additional frame(s) are synthetic "
        "domain-randomized renders (status=='synthetic') and are NEITHER independent human "
        "labels NOR owner-approved copies; "
        f"{external_dataset_frame_count} additional frame(s) are human-annotated third-party "
        "dataset rows (status=='reviewed_external_dataset') and are counted separately, never "
        "as independent owner-human verification."
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_pretraining_run",
        "status": "trained_not_phase_verified",
        "checkpoint": str(checkpoint),
        "gate": {
            "metric": (
                "heldout_line_intersection_pck_at_5px"
                if use_line_segmentation
                else "heldout_static_camera_aggregate_pck_at_5px"
                if use_static_camera_aggregation
                else "heldout_pck_at_5px"
            ),
            "value": gate_value,
            "threshold": 0.95,
            "pck_threshold_px": 5.0,
            "passed": bool(gate_value is not None and float(gate_value) >= 0.95),
            "not_cal3_verified": True,
            # Report independent human-verified frames separately from owner-approved
            # static-camera copies and from synthetic renders; never collapse these into a
            # single "reviewed" count (CAL-R2 provenance fix -- gates must never count
            # synthetic frames as human).
            "independent_reviewed_frame_count": independent_reviewed_frame_count,
            "copied_frame_count": copied_frame_count,
            "synthetic_frame_count": synthetic_frame_count,
            "external_dataset_frame_count": external_dataset_frame_count,
            "human_verification_note": human_verification_note,
        },
        "before": before,
        "after": after,
        "history": history,
        "holdout_artifacts": [],
        "holdout_strategy": holdout_strategy,
        "real_train_count": len(train_real),
        "real_holdout_count": len(holdout_real),
        "eval_guard": eval_guard_summary,
        "labels_independent_human_frames": independent_reviewed_frame_count,
        "labels_static_camera_copy_frame_count": copied_frame_count,
        "labels_synthetic_frame_count": synthetic_frame_count,
        "labels_external_dataset_frame_count": external_dataset_frame_count,
        "postprocess": {
            "prediction_mode": "line_segmentation_intersection" if use_line_segmentation else "keypoint_heatmap_argmax",
            "homography_refinement": use_homography_refinement,
            "homography_refinement_max_inlier_error_px": 30.0,
            "homography_refinement_min_inliers": 8,
            "static_camera_aggregation": use_static_camera_aggregation,
            "static_camera_aggregation_source": "holdout_rows_self_referential" if use_static_camera_aggregation else None,
            "static_camera_aggregation_row_count": len(holdout_real) if use_static_camera_aggregation else 0,
        },
        "architecture": {
            "name": model_architecture,
            "network_architecture": net_architecture,
            "line_names": line_names if use_line_segmentation else [],
            "net_keypoint_height_convention": "regulation_net_top",
        },
        "round3_input_resolution": round3_input_resolution,
        "geometric_loss": {
            "enabled": geometric_loss_weight > 0.0,
            "weight": geometric_loss_weight,
            "colinearity_weight": geometric_colinearity_weight,
            "homography_weight": geometric_homography_weight,
        },
        "curriculum": {
            "synthetic_curriculum_start_fraction": synthetic_curriculum_start_fraction,
            "synthetic_curriculum_end_fraction": synthetic_curriculum_end_fraction,
        },
        "note": (
            "Synthetic pretraining plus court-keypoint fine-tune; not a "
            "verified CAL-3 no-tap solver. " + human_verification_note
        ),
    }
    _write_training_summary(args.out, summary)
    if not bool(getattr(args, "skip_holdout_artifacts", False)):
        summary["holdout_artifacts"] = _write_holdout_prediction_artifacts(
            model,
            holdout_real,
            cv2=cv2,
            np=np,
            torch=torch,
            device=device,
            keypoint_names=keypoint_names,
            model_width=width,
            model_height=height,
            out_dir=args.out,
            use_homography_refinement=use_homography_refinement,
        )
        _write_training_summary(args.out, summary)
    return summary


def _write_training_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "court_keypoint_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def training_cli_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": summary["checkpoint"],
        "gate": summary["gate"],
        "before": summary["before"],
        "after": summary["after"],
        "holdout_artifacts": summary.get("holdout_artifacts", []),
        "labels_independent_human_frames": summary.get("labels_independent_human_frames"),
        "labels_static_camera_copy_frame_count": summary.get("labels_static_camera_copy_frame_count"),
        "labels_synthetic_frame_count": summary.get("labels_synthetic_frame_count"),
        "labels_external_dataset_frame_count": summary.get("labels_external_dataset_frame_count"),
    }


def load_label_image(row: dict[str, Any], *, cv2: Any, image_module: Any) -> Any:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path and Path(image_path).is_file():
        return image_module.open(image_path).convert("RGB")

    video_path = row.get("video_path")
    frame_index = row.get("frame_index")
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("real court label row is missing video_path")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("real court label row is missing non-negative frame_index")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open court label video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"could not read frame {frame_index} from court label video: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return image_module.fromarray(frame_rgb).convert("RGB")


def _label_coordinate_size(row: dict[str, Any], *, fallback_size: tuple[int, int]) -> tuple[float, float]:
    source_size = row.get("source_video_size")
    if (
        isinstance(source_size, list)
        and len(source_size) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 for value in source_size)
    ):
        return float(source_size[0]), float(source_size[1])
    return float(fallback_size[0]), float(fallback_size[1])


def _write_holdout_prediction_artifacts(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    device: Any,
    keypoint_names: list[str],
    model_width: int,
    model_height: int,
    out_dir: Path,
    use_homography_refinement: bool,
) -> list[dict[str, Any]]:
    prediction_dir = out_dir / "holdout_predictions"
    overlay_dir = out_dir / "holdout_overlays"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    model.eval()

    rows_by_clip_video: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        video_path = Path(str(row.get("video_path")))
        clip = str(row.get("clip") or video_path.parent.name)
        rows_by_clip_video.setdefault((clip, str(video_path)), []).append(row)

    for (clip, video_path_text), label_rows in sorted(rows_by_clip_video.items()):
        video_path = Path(video_path_text)
        prediction_path = prediction_dir / f"{clip}_court_keypoints.json"
        overlay_path = overlay_dir / f"{clip}_court_keypoints_overlay.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open holdout court video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_width <= 0 or source_height <= 0:
            capture.release()
            raise ValueError(f"could not determine holdout court video size: {video_path}")
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (source_width, source_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"could not open court keypoint overlay writer: {overlay_path}")

        frames: list[dict[str, Any]] = []
        label_errors: list[float] = []
        labels_by_frame: dict[int, list[dict[str, Any]]] = {}
        for label_row in label_rows:
            frame_index_value = label_row.get("frame_index")
            if isinstance(frame_index_value, int) and not isinstance(frame_index_value, bool):
                labels_by_frame.setdefault(frame_index_value, []).append(label_row)
        frame_index = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                keypoints = _predict_frame_keypoints(
                    frame_bgr,
                    model,
                    cv2=cv2,
                    np=np,
                    torch=torch,
                    device=device,
                    keypoint_names=keypoint_names,
                    source_width=source_width,
                    source_height=source_height,
                    model_width=model_width,
                    model_height=model_height,
                    use_homography_refinement=use_homography_refinement,
                )
                frames.append({"frame_index": frame_index, "keypoints": keypoints})
                for label_row in labels_by_frame.get(frame_index, []):
                    label_errors.extend(_keypoint_errors(keypoints, label_row["keypoints"]))
                _draw_court_keypoints(cv2, frame_bgr, keypoints)
                writer.write(frame_bgr)
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        prediction_payload = {
            "schema_version": 1,
            "artifact_type": "court_keypoint_holdout_predictions",
            "clip": clip,
            "video": str(video_path),
            "coordinate_space": "source_video_pixels",
            "model_input_size": [model_width, model_height],
            "source_size": [source_width, source_height],
            "frames": frames,
            "verified": False,
            "not_cal3_verified": True,
        }
        prediction_path.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        errors = _error_summary(label_errors)
        heldout_frame_indices = sorted(labels_by_frame)
        artifacts.append(
            {
                "clip": clip,
                "prediction_artifact": str(prediction_path),
                "overlay_artifact": str(overlay_path),
                "overlay_frame_count": len(frames),
                "heldout_label_frame_index": heldout_frame_indices[0] if len(heldout_frame_indices) == 1 else None,
                "heldout_label_frame_indices": heldout_frame_indices,
                "heldout_keypoint_count": errors["count"],
                "median_keypoint_reprojection_px": errors["median"],
                "p95_keypoint_reprojection_px": errors["p95"],
                "not_cal3_verified": True,
            }
        )
    return artifacts


def _predict_frame_keypoints(
    frame_bgr: Any,
    model: Any,
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    device: Any,
    keypoint_names: list[str],
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
    use_homography_refinement: bool,
) -> dict[str, dict[str, Any]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (model_width, model_height), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = court_keypoint_probabilities(_keypoint_heatmap_logits(model(tensor))).detach().cpu()[0]
    scale_x = source_width / float(model_width)
    scale_y = source_height / float(model_height)
    keypoints: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(keypoint_names):
        decoded = decode_subpixel_heatmap(pred[idx].tolist())
        keypoints[name] = {
            "xy": [decoded.x * scale_x, decoded.y * scale_y],
            "confidence": max(0.0, min(1.0, float(decoded.score))),
            "heatmap_score": float(decoded.score),
        }
    if use_homography_refinement:
        refined = refine_keypoint_xy_with_planar_homography({name: item["xy"] for name, item in keypoints.items()})
        for name, xy in refined.items():
            if name not in keypoints:
                continue
            keypoints[name]["raw_xy"] = keypoints[name]["xy"]
            keypoints[name]["xy"] = xy
            keypoints[name]["postprocess"] = "planar_homography_ransac_v1"
    return keypoints


def _keypoint_errors(predictions: dict[str, dict[str, Any]], labels: dict[str, list[float]]) -> list[float]:
    errors: list[float] = []
    for name, label_xy in labels.items():
        prediction = predictions.get(name)
        if prediction is None:
            continue
        pred_xy = prediction.get("xy")
        if not isinstance(pred_xy, list) or len(pred_xy) != 2:
            continue
        errors.append(math.hypot(float(pred_xy[0]) - float(label_xy[0]), float(pred_xy[1]) - float(label_xy[1])))
    return errors


def _draw_court_keypoints(cv2: Any, frame_bgr: Any, keypoints: dict[str, dict[str, Any]]) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    for start, end in (
        ("near_left_corner", "near_right_corner"),
        ("near_right_corner", "far_right_corner"),
        ("far_right_corner", "far_left_corner"),
        ("far_left_corner", "near_left_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
        ("net_left_sideline", "net_right_sideline"),
        ("near_baseline_center", "near_nvz_center"),
        ("far_nvz_center", "far_baseline_center"),
    ):
        p0 = _prediction_point(keypoints.get(start))
        p1 = _prediction_point(keypoints.get(end))
        if p0 is not None and p1 is not None:
            cv2.line(frame_bgr, p0, p1, (0, 255, 255), 1, line_type)
    for name, prediction in keypoints.items():
        point = _prediction_point(prediction)
        if point is None:
            continue
        color = (0, 255, 0) if name.endswith("corner") else (255, 200, 0)
        cv2.circle(frame_bgr, point, 3, (0, 0, 0), -1, line_type)
        cv2.circle(frame_bgr, point, 2, color, -1, line_type)


def _prediction_point(prediction: dict[str, Any] | None) -> tuple[int, int] | None:
    if prediction is None:
        return None
    xy = prediction.get("xy")
    if not isinstance(xy, list) or len(xy) != 2:
        return None
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight pickleball court-keypoint heatmap model.")
    parser.add_argument(
        "--real-root",
        type=Path,
        default=None,
        action="append",
        help=(
            "Root containing <dataset-or-clip>/labels/court_keypoints.json rows. May be passed "
            "more than once to merge several roots (e.g. multiple external-corpus tiers) into "
            "one training run without needing a merged directory on disk."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-clip", action="append", default=["wolverine_mixed_0200_mid_steep_corner"])
    parser.add_argument(
        "--holdout-frame-stride",
        type=int,
        default=0,
        help="When positive, hold out frames whose frame_index is divisible by this stride across every clip.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument(
        "--model-architecture",
        choices=("keypoint_heatmap_v1", "line_segmentation_intersection_v1"),
        default="keypoint_heatmap_v1",
        help=(
            "Court detector architecture. keypoint_heatmap_v1 is the legacy 15-channel "
            "point heatmap. line_segmentation_intersection_v1 is the CAL-R3 line-mask head "
            "followed by fitted line equations and intersections."
        ),
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="CAL-R3 line-mask target half-width in model pixels.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Reserved CAL-R3 patch inference size; when unset, line mode requires --image-width >= 640.",
    )
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--real-finetune-start-epoch", type=int, default=120)
    parser.add_argument(
        "--real-batch-size",
        type=int,
        default=None,
        help=(
            "Cap each real-finetune epoch to a random sample of this many real-corpus rows "
            "(mini-batch SGD), instead of the full train_real set every epoch. Default (unset) "
            "preserves the original full-batch behavior; set this when train_real is large "
            "(e.g. a multi-hundred-image external-corpus tier) to keep per-epoch cost bounded."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument(
        "--skip-holdout-artifacts",
        action="store_true",
        help="Write the checkpoint and gate metrics without generating full-video holdout prediction overlays.",
    )
    parser.add_argument(
        "--static-camera-aggregate",
        action="store_true",
        help=(
            "Evaluate held-out labels with per-clip median keypoints aggregated across the "
            "held-out rows themselves (never training rows) for static cameras -- see "
            "aggregate_static_camera_predictions()'s eval-integrity note."
        ),
    )
    parser.add_argument(
        "--enable-homography-refinement",
        action="store_true",
        help="Apply planar court-geometry refinement to raw heatmap peaks before scoring.",
    )
    parser.add_argument(
        "--disable-homography-refinement",
        action="store_true",
        help="Compatibility flag: keep planar court-geometry refinement disabled.",
    )
    parser.add_argument(
        "--geometric-loss-weight",
        type=float,
        default=0.0,
        help=(
            "CAL-R2 PnLCalib-style point+line geometric-consistency loss weight, added to the "
            "per-channel heatmap cross-entropy loss every training step (see "
            "threed.racketsport.court_keypoint_geometric_loss). Default 0.0 disables it entirely "
            "(bit-for-bit round-1 behavior, no soft-argmax/DLT overhead)."
        ),
    )
    parser.add_argument(
        "--geometric-colinearity-weight",
        type=float,
        default=1.0,
        help="Relative weight of the colinearity term within the combined geometric-consistency loss.",
    )
    parser.add_argument(
        "--geometric-homography-weight",
        type=float,
        default=1.0,
        help="Relative weight of the homography self-consistency term within the combined geometric-consistency loss.",
    )
    parser.add_argument(
        "--synthetic-curriculum-start-fraction",
        type=float,
        default=0.0,
        help=(
            "CAL-R2 synthetic/real curriculum: fraction of each real-finetune mini-batch drawn "
            "from synthetic-status rows at epoch 0 (requires --real-batch-size and a mix of "
            "synthetic + real rows under --real-root). Default 0.0 (no curriculum, uniform "
            "sampling over train_real, matching pre-curriculum behavior)."
        ),
    )
    parser.add_argument(
        "--synthetic-curriculum-end-fraction",
        type=float,
        default=0.0,
        help="CAL-R2 synthetic/real curriculum: synthetic fraction at the final epoch (see --synthetic-curriculum-start-fraction).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        summary = run_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(training_cli_summary(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
