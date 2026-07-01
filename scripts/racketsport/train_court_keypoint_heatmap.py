from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINTS,
    court_keypoint_heatmap_loss,
    court_keypoint_probabilities,
    decode_subpixel_heatmap,
    keypoint_labels_from_court_corners,
    make_court_keypoint_heatmap_model,
)


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


def court_keypoint_label_rows(payload: dict[str, Any], *, clip_root: Path | None = None) -> list[dict[str, Any]]:
    _require_reviewed(payload)
    return [_court_keypoint_label_row_from_item(payload, item, clip_root=clip_root) for item in _items(payload)]


def court_keypoint_label_row(payload: dict[str, Any], *, clip_root: Path | None = None) -> dict[str, Any]:
    _require_reviewed(payload)
    return _court_keypoint_label_row_from_item(payload, _items(payload)[0], clip_root=clip_root)


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
    raw_keypoints = {
        name: list(_xy_field(keypoints[name], f"court_keypoints.{name}"))
        for name in expected_names
    }
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
        "label_source": "reviewed_15_keypoint_court_labels",
    }


def _require_reviewed(payload: dict[str, Any]) -> None:
    review = payload.get("review")
    if not isinstance(review, dict) or review.get("status") != "reviewed":
        raise ValueError("court_keypoints labels must have review.status == reviewed")


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


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import torch

    keypoint_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    width, height = args.image_width, args.image_height
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    real_labels = load_real_court_keypoint_labels(args.real_root) if args.real_root else []
    holdout = set(args.holdout_clip)
    train_real = [row for row in real_labels if row.get("clip") not in holdout]
    holdout_real = [row for row in real_labels if row.get("clip") in holdout] or real_labels[-1:]

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
            target, mask = heatmaps_for_points(scaled, keypoint_names, width, height, sigma=args.sigma)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
            masks.append(torch.from_numpy(mask))
        return torch.stack(images), torch.stack(targets), torch.stack(masks)

    def evaluate(model: nn.Module, rows: list[dict[str, Any]], synthetic_batches: int = 4) -> dict[str, Any]:
        model.eval()
        real_model_input_errors: list[float] = []
        real_source_errors: list[float] = []
        real_source_errors_by_clip: dict[str, list[float]] = {}
        synthetic_errors: list[float] = []
        with torch.no_grad():
            for row in rows:
                batch = real_batch([row])
                if batch is None:
                    continue
                x, _, _ = [part.to(device) for part in batch]
                pred = court_keypoint_probabilities(model(x)).detach().cpu()[0]
                image = load_label_image(row, cv2=cv2, image_module=Image)
                label_w, label_h = _label_coordinate_size(row, fallback_size=image.size)
                sx, sy = width / label_w, height / label_h
                for name, xy in row["keypoints"].items():
                    idx = keypoint_names.index(name)
                    flat = int(pred[idx].argmax())
                    py, px = divmod(flat, width)
                    real_model_input_errors.append(math.hypot(px - xy[0] * sx, py - xy[1] * sy))
                    source_x = px / sx
                    source_y = py / sy
                    source_error = math.hypot(source_x - xy[0], source_y - xy[1])
                    real_source_errors.append(source_error)
                    real_source_errors_by_clip.setdefault(str(row.get("clip") or "unknown"), []).append(source_error)
            for _ in range(synthetic_batches):
                x, target, _ = synthetic_batch(args.batch_size)
                pred = court_keypoint_probabilities(model(x.to(device))).detach().cpu()
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
        }

    model = make_court_keypoint_heatmap_model(len(keypoint_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    before = evaluate(model, holdout_real)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        x, y, mask = synthetic_batch(args.batch_size)
        if train_real and epoch >= args.real_finetune_start_epoch:
            real = real_batch(train_real)
            if real is not None:
                rx, ry, rm = real
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)
                mask = torch.cat([mask, rm], dim=0)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = court_keypoint_heatmap_loss(model(x), y, mask)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            row = evaluate(model, holdout_real)
            row.update({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after = evaluate(model, holdout_real, synthetic_batches=8)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "court_keypoint_heatmap.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "keypoint_names": keypoint_names,
            "image_size": [width, height],
            "model_architecture": "encoder_decoder_v1",
            "heatmap_activation": "spatial_softmax",
            "loss": "spatial_softmax_cross_entropy",
            "args": vars(args),
        },
        checkpoint,
    )
    holdout_artifacts = _write_holdout_prediction_artifacts(
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
    )
    gate_value = after.get("real_keypoint_pck_at_5px")
    summary = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_pretraining_run",
        "status": "trained_not_phase_verified",
        "checkpoint": str(checkpoint),
        "gate": {
            "metric": "heldout_pck_at_5px",
            "value": gate_value,
            "threshold": 0.95,
            "pck_threshold_px": 5.0,
            "passed": bool(gate_value is not None and float(gate_value) >= 0.95),
            "not_cal3_verified": True,
        },
        "before": before,
        "after": after,
        "history": history,
        "holdout_artifacts": holdout_artifacts,
        "real_train_count": len(train_real),
        "real_holdout_count": len(holdout_real),
        "note": "Synthetic pretraining plus limited reviewed 15-keypoint court fine-tune; not a verified CAL-3 no-tap solver.",
    }
    (args.out / "court_keypoint_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def training_cli_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": summary["checkpoint"],
        "gate": summary["gate"],
        "before": summary["before"],
        "after": summary["after"],
        "holdout_artifacts": summary.get("holdout_artifacts", []),
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
) -> list[dict[str, Any]]:
    prediction_dir = out_dir / "holdout_predictions"
    overlay_dir = out_dir / "holdout_overlays"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    model.eval()

    for row in rows:
        video_path = Path(str(row.get("video_path")))
        clip = str(row.get("clip") or video_path.parent.name)
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
                )
                frames.append({"frame_index": frame_index, "keypoints": keypoints})
                if frame_index == row.get("frame_index"):
                    label_errors = _keypoint_errors(keypoints, row["keypoints"])
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
        artifacts.append(
            {
                "clip": clip,
                "prediction_artifact": str(prediction_path),
                "overlay_artifact": str(overlay_path),
                "overlay_frame_count": len(frames),
                "heldout_label_frame_index": row.get("frame_index"),
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
) -> dict[str, dict[str, Any]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (model_width, model_height), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = court_keypoint_probabilities(model(tensor)).detach().cpu()[0]
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
    parser.add_argument("--real-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-clip", action="append", default=["wolverine_mixed_0200_mid_steep_corner"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--real-finetune-start-epoch", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=20)
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
