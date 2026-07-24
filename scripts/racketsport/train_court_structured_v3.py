#!/usr/bin/env python3
"""Train the confidence-aware 30-point structured court evidence network.

The trainer keeps the public v2 checkpoint untouched.  It consumes the existing reviewed
``court_keypoints.json`` roots, derives auxiliary floor targets only from a regulation homography
anchored by at least four reviewed canonical floor points, and masks every unavailable label.
Owner-reviewed unsupported views supervise only the supported-view head.  External rows use the
same adapter but carry an explicit bounded sample weight (0.25 by default).

This command writes a candidate checkpoint and diagnostic loss history.  It never promotes the
checkpoint or changes ``best_stack.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import (
    ALL_PICKLEBALL_KEYPOINTS,
    COURT_UNET_V2_HEATMAP_STRIDE,
    COURT_UNET_V2_SEG_CLASS_NAMES,
    merge_line_family_and_surface_targets,
)
from threed.racketsport.court_structured_model import (
    COURT_STRUCTURED_V3_ARCHITECTURE,
    STRUCTURED_DISTANCE_CLASS_NAMES,
    STRUCTURED_FLOOR_KEYPOINT_COUNT,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    STRUCTURED_FLOOR_KEYPOINTS,
    initialize_structured_v3_from_v2,
    make_court_structured_v3_model,
)
from threed.racketsport.court_structured_training import (
    semantic_segment_distance_targets,
    structured_floor_training_loss,
    weighted_masked_mean,
)

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except Exception:  # pragma: no cover - torch absence is reported when training starts
    _TorchIterableDataset = object


UNSUPPORTED_VIEW_ARTIFACT_TYPE = "racketsport_court_unsupported_view_labels"
DEFAULT_V2_CHECKPOINT = Path("models/checkpoints/court_unet_v2/court_model_v2.pt")
NET_TOP_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
ALL_KEYPOINT_NAMES = tuple(point.name for point in ALL_PICKLEBALL_KEYPOINTS)
ALL_NAME_TO_INDEX = {name: index for index, name in enumerate(ALL_KEYPOINT_NAMES)}
STRUCTURED_NAME_TO_INDEX = {
    name: index for index, name in enumerate(STRUCTURED_FLOOR_KEYPOINT_NAMES)
}
CANONICAL_FLOOR_NAMES = tuple(
    point.name for point in ALL_PICKLEBALL_KEYPOINTS[:15] if point.name not in NET_TOP_NAMES
)
CANONICAL_FLOOR_SET = frozenset(CANONICAL_FLOOR_NAMES)
AUXILIARY_START_INDEX = len(CANONICAL_FLOOR_NAMES)
SYNTHETIC_UNSUPPORTED_SCENARIOS = frozenset({"portrait_phone"})


@dataclass(frozen=True)
class StructuredLossWeights:
    keypoint_evidence: float = 1.0
    structured_reprojection: float = 1.0
    dense_line_distance: float = 0.5
    segmentation: float = 0.5
    visibility: float = 0.1
    covariance_nll: float = 0.1
    supported_view: float = 0.1


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _model_image(image_bgr: Any, *, width: int, height: int) -> Any:
    import cv2
    import numpy as np

    if image_bgr.shape[:2] != (height, width):
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(image_rgb.transpose(2, 0, 1))


def _gaussian_targets(
    xy_heatmap: Any,
    mask: Any,
    *,
    height: int,
    width: int,
    sigma_px: float,
) -> Any:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    targets = np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT, height, width), dtype=np.float32)
    for index, ((x, y), valid) in enumerate(zip(xy_heatmap, mask, strict=True)):
        if float(valid) <= 0.0:
            continue
        targets[index] = np.exp(
            -((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * sigma_px**2)
        )
    return targets


def _distance_targets(xy_heatmap: Any, *, height: int, width: int, max_distance_px: float) -> Any:
    import torch

    with torch.no_grad():
        value = semantic_segment_distance_targets(
            torch.as_tensor(xy_heatmap, dtype=torch.float32).unsqueeze(0),
            height=height,
            width=width,
            max_distance_px=max_distance_px,
        )[0]
    return value.numpy()


def _model_xy_to_heatmap(
    xy_model: Any,
    *,
    model_width: int,
    model_height: int,
    heatmap_width: int,
    heatmap_height: int,
    coordinate_transform: str,
) -> Any:
    import numpy as np

    if coordinate_transform == "udp":
        scale = np.asarray(
            [
                (heatmap_width - 1) / max(model_width - 1, 1),
                (heatmap_height - 1) / max(model_height - 1, 1),
            ],
            dtype=np.float32,
        )
        return np.asarray(xy_model, dtype=np.float32) * scale
    if coordinate_transform == "legacy_stride":
        return np.asarray(xy_model, dtype=np.float32) / float(COURT_UNET_V2_HEATMAP_STRIDE)
    raise ValueError("coordinate_transform must be udp or legacy_stride")


def _blank_targets(*, width: int, height: int) -> dict[str, Any]:
    import numpy as np

    head_width = max(1, width // COURT_UNET_V2_HEATMAP_STRIDE)
    head_height = max(1, height // COURT_UNET_V2_HEATMAP_STRIDE)
    return {
        "target_xy_heatmap": np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT, 2), dtype=np.float32),
        "keypoint_mask": np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT,), dtype=np.float32),
        "heatmap_target": np.zeros(
            (STRUCTURED_FLOOR_KEYPOINT_COUNT, head_height, head_width), dtype=np.float32
        ),
        "visibility_target": np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT,), dtype=np.float32),
        "visibility_mask": np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT,), dtype=np.float32),
        "segmentation_target": np.zeros((head_height, head_width), dtype=np.int64),
        "segmentation_mask": np.zeros((), dtype=np.float32),
        "distance_target": np.zeros(
            (len(STRUCTURED_DISTANCE_CLASS_NAMES), head_height, head_width), dtype=np.float32
        ),
        "distance_mask": np.zeros((len(STRUCTURED_DISTANCE_CLASS_NAMES),), dtype=np.float32),
        "supported_view_target": np.zeros((), dtype=np.float32),
        "supported_view_mask": np.ones((), dtype=np.float32),
    }


def _fit_human_anchored_floor(
    labeled_xy_model: dict[str, tuple[float, float]],
    *,
    max_median_reprojection_px: float,
    max_p95_reprojection_px: float,
) -> tuple[Any | None, dict[str, Any]]:
    """Fit one regulation projection using reviewed canonical floor points only."""

    import cv2
    import numpy as np

    anchors = [name for name in CANONICAL_FLOOR_NAMES if name in labeled_xy_model]
    if len(anchors) < 4:
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "fewer_than_four_reviewed_floor_anchors",
        }
    world_by_name = {
        point.name: point.world_xyz_m[:2] for point in STRUCTURED_FLOOR_KEYPOINTS
    }
    world = np.asarray([world_by_name[name] for name in anchors], dtype=np.float64)
    image = np.asarray([labeled_xy_model[name] for name in anchors], dtype=np.float64)
    world_hull_area = float(cv2.contourArea(cv2.convexHull(world.astype(np.float32))))
    image_hull_area = float(cv2.contourArea(cv2.convexHull(image.astype(np.float32))))
    if world_hull_area < 1e-4 or image_hull_area < 1.0:
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "degenerate_reviewed_floor_anchors",
            "world_hull_area_m2": world_hull_area,
            "image_hull_area_px2": image_hull_area,
        }
    homography, _ = cv2.findHomography(world, image, method=0)
    if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "degenerate_reviewed_floor_anchors",
        }
    all_world = np.asarray(
        [[point.world_xyz_m[0], point.world_xyz_m[1], 1.0] for point in STRUCTURED_FLOOR_KEYPOINTS],
        dtype=np.float64,
    )
    projected_h = (homography @ all_world.T).T
    if np.any(np.abs(projected_h[:, 2]) < 1e-8):
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "projection_at_infinity",
        }
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    if not np.isfinite(projected).all():
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "non_finite_projection",
        }
    anchor_projection = np.asarray(
        [projected[STRUCTURED_NAME_TO_INDEX[name]] for name in anchors], dtype=np.float64
    )
    residuals = np.linalg.norm(anchor_projection - image, axis=1)
    p95 = float(np.percentile(residuals, 95))
    median = float(np.median(residuals))
    if median > max_median_reprojection_px or p95 > max_p95_reprojection_px:
        return None, {
            "available": False,
            "anchor_count": len(anchors),
            "reason": "reviewed_anchor_reprojection_above_limit",
            "median_reprojection_px": median,
            "p95_reprojection_px": p95,
            "max_reprojection_px": float(residuals.max()),
            "max_median_reprojection_px": float(max_median_reprojection_px),
            "max_p95_reprojection_px": float(max_p95_reprojection_px),
        }
    return projected.astype(np.float32), {
        "available": True,
        "anchor_count": len(anchors),
        "anchor_names": anchors,
        "median_reprojection_px": median,
        "p95_reprojection_px": p95,
        "max_reprojection_px": float(residuals.max()),
        "auxiliary_target_source": "regulation_homography_fit_to_reviewed_floor_anchors",
    }


def real_row_to_structured_arrays(
    row: dict[str, Any],
    *,
    model_width: int,
    model_height: int,
    sigma_px: float,
    sample_weight: float,
    source_kind: str,
    max_anchor_median_reprojection_px: float = 3.0,
    max_anchor_p95_reprojection_px: float = 5.0,
    max_line_distance_px: float = 16.0,
    coordinate_transform: str = "udp",
    photometric_aug: bool = False,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Adapt one reviewed canonical row to the masked structured-v3 target contract."""

    import numpy as np
    from PIL import Image
    import cv2

    from scripts.racketsport.train_court_keypoint_heatmap import load_label_image
    from scripts.racketsport.train_court_model_v2 import _augment_real_image_photometric

    image = load_label_image(row, cv2=cv2, image_module=Image)
    image_bgr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[:, :, ::-1])
    if photometric_aug:
        if rng is None:
            raise ValueError("photometric augmentation requires a seeded rng")
        image_bgr = _augment_real_image_photometric(image_bgr, rng=rng)
    source_size = row.get("source_video_size") or [image.width, image.height]
    if (
        not isinstance(source_size, (list, tuple))
        or len(source_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in source_size)
    ):
        raise ValueError("real row source_video_size must be two numbers")
    source_width, source_height = map(float, source_size)
    if not all(math.isfinite(value) and value > 0 for value in (source_width, source_height)):
        raise ValueError("real row source_video_size must be positive and finite")

    keypoints = row.get("keypoints")
    if not isinstance(keypoints, dict):
        raise ValueError("real row keypoints must be an object")
    unexpected = sorted(set(keypoints) - set(ALL_KEYPOINT_NAMES))
    if unexpected:
        raise ValueError(f"unexpected real keypoints: {unexpected}")
    labeled_xy_model: dict[str, tuple[float, float]] = {}
    for name in CANONICAL_FLOOR_NAMES:
        value = keypoints.get(name)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"real row keypoints.{name} must be a two-item coordinate")
        x, y = value
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (x, y)):
            raise ValueError(f"real row keypoints.{name} must contain numbers")
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError(f"real row keypoints.{name} must contain finite coordinates")
        labeled_xy_model[name] = (
            float(x) * model_width / source_width,
            float(y) * model_height / source_height,
        )
    if not labeled_xy_model:
        raise ValueError("real structured row requires at least one canonical floor label")

    targets = _blank_targets(width=model_width, height=model_height)
    xy_model = np.zeros((STRUCTURED_FLOOR_KEYPOINT_COUNT, 2), dtype=np.float32)
    for name, xy in labeled_xy_model.items():
        index = STRUCTURED_NAME_TO_INDEX[name]
        xy_model[index] = xy
        targets["keypoint_mask"][index] = 1.0
        targets["visibility_target"][index] = 1.0
        targets["visibility_mask"][index] = 1.0

    projected, anchor_report = _fit_human_anchored_floor(
        labeled_xy_model,
        max_median_reprojection_px=max_anchor_median_reprojection_px,
        max_p95_reprojection_px=max_anchor_p95_reprojection_px,
    )
    if projected is not None:
        for index in range(AUXILIARY_START_INDEX, STRUCTURED_FLOOR_KEYPOINT_COUNT):
            x, y = projected[index]
            if 0.0 <= float(x) < model_width and 0.0 <= float(y) < model_height:
                xy_model[index] = projected[index]
                targets["keypoint_mask"][index] = 1.0
        geometry_xy_heatmap = _model_xy_to_heatmap(
            projected,
            model_width=model_width,
            model_height=model_height,
            heatmap_width=targets["distance_target"].shape[2],
            heatmap_height=targets["distance_target"].shape[1],
            coordinate_transform=coordinate_transform,
        )
        targets["distance_target"] = _distance_targets(
            geometry_xy_heatmap,
            height=targets["distance_target"].shape[1],
            width=targets["distance_target"].shape[2],
            max_distance_px=max_line_distance_px,
        )
        targets["distance_mask"][:] = 1.0

    targets["target_xy_heatmap"] = _model_xy_to_heatmap(
        xy_model,
        model_width=model_width,
        model_height=model_height,
        heatmap_width=targets["heatmap_target"].shape[2],
        heatmap_height=targets["heatmap_target"].shape[1],
        coordinate_transform=coordinate_transform,
    )
    targets["heatmap_target"] = _gaussian_targets(
        targets["target_xy_heatmap"],
        targets["keypoint_mask"],
        height=targets["heatmap_target"].shape[1],
        width=targets["heatmap_target"].shape[2],
        sigma_px=sigma_px,
    )
    targets["supported_view_target"] = np.ones((), dtype=np.float32)
    return {
        "image": _model_image(image_bgr, width=model_width, height=model_height),
        **targets,
        "sample_weight": np.asarray(float(sample_weight), dtype=np.float32),
        "source_kind": source_kind,
        "anchor_report": anchor_report,
    }


def synthetic_sample_to_structured_arrays(
    sample: dict[str, Any],
    *,
    model_width: int,
    model_height: int,
    sigma_px: float,
    max_line_distance_px: float = 16.0,
    coordinate_transform: str = "udp",
) -> dict[str, Any]:
    """Adapt an auxiliary CAL-SYNTH sample, structurally removing all top-net channels."""

    import cv2
    import numpy as np

    raw_xy = np.asarray(sample["keypoints_xy"], dtype=np.float32)
    raw_vis = np.asarray(sample["keypoints_vis"], dtype=np.int64)
    if raw_xy.shape != (len(ALL_KEYPOINT_NAMES), 2) or raw_vis.shape != (len(ALL_KEYPOINT_NAMES),):
        raise ValueError("structured synthetic samples must contain canonical-first 33-point targets")
    source_height, source_width = sample["image_bgr"].shape[:2]
    selected = [ALL_NAME_TO_INDEX[name] for name in STRUCTURED_FLOOR_KEYPOINT_NAMES]
    xy_model = raw_xy[selected] * np.asarray(
        [model_width / float(source_width), model_height / float(source_height)], dtype=np.float32
    )
    vis = raw_vis[selected]
    support = np.ones((STRUCTURED_FLOOR_KEYPOINT_COUNT,), dtype=np.float32)
    raw_heatmap_mask = sample.get("keypoint_heatmap_mask")
    if raw_heatmap_mask is not None:
        raw_heatmap_mask = np.asarray(raw_heatmap_mask)[selected]
        support = (raw_heatmap_mask.reshape(STRUCTURED_FLOOR_KEYPOINT_COUNT, -1).max(axis=1) > 0).astype(
            np.float32
        )
    else:
        support = (vis != 0).astype(np.float32)

    targets = _blank_targets(width=model_width, height=model_height)
    scenario = str((sample.get("meta") or {}).get("scenario") or "")
    supported = scenario not in SYNTHETIC_UNSUPPORTED_SCENARIOS
    if supported:
        targets["keypoint_mask"] = support
        targets["visibility_target"] = (vis == 2).astype(np.float32)
        targets["visibility_mask"][:] = 1.0
        targets["target_xy_heatmap"] = _model_xy_to_heatmap(
            xy_model,
            model_width=model_width,
            model_height=model_height,
            heatmap_width=targets["heatmap_target"].shape[2],
            heatmap_height=targets["heatmap_target"].shape[1],
            coordinate_transform=coordinate_transform,
        )
        targets["heatmap_target"] = _gaussian_targets(
            targets["target_xy_heatmap"],
            targets["keypoint_mask"],
            height=targets["heatmap_target"].shape[1],
            width=targets["heatmap_target"].shape[2],
            sigma_px=sigma_px,
        )
        targets["distance_target"] = _distance_targets(
            targets["target_xy_heatmap"],
            height=targets["distance_target"].shape[1],
            width=targets["distance_target"].shape[2],
            max_distance_px=max_line_distance_px,
        )
        targets["distance_mask"][:] = 1.0
        merged = merge_line_family_and_surface_targets(
            sample["line_family_mask"], sample["surface_mask"]
        )
        targets["segmentation_target"] = cv2.resize(
            merged,
            (targets["segmentation_target"].shape[1], targets["segmentation_target"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)
        targets["segmentation_mask"] = np.ones((), dtype=np.float32)
        targets["supported_view_target"] = np.ones((), dtype=np.float32)
    return {
        "image": _model_image(sample["image_bgr"], width=model_width, height=model_height),
        **targets,
        "sample_weight": np.ones((), dtype=np.float32),
        "source_kind": "synthetic_supported" if supported else "synthetic_unsupported",
        "anchor_report": {"available": True, "source": "synthetic_scene_truth"},
    }


def load_unsupported_view_rows(manifests: Sequence[Path] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests or []:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if payload.get("artifact_type") != UNSUPPORTED_VIEW_ARTIFACT_TYPE or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported view manifest has the wrong schema: {manifest_path}")
        if payload.get("authority") != "owner_reviewed":
            raise ValueError(f"unsupported view manifest must be owner_reviewed: {manifest_path}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"unsupported view manifest items must be a list: {manifest_path}")
        for index, item in enumerate(items):
            if not isinstance(item, dict) or item.get("supported_view") is not False:
                raise ValueError(f"unsupported view item {index} must explicitly set supported_view=false")
            image = item.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(f"unsupported view item {index} requires image")
            image_path = manifest_path.parent / image
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            rows.append({"image_path": image_path, "item": item, "source_kind": "owner_unsupported"})
    return rows


def unsupported_row_to_structured_arrays(
    row: dict[str, Any], *, model_width: int, model_height: int
) -> dict[str, Any]:
    import cv2
    import numpy as np

    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode unsupported-view image: {row['image_path']}")
    targets = _blank_targets(width=model_width, height=model_height)
    return {
        "image": _model_image(image, width=model_width, height=model_height),
        **targets,
        "sample_weight": np.ones((), dtype=np.float32),
        "source_kind": "owner_unsupported",
        "anchor_report": {"available": False, "reason": "unsupported_view"},
    }


def _tensor_batch(rows: Sequence[dict[str, Any]], torch: Any) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise ValueError("cannot stack an empty structured batch")
    keys = (
        "image",
        "target_xy_heatmap",
        "keypoint_mask",
        "heatmap_target",
        "visibility_target",
        "visibility_mask",
        "segmentation_target",
        "segmentation_mask",
        "distance_target",
        "distance_mask",
        "supported_view_target",
        "supported_view_mask",
        "sample_weight",
    )
    return {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in keys}


def structured_v3_losses(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    weights: StructuredLossWeights,
) -> dict[str, Any]:
    """Compute every v3 head loss with point/row/source masks applied before reduction."""

    import torch
    import torch.nn.functional as F

    sample_weights = batch["sample_weight"].to(outputs["keypoint_heatmaps"].device).float()
    target_heatmaps = batch["heatmap_target"].to(outputs["keypoint_heatmaps"].device).float()
    keypoint_mask = batch["keypoint_mask"].to(outputs["keypoint_heatmaps"].device).float()
    logits = outputs["keypoint_heatmaps"]
    if logits.shape != target_heatmaps.shape:
        raise ValueError("keypoint heatmap output/target shapes differ")
    target_mass = target_heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    target_distribution = target_heatmaps / target_mass
    log_probability = F.log_softmax(logits.flatten(2), dim=-1).reshape_as(logits)
    per_point_heatmap = -(target_distribution * log_probability).sum(dim=(-2, -1))
    keypoint_evidence = weighted_masked_mean(per_point_heatmap, keypoint_mask, sample_weights)

    structured = structured_floor_training_loss(
        outputs,
        target_xy_heatmap=batch["target_xy_heatmap"].to(logits.device).float(),
        target_mask=keypoint_mask,
        direct_weight=0.0,
        structured_weight=1.0,
        confidence_weight=1.0,
        sample_weights=sample_weights,
    )
    visibility_values = F.binary_cross_entropy_with_logits(
        outputs["keypoint_vis_logits"],
        batch["visibility_target"].to(logits.device).float(),
        reduction="none",
    )
    visibility = weighted_masked_mean(
        visibility_values,
        batch["visibility_mask"].to(logits.device).float(),
        sample_weights,
    )

    distance_values = F.smooth_l1_loss(
        outputs["line_distance_maps"],
        batch["distance_target"].to(logits.device).float(),
        reduction="none",
    ).mean(dim=(-2, -1))
    dense_line_distance = weighted_masked_mean(
        distance_values,
        batch["distance_mask"].to(logits.device).float(),
        sample_weights,
    )

    segmentation_values = F.cross_entropy(
        outputs["line_family_logits"],
        batch["segmentation_target"].to(logits.device).long(),
        reduction="none",
    ).mean(dim=(-2, -1))
    segmentation = weighted_masked_mean(
        segmentation_values,
        batch["segmentation_mask"].to(logits.device).float(),
        sample_weights,
    )
    supported_values = F.binary_cross_entropy_with_logits(
        outputs["supported_view_logit"],
        batch["supported_view_target"].to(logits.device).float(),
        reduction="none",
    )
    supported_view = weighted_masked_mean(
        supported_values,
        batch["supported_view_mask"].to(logits.device).float(),
        sample_weights,
    )
    total = (
        weights.keypoint_evidence * keypoint_evidence
        + weights.structured_reprojection * structured["structured_reprojection_loss"]
        + weights.dense_line_distance * dense_line_distance
        + weights.segmentation * segmentation
        + weights.visibility * visibility
        + weights.covariance_nll * structured["confidence_nll"]
        + weights.supported_view * supported_view
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("structured-v3 loss became non-finite")
    return {
        "loss": total,
        "keypoint_evidence": keypoint_evidence,
        "structured_reprojection": structured["structured_reprojection_loss"],
        "dense_line_distance": dense_line_distance,
        "segmentation": segmentation,
        "visibility": visibility,
        "covariance_nll": structured["confidence_nll"],
        "supported_view": supported_view,
        "structured_row_count": structured["structured_row_mask"].sum(),
    }


class StructuredSyntheticIterableDataset(_TorchIterableDataset):
    """Pickle-safe infinite CAL-SYNTH stream with deterministic per-worker seeds."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        seed: int,
        width: int,
        height: int,
        sigma_px: float,
        coordinate_transform: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.width = width
        self.height = height
        self.sigma_px = sigma_px
        self.coordinate_transform = coordinate_transform

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import torch
        from threed.racketsport.court_synth_stream import iter_synthetic_court_samples

        worker = torch.utils.data.get_worker_info()
        seed = self.seed if worker is None else self.seed + worker.id * 1_000_003
        for sample in iter_synthetic_court_samples(self.config, seed=seed):
            row = synthetic_sample_to_structured_arrays(
                sample,
                model_width=self.width,
                model_height=self.height,
                sigma_px=self.sigma_px,
                coordinate_transform=self.coordinate_transform,
            )
            yield {key: value for key, value in row.items() if key not in {"source_kind", "anchor_report"}}


def _synthetic_loader(args: argparse.Namespace, *, epoch: int, torch: Any) -> Any:
    config: dict[str, Any] = {
        "image_size": [args.image_width, args.image_height],
        "aux_keypoints": True,
        "heatmap_stride": COURT_UNET_V2_HEATMAP_STRIDE,
        "heatmap_sigma_px": args.heatmap_sigma_px,
    }
    if args.synthetic_scenario:
        config["scenarios"] = list(args.synthetic_scenario)
    dataset = StructuredSyntheticIterableDataset(
        config=config,
        seed=args.seed + epoch * 10_000_019,
        width=args.image_width,
        height=args.image_height,
        sigma_px=args.heatmap_sigma_px,
        coordinate_transform=args.coordinate_transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.synthetic_workers,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=False,
        **({"prefetch_factor": 2} if args.synthetic_workers > 0 else {}),
    )


def _load_real_sources(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    from scripts.racketsport.train_court_model_v2 import load_real_training_rows

    weighted: list[dict[str, Any]] = []
    seen: set[str] = set()
    protocol_report = None
    if args.protocol_manifest is not None:
        if args.real_root or args.external_real_root:
            raise ValueError(
                "--protocol-manifest owns the training rows; do not also pass real roots"
            )
        from scripts.racketsport.build_court_v31_protocol import load_protocol_partition_rows

        protocol_rows = load_protocol_partition_rows(
            args.protocol_manifest,
            fold_index=args.protocol_fold_index,
            partition="train",
        )
        sources = []
        for row in protocol_rows:
            row_key = str(row["protocol_row_key"])
            external = row_key.startswith("curated264/")
            sources.append(
                (
                    "external_reviewed" if external else "human_reviewed",
                    [row],
                    args.external_data_weight if external else 1.0,
                )
            )
        protocol_report = {
            "manifest": str(args.protocol_manifest),
            "manifest_sha256": hashlib.sha256(args.protocol_manifest.read_bytes()).hexdigest(),
            "fold_index": args.protocol_fold_index,
            "partition": "train",
            "row_count": len(protocol_rows),
        }
    else:
        sources = [
            ("human_reviewed", load_real_training_rows(list(args.real_root or [])), 1.0),
            (
                "external_reviewed",
                load_real_training_rows(list(args.external_real_root or [])),
                args.external_data_weight,
            ),
        ]
    for source_kind, rows, weight in sources:
        for row in rows:
            identity = str(row.get("image_path") or f"{row.get('clip')}:{row.get('frame_index')}")
            if identity in seen:
                raise ValueError(f"real training row appears in more than one input class: {identity}")
            seen.add(identity)
            weighted.append({"row": row, "sample_weight": float(weight), "source_kind": source_kind})
    unsupported = load_unsupported_view_rows(args.unsupported_view_manifest)
    return weighted, unsupported, protocol_report


def _real_batch(
    positives: Sequence[dict[str, Any]],
    unsupported: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
    torch: Any,
) -> dict[str, Any] | None:
    selected: list[dict[str, Any]] = []
    if positives and args.real_batch_size > 0:
        chosen = rng.sample(list(positives), min(args.real_batch_size, len(positives)))
        selected.extend(
            real_row_to_structured_arrays(
                item["row"],
                model_width=args.image_width,
                model_height=args.image_height,
                sigma_px=args.heatmap_sigma_px,
                sample_weight=item["sample_weight"],
                source_kind=item["source_kind"],
                max_anchor_median_reprojection_px=args.max_anchor_median_reprojection_px,
                max_anchor_p95_reprojection_px=args.max_anchor_p95_reprojection_px,
                max_line_distance_px=args.max_line_distance_px,
                coordinate_transform=args.coordinate_transform,
                photometric_aug=args.real_photometric_aug,
                rng=rng,
            )
            for item in chosen
        )
    if unsupported and args.unsupported_view_batch_size > 0:
        chosen_unsupported = rng.sample(
            list(unsupported), min(args.unsupported_view_batch_size, len(unsupported))
        )
        selected.extend(
            unsupported_row_to_structured_arrays(
                item,
                model_width=args.image_width,
                model_height=args.image_height,
            )
            for item in chosen_unsupported
        )
    return None if not selected else _tensor_batch(selected, torch)


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _loss_numbers(losses: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if key != "loss" and hasattr(value, "detach")
    }


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    args: argparse.Namespace,
    initialization: dict[str, Any],
    weights: StructuredLossWeights,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "court_structured_v3_training_checkpoint",
            "model_architecture": COURT_STRUCTURED_V3_ARCHITECTURE,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "image_size": [args.image_width, args.image_height],
            "heatmap_stride": COURT_UNET_V2_HEATMAP_STRIDE,
            "heatmap_decoder": args.heatmap_decoder,
            "coordinate_transform": args.coordinate_transform,
            "max_line_distance_px": args.max_line_distance_px,
            "keypoint_names": list(STRUCTURED_FLOOR_KEYPOINT_NAMES),
            "distance_class_names": list(STRUCTURED_DISTANCE_CLASS_NAMES),
            "segmentation_class_names": list(COURT_UNET_V2_SEG_CLASS_NAMES),
            "initialization": initialization,
            "loss_weights": asdict(weights),
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        path,
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    for value, name in (
        (args.epochs, "--epochs"),
        (args.steps_per_epoch, "--steps-per-epoch"),
        (args.batch_size, "--batch-size"),
        (args.image_width, "--image-width"),
        (args.image_height, "--image-height"),
    ):
        _validate_positive_int(value, name)
    if args.synthetic_workers < 0 or args.real_batch_size < 0 or args.unsupported_view_batch_size < 0:
        raise ValueError("worker and real-batch counts must be non-negative")
    if not math.isfinite(args.external_data_weight) or not 0 <= args.external_data_weight <= 1:
        raise ValueError("--external-data-weight must be finite and in [0, 1]")
    if args.synthetic_weight < 0 or args.real_weight < 0 or args.synthetic_weight + args.real_weight <= 0:
        raise ValueError("synthetic/real weights must be non-negative with a positive sum")
    if not math.isfinite(args.heatmap_sigma_px) or args.heatmap_sigma_px <= 0:
        raise ValueError("--heatmap-sigma-px must be positive")
    if not math.isfinite(args.max_line_distance_px) or args.max_line_distance_px <= 0:
        raise ValueError("--max-line-distance-px must be positive")
    for value, name in (
        (args.max_anchor_median_reprojection_px, "--max-anchor-median-reprojection-px"),
        (args.max_anchor_p95_reprojection_px, "--max-anchor-p95-reprojection-px"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        not math.isfinite(args.lr)
        or not math.isfinite(args.weight_decay)
        or args.lr <= 0
        or args.weight_decay < 0
    ):
        raise ValueError("--lr must be positive and --weight-decay must be non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    loss_weight_values = {
        "--keypoint-loss-weight": args.keypoint_loss_weight,
        "--structured-loss-weight": args.structured_loss_weight,
        "--distance-loss-weight": args.distance_loss_weight,
        "--seg-loss-weight": args.seg_loss_weight,
        "--visibility-loss-weight": args.visibility_loss_weight,
        "--covariance-loss-weight": args.covariance_loss_weight,
        "--supported-view-loss-weight": args.supported_view_loss_weight,
    }
    invalid_loss_weights = [
        name for name, value in loss_weight_values.items() if not math.isfinite(value) or value < 0
    ]
    if invalid_loss_weights:
        raise ValueError(f"loss weights must be finite and non-negative: {invalid_loss_weights}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "mps" and args.structured_loss_weight > 0:
        raise ValueError(
            "MPS cannot safely backpropagate the SVD-based structured DLT in this PyTorch build; "
            "use CUDA for training or CPU for a local smoke test"
        )
    model = make_court_structured_v3_model(encoder_weights_path=args.encoder_weights_path).to(device)
    initialization: dict[str, Any] = {"mode": "random", "checkpoint": None}
    if args.init_v2_checkpoint is not None and args.resume is None:
        path = Path(args.init_v2_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise ValueError("v2 initialization checkpoint must contain a model state dict")
        report = initialize_structured_v3_from_v2(model, state)
        initialization = {"mode": "v2_model_warm_start", "checkpoint": str(path), "report": report}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    if args.resume is not None:
        payload = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        if payload.get("model_architecture") != COURT_STRUCTURED_V3_ARCHITECTURE:
            raise ValueError("resume checkpoint is not court_structured_v3")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload.get("scaler", {}))
        start_epoch = int(payload.get("epoch", 0))
        initialization = {"mode": "resume", "checkpoint": str(args.resume), "start_epoch": start_epoch}

    positive_rows, unsupported_rows, protocol_report = _load_real_sources(args)
    if args.real_weight > 0 and not positive_rows and not unsupported_rows:
        raise ValueError("--real-weight is positive but no real or unsupported-view rows were supplied")
    loss_weights = StructuredLossWeights(
        keypoint_evidence=args.keypoint_loss_weight,
        structured_reprojection=args.structured_loss_weight,
        dense_line_distance=args.distance_loss_weight,
        segmentation=args.seg_loss_weight,
        visibility=args.visibility_loss_weight,
        covariance_nll=args.covariance_loss_weight,
        supported_view=args.supported_view_loss_weight,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        synthetic_iterator = iter(_synthetic_loader(args, epoch=epoch, torch=torch)) if args.synthetic_weight > 0 else None
        step_losses: list[float] = []
        step_gradient_norms: list[float] = []
        component_sums: dict[str, float] = {}
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            total = None
            if synthetic_iterator is not None:
                synthetic_batch = _move_batch(next(synthetic_iterator), device)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    synthetic_losses = structured_v3_losses(
                        model(synthetic_batch["image"].float()), synthetic_batch, weights=loss_weights
                    )
                total = synthetic_losses["loss"] * args.synthetic_weight
                for key, value in _loss_numbers(synthetic_losses).items():
                    component_sums[f"synthetic_{key}"] = component_sums.get(f"synthetic_{key}", 0.0) + value
            real_batch = _real_batch(
                positive_rows, unsupported_rows, args=args, rng=rng, torch=torch
            ) if args.real_weight > 0 else None
            if real_batch is not None:
                real_batch = _move_batch(real_batch, device)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    real_losses = structured_v3_losses(
                        model(real_batch["image"].float()), real_batch, weights=loss_weights
                    )
                weighted_real = real_losses["loss"] * args.real_weight
                total = weighted_real if total is None else total + weighted_real
                for key, value in _loss_numbers(real_losses).items():
                    component_sums[f"real_{key}"] = component_sums.get(f"real_{key}", 0.0) + value
            if total is None:
                raise RuntimeError("training step has no active data source")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            step_losses.append(float(total.detach().cpu()))
            step_gradient_norms.append(float(gradient_norm.detach().cpu()))
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss_mean": sum(step_losses) / len(step_losses),
            "loss_first": step_losses[0],
            "loss_last": step_losses[-1],
            "gradient_norm_mean_preclip": sum(step_gradient_norms) / len(step_gradient_norms),
            "gradient_norm_max_preclip": max(step_gradient_norms),
            "gradient_clip_max_norm": args.max_grad_norm,
            "lr": scheduler.get_last_lr()[0],
            "wall_time_s": time.time() - started,
            "components": {key: value / args.steps_per_epoch for key, value in sorted(component_sums.items())},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
        if args.checkpoint_every_epoch:
            _save_checkpoint(
                args.out / f"court_structured_v3_epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                args=args,
                initialization=initialization,
                weights=loss_weights,
                torch=torch,
            )

    checkpoint = args.out / "court_structured_v3.pt"
    _save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=args.epochs,
        args=args,
        initialization=initialization,
        weights=loss_weights,
        torch=torch,
    )
    source_counts = {
        "human_reviewed": sum(item["source_kind"] == "human_reviewed" for item in positive_rows),
        "external_reviewed": sum(item["source_kind"] == "external_reviewed" for item in positive_rows),
        "owner_unsupported": len(unsupported_rows),
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "court_structured_v3_training_run",
        "status": "trained_not_phase_verified",
        "verified": False,
        "authority_state": "review_only",
        "measurement_valid": False,
        "checkpoint": str(checkpoint),
        "architecture": {
            "name": COURT_STRUCTURED_V3_ARCHITECTURE,
            "floor_keypoint_count": STRUCTURED_FLOOR_KEYPOINT_COUNT,
            "floor_keypoint_names": list(STRUCTURED_FLOOR_KEYPOINT_NAMES),
            "distance_class_names": list(STRUCTURED_DISTANCE_CLASS_NAMES),
            "segmentation_class_names": list(COURT_UNET_V2_SEG_CLASS_NAMES),
        },
        "heatmap_decoder": args.heatmap_decoder,
        "coordinate_transform": args.coordinate_transform,
        "initialization": initialization,
        "source_counts": source_counts,
        "external_data_weight": args.external_data_weight,
        "evaluation_protocol": protocol_report,
        "loss_weights": asdict(loss_weights),
        "missing_label_policy": "masked_no_fabrication",
        "auxiliary_target_policy": (
            "regulation_homography_fit_to_at_least_four_reviewed_floor_anchors_"
            "median_le_3px_p95_le_5px"
        ),
        "unsupported_view_policy": "supported_view_head_only",
        "history": history,
    }
    (args.out / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, action="append", default=[])
    parser.add_argument("--external-real-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        help="Frozen v3.1 source-grouped manifest; trains only the selected fold's train rows.",
    )
    parser.add_argument("--protocol-fold-index", type=int, default=0)
    parser.add_argument("--unsupported-view-manifest", type=Path, action="append", default=[])
    parser.add_argument("--external-data-weight", type=float, default=0.25)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-v2-checkpoint", type=Path, default=DEFAULT_V2_CHECKPOINT)
    initialization.add_argument("--resume", type=Path)
    parser.add_argument("--encoder-weights-path", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--real-batch-size", type=int, default=16)
    parser.add_argument("--unsupported-view-batch-size", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--heatmap-sigma-px", type=float, default=1.5)
    parser.add_argument("--heatmap-decoder", choices=("parabolic", "dark"), default="dark")
    parser.add_argument(
        "--coordinate-transform",
        choices=("legacy_stride", "udp"),
        default="udp",
    )
    parser.add_argument("--max-line-distance-px", type=float, default=16.0)
    parser.add_argument("--max-anchor-median-reprojection-px", type=float, default=3.0)
    parser.add_argument("--max-anchor-p95-reprojection-px", type=float, default=5.0)
    parser.add_argument("--synthetic-scenario", action="append")
    parser.add_argument("--synthetic-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--synthetic-weight", type=float, default=0.2)
    parser.add_argument("--real-weight", type=float, default=0.8)
    parser.add_argument("--real-photometric-aug", action="store_true")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--structured-loss-weight", type=float, default=1.0)
    parser.add_argument("--distance-loss-weight", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.5)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.1)
    parser.add_argument("--covariance-loss-weight", type=float, default=0.1)
    parser.add_argument("--supported-view-loss-weight", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"checkpoint": summary["checkpoint"], "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
