#!/usr/bin/env python3
"""Adapt Roboflow COCO court keypoints to the trainer's canonical 15-key schema.

The canonical contract comes from ``threed.racketsport.court_keypoint_net``.
Every emitted item contains all 15 names in that exact order; unavailable or
unsafe channels are JSON ``null``.  In approved mode, emission is restricted to
the immutable owner-approved source-row manifest and all external net channels
are nulled because the reviewed Roboflow net markers are bottom-of-net while
the canonical channels are top-of-net.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.eval_guard import PROTECTED_EVAL_CLIP_IDS, assert_not_training_on_eval_clip


LANE_ID = "roboflow_court_adapter_20260723"
DEFAULT_DATASET_ROOT = ROOT / "data/roboflow_universe_20260706"
DEFAULT_LANE_DIR = ROOT / "runs/lanes" / LANE_ID
DEFAULT_OWNER_PACK_ROOT = ROOT / "cvat_upload/court_labelpack3_20260723/frames"
APPROVED_OWNER_ACT = ROOT / "runs/court_unified_training_20260723/external_training_owner_act_final.json"
APPROVED_OWNER_ACT_SHA256 = "e0f8935c5d42a531d144f74f0c527fc51b0cdd7c18e6c59ed5c5faca26893f29"
APPROVED_ROW_MANIFEST = ROOT / "runs/court_unified_training_20260723/final_external_corpus/manifest.json"
APPROVED_ROW_MANIFEST_SHA256 = "8693e56d39b776f725704ee0abcd5f32dfe55908fcbde061af583ab8cf3a977a"
OWNER_APPROVED_STATUS = "OWNER_APPROVED"
EXTERNAL_DATASET_STATUS = "reviewed_external_dataset"
FLOOR_ONLY_TRAINING_CONDITION = "FLOOR_ONLY_ALL_NET_CHANNELS_NULL"
COMPARE_ONLY_IDS = frozenset({"83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"})
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
NET_KEYPOINTS = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})

# This duplicate pin intentionally fails loudly if the trainer silently changes
# channel order. The names and order are authoritative at court_keypoint_net.py.
EXPECTED_CANONICAL_ORDER = (
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
    "far_right_corner",
    "far_baseline_center",
    "far_left_corner",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
)
LABELING_UI_ORDER = (
    "far_left_corner",
    "far_baseline_center",
    "far_right_corner",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
)

# Normalized regulation floor locations. 15 ft separates each baseline from
# its NVZ line, 14 ft separates the two NVZ lines, and the full court is 44 ft.
FLOOR_WORLD_XY = {
    "far_left_corner": (-1.0, 0.0),
    "far_baseline_center": (0.0, 0.0),
    "far_right_corner": (1.0, 0.0),
    "far_nvz_left": (-1.0, 15.0 / 44.0),
    "far_nvz_center": (0.0, 15.0 / 44.0),
    "far_nvz_right": (1.0, 15.0 / 44.0),
    "near_nvz_left": (-1.0, 29.0 / 44.0),
    "near_nvz_center": (0.0, 29.0 / 44.0),
    "near_nvz_right": (1.0, 29.0 / 44.0),
    "near_left_corner": (-1.0, 1.0),
    "near_baseline_center": (0.0, 1.0),
    "near_right_corner": (1.0, 1.0),
}
FLOOR_ROWS = (
    ("far_left_corner", "far_baseline_center", "far_right_corner"),
    ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
    ("near_left_corner", "near_baseline_center", "near_right_corner"),
)


def _mapping(
    *,
    source_group: str,
    schema_kind: str,
    confidence: str,
    rationale: str,
    source_to_canonical: dict[str, str],
    owner_approved_floor_only: bool,
) -> dict[str, Any]:
    return {
        "source_group": source_group,
        "schema_kind": schema_kind,
        "confidence": confidence,
        "rationale": rationale,
        "source_to_canonical": source_to_canonical,
        "owner_approved_floor_only": owner_approved_floor_only,
    }


DATASET_MAPPINGS: dict[str, dict[str, Any]] = {
    "chetan-rajagiri-9abfm__pickleball-court-v2__v1": _mapping(
        source_group="chetan_court_v2",
        schema_kind="semantic_direct_with_source_depth_reversal",
        confidence="high",
        rationale=(
            "All source names are semantic. Fresh overlays show source far_* on the image-near "
            "half and source near_* on the image-far half, so the mapping reverses source depth "
            "to match the trainer/UI camera-relative near/far convention. External net points "
            "are subsequently nulled by the binding floor-only owner act."
        ),
        source_to_canonical={
            "far_left_baseline": "near_left_corner",
            "far_center_baseline": "near_baseline_center",
            "far_right_baseline": "near_right_corner",
            "far_left_kitchen": "near_nvz_left",
            "far_center_kitchen": "near_nvz_center",
            "far_right_kitchen": "near_nvz_right",
            "net_left": "net_left_sideline",
            "net_center": "net_center",
            "net_right": "net_right_sideline",
            "near_left_kitchen": "far_nvz_left",
            "near_center_kitchen": "far_nvz_center",
            "near_right_kitchen": "far_nvz_right",
            "near_left_baseline": "far_left_corner",
            "near_center_baseline": "far_baseline_center",
            "near_right_baseline": "far_right_corner",
        },
        owner_approved_floor_only=True,
    ),
    "n-do-tran__pickleball-court-p3chl__v4": _mapping(
        source_group="p3chl_14kp_family",
        schema_kind="generic_index_geometric_inferred_static",
        confidence="high",
        rationale=(
            "The 14-point skeleton and reviewed overlays resolve 12 regulation floor "
            "intersections plus two bottom-of-net endpoints. Only the 12 floor points emit."
        ),
        source_to_canonical={
            "new-point-0": "far_left_corner",
            "new-point-1": "far_baseline_center",
            "new-point-2": "far_right_corner",
            "new-point-3": "far_nvz_right",
            "new-point-4": "far_nvz_center",
            "new-point-5": "far_nvz_left",
            "new-point-6": "near_nvz_left",
            "new-point-7": "near_nvz_center",
            "new-point-8": "near_nvz_right",
            "new-point-9": "near_right_corner",
            "new-point-10": "near_baseline_center",
            "new-point-11": "near_left_corner",
            "new-point-12": "net_left_sideline",
            "new-point-13": "net_right_sideline",
        },
        owner_approved_floor_only=True,
    ),
    "necromancer__pickleball-court-vbmkq__v2": _mapping(
        source_group="vbmkq_vhpgp_12kp_family",
        schema_kind="semantic_abbreviation_direct",
        confidence="high",
        rationale="Abbreviated sideline/baseline/NVZ names and skeleton resolve all 12 floor intersections.",
        source_to_canonical={
            "SB1": "far_left_corner",
            "BS1": "far_baseline_center",
            "SB2": "far_right_corner",
            "SNZ2": "far_nvz_right",
            "SNZ3": "near_nvz_right",
            "SB3": "near_right_corner",
            "BS2": "near_baseline_center",
            "SB4": "near_left_corner",
            "SNZ4": "near_nvz_left",
            "SNZ1": "far_nvz_left",
            "NS1": "far_nvz_center",
            "NS2": "near_nvz_center",
        },
        owner_approved_floor_only=True,
    ),
    "nigh-workspace__pickleball-court-vhpgp__v11": _mapping(
        source_group="vbmkq_vhpgp_12kp_family",
        schema_kind="semantic_abbreviation_direct",
        confidence="high",
        rationale="Semantic l/r, near/middle/far, baseline/kitchen abbreviations resolve all 12 floor intersections.",
        source_to_canonical={
            "l-n-b": "far_left_corner",
            "l-m-b": "far_baseline_center",
            "l-f-b": "far_right_corner",
            "l-f-k": "far_nvz_right",
            "r-f-k": "near_nvz_right",
            "r-f-b": "near_right_corner",
            "r-m-b": "near_baseline_center",
            "r-n-b": "near_left_corner",
            "r-n-k": "near_nvz_left",
            "l-n-k": "far_nvz_left",
            "l-m-k": "far_nvz_center",
            "r-m-k": "near_nvz_center",
        },
        owner_approved_floor_only=True,
    ),
    "pickleball-ball-detection__pickleball-court-keypoints-syncz__v6": _mapping(
        source_group="syncz_12kp",
        schema_kind="generic_index_geometric_inferred_static",
        confidence="high",
        rationale="Skeleton adjacency and reviewed overlays resolve the generic indices as the 12 floor intersections.",
        source_to_canonical={
            "new-point-0": "near_left_corner",
            "new-point-11": "near_baseline_center",
            "new-point-3": "near_right_corner",
            "new-point-4": "near_nvz_left",
            "new-point-8": "near_nvz_center",
            "new-point-5": "near_nvz_right",
            "new-point-7": "far_nvz_left",
            "new-point-9": "far_nvz_center",
            "new-point-6": "far_nvz_right",
            "new-point-1": "far_left_corner",
            "new-point-10": "far_baseline_center",
            "new-point-2": "far_right_corner",
        },
        owner_approved_floor_only=True,
    ),
    "ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3": _mapping(
        source_group="p3chl_14kp_family",
        schema_kind="generic_index_geometric_inferred_static",
        confidence="high",
        rationale="Same audited 14-point topology and source-image family as p3chl; only the 12 floor points emit.",
        source_to_canonical={
            "new-point-0": "far_left_corner",
            "new-point-1": "far_baseline_center",
            "new-point-2": "far_right_corner",
            "new-point-3": "far_nvz_right",
            "new-point-4": "far_nvz_center",
            "new-point-5": "far_nvz_left",
            "new-point-6": "near_nvz_left",
            "new-point-7": "near_nvz_center",
            "new-point-8": "near_nvz_right",
            "new-point-9": "near_right_corner",
            "new-point-10": "near_baseline_center",
            "new-point-11": "near_left_corner",
            "new-point-12": "net_left_sideline",
            "new-point-13": "net_right_sideline",
        },
        owner_approved_floor_only=True,
    ),
    "stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1": _mapping(
        source_group="stump_front_view_12kp",
        schema_kind="generic_letter_geometric_inferred_static",
        confidence="high",
        rationale="Reviewed front-view overlays resolve A-L as the 12 regulation floor intersections.",
        source_to_canonical={
            "A": "near_left_corner",
            "I": "near_baseline_center",
            "B": "near_right_corner",
            "E": "near_nvz_left",
            "J": "near_nvz_center",
            "F": "near_nvz_right",
            "G": "far_nvz_left",
            "L": "far_nvz_center",
            "H": "far_nvz_right",
            "C": "far_left_corner",
            "K": "far_baseline_center",
            "D": "far_right_corner",
        },
        owner_approved_floor_only=True,
    ),
    "testworkspace-i8nb1__pickle-court-keypoints__v2": _mapping(
        source_group="xuann_testworkspace_12kp_family",
        schema_kind="generic_index_geometric_inferred_static",
        confidence="high",
        rationale=(
            "Skeleton and overlays resolve the 12 floor intersections, but this workspace is "
            "outside the current immutable owner act and is not emitted in approved mode."
        ),
        source_to_canonical={
            "new-point-0": "far_left_corner",
            "new-point-1": "far_baseline_center",
            "new-point-2": "far_right_corner",
            "new-point-3": "far_nvz_right",
            "new-point-4": "far_nvz_center",
            "new-point-5": "far_nvz_left",
            "new-point-6": "near_nvz_left",
            "new-point-7": "near_nvz_center",
            "new-point-8": "near_nvz_right",
            "new-point-9": "near_right_corner",
            "new-point-10": "near_baseline_center",
            "new-point-11": "near_left_corner",
        },
        owner_approved_floor_only=False,
    ),
    "xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10": _mapping(
        source_group="xuann_testworkspace_12kp_family",
        schema_kind="generic_numeric_geometric_inferred_static",
        confidence="high",
        rationale=(
            "Skeleton and overlays resolve the 12 floor intersections, but this workspace is "
            "outside the current immutable owner act and is not emitted in approved mode."
        ),
        source_to_canonical={
            "1": "far_left_corner",
            "2": "far_baseline_center",
            "3": "far_right_corner",
            "4": "far_nvz_right",
            "5": "far_nvz_center",
            "6": "far_nvz_left",
            "7": "near_nvz_left",
            "8": "near_nvz_center",
            "9": "near_nvz_right",
            "10": "near_right_corner",
            "11": "near_baseline_center",
            "12": "near_left_corner",
        },
        owner_approved_floor_only=False,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def pin_canonical_contract() -> dict[str, Any]:
    from threed.racketsport import court_line_keypoints

    trainer_order = tuple(point.name for point in PICKLEBALL_KEYPOINTS)
    line_detector_order = tuple(point.name for point in court_line_keypoints.PICKLEBALL_KEYPOINTS)
    if trainer_order != EXPECTED_CANONICAL_ORDER:
        raise RuntimeError(
            "canonical trainer court-keypoint order drifted; update this adapter only after contract review: "
            f"{trainer_order!r}"
        )
    if line_detector_order != trainer_order:
        raise RuntimeError("court line detector and trainer court-keypoint orders disagree")

    ui_path = ROOT / "cvat_upload/court_labelpack2_20260723/START_HERE.html"
    ui_order: tuple[str, ...] | None = None
    if ui_path.is_file():
        match = re.search(r'"label_order":\s*(\[[^\]]+\])', ui_path.read_text(encoding="utf-8"))
        if match:
            ui_order = tuple(json.loads(match.group(1)))
            if ui_order != LABELING_UI_ORDER:
                raise RuntimeError(f"court labelpack UI order drifted: {ui_order!r}")
            if set(ui_order) != set(trainer_order):
                raise RuntimeError("court labelpack UI names do not match the trainer canonical names")

    return {
        "authority": "threed/racketsport/court_keypoint_net.py:PICKLEBALL_KEYPOINTS",
        "trainer_channel_order": list(trainer_order),
        "court_line_detector_order_matches": True,
        "labeling_ui_order": list(ui_order) if ui_order is not None else None,
        "labeling_ui_names_match": ui_order is not None and set(ui_order) == set(trainer_order),
        "coordinate_convention": (
            "image pixel [x,y]; canonical near/far is camera-relative as pinned by the "
            "trainer world points and owner labeling UI"
        ),
        "net_semantic": "top_of_net",
    }


def read_coco_splits(dataset_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = dataset_dir / split / "_annotations.coco.json"
        if path.is_file():
            result[split] = (path, json.loads(path.read_text(encoding="utf-8")))
    return result


def inventory_court_workspaces(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir() and "court" in path.name.lower()):
        schemas: dict[tuple[str, ...], set[str]] = defaultdict(set)
        counts_by_split: dict[str, dict[str, int]] = {}
        task_types: set[str] = set()
        for split, (_, payload) in read_coco_splits(dataset_dir).items():
            counts_by_split[split] = {
                "images": len(payload.get("images", [])),
                "annotations": len(payload.get("annotations", [])),
            }
            if any(isinstance(annotation.get("keypoints"), list) and annotation["keypoints"] for annotation in payload.get("annotations", [])):
                task_types.add("keypoints")
            if any(annotation.get("segmentation") for annotation in payload.get("annotations", [])):
                task_types.add("segmentation")
            if any(annotation.get("bbox") for annotation in payload.get("annotations", [])):
                task_types.add("bbox")
            for category in payload.get("categories", []):
                names = category.get("keypoints")
                if isinstance(names, list) and names:
                    schemas[tuple(str(name) for name in names)].add(split)

        schema_rows = [
            {"keypoint_count": len(names), "keypoint_names": list(names), "splits": sorted(splits)}
            for names, splits in sorted(schemas.items())
        ]
        mapping = DATASET_MAPPINGS.get(dataset_dir.name)
        rows.append(
            {
                "workspace": dataset_dir.name,
                "counts_by_split": counts_by_split,
                "image_count": sum(value["images"] for value in counts_by_split.values()),
                "annotation_count": sum(value["annotations"] for value in counts_by_split.values()),
                "task_types": sorted(task_types),
                "schemas": schema_rows,
                "mapping_configured": mapping is not None,
                "schema_kind": mapping.get("schema_kind") if mapping else "no_usable_keypoint_mapping",
                "mapping_confidence": mapping.get("confidence") if mapping else "none",
                "mapping_rationale": (
                    mapping.get("rationale")
                    if mapping
                    else "No COCO keypoint category is present; boxes/masks are not converted into guessed point labels."
                ),
                "owner_approved_floor_only": bool(mapping and mapping.get("owner_approved_floor_only")),
                "source_to_canonical": mapping.get("source_to_canonical", {}) if mapping else {},
            }
        )
    return rows


def _visible_keypoints(annotation: dict[str, Any], category: dict[str, Any]) -> dict[str, list[float]]:
    names = [str(name) for name in category.get("keypoints", [])]
    values = annotation.get("keypoints", [])
    result: dict[str, list[float]] = {}
    for index, name in enumerate(names):
        if index * 3 + 2 >= len(values):
            break
        x, y, visibility = values[index * 3 : index * 3 + 3]
        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and isinstance(visibility, (int, float))
            and float(visibility) > 0
            and math.isfinite(float(x))
            and math.isfinite(float(y))
        ):
            result[name] = [float(x), float(y)]
    return result


def _canonical_points(source_points: dict[str, list[float]], mapping: dict[str, Any]) -> dict[str, list[float] | None]:
    by_canonical: dict[str, list[float]] = {}
    for source_name, canonical_name in mapping["source_to_canonical"].items():
        if source_name in source_points:
            by_canonical[canonical_name] = source_points[source_name]
    # The current owner act is floor-only. This also makes the generic 14-point
    # schemas honest: their two net points are never promoted to top-of-net.
    return {
        name: None if name in NET_KEYPOINTS else by_canonical.get(name)
        for name in EXPECTED_CANONICAL_ORDER
    }


def geometric_sanity(
    keypoints: dict[str, list[float] | None],
    *,
    width: int,
    height: int,
    max_reprojection_error_ratio: float,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    missing_floor = [name for name in FLOOR_WORLD_XY if keypoints.get(name) is None]
    if missing_floor:
        return {"pass": False, "reason": "missing_floor_keypoints", "missing": missing_floor}
    if width <= 0 or height <= 0:
        return {"pass": False, "reason": "invalid_image_size"}

    points = {name: keypoints[name] for name in FLOOR_WORLD_XY}
    for name, value in points.items():
        assert value is not None
        x, y = float(value[0]), float(value[1])
        if x < -0.01 * width or x > 1.01 * width or y < -0.01 * height or y > 1.01 * height:
            return {"pass": False, "reason": "point_outside_image", "keypoint": name, "xy": [x, y]}

    corner_names = ("far_left_corner", "far_right_corner", "near_right_corner", "near_left_corner")
    source_quad = np.float32([FLOOR_WORLD_XY[name] for name in corner_names])
    image_quad = np.float32([points[name] for name in corner_names])
    if not bool(cv2.isContourConvex(image_quad.astype(np.int32))):
        return {"pass": False, "reason": "nonconvex_or_self_intersecting_outer_quad"}
    area_ratio = abs(float(cv2.contourArea(image_quad))) / float(width * height)
    if area_ratio < 0.02:
        return {"pass": False, "reason": "outer_quad_too_small", "outer_quad_area_ratio": area_ratio}

    homography = cv2.getPerspectiveTransform(source_quad, image_quad)
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        return {"pass": False, "reason": "degenerate_homography"}

    names = tuple(FLOOR_WORLD_XY)
    world = np.float32([[FLOOR_WORLD_XY[name] for name in names]])
    projected = cv2.perspectiveTransform(world, homography)[0]
    image_diag = math.hypot(width, height)
    errors = [
        float(np.linalg.norm(projected[index] - np.float32(points[name]))) / image_diag
        for index, name in enumerate(names)
    ]
    max_error = max(errors)
    if max_error > max_reprojection_error_ratio:
        return {
            "pass": False,
            "reason": "projective_reprojection_error",
            "max_reprojection_error_ratio": max_error,
            "threshold": max_reprojection_error_ratio,
        }

    inverse = np.linalg.inv(homography)
    observed = np.float32([[points[name] for name in names]])
    normalized = cv2.perspectiveTransform(observed, inverse)[0]
    normalized_by_name = {name: normalized[index] for index, name in enumerate(names)}
    row_y = [float(np.median([normalized_by_name[name][1] for name in row])) for row in FLOOR_ROWS]
    if not all(row_y[index] < row_y[index + 1] for index in range(len(row_y) - 1)):
        return {"pass": False, "reason": "projective_baseline_kitchen_order", "row_y": row_y}
    for row in FLOOR_ROWS:
        row_x = [float(normalized_by_name[name][0]) for name in row]
        if not row_x[0] < row_x[1] < row_x[2]:
            return {"pass": False, "reason": "projective_left_center_right_order", "row": list(row), "row_x": row_x}

    return {
        "pass": True,
        "reason": "pass",
        "outer_quad_area_ratio": area_ratio,
        "median_reprojection_error_ratio": float(sorted(errors)[len(errors) // 2]),
        "max_reprojection_error_ratio": max_error,
        "projective_row_y": row_y,
    }


def perceptual_hash(path: Path) -> int:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image for perceptual hash: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low > median).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bool(bit))
    return result


def phash_hex(value: int) -> str:
    return f"{value:016x}"


def phash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _strip_roboflow_suffix(file_name: str) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+(?=\.[^.]+$)", "", file_name, flags=re.IGNORECASE)


def source_media_family(file_name: str) -> str:
    stem = Path(_strip_roboflow_suffix(file_name)).stem
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    if re.match(r"^frame(?:s)?_", stem, flags=re.IGNORECASE):
        return "frame_sequence"
    if re.match(r"^youtube[-_]", stem, flags=re.IGNORECASE):
        return "youtube_sequence"
    court_match = re.match(r"^(left_court|middle_court|right_court)\d+", stem, flags=re.IGNORECASE)
    if court_match:
        return court_match.group(1).lower()
    video_match = re.match(r"^(.*?_(?:mp4|mov))[-_]\d+", stem, flags=re.IGNORECASE)
    if video_match:
        return video_match.group(1)
    return re.sub(r"[-_]\d+$", "", stem) or stem


def _protected_git_blob_oids() -> tuple[set[str], int]:
    prefixes = [f"eval_clips/ball/{clip_id}" for clip_id in PROTECTED_EVAL_CLIP_IDS]
    completed = subprocess.run(
        ["git", "ls-files", "-s", "-z", "--", *prefixes],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    oids: set[str] = set()
    image_count = 0
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        record = raw_record.decode("utf-8", errors="strict")
        metadata, path_text = record.split("\t", 1)
        if Path(path_text).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        fields = metadata.split()
        if len(fields) < 2:
            raise RuntimeError(f"unexpected git index record: {record!r}")
        oids.add(fields[1])
        image_count += 1
    return oids, image_count


def _owner_pack_hashes(root: Path) -> tuple[set[str], list[int], int]:
    if not root.is_dir():
        return set(), [], 0
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return {sha256_file(path) for path in paths}, [perceptual_hash(path) for path in paths], len(paths)


def _approved_row_scope() -> tuple[set[tuple[str, str]], dict[str, Any]]:
    if sha256_file(APPROVED_OWNER_ACT) != APPROVED_OWNER_ACT_SHA256:
        raise RuntimeError("approved external court owner-act bytes drifted")
    if sha256_file(APPROVED_ROW_MANIFEST) != APPROVED_ROW_MANIFEST_SHA256:
        raise RuntimeError("approved external court row-manifest bytes drifted")
    owner_act = json.loads(APPROVED_OWNER_ACT.read_text(encoding="utf-8"))
    manifest = json.loads(APPROVED_ROW_MANIFEST.read_text(encoding="utf-8"))
    approved_datasets = {
        key.removeprefix("roboflow::")
        for key, value in owner_act.get("final_decisions", {}).items()
        if key.startswith("roboflow::") and value == "APPROVE"
    }
    scope = {
        (str(row.get("scope_id")), str(row.get("image_sha256")))
        for row in manifest.get("rows", [])
        if row.get("kind") == "roboflow"
    }
    return scope, {
        "owner_act_path": str(APPROVED_OWNER_ACT.relative_to(ROOT)),
        "owner_act_sha256": APPROVED_OWNER_ACT_SHA256,
        "row_manifest_path": str(APPROVED_ROW_MANIFEST.relative_to(ROOT)),
        "row_manifest_sha256": APPROVED_ROW_MANIFEST_SHA256,
        "approved_datasets": sorted(approved_datasets),
        "approved_row_count": len(scope),
    }


def _license_by_workspace(dataset_root: Path) -> dict[str, str]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for entry in payload.get("entries", []):
        local_path = entry.get("local_path")
        if isinstance(local_path, str) and local_path:
            result[Path(local_path).name] = str(entry.get("license_as_recorded", ""))
    return result


def build_candidates(
    dataset_root: Path,
    inventory: list[dict[str, Any]],
    *,
    mode: str,
    max_reprojection_error_ratio: float,
    owner_pack_root: Path,
    phash_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Counter[str]], dict[str, Any]]:
    approved_scope: set[tuple[str, str]] = set()
    approval_evidence: dict[str, Any] | None = None
    if mode == "approved":
        approved_scope, approval_evidence = _approved_row_scope()

    protected_oids, protected_image_count = _protected_git_blob_oids()
    owner_exact, owner_phashes, owner_pack_count = _owner_pack_hashes(owner_pack_root)
    licenses = _license_by_workspace(dataset_root)
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: list[dict[str, Any]] = []
    lexical_values: list[Any] = []
    compare_only_hits: list[dict[str, str]] = []
    protected_lexical_hits: list[dict[str, str]] = []
    protected_blob_hits: list[str] = []
    owner_pack_hits: list[str] = []

    inventory_by_workspace = {row["workspace"]: row for row in inventory}
    for workspace, mapping in sorted(DATASET_MAPPINGS.items()):
        if workspace not in inventory_by_workspace:
            continue
        if mode == "approved" and not mapping["owner_approved_floor_only"]:
            counters[workspace]["outside_current_owner_act"] += inventory_by_workspace[workspace]["image_count"]
            continue
        dataset_dir = dataset_root / workspace
        for split, (annotation_path, payload) in read_coco_splits(dataset_dir).items():
            categories = {
                int(category["id"]): category for category in payload.get("categories", []) if "id" in category
            }
            annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for annotation in payload.get("annotations", []):
                if isinstance(annotation.get("keypoints"), list) and annotation["keypoints"]:
                    annotations_by_image[int(annotation.get("image_id", -1))].append(annotation)
            for image in sorted(payload.get("images", []), key=lambda row: int(row.get("id", -1))):
                image_id = int(image.get("id", -1))
                counters[workspace]["raw_images"] += 1
                image_annotations = annotations_by_image.get(image_id, [])
                if not image_annotations:
                    counters[workspace]["no_keypoint_annotation"] += 1
                    continue
                file_name = str(image.get("file_name", ""))
                source_path = annotation_path.parent / file_name
                lexical_values.extend((workspace, file_name, str(source_path)))
                lowered = " ".join((workspace, file_name, str(source_path))).lower()
                for compare_id in COMPARE_ONLY_IDS:
                    if compare_id in lowered:
                        compare_only_hits.append({"id": compare_id, "path": str(source_path)})
                for clip_id in PROTECTED_EVAL_CLIP_IDS:
                    if clip_id in lowered:
                        protected_lexical_hits.append({"id": clip_id, "path": str(source_path)})
                if not source_path.is_file():
                    counters[workspace]["missing_image"] += 1
                    continue

                best: tuple[int, int, dict[str, Any], dict[str, list[float]]] | None = None
                for annotation in image_annotations:
                    category = categories.get(int(annotation.get("category_id", -1)), {})
                    source_points = _visible_keypoints(annotation, category)
                    mapped_visible = sum(name in source_points for name in mapping["source_to_canonical"])
                    candidate_key = (mapped_visible, -int(annotation.get("id", 0)))
                    if best is None or candidate_key > best[:2]:
                        best = (candidate_key[0], candidate_key[1], annotation, source_points)
                assert best is not None
                _, _, annotation, source_points = best
                canonical = _canonical_points(source_points, mapping)
                if any(canonical[name] is None for name in FLOOR_WORLD_XY):
                    counters[workspace]["incomplete_floor_mapping"] += 1
                    continue

                width, height = int(image.get("width", 0)), int(image.get("height", 0))
                sanity = geometric_sanity(
                    canonical,
                    width=width,
                    height=height,
                    max_reprojection_error_ratio=max_reprojection_error_ratio,
                )
                if not sanity["pass"]:
                    counters[workspace][f"sanity_{sanity['reason']}"] += 1
                    continue
                counters[workspace]["sanity_pass"] += 1

                digest = sha256_file(source_path)
                if mode == "approved" and (workspace, digest) not in approved_scope:
                    counters[workspace]["outside_pinned_owner_row_scope"] += 1
                    continue
                blob_oid = git_blob_sha1(source_path)
                if blob_oid in protected_oids:
                    protected_blob_hits.append(str(source_path))
                    counters[workspace]["protected_image_blob"] += 1
                    continue
                p_hash = perceptual_hash(source_path)
                if digest in owner_exact or any(phash_distance(p_hash, owner_hash) <= phash_threshold for owner_hash in owner_phashes):
                    owner_pack_hits.append(str(source_path))
                    counters[workspace]["owner_pack_near_match_excluded"] += 1
                    continue

                source_group = str(mapping["source_group"])
                original_name = _strip_roboflow_suffix(file_name)
                media_family = source_media_family(file_name)
                candidates.append(
                    {
                        "workspace": workspace,
                        "split": split,
                        "image_id": image_id,
                        "annotation_id": int(annotation.get("id", -1)),
                        "source_path": source_path,
                        "source_file_name": file_name,
                        "source_original_name": original_name,
                        "source_group": source_group,
                        "source_media_family": media_family,
                        "apparent_venue_id": f"{source_group}:{media_family}",
                        "origin_key": f"{source_group}:{original_name}",
                        "width": width,
                        "height": height,
                        "sha256": digest,
                        "git_blob_sha1": blob_oid,
                        "phash": p_hash,
                        "keypoints": canonical,
                        "source_visible_keypoint_names": sorted(source_points),
                        "mapping_mode": mapping["schema_kind"],
                        "mapping_confidence": mapping["confidence"],
                        "mapping_used": dict(mapping["source_to_canonical"]),
                        "license": licenses.get(workspace, ""),
                        "sanity": sanity,
                    }
                )

    assert_not_training_on_eval_clip(lexical_values, allow_internal_val=False)
    if compare_only_hits or protected_lexical_hits or protected_blob_hits:
        raise RuntimeError(
            "protected/compare-only Roboflow collision assertion failed: "
            + json.dumps(
                {
                    "compare_only": compare_only_hits,
                    "protected_lexical": protected_lexical_hits,
                    "protected_blobs": protected_blob_hits,
                },
                sort_keys=True,
            )
        )
    return candidates, counters, {
        "approval_evidence": approval_evidence,
        "compare_only_ids_checked": sorted(COMPARE_ONLY_IDS),
        "compare_only_matches": 0,
        "protected_clip_ids_checked": list(PROTECTED_EVAL_CLIP_IDS),
        "protected_git_index_image_count": protected_image_count,
        "protected_git_blob_matches": 0,
        "protected_lexical_matches": 0,
        "owner_pack_root": str(owner_pack_root),
        "owner_pack_images_hashed": owner_pack_count,
        "owner_pack_near_match_excluded_count": len(owner_pack_hits),
        "owner_pack_near_match_excluded_examples": owner_pack_hits[:20],
        "selected_owner_pack_collision_count": 0,
    }


def perceptual_deduplicate(
    candidates: list[dict[str, Any]],
    counters: dict[str, Counter[str]],
    *,
    phash_threshold: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_origin: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda item: (item["workspace"], item["split"], item["image_id"], item["annotation_id"]),
    ):
        workspace = str(row["workspace"])
        if row["sha256"] in seen_exact:
            counters[workspace]["dedup_exact"] += 1
            continue
        if row["origin_key"] in seen_origin:
            counters[workspace]["dedup_same_roboflow_origin"] += 1
            continue
        if any(phash_distance(int(row["phash"]), int(prior["phash"])) <= phash_threshold for prior in kept):
            counters[workspace]["dedup_perceptual"] += 1
            continue
        kept.append(row)
        seen_exact.add(str(row["sha256"]))
        seen_origin.add(str(row["origin_key"]))
    return kept


def _diverse_cap(rows: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (row["workspace"], row["split"], row["image_id"]))
    if len(ordered) <= limit:
        return ordered, []
    selected = [ordered.pop(0)]
    while ordered and len(selected) < limit:
        best_index = max(
            range(len(ordered)),
            key=lambda index: (
                min(phash_distance(int(ordered[index]["phash"]), int(item["phash"])) for item in selected),
                -index,
            ),
        )
        selected.append(ordered.pop(best_index))
    return selected, ordered


def cap_apparent_venues(
    candidates: list[dict[str, Any]],
    counters: dict[str, Counter[str]],
    *,
    max_per_apparent_venue: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_venue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_venue[str(row["apparent_venue_id"])].append(row)
    selected: list[dict[str, Any]] = []
    venue_counts: dict[str, int] = {}
    for venue_id, rows in sorted(by_venue.items()):
        kept, dropped = _diverse_cap(rows, max_per_apparent_venue)
        selected.extend(kept)
        venue_counts[venue_id] = len(kept)
        for row in dropped:
            counters[str(row["workspace"])]["apparent_venue_cap"] += 1
    selected.sort(key=lambda row: (row["workspace"], row["split"], row["image_id"], row["annotation_id"]))
    return selected, venue_counts


def _clean_generated_outputs(lane_dir: Path) -> None:
    if lane_dir.resolve() != DEFAULT_LANE_DIR.resolve() and LANE_ID not in lane_dir.name:
        # Tests use arbitrary temporary lane names; production refuses broad paths.
        if not str(lane_dir.resolve()).startswith(("/private/tmp/", "/tmp/", "/private/var/folders/")):
            raise ValueError(f"refusing to write adapter outputs outside an adapter lane or temporary test dir: {lane_dir}")
    for directory_name in ("adapted_corpus", "validation_pngs"):
        path = lane_dir / directory_name
        if path.exists():
            shutil.rmtree(path)
    for file_name in (
        "adapted_labels.jsonl",
        "schema_map.json",
        "schema_map.md",
        "PROPOSED_LEDGER_ROW.json",
        "report.json",
    ):
        path = lane_dir / file_name
        if path.exists():
            path.unlink()


def emit_loader_corpus(
    selected: list[dict[str, Any]],
    *,
    lane_dir: Path,
    mode: str,
    adapter_sha256: str,
) -> tuple[Path, list[dict[str, Any]]]:
    corpus_root = lane_dir / "adapted_corpus"
    corpus_root.mkdir(parents=True, exist_ok=True)
    emitted_jsonl: list[dict[str, Any]] = []
    by_clip: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_clip[(str(row["workspace"]), str(row["split"]))].append(row)

    for (workspace, split), rows in sorted(by_clip.items()):
        clip = f"{workspace}__{split}"
        clip_root = corpus_root / clip
        frame_dir = clip_root / "frames"
        label_dir = clip_root / "labels"
        frame_dir.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            suffix = Path(str(row["source_file_name"])).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".jpg"
            frame_name = f"frame_{index:06d}{suffix}"
            destination = frame_dir / frame_name
            os.symlink(Path(row["source_path"]).resolve(), destination)
            provenance = {
                "dataset": workspace,
                "workspace": workspace,
                "split": split,
                "coco_image_id": row["image_id"],
                "coco_annotation_id": row["annotation_id"],
                "original_file_name": row["source_file_name"],
                "roboflow_original_name": row["source_original_name"],
                "original_image": display_path(Path(row["source_path"])),
                "source_group": row["source_group"],
                "source_media_family": row["source_media_family"],
                "apparent_venue_id": row["apparent_venue_id"],
                "license": row["license"],
                "image_sha256": row["sha256"],
                "perceptual_hash64": phash_hex(int(row["phash"])),
                "mapping_mode": row["mapping_mode"],
                "mapping_confidence": row["mapping_confidence"],
                "mapping_used": row["mapping_used"],
                "source_visible_keypoint_names": row["source_visible_keypoint_names"],
                "adapter_script": "scripts/racketsport/adapt_roboflow_court_keypoints.py",
                "adapter_sha256": adapter_sha256,
                "geometric_sanity": row["sanity"],
            }
            item: dict[str, Any] = {
                "frame": frame_name,
                "status": EXTERNAL_DATASET_STATUS,
                "keypoints": row["keypoints"],
                "provenance": provenance,
            }
            if mode == "approved":
                item["pseudo_label_status"] = OWNER_APPROVED_STATUS
            else:
                item["pseudo_label_status"] = "PENDING_SPOTCHECK"
            items.append(item)
            emitted_jsonl.append({"clip": clip, "frame_dir": f"{clip}/frames", **item})

        sizes = {(int(row["width"]), int(row["height"])) for row in rows}
        frames: dict[str, Any] = {"frame_dir": f"{clip}/frames", "path_base": "corpus_root"}
        if len(sizes) == 1:
            width, height = next(iter(sizes))
            frames["source_resolution"] = [width, height]
            frames["label_coordinate_space"] = [width, height]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "clip": clip,
            "status": OWNER_APPROVED_STATUS if mode == "approved" else "PENDING_SPOTCHECK",
            "annotation": {"items": items},
            "frames": frames,
            "review": {
                "status": "reviewed",
                "reviewer": "upstream_roboflow_human_annotations",
                "note": (
                    "Mapped external floor points only. All 15 canonical keys are present; "
                    "all three unsafe external bottom-of-net channels are JSON null."
                ),
            },
            "training_eligibility": {
                "queued": mode == "approved",
                "reason": (
                    "owner-approved floor-only source row; external net channels nulled"
                    if mode == "approved"
                    else "diagnostic adapter output; no positive training-eligibility claim"
                ),
            },
        }
        if mode == "approved":
            payload["training_eligibility"]["owner_adjudication"] = {
                "path": str(APPROVED_OWNER_ACT.relative_to(ROOT)),
                "sha256": APPROVED_OWNER_ACT_SHA256,
                "decision": "APPROVE",
                "scope_type": "roboflow_dataset",
                "scope_id": workspace,
                "training_condition": FLOOR_ONLY_TRAINING_CONDITION,
            }
        write_json(label_dir / "court_keypoints.json", payload)

    with (lane_dir / "adapted_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in emitted_jsonl:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return corpus_root, emitted_jsonl


def validate_loader_contract(corpus_root: Path, *, mode: str, expected_count: int) -> dict[str, Any]:
    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(
        corpus_root,
        allow_pending_diagnostic_only=mode != "approved",
    )
    if len(rows) != expected_count:
        raise RuntimeError(f"loader row count mismatch: {len(rows)} != {expected_count}")
    labeled_histogram = Counter(str(len(row["keypoints"])) for row in rows)
    if set(labeled_histogram) != {"12"}:
        raise RuntimeError(f"external floor-only loader rows must expose exactly 12 labels: {labeled_histogram}")
    raw_key_order_errors = 0
    nonnull_net_values = 0
    for path in sorted(corpus_root.glob("*/labels/court_keypoints.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["annotation"]["items"]:
            if tuple(item["keypoints"]) != EXPECTED_CANONICAL_ORDER:
                raw_key_order_errors += 1
            nonnull_net_values += sum(item["keypoints"][name] is not None for name in NET_KEYPOINTS)
    if raw_key_order_errors or nonnull_net_values:
        raise RuntimeError(
            f"emitted raw contract invalid: order_errors={raw_key_order_errors}, "
            f"nonnull_external_net_values={nonnull_net_values}"
        )
    return {
        "status": "PASS",
        "mode": mode,
        "rows_loaded": len(rows),
        "labeled_keypoint_histogram_after_null_removal": dict(sorted(labeled_histogram.items())),
        "raw_items_with_exact_15_key_order": len(rows) - raw_key_order_errors,
        "raw_key_order_errors": raw_key_order_errors,
        "nonnull_external_net_values": nonnull_net_values,
        "default_training_loader_passed": mode == "approved",
    }


def render_validation_samples(
    selected: list[dict[str, Any]],
    *,
    lane_dir: Path,
    limit: int,
) -> list[str]:
    from PIL import Image, ImageDraw

    by_workspace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_workspace[str(row["workspace"])].append(row)
    chosen: list[dict[str, Any]] = []
    workspace_names = sorted(by_workspace)
    offset = 0
    while len(chosen) < limit:
        added = False
        for workspace in workspace_names:
            rows = by_workspace[workspace]
            if offset < len(rows):
                chosen.append(rows[offset])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        offset += 1

    output_dir = lane_dir / "validation_pngs"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    colors = ("#ff375f", "#ffd60a", "#64d2ff", "#30d158")
    for sample_index, row in enumerate(chosen, start=1):
        with Image.open(row["source_path"]) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        scale = max(1, round(max(image.size) / 800))
        for row_index, line_names in enumerate(FLOOR_ROWS):
            line_points = [tuple(row["keypoints"][name]) for name in line_names]
            draw.line(line_points, fill=colors[row_index], width=2 * scale)
        for index, name in enumerate(EXPECTED_CANONICAL_ORDER, start=1):
            point = row["keypoints"][name]
            if point is None:
                continue
            x, y = float(point[0]), float(point[1])
            radius = 4 * scale
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="#ff375f",
                outline="white",
                width=scale,
            )
            draw.text(
                (x + radius + scale, y - radius),
                f"{index}:{name}",
                fill="#fff59d",
                stroke_width=scale,
                stroke_fill="black",
            )
        banner = (
            f"{row['workspace']} | {row['mapping_mode']} | 12/15 floor labels | "
            f"venue={row['apparent_venue_id']}"
        )
        draw.rectangle((0, 0, image.width, 24 * scale), fill="black")
        draw.text((5 * scale, 5 * scale), banner, fill="white")
        output_path = output_dir / f"{sample_index:02d}_{row['workspace']}.png"
        image.save(output_path)
        paths.append(display_path(output_path))
    return paths


def _schema_map_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Roboflow court-keypoint schema map",
        "",
        "Canonical output is the trainer's 15-key dictionary order. `null` channels are unsupervised. "
        "The current owner act permits only the 12 floor channels from approved external workspaces.",
        "",
        "| Workspace | Raw images | Source keypoints | Schema/mapping | Confidence | Approved-mode yield | Disposition |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        schemas = row.get("schemas", [])
        source = "<br>".join(
            f"{schema['keypoint_count']}: {', '.join(schema['keypoint_names'])}" for schema in schemas
        ) or "none (boxes/masks only)"
        disposition = str(row.get("disposition", "")).replace("|", "/")
        lines.append(
            f"| `{row['workspace']}` | {row['image_count']} | {source} | "
            f"{row['schema_kind']} | {row['mapping_confidence']} | "
            f"{row.get('yield_after_dedup_sanity_cap', 0)} | {disposition} |"
        )
    lines.append("")
    return "\n".join(lines)


def _workspace_report_rows(
    inventory: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    counters: dict[str, Counter[str]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    yield_by_workspace = Counter(str(row["workspace"]) for row in selected)
    result: list[dict[str, Any]] = []
    for row in inventory:
        workspace = str(row["workspace"])
        configured = workspace in DATASET_MAPPINGS
        approved = bool(configured and DATASET_MAPPINGS[workspace]["owner_approved_floor_only"])
        if not configured:
            disposition = "skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections"
        elif mode == "approved" and not approved:
            disposition = "schema mapped but skipped: outside current immutable owner act"
        elif yield_by_workspace[workspace] == 0:
            disposition = "mapped but zero yield after sanity/dedup/apparent-venue policy"
        else:
            disposition = "mapped and emitted"
        merged = dict(row)
        merged["yield_after_dedup_sanity_cap"] = yield_by_workspace[workspace]
        merged["drop_and_pass_counts"] = dict(sorted(counters[workspace].items()))
        merged["disposition"] = disposition
        result.append(merged)
    return result


def _ledger_proposal(
    *,
    lane_dir: Path,
    adapter_sha256: str,
    adapted_labels_sha256: str,
    schema_map_sha256: str,
    selected: list[dict[str, Any]],
    venue_counts: dict[str, int],
    per_workspace: list[dict[str, Any]],
    exclusion_evidence: dict[str, Any],
) -> dict[str, Any]:
    yield_by_workspace = {
        row["workspace"]: row["yield_after_dedup_sanity_cap"]
        for row in per_workspace
        if row["yield_after_dedup_sanity_cap"]
    }
    corpus_path = display_path(lane_dir / "adapted_corpus")
    labels_path = display_path(lane_dir / "adapted_labels.jsonl")
    schema_path = display_path(lane_dir / "schema_map.json")
    return {
        "proposal_only": True,
        "must_not_apply_automatically": True,
        "integration_instruction": (
            "Integration manager should review this row and update runs/manager/data_ledger.json; "
            "this adapter did not modify the ledger."
        ),
        "asset_id": "roboflow_court_keypoints_adapted_20260723",
        "source_asset_id": "roboflow_court_taxonomy_20260706",
        "acquired_utc": "2026-07-06T20:00:00Z",
        "paths": [
            {
                "path": corpus_path,
                "role": "trainer-compatible canonical-15 dictionaries with floor-only non-null supervision",
                "present": True,
            },
            {
                "path": labels_path,
                "role": "flat provenance index for adapted rows",
                "present": True,
            },
        ],
        "counts": {
            "adapted_usable_label_count": len(selected),
            "label_unit": "deduplicated_sanity_passed_floor_supervision_images",
            "effective_non_null_keypoints_per_row": 12,
            "canonical_dictionary_key_count_per_row": 15,
            "workspace_yield": yield_by_workspace,
            "distinct_apparent_venue_count": len(venue_counts),
        },
        "source_lineage": {
            "original_sources": sorted(yield_by_workspace),
            "sessions": ["roboflow_universe_20260706"],
            "apparent_venue_definition": "source-group plus normalized source-media filename family",
        },
        "immutable_hashes": [
            {
                "path": "scripts/racketsport/adapt_roboflow_court_keypoints.py",
                "algorithm": "sha256",
                "digest": adapter_sha256,
                "role": "adapter implementation",
            },
            {
                "path": labels_path,
                "algorithm": "sha256",
                "digest": adapted_labels_sha256,
                "role": "adapted row index",
            },
            {
                "path": schema_path,
                "algorithm": "sha256",
                "digest": schema_map_sha256,
                "role": "workspace schema and mapping evidence",
            },
        ],
        "rights": {
            "posture": "Preserve source license/attribution metadata and the current owner floor-only act.",
            "component_rulings": {
                "COURT": {
                    "decision": "AUTHORIZE_TRACK_A_FLOOR_ONLY_AFTER_LEDGER_INTEGRATION_REVIEW",
                    "ruling": (
                        "Use only the emitted 12 floor channels; all three external net channels "
                        "remain null. This is training data, never independent gate truth."
                    ),
                },
                "BALL": {"decision": "FORBID", "ruling": "Court labels are not ball labels."},
                "EVENT": {"decision": "FORBID", "ruling": "Court labels are not event labels."},
            },
        },
        "protection": {
            "compare_only_matches": exclusion_evidence["compare_only_matches"],
            "protected_matches": exclusion_evidence["protected_git_blob_matches"],
            "owner_pack_near_match_excluded_count": exclusion_evidence[
                "owner_pack_near_match_excluded_count"
            ],
            "selected_owner_pack_collision_count": exclusion_evidence[
                "selected_owner_pack_collision_count"
            ],
        },
        "partitions": {
            "strategy": "group by source_group/apparent_venue_id; never frame-random split related frames",
            "train": sorted(yield_by_workspace),
            "val": [],
            "test": [],
        },
        "disposition": {
            "consumer_track": "A",
            "training_intent": True,
            "next_queue_action": (
                "Manager reviews this proposed row, preserves source-group splits, then points Track A "
                f"at {corpus_path}. Do not promote model authority from these rows."
            ),
        },
        "state": "PROPOSED_READY_FOR_LEDGER_REVIEW",
        "state_reason": (
            "Rows passed canonical-schema loading, projective geometry, dedup, venue caps, and "
            "protected/owner-pack guards; VERIFIED=0 remains binding."
        ),
        "verified": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    lane_dir = args.lane_dir.resolve()
    owner_pack_root = args.owner_pack_root.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is missing: {dataset_root}")
    lane_dir.mkdir(parents=True, exist_ok=True)
    _clean_generated_outputs(lane_dir)

    canonical_contract = pin_canonical_contract()
    inventory = inventory_court_workspaces(dataset_root)
    adapter_sha256 = sha256_file(Path(__file__))
    candidates, counters, exclusion_evidence = build_candidates(
        dataset_root,
        inventory,
        mode=args.mode,
        max_reprojection_error_ratio=args.max_reprojection_error_ratio,
        owner_pack_root=owner_pack_root,
        phash_threshold=args.phash_hamming_threshold,
    )
    deduplicated = perceptual_deduplicate(
        candidates,
        counters,
        phash_threshold=args.phash_hamming_threshold,
    )
    selected, venue_counts = cap_apparent_venues(
        deduplicated,
        counters,
        max_per_apparent_venue=args.max_per_apparent_venue,
    )
    if not selected:
        raise RuntimeError("adapter produced zero usable rows after sanity, eligibility, dedup, and venue caps")

    per_workspace = _workspace_report_rows(inventory, selected, counters, mode=args.mode)
    write_json(lane_dir / "schema_map.json", {"schema_version": 1, "workspaces": per_workspace})
    (lane_dir / "schema_map.md").write_text(_schema_map_markdown(per_workspace), encoding="utf-8")
    corpus_root, emitted_jsonl = emit_loader_corpus(
        selected,
        lane_dir=lane_dir,
        mode=args.mode,
        adapter_sha256=adapter_sha256,
    )
    loader_validation = validate_loader_contract(corpus_root, mode=args.mode, expected_count=len(selected))
    validation_paths = render_validation_samples(
        selected,
        lane_dir=lane_dir,
        limit=args.validation_samples,
    )

    adapted_labels_sha256 = sha256_file(lane_dir / "adapted_labels.jsonl")
    schema_map_sha256 = sha256_file(lane_dir / "schema_map.json")
    proposal = _ledger_proposal(
        lane_dir=lane_dir,
        adapter_sha256=adapter_sha256,
        adapted_labels_sha256=adapted_labels_sha256,
        schema_map_sha256=schema_map_sha256,
        selected=selected,
        venue_counts=venue_counts,
        per_workspace=per_workspace,
        exclusion_evidence=exclusion_evidence,
    )
    write_json(lane_dir / "PROPOSED_LEDGER_ROW.json", proposal)

    configured_present = [row for row in inventory if row["workspace"] in DATASET_MAPPINGS]
    approved_present = [
        row
        for row in configured_present
        if DATASET_MAPPINGS[row["workspace"]]["owner_approved_floor_only"]
    ]
    skipped = [row for row in per_workspace if row["disposition"].startswith("skipped")]
    sanity_pass_total = sum(counter["sanity_pass"] for counter in counters.values())
    report = {
        "artifact_type": "racketsport_roboflow_court_keypoint_adapter_report",
        "schema_version": 1,
        "lane": LANE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "verified": False,
        "status": "PASS",
        "canonical_contract": canonical_contract,
        "adapter": {
            "path": "scripts/racketsport/adapt_roboflow_court_keypoints.py",
            "sha256": adapter_sha256,
        },
        "counts": {
            "court_workspaces_enumerated": len(inventory),
            "workspaces_with_resolved_keypoint_mapping": len(configured_present),
            "workspaces_in_current_owner_approved_floor_scope": len(approved_present),
            "workspaces_skipped": len(skipped),
            "workspaces_with_nonzero_yield": sum(
                1 for row in per_workspace if row["yield_after_dedup_sanity_cap"] > 0
            ),
            "raw_court_workspace_images": sum(row["image_count"] for row in inventory),
            "geometric_sanity_pass_before_dedup": sanity_pass_total,
            "adapted_usable_labels_after_dedup_sanity_and_venue_cap": len(selected),
            "effective_floor_keypoints": sum(
                sum(value is not None for value in row["keypoints"].values()) for row in selected
            ),
            "canonical_dictionary_keys": len(selected) * len(EXPECTED_CANONICAL_ORDER),
            "distinct_apparent_venue_count": len(venue_counts),
            "validation_png_count": len(validation_paths),
        },
        "per_workspace": per_workspace,
        "apparent_venue_counts": venue_counts,
        "dedup_policy": {
            "exact_sha256": True,
            "roboflow_origin_key": "source_group plus filename with .rf.<hash> removed",
            "perceptual_hash": "64-bit DCT pHash",
            "phash_hamming_threshold": args.phash_hamming_threshold,
            "max_per_apparent_venue": args.max_per_apparent_venue,
            "cap_selection": "deterministic farthest-first pHash diversity",
        },
        "geometric_sanity": {
            "outer_quad_convex": True,
            "minimum_outer_quad_area_ratio": 0.02,
            "max_reprojection_error_ratio": args.max_reprojection_error_ratio,
            "projective_baseline_kitchen_and_left_center_right_order": True,
            "failed_rows_dropped": True,
        },
        "exclusions": exclusion_evidence,
        "loader_validation": loader_validation,
        "validation_pngs": validation_paths,
        "artifacts": {
            "adapted_corpus": display_path(corpus_root),
            "adapted_labels_jsonl": display_path(lane_dir / "adapted_labels.jsonl"),
            "adapted_labels_sha256": adapted_labels_sha256,
            "schema_map_json": display_path(lane_dir / "schema_map.json"),
            "schema_map_md": display_path(lane_dir / "schema_map.md"),
            "proposed_ledger_row": display_path(lane_dir / "PROPOSED_LEDGER_ROW.json"),
        },
        "honest_caveats": [
            "VERIFIED=0. These external training rows are not independent calibration gate truth.",
            (
                "The dispatch's example names are Roboflow source names, not our canonical trainer names; "
                "the adapter pins the actual near_left_corner/... order."
            ),
            (
                "Chetan source far/near is reversed relative to our camera-depth convention. The static "
                "semantic mapping corrects that reversal before projective sanity checks."
            ),
            (
                "All three external net channels are null. The binding owner review found bottom-of-net "
                "markers, while our canonical net channels require top-of-net."
            ),
            (
                "Generic index/letter mappings are static geometry-inferred mappings supported by skeleton "
                "topology and overlays; they are not inferred independently per image."
            ),
            (
                "Distinct venue count is an apparent source-camera-family count inferred from filenames, "
                "not a claim of known physical venue identity; Roboflow metadata cannot prove venue identity."
            ),
            (
                "Two resolved generic workspaces remain outside the current immutable owner act and are "
                "enumerated but not emitted in approved mode."
            ),
            (
                "The proposed ledger row was written only to this lane; runs/manager/data_ledger.json was not modified."
            ),
        ],
        "verified_state": "VERIFIED=0",
    }
    write_json(lane_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt Roboflow COCO court keypoints to the canonical 15-key trainer schema."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lane-dir", type=Path, default=DEFAULT_LANE_DIR)
    parser.add_argument("--owner-pack-root", type=Path, default=DEFAULT_OWNER_PACK_ROOT)
    parser.add_argument(
        "--mode",
        choices=("approved", "diagnostic"),
        default="approved",
        help=(
            "approved restricts rows to the pinned owner act and immutable source-row manifest; "
            "diagnostic emits PENDING_SPOTCHECK rows and never claims training eligibility"
        ),
    )
    parser.add_argument("--phash-hamming-threshold", type=int, default=4)
    parser.add_argument("--max-per-apparent-venue", type=int, default=50)
    parser.add_argument("--max-reprojection-error-ratio", type=float, default=0.04)
    parser.add_argument("--validation-samples", type=int, default=12)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 0 <= args.phash_hamming_threshold <= 64:
        parser.error("--phash-hamming-threshold must be in [0,64]")
    if args.max_per_apparent_venue <= 0:
        parser.error("--max-per-apparent-venue must be positive")
    if not 0 < args.max_reprojection_error_ratio < 1:
        parser.error("--max-reprojection-error-ratio must be in (0,1)")
    if args.validation_samples <= 0:
        parser.error("--validation-samples must be positive")
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verified": report["verified"],
                "adapted_usable_labels": report["counts"][
                    "adapted_usable_labels_after_dedup_sanity_and_venue_cap"
                ],
                "distinct_apparent_venue_count": report["counts"]["distinct_apparent_venue_count"],
                "report": str((args.lane_dir.resolve() / "report.json")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
