"""SAM-3D Body input preparation helpers.

These helpers are CPU-safe: they prepare bbox padding, optional mask prompt
paths, static camera intrinsics, and optional soft-background images without
importing the Fast-SAM-3D-Body runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .court_calibration import calibration_image_size
from .schemas import CourtCalibration


ARTIFACT_TYPE = "racketsport_sam3d_body_input_prep"
MASK_PROMPT_ARTIFACT_TYPE = "racketsport_sam3d_body_mask_prompts"
ACCURACY_OPT_SOURCE = "sam3d_accuracy_opt_20260703T0xZ"
SUPPORTED_BODY_INPUT_SIZES = (384, 448, 512)


def normalize_body_input_size(value: int | None) -> int | None:
    if value is None:
        return None
    size = int(value)
    if size not in SUPPORTED_BODY_INPUT_SIZES:
        raise ValueError(f"sam3d_body_input_size_px must be one of {SUPPORTED_BODY_INPUT_SIZES}, got {size}")
    return size


def normalize_crop_padding_scale(value: float) -> float:
    scale = float(value)
    if scale < 1.0:
        raise ValueError("sam3d_crop_padding_scale must be >= 1.0")
    if scale > 2.0:
        raise ValueError("sam3d_crop_padding_scale must be <= 2.0")
    return scale


def normalize_soft_background_alpha(value: float) -> float:
    alpha = float(value)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("sam3d_soft_background_alpha must be in (0, 1]")
    return alpha


def padded_bbox_xyxy(
    bbox_xyxy: Sequence[float],
    *,
    image_size_px: Sequence[int],
    padding_scale: float,
) -> list[float]:
    bbox = [float(value) for value in bbox_xyxy]
    if len(bbox) != 4:
        raise ValueError("bbox_xyxy must have 4 values")
    width, height = int(image_size_px[0]), int(image_size_px[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size_px must be positive")
    scale = normalize_crop_padding_scale(padding_scale)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * scale * 0.5
    half_h = (y2 - y1) * scale * 0.5
    return [
        _clean_float(max(0.0, cx - half_w)),
        _clean_float(max(0.0, cy - half_h)),
        _clean_float(min(float(width), cx + half_w)),
        _clean_float(min(float(height), cy + half_h)),
    ]


def static_camera_intrinsics_k(
    calibration: CourtCalibration,
    *,
    image_size_px: Sequence[int],
) -> list[list[float]]:
    """Scale calibration intrinsics once to the BODY frame coordinate space."""

    width, height = float(image_size_px[0]), float(image_size_px[1])
    calibration_width, calibration_height = calibration_image_size(
        calibration,
        fallback_target=(width, height),
    )
    scale_x = width / calibration_width if calibration_width > 0.0 else 1.0
    scale_y = height / calibration_height if calibration_height > 0.0 else 1.0
    intrinsics = calibration.intrinsics
    return [
        [_clean_float(float(intrinsics.fx) * scale_x), 0.0, _clean_float(float(intrinsics.cx) * scale_x)],
        [0.0, _clean_float(float(intrinsics.fy) * scale_y), _clean_float(float(intrinsics.cy) * scale_y)],
        [0.0, 0.0, 1.0],
    ]


class MaskPromptLookup:
    def __init__(self, *, mode: str, manifest_path: Path | None, lookup: dict[tuple[int, int], Path]) -> None:
        self.mode = mode
        self.manifest_path = manifest_path
        self.lookup = lookup

    @property
    def configured(self) -> bool:
        return self.mode == "manifest" and self.manifest_path is not None

    def path_for(self, *, frame_idx: int, player_id: int) -> Path | None:
        path = self.lookup.get((int(frame_idx), int(player_id)))
        if path is None or not path.is_file():
            return None
        return path


def load_mask_prompt_lookup(run_dir: Path, *, artifact_name: str, mode: str) -> MaskPromptLookup:
    if mode not in {"off", "manifest"}:
        raise ValueError("sam3d_mask_prompt_mode must be 'off' or 'manifest'")
    if mode == "off":
        return MaskPromptLookup(mode=mode, manifest_path=None, lookup={})
    manifest_path = run_dir / artifact_name
    if not manifest_path.is_file():
        return MaskPromptLookup(mode=mode, manifest_path=None, lookup={})
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_frames = payload.get("frames", []) if isinstance(payload, Mapping) else []
    lookup: dict[tuple[int, int], Path] = {}
    for item in raw_frames:
        if not isinstance(item, Mapping):
            continue
        try:
            frame_idx = int(item["frame_idx"])
            player_id = int(item["player_id"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_path = item.get("mask_path") or item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        mask_path = Path(raw_path)
        if not mask_path.is_absolute():
            mask_path = manifest_path.parent / mask_path
        lookup[(frame_idx, player_id)] = mask_path
    return MaskPromptLookup(mode=mode, manifest_path=manifest_path, lookup=lookup)


def write_soft_background_image(
    *,
    image_path: Path,
    mask_path: Path,
    out_path: Path,
    background_alpha: float,
) -> tuple[Path, bool, str]:
    """Write a soft-suppressed copy and return a safe fallback on any issue."""

    alpha = normalize_soft_background_alpha(background_alpha)
    if alpha >= 1.0:
        return image_path, False, "disabled"
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return image_path, False, "opencv_or_numpy_unavailable"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return image_path, False, "image_unreadable"
    if mask is None:
        return image_path, False, "mask_unreadable"
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=8.0)
    softened = (alpha * image.astype("float32") + (1.0 - alpha) * background.astype("float32")).clip(0, 255).astype("uint8")
    foreground = mask > 0
    output = np.where(foreground[:, :, None], image, softened)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), output):
        return image_path, False, "write_failed"
    return out_path, True, "applied"


def load_mask_prompt_arrays(mask_paths: Sequence[str | Path | None] | None) -> Any | None:
    paths = [Path(path) for path in (mask_paths or []) if path]
    if not paths:
        return None
    if any(not path.is_file() for path in paths):
        return None
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None
    masks = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        masks.append(mask > 0)
    return np.asarray(masks, dtype="uint8")


def request_prep_artifact(
    *,
    camera_intrinsics_k: Sequence[Sequence[float]],
    camera_intrinsics_image_size_px: Sequence[int],
    mask_mode: str,
    mask_manifest_path: Path | None,
    records: Sequence[Mapping[str, Any]],
    body_input_size_px: int | None,
    crop_padding_scale: float,
    soft_background_alpha: float,
) -> dict[str, Any]:
    available_count = sum(1 for record in records if record.get("mask_prompt_path"))
    missing_count = sum(1 for record in records if not record.get("mask_prompt_path"))
    soft_count = sum(1 for record in records if record.get("soft_background_applied") is True)
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "source": ACCURACY_OPT_SOURCE,
        "camera_intrinsics": {
            "source": "court_calibration.json",
            "static_per_clip": True,
            "image_size_px": [int(camera_intrinsics_image_size_px[0]), int(camera_intrinsics_image_size_px[1])],
            "K": [[float(value) for value in row] for row in camera_intrinsics_k],
        },
        "crop": {
            "sam3d_body_input_size_px": body_input_size_px,
            "supported_sweep_sizes_px": list(SUPPORTED_BODY_INPUT_SIZES),
            "padding_scale": crop_padding_scale,
        },
        "mask_prompts": {
            "mode": mask_mode,
            "manifest_path": str(mask_manifest_path) if mask_manifest_path is not None else "",
            "available_count": available_count,
            "missing_count": missing_count,
            "fallback": "box_only_when_mask_absent",
        },
        "soft_background": {
            "background_alpha": soft_background_alpha,
            "applied_count": soft_count,
            "fallback": "original_image_when_mask_or_cv2_unavailable",
        },
        "records": list(records),
        "validation": {
            "protected_eval_labels_used": False,
            "gpu_required_for_model_accuracy": True,
            "local_cpu_safe": True,
        },
    }


def _clean_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if abs(rounded) < 1e-9 else rounded
