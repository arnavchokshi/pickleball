"""Court-surface and paint evidence for detector v2."""

from __future__ import annotations

from typing import Any


def build_surface_paint_evidence(image_bgr: Any, *, roi: dict[str, int] | None = None) -> dict[str, Any]:
    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")

    import numpy as np

    if len(image_bgr.shape) == 2:
        bgr = np.repeat(image_bgr[:, :, None], 3, axis=2).astype(np.float32)
    else:
        bgr = image_bgr.astype(np.float32)
    height, width = bgr.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image_bgr must have positive dimensions")

    sample = _roi_sample(bgr, roi=roi)
    surface_color = np.median(sample.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(bgr - surface_color.reshape(1, 1, 3), axis=2)
    sample_distance = np.linalg.norm(sample.reshape(-1, 3) - surface_color.reshape(1, 3), axis=1)
    local_threshold = max(34.0, float(np.percentile(sample_distance, 82)) + 10.0)

    channel_min = bgr.min(axis=2)
    channel_max = bgr.max(axis=2)
    white_mask = (channel_min >= 190.0) & ((channel_max - channel_min) <= 90.0)
    local_mask = distance >= local_threshold
    top_cutoff = int(round(height * 0.20))
    white_mask[:top_cutoff, :] = False
    local_mask[:top_cutoff, :] = False

    if _support_ratio(white_mask) >= 0.001:
        mask = white_mask
        mode = "white"
    else:
        mask = local_mask
        mode = "local_contrast"

    line_pixels = bgr[mask]
    line_color = np.median(line_pixels, axis=0) if line_pixels.size else surface_color

    line_candidates = _line_candidates_from_mask(mask)
    return {
        "surface_color_bgr": [round(float(value), 3) for value in surface_color],
        "line_color_bgr": [round(float(value), 3) for value in line_color],
        "line_color_mode": mode,
        "mask_support_ratio": round(_support_ratio(mask), 6),
        "shadow_normalization_applied": False,
        "semantic_line_candidates": line_candidates,
    }


def _roi_sample(image: Any, *, roi: dict[str, int] | None) -> Any:
    height, width = image.shape[:2]
    if roi:
        x_min = max(0, min(width - 1, int(roi.get("x_min", 0))))
        x_max = max(x_min + 1, min(width, int(roi.get("x_max", width))))
        y_min = max(0, min(height - 1, int(roi.get("y_min", 0))))
        y_max = max(y_min + 1, min(height, int(roi.get("y_max", height))))
    else:
        x_min, x_max = int(round(width * 0.08)), int(round(width * 0.92))
        y_min, y_max = int(round(height * 0.35)), int(round(height * 0.95))
    return image[y_min:y_max, x_min:x_max]


def _support_ratio(mask: Any) -> float:
    import numpy as np

    total = int(mask.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total)


def _line_candidates_from_mask(mask: Any) -> list[dict[str, Any]]:
    import math
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    mask_u8 = mask.astype(np.uint8) * 255
    height, width = mask_u8.shape[:2]
    edges = cv2.Canny(mask_u8, 50, 150)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=24,
        minLineLength=max(20, int(round(min(width, height) * 0.12))),
        maxLineGap=max(8, int(round(max(width, height) * 0.02))),
    )
    if raw is None:
        return []
    candidates: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 20.0:
            continue
        candidates.append(
            {
                "p1": [float(x1), float(y1)],
                "p2": [float(x2), float(y2)],
                "angle_deg": round(float(math.degrees(math.atan2(dy, dx))), 3),
                "length_px": round(float(length), 3),
            }
        )
    candidates.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return candidates[:32]
