"""Court-surface and paint evidence for detector v2."""

from __future__ import annotations

from typing import Any


def build_surface_paint_evidence(image_bgr: Any, *, roi: dict[str, int] | None = None) -> dict[str, Any]:
    result = _build_surface_paint_evidence_core(image_bgr, roi=roi)
    try:
        polygon_evidence = segment_court_surface_regions(image_bgr)
    except Exception as exc:  # pragma: no cover - defensive, never break line-color evidence
        polygon_evidence = {"interior_polygon": None, "method": "lab_kmeans_convex_quad", "reason": str(exc)}
    result["surface_polygon"] = polygon_evidence
    return result


def _build_surface_paint_evidence_core(image_bgr: Any, *, roi: dict[str, int] | None = None) -> dict[str, Any]:
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


def segment_court_surface_regions(image_bgr: Any, *, max_dim: int = 320, k: int = 4) -> dict[str, Any]:
    """Real dominant-color court-surface segmentation (Stage 1(b), NEW signal).

    Pickleball/tennis court interiors are almost always a single solid
    saturated paint color (blue/purple/green) distinct from their surround.
    This clusters a downsampled frame in LAB color space, picks the cluster
    that best matches a plausible court-interior region (large connected
    component, not dominated by the top-of-frame background band, moderate
    chroma), and fits a convex quadrilateral to its largest component. This is
    the surface counterpart to the existing (line-color) HSV masking, which
    only ever masked LINES -- this masks the SURFACE, which is new signal.

    Never raises for a readable image; returns interior_polygon=None with a
    reason when no plausible court-like cluster exists.
    """

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    bgr = image_bgr if len(image_bgr.shape) == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    height, width = bgr.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image_bgr must have positive dimensions")

    scale = min(1.0, float(max_dim) / float(max(width, height)))
    small = cv2.resize(bgr, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    small_h, small_w = small.shape[:2]
    lab = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab.reshape(-1, 3)

    k_effective = max(2, min(int(k), max(2, samples.shape[0] // 64)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _compactness, labels, centers = cv2.kmeans(samples, k_effective, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small_h, small_w)

    best: dict[str, Any] | None = None
    top_band = max(1, int(round(small_h * 0.12)))
    for cluster_id in range(k_effective):
        cluster_mask = (labels == cluster_id).astype(np.uint8)
        area = int(cluster_mask.sum())
        if area < max(64, small_h * small_w * 0.05):
            continue
        num_components, component_labels, stats, centroids = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)
        if num_components <= 1:
            continue
        component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        area_fraction = component_area / float(small_h * small_w)
        if area_fraction < 0.05:
            continue
        component_mask = (component_labels == component_id).astype(np.uint8)
        top_band_fraction = float(component_mask[:top_band, :].sum()) / float(max(1, top_band * small_w))
        center_lab = centers[cluster_id]
        chroma = float(np.hypot(center_lab[1] - 128.0, center_lab[2] - 128.0))
        # Prefer large, roughly centered regions that are not dominated by the
        # top band (sky/background/crowd) and have plausible court chroma.
        score = area_fraction * 2.2 + min(1.0, chroma / 40.0) * 0.8 - top_band_fraction * 1.6
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "mask": component_mask,
                "area_fraction": area_fraction,
                "center_lab": center_lab,
            }

    if best is None:
        return {"interior_polygon": None, "method": "lab_kmeans_convex_quad", "reason": "no_plausible_court_cluster"}

    contours, _ = cv2.findContours(best["mask"] * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"interior_polygon": None, "method": "lab_kmeans_convex_quad", "reason": "no_contour"}
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for epsilon_frac in (0.02, 0.035, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, epsilon_frac * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect)

    ordered = _order_quad_corners(quad)
    inverse_scale = 1.0 / scale if scale > 0 else 1.0
    polygon = [[round(float(x) * inverse_scale, 2), round(float(y) * inverse_scale, 2)] for x, y in ordered]
    boundary_edges = [
        {"p1": polygon[index], "p2": polygon[(index + 1) % 4]}
        for index in range(4)
    ]
    return {
        "interior_polygon": polygon,
        "boundary_edges": boundary_edges,
        "dominant_color_lab": [round(float(value), 2) for value in best["center_lab"]],
        "area_fraction": round(float(best["area_fraction"]), 4),
        "method": "lab_kmeans_convex_quad",
        "cluster_count": int(k_effective),
    }


def _order_quad_corners(points: Any) -> list[tuple[float, float]]:
    """Order 4 arbitrary quad points as top-left, top-right, bottom-right, bottom-left."""

    import numpy as np

    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    top_left = pts[int(sums.argmin())]
    bottom_right = pts[int(sums.argmax())]
    top_right = pts[int(diffs.argmax())]
    bottom_left = pts[int(diffs.argmin())]
    ordered = [top_left, top_right, bottom_right, bottom_left]
    # Guard against degenerate argmin/argmax collisions (e.g. near-axis-aligned
    # quads) picking the same point twice.
    if len({tuple(point) for point in ordered}) < 4:
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        ordered_indices = np.argsort(angles)
        ordered = [pts[index] for index in ordered_indices]
    return [(float(x), float(y)) for x, y in ordered]
