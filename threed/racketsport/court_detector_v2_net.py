"""Net evidence extraction for court detector v2."""

from __future__ import annotations

from typing import Any

from .net_anchor_court import detect_net_anchor


def detect_court_net_evidence(frame_bgr: Any) -> dict[str, Any]:
    """Return net evidence used only as ROI/orientation/scale anchor.

    Top-tape points are intentionally marked as non-floor evidence. The floor
    homography path must consume only court-floor correspondences.
    """

    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        raise ValueError("frame_bgr must be an image array")
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("frame_bgr must have positive dimensions")

    try:
        net = detect_net_anchor(frame_bgr)
        top_tape = [[float(x), float(y)] for x, y in net.tape_line]
        post_tops = [[float(x), float(y)] for x, y in net.post_tops]
        post_bases = [[float(x), float(y)] for x, y in net.post_bases]
        confidence = float(net.confidence)
        evidence = dict(net.evidence)
    except Exception as exc:  # pragma: no cover - defensive for unreadable frames
        top_tape = [[0.1 * width, 0.55 * height], [0.9 * width, 0.55 * height]]
        post_tops = []
        post_bases = []
        confidence = 0.0
        evidence = {"fallback_reason": str(exc)}

    xs = [point[0] for point in top_tape] + [point[0] for point in post_tops] + [point[0] for point in post_bases]
    ys = [point[1] for point in top_tape] + [point[1] for point in post_tops] + [point[1] for point in post_bases]
    if not xs or not ys:
        xs = [0.1 * width, 0.9 * width]
        ys = [0.45 * height, 0.65 * height]
    x_pad = max(20.0, (max(xs) - min(xs)) * 0.25)
    y_pad = max(40.0, height * 0.25)
    y_mid = sum(top_tape_point[1] for top_tape_point in top_tape) / max(1, len(top_tape))
    roi = {
        "x_min": int(max(0, round(min(xs) - x_pad))),
        "y_min": int(max(0, round(y_mid - y_pad))),
        "x_max": int(min(width, round(max(xs) + x_pad))),
        "y_max": int(min(height, round(max(ys) + y_pad))),
    }

    return {
        "anchor_role": "roi_orientation_scale_only",
        "uses_top_net_as_floor_point": False,
        "top_tape_line": top_tape,
        "post_candidates": {"tops": post_tops, "bases": post_bases},
        "mesh_band": evidence.get("mesh_band", {}),
        "bottom_band_or_floor_net_candidate": post_bases,
        "roi": roi,
        "confidence": max(0.0, min(1.0, confidence)),
        "legacy_net_anchor_evidence": evidence,
    }
