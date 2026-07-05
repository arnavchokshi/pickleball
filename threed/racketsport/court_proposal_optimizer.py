"""Guarded proposal refinement helpers."""

from __future__ import annotations

from typing import Any


def accept_refinement(before: dict[str, float], after: dict[str, float]) -> tuple[bool, list[str]]:
    """Accept a refinement only when line, pixel, median, and tail metrics hold."""

    reasons: list[str] = []
    if after.get("line_rmse_px", 1e9) > before.get("line_rmse_px", 1e9) * 0.95:
        reasons.append("line_residual_not_improved")
    if after.get("pixel_support", 0.0) < before.get("pixel_support", 0.0) - 0.02:
        reasons.append("pixel_support_worsened")
    if after.get("p95_px", 1e9) > before.get("p95_px", 1e9) + 10.0:
        reasons.append("p95_worsened")
    if after.get("median_px", 1e9) > before.get("median_px", 1e9) + 5.0:
        reasons.append("median_worsened")
    return not reasons, reasons


def refine_homography_with_lines(
    initial_h: list[list[float]],
    semantic_lines: dict[str, object],
    line_distance_map: object,
    keypoint_priors: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Run a guarded refinement shell and keep the initial homography on reject."""

    before = score_homography_support(initial_h, semantic_lines, line_distance_map, keypoint_priors)
    refined_h = run_guarded_line_refinement(initial_h, semantic_lines, line_distance_map, keypoint_priors)
    after = score_homography_support(refined_h, semantic_lines, line_distance_map, keypoint_priors)
    accepted, reasons = accept_refinement(before, after)
    if refined_h == initial_h:
        accepted = False
        reasons = [*reasons, "optimizer_not_wired"]
    return {
        "accepted": accepted,
        "homography_image_from_court": refined_h if accepted else initial_h,
        "scores_before": before,
        "scores_after": after,
        "reject_reasons": reasons,
    }


def score_homography_support(
    homography: list[list[float]],
    semantic_lines: dict[str, object],
    line_distance_map: object,
    keypoint_priors: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Return support metrics for a homography.

    The full optimizer is benchmark-owned today. This helper intentionally starts
    conservative so callers can record guard telemetry before any optimization is
    trusted.
    """

    _ = homography, semantic_lines, line_distance_map, keypoint_priors
    return {"line_rmse_px": 0.0, "pixel_support": 0.0, "p95_px": 0.0, "median_px": 0.0}


def run_guarded_line_refinement(
    initial_h: list[list[float]],
    semantic_lines: dict[str, object],
    line_distance_map: object,
    keypoint_priors: dict[str, tuple[float, float]] | None = None,
) -> list[list[float]]:
    """Return the input homography until the optimizer is wired to real metrics."""

    _ = semantic_lines, line_distance_map, keypoint_priors
    return initial_h
