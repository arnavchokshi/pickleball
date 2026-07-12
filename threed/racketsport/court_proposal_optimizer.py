"""Guarded point-and-line court-pose refinement.

The optimizer in this module is an original implementation of the point/line
objectives summarized in the repository's 2026-07-12 court-lock research.  It
does not contain or derive from PnLCalib or No-Bells source code.

Two parameterizations are supported.  A planar homography is optimized
directly when no camera model is supplied.  When a calibration contains a
fixed intrinsic/distortion model, camera pose is optimized directly against
undistorted observations.  Its pose update is accompanied by a raw-space
planar proxy synthesized through the declared distortion model, so raw and
undistorted pixels are never mixed in one solve or output field.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from .court_calibration import homography_from_planar_points, project_planar_points
from .court_keypoint_net import PICKLEBALL_KEYPOINTS, refine_keypoint_xy_with_planar_homography
from .court_templates import get_court_template


NET_TOP_KEYPOINT_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
FLOOR_KEYPOINTS = tuple(point for point in PICKLEBALL_KEYPOINTS if point.name not in NET_TOP_KEYPOINT_NAMES)
FLOOR_KEYPOINT_BY_NAME = {point.name: point for point in FLOOR_KEYPOINTS}


@dataclass(frozen=True)
class RefinementConfig:
    """Explicit weights and fail-closed thresholds for one refinement."""

    line_weight: float = 0.60
    point_weight: float = 0.40
    line_samples_per_segment: int = 33
    heldout_stride: int = 5
    robust_loss: str = "soft_l1"
    robust_scale_px: float = 3.0
    magsac_reprojection_threshold_px: float = 5.0
    min_line_observations: int = 2
    min_line_samples: int = 24
    min_point_correspondences: int = 4
    max_nfev: int = 500
    max_condition_number: float = 1.0e12
    max_corner_shift_px: float = 80.0
    heldout_p90_tolerance_px: float = 0.25
    heldout_max_coverage_drop: float = 0.01
    heldout_max_line_family_p90_regression_px: float = 1.0
    evidence_inlier_px: float = 14.0
    identifiability_regularization: bool = True
    information_noise_px: float = 1.0
    stability_guard_enabled: bool = True
    stability_bootstrap_draws: int = 64
    stability_observation_sigma_px: float = 1.0
    stability_max_regression_fraction: float = 0.10
    stability_floor_only_reserve_fraction: float = 0.02
    stability_random_seed: int = 20260712
    hybrid_variance_floor_px2: float = 0.04
    hybrid_intersection_point_fraction: float = 0.50

    def validate(self) -> None:
        if self.line_weight < 0.0 or self.point_weight < 0.0:
            raise ValueError("point and line weights must be non-negative")
        if not math.isclose(self.line_weight + self.point_weight, 1.0, abs_tol=1e-9):
            raise ValueError("point and line weights must sum to 1")
        if self.line_samples_per_segment < 5:
            raise ValueError("line_samples_per_segment must be at least 5")
        if self.heldout_stride < 3:
            raise ValueError("heldout_stride must be at least 3")
        if self.robust_scale_px <= 0.0:
            raise ValueError("robust_scale_px must be positive")
        if self.information_noise_px <= 0.0:
            raise ValueError("information_noise_px must be positive")
        if self.stability_bootstrap_draws <= 0:
            raise ValueError("stability_bootstrap_draws must be positive")
        if self.stability_observation_sigma_px <= 0.0:
            raise ValueError("stability_observation_sigma_px must be positive")
        if self.stability_max_regression_fraction < 0.0:
            raise ValueError("stability_max_regression_fraction must be non-negative")
        if not 0.0 <= self.stability_floor_only_reserve_fraction < self.stability_max_regression_fraction:
            raise ValueError("stability_floor_only_reserve_fraction must be below the regression limit")
        if self.hybrid_variance_floor_px2 <= 0.0:
            raise ValueError("hybrid_variance_floor_px2 must be positive")
        if not 0.0 <= self.hybrid_intersection_point_fraction <= 1.0:
            raise ValueError("hybrid_intersection_point_fraction must be in [0, 1]")


@dataclass(frozen=True)
class _HybridLineSample:
    xy: tuple[float, float]
    covariance_px2: np.ndarray
    provenance: str


@dataclass(frozen=True)
class _HybridLineSegment:
    endpoints: np.ndarray
    samples: tuple[_HybridLineSample, ...]


@dataclass(frozen=True)
class _LineEvidence:
    line_id: str
    world_segment: np.ndarray
    optimize_segments: tuple[np.ndarray, ...]
    heldout_segments: tuple[np.ndarray, ...]
    optimize_hybrid_segments: tuple[_HybridLineSegment, ...]
    heldout_hybrid_segments: tuple[_HybridLineSegment, ...]
    confidence: float


@dataclass(frozen=True)
class _PointEvidence:
    name: str
    world_xy: tuple[float, float]
    image_xy: tuple[float, float]
    confidence: float
    source: str
    covariance_px2: np.ndarray | None = None


def accept_refinement(before: dict[str, float], after: dict[str, float]) -> tuple[bool, list[str]]:
    """Retain the historical shell guard for callers that score these fields."""

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


def diagnose_planar_keypoint_refinement(
    keypoints: Mapping[str, Sequence[float]],
    *,
    max_inlier_error_px: float = 30.0,
    min_inliers: int = 8,
) -> dict[str, Any]:
    """Instrument the existing learned-keypoint postprocess without changing it.

    This deliberately mirrors its proposal census, then invokes the production
    function for the output.  The census makes the two fallback modes and the
    accidental flattening of physical net-top points machine-readable.
    """

    candidates: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for point in PICKLEBALL_KEYPOINTS:
        value = keypoints.get(point.name)
        if value is None:
            continue
        image_xy = _xy(value, f"keypoints.{point.name}")
        candidates.append((point.name, (point.world_xyz_m[0], point.world_xyz_m[1]), image_xy))

    nondegenerate = 0
    net_contaminated_subsets = 0
    best: tuple[int, float, tuple[str, ...], tuple[str, ...], np.ndarray, list[float]] | None = None
    all_world = [item[1] for item in candidates]
    for subset in combinations(candidates, 4):
        world = [item[1] for item in subset]
        image = [item[2] for item in subset]
        if _max_triangle_area(world) <= 1e-6 or _max_triangle_area(image) <= 1e-3:
            continue
        nondegenerate += 1
        subset_names = tuple(item[0] for item in subset)
        if NET_TOP_KEYPOINT_NAMES.intersection(subset_names):
            net_contaminated_subsets += 1
        try:
            homography = np.asarray(homography_from_planar_points(world, image), dtype=np.float64)
            projected = np.asarray(project_planar_points(homography.tolist(), all_world), dtype=np.float64)
        except ValueError:
            continue
        residuals = [float(np.linalg.norm(point - np.asarray(item[2]))) for point, item in zip(projected, candidates, strict=True)]
        inlier_names = tuple(
            candidates[index][0] for index, residual in enumerate(residuals) if residual <= max_inlier_error_px
        )
        if len(inlier_names) < min_inliers:
            continue
        mean_error = float(np.mean([residuals[index] for index, item in enumerate(candidates) if item[0] in inlier_names]))
        row = (len(inlier_names), mean_error, subset_names, inlier_names, homography, residuals)
        if best is None or row[0] > best[0] or (row[0] == best[0] and row[1] < best[1]):
            best = row

    raw = {name: [xy[0], xy[1]] for name, _, xy in candidates}
    output = refine_keypoint_xy_with_planar_homography(
        keypoints, max_inlier_error_px=max_inlier_error_px, min_inliers=min_inliers
    )
    common = sorted(set(raw).intersection(output))
    deltas = {
        name: float(np.linalg.norm(np.asarray(output[name], dtype=np.float64) - np.asarray(raw[name], dtype=np.float64)))
        for name in common
    }
    if len(candidates) < max(4, min_inliers):
        fallback_reason = "insufficient_candidates"
    elif best is None:
        fallback_reason = "no_consensus_meets_min_inliers"
    elif max(deltas.values(), default=0.0) <= 1e-12:
        fallback_reason = "output_exactly_equals_input"
    else:
        fallback_reason = None

    per_point_residual = {}
    if best is not None:
        per_point_residual = {
            item[0]: float(residual) for item, residual in zip(candidates, best[5], strict=True)
        }
    return {
        "candidate_count": len(candidates),
        "non_degenerate_4_subset_count": nondegenerate,
        "net_contaminated_4_subset_count": net_contaminated_subsets,
        "best_inlier_count": int(best[0]) if best is not None else 0,
        "best_subset_names": list(best[2]) if best is not None else [],
        "best_inlier_names": list(best[3]) if best is not None else [],
        "per_point_residual_px": per_point_residual,
        "fallback_reason": fallback_reason,
        "output_delta_px": {
            "per_point": deltas,
            "max": max(deltas.values(), default=0.0),
            "median": float(np.median(list(deltas.values()))) if deltas else 0.0,
        },
        "net_top_candidate_count": sum(name in NET_TOP_KEYPOINT_NAMES for name, _, _ in candidates),
        "net_top_points_entered_planar_fit": net_contaminated_subsets > 0,
        "net_top_in_best_subset": bool(best and NET_TOP_KEYPOINT_NAMES.intersection(best[2])),
        "net_top_in_best_inliers": bool(best and NET_TOP_KEYPOINT_NAMES.intersection(best[3])),
        "raw_output": raw,
        "refined_output": output,
    }


def magsac_homography_from_points(
    points: Sequence[_PointEvidence] | Sequence[Mapping[str, Any]],
    *,
    threshold_px: float = 5.0,
) -> dict[str, Any]:
    """Fit a floor-only homography with MAGSAC++ and named fallbacks."""

    parsed: list[_PointEvidence] = []
    for index, point in enumerate(points):
        if isinstance(point, _PointEvidence):
            parsed.append(point)
            continue
        name = str(point.get("name", f"point_{index}"))
        if name in NET_TOP_KEYPOINT_NAMES:
            continue
        parsed.append(
            _PointEvidence(
                name=name,
                world_xy=_xy(point["world_xy"], f"points[{index}].world_xy"),
                image_xy=_xy(point["image_xy"], f"points[{index}].image_xy"),
                confidence=float(point.get("confidence", 1.0)),
                source=str(point.get("source", "provided")),
            )
        )
    if len(parsed) < 4:
        return {
            "success": False,
            "reason": "insufficient_floor_correspondences",
            "homography": None,
            "method": None,
            "fallback_reason": None,
            "inlier_count": 0,
            "opencv_version": cv2.__version__,
        }

    world = np.asarray([point.world_xy for point in parsed], dtype=np.float64)
    image = np.asarray([point.image_xy for point in parsed], dtype=np.float64)
    fallback_reason = None
    if hasattr(cv2, "USAC_MAGSAC"):
        method = int(cv2.USAC_MAGSAC)
        method_name = "cv2.USAC_MAGSAC"
    elif hasattr(cv2, "LMEDS"):
        method = int(cv2.LMEDS)
        method_name = "cv2.LMEDS"
        fallback_reason = "opencv_usac_magsac_unavailable"
    else:
        method = int(cv2.RANSAC)
        method_name = "cv2.RANSAC"
        fallback_reason = "opencv_usac_magsac_and_lmeds_unavailable"
    try:
        homography, mask = cv2.findHomography(world, image, method=method, ransacReprojThreshold=float(threshold_px))
    except cv2.error:
        homography, mask = cv2.findHomography(world, image, method=cv2.RANSAC, ransacReprojThreshold=float(threshold_px))
        method_name = "cv2.RANSAC"
        fallback_reason = "selected_robust_method_raised_cv2_error"
    if homography is None or not np.all(np.isfinite(homography)):
        return {
            "success": False,
            "reason": "robust_homography_fit_failed",
            "homography": None,
            "method": method_name,
            "fallback_reason": fallback_reason,
            "inlier_count": 0,
            "opencv_version": cv2.__version__,
        }
    homography = _normalize_h(homography)
    return {
        "success": True,
        "reason": None,
        "homography": homography.tolist(),
        "method": method_name,
        "fallback_reason": fallback_reason,
        "inlier_count": int(np.asarray(mask).sum()) if mask is not None else len(parsed),
        "point_count": len(parsed),
        "opencv_version": cv2.__version__,
    }


def refine_homography_with_lines(
    initial_h: Sequence[Sequence[float]],
    semantic_lines: Mapping[str, object],
    line_distance_map: object,
    keypoint_priors: Mapping[str, object] | None = None,
    *,
    calibration: Mapping[str, Any] | None = None,
    coordinate_space: str | None = None,
    config: RefinementConfig | None = None,
) -> dict[str, Any]:
    """Run guarded direct point+line refinement and fail closed to ``initial_h``."""

    resolved = config or RefinementConfig()
    resolved.validate()
    seed = _validate_h(initial_h)
    raw_stability_points = _raw_floor_points(keypoint_priors or {})
    try:
        lines, points, parse_telemetry = _prepare_evidence(
            semantic_lines,
            keypoint_priors or {},
            calibration=calibration,
            coordinate_space=coordinate_space,
        )
    except ValueError as exc:
        return _return_seed(seed, str(exc), config=resolved)

    line_observation_count = sum(
        len(line.optimize_hybrid_segments) if line.optimize_hybrid_segments else len(line.optimize_segments)
        for line in lines
    )
    line_sample_count = sum(
        sum(len(segment.samples) for segment in line.optimize_hybrid_segments)
        if line.optimize_hybrid_segments
        else len(line.optimize_segments)
        * (resolved.line_samples_per_segment - math.ceil(resolved.line_samples_per_segment / resolved.heldout_stride))
        for line in lines
    )
    if line_observation_count < resolved.min_line_observations or line_sample_count < resolved.min_line_samples:
        return _return_seed(
            seed,
            "insufficient_line_evidence",
            config=resolved,
            telemetry={**parse_telemetry, "line_observation_count": line_observation_count, "line_sample_count": line_sample_count},
        )

    enriched = _synthesize_intersection_points(lines)
    hybrid_intersections = _synthesize_hybrid_intersection_points(lines)
    objective_points = [*points, *hybrid_intersections]
    initializer_points = [*points, *enriched, *hybrid_intersections]
    robust_init = magsac_homography_from_points(
        initializer_points, threshold_px=resolved.magsac_reprojection_threshold_px
    )
    initial_for_solve = seed
    if robust_init["success"]:
        candidate_init = np.asarray(robust_init["homography"], dtype=np.float64)
        # MAGSAC is an initializer, never authority.  Keep it only inside the
        # seed's declared attraction basin.
        if _canonical_corner_shift(seed, candidate_init) <= resolved.max_corner_shift_px:
            initial_for_solve = candidate_init
        else:
            robust_init["fallback_reason"] = "magsac_init_outside_seed_attraction_basin"

    pose_mode = _camera_pose_mode(calibration)
    if pose_mode:
        params_seed = _pose_params_from_calibration(calibration)
        params0 = params_seed.copy()
        seed_undistorted_h = _homography_from_pose_params(params_seed, calibration)
        seed_guard_h = seed
        if robust_init["success"]:
            candidate_init = np.asarray(robust_init["homography"], dtype=np.float64)
            if _canonical_corner_shift(seed_undistorted_h, candidate_init) <= resolved.max_corner_shift_px:
                params0 = _pose_params_from_homography(candidate_init, calibration)
            else:
                robust_init["fallback_reason"] = "magsac_init_outside_seed_pose_attraction_basin"
        projection = lambda params, world: _project_pose_undistorted(params, world, calibration)  # noqa: E731
    else:
        params_seed = _pack_h(seed)
        params0 = _pack_h(initial_for_solve)
        seed_guard_h = seed
        projection = lambda params, world: _project_h(_unpack_h(params), world)  # noqa: E731

    # Endpoint-only geometry intersections enrich initialization only. Hybrid
    # intersections have independently refined band samples and propagated
    # covariance, so they are valid point observations in the point term.
    optimize_residuals = _residual_vector(
        params0, lines, objective_points, projection=projection, config=resolved, heldout=False
    )
    if optimize_residuals.size < len(params0):
        return _return_seed(
            seed,
            "insufficient_independent_residuals",
            config=resolved,
            telemetry={**parse_telemetry, "residual_count": int(optimize_residuals.size)},
        )
    data_residual = lambda params: _residual_vector(  # noqa: E731
        params, lines, objective_points, projection=projection, config=resolved, heldout=False
    )
    identifiability = _identifiability_model(
        params_seed,
        data_residual=data_residual,
        projection=projection,
        config=resolved,
    )
    solve_residual = lambda params: _regularized_residual_vector(  # noqa: E731
        params,
        data_residual=data_residual,
        params_seed=params_seed,
        identifiability=identifiability,
        enabled=resolved.identifiability_regularization,
    )
    try:
        result = least_squares(
            solve_residual,
            params0,
            method="trf",
            loss=resolved.robust_loss,
            f_scale=resolved.robust_scale_px,
            x_scale="jac",
            max_nfev=resolved.max_nfev,
        )
    except (ValueError, FloatingPointError) as exc:
        return _return_seed(seed, "optimizer_exception", config=resolved, telemetry={"exception": str(exc)})
    if not result.success or not np.all(np.isfinite(result.x)):
        return _return_seed(
            seed,
            "optimizer_diverged",
            config=resolved,
            telemetry={"optimizer_status": int(result.status), "optimizer_message": str(result.message)},
        )

    seed_posterior = _posterior_diagnostics(
        params_seed, data_residual=data_residual, identifiability=identifiability
    )
    solution_posterior = _posterior_diagnostics(
        result.x, data_residual=data_residual, identifiability=identifiability
    )
    condition_number = float(solution_posterior["scaled_condition_number"])
    if not math.isfinite(condition_number) or condition_number > resolved.max_condition_number:
        return _return_seed(
            seed,
            "poor_conditioning",
            config=resolved,
            telemetry={"condition_number": condition_number, "optimizer_status": int(result.status)},
        )
    seed_projection = _projection_for_h_or_seed_pose(seed, calibration)
    before = _score_evidence(lines, points, projection=seed_projection, config=resolved, heldout=True)
    stability_points = raw_stability_points if pose_mode else points
    seed_stability = _observation_bootstrap_stability(
        seed_guard_h,
        stability_points,
        draws=resolved.stability_bootstrap_draws,
        sigma_px=resolved.stability_observation_sigma_px,
        random_seed=resolved.stability_random_seed,
    )
    line_search: list[dict[str, Any]] = []
    accepted_rows: list[tuple[float, float, float, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        candidate_params = params_seed + alpha * (result.x - params_seed)
        candidate_h = (
            _raw_homography_proxy_from_pose(candidate_params, calibration)
            if pose_mode
            else _unpack_h(candidate_params)
        )
        corner_shift = _canonical_corner_shift(seed_guard_h, candidate_h)
        candidate_projection = (
            (lambda world, params=candidate_params: _project_pose_undistorted(params, world, calibration))
            if pose_mode
            else (lambda world, h=candidate_h: _project_h(h, world))
        )
        candidate_score = _score_evidence(
            lines, points, projection=candidate_projection, config=resolved, heldout=True
        )
        selection = (
            "refinement_left_seed_attraction_basin"
            if corner_shift > resolved.max_corner_shift_px
            else _heldout_selection_reason(before, candidate_score, resolved)
        )
        candidate_stability = _observation_bootstrap_stability(
            candidate_h,
            stability_points,
            draws=resolved.stability_bootstrap_draws,
            sigma_px=resolved.stability_observation_sigma_px,
            random_seed=resolved.stability_random_seed,
        )
        stability_ratio = _stability_ratio(seed_stability, candidate_stability)
        if (
            selection == "refined_wins_heldout"
            and resolved.stability_guard_enabled
            and (
                not math.isfinite(stability_ratio)
                or stability_ratio
                > 1.0 + resolved.stability_max_regression_fraction - resolved.stability_floor_only_reserve_fraction
            )
        ):
            selection = "seed_wins_observation_bootstrap_stability"
        line_search.append(
            {
                "alpha": alpha,
                "selection_reason": selection,
                "corner_shift_px": corner_shift,
                "p90_px": candidate_score["p90_px"],
                "median_px": candidate_score["median_px"],
                "pixel_support": candidate_score["pixel_support"],
                "observation_bootstrap_worst_point_p95_cm": candidate_stability.get("worst_point_p95_cm"),
                "observation_bootstrap_ratio_vs_seed": stability_ratio,
            }
        )
        if selection == "refined_wins_heldout":
            accepted_rows.append(
                (
                    float(candidate_score["p90_px"]),
                    float(candidate_score["median_px"]),
                    alpha,
                    candidate_params,
                    candidate_h,
                    candidate_score,
                )
            )

    if accepted_rows:
        _p90, _median, selected_alpha, selected_params, refined, after = min(
            accepted_rows, key=lambda row: (row[0], row[1], -row[2])
        )
        accepted = True
        selection_reason = "refined_wins_heldout"
        corner_shift_px = _canonical_corner_shift(seed_guard_h, refined)
        pose_update = _pose_update_payload(selected_params) if pose_mode else None
    else:
        selected_alpha = 0.0
        selected_params = params_seed
        refined = seed
        after = _score_evidence(
            lines,
            points,
            projection=(
                (lambda world: _project_pose_undistorted(result.x, world, calibration))
                if pose_mode
                else (lambda world: _project_h(_unpack_h(result.x), world))
            ),
            config=resolved,
            heldout=True,
        )
        accepted = False
        selection_reason = min(
            line_search, key=lambda row: (row["p90_px"], row["median_px"])
        )["selection_reason"]
        corner_shift_px = 0.0
        pose_update = None
    output_h = refined if accepted else seed
    return {
        "accepted": accepted,
        "homography_image_from_court": output_h.tolist(),
        "scores_before": before,
        "scores_after": after,
        "reject_reasons": [] if accepted else [selection_reason],
        "selection": "refined" if accepted else "seed",
        "selection_reason": selection_reason,
        "covariance_inflation_required": not accepted,
        "guarded": True,
        "objective": {
            "parameterization": "direct_camera_pose" if pose_mode else "direct_homography_8dof",
            "line_weight": resolved.line_weight,
            "point_weight": resolved.point_weight,
            "robust_loss": resolved.robust_loss,
            "robust_scale_px": resolved.robust_scale_px,
            "heldout_policy": f"every_{resolved.heldout_stride}th_line_sample_excluded_from_optimizer",
            "synthesized_intersections_policy": (
                "endpoint_only_initializer_only; band_refined_covariance_intersections_in_point_objective"
            ),
            "hybrid_variance_floor_px2": resolved.hybrid_variance_floor_px2,
            "hybrid_intersection_point_fraction": resolved.hybrid_intersection_point_fraction,
            "identifiability_regularization": resolved.identifiability_regularization,
            "stability_guard": resolved.stability_guard_enabled,
            "stability_guard_effective_limit_fraction": (
                resolved.stability_max_regression_fraction - resolved.stability_floor_only_reserve_fraction
            ),
        },
        "robust_initialization": robust_init,
        "telemetry": {
            **parse_telemetry,
            "line_observation_count": line_observation_count,
            "optimizer_point_count": len(points),
            "geometry_synthesized_point_count": len(enriched),
            "hybrid_intersection_point_count": len(hybrid_intersections),
            "hybrid_band_refined_sample_count": sum(
                sample.provenance == "band_refined"
                for line in lines
                for segment in line.optimize_hybrid_segments
                for sample in segment.samples
            ),
            "net_top_point_count_in_planar_fit": 0,
            "condition_number": condition_number,
            "seed_posterior": seed_posterior,
            "solution_posterior": solution_posterior,
            "identifiability_prior": _json_identifiability(identifiability),
            "influence_at_seed": _influence_diagnostics(
                params_seed, lines, objective_points, projection=projection, config=resolved
            ),
            "influence_at_solution": _influence_diagnostics(
                result.x, lines, objective_points, projection=projection, config=resolved
            ),
            "seed_observation_bootstrap": seed_stability,
            "corner_shift_from_seed_parameterization_px": corner_shift_px,
            "selected_line_search_alpha": selected_alpha,
            "guard_line_search": line_search,
            "optimizer_nfev": int(result.nfev),
            "optimizer_status": int(result.status),
            "coordinate_space": "pixels_undistorted_native" if pose_mode else (coordinate_space or "declared_input_pixels"),
            "homography_output_space": (
                "pixels_raw_native_distortion_proxy" if pose_mode else (coordinate_space or "declared_input_pixels")
            ),
            "opencv_version": cv2.__version__,
        },
        "pose_update": pose_update if accepted else None,
    }


def run_guarded_line_refinement(
    initial_h: Sequence[Sequence[float]],
    semantic_lines: Mapping[str, object],
    line_distance_map: object,
    keypoint_priors: Mapping[str, object] | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """Compatibility helper returning only the guarded selected homography."""

    return refine_homography_with_lines(
        initial_h,
        semantic_lines,
        line_distance_map,
        keypoint_priors,
        **kwargs,
    )["homography_image_from_court"]


def score_homography_support(
    homography: Sequence[Sequence[float]],
    semantic_lines: Mapping[str, object],
    line_distance_map: object,
    keypoint_priors: Mapping[str, object] | None = None,
    *,
    coordinate_space: str | None = None,
    config: RefinementConfig | None = None,
) -> dict[str, float]:
    """Score a homography on parsed point/line evidence without optimizing it."""

    _ = line_distance_map
    resolved = config or RefinementConfig()
    lines, points, _telemetry = _prepare_evidence(
        semantic_lines, keypoint_priors or {}, calibration=None, coordinate_space=coordinate_space
    )
    score = _score_evidence(
        lines,
        points,
        projection=lambda world: _project_h(_validate_h(homography), world),
        config=resolved,
        heldout=False,
    )
    return {
        "line_rmse_px": float(score["line_rmse_px"]),
        "pixel_support": float(score["pixel_support"]),
        "p95_px": float(score["p95_px"]),
        "median_px": float(score["median_px"]),
    }


def _return_seed(
    seed: np.ndarray,
    reason: str,
    *,
    config: RefinementConfig,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    empty = {"line_rmse_px": 0.0, "pixel_support": 0.0, "p95_px": 0.0, "median_px": 0.0}
    return {
        "accepted": False,
        "homography_image_from_court": seed.tolist(),
        "scores_before": empty,
        "scores_after": empty,
        "reject_reasons": [reason],
        "selection": "seed",
        "selection_reason": reason,
        "covariance_inflation_required": True,
        "guarded": True,
        "objective": {
            "line_weight": config.line_weight,
            "point_weight": config.point_weight,
            "robust_loss": config.robust_loss,
        },
        "robust_initialization": None,
        "telemetry": {"net_top_point_count_in_planar_fit": 0, **dict(telemetry or {})},
        "pose_update": None,
    }


def _prepare_evidence(
    semantic_lines: Mapping[str, object],
    keypoint_priors: Mapping[str, object],
    *,
    calibration: Mapping[str, Any] | None,
    coordinate_space: str | None,
) -> tuple[list[_LineEvidence], list[_PointEvidence], dict[str, Any]]:
    template = get_court_template("pickleball")
    distortion = _distortion_coefficients(calibration)
    has_distortion = bool(distortion is not None and np.any(np.abs(distortion) > 1e-12))
    if has_distortion and coordinate_space not in {"pixels_raw_native", "pixels_undistorted_native"}:
        raise ValueError("coordinate_space_missing_for_distorted_calibration")

    lines: list[_LineEvidence] = []
    for line_id, raw in semantic_lines.items():
        if line_id == "net" or line_id not in template.line_segments_m:
            continue
        optimize_raw, heldout_raw, confidence = _line_payload(raw)
        optimize = tuple(_segment(segment, f"semantic_lines.{line_id}") for segment in optimize_raw)
        heldout = tuple(_segment(segment, f"semantic_lines.{line_id}.heldout") for segment in heldout_raw)
        optimize_hybrid, heldout_hybrid = _hybrid_line_payload(raw, f"semantic_lines.{line_id}")
        if not heldout and not heldout_hybrid:
            heldout = optimize
            heldout_hybrid = optimize_hybrid
        if has_distortion and coordinate_space == "pixels_raw_native":
            optimize = tuple(_undistort_segment(segment, calibration) for segment in optimize)
            heldout = tuple(_undistort_segment(segment, calibration) for segment in heldout)
            optimize_hybrid = tuple(_undistort_hybrid_segment(segment, calibration) for segment in optimize_hybrid)
            heldout_hybrid = tuple(_undistort_hybrid_segment(segment, calibration) for segment in heldout_hybrid)
        lines.append(
            _LineEvidence(
                line_id=line_id,
                world_segment=np.asarray(template.line_segments_m[line_id], dtype=np.float64)[:, :2],
                optimize_segments=optimize,
                heldout_segments=heldout,
                optimize_hybrid_segments=optimize_hybrid,
                heldout_hybrid_segments=heldout_hybrid,
                confidence=confidence,
            )
        )

    points: list[_PointEvidence] = []
    excluded_net = 0
    for name, raw in keypoint_priors.items():
        if name in NET_TOP_KEYPOINT_NAMES:
            excluded_net += 1
            continue
        point = FLOOR_KEYPOINT_BY_NAME.get(name)
        if point is None:
            continue
        image_xy, confidence = _point_payload(raw, f"keypoint_priors.{name}")
        image = np.asarray([image_xy], dtype=np.float64)
        if has_distortion and coordinate_space == "pixels_raw_native":
            image = _undistort_points(image, calibration)
        points.append(
            _PointEvidence(
                name=name,
                world_xy=(float(point.world_xyz_m[0]), float(point.world_xyz_m[1])),
                image_xy=(float(image[0, 0]), float(image[0, 1])),
                confidence=confidence,
                source="keypoint_prior",
            )
        )
    return lines, points, {
        "semantic_line_family_count": len(lines),
        "hybrid_line_segment_count": sum(
            len(line.optimize_hybrid_segments) + len(line.heldout_hybrid_segments) for line in lines
        ),
        "provided_floor_point_count": len(points),
        "excluded_net_top_point_count": excluded_net,
        "distortion_present": has_distortion,
        "input_coordinate_space": coordinate_space,
    }


def _raw_floor_points(keypoint_priors: Mapping[str, object]) -> list[_PointEvidence]:
    """Parse floor priors without conversion for raw-output stability checks."""

    points: list[_PointEvidence] = []
    for name, raw in keypoint_priors.items():
        point = FLOOR_KEYPOINT_BY_NAME.get(name)
        if point is None:
            continue
        image_xy, confidence = _point_payload(raw, f"keypoint_priors.{name}")
        points.append(
            _PointEvidence(
                name=name,
                world_xy=(float(point.world_xyz_m[0]), float(point.world_xyz_m[1])),
                image_xy=image_xy,
                confidence=confidence,
                source="raw_keypoint_prior_for_stability_guard",
            )
        )
    return points


def _line_payload(raw: object) -> tuple[list[object], list[object], float]:
    if isinstance(raw, Mapping):
        optimize = raw.get("optimize", raw.get("segments", raw.get("image_segment")))
        heldout = raw.get("heldout", [])
        confidence = float(raw.get("confidence", 1.0))
    else:
        optimize, heldout, confidence = raw, [], 1.0
    optimize_list = _as_segment_list(optimize)
    heldout_list = _as_segment_list(heldout)
    return optimize_list, heldout_list, max(0.01, min(1.0, confidence))


def _hybrid_line_payload(
    raw: object, field: str
) -> tuple[tuple[_HybridLineSegment, ...], tuple[_HybridLineSegment, ...]]:
    if not isinstance(raw, Mapping):
        return (), ()
    optimize = raw.get("optimize_hybrid", [])
    heldout = raw.get("heldout_hybrid", [])
    return (
        tuple(_coerce_hybrid_segment(value, f"{field}.optimize_hybrid") for value in optimize),
        tuple(_coerce_hybrid_segment(value, f"{field}.heldout_hybrid") for value in heldout),
    )


def _coerce_hybrid_segment(raw: object, field: str) -> _HybridLineSegment:
    endpoints_raw = raw.get("endpoints") if isinstance(raw, Mapping) else getattr(raw, "endpoints", None)
    samples_raw = raw.get("sampled_points") if isinstance(raw, Mapping) else getattr(raw, "sampled_points", None)
    endpoints = _segment(endpoints_raw, f"{field}.endpoints")
    if not isinstance(samples_raw, Sequence) or len(samples_raw) < 2:
        raise ValueError(f"{field}.sampled_points must contain at least two samples")
    samples: list[_HybridLineSample] = []
    for index, sample in enumerate(samples_raw):
        if isinstance(sample, Mapping):
            xy_raw = sample.get("xy")
            covariance_raw = sample.get("normal_covariance_px2")
            provenance = str(sample.get("provenance", "legacy_raw"))
        else:
            xy_raw = getattr(sample, "xy", None)
            covariance_raw = getattr(sample, "normal_covariance_px2", None)
            provenance = str(getattr(sample, "provenance", "legacy_raw"))
        xy = _xy(xy_raw, f"{field}.sampled_points[{index}].xy")
        covariance = np.asarray(covariance_raw, dtype=np.float64)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{field}.sampled_points[{index}].normal_covariance_px2 must be finite 2x2")
        covariance = (covariance + covariance.T) * 0.5
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-9:
            raise ValueError(f"{field}.sampled_points[{index}] covariance must be positive semidefinite")
        if provenance not in {"band_refined", "legacy_raw"}:
            raise ValueError(f"{field}.sampled_points[{index}] has unsupported provenance")
        samples.append(_HybridLineSample(xy=xy, covariance_px2=covariance, provenance=provenance))
    return _HybridLineSegment(endpoints=endpoints, samples=tuple(samples))


def _as_segment_list(value: object) -> list[object]:
    if value is None or value == []:
        return []
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (2, 2):
        return [value]
    if array.ndim == 3 and array.shape[1:] == (2, 2):
        return list(value)  # type: ignore[arg-type]
    raise ValueError("line evidence must be a 2x2 segment or a list of 2x2 segments")


def _point_payload(raw: object, field: str) -> tuple[tuple[float, float], float]:
    if isinstance(raw, Mapping):
        value = raw.get("xy", raw.get("uv", raw.get("image_xy")))
        confidence = float(raw.get("confidence", 1.0))
    else:
        value, confidence = raw, 1.0
    return _xy(value, field), max(0.01, min(1.0, confidence))


def _synthesize_intersection_points(lines: Sequence[_LineEvidence]) -> list[_PointEvidence]:
    result: list[_PointEvidence] = []
    for first, second in combinations(lines, 2):
        if not first.optimize_segments or not second.optimize_segments:
            continue
        world = _line_intersection(first.world_segment, second.world_segment, require_within=True)
        image = _line_intersection(first.optimize_segments[0], second.optimize_segments[0], require_within=False)
        if world is None or image is None:
            continue
        result.append(
            _PointEvidence(
                name=f"intersection:{first.line_id}:{second.line_id}",
                world_xy=(float(world[0]), float(world[1])),
                image_xy=(float(image[0]), float(image[1])),
                confidence=min(first.confidence, second.confidence) * 0.5,
                source="geometry_synthesized_line_intersection",
            )
        )
    return result


def synthesize_hybrid_intersection_points(
    semantic_lines: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Expose covariance-propagated band intersections for adapter tests/tools."""

    lines, _points, _telemetry = _prepare_evidence(
        semantic_lines, {}, calibration=None, coordinate_space="pixels_raw_native"
    )
    return [
        {
            "name": point.name,
            "world_xy": list(point.world_xy),
            "image_xy": list(point.image_xy),
            "covariance_px2": np.asarray(point.covariance_px2).tolist(),
            "source": point.source,
        }
        for point in _synthesize_hybrid_intersection_points(lines)
    ]


def _synthesize_hybrid_intersection_points(lines: Sequence[_LineEvidence]) -> list[_PointEvidence]:
    result: list[_PointEvidence] = []
    for first, second in combinations(lines, 2):
        world = _line_intersection(first.world_segment, second.world_segment, require_within=True)
        if world is None or not first.optimize_hybrid_segments or not second.optimize_hybrid_segments:
            continue
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for first_segment in first.optimize_hybrid_segments:
            first_fit = _fit_band_refined_line(first_segment)
            if first_fit is None:
                continue
            for second_segment in second.optimize_hybrid_segments:
                second_fit = _fit_band_refined_line(second_segment)
                if second_fit is None:
                    continue
                normals = np.vstack([first_fit[0], second_fit[0]])
                if abs(float(np.linalg.det(normals))) <= 1e-4:
                    continue
                inverse = np.linalg.inv(normals)
                image = inverse @ np.asarray([first_fit[1], second_fit[1]], dtype=np.float64)
                covariance = inverse @ np.diag([first_fit[2], second_fit[2]]) @ inverse.T
                if np.all(np.isfinite(image)) and np.all(np.isfinite(covariance)):
                    candidates.append((float(np.trace(covariance)), image, covariance))
        if not candidates:
            continue
        _trace, image, covariance = min(candidates, key=lambda row: row[0])
        result.append(
            _PointEvidence(
                name=f"hybrid_intersection:{first.line_id}:{second.line_id}",
                world_xy=(float(world[0]), float(world[1])),
                image_xy=(float(image[0]), float(image[1])),
                confidence=math.sqrt(first.confidence * second.confidence),
                source="band_refined_intersection",
                covariance_px2=(covariance + covariance.T) * 0.5,
            )
        )
    return result


def _fit_band_refined_line(
    segment: _HybridLineSegment,
) -> tuple[np.ndarray, float, float] | None:
    refined = [sample for sample in segment.samples if sample.provenance == "band_refined"]
    if len(refined) < 2:
        return None
    points = np.asarray([sample.xy for sample in refined], dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    variances = np.asarray(
        [max(float(normal @ sample.covariance_px2 @ normal), 1e-6) for sample in refined],
        dtype=np.float64,
    )
    weights = 1.0 / variances
    centroid = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
    weighted = (points - centroid) * np.sqrt(weights[:, None])
    _u, _s, vh = np.linalg.svd(weighted, full_matrices=False)
    direction = vh[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    variances = np.asarray(
        [max(float(normal @ sample.covariance_px2 @ normal), 1e-6) for sample in refined],
        dtype=np.float64,
    )
    offset_variance = 1.0 / float(np.sum(1.0 / variances))
    offset = float(normal @ centroid)
    return normal, offset, offset_variance


def _residual_vector(
    params: np.ndarray,
    lines: Sequence[_LineEvidence],
    points: Sequence[_PointEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
    heldout: bool,
) -> np.ndarray:
    line_values = _line_residuals(params, lines, projection=projection, config=config, heldout=heldout)
    chunks: list[np.ndarray] = []
    if line_values:
        chunks.append(np.asarray(line_values) * math.sqrt(config.line_weight / len(line_values)))
    if not heldout and config.point_weight > 0.0:
        point_vector = _point_objective_vector(params, points, projection=projection, config=config)
        if point_vector.size:
            chunks.append(point_vector)
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)


def _point_objective_vector(
    params: np.ndarray,
    points: Sequence[_PointEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
) -> np.ndarray:
    provided: list[float] = []
    intersections: list[float] = []
    for point in points:
        projected = projection(params, np.asarray([point.world_xy], dtype=np.float64))[0]
        delta = projected - np.asarray(point.image_xy, dtype=np.float64)
        if point.covariance_px2 is not None:
            covariance = (
                np.asarray(point.covariance_px2, dtype=np.float64)
                + np.eye(2) * config.hybrid_variance_floor_px2
            )
            whitened = np.linalg.solve(np.linalg.cholesky(covariance), delta) * math.sqrt(point.confidence)
            intersections.extend((float(whitened[0]), float(whitened[1])))
        else:
            weighted = delta * math.sqrt(point.confidence)
            provided.extend((float(weighted[0]), float(weighted[1])))
    chunks: list[np.ndarray] = []
    if provided and intersections:
        provided_weight = config.point_weight * (1.0 - config.hybrid_intersection_point_fraction)
        intersection_weight = config.point_weight * config.hybrid_intersection_point_fraction
    elif provided:
        provided_weight, intersection_weight = config.point_weight, 0.0
    else:
        provided_weight, intersection_weight = 0.0, config.point_weight
    if provided and provided_weight > 0.0:
        chunks.append(np.asarray(provided) * math.sqrt(provided_weight / len(provided)))
    if intersections and intersection_weight > 0.0:
        chunks.append(np.asarray(intersections) * math.sqrt(intersection_weight / len(intersections)))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)


def _residual_family_vectors(
    params: np.ndarray,
    lines: Sequence[_LineEvidence],
    points: Sequence[_PointEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
) -> dict[str, np.ndarray]:
    """Return the two explicitly normalized objective families."""

    line_raw = np.asarray(
        _line_residuals(params, lines, projection=projection, config=config, heldout=False),
        dtype=np.float64,
    )
    point_array = _point_objective_vector(params, points, projection=projection, config=config)
    return {
        "line": line_raw * math.sqrt(config.line_weight / len(line_raw)) if line_raw.size else line_raw,
        "point": point_array,
    }


def _finite_difference_jacobian(function: Any, params: np.ndarray) -> np.ndarray:
    baseline = np.asarray(function(params), dtype=np.float64)
    jacobian = np.empty((baseline.size, params.size), dtype=np.float64)
    epsilon = np.cbrt(np.finfo(np.float64).eps)
    for index in range(params.size):
        step = epsilon * max(1.0, abs(float(params[index])))
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=np.float64) - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


def _parameter_observable_scales(params: np.ndarray, *, projection: Any) -> np.ndarray:
    """Parameter deltas that produce one-pixel RMS canonical motion."""

    template = get_court_template("pickleball")
    corners = np.asarray(template.corners_m, dtype=np.float64)[:, :2]
    canonical = np.vstack([corners, np.asarray([[0.0, 0.0], [0.0, -2.1336], [0.0, 2.1336]])])
    jacobian = _finite_difference_jacobian(
        lambda value: projection(value, canonical).reshape(-1), params
    )
    rms_per_parameter = np.sqrt(np.mean(jacobian**2, axis=0))
    return 1.0 / np.maximum(rms_per_parameter, 1e-12)


def _identifiability_model(
    params_seed: np.ndarray,
    *,
    data_residual: Any,
    projection: Any,
    config: RefinementConfig,
) -> dict[str, np.ndarray | float]:
    """Build a one-pixel Fisher floor in observable parameter coordinates.

    One unit in each scaled parameter moves canonical court support by one RMS
    pixel.  Under the declared one-pixel observation model, singular directions
    below unit Fisher information are brought to unit information.  This makes
    the penalty evidence/identifiability-derived rather than score-tuned.
    """

    scales = _parameter_observable_scales(params_seed, projection=projection)
    jacobian = _finite_difference_jacobian(data_residual, params_seed)
    scaled = jacobian @ np.diag(scales)
    _left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    threshold = 1.0 / config.information_noise_px
    precision = np.maximum(0.0, threshold**2 - singular**2)
    return {
        "parameter_scales": scales,
        "singular_values": singular,
        "right_singular_vectors": right,
        "prior_precision": precision,
        "information_floor": float(threshold),
    }


def _regularized_residual_vector(
    params: np.ndarray,
    *,
    data_residual: Any,
    params_seed: np.ndarray,
    identifiability: Mapping[str, Any],
    enabled: bool,
) -> np.ndarray:
    data = np.asarray(data_residual(params), dtype=np.float64)
    if not enabled:
        return data
    scales = np.asarray(identifiability["parameter_scales"], dtype=np.float64)
    right = np.asarray(identifiability["right_singular_vectors"], dtype=np.float64)
    precision = np.asarray(identifiability["prior_precision"], dtype=np.float64)
    normalized_delta = (params - params_seed) / scales
    prior = np.sqrt(precision) * (right @ normalized_delta)
    return np.concatenate([data, prior])


def _posterior_diagnostics(
    params: np.ndarray,
    *,
    data_residual: Any,
    identifiability: Mapping[str, Any],
) -> dict[str, Any]:
    jacobian = _finite_difference_jacobian(data_residual, params)
    scales = np.asarray(identifiability["parameter_scales"], dtype=np.float64)
    scaled = jacobian @ np.diag(scales)
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = math.inf if singular.size == 0 or singular[-1] <= 1e-12 else float(singular[0] / singular[-1])
    raw_singular = np.linalg.svd(jacobian, compute_uv=False)
    raw_condition = (
        math.inf
        if raw_singular.size == 0 or raw_singular[-1] <= 1e-12
        else float(raw_singular[0] / raw_singular[-1])
    )
    covariance_scaled = np.linalg.pinv(scaled.T @ scaled, rcond=1e-12)
    covariance = np.diag(scales) @ covariance_scaled @ np.diag(scales)
    names = (
        ["rvec_x", "rvec_y", "rvec_z", "t_x", "t_y", "t_z"]
        if params.size == 6
        else ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
    )
    return {
        "scaled_condition_number": condition,
        "raw_condition_number": raw_condition,
        "scaled_singular_values": singular.tolist(),
        "raw_singular_values": raw_singular.tolist(),
        "per_parameter_posterior_variance": {
            name: float(covariance[index, index]) for index, name in enumerate(names)
        },
    }


def _json_identifiability(model: Mapping[str, Any]) -> dict[str, Any]:
    precision = np.asarray(model["prior_precision"], dtype=np.float64)
    return {
        "method": "one_pixel_canonical_observable_scaling_and_unit_fisher_floor",
        "information_noise_px": float(1.0 / float(model["information_floor"])),
        "parameter_scales": np.asarray(model["parameter_scales"]).tolist(),
        "seed_scaled_singular_values": np.asarray(model["singular_values"]).tolist(),
        "prior_precision_by_singular_direction": precision.tolist(),
        "regularized_direction_count": int(np.sum(precision > 0.0)),
    }


def _influence_diagnostics(
    params: np.ndarray,
    lines: Sequence[_LineEvidence],
    points: Sequence[_PointEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
) -> dict[str, Any]:
    families = _residual_family_vectors(params, lines, points, projection=projection, config=config)
    costs = {name: float(np.sum(values**2)) for name, values in families.items()}
    gradients: dict[str, float] = {}
    for name, values in families.items():
        if not values.size:
            gradients[name] = 0.0
            continue
        function = lambda value, family=name: _residual_family_vectors(  # noqa: E731
            value, lines, points, projection=projection, config=config
        )[family]
        jacobian = _finite_difference_jacobian(function, params)
        gradients[name] = float(np.linalg.norm(2.0 * jacobian.T @ values))
    total_cost = sum(costs.values())
    total_gradient = sum(gradients.values())
    return {
        "weighted_squared_cost": costs,
        "weighted_cost_fraction": {
            name: value / total_cost if total_cost > 0.0 else 0.0 for name, value in costs.items()
        },
        "gradient_norm": gradients,
        "gradient_norm_fraction": {
            name: value / total_gradient if total_gradient > 0.0 else 0.0 for name, value in gradients.items()
        },
        "point_residual_count": int(families["point"].size),
        "line_residual_count": int(families["line"].size),
    }


_STABILITY_CANONICAL_POINTS = np.asarray(
    [
        [-3.048, -6.7056], [0.0, -6.7056], [3.048, -6.7056],
        [-3.048, 6.7056], [0.0, 6.7056], [3.048, 6.7056],
        [0.0, -2.1336], [0.0, 2.1336], [-3.048, 0.0], [3.048, 0.0],
    ],
    dtype=np.float64,
)


def _observation_bootstrap_stability(
    candidate_h: np.ndarray,
    points: Sequence[_PointEvidence],
    *,
    draws: int,
    sigma_px: float,
    random_seed: int,
) -> dict[str, Any]:
    if len(points) < 4:
        return {"status": "absent", "reason": "fewer_than_four_floor_point_observations", "worst_point_p95_cm": None}
    world = np.asarray([point.world_xy for point in points], dtype=np.float64)
    image = np.asarray([point.image_xy for point in points], dtype=np.float64)
    candidate_pixels = _project_h(candidate_h, _STABILITY_CANONICAL_POINTS)
    rng = np.random.default_rng(random_seed)
    by_point: list[list[float]] = [[] for _ in range(len(_STABILITY_CANONICAL_POINTS))]
    failed = 0
    for _draw in range(draws):
        perturbed = image + rng.normal(0.0, sigma_px, size=image.shape)
        try:
            fitted = np.asarray(homography_from_planar_points(world.tolist(), perturbed.tolist()), dtype=np.float64)
            estimated = _project_h(np.linalg.inv(fitted), candidate_pixels)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            failed += 1
            continue
        delta_cm = np.abs(estimated - _STABILITY_CANONICAL_POINTS) * 100.0
        for index in range(len(by_point)):
            by_point[index].append(float(np.max(delta_cm[index])))
    p95 = [float(np.percentile(values, 95)) if values else math.inf for values in by_point]
    return {
        "status": "present" if failed < draws else "absent",
        "method": "floor_point_observation_perturbation_refit",
        "sigma_px": sigma_px,
        "draws_requested": draws,
        "draws_completed": draws - failed,
        "observation_count": len(points),
        "per_canonical_point_p95_cm": p95,
        "worst_point_index": int(np.argmax(p95)),
        "worst_point_p95_cm": float(max(p95)),
    }


def _stability_ratio(seed: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    before = seed.get("worst_point_p95_cm")
    after = candidate.get("worst_point_p95_cm")
    if before is None or after is None or float(before) <= 0.0:
        return math.inf
    return float(after) / float(before)


def _line_residuals(
    params: Any,
    lines: Sequence[_LineEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
    heldout: bool,
    covariance_weighted: bool = True,
) -> list[float]:
    values: list[float] = []
    indexes = range(config.line_samples_per_segment)
    indexes = [index for index in indexes if (index % config.heldout_stride == 0) == heldout]
    fractions = np.asarray(indexes, dtype=np.float64) / float(config.line_samples_per_segment - 1)
    for line in lines:
        hybrid_segments = line.heldout_hybrid_segments if heldout else line.optimize_hybrid_segments
        if hybrid_segments:
            predicted = projection(params, line.world_segment)
            direction = predicted[1] - predicted[0]
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                values.extend([1e6] * sum(len(segment.samples) for segment in hybrid_segments))
                continue
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
            for segment in hybrid_segments:
                for sample in segment.samples:
                    signed = float((np.asarray(sample.xy) - predicted[0]) @ normal)
                    if covariance_weighted:
                        variance = max(
                            float(normal @ sample.covariance_px2 @ normal),
                            config.hybrid_variance_floor_px2,
                        )
                        signed /= math.sqrt(variance)
                    values.append(signed * math.sqrt(line.confidence))
            continue
        segments = line.heldout_segments if heldout else line.optimize_segments
        if not segments:
            continue
        world = line.world_segment[0] + fractions[:, None] * (line.world_segment[1] - line.world_segment[0])
        projected = projection(params, world)
        for segment in segments:
            signed = _signed_normal_distances(projected, segment)
            values.extend((signed * math.sqrt(line.confidence)).tolist())
    return values


def _score_evidence(
    lines: Sequence[_LineEvidence],
    points: Sequence[_PointEvidence],
    *,
    projection: Any,
    config: RefinementConfig,
    heldout: bool,
) -> dict[str, float]:
    # Adapt a projection(world)->pixels closure to the residual helper's
    # projection(params, world) convention.
    values = _line_residuals(
        np.asarray([]),
        lines,
        projection=lambda _params, world: projection(world),
        config=config,
        heldout=heldout,
        covariance_weighted=False,
    )
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    if absolute.size == 0:
        return {"line_rmse_px": 1e9, "pixel_support": 0.0, "p95_px": 1e9, "median_px": 1e9}
    line_family_p90_px: dict[str, float] = {}
    for line in lines:
        family_values = _line_residuals(
            np.asarray([]),
            [line],
            projection=lambda _params, world: projection(world),
            config=config,
            heldout=heldout,
            covariance_weighted=False,
        )
        if family_values:
            line_family_p90_px[line.line_id] = float(np.percentile(np.abs(family_values), 90))
    return {
        "line_rmse_px": float(np.sqrt(np.mean(absolute**2))),
        "pixel_support": float(np.mean(absolute <= config.evidence_inlier_px)),
        "p95_px": float(np.percentile(absolute, 95)),
        "p90_px": float(np.percentile(absolute, 90)),
        "median_px": float(np.median(absolute)),
        "sample_count": int(absolute.size),
        "line_family_p90_px": line_family_p90_px,
    }


def _heldout_selection_reason(before: Mapping[str, Any], after: Mapping[str, Any], config: RefinementConfig) -> str:
    if after["pixel_support"] < before["pixel_support"] - config.heldout_max_coverage_drop:
        return "seed_wins_heldout_coverage"
    before_families = before.get("line_family_p90_px", {})
    after_families = after.get("line_family_p90_px", {})
    for line_id in sorted(set(before_families).intersection(after_families)):
        if (
            float(after_families[line_id])
            > float(before_families[line_id]) + config.heldout_max_line_family_p90_regression_px
        ):
            return f"seed_wins_heldout_line_family:{line_id}"
    if after["p90_px"] > before["p90_px"] + config.heldout_p90_tolerance_px:
        return "seed_wins_heldout_p90"
    if after["median_px"] >= before["median_px"] - 1e-9 and after["p90_px"] >= before["p90_px"] - 1e-9:
        return "seed_wins_heldout_no_improvement"
    return "refined_wins_heldout"


def _projection_for_h_or_seed_pose(seed: np.ndarray, calibration: Mapping[str, Any] | None) -> Any:
    if _camera_pose_mode(calibration):
        params = _pose_params_from_calibration(calibration)
        return lambda world: _project_pose_undistorted(params, world, calibration)
    return lambda world: _project_h(seed, world)


def _camera_pose_mode(calibration: Mapping[str, Any] | None) -> bool:
    has_camera = bool(
        isinstance(calibration, Mapping)
        and isinstance(calibration.get("intrinsics"), Mapping)
        and isinstance(calibration.get("extrinsics"), Mapping)
    )
    distortion = _distortion_coefficients(calibration)
    return bool(has_camera and distortion is not None and np.any(np.abs(distortion) > 1e-12))


def _pose_params_from_calibration(calibration: Mapping[str, Any] | None) -> np.ndarray:
    assert calibration is not None
    extrinsics = calibration["extrinsics"]
    rotation = np.asarray(extrinsics["R"], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.asarray(extrinsics["t"], dtype=np.float64).reshape(3)
    return np.concatenate([rvec.reshape(3), translation])


def _pose_params_from_homography(
    homography: np.ndarray, calibration: Mapping[str, Any] | None
) -> np.ndarray:
    """Turn a robust planar initializer into a proper pose parameter seed."""

    assert calibration is not None
    intrinsics = calibration["intrinsics"]
    camera = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = np.linalg.inv(camera) @ homography
    scale = 2.0 / (np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1]))
    first = normalized[:, 0] * scale
    second = normalized[:, 1] * scale
    translation = normalized[:, 2] * scale
    if translation[2] < 0.0:
        first, second, translation = -first, -second, -translation
    raw_rotation = np.column_stack([first, second, np.cross(first, second)])
    left, _singular, right = np.linalg.svd(raw_rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    rvec, _ = cv2.Rodrigues(rotation)
    return np.concatenate([rvec.reshape(3), translation])


def _project_pose_undistorted(params: Sequence[float], world_xy: np.ndarray, calibration: Mapping[str, Any] | None) -> np.ndarray:
    assert calibration is not None
    intrinsics = calibration["intrinsics"]
    camera = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    world = np.column_stack([world_xy, np.zeros(len(world_xy), dtype=np.float64)])
    projected, _ = cv2.projectPoints(world, np.asarray(params[:3]), np.asarray(params[3:6]), camera, None)
    return projected.reshape(-1, 2)


def _homography_from_pose_params(params: Sequence[float], calibration: Mapping[str, Any] | None) -> np.ndarray:
    assert calibration is not None
    intrinsics = calibration["intrinsics"]
    camera = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64))
    return _normalize_h(camera @ np.column_stack([rotation[:, 0], rotation[:, 1], np.asarray(params[3:6])]))


def _raw_homography_proxy_from_pose(
    params: Sequence[float], calibration: Mapping[str, Any] | None
) -> np.ndarray:
    """Approximate a distorted plane projection in the raw homography field."""

    assert calibration is not None
    intrinsics = calibration["intrinsics"]
    camera = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    world_xy = np.vstack(
        [
            _STABILITY_CANONICAL_POINTS,
            np.asarray(get_court_template("pickleball").corners_m, dtype=np.float64)[:, :2],
        ]
    )
    world_xyz = np.column_stack([world_xy, np.zeros(len(world_xy), dtype=np.float64)])
    projected, _ = cv2.projectPoints(
        world_xyz,
        np.asarray(params[:3], dtype=np.float64),
        np.asarray(params[3:6], dtype=np.float64),
        camera,
        _distortion_coefficients(calibration),
    )
    return _validate_h(homography_from_planar_points(world_xy.tolist(), projected.reshape(-1, 2).tolist()))


def _pose_update_payload(params: Sequence[float]) -> dict[str, Any]:
    rotation, _ = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64))
    return {"R": rotation.tolist(), "t": np.asarray(params[3:6], dtype=np.float64).tolist()}


def _distortion_coefficients(calibration: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(calibration, Mapping) or not isinstance(calibration.get("intrinsics"), Mapping):
        return None
    return np.asarray(calibration["intrinsics"].get("dist", []), dtype=np.float64)


def _undistort_points(points: np.ndarray, calibration: Mapping[str, Any] | None) -> np.ndarray:
    assert calibration is not None
    intrinsics = calibration["intrinsics"]
    camera = np.asarray(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cv2.undistortPoints(
        points.reshape(-1, 1, 2), camera, _distortion_coefficients(calibration), P=camera
    ).reshape(-1, 2)


def _undistort_segment(segment: np.ndarray, calibration: Mapping[str, Any] | None) -> np.ndarray:
    return _undistort_points(segment, calibration)


def _undistort_hybrid_segment(
    segment: _HybridLineSegment, calibration: Mapping[str, Any] | None
) -> _HybridLineSegment:
    endpoints = _undistort_segment(segment.endpoints, calibration)
    samples: list[_HybridLineSample] = []
    for sample in segment.samples:
        raw = np.asarray(sample.xy, dtype=np.float64)
        undistorted = _undistort_points(raw[None, :], calibration)[0]
        step = 0.01
        basis = np.asarray([[step, 0.0], [0.0, step]], dtype=np.float64)
        plus = _undistort_points(raw[None, :] + basis, calibration)
        minus = _undistort_points(raw[None, :] - basis, calibration)
        jacobian = ((plus - minus) / (2.0 * step)).T
        covariance = jacobian @ sample.covariance_px2 @ jacobian.T
        samples.append(
            _HybridLineSample(
                xy=(float(undistorted[0]), float(undistorted[1])),
                covariance_px2=(covariance + covariance.T) * 0.5,
                provenance=sample.provenance,
            )
        )
    return _HybridLineSegment(endpoints=endpoints, samples=tuple(samples))


def _project_h(homography: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([world_xy, np.ones(len(world_xy), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-9):
        return np.full((len(world_xy), 2), 1e6, dtype=np.float64)
    return projected[:, :2] / denominator[:, None]


def _signed_normal_distances(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    direction = segment[1] - segment[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return np.full(len(points), 1e6, dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
    return (points - segment[0]) @ normal


def _line_intersection(first: np.ndarray, second: np.ndarray, *, require_within: bool) -> np.ndarray | None:
    p, r = first[0], first[1] - first[0]
    q, s = second[0], second[1] - second[0]
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= 1e-8:
        return None
    delta = q - p
    t = float((delta[0] * s[1] - delta[1] * s[0]) / cross)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / cross)
    if require_within and not (-1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6):
        return None
    return p + t * r


def _canonical_corner_shift(first: np.ndarray, second: np.ndarray) -> float:
    corners = np.asarray(get_court_template("pickleball").corners_m, dtype=np.float64)[:, :2]
    a = _project_h(first, corners)
    b = _project_h(second, corners)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.inf
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def _jacobian_condition_number(jacobian: np.ndarray) -> float:
    if jacobian.size == 0:
        return math.inf
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular.size == 0 or singular[-1] <= 1e-12:
        return math.inf
    return float(singular[0] / singular[-1])


def _pack_h(homography: np.ndarray) -> np.ndarray:
    h = _normalize_h(homography)
    return np.asarray([h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]])


def _unpack_h(params: Sequence[float]) -> np.ndarray:
    return _normalize_h(np.asarray([[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]], dtype=np.float64))


def _normalize_h(homography: np.ndarray) -> np.ndarray:
    if abs(float(homography[2, 2])) <= 1e-12:
        return homography
    return homography / float(homography[2, 2])


def _validate_h(homography: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(homography, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)) or abs(float(np.linalg.det(value))) <= 1e-12:
        raise ValueError("initial_h must be a finite nonsingular 3x3 matrix")
    return _normalize_h(value)


def _segment(value: object, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must be a finite 2x2 segment")
    if np.linalg.norm(result[1] - result[0]) <= 1e-9:
        raise ValueError(f"{field} segment must have positive length")
    return result


def _xy(value: object, field: str) -> tuple[float, float]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain two finite numbers") from exc
    if len(result) != 2 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain two finite numbers")
    return result[0], result[1]


def _max_triangle_area(points: Sequence[Sequence[float]]) -> float:
    best = 0.0
    for first, second, third in combinations(points, 3):
        area = abs(
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        ) / 2.0
        best = max(best, float(area))
    return best
