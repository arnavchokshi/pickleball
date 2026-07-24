"""Confidence-aware, floor-only best-effort pickleball-court solver.

This module deliberately does not produce calibration authority.  It searches a
bounded set of regulation-court homographies, robustly chooses one coherent
floor solution, and projects every canonical floor point from that *single*
homography.  The result is useful for review and downstream candidate ranking,
but always remains ``measurement_valid=false`` and ``review_only``.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations, product
import math
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.court_keypoint_net import ALL_PICKLEBALL_KEYPOINTS, PICKLEBALL_KEYPOINTS
from threed.racketsport.court_structured_evidence import CourtEvidenceBundle
from threed.racketsport.court_camera_geometry import (
    PinholeIntrinsics,
    distort_pixels_radial_k1,
)
from threed.racketsport.court_distortion_fit import refine_planar_homography_and_k1


FLOOR_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    point.name for point in PICKLEBALL_KEYPOINTS if abs(float(point.world_xyz_m[2])) <= 1.0e-12
)
NET_TOP_KEYPOINT_NAMES: frozenset[str] = frozenset(
    point.name for point in PICKLEBALL_KEYPOINTS if abs(float(point.world_xyz_m[2])) > 1.0e-12
)
FLOOR_WORLD_XY_M: dict[str, tuple[float, float]] = {
    point.name: (float(point.world_xyz_m[0]), float(point.world_xyz_m[1]))
    for point in PICKLEBALL_KEYPOINTS
    if abs(float(point.world_xyz_m[2])) <= 1.0e-12
}
EVIDENCE_WORLD_XY_M: dict[str, tuple[float, float]] = {
    point.name: (float(point.world_xyz_m[0]), float(point.world_xyz_m[1]))
    for point in ALL_PICKLEBALL_KEYPOINTS
    if abs(float(point.world_xyz_m[2])) <= 1.0e-12
}

_OUTER_CORNERS = (
    "near_left_corner",
    "near_right_corner",
    "far_right_corner",
    "far_left_corner",
)
_TRANSVERSE_TRIPLES = (
    ("near_left_corner", "near_baseline_center", "near_right_corner"),
    ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
    ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ("far_left_corner", "far_baseline_center", "far_right_corner"),
)
_SIDELINE_SEQUENCES = (
    ("near_left_corner", "near_nvz_left", "far_nvz_left", "far_left_corner"),
    ("near_right_corner", "near_nvz_right", "far_nvz_right", "far_right_corner"),
)
SEMANTIC_FLOOR_SEGMENTS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_nvz_center", "far_baseline_center"),
}


@dataclass(frozen=True)
class _Observation:
    semantic: str
    candidate_id: str
    xy: tuple[float, float]
    confidence: float
    visibility: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    sigma_px: float
    quality: float
    input_index: int
    frame_indices: tuple[int, ...]
    line_support: float
    temporal_support: float


@dataclass(frozen=True)
class _ScoredModel:
    homography: np.ndarray
    source: str
    score: float
    score_components: dict[str, float | int]
    evaluations: tuple[dict[str, Any], ...]
    inliers: tuple[dict[str, Any], ...]
    residual_stats: dict[str, float | int | None]
    seed_semantics: tuple[str, ...]


@dataclass(frozen=True)
class _DenseEvidence:
    image_size: tuple[int, int]
    line_distance_maps: Mapping[str, np.ndarray]
    surface_probability: np.ndarray | None
    temporal_support: float | None


def _dense_evidence_from_bundle(bundle: CourtEvidenceBundle | None) -> _DenseEvidence | None:
    if bundle is None:
        return None
    return _DenseEvidence(
        image_size=bundle.image_size,
        line_distance_maps={
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in bundle.line_distance_maps.items()
        },
        surface_probability=(
            None
            if bundle.surface_probability is None
            else np.asarray(bundle.surface_probability, dtype=np.float64)
        ),
        temporal_support=bundle.temporal_support,
    )


def solve_best_floor_court(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | CourtEvidenceBundle,
    *,
    prior_homography: Sequence[Sequence[float]] | None = None,
    max_hypotheses: int = 256,
    shortlist_size: int = 16,
    refine_size: int = 4,
    inlier_threshold_px: float = 10.0,
    duplicate_tolerance_px: float = 2.0,
) -> dict[str, Any]:
    """Return one robust regulation-court projection from uncertain floor points.

    ``observations`` may be a mapping from semantic name to one candidate or a
    candidate list, or a flat sequence whose items contain ``semantic``/``name``.
    A candidate accepts ``xy`` (or ``image_xy``), ``confidence``, ``visibility``
    and covariance as a scalar, diagonal pair, or 2x2 matrix.  At most the two
    strongest valid candidates per semantic participate in the solve.
    """

    if isinstance(max_hypotheses, bool) or not isinstance(max_hypotheses, int) or max_hypotheses <= 0:
        raise ValueError("max_hypotheses must be a positive integer")
    if isinstance(shortlist_size, bool) or not isinstance(shortlist_size, int) or shortlist_size <= 0:
        raise ValueError("shortlist_size must be a positive integer")
    if isinstance(refine_size, bool) or not isinstance(refine_size, int) or refine_size <= 0:
        raise ValueError("refine_size must be a positive integer")
    if not math.isfinite(float(inlier_threshold_px)) or float(inlier_threshold_px) <= 0.0:
        raise ValueError("inlier_threshold_px must be positive and finite")
    if not math.isfinite(float(duplicate_tolerance_px)) or float(duplicate_tolerance_px) < 0.0:
        raise ValueError("duplicate_tolerance_px must be non-negative and finite")

    bundle = observations if isinstance(observations, CourtEvidenceBundle) else None
    raw_observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] = (
        list(bundle.observations) if bundle is not None else observations
    )
    dense_evidence = _dense_evidence_from_bundle(bundle)
    if prior_homography is None and bundle is not None:
        for candidate in bundle.homography_candidates:
            if str(candidate.get("source", "")).startswith("previous_static_lock"):
                prior_homography = candidate.get("homography")
                break
    candidates, initially_ignored, observation_counts = _normalize_observations(raw_observations)
    hypothesis_specs, hypothesis_diagnostics = _prioritized_hypotheses(
        candidates,
        max_hypotheses=max_hypotheses,
    )

    scored: list[_ScoredModel] = []
    for priority, semantics, selected in hypothesis_specs:
        del priority
        try:
            homography = _fit_homography(
                [EVIDENCE_WORLD_XY_M[name] for name in semantics],
                [candidate.xy for candidate in selected],
                [max(candidate.quality / (candidate.sigma_px**2), 1.0e-6) for candidate in selected],
            )
        except ValueError:
            continue
        scored.append(
            _score_model(
                homography,
                candidates,
                source="observation_hypothesis",
                seed_semantics=semantics,
                inlier_threshold_px=float(inlier_threshold_px),
                duplicate_tolerance_px=float(duplicate_tolerance_px),
                dense_evidence=dense_evidence,
            )
        )

    if bundle is not None:
        for raw_candidate in bundle.homography_candidates:
            try:
                candidate_homography = _validated_homography(raw_candidate.get("homography"))
            except (TypeError, ValueError):
                continue
            scored.append(
                _score_model(
                    candidate_homography,
                    candidates,
                    source=str(raw_candidate.get("source") or "line_only_candidate"),
                    seed_semantics=(),
                    inlier_threshold_px=float(inlier_threshold_px),
                    duplicate_tolerance_px=float(duplicate_tolerance_px),
                    dense_evidence=dense_evidence,
                )
            )

    prior_error: str | None = None
    if prior_homography is not None:
        try:
            prior = _validated_homography(prior_homography)
        except ValueError as exc:
            prior_error = str(exc)
        else:
            scored.append(
                _score_model(
                    prior,
                    candidates,
                    source="prior_homography",
                    seed_semantics=(),
                    inlier_threshold_px=float(inlier_threshold_px),
                    duplicate_tolerance_px=float(duplicate_tolerance_px),
                    dense_evidence=dense_evidence,
                )
            )

    if not scored:
        ignored = list(initially_ignored)
        for semantic_candidates in candidates.values():
            for candidate in semantic_candidates:
                ignored.append(_ignored_record(candidate, "no_valid_hypothesis"))
        return _empty_result(
            ignored=ignored,
            observation_counts=observation_counts,
            hypothesis_diagnostics=hypothesis_diagnostics,
            prior_error=prior_error,
        )

    scored.sort(key=_model_sort_key)
    shortlisted = scored[: min(shortlist_size, len(scored))]
    refined: list[_ScoredModel] = []
    for model in shortlisted[: min(refine_size, len(shortlisted))]:
        point_refined = _refine_model(
            model,
            candidates,
            inlier_threshold_px=float(inlier_threshold_px),
            duplicate_tolerance_px=float(duplicate_tolerance_px),
            dense_evidence=dense_evidence,
        )
        refined.append(
            _refine_dense_model(
                point_refined,
                candidates,
                inlier_threshold_px=float(inlier_threshold_px),
                duplicate_tolerance_px=float(duplicate_tolerance_px),
                dense_evidence=dense_evidence,
            )
        )
    finalists = scored + refined
    finalists.sort(key=_model_sort_key)
    best = finalists[0]
    alternate = next(
        (
            model
            for model in finalists[1:]
            if not np.allclose(model.homography, best.homography, atol=1.0e-9, rtol=1.0e-9)
        ),
        None,
    )
    margin = None if alternate is None else max(0.0, float(best.score - alternate.score))

    distortion_fit = _camera_distortion_refinement(best, candidates, bundle)
    if distortion_fit is None:
        projected = _project_floor(best.homography)
        diagnostic_homography = best.homography
        diagnostic_projected = projected
    else:
        diagnostic_homography = np.asarray(
            distortion_fit["homography_undistorted_from_court"], dtype=np.float64
        )
        diagnostic_projected = _project_floor(diagnostic_homography)
        intrinsics = distortion_fit.pop("_intrinsics")
        undistorted_points = np.asarray(
            [diagnostic_projected[name] for name in FLOOR_KEYPOINT_NAMES],
            dtype=np.float64,
        )
        distorted_points = distort_pixels_radial_k1(
            undistorted_points,
            intrinsics,
            k1=float(distortion_fit["k1"]),
        )
        projected = {
            name: (float(distorted_points[index, 0]), float(distorted_points[index, 1]))
            for index, name in enumerate(FLOOR_KEYPOINT_NAMES)
        }
    inlier_by_semantic = {str(item["semantic"]): item for item in best.inliers}
    court_confidence = _court_confidence(
        best,
        margin=margin,
        has_alternate=alternate is not None,
    )
    transform_covariance, projected_covariance = _estimate_transform_covariance(best, candidates)
    point_confidence: dict[str, float] = {}
    projected_points: dict[str, dict[str, Any]] = {}
    for name in FLOOR_KEYPOINT_NAMES:
        inlier = inlier_by_semantic.get(name)
        if inlier is not None:
            confidence = math.sqrt(
                max(0.0, court_confidence) * max(0.0, float(inlier["effective_confidence"]))
            )
        elif name in candidates:
            confidence = court_confidence * 0.50
        else:
            confidence = court_confidence * 0.35
        confidence = float(min(max(confidence, 0.0), 0.99))
        if distortion_fit is not None:
            confidence *= math.exp(-math.sqrt(float(distortion_fit["k1_variance"])) / 0.15)
        point_confidence[name] = confidence
        projected_points[name] = {
            "xy": [float(value) for value in projected[name]],
            "confidence": confidence,
            "source": "single_regulation_floor_homography",
            "covariance_px2": projected_covariance.get(name),
        }

    evaluation_by_id = {
        (str(row["semantic"]), str(row["candidate_id"])): row for row in best.evaluations
    }
    inlier_ids = {
        (str(row["semantic"]), str(row["candidate_id"])) for row in best.inliers
    }
    ignored = list(initially_ignored)
    for semantic_candidates in candidates.values():
        for candidate in semantic_candidates:
            key = (candidate.semantic, candidate.candidate_id)
            if key in inlier_ids:
                continue
            evaluation = evaluation_by_id.get(key)
            reason = (
                str(evaluation["ignore_reason"])
                if evaluation is not None and evaluation.get("ignore_reason")
                else "alternate_candidate_not_selected"
            )
            ignored.append(_ignored_record(candidate, reason, evaluation=evaluation))

    diagnostics = _geometry_diagnostics(diagnostic_homography, diagnostic_projected)
    diagnostics["distortion_refinement"] = (
        {key: value for key, value in distortion_fit.items() if not key.startswith("_")}
        if distortion_fit is not None
        else {"status": "not_run_no_explicit_intrinsics"}
    )
    diagnostics["hypothesis_search"] = {
        **hypothesis_diagnostics,
        "valid_homographies_scored": len(scored),
        "shortlist_size": len(shortlisted),
        "refined_models_scored": len(refined),
        "refine_size": min(refine_size, len(shortlisted)),
    }
    if prior_error is not None:
        diagnostics["prior_homography"] = {"accepted": False, "reason": prior_error}
    elif prior_homography is not None:
        diagnostics["prior_homography"] = {"accepted": True, "reason": None}

    return {
        "schema_version": 1,
        "solver": "confidence_aware_floor_only_robust_homography_v1",
        "status": (
            "prior_only_best_effort"
            if best.source == "prior_homography" and not best.inliers
            else (
                "line_only_best_effort"
                if not best.inliers and float(best.score_components.get("line_alignment", 0.0)) > 0.0
                else "solved_best_effort"
            )
        ),
        "measurement_valid": False,
        "authority_state": "review_only",
        "solution_role": "best_effort",
        "verified": False,
        "floor_only": True,
        "excluded_semantics": sorted(NET_TOP_KEYPOINT_NAMES),
        "homography_image_from_court": best.homography.tolist(),
        "camera_parameters": (
            None
            if distortion_fit is None
            else {
                "intrinsics": distortion_fit["intrinsics"],
                "intrinsics_source": distortion_fit["intrinsics_source"],
                "homography_undistorted_from_court": distortion_fit[
                    "homography_undistorted_from_court"
                ],
            }
        ),
        "distortion": (
            {"model": "not_estimated", "k1": None, "source": "not_available"}
            if distortion_fit is None
            else {
                "model": "radial_k1",
                "k1": distortion_fit["k1"],
                "k1_variance": distortion_fit["k1_variance"],
                "bounds": distortion_fit["bounds"],
                "source": "joint_point_camera_refinement",
                "uncertainty_status": distortion_fit["uncertainty_status"],
            }
        ),
        "transform_covariance": transform_covariance,
        "projected_floor_keypoints": projected_points,
        "point_confidence": point_confidence,
        "court_confidence": court_confidence,
        "margin": margin,
        "inliers": list(best.inliers),
        "ignored_observations": ignored,
        "residual_stats_px": best.residual_stats,
        "score_components": {
            **best.score_components,
            "total": float(best.score),
        },
        "selected_hypothesis": {
            "source": best.source,
            "seed_semantics": list(best.seed_semantics),
        },
        "observation_counts": observation_counts,
        "diagnostics": diagnostics,
    }


def solve_structured_court(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Stable descriptive alias for :func:`solve_best_floor_court`."""

    return solve_best_floor_court(observations, **kwargs)


def _normalize_observations(
    raw: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[_Observation, ...]], list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[Any]] = {}
    if isinstance(raw, Mapping):
        for semantic, value in raw.items():
            grouped[str(semantic)] = _candidate_values(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if not isinstance(item, Mapping):
                grouped.setdefault("<invalid>", []).append(item)
                continue
            semantic = item.get("semantic", item.get("name"))
            grouped.setdefault(str(semantic) if semantic is not None else "<missing>", []).append(item)
    else:
        raise ValueError("observations must be a semantic mapping or a candidate sequence")

    kept: dict[str, tuple[_Observation, ...]] = {}
    ignored: list[dict[str, Any]] = []
    input_count = 0
    valid_count = 0
    for semantic in sorted(grouped):
        values = grouped[semantic]
        input_count += len(values)
        if semantic in NET_TOP_KEYPOINT_NAMES:
            for index, value in enumerate(values):
                ignored.append(
                    _raw_ignored_record(
                        semantic,
                        value,
                        index,
                        "net_top_excluded_floor_only_solver",
                    )
                )
            continue
        if semantic not in EVIDENCE_WORLD_XY_M:
            for index, value in enumerate(values):
                ignored.append(_raw_ignored_record(semantic, value, index, "unknown_semantic"))
            continue
        parsed: list[_Observation] = []
        for index, value in enumerate(values):
            try:
                candidate = _parse_observation(semantic, value, index)
            except ValueError as exc:
                ignored.append(_raw_ignored_record(semantic, value, index, f"invalid_candidate:{exc}"))
                continue
            if candidate.visibility <= 0.0:
                ignored.append(_ignored_record(candidate, "visibility_zero"))
                continue
            if candidate.confidence <= 0.0:
                ignored.append(_ignored_record(candidate, "nonpositive_confidence"))
                continue
            parsed.append(candidate)
            valid_count += 1
        parsed.sort(
            key=lambda item: (
                -item.quality,
                -item.confidence,
                item.sigma_px,
                item.candidate_id,
            )
        )
        for candidate in parsed[2:]:
            ignored.append(_ignored_record(candidate, "below_top2_confidence"))
        if parsed[:2]:
            kept[semantic] = tuple(parsed[:2])

    return kept, ignored, {
        "input": input_count,
        "valid_before_top2": valid_count,
        "retained_top2": sum(len(value) for value in kept.values()),
        "semantic_count": len(kept),
    }


def _candidate_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 2 and all(_is_finite_number(item) for item in value):
            return [{"xy": value, "confidence": 1.0}]
        return list(value)
    return [value]


def _parse_observation(semantic: str, value: Any, index: int) -> _Observation:
    if isinstance(value, Mapping):
        xy_raw = value.get("xy", value.get("image_xy"))
        confidence_raw = value.get("confidence")
        visibility_raw = value.get("visibility", 1.0)
        covariance_raw = value.get("covariance", value.get("covariance_px2", 4.0))
        candidate_id = str(value.get("candidate_id", value.get("id", f"{semantic}:{index}")))
        line_support_raw = value.get("line_support", 0.0)
        temporal_support_raw = value.get("temporal_support", 0.0)
        frames_raw = value.get("contributing_frame_indices")
        if frames_raw is None:
            frame = value.get("frame", value.get("frame_index"))
            frames_raw = [] if frame is None else [frame]
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        xy_raw = value
        confidence_raw = 1.0
        visibility_raw = 1.0
        covariance_raw = 4.0
        candidate_id = f"{semantic}:{index}"
        frames_raw = []
        line_support_raw = 0.0
        temporal_support_raw = 0.0
    else:
        raise ValueError("candidate must be a mapping or xy pair")
    if (
        not isinstance(xy_raw, Sequence)
        or isinstance(xy_raw, (str, bytes, bytearray))
        or len(xy_raw) != 2
        or not all(_is_finite_number(item) for item in xy_raw)
    ):
        raise ValueError("xy must contain two finite numbers")
    if not _is_finite_number(confidence_raw):
        raise ValueError("confidence must be finite")
    if isinstance(visibility_raw, bool):
        visibility = 1.0 if visibility_raw else 0.0
    elif _is_finite_number(visibility_raw):
        visibility = float(visibility_raw)
    else:
        raise ValueError("visibility must be boolean or finite")
    confidence = min(max(float(confidence_raw), 0.0), 1.0)
    visibility = min(max(visibility, 0.0), 1.0)
    covariance = _parse_covariance(covariance_raw)
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    if float(eigenvalues.min()) < -1.0e-9:
        raise ValueError("covariance must be positive semidefinite")
    sigma_px = math.sqrt(max(float(np.trace(covariance)) * 0.5, 0.25))
    covariance_reliability = 1.0 / (1.0 + sigma_px / 12.0)
    if not _is_finite_number(line_support_raw) or not _is_finite_number(temporal_support_raw):
        raise ValueError("line and temporal support must be finite")
    line_support = min(max(float(line_support_raw), 0.0), 1.0)
    temporal_support = min(max(float(temporal_support_raw), 0.0), 1.0)
    evidence_factor = 0.75 + 0.15 * line_support + 0.10 * temporal_support
    quality = confidence * visibility * covariance_reliability * evidence_factor
    if (
        not isinstance(frames_raw, Sequence)
        or isinstance(frames_raw, (str, bytes, bytearray))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in frames_raw)
    ):
        raise ValueError("frame indices must be non-negative integers")
    frame_indices = tuple(sorted(set(int(value) for value in frames_raw)))
    return _Observation(
        semantic=semantic,
        candidate_id=candidate_id,
        xy=(float(xy_raw[0]), float(xy_raw[1])),
        confidence=confidence,
        visibility=visibility,
        covariance=covariance,
        sigma_px=sigma_px,
        quality=quality,
        input_index=index,
        frame_indices=frame_indices,
        line_support=line_support,
        temporal_support=temporal_support,
    )


def _parse_covariance(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if _is_finite_number(value):
        variance = max(float(value), 0.0)
        return ((variance, 0.0), (0.0, variance))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("covariance must be scalar, diagonal pair, or 2x2 matrix")
    if len(value) == 2 and all(_is_finite_number(item) for item in value):
        return ((max(float(value[0]), 0.0), 0.0), (0.0, max(float(value[1]), 0.0)))
    if (
        len(value) == 2
        and all(isinstance(row, Sequence) and len(row) == 2 for row in value)
        and all(_is_finite_number(item) for row in value for item in row)
    ):
        matrix = np.asarray(value, dtype=np.float64)
        if not np.allclose(matrix, matrix.T, atol=1.0e-9, rtol=1.0e-9):
            raise ValueError("covariance matrix must be symmetric")
        return (
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        )
    raise ValueError("covariance must be scalar, diagonal pair, or 2x2 matrix")


def _prioritized_hypotheses(
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    max_hypotheses: int,
) -> tuple[
    list[tuple[float, tuple[str, ...], tuple[_Observation, ...]]],
    dict[str, int],
]:
    # A full materialization is combinatorial: 30 semantics with two peaks can
    # exceed 400k candidate tuples even though the solver consumes at most 256.
    # Use a deterministic best-first merge instead.  Each semantic group enters
    # the heap with an admissible upper bound; candidate products are expanded
    # only when that group can still contribute to the global top-k.  This is
    # the confidence-prioritized, progressively widened behavior intended by
    # the PROSAC-style contract.
    specs: list[tuple[float, tuple[str, ...], tuple[_Observation, ...]]] = []
    world_degenerate = 0
    image_degenerate = 0
    semantic_group_count = 0
    candidate_products_expanded = 0
    heap: list[tuple[float, int, tuple[str, ...], tuple[str, ...], int]] = []
    local_specs: dict[
        tuple[str, ...],
        list[tuple[float, tuple[_Observation, ...]]],
    ] = {}

    for semantics in combinations(sorted(candidates), 4):
        world_points = [EVIDENCE_WORLD_XY_M[name] for name in semantics]
        if not _nondegenerate_four_points(world_points):
            world_degenerate += 1
            continue
        semantic_group_count += 1
        upper_bound = sum(
            max(_hypothesis_candidate_priority(candidate) for candidate in candidates[name])
            for name in semantics
        )
        heapq.heappush(heap, (-upper_bound, 0, semantics, (), 0))

    while heap and len(specs) < max_hypotheses:
        negative_priority, kind, semantics, candidate_ids, index = heapq.heappop(heap)
        if kind == 0:
            expanded: list[tuple[float, tuple[_Observation, ...]]] = []
            for selected in product(*(candidates[name] for name in semantics)):
                candidate_products_expanded += 1
                if not _nondegenerate_four_points([candidate.xy for candidate in selected]):
                    image_degenerate += 1
                    continue
                priority = sum(_hypothesis_candidate_priority(candidate) for candidate in selected)
                expanded.append((priority, tuple(selected)))
            expanded.sort(
                key=lambda item: (
                    -item[0],
                    tuple(candidate.candidate_id for candidate in item[1]),
                )
            )
            if not expanded:
                continue
            local_specs[semantics] = expanded
            priority, selected = expanded[0]
            heapq.heappush(
                heap,
                (
                    -priority,
                    1,
                    semantics,
                    tuple(candidate.candidate_id for candidate in selected),
                    0,
                ),
            )
            continue

        expanded = local_specs[semantics]
        priority, selected = expanded[index]
        specs.append((priority, semantics, selected))
        next_index = index + 1
        if next_index < len(expanded):
            next_priority, next_selected = expanded[next_index]
            heapq.heappush(
                heap,
                (
                    -next_priority,
                    1,
                    semantics,
                    tuple(candidate.candidate_id for candidate in next_selected),
                    next_index,
                ),
            )

    return specs, {
        "candidate_hypotheses_nondegenerate": len(specs),
        "hypotheses_retained_cap": len(specs),
        "hypothesis_cap": max_hypotheses,
        "world_degenerate_groups_ignored": world_degenerate,
        "image_degenerate_hypotheses_ignored": image_degenerate,
        "semantic_groups_queued": semantic_group_count,
        "candidate_products_expanded": candidate_products_expanded,
    }


def _hypothesis_candidate_priority(candidate: _Observation) -> float:
    return math.log(max(candidate.quality, 1.0e-6)) - 0.05 * math.log1p(candidate.sigma_px)


def _nondegenerate_four_points(points: Sequence[Sequence[float]]) -> bool:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        return False
    extent = np.ptp(array, axis=0)
    scale = max(float(extent[0]), float(extent[1]), 1.0)
    if min(
        abs(_orientation(array[a], array[b], array[c]))
        for a, b, c in combinations(range(4), 3)
    ) <= 1.0e-6 * scale * scale:
        return False
    return True


def _fit_homography(
    world_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> np.ndarray:
    world = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if world.shape != image.shape or world.ndim != 2 or world.shape[1] != 2 or len(world) < 4:
        raise ValueError("homography fit requires at least four paired 2D points")
    if weight.shape != (len(world),) or not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("homography weights must be positive finite values")
    world_n, world_transform = _normalize_points(world)
    image_n, image_transform = _normalize_points(image)
    rows: list[list[float]] = []
    for (x, y), (u, v), point_weight in zip(world_n, image_n, weight, strict=True):
        root_weight = math.sqrt(float(point_weight))
        rows.append(
            [
                -x * root_weight,
                -y * root_weight,
                -root_weight,
                0.0,
                0.0,
                0.0,
                u * x * root_weight,
                u * y * root_weight,
                u * root_weight,
            ]
        )
        rows.append(
            [
                0.0,
                0.0,
                0.0,
                -x * root_weight,
                -y * root_weight,
                -root_weight,
                v * x * root_weight,
                v * y * root_weight,
                v * root_weight,
            ]
        )
    design = np.asarray(rows, dtype=np.float64)
    _u, singular, vh = np.linalg.svd(design, full_matrices=True)
    if len(singular) < 8 or singular[0] <= 0.0 or singular[7] / singular[0] <= 1.0e-12:
        raise ValueError("homography design is rank deficient")
    normalized = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(image_transform) @ normalized @ world_transform
    return _validated_homography(homography)


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    mean_distance = float(np.mean(distances))
    if not math.isfinite(mean_distance) or mean_distance <= 1.0e-12:
        raise ValueError("point normalization is degenerate")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def _validated_homography(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    homography = np.asarray(value, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    norm = float(homography[2, 2])
    if abs(norm) > 1.0e-12:
        homography = homography / norm
    else:
        matrix_norm = float(np.linalg.norm(homography))
        if matrix_norm <= 1.0e-12:
            raise ValueError("homography has zero norm")
        homography = homography / matrix_norm
    determinant = float(np.linalg.det(homography))
    condition = float(np.linalg.cond(homography))
    if (
        not math.isfinite(determinant)
        or abs(determinant) <= 1.0e-12
        or not math.isfinite(condition)
        or condition > 1.0e14
    ):
        raise ValueError("homography is non-invertible or ill-conditioned")
    _project_xy(homography, list(EVIDENCE_WORLD_XY_M.values()))
    return homography


def _score_model(
    homography: np.ndarray,
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    source: str,
    seed_semantics: tuple[str, ...],
    inlier_threshold_px: float,
    duplicate_tolerance_px: float,
    dense_evidence: _DenseEvidence | None,
) -> _ScoredModel:
    projected = _project_evidence(homography)
    evaluations: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for semantic in sorted(candidates):
        rows: list[dict[str, Any]] = []
        expected = projected[semantic]
        for candidate in candidates[semantic]:
            residual = math.dist(expected, candidate.xy)
            threshold = max(inlier_threshold_px, 2.5 * candidate.sigma_px)
            delta = np.asarray(expected, dtype=np.float64) - np.asarray(candidate.xy, dtype=np.float64)
            covariance = np.asarray(candidate.covariance, dtype=np.float64) + np.eye(2) * 1.0e-6
            mahalanobis = float(delta.T @ np.linalg.pinv(covariance) @ delta)
            # A bounded covariance likelihood ranks precise evidence, while the
            # explicit uniform outlier component prevents one overconfident bad
            # observation from driving the posterior to numerical zero.
            gaussian_likelihood = math.exp(-0.5 * min(mahalanobis, 80.0))
            mixture_likelihood = 0.95 * gaussian_likelihood + 0.05 * 1.0e-3
            support = candidate.quality * gaussian_likelihood
            row = {
                "semantic": semantic,
                "candidate_id": candidate.candidate_id,
                "xy": [float(candidate.xy[0]), float(candidate.xy[1])],
                "residual_px": float(residual),
                "threshold_px": float(threshold),
                "effective_confidence": float(candidate.quality),
                "frame_indices": list(candidate.frame_indices),
                "line_support": float(candidate.line_support),
                "temporal_support": float(candidate.temporal_support),
                "support": float(support),
                "mahalanobis_squared": mahalanobis,
                "outlier_mixture_likelihood": mixture_likelihood,
                "preliminary_inlier": bool(residual <= threshold),
                "selected_for_semantic": False,
                "ignore_reason": None,
            }
            rows.append(row)
        rows.sort(
            key=lambda row: (
                -float(row["support"]),
                float(row["residual_px"]),
                str(row["candidate_id"]),
            )
        )
        rows[0]["selected_for_semantic"] = True
        selected.append(rows[0])
        for row in rows[1:]:
            row["ignore_reason"] = "alternate_candidate_not_selected"
        evaluations.extend(rows)

    accepted: list[dict[str, Any]] = []
    duplicate_penalty = 0.0
    for row in sorted(
        selected,
        key=lambda item: (
            -float(item["support"]),
            -float(item["effective_confidence"]),
            str(item["semantic"]),
        ),
    ):
        if not row["preliminary_inlier"]:
            row["ignore_reason"] = "residual_outlier"
            continue
        duplicate = next(
            (
                prior
                for prior in accepted
                if math.dist(row["xy"], prior["xy"]) <= duplicate_tolerance_px
            ),
            None,
        )
        if duplicate is not None:
            row["ignore_reason"] = "duplicate_image_location"
            duplicate_penalty += float(row["effective_confidence"])
            continue
        accepted.append(row)

    residuals = [float(row["residual_px"]) for row in accepted]
    weighted_consensus = sum(float(row["support"]) for row in accepted)
    confidence_support = sum(float(row["effective_confidence"]) for row in accepted)
    outlier_log_likelihood = sum(
        float(row["effective_confidence"])
        * math.log(max(float(row["outlier_mixture_likelihood"]), 1.0e-12))
        for row in selected
    ) / max(len(selected), 1)
    # Four independent correspondences are the minimum for a point-derived
    # planar solve. A three-point result backed only by a prior must therefore
    # retain visibly lower confidence even if all three available observations
    # agree with that prior.
    coverage = len(accepted) / max(len(candidates), 4)
    if accepted:
        residual_penalty = sum(
            float(row["effective_confidence"])
            * min((float(row["residual_px"]) / float(row["threshold_px"])) ** 2, 4.0)
            for row in accepted
        ) / max(confidence_support, 1.0e-9)
    else:
        residual_penalty = 4.0
    dense_scores = _score_dense_evidence(homography, dense_evidence)
    score = (
        4.0 * weighted_consensus
        + 8.0 * coverage
        + 1.5 * confidence_support
        + 3.0 * float(dense_scores["line_alignment"])
        + 2.0 * float(dense_scores["surface_overlap"])
        + 1.0 * float(dense_scores["temporal_support"])
        + 0.5 * outlier_log_likelihood
        - 2.0 * residual_penalty
        - 2.5 * duplicate_penalty
    )
    return _ScoredModel(
        homography=homography,
        source=source,
        score=float(score),
        score_components={
            "weighted_consensus": float(weighted_consensus),
            "semantic_coverage": float(coverage),
            "inlier_count": len(accepted),
            "confidence_support": float(confidence_support),
            "residual_penalty": float(residual_penalty),
            "duplicate_penalty": float(duplicate_penalty),
            "outlier_mixture_log_likelihood": float(outlier_log_likelihood),
            **dense_scores,
        },
        evaluations=tuple(evaluations),
        inliers=tuple(
            {
                "semantic": str(row["semantic"]),
                "candidate_id": str(row["candidate_id"]),
                "xy": list(row["xy"]),
                "residual_px": float(row["residual_px"]),
                "effective_confidence": float(row["effective_confidence"]),
                "frame_indices": list(row.get("frame_indices") or []),
            }
            for row in sorted(accepted, key=lambda item: str(item["semantic"]))
        ),
        residual_stats=_residual_stats(residuals),
        seed_semantics=seed_semantics,
    )


def _score_dense_evidence(
    homography: np.ndarray,
    evidence: _DenseEvidence | None,
) -> dict[str, float]:
    if evidence is None:
        return {
            "line_alignment": 0.0,
            "surface_overlap": 0.0,
            "temporal_support": 0.0,
            "line_mean_distance_px": 0.0,
            "line_visible_fraction": 0.0,
            "surface_visible_fraction": 0.0,
        }

    line_distances: list[float] = []
    requested_line_samples = 0
    for segment_name, (start_name, end_name) in SEMANTIC_FLOOR_SEGMENTS.items():
        distance_map = evidence.line_distance_maps.get(segment_name)
        if distance_map is None:
            distance_map = evidence.line_distance_maps.get("pickleball_line")
        if distance_map is None:
            distance_map = evidence.line_distance_maps.get("all_pickleball_lines")
        if distance_map is None:
            continue
        start = np.asarray(FLOOR_WORLD_XY_M[start_name], dtype=np.float64)
        end = np.asarray(FLOOR_WORLD_XY_M[end_name], dtype=np.float64)
        world_samples = [
            tuple((start * (1.0 - alpha) + end * alpha).tolist())
            for alpha in np.linspace(0.0, 1.0, 33)
        ]
        try:
            projected = _project_xy(homography, world_samples)
        except ValueError:
            continue
        requested_line_samples += len(projected)
        line_distances.extend(_sample_image(distance_map, projected))
    if line_distances:
        mean_line_distance = float(np.mean(line_distances))
        visible_line_fraction = len(line_distances) / max(requested_line_samples, 1)
        line_alignment = math.exp(-mean_line_distance / 8.0) * math.sqrt(visible_line_fraction)
    else:
        mean_line_distance = 0.0
        visible_line_fraction = 0.0
        line_alignment = 0.0

    surface_overlap = 0.0
    surface_visible_fraction = 0.0
    if evidence.surface_probability is not None:
        xs = np.linspace(-3.048, 3.048, 13)
        ys = np.linspace(-6.7056, 6.7056, 25)
        world_grid = [(float(x), float(y)) for y in ys for x in xs]
        try:
            projected_grid = _project_xy(homography, world_grid)
        except ValueError:
            projected_grid = []
        surface_values = _sample_image(evidence.surface_probability, projected_grid)
        surface_visible_fraction = len(surface_values) / len(projected_grid)
        if surface_values:
            coverage_factor = min(1.0, surface_visible_fraction / 0.25)
            surface_overlap = float(np.mean(surface_values)) * coverage_factor

    return {
        "line_alignment": float(min(max(line_alignment, 0.0), 1.0)),
        "surface_overlap": float(min(max(surface_overlap, 0.0), 1.0)),
        "temporal_support": float(evidence.temporal_support or 0.0),
        "line_mean_distance_px": mean_line_distance,
        "line_visible_fraction": float(visible_line_fraction),
        "surface_visible_fraction": float(surface_visible_fraction),
    }


def _sample_image(image: np.ndarray, points: Sequence[Sequence[float]]) -> list[float]:
    height, width = image.shape
    if len(points) == 0:
        return []
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("sample points must be an Nx2 array")
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    if not bool(np.any(keep)):
        return []
    x = x[keep]
    y = y[keep]
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    values = (
        (1.0 - dx) * (1.0 - dy) * image[y0, x0]
        + dx * (1.0 - dy) * image[y0, x1]
        + (1.0 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    )
    return [float(value) for value in values]


def _refine_model(
    model: _ScoredModel,
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    inlier_threshold_px: float,
    duplicate_tolerance_px: float,
    dense_evidence: _DenseEvidence | None,
) -> _ScoredModel:
    current = model
    candidate_lookup = {
        (candidate.semantic, candidate.candidate_id): candidate
        for semantic_candidates in candidates.values()
        for candidate in semantic_candidates
    }
    for _iteration in range(2):
        observations = [
            candidate_lookup[(str(item["semantic"]), str(item["candidate_id"]))]
            for item in current.inliers
        ]
        if len(observations) < 4:
            break
        world = [EVIDENCE_WORLD_XY_M[item.semantic] for item in observations]
        image = [item.xy for item in observations]
        if not _point_set_supports_homography(world) or not _point_set_supports_homography(image):
            break
        try:
            homography = _fit_homography(
                world,
                image,
                [max(item.quality / (item.sigma_px**2), 1.0e-6) for item in observations],
            )
        except ValueError:
            break
        candidate_model = _score_model(
            homography,
            candidates,
            source=f"{model.source}_refined",
            seed_semantics=model.seed_semantics,
            inlier_threshold_px=inlier_threshold_px,
            duplicate_tolerance_px=duplicate_tolerance_px,
            dense_evidence=dense_evidence,
        )
        if _model_sort_key(candidate_model) < _model_sort_key(current):
            current = candidate_model
        else:
            break
    return current


def _refine_dense_model(
    model: _ScoredModel,
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    inlier_threshold_px: float,
    duplicate_tolerance_px: float,
    dense_evidence: _DenseEvidence | None,
) -> _ScoredModel:
    if dense_evidence is None or (
        not dense_evidence.line_distance_maps and dense_evidence.surface_probability is None
    ):
        return model
    current = model
    base = np.asarray(
        [
            model.homography[0, 0],
            model.homography[0, 1],
            model.homography[0, 2],
            model.homography[1, 0],
            model.homography[1, 1],
            model.homography[1, 2],
            model.homography[2, 0],
            model.homography[2, 1],
        ],
        dtype=np.float64,
    )
    initial_steps = np.asarray(
        [
            max(abs(base[0]) * 0.01, 0.1),
            max(abs(base[1]) * 0.01, 0.1),
            2.0,
            max(abs(base[3]) * 0.01, 0.1),
            max(abs(base[4]) * 0.01, 0.1),
            2.0,
            max(abs(base[6]) * 0.05, 1.0e-5),
            max(abs(base[7]) * 0.05, 1.0e-5),
        ],
        dtype=np.float64,
    )
    for scale in (1.0, 0.5, 0.25):
        improved = True
        while improved:
            improved = False
            current_parameters = np.asarray(
                [
                    current.homography[0, 0],
                    current.homography[0, 1],
                    current.homography[0, 2],
                    current.homography[1, 0],
                    current.homography[1, 1],
                    current.homography[1, 2],
                    current.homography[2, 0],
                    current.homography[2, 1],
                ],
                dtype=np.float64,
            )
            for index, step in enumerate(initial_steps * scale):
                for direction in (-1.0, 1.0):
                    proposal = current_parameters.copy()
                    proposal[index] += direction * step
                    homography = np.asarray(
                        [
                            proposal[0:3],
                            proposal[3:6],
                            [proposal[6], proposal[7], 1.0],
                        ],
                        dtype=np.float64,
                    )
                    try:
                        homography = _validated_homography(homography)
                    except ValueError:
                        continue
                    candidate_model = _score_model(
                        homography,
                        candidates,
                        source=f"{model.source}_dense_refined",
                        seed_semantics=model.seed_semantics,
                        inlier_threshold_px=inlier_threshold_px,
                        duplicate_tolerance_px=duplicate_tolerance_px,
                        dense_evidence=dense_evidence,
                    )
                    if _model_sort_key(candidate_model) < _model_sort_key(current):
                        current = candidate_model
                        improved = True
            # One complete coordinate pass per scale keeps runtime bounded and deterministic.
            break
    return current


def _point_set_supports_homography(points: Sequence[Sequence[float]]) -> bool:
    if len(points) < 4:
        return False
    for indices in combinations(range(len(points)), 4):
        if _nondegenerate_four_points([points[index] for index in indices]):
            return True
    return False


def _model_sort_key(model: _ScoredModel) -> tuple[Any, ...]:
    homography_signature = tuple(float(value) for value in np.round(model.homography, 12).ravel())
    return (
        -model.score,
        -len(model.inliers),
        float(model.residual_stats.get("median") or math.inf),
        model.source,
        model.seed_semantics,
        homography_signature,
    )


def _project_xy(homography: np.ndarray, points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    source = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    denominators = projected[:, 2]
    if not np.isfinite(projected).all() or np.any(np.abs(denominators) <= 1.0e-10):
        raise ValueError("homography projects a floor point to infinity")
    xy = projected[:, :2] / denominators[:, None]
    if not np.isfinite(xy).all():
        raise ValueError("homography projection is non-finite")
    return [(float(point[0]), float(point[1])) for point in xy]


def _project_floor(homography: np.ndarray) -> dict[str, tuple[float, float]]:
    projected = _project_xy(homography, [FLOOR_WORLD_XY_M[name] for name in FLOOR_KEYPOINT_NAMES])
    return dict(zip(FLOOR_KEYPOINT_NAMES, projected, strict=True))


def _project_evidence(homography: np.ndarray) -> dict[str, tuple[float, float]]:
    names = tuple(EVIDENCE_WORLD_XY_M)
    projected = _project_xy(homography, [EVIDENCE_WORLD_XY_M[name] for name in names])
    return dict(zip(names, projected, strict=True))


def _homography_projection_jacobian(
    homography: np.ndarray,
    world_xy: tuple[float, float],
) -> np.ndarray:
    x, y = world_xy
    denominator = homography[2, 0] * x + homography[2, 1] * y + 1.0
    if abs(float(denominator)) <= 1.0e-10:
        raise ValueError("homography covariance projection is singular")
    u, v = _project_xy(homography, [world_xy])[0]
    inverse_denominator = 1.0 / denominator
    return np.asarray(
        [
            [
                x * inverse_denominator,
                y * inverse_denominator,
                inverse_denominator,
                0.0,
                0.0,
                0.0,
                -u * x * inverse_denominator,
                -u * y * inverse_denominator,
            ],
            [
                0.0,
                0.0,
                0.0,
                x * inverse_denominator,
                y * inverse_denominator,
                inverse_denominator,
                -v * x * inverse_denominator,
                -v * y * inverse_denominator,
            ],
        ],
        dtype=np.float64,
    )


def _estimate_transform_covariance(
    model: _ScoredModel,
    candidates: Mapping[str, tuple[_Observation, ...]],
) -> tuple[list[list[float]] | None, dict[str, list[list[float]]]]:
    if len(model.inliers) < 4:
        return None, {}
    lookup = {
        (candidate.semantic, candidate.candidate_id): candidate
        for semantic_candidates in candidates.values()
        for candidate in semantic_candidates
    }
    information = np.zeros((8, 8), dtype=np.float64)
    squared_residuals: list[float] = []
    for row in model.inliers:
        key = (str(row["semantic"]), str(row["candidate_id"]))
        candidate = lookup.get(key)
        if candidate is None:
            continue
        jacobian = _homography_projection_jacobian(
            model.homography,
            EVIDENCE_WORLD_XY_M[candidate.semantic],
        )
        covariance = np.asarray(candidate.covariance, dtype=np.float64) + np.eye(2) * 1.0e-6
        information += jacobian.T @ np.linalg.pinv(covariance) @ jacobian
        squared_residuals.append(float(row["residual_px"]) ** 2)
    if np.linalg.matrix_rank(information, tol=1.0e-10) < 8:
        return None, {}
    residual_variance = max(float(np.mean(squared_residuals or [1.0])), 1.0)
    covariance_h = np.linalg.pinv(information) * residual_variance
    if not np.isfinite(covariance_h).all():
        return None, {}
    # ``pinv`` of a theoretically symmetric information matrix can differ
    # from its transpose by more than the strict court-lock tolerance once
    # homography parameter scales become large.  Normalize the estimate back
    # onto the symmetric PSD cone before it crosses the public artifact
    # boundary; this changes only floating-point noise, not the fitted court.
    covariance_h = 0.5 * (covariance_h + covariance_h.T)
    values_h, vectors_h = np.linalg.eigh(covariance_h)
    covariance_h = vectors_h @ np.diag(np.maximum(values_h, 0.0)) @ vectors_h.T
    covariance_h = 0.5 * (covariance_h + covariance_h.T)
    point_covariance: dict[str, list[list[float]]] = {}
    for name in FLOOR_KEYPOINT_NAMES:
        jacobian = _homography_projection_jacobian(model.homography, FLOOR_WORLD_XY_M[name])
        covariance_xy = jacobian @ covariance_h @ jacobian.T
        covariance_xy = 0.5 * (covariance_xy + covariance_xy.T)
        values, vectors = np.linalg.eigh(covariance_xy)
        covariance_xy = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        point_covariance[name] = covariance_xy.tolist()
    return covariance_h.tolist(), point_covariance


def _camera_distortion_refinement(
    model: _ScoredModel,
    candidates: Mapping[str, tuple[_Observation, ...]],
    bundle: CourtEvidenceBundle | None,
) -> dict[str, Any] | None:
    if bundle is None or len(model.inliers) < 4:
        return None
    metadata = bundle.camera_metadata
    if not isinstance(metadata, Mapping):
        return None
    intrinsics_payload = metadata.get("intrinsics")
    if not isinstance(intrinsics_payload, Mapping):
        return None
    if not all(key in intrinsics_payload for key in ("fx", "fy", "cx", "cy")):
        return None
    try:
        intrinsics = PinholeIntrinsics(
            fx=float(intrinsics_payload["fx"]),
            fy=float(intrinsics_payload["fy"]),
            cx=float(intrinsics_payload["cx"]),
            cy=float(intrinsics_payload["cy"]),
        )
    except (TypeError, ValueError):
        return None
    distortion_payload = metadata.get("distortion")
    if not isinstance(distortion_payload, Mapping):
        distortion_payload = {}
    bounds_raw = distortion_payload.get("k1_bounds", (-0.45, 0.25))
    if (
        not isinstance(bounds_raw, Sequence)
        or isinstance(bounds_raw, (str, bytes))
        or len(bounds_raw) != 2
    ):
        return None
    lookup = {
        (candidate.semantic, candidate.candidate_id): candidate
        for semantic_candidates in candidates.values()
        for candidate in semantic_candidates
    }
    selected = [
        lookup.get((str(row["semantic"]), str(row["candidate_id"])))
        for row in model.inliers
    ]
    observations = [candidate for candidate in selected if candidate is not None]
    if len(observations) < 4:
        return None
    try:
        result = refine_planar_homography_and_k1(
            [EVIDENCE_WORLD_XY_M[candidate.semantic] for candidate in observations],
            [candidate.xy for candidate in observations],
            intrinsics,
            weights=[candidate.quality for candidate in observations],
            k1_bounds=(float(bounds_raw[0]), float(bounds_raw[1])),
            k1_initial=float(distortion_payload.get("k1", 0.0)),
            grid_steps=int(distortion_payload.get("grid_steps", 41)),
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    result["intrinsics"] = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
    }
    result["intrinsics_source"] = str(
        intrinsics_payload.get("source") or metadata.get("source") or "explicit_camera_metadata"
    )
    result["_intrinsics"] = intrinsics
    result["status"] = "jointly_refined"
    return result


def _court_confidence(
    model: _ScoredModel,
    *,
    margin: float | None,
    has_alternate: bool,
) -> float:
    inlier_count = len(model.inliers)
    if inlier_count == 0:
        line = float(model.score_components.get("line_alignment", 0.0))
        surface = float(model.score_components.get("surface_overlap", 0.0))
        if line > 0.0 or surface > 0.0:
            return float(min(0.35, 0.25 * line + 0.10 * surface))
        return 0.05 if model.source.startswith("prior_homography") else 0.0
    coverage = float(model.score_components.get("semantic_coverage", 0.0))
    support = float(model.score_components["weighted_consensus"]) / inlier_count
    median = float(model.residual_stats.get("median") or 0.0)
    residual_factor = math.exp(-median / 10.0)
    if has_alternate:
        margin_factor = 0.55 + 0.45 * (1.0 - math.exp(-max(float(margin or 0.0), 0.0) / 4.0))
    else:
        margin_factor = 0.55
    confidence = coverage * min(max(support, 0.0), 1.0) * residual_factor * margin_factor
    return float(min(max(confidence, 0.0), 0.99))


def _residual_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _geometry_diagnostics(
    homography: np.ndarray,
    projected: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    world = [FLOOR_WORLD_XY_M[name] for name in FLOOR_KEYPOINT_NAMES]
    homogeneous = np.column_stack([np.asarray(world), np.ones(len(world), dtype=np.float64)])
    denominators = (homography @ homogeneous.T).T[:, 2]
    projective = {
        "finite": bool(np.isfinite(homography).all()),
        "determinant": float(np.linalg.det(homography)),
        "condition_number": float(np.linalg.cond(homography)),
        "minimum_floor_denominator_abs": float(np.min(np.abs(denominators))),
        "invertible": bool(abs(float(np.linalg.det(homography))) > 1.0e-12),
    }

    order_checks: list[dict[str, Any]] = []
    for left, center, right in _TRANSVERSE_TRIPLES:
        a, b, c = projected[left], projected[center], projected[right]
        gap = abs((math.dist(a, b) + math.dist(b, c)) - math.dist(a, c))
        order_checks.append(
            {
                "semantics": [left, center, right],
                "middle_between_endpoints": bool(gap <= max(1.0e-6, 1.0e-6 * math.dist(a, c))),
                "collinearity_error_px": float(
                    abs(_orientation(a, b, c)) / max(math.dist(a, c), 1.0e-9)
                ),
            }
        )
    for sequence in _SIDELINE_SEQUENCES:
        points = [projected[name] for name in sequence]
        direction = np.asarray(points[-1]) - np.asarray(points[0])
        denominator = float(np.dot(direction, direction))
        parameters = (
            [0.0 for _ in points]
            if denominator <= 1.0e-12
            else [
                float(np.dot(np.asarray(point) - np.asarray(points[0]), direction) / denominator)
                for point in points
            ]
        )
        order_checks.append(
            {
                "semantics": list(sequence),
                "projective_parameter_order_preserved": bool(
                    all(parameters[index] < parameters[index + 1] for index in range(len(parameters) - 1))
                ),
                "projective_parameters": parameters,
            }
        )

    corners = [projected[name] for name in _OUTER_CORNERS]
    turns = [
        _orientation(corners[index], corners[(index + 1) % 4], corners[(index + 2) % 4])
        for index in range(4)
    ]
    signed_area_twice = sum(
        corners[index][0] * corners[(index + 1) % 4][1]
        - corners[(index + 1) % 4][0] * corners[index][1]
        for index in range(4)
    )
    convexity = {
        "outer_court_convex": bool(all(value > 0.0 for value in turns) or all(value < 0.0 for value in turns)),
        "outer_court_signed_area_px2": float(0.5 * signed_area_twice),
        "self_intersecting": bool(
            _segments_properly_intersect(corners[0], corners[1], corners[2], corners[3])
            or _segments_properly_intersect(corners[1], corners[2], corners[3], corners[0])
        ),
    }

    diagonal_intersection = _line_intersection(corners[0], corners[2], corners[1], corners[3])
    projected_center = _project_xy(homography, [(0.0, 0.0)])[0]
    diagonal_center = {
        "diagonals_intersect_finitely": diagonal_intersection is not None,
        "projected_regulation_center_xy": [float(value) for value in projected_center],
        "diagonal_intersection_xy": (
            None
            if diagonal_intersection is None
            else [float(value) for value in diagonal_intersection]
        ),
        "diagonal_center_residual_px": (
            None
            if diagonal_intersection is None
            else float(math.dist(diagonal_intersection, projected_center))
        ),
    }
    return {
        "projective": projective,
        "order": {
            "checks": order_checks,
            "all_passed": all(
                bool(
                    row.get(
                        "middle_between_endpoints",
                        row.get("projective_parameter_order_preserved", False),
                    )
                )
                for row in order_checks
            ),
        },
        "convexity": convexity,
        "diagonal_center": diagonal_center,
        "diagnostic_note": (
            "Projective incidence/order/convexity diagnostics only; no image-angle or "
            "image-length equality assumptions are used."
        ),
    }


def _empty_result(
    *,
    ignored: list[dict[str, Any]],
    observation_counts: dict[str, int],
    hypothesis_diagnostics: dict[str, int],
    prior_error: str | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "hypothesis_search": {
            **hypothesis_diagnostics,
            "valid_homographies_scored": 0,
            "shortlist_size": 0,
            "refined_models_scored": 0,
        },
        "diagnostic_note": (
            "No finite regulation homography existed; output remains review-only and contains "
            "no independently selected points."
        ),
    }
    if prior_error is not None:
        diagnostics["prior_homography"] = {"accepted": False, "reason": prior_error}
    return {
        "schema_version": 1,
        "solver": "confidence_aware_floor_only_robust_homography_v1",
        "status": "insufficient_floor_hypothesis",
        "measurement_valid": False,
        "authority_state": "review_only",
        "solution_role": "best_effort",
        "verified": False,
        "floor_only": True,
        "excluded_semantics": sorted(NET_TOP_KEYPOINT_NAMES),
        "homography_image_from_court": None,
        "transform_covariance": None,
        "projected_floor_keypoints": {},
        "point_confidence": {},
        "court_confidence": 0.0,
        "margin": None,
        "inliers": [],
        "ignored_observations": ignored,
        "residual_stats_px": _residual_stats([]),
        "score_components": {
            "weighted_consensus": 0.0,
            "semantic_coverage": 0.0,
            "inlier_count": 0,
            "confidence_support": 0.0,
            "residual_penalty": 0.0,
            "duplicate_penalty": 0.0,
            "outlier_mixture_log_likelihood": 0.0,
            "line_alignment": 0.0,
            "surface_overlap": 0.0,
            "temporal_support": 0.0,
            "line_mean_distance_px": 0.0,
            "line_visible_fraction": 0.0,
            "surface_visible_fraction": 0.0,
            "total": 0.0,
        },
        "selected_hypothesis": None,
        "observation_counts": observation_counts,
        "diagnostics": diagnostics,
    }


def _ignored_record(
    candidate: _Observation,
    reason: str,
    *,
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "semantic": candidate.semantic,
        "candidate_id": candidate.candidate_id,
        "xy": [float(candidate.xy[0]), float(candidate.xy[1])],
        "reason": reason,
        "confidence": float(candidate.confidence),
        "visibility": float(candidate.visibility),
        "effective_confidence": float(candidate.quality),
        "frame_indices": list(candidate.frame_indices),
        "frame": candidate.frame_indices[0] if len(candidate.frame_indices) == 1 else None,
        "line_support": float(candidate.line_support),
        "temporal_support": float(candidate.temporal_support),
    }
    if evaluation is not None and evaluation.get("residual_px") is not None:
        record["residual_px"] = float(evaluation["residual_px"])
    return record


def _raw_ignored_record(
    semantic: str,
    value: Any,
    index: int,
    reason: str,
) -> dict[str, Any]:
    candidate_id = (
        str(value.get("candidate_id", value.get("id", f"{semantic}:{index}")))
        if isinstance(value, Mapping)
        else f"{semantic}:{index}"
    )
    frame = (
        value.get("frame", value.get("frame_index"))
        if isinstance(value, Mapping)
        else None
    )
    return {
        "semantic": semantic,
        "candidate_id": candidate_id,
        "reason": reason,
        "frame": frame if isinstance(frame, int) and not isinstance(frame, bool) else None,
    }


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_properly_intersect(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (ab_c > 0.0 > ab_d or ab_d > 0.0 > ab_c) and (
        cd_a > 0.0 > cd_b or cd_b > 0.0 > cd_a
    )


def _line_intersection(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> tuple[float, float] | None:
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    x3, y3 = float(c[0]), float(c[1])
    x4, y4 = float(d[0]), float(d[1])
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1.0e-12:
        return None
    determinant_ab = x1 * y2 - y1 * x2
    determinant_cd = x3 * y4 - y3 * x4
    return (
        (determinant_ab * (x3 - x4) - (x1 - x2) * determinant_cd) / denominator,
        (determinant_ab * (y3 - y4) - (y1 - y2) * determinant_cd) / denominator,
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


__all__ = [
    "EVIDENCE_WORLD_XY_M",
    "FLOOR_KEYPOINT_NAMES",
    "FLOOR_WORLD_XY_M",
    "NET_TOP_KEYPOINT_NAMES",
    "SEMANTIC_FLOOR_SEGMENTS",
    "solve_best_floor_court",
    "solve_structured_court",
]
