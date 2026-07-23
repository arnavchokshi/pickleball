"""Confidence-aware, floor-only best-effort pickleball-court solver.

This module deliberately does not produce calibration authority.  It searches a
bounded set of regulation-court homographies, robustly chooses one coherent
floor solution, and projects every canonical floor point from that *single*
homography.  The result is useful for review and downstream candidate ranking,
but always remains ``measurement_valid=false`` and ``review_only``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


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


def solve_best_floor_court(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    prior_homography: Sequence[Sequence[float]] | None = None,
    max_hypotheses: int = 256,
    shortlist_size: int = 8,
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
    if not math.isfinite(float(inlier_threshold_px)) or float(inlier_threshold_px) <= 0.0:
        raise ValueError("inlier_threshold_px must be positive and finite")
    if not math.isfinite(float(duplicate_tolerance_px)) or float(duplicate_tolerance_px) < 0.0:
        raise ValueError("duplicate_tolerance_px must be non-negative and finite")

    candidates, initially_ignored, observation_counts = _normalize_observations(observations)
    hypothesis_specs, hypothesis_diagnostics = _prioritized_hypotheses(
        candidates,
        max_hypotheses=max_hypotheses,
    )

    scored: list[_ScoredModel] = []
    for priority, semantics, selected in hypothesis_specs:
        del priority
        try:
            homography = _fit_homography(
                [FLOOR_WORLD_XY_M[name] for name in semantics],
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
    for model in shortlisted:
        refined.append(
            _refine_model(
                model,
                candidates,
                inlier_threshold_px=float(inlier_threshold_px),
                duplicate_tolerance_px=float(duplicate_tolerance_px),
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

    projected = _project_floor(best.homography)
    inlier_by_semantic = {str(item["semantic"]): item for item in best.inliers}
    court_confidence = _court_confidence(
        best,
        margin=margin,
        has_alternate=alternate is not None,
    )
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
        point_confidence[name] = confidence
        projected_points[name] = {
            "xy": [float(value) for value in projected[name]],
            "confidence": confidence,
            "source": "single_regulation_floor_homography",
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

    diagnostics = _geometry_diagnostics(best.homography, projected)
    diagnostics["hypothesis_search"] = {
        **hypothesis_diagnostics,
        "valid_homographies_scored": len(scored),
        "shortlist_size": len(shortlisted),
        "refined_models_scored": len(refined),
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
            else "solved_best_effort"
        ),
        "measurement_valid": False,
        "authority_state": "review_only",
        "solution_role": "best_effort",
        "verified": False,
        "floor_only": True,
        "excluded_semantics": sorted(NET_TOP_KEYPOINT_NAMES),
        "homography_image_from_court": best.homography.tolist(),
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
        if semantic not in FLOOR_WORLD_XY_M:
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
    quality = confidence * visibility * covariance_reliability
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
    specs: list[tuple[float, tuple[str, ...], tuple[_Observation, ...]]] = []
    world_degenerate = 0
    image_degenerate = 0
    for semantics in combinations(sorted(candidates), 4):
        world_points = [FLOOR_WORLD_XY_M[name] for name in semantics]
        if not _nondegenerate_four_points(world_points):
            world_degenerate += 1
            continue
        for selected in product(*(candidates[name] for name in semantics)):
            if not _nondegenerate_four_points([candidate.xy for candidate in selected]):
                image_degenerate += 1
                continue
            quality = sum(math.log(max(candidate.quality, 1.0e-6)) for candidate in selected)
            covariance = sum(math.log1p(candidate.sigma_px) for candidate in selected)
            priority = quality - 0.05 * covariance
            specs.append((priority, semantics, tuple(selected)))
    specs.sort(
        key=lambda item: (
            -item[0],
            item[1],
            tuple(candidate.candidate_id for candidate in item[2]),
        )
    )
    retained = specs[:max_hypotheses]
    return retained, {
        "candidate_hypotheses_nondegenerate": len(specs),
        "hypotheses_retained_cap": len(retained),
        "hypothesis_cap": max_hypotheses,
        "world_degenerate_groups_ignored": world_degenerate,
        "image_degenerate_hypotheses_ignored": image_degenerate,
    }


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
    _project_xy(homography, list(FLOOR_WORLD_XY_M.values()))
    return homography


def _score_model(
    homography: np.ndarray,
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    source: str,
    seed_semantics: tuple[str, ...],
    inlier_threshold_px: float,
    duplicate_tolerance_px: float,
) -> _ScoredModel:
    projected = _project_floor(homography)
    evaluations: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for semantic in sorted(candidates):
        rows: list[dict[str, Any]] = []
        expected = projected[semantic]
        for candidate in candidates[semantic]:
            residual = math.dist(expected, candidate.xy)
            threshold = max(inlier_threshold_px, 2.5 * candidate.sigma_px)
            ratio = residual / threshold
            support = candidate.quality * math.exp(-0.5 * ratio * ratio)
            row = {
                "semantic": semantic,
                "candidate_id": candidate.candidate_id,
                "xy": [float(candidate.xy[0]), float(candidate.xy[1])],
                "residual_px": float(residual),
                "threshold_px": float(threshold),
                "effective_confidence": float(candidate.quality),
                "support": float(support),
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
    coverage = len(accepted) / len(FLOOR_KEYPOINT_NAMES)
    if accepted:
        residual_penalty = sum(
            float(row["effective_confidence"])
            * min((float(row["residual_px"]) / float(row["threshold_px"])) ** 2, 4.0)
            for row in accepted
        ) / max(confidence_support, 1.0e-9)
    else:
        residual_penalty = 4.0
    score = (
        4.0 * weighted_consensus
        + 8.0 * coverage
        + 1.5 * confidence_support
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
        },
        evaluations=tuple(evaluations),
        inliers=tuple(
            {
                "semantic": str(row["semantic"]),
                "candidate_id": str(row["candidate_id"]),
                "xy": list(row["xy"]),
                "residual_px": float(row["residual_px"]),
                "effective_confidence": float(row["effective_confidence"]),
            }
            for row in sorted(accepted, key=lambda item: str(item["semantic"]))
        ),
        residual_stats=_residual_stats(residuals),
        seed_semantics=seed_semantics,
    )


def _refine_model(
    model: _ScoredModel,
    candidates: Mapping[str, tuple[_Observation, ...]],
    *,
    inlier_threshold_px: float,
    duplicate_tolerance_px: float,
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
        world = [FLOOR_WORLD_XY_M[item.semantic] for item in observations]
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
        )
        if _model_sort_key(candidate_model) < _model_sort_key(current):
            current = candidate_model
        else:
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


def _court_confidence(
    model: _ScoredModel,
    *,
    margin: float | None,
    has_alternate: bool,
) -> float:
    inlier_count = len(model.inliers)
    if inlier_count == 0:
        return 0.05 if model.source.startswith("prior_homography") else 0.0
    coverage = inlier_count / len(FLOOR_KEYPOINT_NAMES)
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
    return {
        "semantic": semantic,
        "candidate_id": candidate_id,
        "reason": reason,
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
    "FLOOR_KEYPOINT_NAMES",
    "FLOOR_WORLD_XY_M",
    "NET_TOP_KEYPOINT_NAMES",
    "solve_best_floor_court",
    "solve_structured_court",
]
