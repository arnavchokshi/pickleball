"""Diagnostic-only CAL court-keypoint label-space replay checks.

This module compares no-tap court-keypoint predictions against trusted
calibration projections. It is intentionally a confusion diagnostic, not a
CAL-3 promotion path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINTS,
    validate_heatmap_prediction_payload,
)
from threed.racketsport.schemas import CourtCalibration


DEFAULT_EVAL_REPORT = Path(
    "runs/pickleball_pretraining/court_keypoint_20260628/"
    "diagnostic_20260630_cal_lead_verify_current/court_keypoint_no_tap_eval_cpu.json"
)
DEFAULT_OUTPUT = Path(
    "runs/pickleball_pretraining/court_keypoint_20260628/"
    "diagnostic_20260630_cal_lead_verify_current/labelspace_diagnostic/"
    "court_keypoint_labelspace_diagnostic.json"
)
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_OUTPUT.with_suffix(".md")
DEFAULT_THRESHOLD = 0.05
PIXEL_EXPLAINED_GATE_PX = 5.0
PIXEL_BAD_GATE_PX = 20.0


Point2 = tuple[float, float]


def analyze_court_keypoint_labelspace(
    *,
    prediction_payload: Mapping[str, Any],
    calibration_payload: CourtCalibration | Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Compare predicted keypoints to trusted calibration projections.

    Returned payloads are diagnostic-only. They can identify likely coordinate
    transforms or label confusion, but they are not ground-truth keypoint
    validation and must not promote CAL.
    """

    confidence_threshold = _unit_interval(threshold, "threshold")
    calibration = (
        calibration_payload
        if isinstance(calibration_payload, CourtCalibration)
        else CourtCalibration.model_validate(calibration_payload)
    )
    expected_by_name = _trusted_keypoint_projection(calibration)
    source_width, source_height = _prediction_source_size(prediction_payload, calibration)
    frames = _prediction_frames(prediction_payload, max_frames=max_frames)
    pairs = _selected_pairs(frames, expected_by_name=expected_by_name, threshold=confidence_threshold)
    frame_points = _selected_frame_points(frames, threshold=confidence_threshold)

    transform_results = _transform_results(
        pairs,
        source_width=source_width,
        source_height=source_height,
    )
    nearest_result, confusion_pairs, same_label_ratio = _nearest_keypoint_permutation_result(
        frame_points,
        expected_by_name=expected_by_name,
    )
    transform_results.append(nearest_result)

    best_label = _best_transform(
        transform_results,
        names=[
            "identity",
            "scale_translate_xy",
            "flip_x_scale_translate_xy",
            "flip_y_scale_translate_xy",
            "flip_xy_scale_translate_xy",
            "affine",
        ],
    )
    best_overall = _best_transform(transform_results)
    identity = _result_by_name(transform_results, "identity")
    likely_failure_mode = _likely_failure_mode(
        identity=identity,
        best_label_preserving=best_label,
        nearest=nearest_result,
        nearest_same_label_ratio=same_label_ratio,
        pair_count=len(pairs),
    )
    blockers = _clip_blockers(pairs=pairs, best_label=best_label, nearest=nearest_result)

    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_labelspace_clip_diagnostic",
        "status": "diagnostic_only" if pairs else "blocked",
        "claim_scope": "diagnostic_only_no_cal_promotion",
        "clip": str(prediction_payload.get("clip") or "unknown"),
        "threshold": confidence_threshold,
        "frame_count": len(frames),
        "compared_keypoint_count": len(pairs),
        "source_size": [source_width, source_height],
        "verified": False,
        "promote_cal": False,
        "cal3_verified": False,
        "best_label_preserving_transform": best_label["name"] if best_label is not None else None,
        "best_overall_transform": best_overall["name"] if best_overall is not None else None,
        "nearest_keypoint_same_label_ratio": same_label_ratio,
        "nearest_keypoint_confusion_top_pairs": confusion_pairs,
        "likely_failure_mode": likely_failure_mode,
        "blockers": blockers,
        "transform_results": transform_results,
        "notes": [
            "diagnostic only; compares predictions against trusted calibration projection",
            "not CAL-3 verified; no promotion decision is made here",
        ],
    }


def build_court_keypoint_labelspace_report(
    *,
    eval_report_path: str | Path = DEFAULT_EVAL_REPORT,
    out: str | Path = DEFAULT_OUTPUT,
    markdown_out: str | Path | None = DEFAULT_MARKDOWN_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
    max_frames_per_clip: int | None = None,
) -> dict[str, Any]:
    """Build a JSON/Markdown label-space diagnostic from an eval report."""

    eval_path = Path(eval_report_path)
    out_path = Path(out)
    markdown_path = Path(markdown_out) if markdown_out is not None else None
    confidence_threshold = _unit_interval(threshold, "threshold")
    if not eval_path.is_file():
        report = _blocked_report(
            eval_report_path=eval_path,
            threshold=confidence_threshold,
            blockers=["missing_eval_report"],
            notes=[f"missing eval report: {eval_path}"],
        )
        _write_report(report, out_path=out_path, markdown_path=markdown_path)
        return report

    try:
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = _blocked_report(
            eval_report_path=eval_path,
            threshold=confidence_threshold,
            blockers=["invalid_eval_report_json"],
            notes=[f"could not parse eval report: {exc}"],
        )
        _write_report(report, out_path=out_path, markdown_path=markdown_path)
        return report

    clips_raw = eval_payload.get("clips")
    if not isinstance(clips_raw, Sequence) or isinstance(clips_raw, (str, bytes)):
        report = _blocked_report(
            eval_report_path=eval_path,
            threshold=confidence_threshold,
            blockers=["eval_report_missing_clips"],
            notes=["eval report does not contain a clips array"],
        )
        _write_report(report, out_path=out_path, markdown_path=markdown_path)
        return report

    clips: list[dict[str, Any]] = []
    for clip_raw in clips_raw:
        if not isinstance(clip_raw, Mapping):
            continue
        clips.append(
            _clip_from_eval_entry(
                clip_raw,
                threshold=confidence_threshold,
                max_frames=max_frames_per_clip,
            )
        )

    analyzed_count = sum(1 for clip in clips if clip.get("status") == "diagnostic_only")
    blocked_count = sum(1 for clip in clips if clip.get("status") == "blocked")
    blockers = [] if analyzed_count else ["no_analyzable_prediction_calibration_pairs"]
    report = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_labelspace_diagnostic",
        "status": "diagnostic_only" if analyzed_count else "blocked",
        "claim_scope": "diagnostic_only_no_cal_promotion",
        "eval_report": str(eval_path),
        "threshold": confidence_threshold,
        "verified": False,
        "promote_cal": False,
        "cal3_verified": False,
        "blockers": blockers,
        "summary": {
            "clip_count": len(clips),
            "analyzed_clip_count": analyzed_count,
            "blocked_clip_count": blocked_count,
            "likely_failure_modes": _failure_mode_counts(clips),
        },
        "clips": clips,
        "notes": [
            "diagnostic only; this report cannot promote CAL",
            "passing a transform diagnostic would still require independent real keypoint labels and phase gates",
        ],
    }
    _write_report(report, out_path=out_path, markdown_path=markdown_path)
    return report


def render_court_keypoint_labelspace_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Court Keypoint Label-Space Diagnostic",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- claim scope: `{report.get('claim_scope', 'diagnostic_only_no_cal_promotion')}`",
        "- verdict: not CAL-3 verified",
        f"- threshold: `{report.get('threshold', DEFAULT_THRESHOLD)}`",
        f"- blockers: {', '.join(str(item) for item in report.get('blockers', [])) or 'none'}",
        "",
        "| clip | samples | identity p95 px | best label transform | best p95 px | nearest p95 px | same-label | likely failure |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for clip in report.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        results = clip.get("transform_results", [])
        identity = _result_by_name(results, "identity") if isinstance(results, list) else None
        best = _result_by_name(results, str(clip.get("best_label_preserving_transform"))) if isinstance(results, list) else None
        nearest = _result_by_name(results, "nearest_keypoint_permutation") if isinstance(results, list) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(clip.get("clip", "unknown")),
                    str(clip.get("compared_keypoint_count", 0)),
                    _result_p95(identity),
                    str(clip.get("best_label_preserving_transform") or "n/a"),
                    _result_p95(best),
                    _result_p95(nearest),
                    _format_float(clip.get("nearest_keypoint_same_label_ratio")),
                    str(clip.get("likely_failure_mode") or ", ".join(clip.get("blockers", [])) or "n/a"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a diagnostic-only replay against existing trusted calibration projections.",
            "- It does not use independent court-keypoint ground truth and must not be used as CAL promotion evidence.",
        ]
    )
    if report.get("notes"):
        lines.append("")
        for note in report.get("notes", []):
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _clip_from_eval_entry(
    clip_entry: Mapping[str, Any],
    *,
    threshold: float,
    max_frames: int | None,
) -> dict[str, Any]:
    clip_name = str(clip_entry.get("clip") or "unknown")
    prediction_path = _resolve_existing_path(clip_entry.get("prediction_artifact"))
    calibration_path = _resolve_existing_path(clip_entry.get("court_calibration"))
    missing: list[str] = []
    if prediction_path is None:
        missing.append("missing_prediction_artifact")
    if calibration_path is None:
        missing.append("missing_court_calibration")
    if missing:
        return _blocked_clip(clip_name, blockers=missing, clip_entry=clip_entry)

    try:
        prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        clip = analyze_court_keypoint_labelspace(
            prediction_payload=prediction_payload,
            calibration_payload=calibration_payload,
            threshold=threshold,
            max_frames=max_frames,
        )
    except Exception as exc:
        return _blocked_clip(clip_name, blockers=["labelspace_analysis_failed"], clip_entry=clip_entry, notes=[str(exc)])

    clip["prediction_artifact"] = str(prediction_path)
    clip["court_calibration"] = str(calibration_path)
    clip["source_eval_status"] = str(clip_entry.get("status") or "unknown")
    return clip


def _transform_results(
    pairs: list[dict[str, Any]],
    *,
    source_width: float,
    source_height: float,
) -> list[dict[str, Any]]:
    predicted = [pair["predicted"] for pair in pairs]
    expected = [pair["expected"] for pair in pairs]
    transforms: list[tuple[str, Callable[[Point2], Point2], dict[str, Any], list[str]]] = [
        ("identity", lambda point: point, {}, []),
    ]

    scale = _fit_scale_translate(predicted, expected)
    if scale is not None:
        transforms.append(("scale_translate_xy", scale[0], scale[1], scale[2]))

    flip_x_points = [(source_width - point[0], point[1]) for point in predicted]
    flip_x = _fit_scale_translate(flip_x_points, expected)
    if flip_x is not None:
        transform, params, notes = flip_x
        transforms.append(
            (
                "flip_x_scale_translate_xy",
                lambda point, transform=transform: transform((source_width - point[0], point[1])),
                {"pre_transform": "flip_x", **params},
                notes,
            )
        )

    flip_y_points = [(point[0], source_height - point[1]) for point in predicted]
    flip_y = _fit_scale_translate(flip_y_points, expected)
    if flip_y is not None:
        transform, params, notes = flip_y
        transforms.append(
            (
                "flip_y_scale_translate_xy",
                lambda point, transform=transform: transform((point[0], source_height - point[1])),
                {"pre_transform": "flip_y", **params},
                notes,
            )
        )

    flip_xy_points = [(source_width - point[0], source_height - point[1]) for point in predicted]
    flip_xy = _fit_scale_translate(flip_xy_points, expected)
    if flip_xy is not None:
        transform, params, notes = flip_xy
        transforms.append(
            (
                "flip_xy_scale_translate_xy",
                lambda point, transform=transform: transform((source_width - point[0], source_height - point[1])),
                {"pre_transform": "flip_xy", **params},
                notes,
            )
        )

    affine = _fit_affine(predicted, expected)
    if affine is not None:
        transforms.append(("affine", affine[0], affine[1], affine[2]))

    results = [_transform_result(name, transform, pairs=pairs, parameters=params, notes=notes) for name, transform, params, notes in transforms]
    present = {result["name"] for result in results}
    for name in (
        "scale_translate_xy",
        "flip_x_scale_translate_xy",
        "flip_y_scale_translate_xy",
        "flip_xy_scale_translate_xy",
        "affine",
    ):
        if name not in present:
            results.append(_unavailable_result(name, sample_count=len(pairs), notes=["not_enough_independent_points"]))
    return results


def _transform_result(
    name: str,
    transform: Callable[[Point2], Point2],
    *,
    pairs: list[dict[str, Any]],
    parameters: Mapping[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    residuals: list[float] = []
    for pair in pairs:
        residuals.append(_distance(transform(pair["predicted"]), pair["expected"]))
    return {
        "name": name,
        "available": bool(pairs),
        "sample_count": len(residuals),
        "residual_px": _metric_summary(residuals),
        "parameters": dict(parameters),
        "notes": notes,
    }


def _nearest_keypoint_permutation_result(
    frame_points: list[dict[str, Any]],
    *,
    expected_by_name: Mapping[str, Point2],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    residuals: list[float] = []
    confusion: dict[tuple[str, str], int] = {}
    same_label_count = 0
    assignment_count = 0
    expected_items = list(expected_by_name.items())
    for frame in frame_points:
        assignments = _best_unique_assignment(frame["points"], expected_items)
        for predicted_name, assigned_name, residual in assignments:
            residuals.append(residual)
            confusion[(predicted_name, assigned_name)] = confusion.get((predicted_name, assigned_name), 0) + 1
            assignment_count += 1
            if predicted_name == assigned_name:
                same_label_count += 1
    top_pairs = [
        {"predicted": predicted, "nearest": assigned, "count": count}
        for (predicted, assigned), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:20]
    ]
    result = {
        "name": "nearest_keypoint_permutation",
        "available": bool(residuals),
        "sample_count": len(residuals),
        "residual_px": _metric_summary(residuals),
        "parameters": {"assignment": "per_frame_min_sum_unique_keypoints"},
        "notes": ["ignores predicted keypoint labels to expose permutation-style confusion"],
    }
    return result, top_pairs, _ratio(same_label_count, assignment_count)


def _best_unique_assignment(
    predicted_points: list[tuple[str, Point2]],
    expected_items: list[tuple[str, Point2]],
) -> list[tuple[str, str, float]]:
    if not predicted_points:
        return []
    if len(predicted_points) > len(expected_items):
        predicted_points = predicted_points[: len(expected_items)]

    dp: dict[int, tuple[float, list[tuple[str, str, float]]]] = {0: (0.0, [])}
    for predicted_name, predicted_point in predicted_points:
        next_dp: dict[int, tuple[float, list[tuple[str, str, float]]]] = {}
        for mask, (cost, assignments) in dp.items():
            for index, (expected_name, expected_point) in enumerate(expected_items):
                bit = 1 << index
                if mask & bit:
                    continue
                residual = _distance(predicted_point, expected_point)
                next_mask = mask | bit
                candidate = (cost + residual, [*assignments, (predicted_name, expected_name, residual)])
                current = next_dp.get(next_mask)
                if current is None or candidate[0] < current[0]:
                    next_dp[next_mask] = candidate
        dp = next_dp
    if not dp:
        return []
    return min(dp.values(), key=lambda item: item[0])[1]


def _fit_scale_translate(
    predicted: Sequence[Point2],
    expected: Sequence[Point2],
) -> tuple[Callable[[Point2], Point2], dict[str, float], list[str]] | None:
    if len(predicted) < 2 or len(expected) < 2:
        return None
    sx_tx = _fit_line([point[0] for point in predicted], [point[0] for point in expected])
    sy_ty = _fit_line([point[1] for point in predicted], [point[1] for point in expected])
    if sx_tx is None or sy_ty is None:
        return None
    sx, tx = sx_tx
    sy, ty = sy_ty
    notes: list[str] = []
    if sx < 0.0 or sy < 0.0:
        notes.append("negative_scale_indicates_axis_flip_or_label_confusion")

    def transform(point: Point2) -> Point2:
        return sx * point[0] + tx, sy * point[1] + ty

    return transform, {"scale_x": sx, "translate_x": tx, "scale_y": sy, "translate_y": ty}, notes


def _fit_affine(
    predicted: Sequence[Point2],
    expected: Sequence[Point2],
) -> tuple[Callable[[Point2], Point2], dict[str, float], list[str]] | None:
    if len(predicted) < 3 or len(expected) < 3:
        return None
    features = [[point[0], point[1], 1.0] for point in predicted]
    coeff_x = _least_squares(features, [point[0] for point in expected])
    coeff_y = _least_squares(features, [point[1] for point in expected])
    if coeff_x is None or coeff_y is None:
        return None
    a, b, tx = coeff_x
    c, d, ty = coeff_y

    def transform(point: Point2) -> Point2:
        x, y = point
        return a * x + b * y + tx, c * x + d * y + ty

    return transform, {"a": a, "b": b, "c": c, "d": d, "translate_x": tx, "translate_y": ty}, []


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return 0.0, sum_y / n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _least_squares(features: Sequence[Sequence[float]], values: Sequence[float]) -> list[float] | None:
    if not features or len(features) != len(values):
        return None
    width = len(features[0])
    ata = [[0.0 for _ in range(width)] for _ in range(width)]
    atb = [0.0 for _ in range(width)]
    for row, value in zip(features, values):
        if len(row) != width:
            return None
        for i in range(width):
            atb[i] += row[i] * value
            for j in range(width):
                ata[i][j] += row[i] * row[j]
    return _solve_linear_system(ata, atb)


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float] | None:
    size = len(values)
    aug = [list(row) + [values[index]] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-9:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, size + 1):
            aug[col][j] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, size + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[row][size] for row in range(size)]


def _selected_pairs(
    frames: list[Mapping[str, Any]],
    *,
    expected_by_name: Mapping[str, Point2],
    threshold: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for frame in frames:
        frame_index = int(frame.get("frame_index", len(pairs)) or 0)
        predictions = validate_heatmap_prediction_payload({"keypoints": frame.get("keypoints", {})})
        for name, prediction in predictions.items():
            if prediction.confidence < threshold or name not in expected_by_name:
                continue
            pairs.append(
                {
                    "frame_index": frame_index,
                    "name": name,
                    "predicted": prediction.image_xy,
                    "expected": expected_by_name[name],
                    "confidence": prediction.confidence,
                }
            )
    return pairs


def _selected_frame_points(
    frames: list[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frames:
        predictions = validate_heatmap_prediction_payload({"keypoints": frame.get("keypoints", {})})
        points = [
            (name, prediction.image_xy)
            for name, prediction in predictions.items()
            if prediction.confidence >= threshold
        ]
        if points:
            selected.append({"frame_index": frame.get("frame_index"), "points": points})
    return selected


def _trusted_keypoint_projection(calibration: CourtCalibration) -> dict[str, Point2]:
    return {
        point.name: tuple(project_planar_points(calibration.homography, [point.world_xyz_m[:2]])[0])  # type: ignore[misc]
        for point in PICKLEBALL_KEYPOINTS
    }


def _prediction_frames(prediction_payload: Mapping[str, Any], *, max_frames: int | None) -> list[Mapping[str, Any]]:
    frames = prediction_payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("prediction payload must contain frames")
    parsed = [frame for frame in frames if isinstance(frame, Mapping)]
    if max_frames is not None:
        return parsed[: max(0, int(max_frames))]
    return parsed


def _prediction_source_size(
    prediction_payload: Mapping[str, Any],
    calibration: CourtCalibration,
) -> tuple[float, float]:
    source_size = prediction_payload.get("source_size")
    if _is_point(source_size):
        return float(source_size[0]), float(source_size[1])
    if calibration.image_size is not None and len(calibration.image_size) == 2:
        return float(calibration.image_size[0]), float(calibration.image_size[1])
    return float(calibration.intrinsics.cx) * 2.0, float(calibration.intrinsics.cy) * 2.0


def _likely_failure_mode(
    *,
    identity: Mapping[str, Any] | None,
    best_label_preserving: Mapping[str, Any] | None,
    nearest: Mapping[str, Any],
    nearest_same_label_ratio: float,
    pair_count: int,
) -> str:
    if pair_count == 0:
        return "no_confident_keypoints"
    identity_p95 = _p95(identity)
    best_label_p95 = _p95(best_label_preserving)
    nearest_p95 = _p95(nearest)
    if identity_p95 <= PIXEL_EXPLAINED_GATE_PX:
        return "labelspace_matches_trusted_projection"
    if nearest_p95 <= PIXEL_EXPLAINED_GATE_PX and nearest_same_label_ratio < 0.5 and identity_p95 > PIXEL_BAD_GATE_PX:
        return "label_permutation_or_confusion"
    if best_label_p95 <= PIXEL_EXPLAINED_GATE_PX and identity_p95 > PIXEL_BAD_GATE_PX:
        return "global_coordinate_transform_mismatch"
    return "non_localizing_underfit_or_unmodeled_error"


def _clip_blockers(
    *,
    pairs: list[dict[str, Any]],
    best_label: Mapping[str, Any] | None,
    nearest: Mapping[str, Any],
) -> list[str]:
    if not pairs:
        return ["no_confident_keypoints_at_threshold"]
    blockers: list[str] = []
    if _p95(best_label) > PIXEL_EXPLAINED_GATE_PX:
        blockers.append("no_label_preserving_transform_explains_predictions")
    if _p95(nearest) > PIXEL_EXPLAINED_GATE_PX:
        blockers.append("nearest_keypoint_residual_still_high")
    return blockers


def _best_transform(
    transform_results: Sequence[Mapping[str, Any]],
    names: Sequence[str] | None = None,
) -> Mapping[str, Any] | None:
    allowed = set(names) if names is not None else None
    candidates = [
        result
        for result in transform_results
        if result.get("available") is True
        and (allowed is None or result.get("name") in allowed)
        and isinstance(result.get("residual_px"), Mapping)
    ]
    if not candidates:
        return None
    priority = {name: index for index, name in enumerate(names or [str(result.get("name")) for result in candidates])}
    return min(
        candidates,
        key=lambda result: (
            round(float(result["residual_px"]["p95"]), 9),
            priority.get(str(result.get("name")), 999),
        ),
    )


def _result_by_name(results: Sequence[Any], name: str | None) -> Mapping[str, Any] | None:
    if name is None:
        return None
    for result in results:
        if isinstance(result, Mapping) and result.get("name") == name:
            return result
    return None


def _blocked_report(
    *,
    eval_report_path: Path,
    threshold: float,
    blockers: list[str],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_labelspace_diagnostic",
        "status": "blocked",
        "claim_scope": "diagnostic_only_no_cal_promotion",
        "eval_report": str(eval_report_path),
        "threshold": threshold,
        "verified": False,
        "promote_cal": False,
        "cal3_verified": False,
        "blockers": blockers,
        "summary": {
            "clip_count": 0,
            "analyzed_clip_count": 0,
            "blocked_clip_count": 0,
            "likely_failure_modes": {},
        },
        "clips": [],
        "notes": notes,
    }


def _blocked_clip(
    clip_name: str,
    *,
    blockers: list[str],
    clip_entry: Mapping[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_labelspace_clip_diagnostic",
        "status": "blocked",
        "claim_scope": "diagnostic_only_no_cal_promotion",
        "clip": clip_name,
        "source_eval_status": str(clip_entry.get("status") or "unknown"),
        "prediction_artifact": clip_entry.get("prediction_artifact"),
        "court_calibration": clip_entry.get("court_calibration"),
        "threshold": None,
        "frame_count": 0,
        "compared_keypoint_count": 0,
        "verified": False,
        "promote_cal": False,
        "cal3_verified": False,
        "best_label_preserving_transform": None,
        "best_overall_transform": None,
        "nearest_keypoint_same_label_ratio": 0.0,
        "nearest_keypoint_confusion_top_pairs": [],
        "likely_failure_mode": "blocked_missing_inputs" if blockers else "blocked",
        "blockers": blockers,
        "transform_results": [],
        "notes": notes or [],
    }


def _write_report(report: Mapping[str, Any], *, out_path: Path, markdown_path: Path | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_court_keypoint_labelspace_markdown(report), encoding="utf-8")


def _resolve_existing_path(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path
    return None


def _failure_mode_counts(clips: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in clips:
        mode = str(clip.get("likely_failure_mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _unavailable_result(name: str, *, sample_count: int, notes: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "available": False,
        "sample_count": sample_count,
        "residual_px": None,
        "parameters": {},
        "notes": notes,
    }


def _metric_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": _percentile(finite, 50.0),
        "p95": _percentile(finite, 95.0),
        "max": finite[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (percentile / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _p95(result: Mapping[str, Any] | None) -> float:
    if result is None:
        return math.inf
    residual = result.get("residual_px")
    if not isinstance(residual, Mapping):
        return math.inf
    value = residual.get("p95")
    if isinstance(value, (int, float)):
        return float(value)
    return math.inf


def _result_p95(result: Mapping[str, Any] | None) -> str:
    value = _p95(result)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.3f}"
    return "n/a"


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _is_point(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return False
    return all(isinstance(item, (int, float)) for item in value)


def _unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric
