"""Confidence propagation and gating."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


CAPTURE_QUALITY_FACTORS = {
    "good": 1.0,
    "warn": 0.75,
    "poor": 0.45,
}


def confidence_grade(
    metric_confidence: float | int | None,
    capture_quality: str | Mapping[str, Any] | Any = "good",
    *,
    missing_upstream_reasons: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic confidence grade for a metric-derived claim."""

    missing_reasons = _clean_reasons(missing_upstream_reasons)
    if missing_reasons:
        return {
            "grade": "omit",
            "score": 0.0,
            "reasons": [f"missing_upstream:{reason}" for reason in missing_reasons],
        }

    normalized_confidence = _normalize_confidence(metric_confidence)
    if normalized_confidence is None:
        return {"grade": "omit", "score": 0.0, "reasons": ["invalid_metric_confidence"]}

    quality_grade, quality_reasons = _capture_quality_parts(capture_quality)
    factor = CAPTURE_QUALITY_FACTORS.get(quality_grade)
    reasons: list[str] = []
    if factor is None:
        factor = CAPTURE_QUALITY_FACTORS["poor"]
        reasons.append(f"capture_quality:{quality_grade}")
    elif quality_grade != "good":
        reasons.append(f"capture_quality:{quality_grade}")
    reasons.extend(f"capture_reason:{reason}" for reason in quality_reasons)

    score = round(normalized_confidence * factor, 6)
    return {
        "grade": _grade_from_score(score),
        "score": score,
        "reasons": reasons,
    }


def gate_metric_claim(
    value: Any,
    conf: float | int | None,
    *,
    threshold: float,
    claim_name: str,
    capture_quality: str | Mapping[str, Any] | Any = "good",
    missing_upstream_reasons: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Gate an absolute metric claim into allow, relative, or omit."""

    _validate_threshold(threshold)
    grade = confidence_grade(
        conf,
        capture_quality,
        missing_upstream_reasons=missing_upstream_reasons,
    )
    reasons = list(grade["reasons"])
    decision = "omit"
    if value is None:
        reasons.append("missing_value")
    elif grade["grade"] != "omit":
        score = float(grade["score"])
        if score >= threshold:
            decision = "allow"
        elif score >= max(0.35, threshold * 0.6):
            decision = "relative"

    return {
        "claim_name": claim_name,
        "decision": decision,
        "value": value,
        "metric_confidence": conf,
        "confidence": grade["score"],
        "threshold": threshold,
        "grade": grade["grade"],
        "reasons": reasons,
    }


def _capture_quality_parts(capture_quality: str | Mapping[str, Any] | Any) -> tuple[str, list[str]]:
    if isinstance(capture_quality, str):
        return capture_quality, []
    if isinstance(capture_quality, Mapping):
        grade = capture_quality.get("grade", "good")
        reasons = capture_quality.get("reasons", [])
    else:
        grade = getattr(capture_quality, "grade", "good")
        reasons = getattr(capture_quality, "reasons", [])

    grade_text = str(grade) if grade is not None else "good"
    return grade_text, _clean_reasons(reasons)


def _clean_reasons(reasons: Iterable[str] | None) -> list[str]:
    if reasons is None:
        return []
    return [str(reason) for reason in reasons if str(reason)]


def _grade_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.35:
        return "low"
    return "omit"


def _normalize_confidence(confidence: float | int | None) -> float | None:
    if confidence is None or isinstance(confidence, bool):
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if value < 0.0 or value > 1.0:
        return None
    return value


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool):
        raise ValueError("threshold must be between 0 and 1")
    value = float(threshold)
    if value < 0.0 or value > 1.0:
        raise ValueError("threshold must be between 0 and 1")
