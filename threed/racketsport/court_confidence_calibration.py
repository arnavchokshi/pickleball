"""Serializable calibration for structured-court confidence values.

Calibration is intentionally fitted outside training on a source-disjoint
development split.  A calibration artifact improves probability semantics; it
does not grant court authority or change the underlying point geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class IsotonicConfidenceCalibrator:
    upper_bounds: tuple[float, ...]
    probabilities: tuple[float, ...]
    sample_count: int
    positive_count: int
    threshold_px: float = 5.0

    def predict(self, value: float) -> float:
        score = _probability(value, "confidence")
        if not self.upper_bounds:
            return score
        index = bisect.bisect_left(self.upper_bounds, score)
        index = min(index, len(self.probabilities) - 1)
        return float(self.probabilities[index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "calibrator": "isotonic_pav",
            "upper_bounds": list(self.upper_bounds),
            "probabilities": list(self.probabilities),
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "threshold_px": self.threshold_px,
            "promotion_allowed": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IsotonicConfidenceCalibrator":
        if payload.get("calibrator") != "isotonic_pav":
            raise ValueError("unsupported confidence calibrator")
        bounds = tuple(_probability(value, "upper_bounds") for value in payload.get("upper_bounds", ()))
        probabilities = tuple(
            _probability(value, "probabilities") for value in payload.get("probabilities", ())
        )
        if len(bounds) != len(probabilities) or not bounds:
            raise ValueError("isotonic calibrator requires equal non-empty bounds/probabilities")
        if any(left >= right for left, right in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("isotonic upper bounds must be strictly increasing")
        if any(left > right for left, right in zip(probabilities, probabilities[1:], strict=False)):
            raise ValueError("isotonic probabilities must be nondecreasing")
        sample_count = _nonnegative_int(payload.get("sample_count"), "sample_count")
        positive_count = _nonnegative_int(payload.get("positive_count"), "positive_count")
        if positive_count > sample_count:
            raise ValueError("positive_count cannot exceed sample_count")
        threshold_px = _positive_float(payload.get("threshold_px", 5.0), "threshold_px")
        return cls(bounds, probabilities, sample_count, positive_count, threshold_px)


def fit_isotonic_confidence(
    scores: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    threshold_px: float = 5.0,
) -> IsotonicConfidenceCalibrator:
    """Fit a monotone probability mapping with the pool-adjacent-violators algorithm."""

    if len(scores) != len(outcomes) or not scores:
        raise ValueError("scores and outcomes must be equal, non-empty sequences")
    pairs = sorted(
        (_probability(score, "score"), _binary(outcome, "outcome"))
        for score, outcome in zip(scores, outcomes, strict=True)
    )
    grouped: list[dict[str, float]] = []
    for score, outcome in pairs:
        if grouped and math.isclose(grouped[-1]["upper"], score, rel_tol=0.0, abs_tol=1e-15):
            grouped[-1]["sum"] += outcome
            grouped[-1]["count"] += 1.0
        else:
            grouped.append({"lower": score, "upper": score, "sum": outcome, "count": 1.0})

    blocks: list[dict[str, float]] = []
    for group in grouped:
        blocks.append(dict(group))
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["count"] <= right["sum"] / right["count"]:
                break
            merged = {
                "lower": left["lower"],
                "upper": right["upper"],
                "sum": left["sum"] + right["sum"],
                "count": left["count"] + right["count"],
            }
            blocks[-2:] = [merged]

    return IsotonicConfidenceCalibrator(
        upper_bounds=tuple(float(block["upper"]) for block in blocks),
        probabilities=tuple(float(block["sum"] / block["count"]) for block in blocks),
        sample_count=len(pairs),
        positive_count=int(sum(outcome for _, outcome in pairs)),
        threshold_px=_positive_float(threshold_px, "threshold_px"),
    )


def apply_point_calibration(
    point_confidence: Mapping[str, float],
    calibrator: IsotonicConfidenceCalibrator,
) -> dict[str, float]:
    return {str(name): calibrator.predict(float(value)) for name, value in point_confidence.items()}


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1]")
    return result


def _binary(value: Any, field: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    result = _probability(value, field)
    if result not in {0.0, 1.0}:
        raise ValueError(f"{field} must be binary")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


__all__ = [
    "IsotonicConfidenceCalibrator",
    "apply_point_calibration",
    "fit_isotonic_confidence",
]
