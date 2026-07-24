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
class TemperatureConfidenceCalibrator:
    temperature: float
    sample_count: int
    positive_count: int
    quality_threshold_px: float = 5.0
    measurement_threshold: float | None = None
    zero_unsupported_false_accepts: bool = False
    promotion_allowed: bool = False

    def predict_logit(self, logit: float) -> float:
        value = _finite_float(logit, "logit") / self.temperature
        if value >= 0.0:
            return float(1.0 / (1.0 + math.exp(-value)))
        exponential = math.exp(value)
        return float(exponential / (1.0 + exponential))

    def predict_probability(self, probability: float) -> float:
        value = min(max(_probability(probability, "probability"), 1.0e-8), 1.0 - 1.0e-8)
        return self.predict_logit(math.log(value / (1.0 - value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "calibrator": "temperature_scaling",
            "temperature": self.temperature,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "quality_threshold_px": self.quality_threshold_px,
            "measurement_threshold": self.measurement_threshold,
            "zero_unsupported_false_accepts": self.zero_unsupported_false_accepts,
            "promotion_allowed": self.promotion_allowed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemperatureConfidenceCalibrator":
        if payload.get("calibrator") != "temperature_scaling":
            raise ValueError("unsupported court confidence calibrator")
        measurement_threshold = payload.get("measurement_threshold")
        sample_count = _nonnegative_int(payload.get("sample_count"), "sample_count")
        positive_count = _nonnegative_int(payload.get("positive_count"), "positive_count")
        if positive_count > sample_count:
            raise ValueError("positive_count cannot exceed sample_count")
        return cls(
            temperature=_positive_float(payload.get("temperature"), "temperature"),
            sample_count=sample_count,
            positive_count=positive_count,
            quality_threshold_px=_positive_float(
                payload.get("quality_threshold_px", 5.0), "quality_threshold_px"
            ),
            measurement_threshold=(
                None
                if measurement_threshold is None
                else _probability(measurement_threshold, "measurement_threshold")
            ),
            zero_unsupported_false_accepts=bool(
                payload.get("zero_unsupported_false_accepts", False)
            ),
            promotion_allowed=bool(payload.get("promotion_allowed", False)),
        )


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


def fit_temperature_confidence(
    logits: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    quality_threshold_px: float = 5.0,
) -> TemperatureConfidenceCalibrator:
    """Fit one positive temperature by deterministic bounded NLL minimization."""

    if len(logits) != len(outcomes) or not logits:
        raise ValueError("logits and outcomes must be equal, non-empty sequences")
    parsed_logits = [_finite_float(value, "logit") for value in logits]
    parsed_outcomes = [_binary(value, "outcome") for value in outcomes]

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        loss = 0.0
        for logit, outcome in zip(parsed_logits, parsed_outcomes, strict=True):
            scaled = logit / temperature
            # Stable Bernoulli negative log likelihood from logits.
            loss += max(scaled, 0.0) - scaled * outcome + math.log1p(math.exp(-abs(scaled)))
        return loss / len(parsed_logits)

    left, right = -4.0, 4.0
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(96):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)
    temperature = math.exp((left + right) * 0.5)
    return TemperatureConfidenceCalibrator(
        temperature=temperature,
        sample_count=len(parsed_logits),
        positive_count=int(sum(parsed_outcomes)),
        quality_threshold_px=_positive_float(quality_threshold_px, "quality_threshold_px"),
    )


def confidence_calibration_report(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes must be equal, non-empty sequences")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    scores = [_probability(value, "probability") for value in probabilities]
    labels = [_binary(value, "outcome") for value in outcomes]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            row
            for row in zip(scores, labels, strict=True)
            if lower <= row[0] <= upper and (index == bin_count - 1 or row[0] < upper)
        ]
        if members:
            mean_confidence = sum(row[0] for row in members) / len(members)
            empirical_accuracy = sum(row[1] for row in members) / len(members)
            ece += len(members) / len(scores) * abs(mean_confidence - empirical_accuracy)
        else:
            mean_confidence = None
            empirical_accuracy = None
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
            }
        )
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels, strict=True)) / len(scores)
    return {"sample_count": len(scores), "brier_score": brier, "ece": ece, "bins": bins}


def select_zero_false_accept_threshold(
    probabilities: Sequence[float],
    quality_outcomes: Sequence[bool | int | float],
    unsupported_view: Sequence[bool | int | float],
) -> float | None:
    """Return the lowest observed threshold with no bad-quality or unsupported accepts."""

    if not probabilities or not (
        len(probabilities) == len(quality_outcomes) == len(unsupported_view)
    ):
        raise ValueError("threshold inputs must be equal, non-empty sequences")
    scores = [_probability(value, "probability") for value in probabilities]
    quality = [_binary(value, "quality_outcome") for value in quality_outcomes]
    unsupported = [_binary(value, "unsupported_view") for value in unsupported_view]
    for threshold in sorted(set(scores)):
        accepted = [index for index, score in enumerate(scores) if score >= threshold]
        if accepted and all(quality[index] == 1.0 and unsupported[index] == 0.0 for index in accepted):
            return float(threshold)
    return None


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1]")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
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
    "TemperatureConfidenceCalibrator",
    "apply_point_calibration",
    "confidence_calibration_report",
    "fit_isotonic_confidence",
    "fit_temperature_confidence",
    "select_zero_false_accept_threshold",
]
