"""Template competition for court proposal evidence."""

from __future__ import annotations

from typing import Any


def _distance_score(actual: float | None, expected: float, tolerance: float) -> float:
    if actual is None:
        return 0.0
    return max(0.0, 1.0 - abs(actual - expected) / tolerance)


def _line_y(lines: dict[str, dict[str, Any]], name: str) -> float | None:
    value = lines.get(name, {}).get("court_y_ft")
    return float(value) if value is not None else None


def score_template_competition(semantic_lines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score semantic cross-lines against pickleball and tennis templates."""

    net_y = _line_y(semantic_lines, "net")
    near_nvz = _line_y(semantic_lines, "near_nvz")
    far_nvz = _line_y(semantic_lines, "far_nvz")
    near_service = _line_y(semantic_lines, "near_service")
    far_service = _line_y(semantic_lines, "far_service")
    near_baseline = _line_y(semantic_lines, "near_baseline")
    far_baseline = _line_y(semantic_lines, "far_baseline")

    pickleball_score = 0.0
    tennis_score = 0.0

    if net_y is not None and near_nvz is not None:
        pickleball_score += _distance_score(net_y - near_nvz, 7.0, 3.0)
    if net_y is not None and far_nvz is not None:
        pickleball_score += _distance_score(far_nvz - net_y, 7.0, 3.0)
    if near_baseline is not None and far_baseline is not None:
        pickleball_score += _distance_score(far_baseline - near_baseline, 44.0, 8.0)
        tennis_score += _distance_score(far_baseline - near_baseline, 78.0, 10.0)
    if net_y is not None and near_service is not None:
        tennis_score += _distance_score(net_y - near_service, 21.0, 5.0)
    if net_y is not None and far_service is not None:
        tennis_score += _distance_score(far_service - net_y, 21.0, 5.0)

    pickleball_score /= 3.0
    tennis_score /= 3.0
    winner = "pickleball" if pickleball_score > tennis_score else "tennis"
    margin = abs(pickleball_score - tennis_score)
    pickleball_reject_reasons: list[str] = []
    if winner == "tennis":
        pickleball_reject_reasons.append("tennis_template_wins")
    if margin < 0.2:
        pickleball_reject_reasons.append("template_margin_too_small")

    return {
        "winner": winner,
        "margin": float(margin),
        "pickleball": {
            "score": float(pickleball_score),
            "reject_reasons": pickleball_reject_reasons,
        },
        "tennis": {"score": float(tennis_score), "reject_reasons": []},
    }
