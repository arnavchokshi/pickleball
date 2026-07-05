"""Template competition for court proposal evidence."""

from __future__ import annotations

from typing import Any, Sequence


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


# ---------------------------------------------------------------------------
# Joint 2-D pickleball vs tennis template competition (CAL-GEO 2026-07-05).
#
# The stub above only ever compared cross-court (net/NVZ/baseline) spacing.
# The research evidence explicitly calls out sideline-width, service-line, and
# overlong-line competition as still missing. This adds a genuine 2-D
# assignment: every observed cross-line AND sideline group is scored against
# BOTH templates (not just pickleball vs a single tennis-service check), so a
# segment that is well explained by the tennis template stops being used as
# pickleball evidence instead of merely being penalized once.
# ---------------------------------------------------------------------------

PICKLEBALL_CROSS_Y_FT: dict[str, float] = {
    "far_baseline": 22.0,
    "far_nvz": 7.0,
    "net": 0.0,
    "near_nvz": -7.0,
    "near_baseline": -22.0,
}
TENNIS_CROSS_Y_FT: dict[str, float] = {
    "far_baseline": 39.0,
    "far_service": 21.0,
    "net": 0.0,
    "near_service": -21.0,
    "near_baseline": -39.0,
}
PICKLEBALL_LONG_X_FT: dict[str, float] = {
    "left_sideline": -10.0,
    "centerline": 0.0,
    "right_sideline": 10.0,
}
TENNIS_DOUBLES_LONG_X_FT: dict[str, float] = {
    "doubles_left": -18.0,
    "singles_left": -13.5,
    "center_service": 0.0,
    "singles_right": 13.5,
    "doubles_right": 18.0,
}
PICKLEBALL_COURT_WIDTH_FT = 20.0
PICKLEBALL_COURT_LENGTH_FT = 44.0
TENNIS_DOUBLES_WIDTH_FT = 36.0
TENNIS_SINGLES_WIDTH_FT = 27.0
TENNIS_COURT_LENGTH_FT = 78.0

# Structural-slot mapping: pickleball and tennis cross-line labels that
# occupy the SAME physical role (far baseline, far mid-line, net, near
# mid-line, near baseline) map to the same slot. This lets any actually-
# assigned label subset (e.g. ("far_baseline", "far_nvz", "near_nvz") -- an
# ASYMMETRIC 3-line subset with no near_baseline at all) be scored against
# both templates correctly, unlike a naive "count implies baseline/net/
# baseline" assumption, which is symmetric and therefore has zero
# discriminating power between differently-scaled but equally-symmetric
# templates.
_CROSS_LABEL_TO_SLOT: dict[str, str] = {
    "far_baseline": "far_baseline",
    "far_nvz": "far_mid",
    "far_service": "far_mid",
    "net": "net",
    "near_nvz": "near_mid",
    "near_service": "near_mid",
    "near_baseline": "near_baseline",
}
_SLOT_TO_PICKLEBALL_FT: dict[str, float] = {"far_baseline": 22.0, "far_mid": 7.0, "net": 0.0, "near_mid": -7.0, "near_baseline": -22.0}
_SLOT_TO_TENNIS_FT: dict[str, float] = {"far_baseline": 39.0, "far_mid": 21.0, "net": 0.0, "near_mid": -21.0, "near_baseline": -39.0}


def _spacing_error(observed: Sequence[float], expected: Sequence[float]) -> float:
    """Scale-invariant residual between observed pixel gaps and expected foot gaps."""

    if len(observed) != len(expected) or not observed:
        return 0.0
    denom = sum(value * value for value in expected)
    if denom <= 1e-6:
        return 0.0
    scale = sum(float(obs) * float(exp) for obs, exp in zip(observed, expected)) / denom
    if scale <= 1e-6:
        return float("inf")
    return sum(
        abs(float(obs) - scale * float(exp)) / max(1.0, scale * float(exp)) for obs, exp in zip(observed, expected)
    ) / len(observed)


def score_cross_line_spacing(cross_labels: Sequence[str], observed_y_px: Sequence[float]) -> dict[str, Any]:
    """Score observed (sorted far->near) cross-line pixel positions against both templates.

    `cross_labels` must be the ACTUAL assigned label for each observed
    position, in the same far->near sorted order (e.g. a subset like
    `("far_baseline", "far_nvz", "near_nvz")`, which has no near_baseline at
    all). Using the real labels -- rather than assuming a fixed generic
    baseline/net/baseline structure for every 3-line case -- is required for
    correct discrimination: a baseline/net/baseline subset is symmetric for
    BOTH templates and can never distinguish them by shape alone.
    """

    count = len(observed_y_px)
    if count < 3 or len(cross_labels) != count:
        return {"available": False, "reason": "unsupported_cross_line_count"}
    try:
        slots = [_CROSS_LABEL_TO_SLOT[str(label)] for label in cross_labels]
        pickleball_expected = [_SLOT_TO_PICKLEBALL_FT[slot] for slot in slots]
        tennis_expected = [_SLOT_TO_TENNIS_FT[slot] for slot in slots]
    except KeyError:
        return {"available": False, "reason": "unknown_cross_label"}
    gaps = [float(observed_y_px[i + 1]) - float(observed_y_px[i]) for i in range(count - 1)]
    pickleball_gaps = [abs(pickleball_expected[i + 1] - pickleball_expected[i]) for i in range(len(pickleball_expected) - 1)]
    tennis_gaps = [abs(tennis_expected[i + 1] - tennis_expected[i]) for i in range(len(tennis_expected) - 1)]
    return {
        "available": True,
        "pickleball_error": _spacing_error(gaps, pickleball_gaps),
        "tennis_error": _spacing_error(gaps, tennis_gaps),
        "observed_gaps_px": [round(float(v), 3) for v in gaps],
    }


def score_sideline_width_competition(left_x_px: float, right_x_px: float, near_y_px: float, far_y_px: float) -> dict[str, Any]:
    """Compare the observed width/length ratio against pickleball vs tennis regulation ratios.

    This is dimensionless (width-px / length-px vs width-ft / length-ft), so it
    does not require solving for the unknown pixels-per-foot scale first. It is
    the concrete "sideline-width consistency" check the research evidence
    flagged as still missing from tennis-overlay rejection.
    """

    width_px = abs(float(right_x_px) - float(left_x_px))
    length_px = abs(float(far_y_px) - float(near_y_px))
    if length_px <= 1e-6:
        return {"available": False, "reason": "degenerate_length"}
    observed_ratio = width_px / length_px
    pickleball_ratio = PICKLEBALL_COURT_WIDTH_FT / PICKLEBALL_COURT_LENGTH_FT
    tennis_doubles_ratio = TENNIS_DOUBLES_WIDTH_FT / TENNIS_COURT_LENGTH_FT
    tennis_singles_ratio = TENNIS_SINGLES_WIDTH_FT / TENNIS_COURT_LENGTH_FT
    # This ratio is only a valid template-competition cue for roughly
    # fronto-parallel views: real camera perspective foreshortens the x and y
    # directions differently, so width_px/length_px is NOT camera-invariant in
    # general (unlike the cross-line spacing scale-fit, which only ever fits a
    # single shared y-axis scale). When the observed ratio is wildly outside
    # what either template could plausibly produce even under perspective,
    # this heuristic is unreliable and must not be allowed to inject a
    # spurious tennis/pickleball bias -- treat it as unavailable rather than
    # emitting a huge, direction-biased error.
    plausible_ceiling = 6.0 * max(pickleball_ratio, tennis_doubles_ratio)
    if observed_ratio <= 0.0 or observed_ratio > plausible_ceiling:
        return {
            "available": False,
            "reason": "implausible_width_length_ratio",
            "observed_width_over_length": round(float(observed_ratio), 4),
        }
    pickleball_error = min(3.0, abs(observed_ratio - pickleball_ratio) / pickleball_ratio)
    tennis_error = min(
        3.0,
        min(
            abs(observed_ratio - tennis_doubles_ratio) / tennis_doubles_ratio,
            abs(observed_ratio - tennis_singles_ratio) / tennis_singles_ratio,
        ),
    )
    best_tennis_ratio = min((tennis_doubles_ratio, tennis_singles_ratio), key=lambda r: abs(observed_ratio - r))
    return {
        "available": True,
        "observed_width_over_length": round(float(observed_ratio), 4),
        "pickleball_expected_ratio": round(float(pickleball_ratio), 4),
        "tennis_expected_ratio_best": round(float(best_tennis_ratio), 4),
        "pickleball_error": round(float(pickleball_error), 6),
        "tennis_error": round(float(tennis_error), 6),
    }


def score_joint_template_competition(
    *,
    cross_labels: Sequence[str] | None = None,
    cross_y_px: Sequence[float] | None = None,
    left_right_top_bottom_px: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Real Stage-3 2-D joint pickleball-vs-tennis template competition.

    Combines cross-line spacing (net/NVZ/baseline vs net/service/baseline) with
    the dimensionless sideline-width/length ratio (20/44 vs 36/78 or 27/78).
    `cross_labels` must be the actual assigned label per `cross_y_px` entry
    (see `score_cross_line_spacing`). Either input may be omitted when that
    evidence is unavailable; the remaining evidence still yields a verdict.
    Returns a winner + margin: a hypothesis's observed line pool is only
    "pickleball-explained" when it wins by a clear margin, otherwise it is
    tennis-explained (or ambiguous) and must not be used to promote a
    pickleball hypothesis -- this is what stops a tennis-overlay segment pool
    from polluting the pickleball fit.
    """

    components: dict[str, Any] = {}
    pickleball_costs: list[float] = []
    tennis_costs: list[float] = []

    if cross_y_px is not None and cross_labels is not None:
        cross = score_cross_line_spacing(cross_labels, cross_y_px)
        components["cross_line_spacing"] = cross
        if cross.get("available"):
            pickleball_costs.append(float(cross["pickleball_error"]))
            tennis_costs.append(float(cross["tennis_error"]))

    if left_right_top_bottom_px is not None:
        left_x, right_x, near_y, far_y = left_right_top_bottom_px
        width_ratio = score_sideline_width_competition(left_x, right_x, near_y, far_y)
        components["sideline_width"] = width_ratio
        if width_ratio.get("available"):
            pickleball_costs.append(float(width_ratio["pickleball_error"]))
            tennis_costs.append(float(width_ratio["tennis_error"]))

    if not pickleball_costs:
        return {"available": False, "reason": "no_joint_evidence", "components": components}

    pickleball_cost = sum(pickleball_costs) / len(pickleball_costs)
    tennis_cost = sum(tennis_costs) / len(tennis_costs)
    margin = float(tennis_cost - pickleball_cost)
    winner = "pickleball" if margin > 0.05 else ("tennis" if margin < -0.05 else "ambiguous")
    return {
        "available": True,
        "winner": winner,
        "pickleball_cost": round(float(pickleball_cost), 6),
        "tennis_cost": round(float(tennis_cost), 6),
        "margin": round(float(margin), 6),
        "components": components,
    }
