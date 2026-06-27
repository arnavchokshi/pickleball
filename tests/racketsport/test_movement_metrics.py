from __future__ import annotations

import math

import pytest

from threed.racketsport.biomech import balance_score_metric, x_factor_angle_metric
from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.movement_metrics import inter_player_spacing_metric, nvz_margin_metric
from threed.racketsport.schemas import MetricValue


def test_nvz_margin_metric_is_positive_outside_kitchen_and_negative_inside() -> None:
    outside_near = nvz_margin_metric([0.0, -9.5 * FT_TO_M], conf=0.86)
    inside_far = nvz_margin_metric([0.0, 3.0 * FT_TO_M], conf=0.71)

    assert outside_near == {
        "nvz_margin_ft": {
            "value": pytest.approx(2.5),
            "conf": pytest.approx(0.86),
            "unit": "ft",
            "gated": False,
            "source": "cpu_world_foot_point",
        }
    }
    assert inside_far["nvz_margin_ft"]["value"] == pytest.approx(-4.0)
    assert inside_far["nvz_margin_ft"]["conf"] == pytest.approx(0.71)

    MetricValue(**outside_near["nvz_margin_ft"])


def test_inter_player_spacing_metric_reports_planar_distance_in_feet() -> None:
    metric = inter_player_spacing_metric([0.0, 0.0], [3.0 * FT_TO_M, 4.0 * FT_TO_M], conf=0.92)

    assert metric == {
        "inter_player_spacing_ft": {
            "value": pytest.approx(5.0),
            "conf": pytest.approx(0.92),
            "unit": "ft",
            "gated": False,
            "source": "cpu_world_foot_points",
        }
    }
    MetricValue(**metric["inter_player_spacing_ft"])


def test_balance_score_metric_scores_center_of_mass_against_support_proxy() -> None:
    centered = balance_score_metric(
        center_of_mass_xy=[0.0, 0.0],
        support_points_xy=[[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [-0.5, 0.3]],
        conf=0.88,
    )
    offset = balance_score_metric(
        center_of_mass_xy=[1.0, 0.0],
        support_points_xy=[[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [-0.5, 0.3]],
        conf=0.77,
    )

    assert centered == {
        "balance_score": {
            "value": pytest.approx(1.0),
            "conf": pytest.approx(0.88),
            "unit": "score",
            "gated": False,
            "source": "cpu_com_support_proxy",
        }
    }
    assert offset["balance_score"]["value"] == pytest.approx(0.5)
    assert offset["balance_score"]["conf"] == pytest.approx(0.77)
    MetricValue(**centered["balance_score"])


def test_x_factor_angle_metric_returns_signed_smallest_angle_between_axes() -> None:
    open_torso = x_factor_angle_metric(shoulder_vector_xy=[0.0, 1.0], hip_vector_xy=[1.0, 0.0], conf=0.93)
    closed_torso = x_factor_angle_metric(shoulder_vector_xy=[1.0, 0.0], hip_vector_xy=[0.0, 1.0], conf=0.81)

    assert open_torso == {
        "x_factor_deg": {
            "value": pytest.approx(90.0),
            "conf": pytest.approx(0.93),
            "unit": "deg",
            "gated": False,
            "source": "cpu_shoulder_hip_vectors",
        }
    }
    assert closed_torso["x_factor_deg"]["value"] == pytest.approx(-90.0)
    assert closed_torso["x_factor_deg"]["conf"] == pytest.approx(0.81)
    MetricValue(**open_torso["x_factor_deg"])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: nvz_margin_metric([math.nan, 0.0]), "finite"),
        (lambda: inter_player_spacing_metric([0.0, 0.0], [1.0]), "world_xy"),
        (lambda: balance_score_metric([0.0, 0.0], [[0.0, 0.0]]), "support_points_xy"),
        (lambda: x_factor_angle_metric([0.0, 0.0], [1.0, 0.0]), "non-zero"),
    ],
)
def test_metric_primitives_fail_closed_on_invalid_geometry(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
