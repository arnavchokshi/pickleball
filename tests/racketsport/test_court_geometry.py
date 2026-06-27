from __future__ import annotations

import pytest

from threed.racketsport.court_templates import FT_TO_M, get_court_template
from threed.racketsport.court_zones import build_court_zones
from threed.racketsport.net_plane import build_net_plane, net_top_height_m_at_x
from threed.racketsport.schemas import CourtZones, NetPlane


def _ft(value_m: float) -> float:
    return value_m / FT_TO_M


def _assert_xy_ft(point: list[float], expected_x_ft: float, expected_y_ft: float) -> None:
    assert _ft(point[0]) == pytest.approx(expected_x_ft)
    assert _ft(point[1]) == pytest.approx(expected_y_ft)


def test_pickleball_template_matches_regulation_dimensions():
    template = get_court_template("pickleball")

    assert template.length_ft == pytest.approx(44.0)
    assert template.width_ft == pytest.approx(20.0)
    assert template.net_width_ft == pytest.approx(22.0)
    assert template.non_volley_zone_ft == pytest.approx(7.0)
    assert template.center_net_height_in == pytest.approx(34.0)
    assert template.post_net_height_in == pytest.approx(36.0)
    assert template.coordinate_frame == "origin_net_center_x_width_y_length_z_up_m"
    assert len(template.corners_m) == 4
    _assert_xy_ft(template.corners_m[0], -10.0, -22.0)
    _assert_xy_ft(template.corners_m[2], 10.0, 22.0)


def test_tennis_template_matches_doubles_and_singles_geometry():
    template = get_court_template("tennis")

    assert template.length_ft == pytest.approx(78.0)
    assert template.width_ft == pytest.approx(36.0)
    assert template.singles_width_ft == pytest.approx(27.0)
    assert template.service_line_distance_ft == pytest.approx(21.0)
    assert template.net_post_offset_ft == pytest.approx(3.0)
    assert template.net_width_ft == pytest.approx(42.0)
    assert template.center_net_height_in == pytest.approx(36.0)
    assert template.post_net_height_in == pytest.approx(42.0)
    _assert_xy_ft(template.corners_m[0], -18.0, -39.0)
    _assert_xy_ft(template.corners_m[2], 18.0, 39.0)


def test_pickleball_zones_include_nvz_and_service_boxes():
    zones = build_court_zones("pickleball")

    assert isinstance(zones, CourtZones)
    assert set(zones.zones) == {
        "court",
        "far_left_service",
        "far_nvz",
        "far_right_service",
        "near_left_service",
        "near_nvz",
        "near_right_service",
    }
    _assert_xy_ft(zones.zones["court"][0], -10.0, -22.0)
    _assert_xy_ft(zones.zones["court"][2], 10.0, 22.0)
    _assert_xy_ft(zones.zones["near_nvz"][0], -10.0, -7.0)
    _assert_xy_ft(zones.zones["near_nvz"][2], 10.0, 0.0)
    _assert_xy_ft(zones.zones["near_left_service"][0], -10.0, -22.0)
    _assert_xy_ft(zones.zones["near_left_service"][2], 0.0, -7.0)
    _assert_xy_ft(zones.zones["far_right_service"][0], 0.0, 7.0)
    _assert_xy_ft(zones.zones["far_right_service"][2], 10.0, 22.0)


def test_tennis_zones_include_service_boxes_and_alleys():
    zones = build_court_zones("tennis")

    assert isinstance(zones, CourtZones)
    assert {
        "court",
        "singles_court",
        "near_left_service",
        "near_right_service",
        "far_left_service",
        "far_right_service",
        "left_doubles_alley",
        "right_doubles_alley",
    } <= set(zones.zones)
    _assert_xy_ft(zones.zones["court"][0], -18.0, -39.0)
    _assert_xy_ft(zones.zones["singles_court"][0], -13.5, -39.0)
    _assert_xy_ft(zones.zones["near_left_service"][0], -13.5, -21.0)
    _assert_xy_ft(zones.zones["near_left_service"][2], 0.0, 0.0)
    _assert_xy_ft(zones.zones["far_right_service"][0], 0.0, 0.0)
    _assert_xy_ft(zones.zones["far_right_service"][2], 13.5, 21.0)


def test_net_planes_use_vertical_net_plane_and_top_cable_heights():
    pickleball = build_net_plane("pickleball")
    tennis = build_net_plane("tennis")

    assert isinstance(pickleball, NetPlane)
    assert pickleball.plane.point == [0.0, 0.0, 0.0]
    assert pickleball.plane.normal == [0.0, 1.0, 0.0]
    assert _ft(pickleball.endpoints[0][0]) == pytest.approx(-11.0)
    assert _ft(pickleball.endpoints[1][0]) == pytest.approx(11.0)
    assert pickleball.center_height_in == pytest.approx(34.0)
    assert pickleball.post_height_in == pytest.approx(36.0)
    assert net_top_height_m_at_x("pickleball", 0.0) == pytest.approx(34.0 * 0.0254)

    assert _ft(tennis.endpoints[0][0]) == pytest.approx(-21.0)
    assert _ft(tennis.endpoints[1][0]) == pytest.approx(21.0)
    assert tennis.center_height_in == pytest.approx(36.0)
    assert tennis.post_height_in == pytest.approx(42.0)
    assert net_top_height_m_at_x("tennis", tennis.endpoints[1][0]) == pytest.approx(42.0 * 0.0254)


def test_court_geometry_rejects_unknown_sports():
    with pytest.raises(ValueError, match="Unsupported sport"):
        get_court_template("badminton")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported sport"):
        build_court_zones("badminton")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported sport"):
        build_net_plane("badminton")  # type: ignore[arg-type]
