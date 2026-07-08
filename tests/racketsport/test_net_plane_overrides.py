from __future__ import annotations

import pytest

from threed.racketsport.net_plane import build_net_plane, net_top_height_m_at_x


def test_build_net_plane_uses_tape_measured_height_overrides() -> None:
    plane = build_net_plane("pickleball", post_height_in=35.5, center_height_in=33.5)

    assert plane.center_height_in == pytest.approx(33.5)
    assert plane.post_height_in == pytest.approx(35.5)
    assert plane.endpoints[0][2] == pytest.approx(35.5 * 0.0254)
    assert plane.endpoints[1][2] == pytest.approx(35.5 * 0.0254)
    assert [0.0, 0.0, plane.center_height_in * 0.0254][2] == pytest.approx(33.5 * 0.0254)


def test_build_net_plane_default_payload_is_unchanged_without_overrides() -> None:
    default_plane = build_net_plane("pickleball")
    explicit_none_plane = build_net_plane("pickleball", post_height_in=None, center_height_in=None)

    assert explicit_none_plane.model_dump(mode="json") == default_plane.model_dump(mode="json")
    assert net_top_height_m_at_x("pickleball", 0.0) == pytest.approx(34.0 * 0.0254)
