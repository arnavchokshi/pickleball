from __future__ import annotations

from threed.racketsport.court_assist_seed import CourtAssistSeed


def test_one_tap_seed_is_target_region_only() -> None:
    seed = CourtAssistSeed.one_inside_tap((100.0, 200.0), image_size=(1920, 1080))

    constraints = seed.to_constraints()

    assert constraints["mode"] == "one_inside_tap"
    assert constraints["target_region_contains"] == [100.0, 200.0]
    assert constraints["trusted_calibration"] is False


def test_two_line_taps_require_line_label() -> None:
    seed = CourtAssistSeed.two_line_taps(
        (10.0, 20.0),
        (300.0, 24.0),
        line_label="near_nvz",
        image_size=(1920, 1080),
    )

    constraints = seed.to_constraints()

    assert constraints["mode"] == "two_line_taps"
    assert constraints["line_label"] == "near_nvz"
    assert constraints["trusted_calibration"] is False
