from __future__ import annotations

from threed.racketsport.court_motion_mode import classify_motion_mode


def test_static_motion_mode_for_low_jitter() -> None:
    result = classify_motion_mode(frame_transforms=[{"translation_px": 0.5, "rotation_deg": 0.05, "inliers": 120}])

    assert result["motion_mode"] == "static"


def test_untrusted_motion_for_low_inliers() -> None:
    result = classify_motion_mode(frame_transforms=[{"translation_px": 12.0, "rotation_deg": 2.0, "inliers": 6}])

    assert result["motion_mode"] == "untrusted_motion"
    assert "low_inliers" in result["reasons"]
