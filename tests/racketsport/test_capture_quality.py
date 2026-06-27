from __future__ import annotations

from threed.racketsport.capture_quality import score_capture_quality


def test_score_capture_quality_accepts_clean_locked_capture():
    quality = score_capture_quality(
        corners_visible=4,
        reprojection_rmse_px=2.5,
        blur_laplacian_var=180.0,
        luminance_mean=0.48,
        luminance_std_fraction=0.01,
        fps=120.0,
        exposure_s=1 / 800,
        shake_rms_deg=0.4,
        arkit_tracking_state="normal",
    )

    assert quality.grade == "good"
    assert quality.reasons == []


def test_score_capture_quality_fails_closed_for_unusable_framing_and_shutter():
    quality = score_capture_quality(
        corners_visible=2,
        reprojection_rmse_px=15.0,
        blur_laplacian_var=20.0,
        luminance_mean=0.03,
        luminance_std_fraction=0.12,
        fps=30.0,
        exposure_s=1 / 60,
        shake_rms_deg=4.0,
        arkit_tracking_state="limited",
    )

    assert quality.grade == "poor"
    assert "court_corners_missing" in quality.reasons
    assert "shutter_too_slow" in quality.reasons
    assert "arkit_tracking_limited" in quality.reasons


def test_score_capture_quality_warns_on_marginal_locked_capture():
    quality = score_capture_quality(
        corners_visible=4,
        reprojection_rmse_px=6.5,
        blur_laplacian_var=95.0,
        luminance_mean=0.18,
        luminance_std_fraction=0.035,
        fps=58.5,
        exposure_s=1 / 320,
        shake_rms_deg=1.5,
        arkit_tracking_state="normal",
    )

    assert quality.grade == "warn"
    assert "reprojection_high" in quality.reasons
    assert "luminance_unstable" in quality.reasons
    assert "shutter_marginal" in quality.reasons
