from __future__ import annotations

from threed.racketsport.schemas import CaptureQuality


def score_capture_quality(
    *,
    corners_visible: int | None = None,
    reprojection_rmse_px: float | None = None,
    blur_laplacian_var: float | None = None,
    luminance_mean: float | None = None,
    luminance_std_fraction: float | None = None,
    fps: float | None = None,
    exposure_s: float | None = None,
    shake_rms_deg: float | None = None,
    arkit_tracking_state: str | None = None,
) -> CaptureQuality:
    """Score capture readiness from iOS/server measurable quality signals.

    The thresholds are intentionally conservative: poor framing or unstable
    capture should fail closed before downstream metrics pretend to be precise.
    """

    reasons: list[str] = []
    poor = False

    def warn(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def fail(reason: str) -> None:
        nonlocal poor
        poor = True
        warn(reason)

    if corners_visible is not None and corners_visible < 4:
        fail("court_corners_missing")

    if reprojection_rmse_px is not None:
        if reprojection_rmse_px > 10.0:
            fail("reprojection_unusable")
        elif reprojection_rmse_px > 5.0:
            warn("reprojection_high")

    if blur_laplacian_var is not None:
        if blur_laplacian_var < 40.0:
            fail("motion_blur_high")
        elif blur_laplacian_var < 120.0:
            warn("motion_blur_marginal")

    if luminance_mean is not None:
        if luminance_mean < 0.05 or luminance_mean > 0.95:
            fail("exposure_clipped")
        elif luminance_mean < 0.20 or luminance_mean > 0.85:
            warn("exposure_marginal")

    if luminance_std_fraction is not None:
        if luminance_std_fraction > 0.08:
            fail("luminance_pumping")
        elif luminance_std_fraction > 0.02:
            warn("luminance_unstable")

    if fps is not None:
        if fps < 55.0:
            fail("fps_below_floor")
        elif fps < 60.0:
            warn("fps_marginal")

    if exposure_s is not None:
        if exposure_s > 1 / 120:
            fail("shutter_too_slow")
        elif exposure_s > 1 / 500:
            warn("shutter_marginal")

    if shake_rms_deg is not None:
        if shake_rms_deg > 3.0:
            fail("camera_shake_high")
        elif shake_rms_deg > 1.0:
            warn("camera_shake_marginal")

    if arkit_tracking_state is not None and arkit_tracking_state != "normal":
        if arkit_tracking_state in {"limited", "failed"}:
            fail(f"arkit_tracking_{arkit_tracking_state}")
        else:
            warn(f"arkit_tracking_{arkit_tracking_state}")

    grade = "poor" if poor else "warn" if reasons else "good"
    return CaptureQuality(grade=grade, reasons=reasons)
