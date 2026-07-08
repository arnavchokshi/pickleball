"""Regulation net-plane construction."""

from __future__ import annotations

from .court_templates import Sport, get_court_template, in_to_m
from .court_calibration import project_world_points
from .schemas import CourtCalibration, NetPlane


def build_net_plane(
    sport: Sport,
    *,
    post_height_in: float | None = None,
    center_height_in: float | None = None,
) -> NetPlane:
    template = get_court_template(sport)
    half_net_width_m = template.net_width_m / 2.0
    resolved_post_height_in = template.post_net_height_in if post_height_in is None else float(post_height_in)
    resolved_center_height_in = template.center_net_height_in if center_height_in is None else float(center_height_in)
    post_height_m = in_to_m(resolved_post_height_in)

    return NetPlane(
        schema_version=1,
        plane={"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
        endpoints=(
            [-half_net_width_m, 0.0, post_height_m],
            [half_net_width_m, 0.0, post_height_m],
        ),
        center_height_in=resolved_center_height_in,
        post_height_in=resolved_post_height_in,
    )


def net_plane_from_template(calibration: CourtCalibration) -> NetPlane:
    """Build the regulation net plane for a solved calibration's sport."""

    return build_net_plane(calibration.sport)


def net_top_height_m_at_x(sport: Sport, x_m: float) -> float:
    """Approximate the top cable as symmetric linear sag from post to center."""

    template = get_court_template(sport)
    half_net_width_m = template.net_width_m / 2.0
    if half_net_width_m <= 0:
        raise ValueError(f"Invalid net width for sport: {sport}")

    clamped_ratio = min(abs(x_m) / half_net_width_m, 1.0)
    return template.center_net_height_m + (template.post_net_height_m - template.center_net_height_m) * clamped_ratio


def project_net_plane(calibration: CourtCalibration, net_plane: NetPlane) -> dict[str, list[float]]:
    center = [0.0, 0.0, net_plane.center_height_in * 0.0254]
    left, right, center_projected = project_world_points(
        calibration.extrinsics,
        calibration.intrinsics,
        [net_plane.endpoints[0], net_plane.endpoints[1], center],
    )
    return {"left_post": left, "right_post": right, "center": center_projected}
