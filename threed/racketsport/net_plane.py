"""Regulation net-plane construction."""

from __future__ import annotations

from .court_templates import Sport, get_court_template
from .court_calibration import project_world_points
from .schemas import CourtCalibration, NetPlane


def build_net_plane(sport: Sport) -> NetPlane:
    template = get_court_template(sport)
    half_net_width_m = template.net_width_m / 2.0
    post_height_m = template.post_net_height_m

    return NetPlane(
        schema_version=1,
        plane={"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
        endpoints=(
            [-half_net_width_m, 0.0, post_height_m],
            [half_net_width_m, 0.0, post_height_m],
        ),
        center_height_in=template.center_net_height_in,
        post_height_in=template.post_net_height_in,
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
