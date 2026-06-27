"""Court zone geometry for NVZ, transition, baseline, and service boxes."""

from __future__ import annotations

from .court_templates import Sport, get_court_template, rectangle_xy_ft
from .schemas import CourtZones


def build_court_zones(sport: Sport) -> CourtZones:
    template = get_court_template(sport)

    if sport == "pickleball":
        half_width = template.half_width_ft
        half_length = template.half_length_ft
        nvz = template.non_volley_zone_ft
        if nvz is None:
            raise ValueError("Pickleball template missing non-volley zone distance")

        zones = {
            "court": rectangle_xy_ft(-half_width, -half_length, half_width, half_length),
            "near_nvz": rectangle_xy_ft(-half_width, -nvz, half_width, 0.0),
            "far_nvz": rectangle_xy_ft(-half_width, 0.0, half_width, nvz),
            "near_left_service": rectangle_xy_ft(-half_width, -half_length, 0.0, -nvz),
            "near_right_service": rectangle_xy_ft(0.0, -half_length, half_width, -nvz),
            "far_left_service": rectangle_xy_ft(-half_width, nvz, 0.0, half_length),
            "far_right_service": rectangle_xy_ft(0.0, nvz, half_width, half_length),
        }
        return CourtZones(schema_version=1, zones=zones)

    if sport == "tennis":
        half_width = template.half_width_ft
        half_length = template.half_length_ft
        half_singles_width = (template.singles_width_ft or template.width_ft) / 2.0
        service = template.service_line_distance_ft
        if service is None:
            raise ValueError("Tennis template missing service line distance")

        zones = {
            "court": rectangle_xy_ft(-half_width, -half_length, half_width, half_length),
            "singles_court": rectangle_xy_ft(-half_singles_width, -half_length, half_singles_width, half_length),
            "near_left_service": rectangle_xy_ft(-half_singles_width, -service, 0.0, 0.0),
            "near_right_service": rectangle_xy_ft(0.0, -service, half_singles_width, 0.0),
            "far_left_service": rectangle_xy_ft(-half_singles_width, 0.0, 0.0, service),
            "far_right_service": rectangle_xy_ft(0.0, 0.0, half_singles_width, service),
            "left_doubles_alley": rectangle_xy_ft(-half_width, -half_length, -half_singles_width, half_length),
            "right_doubles_alley": rectangle_xy_ft(half_singles_width, -half_length, half_width, half_length),
        }
        return CourtZones(schema_version=1, zones=zones)

    raise ValueError(f"Unsupported sport: {sport}")


def classify_point(sport: Sport, world_xy: list[float]) -> str | None:
    zones = build_court_zones(sport).zones
    broad_zones = {"court", "singles_court"}
    for name, polygon in zones.items():
        if name in broad_zones:
            continue
        if _point_in_polygon(world_xy, polygon):
            return name
    for name in ("singles_court", "court"):
        polygon = zones.get(name)
        if polygon is not None and _point_in_polygon(world_xy, polygon):
            return name
    return None


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    for idx in range(count):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % count]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
