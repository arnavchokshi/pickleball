"""Regulation court geometry templates for pickleball and tennis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Vector2 = list[float]
Vector3 = list[float]
Sport = Literal["pickleball", "tennis"]

FT_TO_M = 0.3048
IN_TO_M = 0.0254
COORDINATE_FRAME = "origin_net_center_x_width_y_length_z_up_m"


def ft_to_m(value_ft: float) -> float:
    return value_ft * FT_TO_M


def in_to_m(value_in: float) -> float:
    return value_in * IN_TO_M


def xy_ft_to_m(x_ft: float, y_ft: float) -> Vector2:
    return [ft_to_m(x_ft), ft_to_m(y_ft)]


def xyz_ft_to_m(x_ft: float, y_ft: float, z_ft: float = 0.0) -> Vector3:
    return [ft_to_m(x_ft), ft_to_m(y_ft), ft_to_m(z_ft)]


def rectangle_xy_ft(x_min_ft: float, y_min_ft: float, x_max_ft: float, y_max_ft: float) -> list[Vector2]:
    """Return a rectangular polygon without repeating the first point."""

    return [
        xy_ft_to_m(x_min_ft, y_min_ft),
        xy_ft_to_m(x_max_ft, y_min_ft),
        xy_ft_to_m(x_max_ft, y_max_ft),
        xy_ft_to_m(x_min_ft, y_max_ft),
    ]


@dataclass(frozen=True)
class CourtTemplate:
    sport: Sport
    length_ft: float
    width_ft: float
    net_width_ft: float
    center_net_height_in: float
    post_net_height_in: float
    singles_width_ft: float | None = None
    service_line_distance_ft: float | None = None
    non_volley_zone_ft: float | None = None
    net_post_offset_ft: float = 0.0
    coordinate_frame: str = COORDINATE_FRAME

    @property
    def half_length_ft(self) -> float:
        return self.length_ft / 2.0

    @property
    def half_width_ft(self) -> float:
        return self.width_ft / 2.0

    @property
    def half_net_width_ft(self) -> float:
        return self.net_width_ft / 2.0

    @property
    def length_m(self) -> float:
        return ft_to_m(self.length_ft)

    @property
    def width_m(self) -> float:
        return ft_to_m(self.width_ft)

    @property
    def net_width_m(self) -> float:
        return ft_to_m(self.net_width_ft)

    @property
    def center_net_height_m(self) -> float:
        return in_to_m(self.center_net_height_in)

    @property
    def post_net_height_m(self) -> float:
        return in_to_m(self.post_net_height_in)

    @property
    def corners_m(self) -> list[Vector3]:
        return [
            xyz_ft_to_m(-self.half_width_ft, -self.half_length_ft),
            xyz_ft_to_m(self.half_width_ft, -self.half_length_ft),
            xyz_ft_to_m(self.half_width_ft, self.half_length_ft),
            xyz_ft_to_m(-self.half_width_ft, self.half_length_ft),
        ]

    @property
    def line_segments_m(self) -> dict[str, tuple[Vector3, Vector3]]:
        lines = {
            "near_baseline": (
                xyz_ft_to_m(-self.half_width_ft, -self.half_length_ft),
                xyz_ft_to_m(self.half_width_ft, -self.half_length_ft),
            ),
            "far_baseline": (
                xyz_ft_to_m(-self.half_width_ft, self.half_length_ft),
                xyz_ft_to_m(self.half_width_ft, self.half_length_ft),
            ),
            "left_sideline": (
                xyz_ft_to_m(-self.half_width_ft, -self.half_length_ft),
                xyz_ft_to_m(-self.half_width_ft, self.half_length_ft),
            ),
            "right_sideline": (
                xyz_ft_to_m(self.half_width_ft, -self.half_length_ft),
                xyz_ft_to_m(self.half_width_ft, self.half_length_ft),
            ),
            "net": (
                xyz_ft_to_m(-self.half_net_width_ft, 0.0),
                xyz_ft_to_m(self.half_net_width_ft, 0.0),
            ),
        }
        if self.non_volley_zone_ft is not None:
            lines["near_nvz"] = (
                xyz_ft_to_m(-self.half_width_ft, -self.non_volley_zone_ft),
                xyz_ft_to_m(self.half_width_ft, -self.non_volley_zone_ft),
            )
            lines["far_nvz"] = (
                xyz_ft_to_m(-self.half_width_ft, self.non_volley_zone_ft),
                xyz_ft_to_m(self.half_width_ft, self.non_volley_zone_ft),
            )
            if self.sport == "pickleball":
                lines["near_centerline"] = (
                    xyz_ft_to_m(0.0, -self.half_length_ft),
                    xyz_ft_to_m(0.0, -self.non_volley_zone_ft),
                )
                lines["far_centerline"] = (
                    xyz_ft_to_m(0.0, self.non_volley_zone_ft),
                    xyz_ft_to_m(0.0, self.half_length_ft),
                )
        if self.service_line_distance_ft is not None:
            service_half_width_ft = (self.singles_width_ft or self.width_ft) / 2.0
            lines["near_service_line"] = (
                xyz_ft_to_m(-service_half_width_ft, -self.service_line_distance_ft),
                xyz_ft_to_m(service_half_width_ft, -self.service_line_distance_ft),
            )
            lines["far_service_line"] = (
                xyz_ft_to_m(-service_half_width_ft, self.service_line_distance_ft),
                xyz_ft_to_m(service_half_width_ft, self.service_line_distance_ft),
            )
        return lines


COURT_TEMPLATES: dict[Sport, CourtTemplate] = {
    "pickleball": CourtTemplate(
        sport="pickleball",
        length_ft=44.0,
        width_ft=20.0,
        net_width_ft=22.0,
        center_net_height_in=34.0,
        post_net_height_in=36.0,
        non_volley_zone_ft=7.0,
        net_post_offset_ft=1.0,
    ),
    "tennis": CourtTemplate(
        sport="tennis",
        length_ft=78.0,
        width_ft=36.0,
        singles_width_ft=27.0,
        net_width_ft=42.0,
        center_net_height_in=36.0,
        post_net_height_in=42.0,
        service_line_distance_ft=21.0,
        net_post_offset_ft=3.0,
    ),
}


def get_court_template(sport: Sport) -> CourtTemplate:
    try:
        return COURT_TEMPLATES[sport]
    except KeyError as exc:
        raise ValueError(f"Unsupported sport: {sport}") from exc
