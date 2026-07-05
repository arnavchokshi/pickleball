"""Assisted court proposal seed constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CourtAssistSeed:
    mode: str
    points: tuple[tuple[float, float], ...]
    image_size: tuple[int, int]
    line_label: str | None = None

    @classmethod
    def one_inside_tap(
        cls,
        point: tuple[float, float],
        *,
        image_size: tuple[int, int],
    ) -> "CourtAssistSeed":
        return cls(mode="one_inside_tap", points=(point,), image_size=image_size)

    @classmethod
    def two_line_taps(
        cls,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        *,
        line_label: str,
        image_size: tuple[int, int],
    ) -> "CourtAssistSeed":
        if not line_label:
            raise ValueError("line_label is required for two_line_taps")
        return cls(
            mode="two_line_taps",
            points=(point_a, point_b),
            image_size=image_size,
            line_label=line_label,
        )

    def to_constraints(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode,
            "tap_points": [[float(x), float(y)] for x, y in self.points],
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "line_label": self.line_label,
            "trusted_calibration": False,
        }
        if self.mode == "one_inside_tap":
            payload["target_region_contains"] = [
                float(self.points[0][0]),
                float(self.points[0][1]),
            ]
        return payload
