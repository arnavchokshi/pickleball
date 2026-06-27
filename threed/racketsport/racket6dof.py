"""CPU-only racket and paddle 6DoF pose primitives.

This module intentionally contains deterministic validation and container
helpers only. It does not run a detector, SAM2, PnP, GigaPose, or a real UKF.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


INCH_TO_CM = 2.54


@dataclass(frozen=True)
class PaddleDimensions:
    """Validated paddle face dimensions in inches."""

    length_in: float
    width_in: float

    def __post_init__(self) -> None:
        length_in = _require_finite_float(self.length_in, "paddle_dims_in.length")
        width_in = _require_finite_float(self.width_in, "paddle_dims_in.width")
        if length_in <= 0.0 or width_in <= 0.0:
            raise ValueError("paddle_dims_in values must be positive")
        object.__setattr__(self, "length_in", length_in)
        object.__setattr__(self, "width_in", width_in)

    @property
    def length_cm(self) -> float:
        return self.length_in * INCH_TO_CM

    @property
    def width_cm(self) -> float:
        return self.width_in * INCH_TO_CM


@dataclass(frozen=True)
class SE3PoseConfidence:
    """A minimal SE3 pose plus confidence placeholder for a future UKF stage."""

    R: Sequence[Sequence[float]]
    t: Sequence[float]
    confidence: float
    source: str = "ukf_placeholder"

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _validate_rotation_matrix(self.R))
        object.__setattr__(self, "t", _validate_vector(self.t, "t", length=3))
        confidence = _require_finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "confidence", confidence)


def validate_paddle_dimensions(paddle_dims_in: Mapping[str, float] | PaddleDimensions) -> PaddleDimensions:
    """Return normalized paddle dimensions from length/width or h/w inch keys."""

    if isinstance(paddle_dims_in, PaddleDimensions):
        return paddle_dims_in

    has_named_dims = "length" in paddle_dims_in and "width" in paddle_dims_in
    has_short_dims = "h" in paddle_dims_in and "w" in paddle_dims_in
    if not has_named_dims and not has_short_dims:
        raise ValueError("paddle_dims_in must include length/width or h/w")

    if has_named_dims:
        length_in = paddle_dims_in["length"]
        width_in = paddle_dims_in["width"]
    else:
        length_in = paddle_dims_in["h"]
        width_in = paddle_dims_in["w"]

    return PaddleDimensions(length_in=length_in, width_in=width_in)


def normalize_face_normal(face_normal: Sequence[float]) -> tuple[float, float, float]:
    """Return a unit face-normal vector from a 3-vector."""

    x, y, z = _validate_vector(face_normal, "face_normal", length=3)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("face_normal must be non-zero")
    return (x / norm, y / norm, z / norm)


def validate_contact_point_face_cm(
    contact_point_face_cm: Sequence[float],
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
) -> tuple[float, float]:
    """Validate a face-local contact point against paddle dimensions.

    Coordinates are centered on the paddle face: x spans paddle width and y
    spans paddle length, both measured in centimeters.
    """

    x_cm, y_cm = _validate_vector(contact_point_face_cm, "contact_point_face_cm", length=2)
    dims = validate_paddle_dimensions(paddle_dims_in)
    half_width_cm = dims.width_cm / 2.0
    half_length_cm = dims.length_cm / 2.0
    epsilon = 1e-9

    if abs(x_cm) > half_width_cm + epsilon:
        raise ValueError("contact_point_face_cm x coordinate exceeds paddle width")
    if abs(y_cm) > half_length_cm + epsilon:
        raise ValueError("contact_point_face_cm y coordinate exceeds paddle length")
    return (x_cm, y_cm)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_vector(values: Sequence[float], name: str, *, length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a {length}-vector")
    try:
        vector = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a {length}-vector") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return tuple(_require_finite_float(value, f"{name}/{index}") for index, value in enumerate(vector))


def _validate_rotation_matrix(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    if isinstance(rows, (str, bytes)):
        raise ValueError("R must be a 3x3 matrix")
    try:
        matrix = tuple(tuple(row) for row in rows)
    except TypeError as exc:
        raise ValueError("R must be a 3x3 matrix") from exc
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("R must be a 3x3 matrix")
    return tuple(
        tuple(_require_finite_float(value, f"R/{row_index}/{col_index}") for col_index, value in enumerate(row))
        for row_index, row in enumerate(matrix)
    )


__all__ = [
    "INCH_TO_CM",
    "PaddleDimensions",
    "SE3PoseConfidence",
    "normalize_face_normal",
    "validate_contact_point_face_cm",
    "validate_paddle_dimensions",
]
