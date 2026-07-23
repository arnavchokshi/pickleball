"""Deterministic structured floor-keypoint evidence from court heatmaps.

The learned court model has 15 canonical channels, but three of those channels describe the
physical *top* of the net.  A planar court solver must never consume those elevated points as
floor observations.  This module therefore emits only the 12 canonical Z=0 court points and
keeps the filtering structural rather than relying on caller discipline.

Inputs are per-keypoint spatial probability maps plus per-keypoint visibility probabilities.
Each output record is composed only of JSON-native values and carries two separated subpixel
peak hypotheses, uncertainty diagnostics, a local covariance estimate, and a conservative raw
confidence.  No confidence field is a promotion or calibration-accuracy claim: the formula is a
fixed evidence combiner intended for downstream weighting and later empirical calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


NET_TOP_KEYPOINT_NAMES: frozenset[str] = frozenset(
    {"net_left_sideline", "net_center", "net_right_sideline"}
)
CANONICAL_FLOOR_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    point.name for point in PICKLEBALL_KEYPOINTS if point.name not in NET_TOP_KEYPOINT_NAMES
)
_CANONICAL_BY_NAME = {point.name: point for point in PICKLEBALL_KEYPOINTS}

DEFAULT_MIN_PEAK_SEPARATION_HEATMAP_PX = 2.0
DEFAULT_COVARIANCE_RADIUS_HEATMAP_PX = 2.0
DEFAULT_COVARIANCE_FLOOR_HEATMAP_PX2 = 0.25

__all__ = [
    "CANONICAL_FLOOR_KEYPOINT_NAMES",
    "DEFAULT_COVARIANCE_FLOOR_HEATMAP_PX2",
    "DEFAULT_COVARIANCE_RADIUS_HEATMAP_PX",
    "DEFAULT_MIN_PEAK_SEPARATION_HEATMAP_PX",
    "NET_TOP_KEYPOINT_NAMES",
    "extract_court_structured_evidence",
]


def extract_court_structured_evidence(
    heatmap_probabilities: Mapping[str, Any],
    visibility: Mapping[str, Any],
    *,
    image_size: Sequence[int] | None = None,
    source_size: Sequence[int] | None = None,
    min_peak_separation_heatmap_px: float = DEFAULT_MIN_PEAK_SEPARATION_HEATMAP_PX,
    covariance_radius_heatmap_px: float = DEFAULT_COVARIANCE_RADIUS_HEATMAP_PX,
    covariance_floor_heatmap_px2: float = DEFAULT_COVARIANCE_FLOOR_HEATMAP_PX2,
) -> list[dict[str, Any]]:
    """Extract canonical floor observations in deterministic taxonomy order.

    ``heatmap_probabilities`` may contain the canonical 15 channels, an auxiliary architecture's
    extra channels, or a partial canonical mapping.  Only canonical floor names are emitted;
    auxiliary and net-top channels are ignored.  Every present floor channel must have a matching
    visibility probability in ``[0, 1]``.

    Sizes use ``[width, height]`` ordering.  Heatmap coordinates are scaled independently to the
    model-input image and original source sizes.  ``observation_xy`` and ``covariance_px2`` use
    source pixels when ``source_size`` is provided, image pixels when only ``image_size`` is
    provided, and heatmap pixels otherwise.
    """

    if not isinstance(heatmap_probabilities, Mapping):
        raise ValueError("heatmap_probabilities must be a keypoint mapping")
    if not isinstance(visibility, Mapping):
        raise ValueError("visibility must be a keypoint mapping")
    separation = _positive_float(
        min_peak_separation_heatmap_px,
        "min_peak_separation_heatmap_px",
    )
    covariance_radius = _positive_float(
        covariance_radius_heatmap_px,
        "covariance_radius_heatmap_px",
    )
    covariance_floor = _positive_float(
        covariance_floor_heatmap_px2,
        "covariance_floor_heatmap_px2",
    )
    parsed_image_size = _optional_size(image_size, "image_size")
    parsed_source_size = _optional_size(source_size, "source_size")

    present_floor_names = [
        name for name in CANONICAL_FLOOR_KEYPOINT_NAMES if name in heatmap_probabilities
    ]
    if not present_floor_names:
        raise ValueError("no canonical floor heatmap channels were provided")

    records: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    for canonical_index, name in enumerate(CANONICAL_FLOOR_KEYPOINT_NAMES):
        if name not in heatmap_probabilities:
            continue
        if name not in visibility:
            raise ValueError(f"visibility is missing canonical floor keypoint {name}")
        probability = _normalized_probability_map(heatmap_probabilities[name], name=name)
        height, width = probability.shape
        if expected_shape is None:
            expected_shape = (height, width)
        elif (height, width) != expected_shape:
            raise ValueError(
                "all canonical floor probability maps must share one shape; "
                f"expected {expected_shape}, got {(height, width)} for {name}"
            )
        visible_probability = _unit_float(visibility[name], f"visibility.{name}")
        primary, secondary = _two_separated_peaks(probability, min_separation=separation)
        normalized_entropy = _normalized_entropy(probability)
        peak_margin = max(0.0, primary["probability"] - secondary["probability"])
        relative_peak_margin = peak_margin / max(primary["probability"], 1.0e-12)
        raw_confidence = visible_probability * primary["probability"]
        calibrated_raw_confidence = (
            raw_confidence
            * max(0.0, 1.0 - normalized_entropy)
            * max(0.0, min(1.0, relative_peak_margin))
        )

        heatmap_covariance = _local_peak_covariance(
            probability,
            center_xy=primary["xy"],
            radius=covariance_radius,
            variance_floor=covariance_floor,
        )
        image_scale = _scale_for_size(parsed_image_size, heatmap_width=width, heatmap_height=height)
        source_scale = _scale_for_size(parsed_source_size, heatmap_width=width, heatmap_height=height)
        if source_scale is not None:
            observation_scale = source_scale
            coordinate_space = "source_pixels"
        elif image_scale is not None:
            observation_scale = image_scale
            coordinate_space = "image_pixels"
        else:
            observation_scale = (1.0, 1.0)
            coordinate_space = "heatmap_pixels"
        covariance_px2 = _scale_covariance(heatmap_covariance, observation_scale)

        point = _CANONICAL_BY_NAME[name]
        records.append(
            {
                "schema_version": 1,
                "observation_type": "canonical_court_floor_keypoint_heatmap",
                "keypoint_name": name,
                "canonical_floor_index": canonical_index,
                "world_xy_m": [float(point.world_xyz_m[0]), float(point.world_xyz_m[1])],
                "coordinate_space": coordinate_space,
                "heatmap_size": [width, height],
                "image_size": None if parsed_image_size is None else list(parsed_image_size),
                "source_size": None if parsed_source_size is None else list(parsed_source_size),
                "observation_xy": _scaled_xy(primary["xy"], observation_scale),
                "primary_peak": _peak_record(
                    primary,
                    image_scale=image_scale,
                    source_scale=source_scale,
                ),
                "secondary_peak": _peak_record(
                    secondary,
                    image_scale=image_scale,
                    source_scale=source_scale,
                ),
                "peak_separation_heatmap_px": float(
                    math.hypot(
                        primary["xy"][0] - secondary["xy"][0],
                        primary["xy"][1] - secondary["xy"][1],
                    )
                ),
                "primary_probability": float(primary["probability"]),
                "secondary_probability": float(secondary["probability"]),
                "normalized_entropy": float(normalized_entropy),
                "peak_margin": float(peak_margin),
                "relative_peak_margin": float(relative_peak_margin),
                "visibility": float(visible_probability),
                "raw_confidence": float(raw_confidence),
                "calibrated_raw_confidence": float(calibrated_raw_confidence),
                "confidence": float(calibrated_raw_confidence),
                "confidence_formula": (
                    "visibility * primary_probability * (1 - normalized_entropy) "
                    "* relative_peak_margin"
                ),
                "covariance_px2": covariance_px2,
                "covariance_policy": {
                    "kind": "local_probability_second_moment",
                    "radius_heatmap_px": float(covariance_radius),
                    "variance_floor_heatmap_px2": float(covariance_floor),
                    "scaled_to": coordinate_space,
                },
                "provenance": {
                    "source": "court_keypoint_heatmap_probabilities_and_visibility",
                    "floor_only": True,
                    "net_top_channels_excluded": True,
                },
            }
        )
    return records


def _normalized_probability_map(value: Any, *, name: str) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"heatmap_probabilities.{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"heatmap_probabilities.{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"heatmap_probabilities.{name} must be non-negative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"heatmap_probabilities.{name} must have positive mass")
    return array / total


def _two_separated_peaks(probability: Any, *, min_separation: float) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    primary_y, primary_x = np.unravel_index(int(np.argmax(probability)), probability.shape)
    primary = _decode_peak_at(probability, x=int(primary_x), y=int(primary_y))
    height, width = probability.shape
    candidates: list[tuple[float, int, int]] = []
    for y in range(height):
        for x in range(width):
            if x == primary_x and y == primary_y:
                continue
            if math.hypot(float(x) - primary["xy"][0], float(y) - primary["xy"][1]) < min_separation:
                continue
            candidates.append((float(probability[y, x]), y, x))
    # Probability descending, then row/column ascending: deterministic even for flat/tied maps.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, y, x in candidates:
        candidate = _decode_peak_at(probability, x=x, y=y)
        if math.hypot(
            candidate["xy"][0] - primary["xy"][0],
            candidate["xy"][1] - primary["xy"][1],
        ) >= min_separation:
            return primary, candidate
    raise ValueError(
        "probability map is too small to provide two subpixel peaks at the requested separation"
    )


def _decode_peak_at(probability: Any, *, x: int, y: int) -> dict[str, Any]:
    height, width = probability.shape
    x_offset = 0.0
    y_offset = 0.0
    if 0 < x < width - 1:
        x_offset = _parabolic_offset(
            float(probability[y, x - 1]),
            float(probability[y, x]),
            float(probability[y, x + 1]),
        )
    if 0 < y < height - 1:
        y_offset = _parabolic_offset(
            float(probability[y - 1, x]),
            float(probability[y, x]),
            float(probability[y + 1, x]),
        )
    return {
        "xy": (float(x) + x_offset, float(y) + y_offset),
        "discrete_xy": (x, y),
        "probability": float(probability[y, x]),
    }


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if denominator >= 0.0 or math.isclose(denominator, 0.0):
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (left - right) / denominator))


def _normalized_entropy(probability: Any) -> float:
    import numpy as np

    count = int(probability.size)
    if count <= 1:
        return 0.0
    positive = probability[probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return max(0.0, min(1.0, entropy / math.log(count)))


def _local_peak_covariance(
    probability: Any,
    *,
    center_xy: tuple[float, float],
    radius: float,
    variance_floor: float,
) -> list[list[float]]:
    import numpy as np

    height, width = probability.shape
    center_x, center_y = center_xy
    samples: list[tuple[float, float, float]] = []
    for y in range(height):
        for x in range(width):
            if math.hypot(float(x) - center_x, float(y) - center_y) <= radius:
                samples.append((float(x), float(y), float(probability[y, x])))
    total = sum(sample[2] for sample in samples)
    if total <= 0.0:
        covariance = np.eye(2, dtype=np.float64) * variance_floor
    else:
        covariance = np.zeros((2, 2), dtype=np.float64)
        for x, y, weight in samples:
            delta = np.asarray([x - center_x, y - center_y], dtype=np.float64)
            covariance += weight * np.outer(delta, delta)
        covariance /= total
        covariance = 0.5 * (covariance + covariance.T)
        values, vectors = np.linalg.eigh(covariance)
        covariance = vectors @ np.diag(np.maximum(values, variance_floor)) @ vectors.T
    return [
        [float(covariance[0, 0]), float(covariance[0, 1])],
        [float(covariance[1, 0]), float(covariance[1, 1])],
    ]


def _scale_covariance(
    covariance: list[list[float]],
    scale: tuple[float, float],
) -> list[list[float]]:
    sx, sy = scale
    return [
        [float(covariance[0][0] * sx * sx), float(covariance[0][1] * sx * sy)],
        [float(covariance[1][0] * sx * sy), float(covariance[1][1] * sy * sy)],
    ]


def _peak_record(
    peak: dict[str, Any],
    *,
    image_scale: tuple[float, float] | None,
    source_scale: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "heatmap_xy": [float(peak["xy"][0]), float(peak["xy"][1])],
        "heatmap_discrete_xy": [int(peak["discrete_xy"][0]), int(peak["discrete_xy"][1])],
        "image_xy": None if image_scale is None else _scaled_xy(peak["xy"], image_scale),
        "source_xy": None if source_scale is None else _scaled_xy(peak["xy"], source_scale),
        "probability": float(peak["probability"]),
    }


def _scaled_xy(xy: tuple[float, float], scale: tuple[float, float]) -> list[float]:
    return [float(xy[0] * scale[0]), float(xy[1] * scale[1])]


def _scale_for_size(
    size: tuple[int, int] | None,
    *,
    heatmap_width: int,
    heatmap_height: int,
) -> tuple[float, float] | None:
    if size is None:
        return None
    return (size[0] / float(heatmap_width), size[1] / float(heatmap_height))


def _optional_size(value: Sequence[int] | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{name} must contain positive integer width and height")
    return (width, height)


def _positive_float(value: Any, name: str) -> float:
    result = _numeric_scalar(value, name)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_float(value: Any, name: str) -> float:
    result = _positive_or_zero_float(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_or_zero_float(value: Any, name: str) -> float:
    result = _numeric_scalar(value, name)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _numeric_scalar(value: Any, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError):
            raise ValueError(f"{name} must be a numeric scalar") from None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    return float(value)
