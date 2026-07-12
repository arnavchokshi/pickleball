"""Strict additive sidecar for raw WASB ball-size observations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .io_decode import time_for_frame


HEATMAP_THRESHOLD = 0.5
RADIUS_PROXY_DEFINITION = "0.5 * sqrt(native_bbox_width_px * native_bbox_height_px)"


class WasbBallSizeBlob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center_xy_px: tuple[FiniteFloat, FiniteFloat]
    extent_xyxy_px: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    width_px: FiniteFloat = Field(gt=0.0)
    height_px: FiniteFloat = Field(gt=0.0)
    component_pixel_count: int = Field(ge=1)
    component_area_px2: FiniteFloat = Field(gt=0.0)
    heatmap_peak: FiniteFloat = Field(ge=0.0, le=1.0)
    heatmap_weight_sum: FiniteFloat = Field(gt=0.0)
    radius_proxy_px: FiniteFloat = Field(gt=0.0)
    source_detector: Literal["wasb_concomp"] = "wasb_concomp"


class WasbBallSizeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    pts_seconds: FiniteFloat = Field(ge=0.0)
    heatmap_peak: FiniteFloat = Field(ge=0.0, le=1.0)
    heatmap_observation_count: int = Field(ge=1)
    selected_heatmap_observation_index: int = Field(ge=0)
    blob_count: int = Field(ge=0)
    blobs: list[WasbBallSizeBlob] = Field(default_factory=list)

    @model_validator(mode="after")
    def _blob_count_matches(self) -> "WasbBallSizeFrame":
        if self.blob_count != len(self.blobs):
            raise ValueError("blob_count must equal len(blobs)")
        if self.selected_heatmap_observation_index >= self.heatmap_observation_count:
            raise ValueError("selected heatmap observation index is out of range")
        return self


class WasbBallSizeObservations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    artifact_type: Literal["racketsport_wasb_ball_size_observations"]
    fps: FiniteFloat = Field(gt=0.0)
    source: Literal["wasb"] = "wasb"
    source_mode: str
    input_preprocessing: Literal["official", "harness_v0"]
    primary_output: str
    coordinate_space: Literal["source_pixels"] = "source_pixels"
    heatmap_threshold: Literal[0.5] = HEATMAP_THRESHOLD
    radius_proxy_definition: Literal[
        "0.5 * sqrt(native_bbox_width_px * native_bbox_height_px)"
    ] = RADIUS_PROXY_DEFINITION
    not_ground_truth: Literal[True] = True
    emission_only: Literal[True] = True
    provenance: dict[str, Any] = Field(default_factory=dict)
    frames: list[WasbBallSizeFrame]

    @model_validator(mode="after")
    def _frames_are_strictly_ordered(self) -> "WasbBallSizeObservations":
        frame_ids = [frame.frame for frame in self.frames]
        if frame_ids != sorted(set(frame_ids)):
            raise ValueError("ball-size observation frame ids must be unique and sorted")
        return self


def connected_component_blob_extents(
    heatmap: Any,
    affine_inv: Any,
    *,
    cv2: Any,
    np: Any,
    threshold: float = HEATMAP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return every WASB connected component in native source-pixel coordinates."""

    array = np.asarray(heatmap, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"WASB heatmap must be 2D, got shape={array.shape}")
    affine = np.asarray(affine_inv, dtype=np.float64)
    if affine.shape != (2, 3):
        raise ValueError(f"WASB affine must be 2x3, got shape={affine.shape}")
    mask = (array > float(threshold)).astype(np.uint8)
    label_count, labels = cv2.connectedComponents(mask)
    area_scale = abs(float(np.linalg.det(affine[:, :2])))
    if not math.isfinite(area_scale) or area_scale <= 0.0:
        raise ValueError("WASB source affine must have a positive finite area scale")

    blobs: list[dict[str, Any]] = []
    for label in range(1, int(label_count)):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        weights = array[ys, xs]
        weight_sum = float(weights.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            continue
        center_h = np.asarray(
            [float(np.sum(xs * weights) / weight_sum), float(np.sum(ys * weights) / weight_sum), 1.0],
            dtype=np.float64,
        )
        center = affine @ center_h

        # Pixel-cell edges preserve a one-pixel component as a non-zero extent.
        x0, x1 = float(xs.min()) - 0.5, float(xs.max()) + 0.5
        y0, y1 = float(ys.min()) - 0.5, float(ys.max()) + 0.5
        corners_h = np.asarray(
            [[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]],
            dtype=np.float64,
        )
        corners = (affine @ corners_h.T).T
        native_x0 = float(corners[:, 0].min())
        native_y0 = float(corners[:, 1].min())
        native_x1 = float(corners[:, 0].max())
        native_y1 = float(corners[:, 1].max())
        width_px = native_x1 - native_x0
        height_px = native_y1 - native_y0
        radius_proxy_px = 0.5 * math.sqrt(width_px * height_px)
        blobs.append(
            {
                "center_xy_px": [float(center[0]), float(center[1])],
                "extent_xyxy_px": [native_x0, native_y0, native_x1, native_y1],
                "width_px": width_px,
                "height_px": height_px,
                "component_pixel_count": int(len(xs)),
                "component_area_px2": float(len(xs)) * area_scale,
                "heatmap_peak": float(weights.max()),
                "heatmap_weight_sum": weight_sum,
                "radius_proxy_px": radius_proxy_px,
                "source_detector": "wasb_concomp",
            }
        )
    blobs.sort(
        key=lambda blob: (
            -float(blob["heatmap_peak"]),
            -float(blob["heatmap_weight_sum"]),
            float(blob["center_xy_px"][0]),
            float(blob["center_xy_px"][1]),
        )
    )
    return blobs


def write_wasb_ball_size_observations(
    *,
    path: str | Path,
    fps: float,
    frame_times: Any,
    source_mode: str,
    input_preprocessing: str,
    primary_output: str | Path,
    frame_ids: Sequence[int],
    raw_frame_observations: Mapping[int, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Any],
) -> WasbBallSizeObservations:
    """Select one deterministic raw heatmap per frame and write the strict sidecar."""

    frames: list[dict[str, Any]] = []
    for frame in sorted({int(value) for value in frame_ids}):
        observations = list(raw_frame_observations.get(frame, []))
        if not observations:
            raise ValueError(f"missing raw WASB heatmap observation for frame {frame}")
        indexed = list(enumerate(observations))
        selected_index, selected = min(
            indexed,
            key=lambda item: (-float(item[1]["heatmap_peak"]), int(item[0])),
        )
        blobs = list(selected.get("blobs", []))
        frames.append(
            {
                "frame": frame,
                "pts_seconds": time_for_frame(frame, frame_times=frame_times, fps=float(fps)),
                "heatmap_peak": float(selected["heatmap_peak"]),
                "heatmap_observation_count": len(observations),
                "selected_heatmap_observation_index": selected_index,
                "blob_count": len(blobs),
                "blobs": blobs,
            }
        )

    payload = WasbBallSizeObservations.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_wasb_ball_size_observations",
            "fps": float(fps),
            "source": "wasb",
            "source_mode": source_mode,
            "input_preprocessing": input_preprocessing,
            "primary_output": str(primary_output),
            "coordinate_space": "source_pixels",
            "heatmap_threshold": HEATMAP_THRESHOLD,
            "radius_proxy_definition": RADIUS_PROXY_DEFINITION,
            "not_ground_truth": True,
            "emission_only": True,
            "provenance": dict(provenance),
            "frames": frames,
        }
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_wasb_ball_size_observations(path: str | Path) -> WasbBallSizeObservations:
    with Path(path).open("r", encoding="utf-8") as handle:
        return WasbBallSizeObservations.model_validate(json.load(handle))


__all__ = [
    "HEATMAP_THRESHOLD",
    "RADIUS_PROXY_DEFINITION",
    "WasbBallSizeBlob",
    "WasbBallSizeFrame",
    "WasbBallSizeObservations",
    "connected_component_blob_extents",
    "load_wasb_ball_size_observations",
    "write_wasb_ball_size_observations",
]
