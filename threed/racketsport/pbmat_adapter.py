"""PB-MAT prediction adapters for schema-valid ball tracks.

This module is intentionally runtime-light. It validates and converts PB-MAT
model outputs, and provides pure-Python heatmap/crop helpers that are easy to
test before a trained GPU model is available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ball_tracknet import ball_frame
from .schemas import BallTrack


ARTIFACT_TYPE = "racketsport_pbmat_predictions"
CONFIDENCE_SEMANTICS = "PB-MAT candidate confidence; visibility_score gates visible frames"


class PBMatModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    checkpoint_sha256: str | None = None


class PBMatCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xy: list[float]
    confidence: float
    source: str = "coarse_heatmap"
    refined_xy: list[float] | None = None
    refined_confidence: float | None = None

    @field_validator("xy", "refined_xy")
    @classmethod
    def _must_be_xy(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("xy must be a 2D point")
        return [_require_finite(component, "xy") for component in value]

    @field_validator("confidence", "refined_confidence")
    @classmethod
    def _must_be_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = _require_finite(value, "confidence")
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value

    def selected_xy(self) -> list[float]:
        return list(self.refined_xy if self.refined_xy is not None else self.xy)

    def selected_confidence(self) -> float:
        if self.refined_xy is not None and self.refined_confidence is not None:
            return float(self.refined_confidence)
        return float(self.confidence)


class PBMatFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int
    t: float | None = None
    visibility_score: float
    blur_score: float | None = None
    occlusion_score: float | None = None
    selected_candidate: int | None = None
    candidates: list[PBMatCandidate] = Field(default_factory=list)

    @field_validator("frame_index")
    @classmethod
    def _frame_index_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame_index must be non-negative")
        return value

    @field_validator("t")
    @classmethod
    def _time_nonnegative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = _require_finite(value, "t")
        if value < 0.0:
            raise ValueError("t must be non-negative")
        return value

    @field_validator("visibility_score", "blur_score", "occlusion_score")
    @classmethod
    def _optional_unit_interval(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = _require_finite(value, "score")
        if not 0.0 <= value <= 1.0:
            raise ValueError("score must be in [0, 1]")
        return value


class PBMatPredictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifact_type: Literal["racketsport_pbmat_predictions"]
    source_mode: str = "pbmat_json"
    fps: float
    image_size: tuple[int, int]
    output_stride: int
    model: PBMatModelInfo
    frames: list[PBMatFrame]

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, value: float) -> float:
        value = _require_finite(value, "fps")
        if value <= 0.0:
            raise ValueError("fps must be positive")
        return value

    @field_validator("image_size")
    @classmethod
    def _image_size_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
            raise ValueError("image_size must contain positive width and height")
        return value

    @field_validator("output_stride")
    @classmethod
    def _stride_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("output_stride must be positive")
        return value


def pbmat_predictions_to_ball_track(
    predictions: Mapping[str, Any] | PBMatPredictions,
    *,
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
    """Convert PB-MAT predictions into the shared ``BallTrack`` artifact."""

    threshold = _unit_interval(visibility_threshold, "visibility_threshold")
    payload = predictions if isinstance(predictions, PBMatPredictions) else PBMatPredictions.model_validate(predictions)
    frames = []
    for frame in sorted(payload.frames, key=lambda item: item.frame_index):
        selected = _select_candidate(frame)
        t = frame.t if frame.t is not None else float(frame.frame_index) / float(payload.fps)
        if selected is None:
            frames.append(ball_frame(t=t, xy=[0.0, 0.0], conf=0.0, visible=False, approx=False))
            continue

        visible = frame.visibility_score >= threshold
        confidence = selected.selected_confidence() if visible else 0.0
        frames.append(
            ball_frame(
                t=t,
                xy=selected.selected_xy(),
                conf=confidence,
                visible=visible,
                approx=False,
            )
        )

    ball_track = {
        "schema_version": 1,
        "fps": float(payload.fps),
        "source": "pbmat",
        "frames": frames,
        "bounces": [],
    }
    BallTrack.model_validate(ball_track)
    return ball_track


def write_ball_track_from_pbmat_predictions(
    predictions: str | Path | Mapping[str, Any] | PBMatPredictions,
    *,
    out: str | Path,
    metadata_out: str | Path | None = None,
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
    """Write ``ball_track.json`` plus PB-MAT run metadata."""

    prediction_payload = _load_predictions(predictions)
    pbmat = PBMatPredictions.model_validate(prediction_payload)
    ball_track = pbmat_predictions_to_ball_track(pbmat, visibility_threshold=visibility_threshold)
    out_path = Path(out)
    _write_json(out_path, ball_track)
    visible_count = sum(1 for frame in ball_track["frames"] if bool(frame["visible"]))
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_pbmat_ball_run",
        "source_mode": pbmat.source_mode,
        "out": str(out_path),
        "fps": float(pbmat.fps),
        "image_size": list(pbmat.image_size),
        "output_stride": int(pbmat.output_stride),
        "frame_count": len(ball_track["frames"]),
        "visible_frame_count": visible_count,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "model": pbmat.model.model_dump(mode="json", exclude_none=True),
        "not_ground_truth": True,
        "verified": False,
    }
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def decode_pbmat_heatmap_candidates(
    *,
    heatmap: Sequence[Sequence[float]],
    offset_x: Sequence[Sequence[float]] | None = None,
    offset_y: Sequence[Sequence[float]] | None = None,
    output_stride: int = 4,
    top_k: int = 5,
    threshold: float = 0.0,
    nms_radius: int = 1,
) -> list[PBMatCandidate]:
    """Decode top-K PB-MAT heatmap peaks into source-frame candidates."""

    if output_stride <= 0:
        raise ValueError("output_stride must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")
    threshold = _unit_interval(threshold, "threshold")
    values = _matrix(heatmap, "heatmap")
    offsets_x = _optional_matrix(offset_x, "offset_x", shape=(len(values), len(values[0])))
    offsets_y = _optional_matrix(offset_y, "offset_y", shape=(len(values), len(values[0])))

    raw_peaks: list[tuple[float, int, int]] = []
    for row_index, row in enumerate(values):
        for col_index, score in enumerate(row):
            if score >= threshold:
                raw_peaks.append((score, row_index, col_index))

    kept: list[tuple[int, int, PBMatCandidate]] = []
    for score, row_index, col_index in sorted(raw_peaks, key=lambda item: item[0], reverse=True):
        if len(kept) >= top_k:
            break
        if any(abs(row_index - kept_row) <= nms_radius and abs(col_index - kept_col) <= nms_radius for kept_row, kept_col, _ in kept):
            continue
        dx = offsets_x[row_index][col_index] if offsets_x is not None else 0.0
        dy = offsets_y[row_index][col_index] if offsets_y is not None else 0.0
        kept.append(
            (
                row_index,
                col_index,
                PBMatCandidate(
                    xy=[(float(col_index) + dx) * output_stride, (float(row_index) + dy) * output_stride],
                    confidence=score,
                    source="coarse_heatmap",
                ),
            )
        )
    return [candidate for _, _, candidate in kept]


def remap_crop_refined_xy(
    *,
    crop_origin_xy: Sequence[float],
    crop_size: Sequence[int],
    refined_crop_xy: Sequence[float],
    image_size: Sequence[int],
) -> list[float]:
    """Map a crop-local refined point back into source-frame coordinates."""

    origin_x, origin_y = _xy(crop_origin_xy, "crop_origin_xy")
    crop_width, crop_height = _size(crop_size, "crop_size")
    image_width, image_height = _size(image_size, "image_size")
    crop_x, crop_y = _xy(refined_crop_xy, "refined_crop_xy")
    crop_x = min(max(crop_x, 0.0), float(crop_width - 1))
    crop_y = min(max(crop_y, 0.0), float(crop_height - 1))
    return [
        min(max(origin_x + crop_x, 0.0), float(image_width - 1)),
        min(max(origin_y + crop_y, 0.0), float(image_height - 1)),
    ]


def _select_candidate(frame: PBMatFrame) -> PBMatCandidate | None:
    if not frame.candidates:
        return None
    if frame.selected_candidate is not None:
        if frame.selected_candidate < 0 or frame.selected_candidate >= len(frame.candidates):
            raise ValueError("selected_candidate index is out of range")
        return frame.candidates[frame.selected_candidate]
    return max(frame.candidates, key=lambda candidate: candidate.selected_confidence())


def _load_predictions(predictions: str | Path | Mapping[str, Any] | PBMatPredictions) -> Any:
    if isinstance(predictions, PBMatPredictions):
        return predictions.model_dump(mode="json")
    if isinstance(predictions, (str, Path)):
        path = Path(predictions)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return predictions


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matrix(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if not values:
        raise ValueError(f"{name} must be a non-empty matrix")
    rows = [[_unit_interval(value, f"{name}/{row_index}/{col_index}") for col_index, value in enumerate(row)] for row_index, row in enumerate(values)]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be a rectangular non-empty matrix")
    return rows


def _optional_matrix(
    values: Sequence[Sequence[float]] | None,
    name: str,
    *,
    shape: tuple[int, int],
) -> list[list[float]] | None:
    if values is None:
        return None
    rows = [[_require_finite(value, f"{name}/{row_index}/{col_index}") for col_index, value in enumerate(row)] for row_index, row in enumerate(values)]
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise ValueError(f"{name} shape must match heatmap")
    return rows


def _xy(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must be a 2D point")
    return (_require_finite(values[0], f"{name}/0"), _require_finite(values[1], f"{name}/1"))


def _size(values: Sequence[int], name: str) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain width and height")
    width, height = int(values[0]), int(values[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive")
    return width, height


def _unit_interval(value: Any, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "ARTIFACT_TYPE",
    "CONFIDENCE_SEMANTICS",
    "PBMatCandidate",
    "PBMatFrame",
    "PBMatModelInfo",
    "PBMatPredictions",
    "decode_pbmat_heatmap_candidates",
    "pbmat_predictions_to_ball_track",
    "remap_crop_refined_xy",
    "write_ball_track_from_pbmat_predictions",
]
