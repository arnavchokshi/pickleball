"""TOTNet prediction adapters for schema-valid ball tracks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from .ball_tracknet import ball_frame
from .schemas import BallTrack


class TOTNetPredictionFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int
    xy: list[float] | None = None
    confidence: float = 0.0
    visible: bool = False

    @field_validator("frame_index")
    @classmethod
    def _frame_index_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame_index must be non-negative")
        return value

    @field_validator("xy")
    @classmethod
    def _xy_pair(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("xy must be a 2D point")
        return [_require_finite(component, "xy") for component in value]

    @field_validator("confidence")
    @classmethod
    def _confidence_unit_interval(cls, value: float) -> float:
        value = _require_finite(value, "confidence")
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value


class TOTNetPredictions(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int
    artifact_type: str
    fps: float
    image_size: tuple[int, int]
    input_size: tuple[int, int]
    model: dict[str, Any]
    frames: list[TOTNetPredictionFrame]

    @field_validator("schema_version")
    @classmethod
    def _schema_version_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("schema_version must be 1")
        return value

    @field_validator("artifact_type")
    @classmethod
    def _artifact_type_supported(cls, value: str) -> str:
        if value != "racketsport_totnet_predictions":
            raise ValueError("artifact_type must be racketsport_totnet_predictions")
        return value

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, value: float) -> float:
        value = _require_finite(value, "fps")
        if value <= 0.0:
            raise ValueError("fps must be positive")
        return value

    @field_validator("image_size", "input_size")
    @classmethod
    def _size_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
            raise ValueError("size must be [width, height]")
        return value


def totnet_predictions_to_ball_track(
    predictions: Mapping[str, Any] | TOTNetPredictions,
    *,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    threshold = _unit_interval(confidence_threshold, "confidence_threshold")
    payload = predictions if isinstance(predictions, TOTNetPredictions) else TOTNetPredictions.model_validate(predictions)
    frames = []
    for frame in sorted(payload.frames, key=lambda item: item.frame_index):
        visible = bool(frame.visible and frame.xy is not None and frame.confidence >= threshold)
        xy = frame.xy if visible and frame.xy is not None else [0.0, 0.0]
        frames.append(
            ball_frame(
                t=float(frame.frame_index) / float(payload.fps),
                xy=xy,
                conf=float(frame.confidence) if visible else 0.0,
                visible=visible,
                approx=False,
            )
        )
    ball_track = {
        "schema_version": 1,
        "fps": float(payload.fps),
        "source": "totnet",
        "frames": frames,
        "bounces": [],
    }
    BallTrack.model_validate(ball_track)
    return ball_track


def write_ball_track_from_totnet_predictions(
    predictions: str | Path | Mapping[str, Any] | TOTNetPredictions,
    *,
    out: str | Path,
    metadata_out: str | Path | None = None,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    prediction_payload = _load_predictions(predictions)
    totnet = TOTNetPredictions.model_validate(prediction_payload)
    ball_track = totnet_predictions_to_ball_track(totnet, confidence_threshold=confidence_threshold)
    out_path = Path(out)
    _write_json(out_path, ball_track)
    visible_count = sum(1 for frame in ball_track["frames"] if bool(frame["visible"]))
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_totnet_ball_run",
        "out": str(out_path),
        "fps": float(totnet.fps),
        "image_size": list(totnet.image_size),
        "input_size": list(totnet.input_size),
        "frame_count": len(ball_track["frames"]),
        "visible_frame_count": visible_count,
        "confidence_threshold": float(confidence_threshold),
        "model": totnet.model,
        "not_ground_truth": True,
        "verified": False,
    }
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def checkpoint_metadata(path: str | Path) -> dict[str, str]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint)}


def _load_predictions(predictions: str | Path | Mapping[str, Any] | TOTNetPredictions) -> Any:
    if isinstance(predictions, TOTNetPredictions):
        return predictions.model_dump(mode="json")
    if isinstance(predictions, Mapping):
        return predictions
    return json.loads(Path(predictions).read_text(encoding="utf-8"))


def _require_finite(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit_interval(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "TOTNetPredictionFrame",
    "TOTNetPredictions",
    "checkpoint_metadata",
    "totnet_predictions_to_ball_track",
    "write_ball_track_from_totnet_predictions",
]
