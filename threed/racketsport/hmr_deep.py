"""CPU-only deep-tier HMR scaffold primitives.

This module validates and packages Fast SAM-3D-Body/MHR-style inputs and
outputs. It intentionally does not download checkpoints, run inference, select
variants, or touch the GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any


SCAFFOLD_NOTE = "cpu_hmr_deep_primitives_no_model_inference"
SCHEMA_VERSION = "body_hmr_deep.v0"
MODEL_FAMILY = "fast_sam_3d_body_mhr_to_smpl"


@dataclass(frozen=True)
class PlayerCropRequest:
    """Validated per-player crop request for the deep mesh tier."""

    frame_idx: int
    player_id: int
    bbox_xyxy: Sequence[float]
    image_size_px: Sequence[int]
    track_confidence: float
    source_track_id: str | None = None
    rally_span_id: str | None = None

    def __post_init__(self) -> None:
        frame_idx = _non_negative_int(self.frame_idx, name="frame_idx")
        player_id = _non_negative_int(self.player_id, name="player_id")
        bbox = _float_vector(self.bbox_xyxy, name="bbox_xyxy", length=4)
        image_size = _image_size(self.image_size_px)
        confidence = _confidence(self.track_confidence, name="track_confidence")

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        if x1 < 0.0 or y1 < 0.0 or x2 > image_size[0] or y2 > image_size[1]:
            raise ValueError("bbox_xyxy must be inside image_size_px")

        object.__setattr__(self, "frame_idx", frame_idx)
        object.__setattr__(self, "player_id", player_id)
        object.__setattr__(self, "bbox_xyxy", tuple(bbox))
        object.__setattr__(self, "image_size_px", tuple(image_size))
        object.__setattr__(self, "track_confidence", confidence)
        if self.source_track_id is not None:
            object.__setattr__(self, "source_track_id", str(self.source_track_id))
        if self.rally_span_id is not None:
            object.__setattr__(self, "rally_span_id", str(self.rally_span_id))

    @property
    def crop_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def area_px(self) -> float:
        return self.crop_xywh[2] * self.crop_xywh[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "player_id": self.player_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "crop_xywh": list(self.crop_xywh),
            "image_size_px": list(self.image_size_px),
            "track_confidence": self.track_confidence,
            "source_track_id": self.source_track_id,
            "rally_span_id": self.rally_span_id,
            "scaffold": SCAFFOLD_NOTE,
        }


def normalize_deep_hmr_payload(
    payload: Mapping[str, Any],
    *,
    request: PlayerCropRequest,
) -> dict[str, Any]:
    """Normalize a model-like MHR/SMPL payload into schema-friendly fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")

    smpl = payload.get("smpl")
    if not isinstance(smpl, Mapping):
        raise ValueError("payload.smpl must be a mapping")

    mhr = payload.get("mhr", {})
    if mhr is None:
        mhr = {}
    if not isinstance(mhr, Mapping):
        raise ValueError("payload.mhr must be a mapping when present")

    model_confidence = _confidence(
        payload.get("confidence", payload.get("model_confidence", request.track_confidence)),
        name="confidence",
    )
    mhr_pose_confidence = _optional_confidence(mhr.get("pose_confidence"), name="mhr.pose_confidence")
    confidence_components = {
        "model_confidence": model_confidence,
        "track_confidence": request.track_confidence,
    }
    confidence_values = [model_confidence, request.track_confidence]
    if mhr_pose_confidence is not None:
        confidence_components["mhr_pose_confidence"] = mhr_pose_confidence
        confidence_values.append(mhr_pose_confidence)

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_idx": request.frame_idx,
        "player_id": request.player_id,
        "model_family": MODEL_FAMILY,
        "representation": "smpl_ish_cpu_normalized",
        "smpl": {
            "global_orient": _float_vector(
                smpl.get("global_orient"),
                name="smpl.global_orient",
                length=3,
            ),
            "body_pose": _float_list(smpl.get("body_pose", []), name="smpl.body_pose"),
            "betas": _float_list(smpl.get("betas", []), name="smpl.betas"),
            "transl": _float_vector(smpl.get("transl"), name="smpl.transl", length=3),
        },
        "mhr": dict(mhr),
        "mesh_vertices_xyz": _vector3_list(
            payload.get("mesh_vertices_xyz", payload.get("vertices", [])),
            name="mesh_vertices_xyz",
        ),
        "joints3d_xyz": _vector3_list(
            payload.get("joints3d_xyz", payload.get("joints3d", [])),
            name="joints3d_xyz",
        ),
        "confidence": min(confidence_values),
        "confidence_components": confidence_components,
        "scaffold": SCAFFOLD_NOTE,
    }


def gate_deep_hmr_artifact(
    hmr_output: Mapping[str, Any],
    *,
    model_inference_ran: bool,
    min_confidence: float = 0.65,
) -> dict[str, Any]:
    """Return deterministic gate metadata for one normalized player output."""

    threshold = _confidence(min_confidence, name="min_confidence")
    confidence = _confidence(hmr_output.get("confidence", 0.0), name="hmr_output.confidence")
    reasons: list[str] = []

    if confidence < threshold:
        reasons.append("low_confidence")
    if not hmr_output.get("mesh_vertices_xyz"):
        reasons.append("missing_mesh_vertices")
    if not hmr_output.get("joints3d_xyz"):
        reasons.append("missing_joints3d")
    if not model_inference_ran:
        reasons.append("scaffold_only_no_model_inference")

    return {
        "decision": "reject" if reasons else "allow",
        "confidence": confidence,
        "threshold": threshold,
        "reasons": reasons,
    }


def build_player_hmr_artifact(
    request: PlayerCropRequest,
    hmr_output: Mapping[str, Any],
    *,
    model_inference_ran: bool = False,
    min_confidence: float = 0.65,
) -> dict[str, Any]:
    """Package one per-player deep-tier HMR artifact."""

    gate = gate_deep_hmr_artifact(
        hmr_output,
        model_inference_ran=model_inference_ran,
        min_confidence=min_confidence,
    )
    return {
        "artifact_type": "deep_hmr_player_frame",
        "schema_version": SCHEMA_VERSION,
        "crop_request": request.to_dict(),
        "hmr_output": dict(hmr_output),
        "gate": gate,
        "metadata": {
            "model_family": MODEL_FAMILY,
            "model_inference_ran": bool(model_inference_ran),
            "scaffold": SCAFFOLD_NOTE,
        },
    }


def _image_size(values: Sequence[int]) -> tuple[int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("image_size_px must be a 2-vector")
    width, height = values
    width_int = _positive_int(width, name="image_size_px/0")
    height_int = _positive_int(height, name="image_size_px/1")
    return (width_int, height_int)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value_int


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _float_vector(values: Any, *, name: str, length: int) -> list[float]:
    result = _float_list(values, name=name)
    if len(result) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return result


def _vector3_list(values: Any, *, name: str) -> list[list[float]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of 3-vectors")
    return [
        _float_vector(vector, name=f"{name}/{idx}", length=3)
        for idx, vector in enumerate(values)
    ]


def _float_list(values: Any, *, name: str) -> list[float]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")

    result: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{name}/{idx} must be finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}/{idx} must be finite") from exc
        if not isfinite(number):
            raise ValueError(f"{name}/{idx} must be finite")
        result.append(number)
    return result


def _confidence(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if confidence < 0.0 or confidence > 1.0 or not isfinite(confidence):
        raise ValueError(f"{name} must be between 0 and 1")
    return confidence


def _optional_confidence(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _confidence(value, name=name)
